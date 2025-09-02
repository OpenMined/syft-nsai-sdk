"""
SyftBox RPC client for communicating with services via cache server
"""
import asyncio
import json
from typing import Dict, Any, Optional, Union
from urllib.parse import quote, urljoin
import httpx
import logging

from ..core.exceptions import NetworkError, RPCError, PollingTimeoutError, PollingError
from ..core.types import ServiceInfo
from ..core.exceptions import PaymentError, AuthenticationError
from ..utils.spinner import AsyncSpinner
from .accounting_client import AccountingClient

logger = logging.getLogger(__name__)


class SyftBoxRPCClient:
    """Client for making RPC calls to SyftBox services via cache server."""
    
    def __init__(self, 
            cache_server_url: str = "https://syftbox.net",
            # from_email: str = None,
            timeout: float = 30.0,
            max_poll_attempts: int = 30,
            poll_interval: float = 3.0,
            accounting_client: Optional[AccountingClient] = None,
        ):
        """Initialize RPC client.
        
        Args:
            cache_server_url: URL of the SyftBox cache server
            from_email: Email to use for x-syft-from header
            timeout: Request timeout in seconds
            max_poll_attempts: Maximum polling attempts for async responses
            poll_interval: Seconds between polling attempts
            accounting_client: Optional accounting client for payments
        """

        self.cache_server_url = cache_server_url.rstrip('/')
        self.from_email = "guest@syft.org" or accounting_client.get_email()
        self.timeout = timeout
        self.max_poll_attempts = max_poll_attempts
        self.poll_interval = poll_interval
        
        # HTTP client with reasonable defaults
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        # Accounting client
        self.accounting_client = accounting_client or AccountingClient()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def build_syft_url(self, datasite: str, service_name: str, endpoint: str) -> str:
        """Build a syft:// URL for RPC calls.
        
        Args:
            datasite: Email of the service datasite
            service_name: Name of the service
            endpoint: RPC endpoint (e.g., 'chat', 'search', 'health')
            
        Returns:
            Complete syft:// URL
        """
        # Clean endpoint - remove leading slash if present
        endpoint = endpoint.lstrip('/')
        
        # Build syft URL: syft://datasite/app_data/service_name/rpc/endpoint
        return f"syft://{datasite}/app_data/{service_name}/rpc/{endpoint}"
        
    async def call_rpc(self, 
                    syft_url: str, 
                    payload: Optional[Dict[str, Any]] = None,  
                    headers: Optional[Dict[str, str]] = None,
                    method: str = "POST",
                    show_spinner: bool = True,
                    ) -> Dict[str, Any]:
        """Make an RPC call to a SyftBox service.
        
        Args:
            syft_url: The syft:// URL to call
            payload: JSON payload to send (optional)
            headers: Additional headers (optional)
            method: HTTP method to use (GET or POST)
            
        Returns:
            Response data from the service
            
        Raises:
            NetworkError: For HTTP/network issues
            RPCError: For RPC-specific errors
            PollingTimeoutError: When polling times out
        """
        try:
            # Build request URL
            request_url = f"{self.cache_server_url}/api/v1/send/msg"

            # Build query parameters
            params = {
                "x-syft-url": syft_url,
                "x-syft-from": self.from_email
            }

            # Add raw parameter if specified in headers
            if headers and headers.get("x-syft-raw"):
                params["x-syft-raw"] = headers["x-syft-raw"]
            
            logger.debug(f"Making RPC call to {syft_url}")

            # Make the request based on method
            if method.upper() == "GET":
                # For GET requests, use minimal headers and no payload
                get_headers = {
                    "Accept": "application/json",
                    "x-syft-from": self.from_email,
                    **(headers or {})
                }
                
                response = await self.client.get(
                    request_url,
                    params=params,
                    headers=get_headers
                )
            else:
                # For POST requests, handle payload and accounting
                if payload is None:
                    payload = {}

                # Extract recipient email for accounting token
                recipient_email = syft_url.split('//')[1].split('/')[0]
                
                # Create accounting token if client is configured
                if self.accounting_client.is_configured():
                    transaction_token = await self.accounting_client.create_transaction_token(
                        recipient_email=recipient_email
                    )
                    payload["transaction_token"] = transaction_token
                
                # Build POST headers
                post_headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "x-syft-from": self.from_email,
                    **(headers or {})
                }

                # Serialize payload and set Content-Length
                payload_json = json.dumps(payload) if payload else None
                if payload_json:
                    post_headers["Content-Length"] = str(len(payload_json.encode('utf-8')))

                response = await self.client.post(
                    request_url,
                    params=params,
                    headers=post_headers,
                    data=payload_json,
                )
            
            # Handle response (same for both GET and POST)
            if response.status_code == 200:
                # Immediate response
                data = response.json()
                logger.debug(f"Got immediate response from {syft_url}")
                return data
            
            elif response.status_code == 202:
                # Async response - need to poll
                data = response.json()
                request_id = data.get("request_id")
                
                if not request_id:
                    raise RPCError("Received 202 but no request_id", syft_url)

                logger.debug(f"Got async response, polling with request_id: {request_id}")

                # Extract poll URL from response
                poll_url_path = None
                if "data" in data and "poll_url" in data["data"]:
                    poll_url_path = data["data"]["poll_url"]
                elif "location" in response.headers:
                    poll_url_path = response.headers["location"]
                
                if not poll_url_path:
                    raise RPCError("Async response but no poll URL found", syft_url)
                
                # Poll for the actual response
                return await self._poll_for_response(poll_url_path, syft_url, request_id, show_spinner)

            else:
                # Error response
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", f"HTTP {response.status_code}")
                    logger.info(f"Got error response from {error_msg}")
                except:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.info(f"Got error message from {error_msg}")
                raise NetworkError(
                    f"RPC call failed: {error_msg}",
                    syft_url,
                    response.status_code
                )
        
        except httpx.TimeoutException:
            raise NetworkError(f"Request timeout after {self.timeout}s", syft_url)
        except httpx.RequestError as e:
            raise NetworkError(f"Request failed: {e}", syft_url)
        except json.JSONDecodeError as e:
            raise RPCError(f"Invalid JSON response: {e}", syft_url)
    
    async def _poll_for_response(self, 
                                 poll_url_path: str, 
                                 syft_url: str, 
                                 request_id: str,
                                 show_spinner: bool = True,
                                ) -> Dict[str, Any]:
        """Poll for an async RPC response.
        
        Args:
            poll_url_path: Path to poll (e.g., '/api/v1/poll/123')
            syft_url: Original syft URL for error context
            request_id: Request ID for logging
            
        Returns:
            Final response data
            
        Raises:
            PollingTimeoutError: When max attempts reached
            PollingError: For polling-specific errors
        """
        # Build full poll URL
        poll_url = urljoin(self.cache_server_url, poll_url_path.lstrip('/'))

        # Start spinner if enabled and requested
        spinner = None
        if show_spinner:
            spinner = AsyncSpinner("Waiting for service response")
            await spinner.start_async()
        try:
            for attempt in range(1, self.max_poll_attempts + 1):
                try:
                    logger.debug(f"Polling attempt {attempt}/{self.max_poll_attempts} for {request_id}")
                    
                    response = await self.client.get(
                        poll_url,
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/json"
                        }
                    )

                    if response.status_code == 200:
                        # Success - parse response
                        try:
                            data = response.json()
                        except json.JSONDecodeError:
                            raise PollingError("Invalid JSON in polling response", syft_url, poll_url)
                        
                        # Check response format
                        if "response" in data:
                            logger.debug(f"Polling complete for {request_id}")
                            return data["response"]
                        elif "status" in data:
                            if data["status"] == "pending":
                                # Still processing, continue polling
                                pass
                            elif data["status"] == "error":
                                error_msg = data.get("message", "Unknown error during processing")
                                raise RPCError(f"Service error: {error_msg}", syft_url)
                            else:
                                # Other status, return as-is
                                return data
                        else:
                            # Assume data is the response
                            return data
                    
                    elif response.status_code == 202:
                        # Still processing
                        try:
                            data = response.json()
                            if data.get("error") == "timeout":
                                # Normal polling timeout, continue
                                pass
                            else:
                                logger.debug(f"202 response: {data}")
                        except json.JSONDecodeError:
                            pass
                    
                    elif response.status_code == 404:
                        # Request not found
                        try:
                            data = response.json()
                            error_msg = data.get("message", "Request not found")
                        except:
                            error_msg = "Request not found"
                        raise PollingError(f"Polling failed: {error_msg}", syft_url, poll_url)
                    
                    elif response.status_code == 500:
                        # Server error
                        try:
                            data = response.json()
                            if data.get("error") == "No response exists. Polling timed out":
                                # This is a normal timeout, continue polling
                                pass
                            else:
                                raise PollingError(f"Server error: {data.get('message', 'Unknown')}", syft_url, poll_url)
                        except json.JSONDecodeError:
                            raise PollingError("Server error during polling", syft_url, poll_url)
                    
                    else:
                        # Other error
                        raise PollingError(f"Polling failed with status {response.status_code}", syft_url, poll_url)
                    
                    # Wait before next attempt
                    if attempt < self.max_poll_attempts:
                        await asyncio.sleep(self.poll_interval)
                
                except httpx.TimeoutException:
                    logger.warning(f"Polling timeout on attempt {attempt} for {request_id}")
                    if attempt == self.max_poll_attempts:
                        raise PollingTimeoutError(syft_url, attempt, self.max_poll_attempts)
                except httpx.RequestError as e:
                    logger.warning(f"Polling request error on attempt {attempt}: {e}")
                    if attempt == self.max_poll_attempts:
                        raise PollingError(f"Network error during polling: {e}", syft_url, poll_url)
            
            # Max attempts reached
            raise PollingTimeoutError(syft_url, self.max_poll_attempts, self.max_poll_attempts)
        finally:
            # Always stop spinner, even if an exception occurs
            if spinner:
                await spinner.stop_async("Response received")

    async def call_health(self, service_info: ServiceInfo) -> Dict[str, Any]:
        """Call the health endpoint of a service.
        
        Args:
            service_info: Service information
            
        Returns:
            Health response data
        """
        syft_url = self.build_syft_url(service_info.datasite, service_info.name, "health")
        # return await self.call_rpc(syft_url, show_spinner=False)
        return await self.call_rpc(syft_url, payload=None, method="GET", show_spinner=False)
        # return await self.call_rpc(syft_url, method="GET", show_spinner=False)
    
    async def call_chat1(self, service_info: ServiceInfo, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the chat endpoint of a service.
        
        Args:
            service_info: Service information
            request_data: Chat request payload
            
        Returns:
            Chat response data
        """
        syft_url = self.build_syft_url(service_info.datasite, service_info.name, "chat")
        return await self.call_rpc(syft_url, request_data)
    
    async def call_search1(self, service_info: ServiceInfo, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the search endpoint of a service.
        
        Args:
            service_info: Service information
            request_data: Search request payload
            
        Returns:
            Search response data
        """
        syft_url = self.build_syft_url(service_info.datasite, service_info.name, "search")
        return await self.call_rpc(syft_url, request_data)
    
    async def call_chat(self, service_info: ServiceInfo, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the chat endpoint of a service."""
        # Map router service names to actual service names if needed
        if "service" in request_data:
            request_data = request_data.copy()  # Don't modify original
            request_data["service"] = self._map_service_name(request_data["service"], service_info)
        
        syft_url = self.build_syft_url(service_info.datasite, service_info.name, "chat")
        return await self.call_rpc(syft_url, request_data)
    
    async def call_search(self, service_info: ServiceInfo, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the search endpoint of a service.
        
        Args:
            service_info: Service information
            request_data: Search request payload
            
        Returns:
            Search response data
        """
        # Map router service names to actual service names if needed
        if "service" in request_data:
            request_data = request_data.copy()  # Don't modify original
            request_data["service"] = self._map_service_name(request_data["service"], service_info)
        
        syft_url = self.build_syft_url(service_info.datasite, service_info.name, "search")
        return await self.call_rpc(syft_url, request_data)

    def _map_service_name(self, service_name: str, service_info: ServiceInfo) -> str:
        """Map router service names to actual underlying service names."""
        # Hard-coded mapping for known Ollama routers
        if service_info.name in ["carl-service", "carl-free"]:
            logger.debug(f"Mapping {service_name} to tinyllama:latest for known Ollama router")
            return "tinyllama:latest"
        
        # Tag-based detection for other Ollama routers
        if service_info.tags and "ollama" in [tag.lower() for tag in service_info.tags]:
            logger.debug(f"Mapping {service_name} to tinyllama:latest based on ollama tag")
            return "tinyllama:latest"
        
        # For other routers, use the original name
        logger.debug(f"Using original service name: {service_name}")
        return service_name
    
    def configure_accounting(self, service_url: str, email: str, password: str):
        """Configure accounting client.
        
        Args:
            service_url: Accounting service URL
            email: User email
            password: User password
        """
        self.accounting_client.configure(service_url, email, password)
    
    def has_accounting_client(self) -> bool:
        """Check if accounting client is configured."""
        return self.accounting_client.is_configured()
    
    def get_accounting_email(self) -> Optional[str]:
        """Get accounting email."""
        return self.accounting_client.get_email()
    
    async def get_account_balance(self) -> float:
        """Get current account balance.
        
        Returns:
            Account balance
        """
        return await self.accounting_client.get_account_balance()
    
    def configure(self, **kwargs):
        """Update client configuration.
        
        Args:
            **kwargs: Configuration options to update
        """
        if "cache_server_url" in kwargs:
            self.cache_server_url = kwargs["cache_server_url"].rstrip('/')
        if "from_email" in kwargs:
            self.from_email = kwargs["from_email"]
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
        if "max_poll_attempts" in kwargs:
            self.max_poll_attempts = kwargs["max_poll_attempts"]
        if "poll_interval" in kwargs:
            self.poll_interval = kwargs["poll_interval"]