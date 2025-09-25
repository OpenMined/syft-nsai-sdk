"""
SyftBox RPC client for communicating with services via cache server
"""
import asyncio
import json
import httpx
import logging

from typing import Dict, Any, Optional, Union
from urllib.parse import quote, urljoin

from ..core.exceptions import NetworkError, RPCError, PollingTimeoutError, PollingError, TransactionTokenCreationError
from ..core.types import ServiceType
from ..models.service_info import ServiceInfo
from ..utils.spinner import AsyncSpinner
from .accounting_client import AccountingClient
from .auth_client import SyftBoxAuthClient
from .endpoint_client import ServiceEndpoints
from .request_client import SyftBoxAPIClient, HTTPClient, RequestArgs

logger = logging.getLogger(__name__)

class SyftBoxRPCClient(SyftBoxAPIClient):
    """Client for making RPC calls to SyftBox services via cache server."""
    
    def __init__(self, 
            cache_server_url: str = "https://syftbox.net",
            timeout: float = 30.0,
            max_poll_attempts: int = 30,
            poll_interval: float = 3.0,
            accounting_client: Optional[AccountingClient] = None,
            syftbox_auth_client: Optional[SyftBoxAuthClient] = None,
            http_client: Optional[HTTPClient] = None,
        ):
        """Initialize RPC client.
        
        Args:
            cache_server_url: URL of the SyftBox cache server
            timeout: Request timeout in seconds
            max_poll_attempts: Maximum polling attempts for async responses
            poll_interval: Seconds between polling attempts
            accounting_client: Optional accounting client for payments
            syftbox_auth_client: Optional SyftBox auth client for authentication
            http_client: Optional HTTP client instance
        """
        # Initialize parent SyftBoxAPIClient
        super().__init__(cache_server_url, http_client)
        
        # RPC-specific configuration
        self.max_poll_attempts = max_poll_attempts
        self.poll_interval = poll_interval
        self.timeout = timeout
        
        # Authentication and accounting clients
        self.accounting_client = accounting_client or AccountingClient()
        self.syftbox_auth_client = syftbox_auth_client or SyftBoxAuthClient()
        
        # Get user email from SyftBox auth (with guest fallback)
        self.from_email = self.syftbox_auth_client.get_user_email()
    
    async def close(self):
        """Close client and cleanup resources."""
        await super().close()
        if self.syftbox_auth_client:
            await self.syftbox_auth_client.close()
    
    async def call_rpc(self, 
                    syft_url: str, 
                    payload: Optional[Dict[str, Any]] = None,  
                    query_params: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None,
                    method: str = "POST",
                    show_spinner: bool = True,
                    args: Optional[RequestArgs] = None,
                    ) -> Dict[str, Any]:
        """Make an RPC call to a SyftBox service.
        
        Args:
            syft_url: The syft:// URL to call
            payload: JSON payload to send (for POST/PUT requests)
            query_params: Query parameters (for GET requests or additional params)
            headers: Additional headers (optional)
            method: HTTP method to use (GET or POST)
            show_spinner: Whether to show spinner during polling
            args: Additional request arguments
            
        Returns:
            Response data from the service
            
        Raises:
            NetworkError: For HTTP/network issues
            RPCError: For RPC-specific errors
            PollingTimeoutError: When polling times out
        """

        if args is None:
            args = RequestArgs()
            
        try:
            # Build base query parameters for SyftBox
            syftbox_params = {
                "suffix-sender": "true",
                "x-syft-url": syft_url,
                "x-syft-from": self.from_email
            }

            # Add raw parameter if specified in headers
            if headers and headers.get("x-syft-raw"):
                syftbox_params["x-syft-raw"] = headers["x-syft-raw"]
            
            # Merge with user-provided query params
            if query_params:
                syftbox_params.update(query_params)

            # Prepare request headers
            request_headers = {
                "Accept": "application/json",
                "x-syft-from": self.from_email,
                **(headers or {})
            }
            
            # Add SyftBox authentication if available
            auth_token = await self.syftbox_auth_client.get_auth_token()
            if auth_token:
                request_headers["Authorization"] = f"Bearer {auth_token}"

            # Prepare request data
            request_data = None
            if payload is not None:
                request_data = payload.copy()
                request_headers["Content-Type"] = "application/json"
            
            # Handle accounting token injection independently
            if args.is_accounting:
                recipient_email, service_identifier = self._parse_service_info_from_url(syft_url)
                if request_data is None:
                    request_data = {}
                
                if self.accounting_client.is_configured():
                    # Use accounting email as sender when we have accounting tokens
                    accounting_email = self.accounting_client.get_email()
                    request_data["user_email"] = accounting_email
                    
                    try:
                        recipient_email = syft_url.split('//')[1].split('/')[0]
                        transaction_token = await self.accounting_client.create_transaction_token(
                            recipient_email=recipient_email
                        )
                        request_data["transaction_token"] = transaction_token
                        logger.debug(f"Added accounting token for {service_identifier}")
                    except Exception as e:
                        raise TransactionTokenCreationError(f"Failed to create accounting token: {e}", recipient_email=recipient_email)
                else:
                    # Guest mode - use the current from_email
                    request_data["user_email"] = self.from_email
                    logger.debug(f"Guest mode request to {service_identifier} - no accounting token available")

            # Make the unified request
            response = await self.http_client.request(
                f"{self.base_url}/api/v1/send/msg",
                method,
                params=syftbox_params,
                json=request_data,
                headers=request_headers,
                args=args
            )

            # Handle response (same for all methods)
            if response.status_code == 200:
                # Immediate response
                data = response.json()
                return data
            
            elif response.status_code == 202:
                # Async response - need to poll
                data = response.json()
                request_id = data.get("request_id")
                
                if not request_id:
                    raise RPCError("Received 202 but no request_id", syft_url)

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
                    logger.debug(f"Error response data: {error_data}")
                    error_msg = error_data.get("message", f"HTTP {response.status_code}")

                    # Don't log "Permission denied" as ERROR - it's often expected for non-existent services
                    if error_msg == "Permission denied.":
                        logger.debug(f"Got expected permission denied response from {syft_url}")
                    else:
                        logger.error(f"Got error response from {error_msg}")
                except:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"Got error message from {error_msg}")
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
            show_spinner: Whether to show spinner during polling
            
        Returns:
            Final response data
            
        Raises:
            PollingTimeoutError: When max attempts reached
            PollingError: For polling-specific errors
        """
        # Build full poll URL
        poll_url = urljoin(self.base_url, poll_url_path.lstrip('/'))

        # Start spinner if enabled and requested
        spinner = None
        if show_spinner:
            spinner = AsyncSpinner("Waiting for service response")
            await spinner.start_async()
        try:
            for attempt in range(1, self.max_poll_attempts + 1):
                try:
                    # Prepare polling headers (include auth if available)
                    poll_headers = {
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    }
                    
                    # Add SyftBox auth token for polling requests too
                    auth_token = await self.syftbox_auth_client.get_auth_token()
                    if auth_token:
                        poll_headers["Authorization"] = f"Bearer {auth_token}"

                    # Make polling request
                    response = await self.http_client.get(
                        poll_url,
                        headers=poll_headers
                    )

                    if response.status_code == 200:
                        # Success - parse response
                        try:
                            data = response.json()
                        except json.JSONDecodeError:
                            raise PollingError("Invalid JSON in polling response", syft_url, poll_url)
                        
                        # Check response format
                        if "response" in data:
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

        # Health checks don't need auth or accounting - use guest mode
        health_args = RequestArgs(is_accounting=False)

        endpoints = ServiceEndpoints(service_info.datasite, service_info.name)
        syft_url = endpoints.health_url()
        return await self.call_rpc(syft_url, payload=None, method="GET", show_spinner=False, args=health_args)
    
    async def call_chat(self, service_info: ServiceInfo, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the chat endpoint of a service.
        
        Args:
            service_info: Service information
            request_data: Chat request payload
            
        Returns:
            Chat response data
        """
        # Hard-code service name to tinyllama:latest
        if "model" in request_data:
            request_data = request_data.copy()
            request_data["model"] = "tinyllama:latest"

        # Check if this is a free service to avoid payment validation errors
        chat_service = service_info.get_service_info(ServiceType.CHAT)
        is_free_service = chat_service and chat_service.pricing == 0.0
        
        chat_args = RequestArgs(
            is_accounting=not is_free_service,  # Disable accounting for free services
            # timeout=60.0,        # Longer timeout for chat
            # skip_loader=False    # Show spinner
            # email=self.accounting_client.get_email() if self.accounting_client.is_configured() else None
        )

        endpoints = ServiceEndpoints(service_info.datasite, service_info.name)
        syft_url = endpoints.chat_url()
        return await self.call_rpc(syft_url, payload=request_data, args=chat_args)

    async def call_search(self, service_info: ServiceInfo, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the search endpoint of a service.
        
        Args:
            service_info: Service information
            request_data: Search request payload
            
        Returns:
            Search response data
        """
        # Hard-code service name to tinyllama:latest
        if "model" in request_data:
            request_data = request_data.copy()
            request_data["model"] = "tinyllama:latest"

        # Check if this is a free service to avoid payment validation errors
        search_service = service_info.get_service_info(ServiceType.SEARCH)
        is_free_service = search_service and search_service.pricing == 0.0

        search_args = RequestArgs(
            is_accounting=not is_free_service,  # Disable accounting for free services
        )

        endpoints = ServiceEndpoints(service_info.datasite, service_info.name)
        syft_url = endpoints.search_url()
        return await self.call_rpc(syft_url, payload=request_data, args=search_args)

    async def call_custom_endpoint(self, 
                                   service_info: ServiceInfo, 
                                   endpoint: str,
                                   request_data: Optional[Dict[str, Any]] = None,
                                   method: str = "POST",
                                   query_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a custom endpoint of a service.
        
        Args:
            service_info: Service information
            endpoint: Custom endpoint name
            request_data: Request payload (for POST/PUT)
            method: HTTP method to use
            query_params: Query parameters (for GET or additional params)
            
        Returns:
            Response data
        """
        endpoints = ServiceEndpoints(service_info.datasite, service_info.name)
        syft_url = endpoints.custom_endpoint_url(endpoint)
        
        return await self.call_rpc(
            syft_url, 
            payload=request_data, 
            query_params=query_params,
            method=method
        )
    
    def _parse_service_info_from_url(self, syft_url: str) -> tuple[str, str]:
        """Parse datasite email and service name from syft URL.
        
        Args:
            syft_url: URL like 'syft://callis@openmined.org/app_data/carl-model/rpc/chat'
            
        Returns:
            Tuple of (recipient_email, service_identifier)
        """
        try:
            # parse datasite
            url_parts = syft_url.split('//')
            if len(url_parts) > 1:
                path_parts = url_parts[1].split('/')
                recipient_email = path_parts[0]  # callis@openmined.org
                service_name = path_parts[2] if len(path_parts) > 2 else "unknown"  # carl-model
                return recipient_email, f"{recipient_email}/{service_name}"
            else:
                return "unknown", "unknown"
        except (IndexError, AttributeError):
            return "unknown", "unknown"