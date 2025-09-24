"""
HTTP client wrapper and utilities for SyftBox networking
"""
import httpx
import asyncio
import logging

from typing import Dict, Any, Optional, Union

from ..core.exceptions import NetworkError

logger = logging.getLogger(__name__)

class RequestArgs:
    """Arguments for HTTP requests."""
    def __init__(
            self,
            is_accounting: bool = False,
            skip_loader: bool = False,
            timeout: Optional[float] = None,
            **kwargs
        ):
        self.is_accounting = is_accounting
        self.skip_loader = skip_loader
        self.timeout = timeout
        self.extra_args = kwargs

class HTTPClient:
    """Wrapper around httpx with SyftBox-specific configurations."""
    
    def __init__(self, 
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 user_agent: str = "syft-hub/0.1.0"):
        """Initialize HTTP client with SyftBox defaults.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            user_agent: User agent string to use
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent
        
        # Track event loop for automatic client recreation
        self._current_loop = None
        self._client = None
    
    @property
    def client(self):
        """Get the httpx client, recreating if event loop has changed."""
        import asyncio
        
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop
            current_loop = None
        
        # Check if we need to recreate the client
        if self._client is None or self._current_loop != current_loop:
            # Clean up old client if it exists and is from a different loop
            if self._client is not None and self._current_loop != current_loop:
                try:
                    # Schedule cleanup of old client
                    if self._current_loop is not None and not self._current_loop.is_closed():
                        self._current_loop.create_task(self._client.aclose())
                except Exception:
                    pass  # Old loop might be closed, ignore
            
            # Create new client for current event loop
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers={"User-Agent": self.user_agent}
            )
            self._current_loop = current_loop
            
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._current_loop = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def request(self,
                      url: str,
                      method: str,
                      params: Optional[Dict[str, Any]] = None,
                      data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                      json: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None,
                      args: Optional[RequestArgs] = None) -> httpx.Response:
        """Make an HTTP request with configurable parameters.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            params: Query parameters
            data: Raw data to send
            json: JSON data to send
            headers: Additional headers
            args: Additional request arguments
            
        Returns:
            HTTP response
            
        Raises:
            NetworkError: For network-related failures
        """
        if args is None:
            args = RequestArgs()
        
        # Build request kwargs - be explicit about what we pass
        request_kwargs = {}
        
        if params is not None:
            request_kwargs["params"] = params
        if headers is not None:
            request_kwargs["headers"] = headers
        if json is not None:
            request_kwargs["json"] = json
        elif data is not None:
            request_kwargs["data"] = data
        
        # Apply timeout if specified
        if args.timeout:
            request_kwargs["timeout"] = httpx.Timeout(args.timeout)
        
        # Only add extra args that are safe for httpx
        if args.extra_args:
            # Filter out any problematic keys
            safe_extra_args = {k: v for k, v in args.extra_args.items() 
                             if k not in ['url', 'method', 'args']}
            request_kwargs.update(safe_extra_args)
        
        return await self._request(method, url, **request_kwargs)

    # Convenience methods that use the unified request method
    async def get(self, 
                  url: str,
                  params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  args: Optional[RequestArgs] = None) -> httpx.Response:
        """Make a GET request."""
        return await self.request(url, "GET", params=params, headers=headers, args=args)
    
    async def post(self,
                   url: str,
                   json: Optional[Dict[str, Any]] = None,
                   data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                   params: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None,
                   args: Optional[RequestArgs] = None) -> httpx.Response:
        """Make a POST request."""
        return await self.request(url, "POST", params=params, data=data, json=json, headers=headers, args=args)
    
    async def put(self,
                  url: str,
                  json: Optional[Dict[str, Any]] = None,
                  data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                  params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  args: Optional[RequestArgs] = None) -> httpx.Response:
        """Make a PUT request."""
        return await self.request(url, "PUT", params=params, data=data, json=json, headers=headers, args=args)
    
    async def delete(self,
                     url: str,
                     params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     args: Optional[RequestArgs] = None) -> httpx.Response:
        """Make a DELETE request."""
        return await self.request(url, "DELETE", params=params, headers=headers, args=args)
    
    async def _request(self,
                       method: str,
                       url: str,
                       **kwargs) -> httpx.Response:
        """Make a request with retry logic.
        
        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Request arguments
            
        Returns:
            HTTP response
            
        Raises:
            NetworkError: For network-related failures
        """
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.request(method, url, **kwargs)
                
                # Log successful request
                logger.debug(f"{method} {url} -> {response.status_code}")
                
                return response
                
            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1}): {url}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                
            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries + 1}): {url}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                
            except httpx.RequestError as e:
                last_exception = e
                logger.warning(f"Request error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                logger.debug(f"httpx.RequestError details: {type(e).__name__}: {e}")
                
                # Don't retry for client errors (4xx)
                break
        
        # All attempts failed
        error_msg = f"{method} {url} failed after {self.max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
        
        raise NetworkError(error_msg, url)


class SyftBoxAPIClient:
    """High-level API client for SyftBox services."""
    
    def __init__(self, 
                 base_url: str,
                 http_client: Optional[HTTPClient] = None):
        """Initialize API client.
        
        Args:
            base_url: Base URL for the API
            http_client: Optional HTTP client instance
        """
        self.base_url = base_url.rstrip('/')
        self.http_client = http_client or HTTPClient()
        self._own_http_client = http_client is None
    
    async def close(self):
        """Close the API client."""
        if self._own_http_client:
            await self.http_client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Full URL
        """
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}/{endpoint}"
    
    async def request(self,
                      endpoint: str,
                      method: str,
                      params: Optional[Dict[str, Any]] = None,
                      data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                      json: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None,
                      args: Optional[RequestArgs] = None) -> Dict[str, Any]:
        """Make an HTTP request to an API endpoint.
        
        Args:
            endpoint: API endpoint
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            params: Query parameters
            data: Raw data to send
            json: JSON data to send
            headers: Additional headers
            args: Additional request arguments
            
        Returns:
            JSON response data
            
        Raises:
            NetworkError: For API errors
        """
        url = self._build_url(endpoint)
        response = await self.http_client.request(
            url, method, params=params, data=data, json=json, headers=headers, args=args
        )
        return await self._handle_response(response)
    
    # Convenience methods for backward compatibility
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, args: Optional[RequestArgs] = None) -> Dict[str, Any]:
        """Make a GET request to an API endpoint."""
        return await self.request(endpoint, "GET", params=params, headers=headers, args=args)
    
    async def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None, data: Optional[Union[str, bytes, Dict[str, Any]]] = None, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, args: Optional[RequestArgs] = None) -> Dict[str, Any]:
        """Make a POST request to an API endpoint."""
        return await self.request(endpoint, "POST", params=params, data=data, json=json, headers=headers, args=args)
    
    async def put(self, endpoint: str, json: Optional[Dict[str, Any]] = None, data: Optional[Union[str, bytes, Dict[str, Any]]] = None, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, args: Optional[RequestArgs] = None) -> Dict[str, Any]:
        """Make a PUT request to an API endpoint."""
        return await self.request(endpoint, "PUT", params=params, data=data, json=json, headers=headers, args=args)
    
    async def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, args: Optional[RequestArgs] = None) -> Dict[str, Any]:
        """Make a DELETE request to an API endpoint."""
        return await self.request(endpoint, "DELETE", params=params, headers=headers, args=args)
    
    async def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API response.
        
        Args:
            response: HTTP response
            
        Returns:
            Parsed JSON data
            
        Raises:
            NetworkError: For API errors
        """
        # Handle different status codes
        if response.status_code == 200:
            try:
                return response.json()
            except Exception as e:
                raise NetworkError(f"Invalid JSON response: {e}", str(response.url))
        
        elif response.status_code == 404:
            raise NetworkError("Resource not found", str(response.url), 404)
        
        elif response.status_code >= 500:
            raise NetworkError(f"Server error: {response.status_code}", str(response.url), response.status_code)
        
        elif response.status_code >= 400:
            # Try to get error message from response
            try:
                error_data = response.json()
                error_msg = error_data.get("message", f"Client error: {response.status_code}")
            except:
                error_msg = f"Client error: {response.status_code}"
            
            raise NetworkError(error_msg, str(response.url), response.status_code)
        
        else:
            # Other success codes (201, 202, etc.)
            try:
                return response.json()
            except:
                return {}


# Utility functions
async def check_connectivity(url: str, timeout: float = 5.0) -> bool:
    """Check if a URL is reachable.
    
    Args:
        url: URL to check
        timeout: Timeout in seconds
        
    Returns:
        True if reachable, False otherwise
    """
    try:
        async with HTTPClient(timeout=timeout, max_retries=0) as client:
            response = await client.get(url)
            return response.status_code < 500
    except Exception:
        return False


async def get_server_info(url: str) -> Optional[Dict[str, Any]]:
    """Get server information from a SyftBox endpoint.
    
    Args:
        url: Server URL
        
    Returns:
        Server info dict or None if unavailable
    """
    try:
        async with SyftBoxAPIClient(url) as client:
            return await client.get("/info")
    except Exception:
        return None


# Connection pool manager
class ConnectionPoolManager:
    """Manages HTTP connection pools for multiple servers."""
    
    def __init__(self):
        self._pools: Dict[str, HTTPClient] = {}
    
    def get_client(self, base_url: str) -> HTTPClient:
        """Get or create an HTTP client for a base URL.
        
        Args:
            base_url: Base URL for the client
            
        Returns:
            HTTP client instance
        """
        if base_url not in self._pools:
            self._pools[base_url] = HTTPClient()
        
        return self._pools[base_url]
    
    async def close_all(self):
        """Close all HTTP clients."""
        for client in self._pools.values():
            await client.close()
        self._pools.clear()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()


# Global connection pool
_connection_pool = ConnectionPoolManager()

def get_http_client(base_url: str) -> HTTPClient:
    """Get a shared HTTP client for a base URL.
    
    Args:
        base_url: Base URL
        
    Returns:
        Shared HTTP client instance
    """
    return _connection_pool.get_client(base_url)