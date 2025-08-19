"""
HTTP client wrapper and utilities for SyftBox networking
"""
import httpx
import asyncio
from typing import Dict, Any, Optional, Union
import logging

from ..core.exceptions import NetworkError

logger = logging.getLogger(__name__)


class HTTPClient:
    """Wrapper around httpx with SyftBox-specific configurations."""
    
    def __init__(self, 
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 user_agent: str = "syft-nsai-sdk/0.1.0"):
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
        
        # Configure httpx client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={"User-Agent": user_agent}
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def get(self, 
                  url: str,
                  params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  **kwargs) -> httpx.Response:
        """Make a GET request with retries.
        
        Args:
            url: URL to request
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional httpx arguments
            
        Returns:
            HTTP response
            
        Raises:
            NetworkError: For network-related failures
        """
        return await self._request("GET", url, params=params, headers=headers, **kwargs)
    
    async def post(self,
                   url: str,
                   json: Optional[Dict[str, Any]] = None,
                   data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                   params: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None,
                   **kwargs) -> httpx.Response:
        """Make a POST request with retries.
        
        Args:
            url: URL to request
            json: JSON data to send
            data: Raw data to send
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional httpx arguments
            
        Returns:
            HTTP response
            
        Raises:
            NetworkError: For network-related failures
        """
        return await self._request(
            "POST", url, 
            json=json, data=data, params=params, headers=headers, 
            **kwargs
        )
    
    async def put(self,
                  url: str,
                  json: Optional[Dict[str, Any]] = None,
                  data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                  params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  **kwargs) -> httpx.Response:
        """Make a PUT request with retries."""
        return await self._request(
            "PUT", url,
            json=json, data=data, params=params, headers=headers,
            **kwargs
        )
    
    async def delete(self,
                     url: str,
                     params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     **kwargs) -> httpx.Response:
        """Make a DELETE request with retries."""
        return await self._request("DELETE", url, params=params, headers=headers, **kwargs)
    
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
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a GET request to an API endpoint.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional arguments for the request
            
        Returns:
            JSON response data
            
        Raises:
            NetworkError: For API errors
        """
        url = self._build_url(endpoint)
        response = await self.http_client.get(url, **kwargs)
        return await self._handle_response(response)
    
    async def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a POST request to an API endpoint.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional arguments for the request
            
        Returns:
            JSON response data
            
        Raises:
            NetworkError: For API errors
        """
        url = self._build_url(endpoint)
        response = await self.http_client.post(url, **kwargs)
        return await self._handle_response(response)
    
    async def put(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a PUT request to an API endpoint."""
        url = self._build_url(endpoint)
        response = await self.http_client.put(url, **kwargs)
        return await self._handle_response(response)
    
    async def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a DELETE request to an API endpoint."""
        url = self._build_url(endpoint)
        response = await self.http_client.delete(url, **kwargs)
        return await self._handle_response(response)
    
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