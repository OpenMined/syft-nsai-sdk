"""
HTTP endpoint construction utilities for SyftBox RPC clients
"""
import logging
from urllib.parse import quote, urljoin
from typing import Dict, Any, Optional

from ..utils.validator import EmailValidator, URLValidator, ValidationError
from ..discovery.filesystem import SyftURLBuilder
from ..utils.constants import (
    SEND_MESSAGE_ENDPOINT, 
    HEALTH_ENDPOINT, 
    OPENAPI_ENDPOINT, 
    OPENAPI_FILENAME
)

logger = logging.getLogger(__name__)

class CacheServerEndpoints:
    """Constructs cache server endpoint URLs for HTTP communication."""
    
    def __init__(self, base_url: str):
        """Initialize with cache server base URL.
        
        Args:
            base_url: Base URL of the cache server (e.g., "https://syftbox.net")
            
        Raises:
            ValidationError: If URL is invalid
        """
        self.base_url = URLValidator.normalize_server_url(base_url)
    
    def build_send_message_url(self, syft_url: str, from_email: str, **params) -> str:
        """Build URL for sending RPC messages.
        
        Args:
            syft_url: The syft:// URL to call
            from_email: Email of the sender
            **params: Additional URL parameters
            
        Returns:
            Complete cache server URL for sending messages
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if not URLValidator.is_valid_syft_url(syft_url):
            raise ValidationError(f"Invalid syft URL: {syft_url}")
        
        from_email = EmailValidator.validate_email(from_email, "from_email")
        
        query_params = {
            "suffix-sender": "true",
            "x-syft-url": syft_url,
            "x-syft-from": from_email,
            **params
        }
        
        return self._build_url_with_params(SEND_MESSAGE_ENDPOINT, query_params)
    
    def build_poll_url(self, poll_path: str) -> str:
        """Build URL for polling responses.
        
        Args:
            poll_path: Path returned from send message response
            
        Returns:
            Complete polling URL
            
        Raises:
            ValidationError: If poll_path is invalid
        """
        if not poll_path:
            raise ValidationError("Poll path cannot be empty")
        
        # Clean the poll path
        clean_path = poll_path.lstrip('/')
        return urljoin(self.base_url + '/', clean_path)
    
    def build_health_check_url(self) -> str:
        """Build URL for cache server health check.
        
        Returns:
            Health check URL
        """
        return urljoin(self.base_url, HEALTH_ENDPOINT)
    
    def build_openapi_url(self) -> str:
        """Build URL for OpenAPI specification.
        
        Returns:
            OpenAPI specification URL
        """
        return urljoin(self.base_url, OPENAPI_ENDPOINT)
    
    def _build_url_with_params(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Build URL with query parameters.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            Complete URL with encoded parameters
        """
        base_url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        if not params:
            return base_url
        
        # Encode parameters
        param_parts = []
        for key, value in params.items():
            if value is not None:
                encoded_key = quote(str(key))
                encoded_value = quote(str(value))
                param_parts.append(f"{encoded_key}={encoded_value}")
        
        if param_parts:
            separator = '&' if '?' in base_url else '?'
            return base_url + separator + '&'.join(param_parts)
        
        return base_url


class ServiceEndpoints:
    """Constructs service-specific endpoint URLs for syft:// communication."""
    
    def __init__(self, datasite: str, service_name: str):
        """Initialize with service details.
        
        Args:
            datasite: Email of the service datasite
            service_name: Name of the service
            
        Raises:
            ValidationError: If inputs are invalid
        """
        self.datasite = EmailValidator.validate_email(datasite, "datasite")
        self.service_name = service_name.strip()
        
        if not self.service_name:
            raise ValidationError("Service name cannot be empty")
    
    def chat_url(self, **params) -> str:
        """Build syft URL for chat endpoint.
        
        Args:
            **params: Optional query parameters
            
        Returns:
            Complete syft:// URL for chat endpoint
        """
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            "chat", 
            params
        )
    
    def search_url(self, query: Optional[str] = None, **params) -> str:
        """Build syft URL for search endpoint.
        
        Args:
            query: Optional search query to include as parameter
            **params: Additional query parameters
            
        Returns:
            Complete syft:// URL for search endpoint
        """
        if query:
            params["query"] = query
        
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            "search", 
            params
        )
    
    def health_url(self) -> str:
        """Build syft URL for health endpoint.
        
        Returns:
            Complete syft:// URL for health endpoint
        """
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            "health"
        )
    
    def openapi_url(self) -> str:
        """Build syft URL for OpenAPI specification.
        
        Returns:
            Complete syft:// URL for OpenAPI specification
        """
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            f"syft/{OPENAPI_FILENAME}"
        )
    
    def custom_endpoint_url(self, endpoint: str, **params) -> str:
        """Build syft URL for custom endpoint.
        
        Args:
            endpoint: Custom endpoint path
            **params: Optional query parameters
            
        Returns:
            Complete syft:// URL for custom endpoint
        """
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            endpoint, 
            params
        )


# Convenience functions for quick URL building
def build_chat_url(datasite: str, service_name: str) -> str:
    """Quick helper to build chat URL.
    
    Args:
        datasite: Email of the service datasite
        service_name: Name of the service
        
    Returns:
        Complete syft:// URL for chat endpoint
    """
    return ServiceEndpoints(datasite, service_name).chat_url()


def build_search_url(datasite: str, service_name: str, query: Optional[str] = None) -> str:
    """Quick helper to build search URL.
    
    Args:
        datasite: Email of the service datasite
        service_name: Name of the service
        query: Optional search query
        
    Returns:
        Complete syft:// URL for search endpoint
    """
    return ServiceEndpoints(datasite, service_name).search_url(query)


def build_health_url(datasite: str, service_name: str) -> str:
    """Quick helper to build health URL.
    
    Args:
        datasite: Email of the service datasite
        service_name: Name of the service
        
    Returns:
        Complete syft:// URL for health endpoint
    """
    return ServiceEndpoints(datasite, service_name).health_url()


def build_custom_endpoint_url(datasite: str, service_name: str, endpoint: str, **params) -> str:
    """Quick helper to build custom endpoint URL.
    
    Args:
        datasite: Email of the service datasite
        service_name: Name of the service
        endpoint: Custom endpoint path
        **params: Optional query parameters
        
    Returns:
        Complete syft:// URL for custom endpoint
    """
    return ServiceEndpoints(datasite, service_name).custom_endpoint_url(endpoint, **params)


# URL utility functions for endpoint clients
def is_cache_server_url(url: str) -> bool:
    """Check if URL is a cache server URL (HTTP/HTTPS).
    
    Args:
        url: URL to check
        
    Returns:
        True if it's an HTTP/HTTPS URL, False otherwise
    """
    return URLValidator.is_valid_http_url(url)


def normalize_cache_server_url(url: str) -> str:
    """Normalize cache server URL format.
    
    Args:
        url: Cache server URL to normalize
        
    Returns:
        Normalized URL without trailing slash
        
    Raises:
        ValidationError: If URL is invalid
    """
    return URLValidator.normalize_server_url(url)