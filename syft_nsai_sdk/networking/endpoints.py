"""
URL construction utilities for SyftBox RPC endpoints
"""
from urllib.parse import quote, urljoin, urlparse, parse_qs
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SyftURLBuilder:
    """Builder for constructing SyftBox URLs."""
    
    @staticmethod
    def build_syft_url(owner: str, app_name: str, endpoint: str, params: Optional[Dict[str, str]] = None) -> str:
        """Build a syft:// URL for RPC calls.
        
        Args:
            owner: Email of the model owner
            app_name: Name of the application/model
            endpoint: RPC endpoint (e.g., 'chat', 'search', 'health')
            params: Optional query parameters
            
        Returns:
            Complete syft:// URL
        """
        # Clean inputs
        owner = owner.strip()
        app_name = app_name.strip()
        endpoint = endpoint.strip().lstrip('/')
        
        # Build base URL
        base_url = f"syft://{owner}/app_data/{app_name}/rpc/{endpoint}"
        
        # Add query parameters if provided
        if params:
            query_parts = []
            for key, value in params.items():
                query_parts.append(f"{quote(key)}={quote(str(value))}")
            
            if query_parts:
                base_url += "?" + "&".join(query_parts)
        
        return base_url
    
    @staticmethod
    def parse_syft_url(syft_url: str) -> Dict[str, Any]:
        """Parse a syft:// URL into components.
        
        Args:
            syft_url: The syft:// URL to parse
            
        Returns:
            Dictionary with parsed components
            
        Raises:
            ValueError: If URL format is invalid
        """
        try:
            parsed = urlparse(syft_url)
            
            if parsed.scheme != 'syft':
                raise ValueError(f"Invalid scheme: {parsed.scheme}, expected 'syft'")
            
            # Extract owner from hostname
            owner = parsed.hostname
            if not owner:
                raise ValueError("Missing owner in syft URL")
            
            # Parse path components
            path_parts = [part for part in parsed.path.split('/') if part]
            
            if len(path_parts) < 4:
                raise ValueError("Invalid syft URL path format")
            
            if path_parts[0] != 'app_data':
                raise ValueError("Expected 'app_data' in path")
            
            if path_parts[2] != 'rpc':
                raise ValueError("Expected 'rpc' in path")
            
            app_name = path_parts[1]
            endpoint = '/'.join(path_parts[3:])  # Support nested endpoints
            
            # Parse query parameters
            params = parse_qs(parsed.query) if parsed.query else {}
            
            # Flatten single-item lists in params
            flattened_params = {}
            for key, values in params.items():
                flattened_params[key] = values[0] if len(values) == 1 else values
            
            return {
                'owner': owner,
                'app_name': app_name,
                'endpoint': endpoint,
                'params': flattened_params,
                'original_url': syft_url
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse syft URL '{syft_url}': {e}")


class CacheServerEndpoints:
    """Constructs cache server endpoint URLs."""
    
    def __init__(self, base_url: str):
        """Initialize with cache server base URL.
        
        Args:
            base_url: Base URL of the cache server (e.g., "https://syftbox.net")
        """
        self.base_url = base_url.rstrip('/')
    
    def build_send_message_url(self, syft_url: str, from_email: str, **params) -> str:
        """Build URL for sending RPC messages.
        
        Args:
            syft_url: The syft:// URL to call
            from_email: Email of the sender
            **params: Additional URL parameters
            
        Returns:
            Complete cache server URL for sending messages
        """
        endpoint = "/api/v1/send/msg"
        
        query_params = {
            "x-syft-url": syft_url,
            "x-syft-from": from_email,
            **params
        }
        
        return self._build_url_with_params(endpoint, query_params)
    
    def build_poll_url(self, poll_path: str) -> str:
        """Build URL for polling responses.
        
        Args:
            poll_path: Path returned from send message response
            
        Returns:
            Complete polling URL
        """
        # Clean the poll path
        clean_path = poll_path.lstrip('/')
        return urljoin(self.base_url + '/', clean_path)
    
    def build_health_check_url(self) -> str:
        """Build URL for cache server health check."""
        return urljoin(self.base_url, "/health")
    
    def build_openapi_url(self) -> str:
        """Build URL for OpenAPI specification."""
        return urljoin(self.base_url, "/openapi.json")
    
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


class ModelEndpoints:
    """Constructs model-specific endpoint URLs."""
    
    def __init__(self, owner: str, model_name: str):
        """Initialize with model details.
        
        Args:
            owner: Email of the model owner
            model_name: Name of the model
        """
        self.owner = owner
        self.model_name = model_name
    
    def chat_url(self, **params) -> str:
        """Build syft URL for chat endpoint."""
        return SyftURLBuilder.build_syft_url(
            self.owner, 
            self.model_name, 
            "chat", 
            params
        )
    
    def search_url(self, query: Optional[str] = None, **params) -> str:
        """Build syft URL for search endpoint."""
        if query:
            params["query"] = query
        
        return SyftURLBuilder.build_syft_url(
            self.owner, 
            self.model_name, 
            "search", 
            params
        )
    
    def health_url(self) -> str:
        """Build syft URL for health endpoint."""
        return SyftURLBuilder.build_syft_url(
            self.owner, 
            self.model_name, 
            "health"
        )
    
    def openapi_url(self) -> str:
        """Build syft URL for OpenAPI specification."""
        return SyftURLBuilder.build_syft_url(
            self.owner, 
            self.model_name, 
            "syft/openapi.json"
        )
    
    def custom_endpoint_url(self, endpoint: str, **params) -> str:
        """Build syft URL for custom endpoint."""
        return SyftURLBuilder.build_syft_url(
            self.owner, 
            self.model_name, 
            endpoint, 
            params
        )


def validate_syft_url(syft_url: str) -> bool:
    """Validate if a string is a properly formatted syft URL.
    
    Args:
        syft_url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parsed = SyftURLBuilder.parse_syft_url(syft_url)
        
        # Check required components
        required_fields = ['owner', 'app_name', 'endpoint']
        for field in required_fields:
            if not parsed.get(field):
                return False
        
        # Validate owner format (should be email)
        owner = parsed['owner']
        if '@' not in owner or '.' not in owner.split('@')[1]:
            return False
        
        return True
        
    except (ValueError, Exception):
        return False


def extract_model_info_from_url(syft_url: str) -> Dict[str, str]:
    """Extract model information from syft URL.
    
    Args:
        syft_url: The syft URL to parse
        
    Returns:
        Dictionary with model information
    """
    try:
        parsed = SyftURLBuilder.parse_syft_url(syft_url)
        
        return {
            'owner': parsed['owner'],
            'model_name': parsed['app_name'],
            'endpoint': parsed['endpoint'],
            'display_name': f"{parsed['app_name']} by {parsed['owner']}"
        }
        
    except ValueError as e:
        logger.error(f"Failed to extract model info from URL '{syft_url}': {e}")
        return {}


def is_cache_server_url(url: str) -> bool:
    """Check if URL is a cache server URL (not syft://).
    
    Args:
        url: URL to check
        
    Returns:
        True if it's an HTTP/HTTPS URL, False if syft:// or invalid
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https']
    except:
        return False


def normalize_cache_server_url(url: str) -> str:
    """Normalize cache server URL format.
    
    Args:
        url: Cache server URL
        
    Returns:
        Normalized URL without trailing slash
    """
    if not url:
        return url
    
    # Remove trailing slash
    normalized = url.rstrip('/')
    
    # Ensure scheme
    if not normalized.startswith(('http://', 'https://')):
        normalized = 'https://' + normalized
    
    return normalized


# Convenience functions
def build_chat_url(owner: str, model_name: str) -> str:
    """Quick helper to build chat URL."""
    return ModelEndpoints(owner, model_name).chat_url()


def build_search_url(owner: str, model_name: str, query: Optional[str] = None) -> str:
    """Quick helper to build search URL."""
    return ModelEndpoints(owner, model_name).search_url(query)


def build_health_url(owner: str, model_name: str) -> str:
    """Quick helper to build health URL."""
    return ModelEndpoints(owner, model_name).health_url()