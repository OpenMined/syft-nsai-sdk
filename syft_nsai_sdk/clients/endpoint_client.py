"""
URL construction utilities for SyftBox RPC endpoints
"""
from urllib.parse import quote, urljoin, urlparse, parse_qs
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SyftURLBuilder:
    """Builder for constructing SyftBox URLs and filesystem paths."""
    
    @staticmethod
    def build_syft_url(datasite: str, app_name: str, endpoint: str, params: Optional[Dict[str, str]] = None) -> str:
        """Build a syft:// URL for RPC calls.
        
        Args:
            datasite: Email of the service datasite
            app_name: Name of the application/service
            endpoint: RPC endpoint (e.g., 'chat', 'search', 'health')
            params: Optional query parameters
            
        Returns:
            Complete syft:// URL
        """
        # Clean inputs
        datasite = datasite.strip()
        app_name = app_name.strip()
        endpoint = endpoint.strip().lstrip('/')
        
        # Build base URL
        base_url = f"syft://{datasite}/app_data/{app_name}/rpc/{endpoint}"
        
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
            
            # Extract datasite from hostname
            datasite = parsed.hostname
            if not datasite:
                raise ValueError("Missing datasite in syft URL")
            
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
                'datasite': datasite,
                'app_name': app_name,
                'endpoint': endpoint,
                'params': flattened_params,
                'original_url': syft_url
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse syft URL '{syft_url}': {e}")
    
    # Filesystem Path Builders
    @staticmethod
    def build_datasite_path(datasites_path: Path, datasite: str) -> Path:
        """Build path to a datasite directory.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            
        Returns:
            Path to datasite directory
        """
        return datasites_path / datasite
    
    @staticmethod
    def build_public_path(datasites_path: Path, datasite: str) -> Path:
        """Build path to a datasite's public directory.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            
        Returns:
            Path to public directory
        """
        return datasites_path / datasite / "public"
    
    @staticmethod
    def build_routers_path(datasites_path: Path, datasite: str) -> Path:
        """Build path to a datasite's routers directory.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            
        Returns:
            Path to routers directory
        """
        return datasites_path / datasite / "public" / "routers"
    
    @staticmethod
    def build_service_directory_path(datasites_path: Path, datasite: str, service_name: str) -> Path:
        """Build path to a service directory in public/routers.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            service_name: Service name
            
        Returns:
            Path to service directory
        """
        return datasites_path / datasite / "public" / "routers" / service_name
    
    @staticmethod
    def build_metadata_path(datasites_path: Path, datasite: str, service_name: str) -> Path:
        """Build path to a service's metadata.json file.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            service_name: Service name
            
        Returns:
            Path to metadata.json file
        """
        return (datasites_path / 
                datasite / 
                "public" / 
                "routers" / 
                service_name / 
                "metadata.json")
    
    @staticmethod
    def build_app_data_path(datasites_path: Path, datasite: str, service_name: str) -> Path:
        """Build path to a service's app_data directory.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            service_name: Service name
            
        Returns:
            Path to app_data directory
        """
        return datasites_path / datasite / "app_data" / service_name
    
    @staticmethod
    def build_rpc_directory_path(datasites_path: Path, datasite: str, service_name: str) -> Path:
        """Build path to a service's RPC directory.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            service_name: Service name
            
        Returns:
            Path to RPC directory
        """
        return datasites_path / datasite / "app_data" / service_name / "rpc"
    
    @staticmethod
    def build_rpc_schema_path(datasites_path: Path, datasite: str, service_name: str) -> Path:
        """Build path to a service's RPC schema file.
        
        Args:
            datasites_path: Base datasites directory path
            datasite: Datasite email
            service_name: Service name
            
        Returns:
            Path to rpc.schema.json file
        """
        return (datasites_path / datasite / 
                "app_data" / service_name / "rpc" / "rpc.schema.json")
    
    @staticmethod
    def syft_url_to_filesystem_path(datasites_path: Path, syft_url: str) -> Path:
        """Convert a syft:// URL to its corresponding filesystem path.
        
        Args:
            datasites_path: Base datasites directory path
            syft_url: The syft:// URL to convert
            
        Returns:
            Corresponding filesystem path
            
        Raises:
            ValueError: If URL format is invalid
        """
        parsed = SyftURLBuilder.parse_syft_url(syft_url)
        
        # Build path: datasites/{datasite}/app_data/{app_name}/rpc/{endpoint}
        return (datasites_path / 
                parsed['datasite'] / 
                "app_data" / 
                parsed['app_name'] / 
                "rpc" / 
                parsed['endpoint'])
    
    @staticmethod
    def filesystem_path_to_syft_url(filesystem_path: Path, datasites_path: Path) -> Optional[str]:
        """Convert a filesystem path to its corresponding syft:// URL.
        
        Args:
            filesystem_path: The filesystem path to convert
            datasites_path: Base datasites directory path
            
        Returns:
            Corresponding syft:// URL or None if path doesn't match expected format
        """
        try:
            # Get relative path from datasites directory
            rel_path = filesystem_path.relative_to(datasites_path)
            path_parts = rel_path.parts
            
            # Expected format: {datasite}/app_data/{service_name}/rpc/{endpoint...}
            if len(path_parts) < 5:
                return None
            
            if path_parts[1] != 'app_data' or path_parts[3] != 'rpc':
                return None
            
            datasite = path_parts[0]
            service_name = path_parts[2]
            endpoint = '/'.join(path_parts[4:])
            
            return SyftURLBuilder.build_syft_url(datasite, service_name, endpoint)
            
        except (ValueError, IndexError):
            return None


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


class ServiceEndpoints:
    """Constructs service-specific endpoint URLs."""
    
    def __init__(self, datasite: str, service_name: str):
        """Initialize with service details.
        
        Args:
            datasite: Email of the service datasite
            service_name: Name of the service
        """
        self.datasite = datasite
        self.service_name = service_name
    
    def chat_url(self, **params) -> str:
        """Build syft URL for chat endpoint."""
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            "chat", 
            params
        )
    
    def search_url(self, query: Optional[str] = None, **params) -> str:
        """Build syft URL for search endpoint."""
        if query:
            params["query"] = query
        
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            "search", 
            params
        )
    
    def health_url(self) -> str:
        """Build syft URL for health endpoint."""
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            "health"
        )
    
    def openapi_url(self) -> str:
        """Build syft URL for OpenAPI specification."""
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
            "syft/openapi.json"
        )
    
    def custom_endpoint_url(self, endpoint: str, **params) -> str:
        """Build syft URL for custom endpoint."""
        return SyftURLBuilder.build_syft_url(
            self.datasite, 
            self.service_name, 
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
        required_fields = ['datasite', 'app_name', 'endpoint']
        for field in required_fields:
            if not parsed.get(field):
                return False
        
        # Validate datasite format (should be email)
        datasite = parsed['datasite']
        if '@' not in datasite or '.' not in datasite.split('@')[1]:
            return False
        
        return True
        
    except (ValueError, Exception):
        return False


def extract_service_info_from_url(syft_url: str) -> Dict[str, str]:
    """Extract service information from syft URL.
    
    Args:
        syft_url: The syft URL to parse
        
    Returns:
        Dictionary with service information
    """
    try:
        parsed = SyftURLBuilder.parse_syft_url(syft_url)
        
        return {
            'datasite': parsed['datasite'],
            'service_name': parsed['app_name'],
            'endpoint': parsed['endpoint'],
            'display_name': f"{parsed['app_name']} by {parsed['datasite']}"
        }
        
    except ValueError as e:
        logger.error(f"Failed to extract service info from URL '{syft_url}': {e}")
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
def build_chat_url(datasite: str, service_name: str) -> str:
    """Quick helper to build chat URL."""
    return ServiceEndpoints(datasite, service_name).chat_url()


def build_search_url(datasite: str, service_name: str, query: Optional[str] = None) -> str:
    """Quick helper to build search URL."""
    return ServiceEndpoints(datasite, service_name).search_url(query)


def build_health_url(datasite: str, service_name: str) -> str:
    """Quick helper to build health URL."""
    return ServiceEndpoints(datasite, service_name).health_url()