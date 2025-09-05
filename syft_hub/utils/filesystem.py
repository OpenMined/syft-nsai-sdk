"""
Filesystem utilities and URL construction for SyftBox
"""
import logging

from urllib.parse import quote, urljoin, urlparse, parse_qs
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.constants import (
    DATASITES_DIR, 
    APP_DATA_DIR, 
    PUBLIC_DIR, 
    ROUTERS_DIR, 
    RPC_DIR,
    METADATA_FILENAME, 
    RPC_SCHEMA_FILENAME,
    SYFT_SCHEME, 
    SEND_MESSAGE_ENDPOINT, 
    HEALTH_ENDPOINT, 
    OPENAPI_ENDPOINT,
    OPENAPI_FILENAME,
)
from ..utils.validator import (
    EmailValidator, 
    URLValidator, 
    PathValidator, 
    ValidationError,
)

logger = logging.getLogger(__name__)

class SyftBoxFilesystem:
    """Filesystem path utilities for SyftBox structure."""
    
    def __init__(self, data_dir: Path):
        """Initialize with SyftBox data directory.
        
        Args:
            data_dir: Path to SyftBox data directory
        """
        self.data_dir = Path(data_dir)
    
    # Core directory paths
    @property
    def datasites_path(self) -> Path:
        """Path to the datasites directory."""
        return self.data_dir / DATASITES_DIR
    
    def datasite_path(self, datasite_email: str) -> Path:
        """Build path to a datasite directory.
        
        Args:
            datasite_email: Email of the datasite
            
        Returns:
            Path to datasite directory
        """
        return self.datasites_path / datasite_email
    
    def public_path(self, datasite_email: str) -> Path:
        """Build path to a datasite's public directory.
        
        Args:
            datasite_email: Email of the datasite
            
        Returns:
            Path to public directory
        """
        return self.datasite_path(datasite_email) / PUBLIC_DIR
    
    def routers_path(self, datasite_email: str) -> Path:
        """Build path to a datasite's routers directory.
        
        Args:
            datasite_email: Email of the datasite
            
        Returns:
            Path to routers directory
        """
        return self.public_path(datasite_email) / ROUTERS_DIR
    
    def app_data_path(self, datasite_email: str, app_name: str) -> Path:
        """Build path to an app's data directory.
        
        Args:
            datasite_email: Email of the datasite
            app_name: Name of the application
            
        Returns:
            Path to app_data directory
        """
        return self.datasite_path(datasite_email) / APP_DATA_DIR / app_name
    
    def rpc_directory_path(self, datasite_email: str, app_name: str) -> Path:
        """Build path to an app's RPC directory.
        
        Args:
            datasite_email: Email of the datasite
            app_name: Name of the application
            
        Returns:
            Path to RPC directory
        """
        return self.app_data_path(datasite_email, app_name) / RPC_DIR
    
    # Service-specific paths
    def service_directory_path(self, datasite_email: str, service_name: str) -> Path:
        """Build path to a service directory in public/routers.
        
        Args:
            datasite_email: Email of the datasite
            service_name: Name of the service
            
        Returns:
            Path to service directory
        """
        return self.routers_path(datasite_email) / service_name
    
    def metadata_path(self, datasite_email: str, service_name: str) -> Path:
        """Build path to a service's metadata.json file.
        
        Args:
            datasite_email: Email of the datasite
            service_name: Name of the service
            
        Returns:
            Path to metadata.json file
        """
        return self.service_directory_path(datasite_email, service_name) / METADATA_FILENAME
    
    def rpc_schema_path(self, datasite_email: str, service_name: str) -> Path:
        """Build path to a service's RPC schema file.
        
        Args:
            datasite_email: Email of the datasite
            service_name: Name of the service
            
        Returns:
            Path to rpc.schema.json file
        """
        return self.rpc_directory_path(datasite_email, service_name) / RPC_SCHEMA_FILENAME
    
    # Path validation and existence checking
    def validate_structure(self, datasite_email: str) -> None:
        """Validate basic SyftBox structure exists for a datasite.
        
        Args:
            datasite_email: Email of the datasite to validate
            
        Raises:
            ValidationError: If required structure is missing
        """
        try:
            # Validate email
            EmailValidator.validate_email(datasite_email, "datasite email")
            
            # Check core directories
            PathValidator.validate_directory_exists(self.data_dir, "SyftBox data")
            PathValidator.validate_directory_exists(self.datasites_path, "Datasites")
            PathValidator.validate_directory_exists(
                self.datasite_path(datasite_email), 
                f"Datasite ({datasite_email})"
            )
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Structure validation failed: {e}")
    
    def check_service_exists(self, datasite_email: str, service_name: str) -> bool:
        """Check if a service exists in the filesystem.
        
        Args:
            datasite_email: Email of the datasite
            service_name: Name of the service
            
        Returns:
            True if service directory exists, False otherwise
        """
        try:
            service_path = self.service_directory_path(datasite_email, service_name)
            return service_path.exists() and service_path.is_dir()
        except Exception:
            return False
    
    def check_rpc_endpoint_exists(self, datasite_email: str, app_name: str, endpoint: str) -> bool:
        """Check if an RPC endpoint exists in the filesystem.
        
        Args:
            datasite_email: Email of the datasite
            app_name: Name of the application
            endpoint: RPC endpoint path
            
        Returns:
            True if endpoint directory exists, False otherwise
        """
        try:
            endpoint_path = self.rpc_directory_path(datasite_email, app_name) / endpoint
            return endpoint_path.exists()
        except Exception:
            return False
    
    def list_datasites(self) -> List[str]:
        """List all available datasites.
        
        Returns:
            List of datasite email addresses
        """
        if not self.datasites_path.exists():
            return []
        
        datasites = []
        for item in self.datasites_path.iterdir():
            if item.is_dir() and EmailValidator.is_valid_email(item.name):
                datasites.append(item.name)
        
        return sorted(datasites)
    
    def list_services(self, datasite_email: str) -> List[str]:
        """List all services for a datasite.
        
        Args:
            datasite_email: Email of the datasite
            
        Returns:
            List of service names
        """
        try:
            routers_path = self.routers_path(datasite_email)
            if not routers_path.exists():
                return []
            
            services = []
            for item in routers_path.iterdir():
                if item.is_dir():
                    services.append(item.name)
            
            return sorted(services)
            
        except Exception:
            return []


class SyftURLBuilder:
    """Builder for constructing SyftBox URLs and converting between URLs and filesystem paths."""
    
    @staticmethod
    def build_syft_url(datasite: str, app_name: str, endpoint: str, 
                       params: Optional[Dict[str, str]] = None) -> str:
        """Build a syft:// URL for RPC calls.
        
        Args:
            datasite: Email of the service datasite
            app_name: Name of the application/service
            endpoint: RPC endpoint (e.g., 'chat', 'search', 'health')
            params: Optional query parameters
            
        Returns:
            Complete syft:// URL
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        try:
            datasite = EmailValidator.validate_email(datasite.strip(), "datasite")
            app_name = app_name.strip()
            endpoint = endpoint.strip().lstrip('/')
            
            if not app_name:
                raise ValidationError("app_name cannot be empty")
            if not endpoint:
                raise ValidationError("endpoint cannot be empty")
                
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Invalid URL inputs: {e}")
        
        # Build base URL
        base_url = f"{SYFT_SCHEME}://{datasite}/{APP_DATA_DIR}/{app_name}/{RPC_DIR}/{endpoint}"
        
        # Add query parameters if provided
        if params:
            query_parts = []
            for key, value in params.items():
                if value is not None:
                    query_parts.append(f"{quote(key)}={quote(str(value))}")
            
            if query_parts:
                base_url += "?" + "&".join(query_parts)
        
        return base_url
    
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
    def parse_syft_url(syft_url: str) -> Dict[str, Any]:
        """Parse a syft:// URL into components.
        
        Args:
            syft_url: The syft:// URL to parse
            
        Returns:
            Dictionary with parsed components
            
        Raises:
            ValidationError: If URL format is invalid
        """
        try:
            parsed = urlparse(syft_url)
            
            if parsed.scheme != SYFT_SCHEME:
                raise ValidationError(f"Invalid scheme: {parsed.scheme}, expected '{SYFT_SCHEME}'")
            
            # Extract and validate datasite
            datasite = parsed.hostname
            if not datasite:
                raise ValidationError("Missing datasite in syft URL")
            
            datasite = EmailValidator.validate_email(datasite, "datasite")
            
            # Parse path components
            path_parts = [part for part in parsed.path.split('/') if part]
            
            if len(path_parts) < 4:
                raise ValidationError("Invalid syft URL path format - too few components")
            
            if path_parts[0] != APP_DATA_DIR:
                raise ValidationError(f"Expected '{APP_DATA_DIR}' in path, got '{path_parts[0]}'")
            
            if path_parts[2] != RPC_DIR:
                raise ValidationError(f"Expected '{RPC_DIR}' in path, got '{path_parts[2]}'")
            
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
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to parse syft URL '{syft_url}': {e}")
    
    @staticmethod
    def syft_url_to_filesystem_path(filesystem: SyftBoxFilesystem, syft_url: str) -> Path:
        """Convert a syft:// URL to its corresponding filesystem path.
        
        Args:
            filesystem: SyftBoxFilesystem instance
            syft_url: The syft:// URL to convert
            
        Returns:
            Corresponding filesystem path
            
        Raises:
            ValidationError: If URL format is invalid
        """
        parsed = SyftURLBuilder.parse_syft_url(syft_url)
        
        # Build path: datasites/{datasite}/app_data/{app_name}/rpc/{endpoint}
        return filesystem.rpc_directory_path(
            parsed['datasite'], 
            parsed['app_name']
        ) / parsed['endpoint']
    
    @staticmethod
    def filesystem_path_to_syft_url(filesystem: SyftBoxFilesystem, filesystem_path: Path) -> Optional[str]:
        """Convert a filesystem path to its corresponding syft:// URL.
        
        Args:
            filesystem: SyftBoxFilesystem instance
            filesystem_path: The filesystem path to convert
            
        Returns:
            Corresponding syft:// URL or None if path doesn't match expected format
        """
        try:
            # Get relative path from datasites directory
            rel_path = filesystem_path.relative_to(filesystem.datasites_path)
            path_parts = rel_path.parts
            
            # Expected format: {datasite}/app_data/{service_name}/rpc/{endpoint...}
            if len(path_parts) < 5:
                return None
            
            if path_parts[1] != APP_DATA_DIR or path_parts[3] != RPC_DIR:
                return None
            
            datasite = path_parts[0]
            service_name = path_parts[2]
            endpoint = '/'.join(path_parts[4:])
            
            # Validate datasite email
            if not EmailValidator.is_valid_email(datasite):
                return None
            
            return SyftURLBuilder.build_syft_url(datasite, service_name, endpoint)
            
        except (ValueError, IndexError, ValidationError):
            return None


# Convenience functions for URL validation
def validate_syft_url(syft_url: str) -> bool:
    """Validate if a string is a properly formatted syft URL."""
    try:
        SyftURLBuilder.parse_syft_url(syft_url)
        return True
    except ValidationError:
        return False


def extract_service_info_from_url(syft_url: str) -> Dict[str, str]:
    """Extract service information from syft URL."""
    try:
        parsed = SyftURLBuilder.parse_syft_url(syft_url)
        
        return {
            'datasite': parsed['datasite'],
            'service_name': parsed['app_name'],
            'endpoint': parsed['endpoint'],
            'display_name': f"{parsed['app_name']} by {parsed['datasite']}"
        }
        
    except ValidationError as e:
        logger.error(f"Failed to extract service info from URL '{syft_url}': {e}")
        return {}