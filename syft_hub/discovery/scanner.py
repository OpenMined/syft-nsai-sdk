"""
File system scanner for discovering SyftBox services across datasites
"""
import os
import logging

from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING

# from ..core.exceptions import ConfigurationError
from .filesystem import SyftURLBuilder

if TYPE_CHECKING:
    from ..core.config import SyftBoxConfig

logger = logging.getLogger(__name__)

class ServiceScanner:
    """Scanner for discovering services across SyftBox datasites."""
    
    def __init__(self, syftbox_config: 'SyftBoxConfig'):
        self.config = syftbox_config
        self.datasites_path = syftbox_config.datasites_path

    def scan_all_datasites(self, exclude_current_user: bool = False) -> List[Path]:
        """Scan all datasites for published services.
        
        Args:
            exclude_current_user: If True, skip current user's datasite
            
        Returns:
            List of paths to metadata.json files
        """
        if not self.datasites_path.exists():
            logger.warning(f"Datasites directory not found: {self.datasites_path}")
            return []
        
        metadata_paths = []
        current_user_email = self.config.email if exclude_current_user else None
        
        for datasite_dir in self.datasites_path.iterdir():
            if not datasite_dir.is_dir():
                continue
            
            # Skip current user if requested
            if current_user_email and datasite_dir.name == current_user_email:
                continue
            
            # Skip directories that don't look like email addresses
            if '@' not in datasite_dir.name:
                continue
            
            try:
                paths = self.scan_datasite(datasite_dir.name)
                metadata_paths.extend(paths)
            except Exception as e:
                logger.warning(f"Error scanning datasite {datasite_dir.name}: {e}")
                continue
        
        logger.debug(f"Found {len(metadata_paths)} services across {len(list(self.datasites_path.iterdir()))} datasites")
        return metadata_paths
    
    def scan_datasite(self, datasite: str) -> List[Path]:
        """Scan a specific datasite for published services.
        
        Args:
            datasite: Email of the datasite
            
        Returns:
            List of paths to metadata.json files for this datasite
        """
        
        # Look for published routers in public/routers/
        routers_path = SyftURLBuilder.build_routers_path(self.datasites_path, datasite)

        if not routers_path.exists():
            logger.debug(f"No published routers found for {datasite}")
            return []
        
        metadata_paths = []
        
        for service_dir in routers_path.iterdir():
            if not service_dir.is_dir():
                continue
            
            metadata_path = SyftURLBuilder.build_metadata_path(self.datasites_path, datasite, service_dir.name)
            if metadata_path.exists() and self.is_valid_metadata_file(metadata_path):
                metadata_paths.append(metadata_path)
            else:
                logger.debug(f"Invalid or missing metadata: {metadata_path}")
        
        logger.debug(f"Found {len(metadata_paths)} services for {datasite}")
        return metadata_paths
    
    def find_metadata_files(self, service_name: Optional[str] = None, 
                           datasite: Optional[str] = None) -> List[Path]:
        """Find specific metadata files with optional filtering.
        
        Args:
            service_name: Optional service name to filter by
            datasite: Optional datasite email to filter by
            
        Returns:
            List of matching metadata.json paths
        """
        if datasite and service_name:
            metadata_path = SyftURLBuilder.build_metadata_path(self.datasites_path, datasite, service_name)
            return [metadata_path] if metadata_path.exists() else []
        elif datasite:
            # Search specific datasite
            return self.scan_datasite(datasite)
        else:
            # Search all datasites
            all_paths = self.scan_all_datasites()
        
            if not service_name:
                return all_paths
            
            # Filter by service name
            filtered_paths = []
            for path in all_paths:
                if path.parent.name == service_name:
                    filtered_paths.append(path)
            
            return filtered_paths
    
    def is_valid_metadata_file(self, metadata_path: Path) -> bool:
        """Check if a metadata.json file is valid and readable.
        
        Args:
            metadata_path: Path to metadata.json file
            
        Returns:
            True if file exists and is readable JSON
        """
        try:
            if not metadata_path.exists() or metadata_path.stat().st_size == 0:
                return False
            
            # Try to read as JSON (basic validation)
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                json.load(f)
            
            return True
        except (json.JSONDecodeError, PermissionError, OSError):
            return False
    
    def is_valid_service_directory(self, service_path: Path) -> bool:
        """Check if a directory contains a valid service.
        
        Args:
            service_path: Path to potential service directory
            
        Returns:
            True if directory contains valid metadata.json
        """
        if not service_path.is_dir():
            return False
        
        metadata_path = service_path / "metadata.json"
        return self.is_valid_metadata_file(metadata_path)
    
    def get_rpc_schema_path(self, metadata_path: Path) -> Optional[Path]:
        """Find the RPC schema file for a given service.
        
        Args:
            metadata_path: Path to the service's metadata.json
            
        Returns:
            Path to rpc.schema.json if found, None otherwise
        """
        # Extract service info from metadata path
        # Expected structure: datasites/{datasite}/public/routers/{service}/metadata.json
        try:
            service_name = metadata_path.parent.name
            datasite = metadata_path.parent.parent.parent.parent.name
            
            # Use centralized path builder for RPC schema
            rpc_schema_path = SyftURLBuilder.build_rpc_schema_path(self.datasites_path, datasite, service_name)
            if rpc_schema_path.exists():
                return rpc_schema_path
            
        except (IndexError, AttributeError):
            pass
        
        # Fallback: look in same directory as metadata
        fallback_path = metadata_path.parent / "rpc.schema.json"
        if fallback_path.exists():
            return fallback_path
        
        return None
    
    def get_service_statistics(self) -> Dict[str, int]:
        """Get statistics about discovered services.
        
        Returns:
            Dictionary with service discovery statistics
        """
        all_paths = self.scan_all_datasites()
        
        # Count by datasite
        datasites = {}
        total_services = len(all_paths)
        
        for path in all_paths:
            try:
                # Extract datasite from path
                datasite = path.parent.parent.parent.parent.name
                datasites[datasite] = datasites.get(datasite, 0) + 1
            except (IndexError, AttributeError):
                continue
        
        return {
            "total_services": total_services,
            "total_datasites": len(datasites),
            "services_per_datasite": datasites,
            "average_services_per_datasite": total_services / len(datasites) if datasites else 0
        }
    
    def list_datasites(self) -> List[str]:
        """List all available datasites.
        
        Returns:
            List of datasite email addresses
        """
        if not self.datasites_path.exists():
            return []
        
        datasites = []
        for item in self.datasites_path.iterdir():
            if item.is_dir() and '@' in item.name:
                datasites.append(item.name)
        
        return sorted(datasites)
    
    def get_services_for_datasite(self, datasite_email: str) -> List[str]:
        """Get list of service names for a specific datasite.
        
        Args:
            datasite_email: Email of the service datasite
            
        Returns:
            List of service names owned by this user
        """
        metadata_paths = self.scan_datasite(datasite_email)
        service_names = []
        
        for path in metadata_paths:
            service_name = path.parent.name
            service_names.append(service_name)
        
        return sorted(service_names)


class FastScanner:
    """Optimized scanner for large numbers of services."""
    
    def __init__(self, syftbox_config: 'SyftBoxConfig'):
        self.config = syftbox_config
        self.datasites_path = syftbox_config.datasites_path
        self._cache: Optional[Dict[str, List[Path]]] = None
    
    def scan_with_cache(self, force_refresh: bool = False) -> List[Path]:
        """Scan with caching for better performance.
        
        Args:
            force_refresh: If True, ignore cache and rescan
            
        Returns:
            List of paths to metadata.json files
        """
        if self._cache is None or force_refresh:
            scanner = ServiceScanner(self.config)
            all_paths = scanner.scan_all_datasites()
            
            # Cache by datasite for faster lookups
            self._cache = {}
            for path in all_paths:
                try:
                    datasite = path.parent.parent.parent.parent.name
                    if datasite not in self._cache:
                        self._cache[datasite] = []
                    self._cache[datasite].append(path)
                except (IndexError, AttributeError):
                    continue
            
            logger.debug(f"Cached {len(all_paths)} services from {len(self._cache)} datasites")
        
        # Return flattened list
        all_paths = []
        for paths in self._cache.values():
            all_paths.extend(paths)
        
        return all_paths
    
    def get_cached_services_for_datasite(self, datasite_email: str) -> List[Path]:
        """Get cached services for specific datasite.
        
        Args:
            datasite_email: Email of the service datasite
            
        Returns:
            List of cached metadata paths for this datasite
        """
        if self._cache is None:
            self.scan_with_cache()
        
        return self._cache.get(datasite_email, [])
    
    def get_service_path(self, datasite: str, service_name: str) -> Optional[Path]:
        """Get direct path to a specific service's metadata file.
        
        Args:
            datasite: Datasite email
            service_name: Service name
            
        Returns:
            Path to metadata.json if it exists, None otherwise
        """
        metadata_path = SyftURLBuilder.build_metadata_path(self.datasites_path, datasite, service_name)
        return metadata_path if metadata_path.exists() else None
    
    def clear_cache(self):
        """Clear the service cache."""
        self._cache = None