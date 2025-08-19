"""
File system scanner for discovering SyftBox models across datasites
"""
import os
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Set
import logging

from ..core.config import SyftBoxConfig
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ModelScanner:
    """Scanner for discovering models across SyftBox datasites."""
    
    def __init__(self, syftbox_config: SyftBoxConfig):
        self.config = syftbox_config
        self.datasites_path = syftbox_config.datasites_path
    
    def scan_all_datasites(self, exclude_current_user: bool = False) -> List[Path]:
        """Scan all datasites for published models.
        
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
        
        logger.info(f"Found {len(metadata_paths)} models across {len(list(self.datasites_path.iterdir()))} datasites")
        return metadata_paths
    
    def scan_datasite(self, owner_email: str) -> List[Path]:
        """Scan a specific datasite for published models.
        
        Args:
            owner_email: Email of the datasite owner
            
        Returns:
            List of paths to metadata.json files for this datasite
        """
        datasite_path = self.datasites_path / owner_email
        
        if not datasite_path.exists():
            logger.debug(f"Datasite not found: {datasite_path}")
            return []
        
        # Look for published routers in public/routers/
        routers_path = datasite_path / "public" / "routers"
        
        if not routers_path.exists():
            logger.debug(f"No published routers found for {owner_email}")
            return []
        
        metadata_paths = []
        
        for model_dir in routers_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists() and self.is_valid_metadata_file(metadata_path):
                metadata_paths.append(metadata_path)
            else:
                logger.debug(f"Invalid or missing metadata: {metadata_path}")
        
        logger.debug(f"Found {len(metadata_paths)} models for {owner_email}")
        return metadata_paths
    
    def find_metadata_files(self, model_name: Optional[str] = None, 
                           owner_email: Optional[str] = None) -> List[Path]:
        """Find specific metadata files with optional filtering.
        
        Args:
            model_name: Optional model name to filter by
            owner_email: Optional owner email to filter by
            
        Returns:
            List of matching metadata.json paths
        """
        if owner_email:
            # Search specific datasite
            all_paths = self.scan_datasite(owner_email)
        else:
            # Search all datasites
            all_paths = self.scan_all_datasites()
        
        if not model_name:
            return all_paths
        
        # Filter by model name
        filtered_paths = []
        for path in all_paths:
            if path.parent.name == model_name:
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
    
    def is_valid_model_directory(self, model_path: Path) -> bool:
        """Check if a directory contains a valid model.
        
        Args:
            model_path: Path to potential model directory
            
        Returns:
            True if directory contains valid metadata.json
        """
        if not model_path.is_dir():
            return False
        
        metadata_path = model_path / "metadata.json"
        return self.is_valid_metadata_file(metadata_path)
    
    def get_rpc_schema_path(self, metadata_path: Path) -> Optional[Path]:
        """Find the RPC schema file for a given model.
        
        Args:
            metadata_path: Path to the model's metadata.json
            
        Returns:
            Path to rpc.schema.json if found, None otherwise
        """
        # Extract model info from metadata path
        # Expected structure: datasites/{owner}/public/routers/{model}/metadata.json
        try:
            model_name = metadata_path.parent.name
            owner_email = metadata_path.parent.parent.parent.parent.name
            
            # Expected RPC schema location: datasites/{owner}/app_data/{model}/rpc/rpc.schema.json
            rpc_schema_path = (self.datasites_path / owner_email / 
                              "app_data" / model_name / "rpc" / "rpc.schema.json")
            
            if rpc_schema_path.exists():
                return rpc_schema_path
        except (IndexError, AttributeError):
            pass
        
        # Fallback: look in same directory as metadata
        fallback_path = metadata_path.parent / "rpc.schema.json"
        if fallback_path.exists():
            return fallback_path
        
        return None
    
    def get_model_statistics(self) -> Dict[str, int]:
        """Get statistics about discovered models.
        
        Returns:
            Dictionary with model discovery statistics
        """
        all_paths = self.scan_all_datasites()
        
        # Count by owner
        owners = {}
        total_models = len(all_paths)
        
        for path in all_paths:
            try:
                # Extract owner from path
                owner = path.parent.parent.parent.parent.name
                owners[owner] = owners.get(owner, 0) + 1
            except (IndexError, AttributeError):
                continue
        
        return {
            "total_models": total_models,
            "total_owners": len(owners),
            "models_per_owner": owners,
            "average_models_per_owner": total_models / len(owners) if owners else 0
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
    
    def get_models_for_owner(self, owner_email: str) -> List[str]:
        """Get list of model names for a specific owner.
        
        Args:
            owner_email: Email of the model owner
            
        Returns:
            List of model names owned by this user
        """
        metadata_paths = self.scan_datasite(owner_email)
        model_names = []
        
        for path in metadata_paths:
            model_name = path.parent.name
            model_names.append(model_name)
        
        return sorted(model_names)


class FastScanner:
    """Optimized scanner for large numbers of models."""
    
    def __init__(self, syftbox_config: SyftBoxConfig):
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
            scanner = ModelScanner(self.config)
            all_paths = scanner.scan_all_datasites()
            
            # Cache by owner for faster lookups
            self._cache = {}
            for path in all_paths:
                try:
                    owner = path.parent.parent.parent.parent.name
                    if owner not in self._cache:
                        self._cache[owner] = []
                    self._cache[owner].append(path)
                except (IndexError, AttributeError):
                    continue
            
            logger.info(f"Cached {len(all_paths)} models from {len(self._cache)} owners")
        
        # Return flattened list
        all_paths = []
        for paths in self._cache.values():
            all_paths.extend(paths)
        
        return all_paths
    
    def get_cached_models_for_owner(self, owner_email: str) -> List[Path]:
        """Get cached models for specific owner.
        
        Args:
            owner_email: Email of the model owner
            
        Returns:
            List of cached metadata paths for this owner
        """
        if self._cache is None:
            self.scan_with_cache()
        
        return self._cache.get(owner_email, [])
    
    def clear_cache(self):
        """Clear the model cache."""
        self._cache = None