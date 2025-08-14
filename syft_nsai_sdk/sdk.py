"""
Main SDK class for model discovery and management.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from tabulate import tabulate

from .config import SyftBoxConfig
from .models import ModelObject
from .types import ModelMetadata, ServiceInfo
from .logger import get_logger

# import pandas as pd
# from IPython.display import display

logger = get_logger(__name__)

class SyftBoxSDK:
    """
    SyftBox SDK for model discovery and interaction.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = SyftBoxConfig(config_path)
        
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.config.exists and self.config.email is not None
    
    @property
    def current_user(self) -> Optional[str]:
        """Get current user email."""
        return self.config.email
    
    def _check_prerequisites(self) -> bool:
        """Check if SDK prerequisites are met."""
        if self.config.exists and self.config.email:
            return True
        
        if not self.config.exists:
            logger.error("SyftBox config not found at ~/.syftbox/config.json")
            logger.info("Please set up SyftBox first to create the config file")
            return False
        
        return True
    
    def _get_datasites_path(self) -> Optional[Path]:
        """Get path to datasites directory."""
        data_dir = self.config.data_dir
        if not data_dir:
            return None
        return data_dir / "datasites"
    
    def _load_metadata(self, metadata_path: Path) -> Optional[ModelMetadata]:
        """Load model metadata from metadata.json file."""
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            services = []
            for service_data in data.get('services', []):
                services.append(ServiceInfo(
                    type=service_data.get('type'),
                    pricing=service_data.get('pricing', 0.0),
                    charge_type=service_data.get('charge_type', 'per_request'),
                    enabled=service_data.get('enabled', True)
                ))
            
            return ModelMetadata(
                name=data.get('project_name'),
                owner=data.get('author'),
                description=data.get('description', ''),
                summary=data.get('summary', ''),
                tags=data.get('tags', []),
                services=services,
                version=data.get('version', '1.0.0'),
                publish_date=data.get('publish_date', ''),
                code_hash=data.get('code_hash', ''),
                delegate_email=data.get('delegate_email')
            )
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return None
    
    def _scan_models(self, owner_filter: Optional[List[str]] = None) -> List[ModelObject]:
        """Scan datasites for available models."""
        if not self._check_prerequisites():
            return []
        
        models = []
        datasites_path = self._get_datasites_path()
        
        if not datasites_path or not datasites_path.exists():
            logger.warning(f"Datasites directory not found at {datasites_path}")
            logger.info("Make sure SyftBox is properly set up and synced")
            return []
        
        for datasite in datasites_path.iterdir():
            if not datasite.is_dir():
                continue
            
            owner = datasite.name
            
            if owner_filter and owner not in owner_filter:
                continue
            
            routers_path = datasite / "public" / "routers"
            if not routers_path.exists():
                continue
            
            for model_dir in routers_path.iterdir():
                if not model_dir.is_dir():
                    continue
                
                metadata_path = model_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                
                metadata = self._load_metadata(metadata_path)
                if not metadata:
                    continue
                
                models.append(ModelObject(metadata, self.config))
        
        return models
    
    def find_models(self, 
                   name: Optional[str] = None, 
                   tags: Optional[List[str]] = None,
                   owners: Optional[List[str]] = None) -> List[ModelObject]:
        """
        Find models based on search criteria.
        
        Args:
            name: Model name to search for (partial match)
            tags: List of tags to filter by
            owners: List of owner emails to filter by
            
        Returns:
            List of matching ModelObject instances
        """
        all_models = self._scan_models(owner_filter=owners)
        filtered_models = []
        
        for model in all_models:
            if name and name.lower() not in model.name.lower():
                continue
            
            if tags:
                model_tags = [tag.lower() for tag in model.tags]
                search_tags = [tag.lower() for tag in tags]
                if not any(tag in model_tags for tag in search_tags):
                    continue
            
            filtered_models.append(model)
        
        return filtered_models
    
    def get_models(self, owners: Optional[List[str]] = None) -> List[ModelObject]:
        """Get all available models."""
        return self._scan_models(owner_filter=owners)
    
    def get_model(self, name: str, owner: Optional[str] = None) -> Optional[ModelObject]:
        """Get a specific model by name."""
        owner_filter = [owner] if owner else None
        models = self.find_models(name=name, owners=owner_filter)
        
        for model in models:
            if model.name == name:
                if owner is None or model.owner == owner:
                    return model
        return models[0] if models else None
    
    def display_models(self, models: List[ModelObject]) -> None:
        """Display models in a nice table format."""
        if not models:
            logger.info("No models found.")
            return
        
        headers = ["Name", "Owner", "Tags", "Services", "Summary", "Status"]
        rows = []
        
        for model in models:
            tags_str = ", ".join(model.tags) if model.tags else "None"
            services_str = ", ".join(model.services) if model.services else "None"
            summary_str = model.summary[:50] + "..." if len(model.summary) > 50 else model.summary
            
            rows.append([
                model.name,
                model.owner,
                tags_str,
                services_str,
                summary_str,
                model.status
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))


    # def format_models(self, models: List[ModelObject]) -> None:
    #     """Display models in a nice table format using pandas."""
    #     if not models:
    #         print("No models found.")
    #         return
        
    #     data = []
    #     for model in models:
    #         tags_str = ", ".join(model.tags) if model.tags else "None"
    #         services_str = ", ".join(model.services) if model.services else "None"
    #         summary_str = model.summary[:50] + "..." if len(model.summary) > 50 else model.summary
            
    #         data.append({
    #             "Name": model.name,
    #             "Owner": model.owner,
    #             "Tags": tags_str,
    #             "Services": services_str,
    #             "Summary": summary_str,
    #             "Status": model.status
    #         })

    #     df = pd.DataFrame(data)
    #     display(df)
    
    def status(self) -> Dict[str, Any]:
        """Get SDK status information."""
        config_exists = self.config.exists
        
        status_info = {
            "config_exists": config_exists,
            "authenticated": config_exists and self.config.email is not None,
            "user": self.config.email if config_exists else None,
            "server": self.config.server_url if config_exists else "https://syftbox.net",
            "data_dir": str(self.config.data_dir) if config_exists and self.config.data_dir else None
        }

        logger.info("SyftBox SDK Status:")
        logger.info(f"   Config Exists: {'Yes' if status_info['config_exists'] else '❌ No'}")
        logger.info(f"   Authenticated: {'Yes' if status_info['authenticated'] else '❌ No'}")

        if status_info['user']:
            logger.info(f"   User: {status_info['user']}")
        logger.info(f"   Server: {status_info['server']}")
        if status_info['data_dir']:
            logger.info(f"   Data Dir: {status_info['data_dir']}")

        if not status_info['config_exists']:
            logger.info("Expected config format at ~/.syftbox/config.json:")
            logger.info("   {")
            logger.info('     "email": "your@email.com",')
            logger.info('     "server_url": "https://syftbox.net",')
            logger.info('     "refresh_token": "your_token",')
            logger.info('     "data_dir": "/path/to/SyftBox"')
            logger.info("   }")

        return status_info