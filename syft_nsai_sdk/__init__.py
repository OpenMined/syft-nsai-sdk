"""
SyftBox SDK

SDK for discovering and interacting with models in the SyftBox ecosystem.
Assumes ~/.syftbox/config.json exists for authentication.
"""

from typing import Dict, List, Optional, Any

# Import the classes first
from .sdk import SyftBoxSDK
from .models import ModelObject  
from .types import ServiceInfo, ModelMetadata
from .config import SyftBoxConfig

# Version info
__version__ = "0.1.0"
__author__ = "OpenMined"

# Global SDK instance - lazy initialization to avoid import issues
_sdk = None

def _get_sdk():
    """Get or create the global SDK instance."""
    global _sdk
    if _sdk is None:
        _sdk = SyftBoxSDK()
    return _sdk

# Public API - Convenience functions
def find_models(name: Optional[str] = None, 
               tags: Optional[List[str]] = None,
               owners: Optional[List[str]] = None) -> List[ModelObject]:
    """Find models matching criteria."""
    return _get_sdk().find_models(name=name, tags=tags, owners=owners)

def get_models(owners: Optional[List[str]] = None) -> List[ModelObject]:
    """Get all available models."""
    return _get_sdk().get_models(owners=owners)

def get_model(name: str, owner: Optional[str] = None) -> Optional[ModelObject]:
    """Get specific model by name.""" 
    return _get_sdk().get_model(name=name, owner=owner)

def display_models(models: List[ModelObject]) -> None:
    """Display models in table format."""
    return _get_sdk().display_models(models)

def status() -> Dict[str, Any]:
    """Show SDK status."""
    return _get_sdk().status()

def current_user() -> Optional[str]:
    """Get current authenticated user."""
    return _get_sdk().current_user

def is_authenticated() -> bool:
    """Check if authenticated."""
    return _get_sdk().is_authenticated

# Export everything
__all__ = [
    "find_models",
    "get_models", 
    "get_model",
    "display_models",
    "status",
    "current_user",
    "is_authenticated",
    "SyftBoxSDK",
    "ModelObject",
    "ServiceInfo", 
    "ModelMetadata",
    "SyftBoxConfig",
]