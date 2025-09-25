"""
SyftBox Authentication Client for managing user authentication
"""
import logging
from typing import Optional
from pathlib import Path

from ..core.config import SyftBoxConfig, ConfigManager
from ..core.exceptions import AuthenticationError, SyftBoxNotFoundError

logger = logging.getLogger(__name__)


class SyftBoxAuthClient:
    """Client for handling SyftBox authentication operations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize auth client.
        
        Args:
            config_path: Optional custom path to SyftBox config file
        """
        self.config_manager = ConfigManager(config_path)
        self._config: Optional[SyftBoxConfig] = None
        
    @property
    def config(self) -> SyftBoxConfig:
        """Get SyftBox configuration."""
        if self._config is None:
            try:
                self._config = self.config_manager.config
            except SyftBoxNotFoundError:
                # Fallback for guest mode
                logger.warning("SyftBox config not found, using guest mode")
                return None
        return self._config
    
    def get_user_email(self) -> str:
        """Get authenticated user email.
        
        Returns:
            User email from SyftBox config, or guest email if not configured
        """
        try:
            if self.config and self.config.email:
                return self.config.email
        except Exception as e:
            logger.warning(f"Failed to get user email from config: {e}")
        
        # Fallback to guest mode
        return "guest@syftbox.net"
    
    async def get_auth_token(self) -> Optional[str]:
        """Get authentication token.
        
        Returns:
            Auth token if available, None otherwise
        """
        try:
            if self.config and self.config.refresh_token:
                return self.config.refresh_token
        except Exception as e:
            logger.warning(f"Failed to get auth token: {e}")
        
        return None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated.
        
        Returns:
            True if user has valid authentication, False otherwise
        """
        try:
            return (self.config is not None and 
                   self.config.email is not None and 
                   self.config.refresh_token is not None)
        except Exception:
            return False
    
    async def close(self):
        """Close auth client and cleanup resources."""
        # No cleanup needed for now
        pass
