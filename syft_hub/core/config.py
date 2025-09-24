"""
SyftBox configuration management
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .exceptions import SyftBoxNotFoundError, ConfigurationError
from ..core.settings import settings
from ..discovery.filesystem import SyftBoxFilesystem
from ..utils.constants import (
    SYFTBOX_DIR, 
    CONFIG_FILENAME, 
    DESKTOP_RELEASES_URL, 
    QUICK_INSTALL_URL, 
    CLI_DOCS_URL, 
    DESKTOP_DOCS_URL
)
from ..utils.validator import (
    EmailValidator, 
    URLValidator, 
    ConfigValidator, 
    ProcessValidator,
    validate_config_file, 
    is_syftbox_running,
)

logger = logging.getLogger(__name__)

@dataclass
class SyftBoxConfig:
    """SyftBox configuration loaded from ~/.syftbox/config.json"""
    
    data_dir: Path
    email: str
    server_url: str
    refresh_token: Optional[str] = None
    config_path: Optional[Path] = None
    _filesystem: Optional[SyftBoxFilesystem] = None
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'SyftBoxConfig':
        """Load SyftBox configuration from file.
        
        Args:
            config_path: Optional custom path to config file
            
        Returns:
            SyftBoxConfig instance
            
        Raises:
            SyftBoxNotFoundError: If config file not found
            ConfigurationError: If config is invalid
        """
        # Determine config path
        if config_path is None:
            config_path = Path.home() / SYFTBOX_DIR / CONFIG_FILENAME
        
        if not config_path.exists():
            raise SyftBoxNotFoundError(
                f"SyftBox config not found at {config_path}.\n"
                "Please install and setup SyftBox first.\n\n"
                f"Install: curl -fsSL {QUICK_INSTALL_URL} | sh\n"
                "Setup: syftbox setup"
            )
        
        # Validate and load config
        try:
            config_data = validate_config_file(config_path)
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load config: {e}", str(config_path))
        
        # Create config object with validated data
        return cls(
            data_dir=Path(config_data["data_dir"]),
            email=config_data["email"],  # Already validated by validate_config_file
            server_url=URLValidator.normalize_server_url(config_data["server_url"]),
            refresh_token=config_data.get("refresh_token"),
            config_path=config_path
        )
    
    @property
    def filesystem(self) -> SyftBoxFilesystem:
        """Get filesystem utility instance."""
        if self._filesystem is None:
            self._filesystem = SyftBoxFilesystem(self.data_dir)
        return self._filesystem
    
    @property
    def datasites_path(self) -> Path:
        """Path to the datasites directory."""
        return self.filesystem.datasites_path
    
    @property
    def my_datasite_path(self) -> Path:
        """Path to the current user's datasite."""
        return self.filesystem.datasite_path(self.email)
    
    @property
    def cache_server_url(self) -> str:
        """URL of the cache server (same as server_url in SyftBox)."""
        return self.server_url
    
    def validate_paths(self) -> None:
        """Validate that required paths exist.
        
        Raises:
            ConfigurationError: If required paths don't exist
        """
        try:
            self.filesystem.validate_structure(self.email)
        except Exception as e:
            raise ConfigurationError(str(e), str(self.config_path))
    
    def get_datasite_path(self, email: str) -> Path:
        """Get path to a specific datasite.
        
        Args:
            email: Email of the datasite
            
        Returns:
            Path to datasite directory
        """
        return self.filesystem.datasite_path(email)
    
    def list_datasites(self) -> List[str]:
        """List all available datasites.
        
        Returns:
            List of datasite email addresses
        """
        return self.filesystem.list_datasites()
    
    def list_services(self, datasite_email: Optional[str] = None) -> List[str]:
        """List services for a datasite.
        
        Args:
            datasite_email: Email of datasite (defaults to current user)
            
        Returns:
            List of service names
        """
        email = datasite_email or self.email
        return self.filesystem.list_services(email)
    
    def get_service_metadata_path(self, service_name: str, datasite_email: Optional[str] = None) -> Path:
        """Get path to service metadata file.
        
        Args:
            service_name: Name of the service
            datasite_email: Email of datasite (defaults to current user)
            
        Returns:
            Path to metadata.json file
        """
        email = datasite_email or self.email
        return self.filesystem.metadata_path(email, service_name)
    
    def get_app_data_path(self, app_name: str, datasite_email: Optional[str] = None) -> Path:
        """Get path to app data directory.
        
        Args:
            app_name: Name of the application
            datasite_email: Email of datasite (defaults to current user)
            
        Returns:
            Path to app_data directory
        """
        email = datasite_email or self.email
        return self.filesystem.app_data_path(email, app_name)
    
    def get_rpc_directory_path(self, app_name: str, datasite_email: Optional[str] = None) -> Path:
        """Get path to RPC directory.
        
        Args:
            app_name: Name of the application
            datasite_email: Email of datasite (defaults to current user)
            
        Returns:
            Path to RPC directory
        """
        email = datasite_email or self.email
        return self.filesystem.rpc_directory_path(email, app_name)
    
    def service_exists(self, service_name: str, datasite_email: Optional[str] = None) -> bool:
        """Check if a service exists.
        
        Args:
            service_name: Name of the service
            datasite_email: Email of datasite (defaults to current user)
            
        Returns:
            True if service exists, False otherwise
        """
        email = datasite_email or self.email
        return self.filesystem.check_service_exists(email, service_name)
    
    def rpc_endpoint_exists(self, app_name: str, endpoint: str, datasite_email: Optional[str] = None) -> bool:
        """Check if an RPC endpoint exists.
        
        Args:
            app_name: Name of the application
            endpoint: RPC endpoint path
            datasite_email: Email of datasite (defaults to current user)
            
        Returns:
            True if endpoint exists, False otherwise
        """
        email = datasite_email or self.email
        return self.filesystem.check_rpc_endpoint_exists(email, app_name, endpoint)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            "data_dir": str(self.data_dir),
            "email": self.email,
            "server_url": self.server_url,
            "refresh_token": self.refresh_token,
            "config_path": str(self.config_path) if self.config_path else None
        }


class ConfigManager:
    """Manages SyftBox configuration with caching and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager.
        
        Args:
            config_path: Optional custom path to config file
        """
        self.config_path = config_path
        self._config: Optional[SyftBoxConfig] = None
    
    @property
    def config(self) -> SyftBoxConfig:
        """Get cached config, loading if necessary.
        
        Returns:
            SyftBoxConfig instance
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self) -> SyftBoxConfig:
        """Load and validate SyftBox configuration.
        
        Returns:
            SyftBoxConfig instance
            
        Raises:
            SyftBoxNotFoundError: If config not found
            ConfigurationError: If config is invalid
        """
        config = SyftBoxConfig.load(self.config_path)
        config.validate_paths()
        return config
    
    def reload_config(self) -> SyftBoxConfig:
        """Force reload configuration from disk.
        
        Returns:
            SyftBoxConfig instance
        """
        self._config = None
        return self.config
    
    def is_syftbox_installed(self) -> bool:
        """Check if SyftBox is installed and configured.
        
        Returns:
            True if installed and configured, False otherwise
        """
        try:
            # Use custom path if provided, otherwise default
            config_path = self.config_path or (Path.home() / SYFTBOX_DIR / CONFIG_FILENAME)
            
            if config_path.exists():
                # Validate the config file
                try:
                    validate_config_file(config_path)
                    logger.info(f"Using config credentials at path: {config_path}")
                    return True
                except ConfigurationError:
                    logger.warning(f"Config file exists but is invalid: {config_path}")
                    return False
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking SyftBox installation: {e}")
            return False
    
    def is_syftbox_running(self) -> bool:
        """Check if SyftBox process is actually running.
        
        Returns:
            True if SyftBox is running, False otherwise
        """
        return ProcessValidator.is_syftbox_running()
    
    def get_installation_instructions1(self) -> str:
        """Instructions for installing SyftBox.
        
        Returns:
            Installation instructions string
        """
        if self.config_path:
            # Custom path was provided but config not found
            return (
                f"SyftBox config not found at: {self.config_path}\n\n"
                "Options:\n"
                "1. Verify the config path is correct\n"
                "2. Install SyftBox if not installed:\n"
                f"   • Desktop App: {DESKTOP_RELEASES_URL}\n"
                f"   • Quick install (terminal): curl -fsSL {QUICK_INSTALL_URL} | sh\n"
                "   • Google Colab: Run the following in a cell:\n"
                "     !pip install syft-installer\n"
                "     import syft_installer as si\n"
                "     si.install_and_run_if_needed()\n"
                "3. Run 'syftbox setup' to create config at this location"
            )
        else:
            # Default path check failed
            return (
                f"SyftBox config not found at default location (~/{SYFTBOX_DIR}/{CONFIG_FILENAME})\n\n"
                "Options:\n"
                "1. Install SyftBox:\n"
                f"   • Desktop App: {DESKTOP_RELEASES_URL}\n"
                f"   • Quick install (terminal): curl -fsSL {QUICK_INSTALL_URL} | sh\n"
                "   • Google Colab: Run the following in a cell:\n"
                "     !pip install syft-installer\n"
                "     import syft_installer as si\n"
                "     si.install_and_run_if_needed()\n"
                "2. If installed in custom location, provide syftbox_config_path parameter\n\n"
                "For detailed instructions, visit the official repositories above."
            )

    def get_installation_instructions(self) -> str:
        """Instructions for installing SyftBox.
        
        Returns:
            Installation instructions string
        """
        # Determine what paths were actually checked
        default_path = Path.home() / SYFTBOX_DIR / CONFIG_FILENAME
        
        if self.config_path and self.config_path != default_path:
            # Custom path was explicitly provided
            return (
                f"SyftBox config not found at custom path: {self.config_path}\n\n"
                "Options:\n"
                "1. Verify the custom config path is correct\n"
                "2. Use default location by calling Client() without syftbox_config_path parameter\n"
                "3. Install SyftBox if not installed:\n"
                f"   • Desktop App: {DESKTOP_RELEASES_URL}\n"
                f"   • Quick install (terminal): curl -fsSL {QUICK_INSTALL_URL} | sh\n"
                "4. Run 'syftbox setup' to create config"
            )
        else:
            # No custom path provided, checked default location
            return (
                f"SyftBox config not found at default location: {default_path}\n\n"
                "This means SyftBox is not installed or not set up on this system.\n\n"
                "Options:\n"
                "1. Install SyftBox:\n"
                f"   • Desktop App: {DESKTOP_RELEASES_URL}\n"
                f"   • Quick install (terminal): curl -fsSL {QUICK_INSTALL_URL} | sh\n"
                "   • Google Colab: Run the following in a cell:\n"
                "     !pip install syft-installer\n"
                "     import syft_installer as si\n"
                "     si.install_and_run_if_needed()\n"
                "2. If SyftBox is installed elsewhere, provide syftbox_config_path parameter\n\n"
                "For detailed instructions, visit the official repositories above."
            )
    
    def get_startup_instructions(self) -> str:
        """Instructions for starting an already installed SyftBox.
        
        Returns:
            Startup instructions string
        """
        return (
            "SyftBox is installed but not running.\n\n"
            "To start SyftBox:\n"
            f"• CLI: Run 'syftbox' in terminal with curl -fsSL {QUICK_INSTALL_URL} | sh\n"
            "• Desktop App: Launch from Applications\n\n"
            "Documentation:\n"
            f"• CLI: {CLI_DOCS_URL}\n"
            f"• Desktop App: {DESKTOP_DOCS_URL}"
        )


# Global config manager instance (environment-aware)
_config_manager = ConfigManager()


def get_config(config_path: Optional[Path] = None) -> SyftBoxConfig:
    """Get SyftBox configuration (convenience function).
    
    Args:
        config_path: Optional custom path to config file
                    If None, uses environment-aware path from settings
        
    Returns:
        SyftBoxConfig instance
    """
    if config_path:
        # Use custom path
        return SyftBoxConfig.load(config_path)
    else:
        # Use global cached config (environment-aware)
        return _config_manager.config


def is_syftbox_running() -> bool:
    """Check if SyftBox is running (convenience function).
    
    Returns:
        True if SyftBox is running, False otherwise
    """
    return _config_manager.is_syftbox_running()


def is_syftbox_installed() -> bool:
    """Check if SyftBox is installed (convenience function).
    
    Returns:
        True if SyftBox is installed and configured, False otherwise
    """
    return _config_manager.is_syftbox_installed()


def get_startup_instructions() -> str:
    """Get SyftBox startup instructions (convenience function).
    
    Returns:
        Startup instructions string
    """
    return _config_manager.get_startup_instructions()


def get_installation_instructions() -> str:
    """Get SyftBox installation instructions (convenience function).
    
    Returns:
        Installation instructions string
    """
    return _config_manager.get_installation_instructions()


def get_app_info() -> Dict[str, Any]:
    """Get application information from settings (convenience function).
    
    Returns:
        Dictionary with app metadata and configuration
    """
    return {
        "app_name": settings.app_name,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "project_name": settings.project_name,
        "project_version": settings.project_version,
        "project_description": settings.project_description,
        "config_path": str(settings.syftbox_config_path),
        "accounting_path": str(settings.accounting_config_path)
    }