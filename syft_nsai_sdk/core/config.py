"""
SyftBox configuration management
"""
import json
import logging
import os
import socket
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .exceptions import (
    SyftBoxNotFoundError, 
    SyftBoxNotRunningError, 
    ConfigurationError
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
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'SyftBoxConfig':
        """Load SyftBox configuration from file."""
        
        # Determine config path
        if config_path is None:
            config_path = Path.home() / ".syftbox" / "config.json"
        
        if not config_path.exists():
            raise SyftBoxNotFoundError(
                f"SyftBox config not found at {config_path}.\n"
                "Please install and setup SyftBox first.\n\n"
                "Install: curl -LsSf https://install.syftbox.openmined.org | sh\n"
                "Setup: syftbox setup"
            )
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in SyftBox config: {e}",
                str(config_path)
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read SyftBox config: {e}",
                str(config_path)
            )
        
        # Validate required fields
        required_fields = ["data_dir", "email", "server_url"]
        missing_fields = [field for field in required_fields if field not in config_data]
        if missing_fields:
            raise ConfigurationError(
                f"Missing required fields in SyftBox config: {', '.join(missing_fields)}",
                str(config_path)
            )
        
        # Create config object
        return cls(
            data_dir=Path(config_data["data_dir"]),
            email=config_data["email"],
            server_url=config_data["server_url"].rstrip('/'),
            refresh_token=config_data.get("refresh_token"),
            config_path=config_path
        )
    
    @property
    def datasites_path(self) -> Path:
        """Path to the datasites directory."""
        return self.data_dir / "datasites"
    
    @property
    def my_datasite_path(self) -> Path:
        """Path to the current user's datasite."""
        return self.datasites_path / self.email
    
    @property
    def cache_server_url(self) -> str:
        """URL of the cache server (same as server_url in SyftBox)."""
        return self.server_url
    
    def validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.data_dir.exists():
            raise ConfigurationError(
                f"SyftBox data directory not found: {self.data_dir}",
                str(self.config_path)
            )
        
        if not self.datasites_path.exists():
            raise ConfigurationError(
                f"Datasites directory not found: {self.datasites_path}",
                str(self.config_path)
            )
    
    def get_datasite_path(self, email: str) -> Path:
        """Get path to a specific datasite."""
        return self.datasites_path / email
    
    def list_datasites(self) -> list[str]:
        """List all available datasites."""
        if not self.datasites_path.exists():
            return []
        
        return [
            item.name for item in self.datasites_path.iterdir()
            if item.is_dir() and '@' in item.name
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
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
        self.config_path = config_path
        self._config: Optional[SyftBoxConfig] = None
    
    @property
    def config(self) -> SyftBoxConfig:
        """Get cached config, loading if necessary."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self) -> SyftBoxConfig:
        """Load and validate SyftBox configuration."""
        config = SyftBoxConfig.load(self.config_path)
        config.validate_paths()
        return config
    
    def reload_config(self) -> SyftBoxConfig:
        """Force reload configuration from disk."""
        self._config = None
        return self.config

    # def is_syftbox_installed1(self) -> bool:
    #     """Check if SyftBox is installed and configured."""
    #     try:
    #         self.config
    #         return True
    #     except (SyftBoxNotFoundError, ConfigurationError):
    #         return False
    
    def is_syftbox_installed(self) -> bool:
        """Check if SyftBox is installed and configured."""
        try:
            # Use custom path if provided, otherwise default
            if self.config_path:
                config_path = self.config_path
            else:
                config_path = Path.home() / ".syftbox" / "config.json"

            # Check if config file exists and is readable
            if config_path.exists():
                # You might want to validate the config file here
                logger.info(f"Using config credentials at path: {config_path}")
                return True
            return False
            
        except Exception:
            return False

    def is_syftbox_running(self) -> bool:
        """Check if SyftBox process is actually running."""
        
        # Check for HTTP server (app version)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            is_app_running = sock.connect_ex(('127.0.0.1', 7938)) == 0
            sock.close()
            if is_app_running:
                return True
        except Exception:
            pass
        
        # Check for any process with "syftbox" in name/path
        try:
            for proc in psutil.process_iter(['name', 'exe']):
                try:
                    name = proc.info.get('name', '').lower()
                    exe = proc.info.get('exe', '').lower()
                    if 'syftbox' in name or 'syftbox' in exe:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        
        return False

    def get_installation_instructions(self) -> str:
        """Instructions for installing SyftBox."""
        if self.config_path:
            # Custom path was provided but config not found
            return (
                f"SyftBox config not found at: {self.config_path}\n\n"
                "Options:\n"
                "1. Verify the config path is correct\n"
                "2. Install SyftBox if not installed:\n"
                "   • Desktop App: https://github.com/OpenMined/SyftUI/releases\n"
                "   • Quick install (terminal): curl -fsSL https://syftbox.net/install.sh | sh\n"
                "   • Google Colab: Run the following in a cell:\n"
                "     !pip install syft-installer\n"
                "     import syft_installer as si\n"
                "     si.install_and_run_if_needed()\n"
                "3. Run 'syftbox setup' to create config at this location"
            )
        else:
            # Default path check failed
            return (
                "SyftBox config not found at default location (~/.syftbox/config.json)\n\n"
                "Options:\n"
                "1. Install SyftBox:\n"
                "   • Desktop App: https://github.com/OpenMined/SyftUI/releases\n"
                "   • Quick install (terminal): curl -fsSL https://syftbox.net/install.sh | sh\n"
                "   • Google Colab: Run the following in a cell:\n"
                "     !pip install syft-installer\n"
                "     import syft_installer as si\n"
                "     si.install_and_run_if_needed()\n"
                "2. If installed in custom location, provide syftbox_config_path parameter\n\n"
                "For detailed instructions, visit the official repositories above."
            )

    def get_startup_instructions(self) -> str:
        """Instructions for starting an already installed SyftBox."""
        return (
            "SyftBox is installed but not running.\n\n"
            "To start SyftBox:\n"
            "• CLI: Run 'syftbox' in terminal with curl -fsSL https://syftbox.net/install.sh | sh\n"
            "• Desktop App: Launch from Applications\n\n"
            "Documentation:\n"
            "• CLI: https://github.com/OpenMined/syftbox\n"
            "• Desktop App: https://github.com/OpenMined/SyftUI"
        )
    
    # def is_syftbox_available(self) -> bool:
    #     """Check if SyftBox is installed and configured."""
    #     try:
    #         self.config
    #         return True
    #     except (SyftBoxNotFoundError, ConfigurationError):
    #         return False
    
    # def get_installation_instructions(self) -> str:
    #     """Get instructions for installing SyftBox."""
    #     return (
    #         "SyftBox is required but not found.\n\n"
    #         "Installation options:\n"
    #         "1. Quick install: curl -LsSf https://install.syftbox.openmined.org | sh\n"
    #         "2. Manual install: Visit https://syftbox.openmined.org/install\n\n"
    #         "After installation, run: syftbox setup\n"
    #         "Then restart this SDK."
    #     )


# Global config manager instance
_config_manager = ConfigManager()

def get_config(config_path: Optional[Path] = None) -> SyftBoxConfig:
    """Get SyftBox configuration (convenience function)."""
    if config_path:
        # Use custom path
        return SyftBoxConfig.load(config_path)
    else:
        # Use global cached config
        return _config_manager.config
    
def is_syftbox_running() -> bool:
    """Check if SyftBox is running (convenience function)."""
    return _config_manager.is_syftbox_running()

def is_syftbox_installed() -> bool:
    """Check if SyftBox is installed (convenience function)."""
    return _config_manager.is_syftbox_installed()

def get_startup_instructions() -> str:
    """Get SyftBox startup instructions (convenience function)."""
    return _config_manager.get_startup_instructions()

def get_installation_instructions() -> str:
    """Get SyftBox installation instructions (convenience function)."""
    return _config_manager.get_installation_instructions()