"""
Configuration management for the SyftBox SDK.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class SyftBoxConfig:
    """SyftBox configuration management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".syftbox" / "config.json"
        self._config_data = None
        
    @property
    def exists(self) -> bool:
        """Check if config file exists."""
        return self.config_path.exists()
    
    @property
    def email(self) -> Optional[str]:
        """Get user email from config."""
        data = self._load_config()
        return data.get("email") if data else None
    
    @property
    def server_url(self) -> str:
        """Get server URL from config."""
        data = self._load_config()
        return data.get("server_url", "https://syftbox.net") if data else "https://syftbox.net"
    
    @property
    def data_dir(self) -> Optional[Path]:
        """Get data directory from config."""
        data = self._load_config()
        if data and "data_dir" in data:
            return Path(data["data_dir"])
        return None
    
    @property
    def refresh_token(self) -> Optional[str]:
        """Get refresh token from config."""
        data = self._load_config()
        return data.get("refresh_token") if data else None
    
    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load config from file."""
        if self._config_data is not None:
            return self._config_data
            
        if not self.exists:
            return None
            
        try:
            with open(self.config_path, 'r') as f:
                self._config_data = json.load(f)
            return self._config_data
        except (json.JSONDecodeError, FileNotFoundError):
            return None