"""
Clean settings configuration for Syft NSAI SDK.
"""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv(override=True)


class AppSettings(BaseSettings):
    """Application settings."""
    
    app_name: str = Field("SYFT NSAI SDK", env="APP_NAME")
    syftbox_config_path: Path = Field("~/.syftbox/config.json", env="SYFTBOX_CONFIG_PATH")
    jwt_secret: str = Field(..., env="JWT_SECRET")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    # Project metadata
    project_name: Optional[str] = Field(None, env="PROJECT_NAME")
    project_version: Optional[str] = Field(None, env="PROJECT_VERSION")
    project_description: Optional[str] = Field(None, env="PROJECT_DESCRIPTION")
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra environment variables
    )

    @field_validator("syftbox_config_path", mode="before")
    def validate_syftbox_config_path(cls, v):
        """Expand tilde to home directory."""
        expanded_path = Path(v).expanduser().resolve()
        if not expanded_path.exists():
            raise ValueError(f"Syftbox config path {expanded_path} does not exist")
        return expanded_path


# Global instance
settings = AppSettings()