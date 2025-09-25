"""
Settings configuration for Syft hub SDK.
"""
import os

from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv(override=True)

DEFAULT_ACCOUNTING_SERVICE_URL = "https://syftaccounting.centralus.cloudapp.azure.com"

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Core app settings
    app_name: str = Field("SYFT HUB SDK", env="APP_NAME")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # accounting service
    accounting_url: str = Field(
        DEFAULT_ACCOUNTING_SERVICE_URL, env="SYFTBOX_ACCOUNTING_URL"
    )
    
    # SyftBox paths
    syftbox_config_path: Path = Field("~/.syftbox/config.json", env="SYFTBOX_CONFIG_PATH")
    accounting_config_path: Path = Field("~/.syftbox/accounting.json", env="ACCOUNTING_CONFIG_PATH")
    
    # Project metadata
    project_name: Optional[str] = Field(None, env="PROJECT_NAME")
    project_version: Optional[str] = Field(None, env="PROJECT_VERSION")
    project_description: Optional[str] = Field(None, env="PROJECT_DESCRIPTION")
    project_author: Optional[str] = Field(None, env="PROJECT_AUTHOR")
    project_url: Optional[str] = Field(None, env="PROJECT_URL")
    project_email: Optional[str] = Field("info@openmined.org", env="PROJECT_EMAIL")

    # Environment and deployment
    environment: str = Field("development", env="ENVIRONMENT")
    deployment_mode: str = Field("local", env="DEPLOYMENT_MODE")  # local, docker, cloud
    
    # Service discovery and networking
    default_cache_server: str = Field("https://syftbox.net", env="DEFAULT_CACHE_SERVER")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    max_retries: int = Field(3, env="MAX_RETRIES")
    
    # Features and behavior
    auto_discovery: bool = Field(True, env="AUTO_DISCOVERY")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    enable_telemetry: bool = Field(False, env="ENABLE_TELEMETRY")
    strict_validation: bool = Field(True, env="STRICT_VALIDATION")
    
    # Performance tuning
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    cache_ttl_seconds: int = Field(300, env="CACHE_TTL_SECONDS")  # 5 minutes
    
    # Development and testing
    test_mode: bool = Field(False, env="TEST_MODE")
    mock_services: bool = Field(False, env="MOCK_SERVICES")
    dev_server_port: int = Field(8000, env="DEV_SERVER_PORT")
    
    # Paths for development
    temp_dir: Optional[Path] = Field(None, env="TEMP_DIR")
    log_file_path: Optional[Path] = Field(None, env="LOG_FILE_PATH")
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra environment variables
        case_sensitive=False
    )

    @field_validator("syftbox_config_path", "accounting_config_path", mode="before")
    def validate_config_paths(cls, v):
        """Expand tilde to home directory and resolve path."""
        if v is None:
            return v
        expanded_path = Path(v).expanduser().resolve()
        return expanded_path

    @field_validator("temp_dir", "log_file_path", mode="before")
    def validate_optional_paths(cls, v):
        """Expand and resolve optional paths."""
        if v is None:
            return v
        return Path(v).expanduser().resolve()

    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of: {', '.join(valid_levels)}")
        return v.upper()

    @field_validator("environment")
    def validate_environment(cls, v):
        """Validate environment name."""
        valid_envs = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"environment must be one of: {', '.join(valid_envs)}")
        return v.lower()

    @field_validator("deployment_mode")
    def validate_deployment_mode(cls, v):
        """Validate deployment mode."""
        valid_modes = ["local", "docker", "cloud", "kubernetes"]
        if v.lower() not in valid_modes:
            raise ValueError(f"deployment_mode must be one of: {', '.join(valid_modes)}")
        return v.lower()

    @field_validator("default_cache_server")
    def validate_cache_server_url(cls, v):
        """Validate cache server URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("default_cache_server must be a valid HTTP/HTTPS URL")
        return v.rstrip('/')

    @field_validator("request_timeout", "max_retries", "max_concurrent_requests", "cache_ttl_seconds")
    def validate_positive_integers(cls, v):
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @computed_field
    @property
    def effective_temp_dir(self) -> Path:
        """Get effective temporary directory."""
        if self.temp_dir:
            return self.temp_dir
        return Path.home() / ".syftbox" / "temp"

    @computed_field
    @property
    def effective_log_file(self) -> Optional[Path]:
        """Get effective log file path."""
        if self.log_file_path:
            return self.log_file_path
        if self.debug or self.environment != "production":
            return Path.home() / ".syftbox" / "logs" / f"{self.app_name.lower().replace(' ', '_')}.log"
        return None

    def get_runtime_info(self) -> dict:
        """Get runtime information for debugging."""
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "deployment_mode": self.deployment_mode,
            "debug": self.debug,
            "test_mode": self.test_mode,
            "python_version": os.sys.version,
            "config_path": str(self.syftbox_config_path),
            "temp_dir": str(self.effective_temp_dir),
            "log_file": str(self.effective_log_file) if self.effective_log_file else None
        }

    def get_feature_flags(self) -> dict:
        """Get current feature flag settings."""
        return {
            "auto_discovery": self.auto_discovery,
            "enable_caching": self.enable_caching,
            "enable_telemetry": self.enable_telemetry,
            "strict_validation": self.strict_validation,
            "mock_services": self.mock_services
        }

    def get_performance_settings(self) -> dict:
        """Get performance-related settings."""
        return {
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "max_concurrent_requests": self.max_concurrent_requests,
            "cache_ttl_seconds": self.cache_ttl_seconds
        }

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        # Create temp directory
        self.effective_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory if log file is specified
        if self.effective_log_file:
            self.effective_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def validate_required_paths(self) -> List[str]:
        """Validate that required paths exist and return list of missing paths."""
        missing_paths = []
        
        # Check SyftBox config (should exist for normal operation)
        if not self.syftbox_config_path.exists():
            missing_paths.append(f"SyftBox config: {self.syftbox_config_path}")
        
        return missing_paths

    def get_env_vars_summary(self) -> dict:
        """Get summary of environment variables (excluding secrets)."""
        env_vars = {}
        for field_name, field_info in self.model_fields.items():
            env_name = getattr(field_info, 'env', field_name.upper())
            if env_name and env_name in os.environ:
                # Don't expose secrets
                if 'secret' in field_name.lower() or 'password' in field_name.lower():
                    env_vars[env_name] = "***"
                else:
                    env_vars[env_name] = os.environ[env_name]
        return env_vars


# Global settings instance
settings = Settings()

# Ensure required directories exist on import
try:
    settings.ensure_directories()
except Exception:
    # Don't fail on import if directories can't be created
    pass