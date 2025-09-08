"""
Shared validation utilities for SyftBox components
"""
import re
import json
import socket
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

from ..core.exceptions import ConfigurationError, ValidationError
from .constants import (
    EMAIL_PATTERN, 
    REQUIRED_CONFIG_FIELDS, 
    OPTIONAL_CONFIG_FIELDS,
    SYFTBOX_PROCESS_NAMES,
    DEFAULT_APP_PORT,
    DEFAULT_HOST,
    DEFAULT_SOCKET_TIMEOUT,
    HTTP_SCHEMES,
    SYFT_SCHEME
)

class EmailValidator:
    """Email validation utilities."""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format."""
        return validate_email(email)
    
    @staticmethod
    def validate_email(email: str, field_name: str = "email") -> str:
        """Validate email and raise error if invalid.
        
        Args:
            email: Email address to validate
            field_name: Name of the field for error messages
            
        Returns:
            Cleaned email address
            
        Raises:
            ValidationError: If email is invalid
        """
        if not email:
            raise ValidationError(f"{field_name} is required")
            
        cleaned_email = email.strip()
        
        if not validate_email(cleaned_email):
            raise ValidationError(f"Invalid {field_name} format: {email}")
            
        return cleaned_email


class URLValidator:
    """URL validation utilities."""
    
    @staticmethod
    def is_valid_http_url(url: str) -> bool:
        """Validate HTTP/HTTPS URL format."""
        return validate_http_url(url)
    
    @staticmethod
    def is_valid_syft_url(syft_url: str) -> bool:
        """Validate syft:// URL format."""
        return validate_syft_url(syft_url)
    
    @staticmethod
    def normalize_server_url(url: str) -> str:
        """Normalize server URL format.
        
        Args:
            url: Server URL to normalize
            
        Returns:
            Normalized URL without trailing slash
            
        Raises:
            ValidationError: If URL is invalid
        """
        if not url:
            raise ValidationError("Server URL is required")
        
        # Clean URL
        normalized = url.strip().rstrip('/')
        
        # Add https if no scheme
        if not normalized.startswith(('http://', 'https://')):
            normalized = 'https://' + normalized
        
        # Validate final URL
        if not URLValidator.is_valid_http_url(normalized):
            raise ValidationError(f"Invalid server URL format: {url}")
        
        return normalized


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_config_data(config_data: Dict[str, Any], config_path: str) -> None:
        """Validate configuration dictionary."""
        # Check required fields
        missing_fields = [
            field for field in REQUIRED_CONFIG_FIELDS 
            if field not in config_data
        ]
        if missing_fields:
            raise ConfigurationError(
                f"Missing required fields in SyftBox config: {', '.join(missing_fields)}",
                config_path
            )
        
        # Validate email
        try:
            EmailValidator.validate_email(config_data["email"])
        except ValidationError as e:
            raise ConfigurationError(str(e), config_path)
        
        # Validate server URL
        try:
            URLValidator.normalize_server_url(config_data["server_url"])
        except ValidationError as e:
            raise ConfigurationError(str(e), config_path)
        
        # Validate data_dir exists
        data_dir = Path(config_data["data_dir"])
        if not data_dir.exists():
            raise ConfigurationError(
                f"Data directory does not exist: {data_dir}",
                config_path
            )
    
    @staticmethod
    def validate_json_file(file_path: Path) -> Dict[str, Any]:
        """Validate and load JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in file: {e}",
                str(file_path)
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read file: {e}",
                str(file_path)
            )


class ProcessValidator:
    """Process and system validation utilities."""
    
    @staticmethod
    def is_port_open(host: str = DEFAULT_HOST, port: int = DEFAULT_APP_PORT, 
                     timeout: float = DEFAULT_SOCKET_TIMEOUT) -> bool:
        """Check if a port is open and listening."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                return sock.connect_ex((host, port)) == 0
        except Exception:
            return False
    
    @staticmethod
    def is_syftbox_process_running() -> bool:
        """Check if SyftBox process is running."""
        try:
            for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
                try:
                    name = proc.info.get('name', '').lower()
                    exe = proc.info.get('exe', '').lower() if proc.info.get('exe') else ''
                    cmdline = proc.info.get('cmdline', [])
                    
                    # Check if any SyftBox process name matches
                    for process_name in SYFTBOX_PROCESS_NAMES:
                        if process_name in name or process_name in exe:
                            return True
                    
                    # Also check command line for syftbox
                    if cmdline:
                        cmdline_str = ' '.join(cmdline).lower()
                        if 'syftbox' in cmdline_str and not 'grep' in cmdline_str:
                            return True
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        
        return False
    
    @staticmethod
    def is_syftbox_running() -> bool:
        """Check if SyftBox is running (port or process)."""
        # Check HTTP server (app version)
        if ProcessValidator.is_port_open():
            return True
        
        # Check for SyftBox processes
        if ProcessValidator.is_syftbox_process_running():
            return True
        
        return False


class PathValidator:
    """Path validation utilities."""
    
    @staticmethod
    def validate_directory_exists(path: Path, description: str) -> None:
        """Validate that a directory exists."""
        if not path.exists():
            raise ValidationError(f"{description} directory does not exist: {path}")
        
        if not path.is_dir():
            raise ValidationError(f"{description} path is not a directory: {path}")
    
    @staticmethod
    def validate_file_exists(path: Path, description: str) -> None:
        """Validate that a file exists."""
        if not path.exists():
            raise ValidationError(f"{description} file does not exist: {path}")
        
        if not path.is_file():
            raise ValidationError(f"{description} path is not a file: {path}")
    
    @staticmethod
    def ensure_directory_exists(path: Path, create: bool = False) -> bool:
        """Ensure directory exists, optionally creating it."""
        if path.exists():
            return path.is_dir()
        
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
                return True
            except Exception:
                return False
        
        return False


class Validator:
    """Input validator with detailed error messages."""
    
    def __init__(self):
        self.errors: List[str] = []
    
    def validate_required(self, value: Any, field_name: str) -> 'Validator':
        """Validate required field."""
        if value is None or (isinstance(value, str) and not value.strip()):
            self.errors.append(f"{field_name} is required")
        return self
    
    def validate_email_field(self, email: str, field_name: str) -> 'Validator':
        """Validate email field."""
        if email and not validate_email(email):
            self.errors.append(f"{field_name} must be a valid email address")
        return self
    
    def validate_url_field(self, url: str, field_name: str) -> 'Validator':
        """Validate URL field."""
        if url and not URLValidator.is_valid_http_url(url):
            self.errors.append(f"{field_name} must be a valid HTTP/HTTPS URL")
        return self
    
    def validate_syft_url_field(self, url: str, field_name: str) -> 'Validator':
        """Validate Syft URL field."""
        if url and not URLValidator.is_valid_syft_url(url):
            self.errors.append(f"{field_name} must be a valid syft:// URL")
        return self
    
    def validate_range(self, value: Union[int, float], min_val: float, max_val: float, field_name: str) -> 'Validator':
        """Validate numeric range."""
        if value is not None:
            try:
                num_val = float(value)
                if not (min_val <= num_val <= max_val):
                    self.errors.append(f"{field_name} must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                self.errors.append(f"{field_name} must be a valid number")
        return self
    
    def validate_length(self, value: str, min_len: int, max_len: int, field_name: str) -> 'Validator':
        """Validate string length."""
        if value is not None:
            if not isinstance(value, str):
                self.errors.append(f"{field_name} must be a string")
            elif not (min_len <= len(value) <= max_len):
                self.errors.append(f"{field_name} must be between {min_len} and {max_len} characters")
        return self
    
    def validate_choices(self, value: Any, choices: List[Any], field_name: str) -> 'Validator':
        """Validate value is in allowed choices."""
        if value is not None and value not in choices:
            self.errors.append(f"{field_name} must be one of: {', '.join(map(str, choices))}")
        return self
    
    def validate_list_field(self, value: List[Any], field_name: str) -> 'Validator':
        """Validate list field."""
        if value is not None:
            if not isinstance(value, list):
                self.errors.append(f"{field_name} must be a list")
            elif len(value) == 0:
                self.errors.append(f"{field_name} cannot be empty")
        return self
    
    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return len(self.errors) == 0
    
    def get_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors.copy()
    
    def raise_if_invalid(self, context: str = "Validation"):
        """Raise ValidationError if any validations failed."""
        if not self.is_valid():
            error_msg = f"{context} failed: " + "; ".join(self.errors)
            raise ValidationError(error_msg)

def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    if not email or not isinstance(email, str):
        return False
    
    # Use the pattern from constants
    if not re.match(EMAIL_PATTERN, email.strip()):
        return False
        
    # Additional RFC compliance checks
    email = email.strip()
    if len(email) > 254:  # RFC 5321 limit
        return False
        
    try:
        local, domain = email.rsplit('@', 1)
        if len(local) > 64:  # RFC 5321 limit
            return False
    except ValueError:
        return False
        
    return True


def validate_service_name(name: str) -> bool:
    """Validate service name format.
    
    Args:
        name: Service name to validate
        
    Returns:
        True if valid service name
    """
    if not name or not isinstance(name, str):
        return False
    
    # Service names should be alphanumeric with hyphens/underscores
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, name)) and 1 <= len(name) <= 100


def validate_syft_url(url: str) -> bool:
    """Validate syft:// URL format.
    
    Args:
        url: Syft URL to validate
        
    Returns:
        True if valid syft URL format
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)
        
        # Must use syft:// scheme
        if parsed.scheme != SYFT_SCHEME:
            return False
        
        # Must have hostname (this will be the email)
        if not parsed.hostname:
            return False
        
        # Validate hostname as email
        if not validate_email(parsed.hostname):
            return False
        
        # Path should follow /app_data/{service}/rpc/{endpoint} pattern
        path_parts = [part for part in parsed.path.split('/') if part]
        if len(path_parts) < 4:
            return False
        
        if path_parts[0] != 'app_data' or path_parts[2] != 'rpc':
            return False
        
        # Validate service name
        service_name = path_parts[1]
        if not validate_service_name(service_name):
            return False
        
        return True
        
    except Exception:
        return False


def validate_http_url(url: str) -> bool:
    """Validate HTTP/HTTPS URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid HTTP URL
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and parsed.netloc
    except Exception:
        return False


def validate_cost(cost: Union[int, float]) -> bool:
    """Validate cost value.
    
    Args:
        cost: Cost value to validate
        
    Returns:
        True if valid cost
    """
    if cost is None:
        return True  # None is valid (means no cost limit)
    
    try:
        float_cost = float(cost)
        return float_cost >= 0 and float_cost <= 1000  # Reasonable range
    except (ValueError, TypeError):
        return False


def validate_temperature(temperature: float) -> bool:
    """Validate temperature parameter for text generation.
    
    Args:
        temperature: Temperature value to validate
        
    Returns:
        True if valid temperature
    """
    if temperature is None:
        return True
    
    try:
        temp = float(temperature)
        return 0.0 <= temp <= 2.0
    except (ValueError, TypeError):
        return False


def validate_max_tokens(max_tokens: int) -> bool:
    """Validate max tokens parameter.
    
    Args:
        max_tokens: Max tokens value to validate
        
    Returns:
        True if valid max tokens
    """
    if max_tokens is None:
        return True
    
    try:
        tokens = int(max_tokens)
        return 1 <= tokens <= 100000  # Reasonable range
    except (ValueError, TypeError):
        return False


def validate_similarity_threshold(threshold: float) -> bool:
    """Validate similarity threshold for search.
    
    Args:
        threshold: Threshold value to validate
        
    Returns:
        True if valid threshold
    """
    if threshold is None:
        return True
    
    try:
        thresh = float(threshold)
        return 0.0 <= thresh <= 1.0
    except (ValueError, TypeError):
        return False


def validate_tags(tags: List[str]) -> bool:
    """Validate list of tags.
    
    Args:
        tags: List of tags to validate
        
    Returns:
        True if valid tags
    """
    if not tags:
        return True
    
    if not isinstance(tags, list):
        return False
    
    for tag in tags:
        if not isinstance(tag, str):
            return False
        if not tag.strip():
            return False
        if len(tag) > 50:  # Reasonable tag length limit
            return False
        # Tags should be alphanumeric with some special chars
        if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
            return False
    
    return len(tags) <= 20  # Reasonable number of tags


def validate_file_path(path: Union[str, Path]) -> bool:
    """Validate file path.
    
    Args:
        path: File path to validate
        
    Returns:
        True if valid and exists
    """
    if not path:
        return False
    
    try:
        path_obj = Path(path)
        return path_obj.exists() and path_obj.is_file()
    except Exception:
        return False


def validate_directory_path(path: Union[str, Path]) -> bool:
    """Validate directory path.
    
    Args:
        path: Directory path to validate
        
    Returns:
        True if valid and exists
    """
    if not path:
        return False
    
    try:
        path_obj = Path(path)
        return path_obj.exists() and path_obj.is_dir()
    except Exception:
        return False


def validate_chat_message(message: str) -> bool:
    """Validate chat message content.
    
    Args:
        message: Message content to validate
        
    Returns:
        True if valid message
    """
    if not message or not isinstance(message, str):
        return False
    
    # Check length
    if not (1 <= len(message.strip()) <= 50000):
        return False
    
    # Check for suspicious content (basic)
    suspicious_patterns = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript URLs
        r'data:text/html',  # HTML data URLs
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return False
    
    return True


def validate_search_query(query: str) -> bool:
    """Validate search query.
    
    Args:
        query: Search query to validate
        
    Returns:
        True if valid query
    """
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    
    # Check length
    if not (1 <= len(query) <= 1000):
        return False
    
    # Query should not be only special characters
    if re.match(r'^[^\w\s]+$', query):
        return False
    
    return True


def validate_json_data(data: Any) -> bool:
    """Validate that data can be serialized to JSON.
    
    Args:
        data: Data to validate
        
    Returns:
        True if JSON serializable
    """
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False
    
# Helper validation functions
def validate_chat_request(message: str, service_name: Optional[str] = None, 
                         max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> None:
    """Validate chat request parameters."""
    validator = Validator()
    
    validator.validate_required(message, "message")
    if message and not validate_chat_message(message):
        validator.errors.append("message contains invalid content or is too long")
    
    if service_name and not validate_service_name(service_name):
        validator.errors.append("service_name contains invalid characters")
    
    validator.validate_range(max_tokens, 1, 100000, "max_tokens")
    validator.validate_range(temperature, 0.0, 2.0, "temperature")
    
    validator.raise_if_invalid("Chat request")


def validate_search_request(query: str, limit: Optional[int] = None, 
                           similarity_threshold: Optional[float] = None) -> None:
    """Validate search request parameters."""
    validator = Validator()
    
    validator.validate_required(query, "query")
    if query and not validate_search_query(query):
        validator.errors.append("query is too long or contains only special characters")
    
    validator.validate_range(limit, 1, 100, "limit")
    validator.validate_range(similarity_threshold, 0.0, 1.0, "similarity_threshold")
    
    validator.raise_if_invalid("Search request")


def validate_service_filter_criteria(**kwargs) -> None:
    """Validate service filter criteria."""
    validator = Validator()
    
    if 'datasite' in kwargs and kwargs['datasite']:
        validator.validate_email_field(kwargs['datasite'], "datasite")
    
    if 'tags' in kwargs and kwargs['tags']:
        if not validate_tags(kwargs['tags']):
            validator.errors.append("tags must be a list of valid tag strings")
    
    if 'max_cost' in kwargs and kwargs['max_cost'] is not None:
        validator.validate_range(kwargs['max_cost'], 0.0, 1000.0, "max_cost")
    
    if 'min_cost' in kwargs and kwargs['min_cost'] is not None:
        validator.validate_range(kwargs['min_cost'], 0.0, 1000.0, "min_cost")
    
    validator.raise_if_invalid("Service filter criteria")


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input by stripping and truncating."""
    if not isinstance(value, str):
        return str(value)
    
    # Strip whitespace
    sanitized = value.strip()
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def sanitize_tags(tags: List[str]) -> List[str]:
    """Sanitize list of tags."""
    if not isinstance(tags, list):
        return []
    
    sanitized = []
    for tag in tags:
        if isinstance(tag, str):
            clean_tag = sanitize_string(tag, 50)
            if clean_tag and validate_service_name(clean_tag):
                sanitized.append(clean_tag)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in sanitized:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_tags.append(tag)
    
    return unique_tags[:20]  # Limit number of tags


def sanitize_request_data(data: Dict[str, Any], request_type: str) -> Dict[str, Any]:
    """Sanitize request data using existing sanitization functions."""
    sanitized = data.copy()
    
    if request_type == 'chat' and 'messages' in sanitized:
        for message in sanitized['messages']:
            if 'content' in message:
                message['content'] = sanitize_string(message['content'], 50000)
                
    elif request_type == 'search' and 'query' in sanitized:
        sanitized['query'] = sanitize_string(sanitized['query'], 1000)
    
    # Sanitize tags if present
    if 'tags' in sanitized and sanitized['tags']:
        sanitized['tags'] = sanitize_tags(sanitized['tags'])
    
    return sanitized


def ensure_valid_email(email: str, field_name: str = "email") -> str:
    """Ensure email is valid, raise ValidationError if not."""
    if not validate_email(email):
        raise ValidationError(f"Invalid email format: {email}")
    return email


def ensure_valid_service_name(name: str, field_name: str = "service_name") -> str:
    """Ensure service name is valid, raise ValidationError if not."""
    if not validate_service_name(name):
        raise ValidationError(f"Invalid service name format: {name}")
    return name


def ensure_valid_cost(cost: Union[int, float], field_name: str = "cost") -> float:
    """Ensure cost is valid, raise ValidationError if not."""
    if not validate_cost(cost):
        raise ValidationError(f"Invalid cost value: {cost}")
    return float(cost)


def ensure_valid_syft_url(url: str, field_name: str = "syft_url") -> str:
    """Ensure syft URL is valid, raise ValidationError if not."""
    if not validate_syft_url(url):
        raise ValidationError(f"Invalid syft:// URL format: {url}")
    return url


def validate_config_file(config_path: Path) -> Dict[str, Any]:
    """Validate configuration file."""
    config_data = ConfigValidator.validate_json_file(config_path)
    ConfigValidator.validate_config_data(config_data, str(config_path))
    return config_data


def is_syftbox_running() -> bool:
    """Check if SyftBox is running."""
    return ProcessValidator.is_syftbox_running()