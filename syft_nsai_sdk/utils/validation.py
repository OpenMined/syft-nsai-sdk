"""
Input validation utilities for SyftBox NSAI SDK
"""
import re
from typing import Any, List, Optional, Union
from pathlib import Path
from urllib.parse import urlparse

from ..core.exceptions import ValidationError


def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_model_name(name: str) -> bool:
    """Validate model name format.
    
    Args:
        name: Model name to validate
        
    Returns:
        True if valid model name
    """
    if not name or not isinstance(name, str):
        return False
    
    # Model names should be alphanumeric with hyphens/underscores
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
        if parsed.scheme != 'syft':
            return False
        
        # Must have hostname (email domain)
        if not parsed.hostname:
            return False
        
        # Must have username (email local part)
        if not parsed.username:
            return False
        
        # Validate email format
        email = f"{parsed.username}@{parsed.hostname}"
        if not validate_email(email):
            return False
        
        # Path should follow /app_data/{model}/rpc/{endpoint} pattern
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) < 4:
            return False
        
        if path_parts[0] != 'app_data' or path_parts[2] != 'rpc':
            return False
        
        # Validate model name
        model_name = path_parts[1]
        if not validate_model_name(model_name):
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
        import json
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False


# Validation decorators and classes

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
        if url and not validate_http_url(url):
            self.errors.append(f"{field_name} must be a valid HTTP/HTTPS URL")
        return self
    
    def validate_syft_url_field(self, url: str, field_name: str) -> 'Validator':
        """Validate Syft URL field."""
        if url and not validate_syft_url(url):
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
            raise ValidationError(error_msg, field="multiple", value=str(self.errors))


def validate_chat_request(message: str, model_name: Optional[str] = None, 
                         max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> None:
    """Validate chat request parameters.
    
    Args:
        message: Chat message
        model_name: Optional model name
        max_tokens: Optional max tokens
        temperature: Optional temperature
        
    Raises:
        ValidationError: If validation fails
    """
    validator = Validator()
    
    validator.validate_required(message, "message")
    if message and not validate_chat_message(message):
        validator.errors.append("message contains invalid content or is too long")
    
    if model_name and not validate_model_name(model_name):
        validator.errors.append("model_name contains invalid characters")
    
    validator.validate_range(max_tokens, 1, 100000, "max_tokens")
    validator.validate_range(temperature, 0.0, 2.0, "temperature")
    
    validator.raise_if_invalid("Chat request")


def validate_search_request(query: str, limit: Optional[int] = None, 
                           similarity_threshold: Optional[float] = None) -> None:
    """Validate search request parameters.
    
    Args:
        query: Search query
        limit: Optional result limit
        similarity_threshold: Optional similarity threshold
        
    Raises:
        ValidationError: If validation fails
    """
    validator = Validator()
    
    validator.validate_required(query, "query")
    if query and not validate_search_query(query):
        validator.errors.append("query is too long or contains only special characters")
    
    validator.validate_range(limit, 1, 100, "limit")
    validator.validate_range(similarity_threshold, 0.0, 1.0, "similarity_threshold")
    
    validator.raise_if_invalid("Search request")


def validate_model_filter_criteria(**kwargs) -> None:
    """Validate model filter criteria.
    
    Args:
        **kwargs: Filter criteria to validate
        
    Raises:
        ValidationError: If validation fails
    """
    validator = Validator()
    
    if 'owner' in kwargs and kwargs['owner']:
        validator.validate_email_field(kwargs['owner'], "owner")
    
    if 'tags' in kwargs and kwargs['tags']:
        if not validate_tags(kwargs['tags']):
            validator.errors.append("tags must be a list of valid tag strings")
    
    if 'max_cost' in kwargs and kwargs['max_cost'] is not None:
        validator.validate_range(kwargs['max_cost'], 0.0, 1000.0, "max_cost")
    
    if 'min_cost' in kwargs and kwargs['min_cost'] is not None:
        validator.validate_range(kwargs['min_cost'], 0.0, 1000.0, "min_cost")
    
    validator.raise_if_invalid("Model filter criteria")


# Quick validation functions for common use cases

def ensure_valid_email(email: str, field_name: str = "email") -> str:
    """Ensure email is valid, raise ValidationError if not."""
    if not validate_email(email):
        raise ValidationError(f"Invalid email format: {email}", field_name, email)
    return email


def ensure_valid_model_name(name: str, field_name: str = "model_name") -> str:
    """Ensure model name is valid, raise ValidationError if not."""
    if not validate_model_name(name):
        raise ValidationError(f"Invalid model name format: {name}", field_name, name)
    return name


def ensure_valid_cost(cost: Union[int, float], field_name: str = "cost") -> float:
    """Ensure cost is valid, raise ValidationError if not."""
    if not validate_cost(cost):
        raise ValidationError(f"Invalid cost value: {cost}", field_name, str(cost))
    return float(cost)


def ensure_valid_syft_url(url: str, field_name: str = "syft_url") -> str:
    """Ensure syft URL is valid, raise ValidationError if not."""
    if not validate_syft_url(url):
        raise ValidationError(f"Invalid syft:// URL format: {url}", field_name, url)
    return url


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input by stripping and truncating.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)
    
    # Strip whitespace
    sanitized = value.strip()
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def sanitize_tags(tags: List[str]) -> List[str]:
    """Sanitize list of tags.
    
    Args:
        tags: List of tags to sanitize
        
    Returns:
        Sanitized list of tags
    """
    if not isinstance(tags, list):
        return []
    
    sanitized = []
    for tag in tags:
        if isinstance(tag, str):
            clean_tag = sanitize_string(tag, 50)
            if clean_tag and validate_model_name(clean_tag):
                sanitized.append(clean_tag)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in sanitized:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_tags.append(tag)
    
    return unique_tags[:20]  # Limit number of tags