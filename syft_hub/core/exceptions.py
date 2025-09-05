"""
Custom exceptions for SyftBox NSAI SDK
"""
from typing import Optional, Dict, Any


class SyftBoxSDKError(Exception):
    """Base exception for all SDK errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def __str__(self):
        return f"{self.error_code}: {self.message}"


class SyftBoxNotFoundError(SyftBoxSDKError):
    """SyftBox installation not found or not configured."""
    
    def __init__(self, message: str = "SyftBox installation not found"):
        super().__init__(message, "SYFTBOX_NOT_FOUND")

class SyftBoxNotRunningError(SyftBoxSDKError):
    """SyftBox is installed but not running."""
    
    def __init__(self, message: str = "SyftBox is not running"):
        super().__init__(message, "SYFTBOX_NOT_RUNNING")


class ConfigurationError(SyftBoxSDKError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_path: Optional[str] = None):
        details = {"config_path": config_path} if config_path else {}
        super().__init__(message, "CONFIGURATION_ERROR", details)


class ServiceNotFoundError(SyftBoxSDKError):
    """Service not found in discovery."""
    
    def __init__(self, service_name: str, available_services: Optional[int] = None):
        message = f"Service '{service_name}' not found"
        if available_services is not None:
            message += f" (searched {available_services} available services)"
        details = {"service_name": service_name, "available_services": available_services}
        super().__init__(message, "SERVICE_NOT_FOUND", details)


class ServiceNotSupportedError(SyftBoxSDKError):
    """Service doesn't support requested service type."""
    
    def __init__(self, service_name: str, service_type: str, supported_services: Optional[list] = None):
        message = f"Service '{service_name}' does not support '{service_type}' service"
        if supported_services:
            message += f" (supports: {', '.join(supported_services)})"
        details = {
            "service_name": service_name,
            "requested_service": service_type,
            "supported_services": supported_services
        }
        super().__init__(message, "SERVICE_NOT_SUPPORTED", details)


class ServiceUnavailableError(SyftBoxSDKError):
    """Service is configured but currently unavailable."""
    
    def __init__(self, service_name: str, service_type: str, reason: Optional[str] = None):
        message = f"Service '{service_type}' on service '{service_name}' is unavailable"
        if reason:
            message += f": {reason}"
        details = {"service_name": service_name, "service_type": service_type, "reason": reason}
        super().__init__(message, "SERVICE_UNAVAILABLE", details)


class NetworkError(SyftBoxSDKError):
    """Network-related errors (RPC calls, polling, etc)."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None):
        details = {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, "NETWORK_ERROR", details)


class RPCError(SyftBoxSDKError):
    """RPC-specific errors."""
    
    def __init__(self, message: str, syft_url: Optional[str] = None, rpc_code: Optional[str] = None):
        details = {}
        if syft_url:
            details["syft_url"] = syft_url
        if rpc_code:
            details["rpc_code"] = rpc_code
        super().__init__(message, "RPC_ERROR", details)


class PollingTimeoutError(RPCError):
    """Polling for response timed out."""
    
    def __init__(self, syft_url: str, attempts: int, max_attempts: int):
        message = f"Polling timed out after {attempts}/{max_attempts} attempts"
        details = {"syft_url": syft_url, "attempts": attempts, "max_attempts": max_attempts}
        super().__init__(message, syft_url, "POLLING_TIMEOUT")
        self.details.update(details)


class PollingError(RPCError):
    """Error during polling for response."""
    
    def __init__(self, message: str, syft_url: str, poll_url: Optional[str] = None):
        details = {"syft_url": syft_url}
        if poll_url:
            details["poll_url"] = poll_url
        super().__init__(message, syft_url, "POLLING_ERROR")
        self.details.update(details)


class AuthenticationError(SyftBoxSDKError):
    """Authentication/authorization errors."""
    
    def __init__(self, message: str, service_url: Optional[str] = None):
        details = {"service_url": service_url} if service_url else {}
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class PaymentError(SyftBoxSDKError):
    """Payment/transaction token related errors."""
    
    def __init__(self, message: str, cost: Optional[float] = None, balance: Optional[float] = None):
        details = {}
        if cost is not None:
            details["cost"] = cost
        if balance is not None:
            details["balance"] = balance
        super().__init__(message, "PAYMENT_ERROR", details)

class TransactionTokenCreationError(SyftBoxSDKError):
    """Exception raised when transaction token creation fails."""
    def __init__(self, message: str, recipient_email: Optional[str] = None):
        details = {"recipient_email": recipient_email} if recipient_email else {}
        super().__init__(message, "TRANSACTION_TOKEN_CREATION_ERROR", details)
        
class ValidationError(SyftBoxSDKError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[str] = None):
        details = {}
        if field:
            details["field"] = field
        if value:
            details["value"] = value
        super().__init__(message, "VALIDATION_ERROR", details)


class MetadataParsingError(SyftBoxSDKError):
    """Error parsing service metadata files."""
    
    def __init__(self, file_path: str, parse_error: str):
        message = f"Failed to parse metadata file: {parse_error}"
        details = {"file_path": file_path, "parse_error": parse_error}
        super().__init__(message, "METADATA_PARSING_ERROR", details)


class HealthCheckError(SyftBoxSDKError):
    """Error during service health checking."""
    
    def __init__(self, service_name: str, reason: str):
        message = f"Health check failed for service '{service_name}': {reason}"
        details = {"service_name": service_name, "reason": reason}
        super().__init__(message, "HEALTH_CHECK_ERROR", details)


# Convenience functions for common error scenarios
def raise_service_not_found(service_name: str, available_services: list = None):
    """Raise ServiceNotFoundError with helpful context."""
    count = len(available_services) if available_services else None
    raise ServiceNotFoundError(service_name, count)


def raise_service_not_supported(service_name: str, service_type: str, service_info=None):
    """Raise ServiceNotSupportedError with service's actual capabilities."""
    supported = None
    if service_info and hasattr(service_info, 'enabled_service_types'):
        supported = [s.value for s in service_info.enabled_service_types]
    raise ServiceNotSupportedError(service_name, service_type, supported)


def raise_network_error(message: str, url: str = None, status_code: int = None):
    """Raise NetworkError with context."""
    raise NetworkError(message, url, status_code)


def raise_rpc_error(message: str, syft_url: str = None):
    """Raise RPCError with context."""
    raise RPCError(message, syft_url)