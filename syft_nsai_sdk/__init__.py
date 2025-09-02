"""
SyftBox NSAI SDK

A Python SDK for discovering and using AI services across the SyftBox network.
"""

__version__ = "0.1.0"
__author__ = "SyftBox Team"
__email__ = "info@openmined.org"

# Main client class
from .main import Client

# Core types and enums
from .core.types import (
    ServiceType,
    ServiceStatus, 
    HealthStatus,
    QualityPreference,
    PricingChargeType,
    ServiceItem,
    ServiceInfo,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    SearchRequest, 
    SearchResponse,
    DocumentResult,
    TransactionToken,
)

# Exceptions
from .core.exceptions import (
    SyftBoxSDKError,
    SyftBoxNotFoundError,
    ConfigurationError,
    ServiceNotFoundError,
    ServiceNotSupportedError,
    ServiceUnavailableError,
    NetworkError,
    RPCError,
    PollingTimeoutError,
    PollingError,
    AuthenticationError,
    PaymentError,
    ValidationError,
    HealthCheckError,
)

# Configuration utilities
from .core.config import (
    SyftBoxConfig,
    get_config,
    is_syftbox_installed,
    is_syftbox_running,
    get_startup_instructions,
    get_installation_instructions,
)

# Service clients (for advanced usage)
from .services.chat import ChatService, ConversationManager
from .services.search import SearchService, BatchSearchService
from .services.health import HealthMonitor

# Filtering utilities
from .discovery.filters import (
    ServiceFilter,
    FilterCriteria,
    FilterBuilder,
    create_chat_services_filter,
    create_search_services_filter,
    create_free_services_filter,
    create_paid_services_filter,
    create_healthy_services_filter,
    create_datasite_services_filter,
    create_tag_services_filter,
)

# Convenience functions
# from .main import (
#     list_available_services,
# )

# Formatting utilities
from .utils.formatting import (
    format_services_table,
    format_service_details,
    format_search_results,
    format_chat_conversation,
    format_health_summary,
    format_statistics,
)


# Package-level convenience functions
def create_client(**kwargs) -> Client:
    """Create a Client with optional configuration.
    
    Args:
        **kwargs: Configuration options for the client
        
    Returns:
        Client instance
    """
    return Client(**kwargs)


def check_installation() -> bool:
    """Check if SyftBox is properly installed and configured.
    
    Returns:
        True if SyftBox is available, False otherwise
    """
    return is_syftbox_installed()


def get_setup_instructions() -> str:
    """Get instructions for setting up SyftBox.
    
    Returns:
        Setup instructions as string
    """
    return get_installation_instructions()

def check_running() -> bool:
    """Check if SyftBox is properly running.

    Returns:
        True if SyftBox is running, False otherwise
    """
    return is_syftbox_running()


def get_startup_instructions() -> str:
    """Get instructions for starting up SyftBox.

    Returns:
        Startup instructions as string
    """
    return get_startup_instructions()


# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Main client
    "Client",
    "create_client",
    
    # Core types
    "ServiceType",
    "ServiceStatus",
    "HealthStatus", 
    "QualityPreference",
    "PricingChargeType",
    "ServiceItem",
    "ServiceInfo",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "SearchRequest",
    "SearchResponse", 
    "DocumentResult",
    "TransactionToken",
    
    # Exceptions
    "SyftBoxSDKError",
    "SyftBoxNotFoundError",
    "ConfigurationError",
    "ServiceNotFoundError",
    "ServiceNotSupportedError",
    "ServiceUnavailableError",
    "NetworkError",
    "RPCError",
    "PollingTimeoutError",
    "PollingError",
    "AuthenticationError",
    "PaymentError",
    "ValidationError",
    "HealthCheckError",
    
    # Configuration
    "SyftBoxConfig",
    "get_config",
    "is_syftbox_installed",
    "is_syftbox_running",
    "get_startup_instructions",
    "get_installation_instructions",
    "check_installation",
    "get_setup_instructions",
    
    # Services
    "ChatService",
    "ConversationManager",
    "SearchService",
    "BatchSearchService", 
    "HealthMonitor",
    
    # Filtering
    "ServiceFilter",
    "FilterCriteria",
    "FilterBuilder",
    "create_chat_services_filter",
    "create_search_services_filter",
    "create_free_services_filter",
    "create_paid_services_filter",
    "create_healthy_services_filter",
    "create_datasite_services_filter",
    "create_tag_services_filter",
    
    # Formatting
    "format_services_table",
    "format_service_details",
    "format_search_results",
    "format_chat_conversation",
    "format_health_summary",
    "format_statistics",
]