"""
SyftBox NSAI SDK

A Python SDK for discovering and using AI models across the SyftBox network.
"""

__version__ = "0.1.0"
__author__ = "SyftBox Team"
__email__ = "support@syftbox.net"

# Main client class
from .main import SyftBoxClient

# Core types and enums
from .core.types import (
    ServiceType,
    ModelStatus, 
    HealthStatus,
    QualityPreference,
    PricingChargeType,
    ModelInfo,
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
    ModelNotFoundError,
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
    is_syftbox_available,
    get_installation_instructions,
)

# Service clients (for advanced usage)
from .services.chat import ChatService, ConversationManager
from .services.search import SearchService, BatchSearchService
from .services.health import HealthMonitor

# Filtering utilities
from .discovery.filters import (
    ModelFilter,
    FilterCriteria,
    FilterBuilder,
    create_chat_models_filter,
    create_search_models_filter,
    create_free_models_filter,
    create_premium_models_filter,
    create_healthy_models_filter,
    create_owner_models_filter,
    create_tag_models_filter,
)

# Convenience functions
# from .main import (
#     list_available_models,
# )

# Formatting utilities
from .utils.formatting import (
    format_models_table,
    format_model_details,
    format_search_results,
    format_chat_conversation,
    format_health_summary,
    format_statistics,
)


# Package-level convenience functions
def create_client(**kwargs) -> SyftBoxClient:
    """Create a SyftBoxClient with optional configuration.
    
    Args:
        **kwargs: Configuration options for the client
        
    Returns:
        SyftBoxClient instance
    """
    return SyftBoxClient(**kwargs)


def check_installation() -> bool:
    """Check if SyftBox is properly installed and configured.
    
    Returns:
        True if SyftBox is available, False otherwise
    """
    return is_syftbox_available()


def get_setup_instructions() -> str:
    """Get instructions for setting up SyftBox.
    
    Returns:
        Setup instructions as string
    """
    return get_installation_instructions()


# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Main client
    "SyftBoxClient",
    "create_client",
    
    # Core types
    "ServiceType",
    "ModelStatus",
    "HealthStatus", 
    "QualityPreference",
    "PricingChargeType",
    "ModelInfo",
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
    "ModelNotFoundError",
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
    "is_syftbox_available",
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
    "ModelFilter",
    "FilterCriteria",
    "FilterBuilder",
    "create_chat_models_filter",
    "create_search_models_filter",
    "create_free_models_filter",
    "create_premium_models_filter",
    "create_healthy_models_filter",
    "create_owner_models_filter",
    "create_tag_models_filter",
    
    # Formatting
    "format_models_table",
    "format_model_details",
    "format_search_results",
    "format_chat_conversation",
    "format_health_summary",
    "format_statistics",
]