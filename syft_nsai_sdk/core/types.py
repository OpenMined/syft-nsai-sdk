"""
Core type definitions for SyftBox NSAI SDK
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


class ServiceType(Enum):
    """Types of services that models can provide."""
    CHAT = "chat"
    SEARCH = "search"


class ModelStatus(Enum):
    """Configuration status of a model based on metadata."""
    ACTIVE = "Active"
    DISABLED = "Disabled"


class HealthStatus(Enum):
    """Runtime health status of a model service."""
    ONLINE = "online"
    OFFLINE = "offline"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "n/a"


class QualityPreference(Enum):
    """Quality preference for model selection."""
    CHEAPEST = "cheapest"
    BALANCED = "balanced"
    PREMIUM = "premium"
    FASTEST = "fastest"


class PricingChargeType(Enum):
    """How models charge for their services."""
    PER_REQUEST = "per_request"
    PER_TOKEN = "per_token"
    PER_MINUTE = "per_minute"
    FLAT_RATE = "flat_rate"


@dataclass
class ServiceInfo:
    """Information about a specific service within a model."""
    type: ServiceType
    enabled: bool
    pricing: float
    charge_type: PricingChargeType


@dataclass
class ModelInfo:
    """Comprehensive information about a discovered model."""
    # Basic metadata from metadata.json
    name: str
    owner: str
    summary: str
    description: str
    tags: List[str]
    
    # Service information
    services: List[ServiceInfo]
    
    # Status information
    config_status: ModelStatus
    health_status: Optional[HealthStatus] = None
    
    # Technical details
    delegate_email: Optional[str] = None
    endpoints: Dict[str, Any] = field(default_factory=dict)
    rpc_schema: Dict[str, Any] = field(default_factory=dict)
    
    # File system paths
    metadata_path: Path = None
    rpc_schema_path: Path = None
    
    # Computed properties
    service_urls: Dict[ServiceType, str] = field(default_factory=dict)
    
    @property
    def has_enabled_services(self) -> bool:
        """Check if model has any enabled services."""
        return any(service.enabled for service in self.services)
    
    @property
    def enabled_service_types(self) -> List[ServiceType]:
        """Get list of enabled service types."""
        return [service.type for service in self.services if service.enabled]
    
    @property
    def min_pricing(self) -> float:
        """Get minimum pricing across all enabled services."""
        enabled_services = [s for s in self.services if s.enabled]
        if not enabled_services:
            return 0.0
        return min(service.pricing for service in enabled_services)
    
    @property
    def max_pricing(self) -> float:
        """Get maximum pricing across all enabled services."""
        enabled_services = [s for s in self.services if s.enabled]
        if not enabled_services:
            return 0.0
        return max(service.pricing for service in enabled_services)
    
    def get_service_info(self, service_type: ServiceType) -> Optional[ServiceInfo]:
        """Get service info for a specific service type."""
        for service in self.services:
            if service.type == service_type:
                return service
        return None
    
    def supports_service(self, service_type: ServiceType) -> bool:
        """Check if model supports and has enabled a specific service type."""
        service = self.get_service_info(service_type)
        return service is not None and service.enabled


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


@dataclass
class GenerationOptions:
    """Options for text generation."""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None


@dataclass
class ChatRequest:
    """Request for chat completion."""
    messages: List[ChatMessage]
    model: Optional[str] = None
    options: Optional[GenerationOptions] = None
    user_email: Optional[str] = None
    transaction_token: Optional[str] = None


@dataclass
class ChatUsage:
    """Token usage information for chat requests."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    """Response from a chat completion request."""
    id: str
    model: str
    message: ChatMessage
    usage: ChatUsage
    cost: Optional[float] = None
    provider_info: Optional[Dict[str, Any]] = None


@dataclass
class SearchOptions:
    """Options for document search."""
    limit: Optional[int] = 3
    similarity_threshold: Optional[float] = None
    include_metadata: Optional[bool] = None
    include_embeddings: Optional[bool] = None


@dataclass
class SearchRequest:
    """Request for document search."""
    query: str
    options: Optional[SearchOptions] = None
    user_email: Optional[str] = None
    transaction_token: Optional[str] = None


@dataclass
class DocumentResult:
    """A document result from search."""
    id: str
    score: float
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None


@dataclass
class SearchResponse:
    """Response from a document search operation."""
    id: str
    query: str
    results: List[DocumentResult]
    cost: Optional[float] = None
    provider_info: Optional[Dict[str, Any]] = None


@dataclass
class TransactionToken:
    """Transaction token for paid services."""
    token: str
    recipient_email: str


# Filter types for model discovery
FilterDict = Dict[str, Any]