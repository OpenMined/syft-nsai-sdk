"""
Core type definitions for SyftBox NSAI SDK
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, EmailStr, field_validator
from pathlib import Path


class ServiceType(Enum):
    """Types of services that services can provide."""
    CHAT = "chat"
    SEARCH = "search"


class ServiceStatus(Enum):
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

class PricingChargeType(Enum):
    """How services charge for their services."""
    PER_REQUEST = "per_request"

class UserAccount(BaseModel):
    """User account information."""
    email: EmailStr = Field(..., description="User email address")
    balance: float = Field(ge=0.0, default=0.0, description="Account balance")
    password: str = Field(..., description="User password")
    organization: Optional[str] = Field(None, description="User organization")

class APIException(Exception):
    """Generic HTTP exception with status code"""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@dataclass
class ServiceItem:
    """Information about a specific item within a service."""
    type: ServiceType = Field(..., description="Type of service")
    enabled: bool = Field(..., description="Is the service enabled?")
    pricing: float = Field(..., description="Pricing for the service")
    charge_type: PricingChargeType = Field(..., description="Charge type for the service")

@dataclass
class ServiceSpec:
    """Internal representation of a service with parameters"""
    name: str = Field(..., description="Service name")
    params: Dict[str, Any] = Field(..., description="Service parameters")

@dataclass
class ServiceInfo:
    """Comprehensive information about a discovered model."""
    # Basic metadata from metadata.json
    name: str = Field(..., description="Model name")
    datasite: str = Field(..., description="Data site")
    summary: str = Field(..., description="Model summary")
    description: str = Field(..., description="Model description")
    tags: List[str] = Field(..., description="Model tags")

    # Service information
    services: List[ServiceItem] = Field(..., description="List of services")

    # Status information
    config_status: ServiceStatus = Field(..., description="Configuration status")
    health_status: Optional[HealthStatus] = Field(None, description="Health status")

    # Technical details
    delegate_email: Optional[str] = Field(None, description="Delegate email address")
    endpoints: Dict[str, Any] = Field(default_factory=dict, description="Service endpoints")
    rpc_schema: Dict[str, Any] = Field(default_factory=dict, description="RPC schema")

    # File system paths
    metadata_path: Path = Field(None, description="Path to metadata file")
    rpc_schema_path: Path = Field(None, description="Path to RPC schema file")

    # Computed properties
    service_urls: Dict[ServiceType, str] = Field(default_factory=dict)

    @property
    def has_enabled_services(self) -> bool:
        """Check if service has any enabled services."""
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

    def get_service_info(self, service_type: ServiceType) -> Optional[ServiceItem]:
        """Get service info for a specific service type."""
        for service in self.services:
            if service.type == service_type:
                return service
        return None
    
    def supports_service(self, service_type: ServiceType) -> bool:
        """Check if service supports and has enabled a specific service type."""
        service = self.get_service_info(service_type)
        return service is not None and service.enabled


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the message sender")


@dataclass
class GenerationOptions:
    """Options for text generation."""
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences for generation")


@dataclass
class ChatRequest:
    """Request for chat completion."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Model name")
    options: Optional[GenerationOptions] = Field(None, description="Generation options")
    user_email: Optional[str] = Field(None, description="User email address")
    transaction_token: Optional[str] = Field(None, description="Transaction token")


@dataclass
class ChatUsage:
    """Token usage information for chat requests."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


@dataclass
class ChatResponse:
    """Response from a chat completion request."""
    id: str = Field(..., description="Unique identifier for the response")
    model: str = Field(..., description="Model used for the response")
    message: ChatMessage = Field(..., description="Message content")
    usage: ChatUsage = Field(..., description="Token usage information")
    cost: Optional[float] = Field(None, description="Cost of the request")
    provider_info: Optional[Dict[str, Any]] = Field(None, description="Information about the service provider")

    def __str__(self) -> str:
        """Return just the message content for easy printing."""
        return self.message.content
    
    def __repr__(self) -> str:
        """Return full object representation for debugging."""
        return f"ChatResponse(id='{self.id}', model='{self.model}', message={self.message!r}, usage={self.usage!r}, cost={self.cost}, provider_info={self.provider_info})"

@dataclass
class SearchOptions:
    """Options for document search."""
    limit: Optional[int] = Field(3, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: Optional[bool] = Field(None, description="Include document metadata")
    include_embeddings: Optional[bool] = Field(None, description="Include vector embeddings")


@dataclass
class SearchRequest:
    """Request for document search."""
    query: str = Field(..., description="Search query")
    options: Optional[SearchOptions] = Field(None, description="Search options")
    user_email: Optional[str] = Field(None, description="User email address")
    transaction_token: Optional[str] = Field(None, description="Transaction token")


@dataclass
class DocumentResult:
    """A document result from search."""
    id: str = Field(..., description="Unique identifier for the document")
    score: float = Field(..., description="Relevance score of the document")
    content: str = Field(..., description="Content of the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the document")


@dataclass
class SearchResponse:
    """Response from a document search operation."""
    id: str = Field(..., description="Unique identifier for the response")
    query: str = Field(..., description="Search query")
    results: List[DocumentResult] = Field(..., description="Search results")
    cost: Optional[float] = Field(None, description="Cost of the request")
    provider_info: Optional[Dict[str, Any]] = Field(None, description="Information about the service provider")

    def __str__(self) -> str:
        """Return formatted search results for easy printing."""
        if not self.results:
            return "No results found."
        
        parts = [f"Search results for: '{self.query}'"]
        for i, result in enumerate(self.results, 1):
            parts.append(f"\n{i}. Score: {result.score:.3f}")
            parts.append(f"   {result.content[:100]}{'...' if len(result.content) > 100 else ''}")
        
        return "\n".join(parts)

@dataclass
class TransactionToken:
    """Transaction token for paid services."""
    token: str = Field(..., description="Transaction token")
    recipient_email: str = Field(..., description="Recipient email address")


# Filter types for service discovery
FilterDict = Dict[str, Any]