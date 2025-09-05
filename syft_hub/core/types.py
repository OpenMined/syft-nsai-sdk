"""
Core type definitions for SyftBox NSAI SDK - Basic types and enums only
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


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


# Basic data classes - keep these in core
@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class ChatUsage:
    """Token usage information for chat requests."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class GenerationOptions:
    """Options for text generation."""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None


@dataclass
class SearchOptions:
    """Options for document search."""
    limit: Optional[int] = 3
    similarity_threshold: Optional[float] = None
    include_metadata: Optional[bool] = None
    include_embeddings: Optional[bool] = None


@dataclass
class DocumentResult:
    """A document result from search."""
    id: str
    score: float
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None


@dataclass
class ServiceItem:
    """Information about a specific item within a service."""
    type: ServiceType
    enabled: bool
    pricing: float
    charge_type: PricingChargeType


@dataclass
class ServiceSpec:
    """Internal representation of a service with parameters"""
    name: str
    params: Dict[str, Any]


@dataclass
class TransactionToken:
    """Transaction token for paid services."""
    token: str
    recipient_email: str


# Filter types for service discovery
FilterDict = Dict[str, Any]


# Exceptions
class APIException(Exception):
    """Generic HTTP exception with status code"""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)