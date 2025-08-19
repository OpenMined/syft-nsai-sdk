"""
Response data classes for SyftBox services
"""
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid

from ..core.types import ChatMessage, ChatUsage, DocumentResult


class ResponseStatus(Enum):
    """Response status values."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    TIMEOUT = "timeout"


class FinishReason(Enum):
    """Reasons why generation finished."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


@dataclass
class BaseResponse:
    """Base class for all responses."""
    id: str
    status: ResponseStatus = ResponseStatus.SUCCESS
    timestamp: datetime = field(default_factory=datetime.now)
    cost: Optional[float] = None
    provider_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


# Pydantic models for validation and serialization
class LogProbsModel(BaseModel):
    """Log probabilities for generated tokens."""
    token_logprobs: Dict[str, float] = Field(..., description="Map of tokens to log probabilities")


class ChatUsageModel(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., ge=0, description="Tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Tokens in the completion")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    
    @validator('total_tokens')
    def validate_total(cls, v, values):
        if 'prompt_tokens' in values and 'completion_tokens' in values:
            expected = values['prompt_tokens'] + values['completion_tokens']
            if v != expected:
                raise ValueError(f'Total tokens {v} != prompt + completion {expected}')
        return v


class ChatMessageModel(BaseModel):
    """Chat message in response."""
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional author name")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v


class ChatResponseModel(BaseModel):
    """Pydantic model for chat responses."""
    id: str = Field(..., description="Unique response ID")
    model: str = Field(..., description="Model that generated the response")
    message: ChatMessageModel = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(None, description="Why generation stopped")
    usage: ChatUsageModel = Field(..., description="Token usage information")
    cost: Optional[float] = Field(None, ge=0, description="Cost of the request")
    provider_info: Optional[Dict[str, Any]] = Field(None, description="Provider-specific information")
    logprobs: Optional[LogProbsModel] = Field(None, description="Log probabilities")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chat-12345",
                "model": "gpt-4",
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                },
                "cost": 0.05
            }
        }


class DocumentResultModel(BaseModel):
    """Document search result."""
    id: str = Field(..., description="Document identifier")
    score: float = Field(..., ge=0, le=1, description="Similarity score")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")


class SearchResponseModel(BaseModel):
    """Pydantic model for search responses."""
    id: str = Field(..., description="Unique response ID")
    query: str = Field(..., description="Original search query")
    results: List[DocumentResultModel] = Field(..., description="Search results")
    cost: Optional[float] = Field(None, ge=0, description="Cost of the request")
    provider_info: Optional[Dict[str, Any]] = Field(None, description="Provider-specific information")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "search-67890",
                "query": "machine learning",
                "results": [
                    {
                        "id": "doc-1",
                        "score": 0.95,
                        "content": "Machine learning is a subset of artificial intelligence...",
                        "metadata": {"filename": "ml_intro.pdf"}
                    }
                ],
                "cost": 0.02
            }
        }


class HealthStatusModel(BaseModel):
    """Health check status information."""
    status: str = Field(..., description="Health status (ok, error)")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    version: Optional[str] = Field(None, description="Service version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class HealthResponseModel(BaseModel):
    """Pydantic model for health check responses."""
    id: str = Field(..., description="Unique response ID")
    project_name: str = Field(..., description="Name of the project/service")
    status: str = Field(..., description="Overall health status")
    services: Dict[str, Any] = Field(..., description="Status of individual services")
    timestamp: Optional[datetime] = Field(None, description="Response timestamp")


# Dataclass versions for internal use
@dataclass
class ChatResponse(BaseResponse):
    """Chat response data class."""
    model: str
    message: ChatMessage
    usage: ChatUsage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, float]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatResponse':
        """Create ChatResponse from dictionary."""
        # Parse message
        message_data = data.get('message', {})
        message = ChatMessage(
            role=message_data.get('role', 'assistant'),
            content=message_data.get('content', ''),
            name=message_data.get('name')
        )
        
        # Parse usage
        usage_data = data.get('usage', {})
        usage = ChatUsage(
            prompt_tokens=usage_data.get('promptTokens', 0),
            completion_tokens=usage_data.get('completionTokens', 0),
            total_tokens=usage_data.get('totalTokens', 0)
        )
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            model=data.get('model', 'unknown'),
            message=message,
            usage=usage,
            finish_reason=data.get('finishReason'),
            cost=data.get('cost'),
            provider_info=data.get('providerInfo'),
            logprobs=data.get('logprobs', {}).get('tokenLogprobs') if data.get('logprobs') else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "model": self.model,
            "message": {
                "role": self.message.role,
                "content": self.message.content,
                **({"name": self.message.name} if self.message.name else {})
            },
            "finish_reason": self.finish_reason,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            },
            "cost": self.cost,
            "provider_info": self.provider_info,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "logprobs": {"token_logprobs": self.logprobs} if self.logprobs else None
        }


@dataclass
class SearchResponse(BaseResponse):
    """Search response data class."""
    query: str
    results: List[DocumentResult]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], original_query: str) -> 'SearchResponse':
        """Create SearchResponse from dictionary."""
        results = []
        
        results_data = data.get('results', [])
        if isinstance(results_data, list):
            for result_data in results_data:
                if isinstance(result_data, dict):
                    result = DocumentResult(
                        id=result_data.get('id', str(uuid.uuid4())),
                        score=float(result_data.get('score', 0.0)),
                        content=result_data.get('content', ''),
                        metadata=result_data.get('metadata'),
                        embedding=result_data.get('embedding')
                    )
                    results.append(result)
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            query=data.get('query', original_query),
            results=results,
            cost=data.get('cost'),
            provider_info=data.get('providerInfo')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "query": self.query,
            "results": [
                {
                    "id": result.id,
                    "score": result.score,
                    "content": result.content,
                    "metadata": result.metadata,
                    "embedding": result.embedding
                }
                for result in self.results
            ],
            "cost": self.cost,
            "provider_info": self.provider_info,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HealthResponse(BaseResponse):
    """Health check response data class."""
    project_name: str
    services: Dict[str, Any]
    uptime: Optional[float] = None
    version: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthResponse':
        """Create HealthResponse from dictionary."""
        # Determine status
        status_str = data.get('status', 'unknown').lower()
        if status_str in ['ok', 'healthy', 'up']:
            status = ResponseStatus.SUCCESS
        else:
            status = ResponseStatus.ERROR
        
        return cls(
            id=str(uuid.uuid4()),
            status=status,
            project_name=data.get('project_name', 'unknown'),
            services=data.get('services', {}),
            uptime=data.get('uptime'),
            version=data.get('version'),
            provider_info=data
        )
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.status == ResponseStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "status": "ok" if self.status == ResponseStatus.SUCCESS else "error",
            "project_name": self.project_name,
            "services": self.services,
            "uptime": self.uptime,
            "version": self.version,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ErrorResponse(BaseResponse):
    """Error response data class."""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.status = ResponseStatus.ERROR
    
    @classmethod
    def from_exception(cls, exception: Exception, request_id: Optional[str] = None) -> 'ErrorResponse':
        """Create ErrorResponse from exception."""
        return cls(
            id=request_id or str(uuid.uuid4()),
            error_code=exception.__class__.__name__,
            error_message=str(exception),
            error_details=getattr(exception, 'details', None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "status": self.status.value,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AsyncResponse(BaseResponse):
    """Response for asynchronous operations."""
    request_id: str
    poll_url: Optional[str] = None
    estimated_completion_time: Optional[datetime] = None
    
    def __post_init__(self):
        self.status = ResponseStatus.PENDING
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AsyncResponse':
        """Create AsyncResponse from dictionary."""
        return cls(
            id=str(uuid.uuid4()),
            request_id=data.get('request_id', ''),
            poll_url=data.get('data', {}).get('poll_url'),
            estimated_completion_time=None  # Could parse from data if available
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "status": self.status.value,
            "request_id": self.request_id,
            "poll_url": self.poll_url,
            "estimated_completion_time": self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            "timestamp": self.timestamp.isoformat()
        }


# Response parsers for different formats
class ResponseParser:
    """Parser for converting raw responses to typed response objects."""
    
    @staticmethod
    def parse_chat_response(data: Dict[str, Any]) -> ChatResponse:
        """Parse chat response from raw data."""
        return ChatResponse.from_dict(data)
    
    @staticmethod
    def parse_search_response(data: Dict[str, Any], query: str) -> SearchResponse:
        """Parse search response from raw data."""
        return SearchResponse.from_dict(data, query)
    
    @staticmethod
    def parse_health_response(data: Dict[str, Any]) -> HealthResponse:
        """Parse health response from raw data."""
        return HealthResponse.from_dict(data)
    
    @staticmethod
    def parse_error_response(data: Dict[str, Any]) -> ErrorResponse:
        """Parse error response from raw data."""
        return ErrorResponse(
            id=str(uuid.uuid4()),
            error_code=data.get('error_code', 'UNKNOWN_ERROR'),
            error_message=data.get('message', data.get('error', 'Unknown error')),
            error_details=data.get('details')
        )
    
    @staticmethod
    def parse_async_response(data: Dict[str, Any]) -> AsyncResponse:
        """Parse async response from raw data."""
        return AsyncResponse.from_dict(data)


# Factory functions for creating responses
def create_successful_chat_response(model: str, content: str, **kwargs) -> ChatResponse:
    """Create a successful chat response."""
    return ChatResponse(
        id=str(uuid.uuid4()),
        model=model,
        message=ChatMessage(role="assistant", content=content),
        usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        **kwargs
    )


def create_successful_search_response(query: str, results: List[DocumentResult], **kwargs) -> SearchResponse:
    """Create a successful search response."""
    return SearchResponse(
        id=str(uuid.uuid4()),
        query=query,
        results=results,
        **kwargs
    )


def create_error_response(error_message: str, error_code: str = "ERROR", **kwargs) -> ErrorResponse:
    """Create an error response."""
    return ErrorResponse(
        id=str(uuid.uuid4()),
        error_code=error_code,
        error_message=error_message,
        **kwargs
    )


def create_health_response(project_name: str, is_healthy: bool = True, **kwargs) -> HealthResponse:
    """Create a health check response."""
    return HealthResponse(
        id=str(uuid.uuid4()),
        status=ResponseStatus.SUCCESS if is_healthy else ResponseStatus.ERROR,
        project_name=project_name,
        services={"status": "ok" if is_healthy else "error"},
        **kwargs
    )