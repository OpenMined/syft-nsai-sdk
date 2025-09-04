"""
Response data classes for SyftBox services
"""
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ResponseStatus = ResponseStatus.SUCCESS
    timestamp: datetime = field(default_factory=datetime.now)
    cost: Optional[float] = None
    provider_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse(BaseResponse):
    """Chat response data class."""
    model: str = ""
    message: Optional[ChatMessage] = None
    usage: Optional[ChatUsage] = None
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

    def __str__(self) -> str:
        """Return just the message content for easy printing."""
        return self.message.content
    
    def __repr__(self) -> str:
        """Return full object representation for debugging."""
        return f"ChatResponse(id='{self.id}', model='{self.model}', message={self.message!r}, usage={self.usage!r}, cost={self.cost}, provider_info={self.provider_info})"


@dataclass
class SearchResponse(BaseResponse):
    """Search response data class."""
    query: str = ""
    results: List[DocumentResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, response_data: Dict[str, Any], original_query: str) -> 'SearchResponse':
        """Create SearchResponse from RPC response data.
        
        Expects schema.py format:
        {
            "id": "uuid-string",
            "query": "search query", 
            "results": [
                {
                    "id": "doc-id",
                    "score": 0.95,
                    "content": "document content",
                    "metadata": {...},
                    "embedding": [...]
                }
            ],
            "provider_info": {...},
            "cost": 0.1
        }
        """
        results = []
        
        results_data = response_data.get('results', [])
        for result_data in results_data:
            result = DocumentResult(
                id=result_data.get('id', str(uuid.uuid4())),
                score=float(result_data.get('score', 0.0)),
                content=result_data.get('content', ''),
                metadata=result_data.get('metadata'),
                embedding=result_data.get('embedding')
            )
            results.append(result)
        
        return cls(
            id=response_data.get('id', str(uuid.uuid4())),
            query=response_data.get('query', original_query),
            results=results,
            cost=response_data.get('cost'),
            provider_info=response_data.get('provider_info')
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
            "provider_info": self.provider_info,
            "cost": self.cost
        }

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
class HealthResponse(BaseResponse):
    """Health check response data class."""
    project_name: str = ""
    services: Dict[str, Any] = field(default_factory=dict)
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
    error_code: str = ""
    error_message: str = ""
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
    request_id: str = ""
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