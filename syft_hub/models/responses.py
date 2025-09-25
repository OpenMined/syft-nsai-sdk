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
    logprobs: Optional[Dict[str, Any]] = None  # Changed from Dict[str, float] to Any

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
            finish_reason=data.get('finishReason'),  # camelCase from endpoint
            cost=data.get('cost'),
            provider_info=data.get('providerInfo'),  # camelCase from endpoint
            logprobs=data.get('logprobs')  # Direct assignment, no nested extraction
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "model": self.model,
            "message": {
                "role": self.message.role if self.message else "assistant",
                "content": self.message.content if self.message else "",
            },
            "finish_reason": self.finish_reason,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens if self.usage else 0,
                "completion_tokens": self.usage.completion_tokens if self.usage else 0,
                "total_tokens": self.usage.total_tokens if self.usage else 0
            },
            "cost": self.cost,
            "provider_info": self.provider_info,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "logprobs": self.logprobs
        }
        
        # Add message name if present
        if self.message and self.message.name:
            result["message"]["name"] = self.message.name
            
        return result

    def __str__(self) -> str:
        """Return just the message content for easy printing."""
        return self.message.content if self.message else ""
    
    def __repr__(self) -> str:
        """Return user-friendly chat response display similar to service repr."""
        try:
            from IPython.display import display, HTML
            # In notebook environment, show rich display
            self._display_rich()
            return ""  # Return empty string to avoid double output
        except ImportError:
            # Not in notebook - provide comprehensive text representation
            lines = [
                f"Chat Response [{self.status.value.title()}]",
                "",
                f"Model:           {self.model}",
            ]
            
            if self.message:
                # Show content preview (first 100 chars)
                content_preview = self.message.content[:100] + "..." if len(self.message.content) > 100 else self.message.content
                lines.append(f"Content:         {content_preview}")
                lines.append(f"Role:            {self.message.role}")
            
            if self.usage:
                lines.append(f"Tokens:          {self.usage.total_tokens} total ({self.usage.prompt_tokens} prompt + {self.usage.completion_tokens} completion)")
            
            if self.cost is not None:
                lines.append(f"Cost:            ${self.cost:.4f}")
            
            if self.finish_reason:
                lines.append(f"Finish Reason:   {self.finish_reason}")
                
            lines.append(f"Timestamp:       {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(lines)
    
    def _display_rich(self) -> None:
        """Display rich HTML representation in Jupyter notebooks."""
        from IPython.display import display, HTML
        
        # Status badge styling
        status_class = "badge-ready" if self.status.value == "success" else "badge-not-ready"
        
        # Content preview
        content_preview = ""
        if self.message:
            content = self.message.content
            if len(content) > 200:
                content_preview = content[:200] + "..."
            else:
                content_preview = content
        
        # Usage info
        usage_text = ""
        if self.usage:
            usage_text = f"{self.usage.total_tokens} tokens ({self.usage.prompt_tokens} + {self.usage.completion_tokens})"
        
        # Cost info
        cost_text = f"${self.cost:.4f}" if self.cost is not None else "Free"
        
        html = f"""
        <style>
            .chat-response-widget {{
                font-family: system-ui, -apple-system, sans-serif;
                padding: 12px 0;
                color: #333;
                line-height: 1.5;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background: #fafafa;
                padding: 16px;
                margin: 8px 0;
            }}
            .widget-title {{
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 12px;
                color: #333;
            }}
            .status-line {{
                display: flex;
                align-items: flex-start;
                margin: 6px 0;
                font-size: 13px;
            }}
            .status-label {{
                color: #666;
                min-width: 100px;
                margin-right: 12px;
                flex-shrink: 0;
            }}
            .status-value {{
                font-family: monospace;
                color: #333;
                word-break: break-word;
            }}
            .status-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                margin-left: 8px;
            }}
            .badge-ready {{
                background: #d4edda;
                color: #155724;
            }}
            .badge-not-ready {{
                background: #f8d7da;
                color: #721c24;
            }}
            .content-preview {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 8px;
                font-family: inherit;
                font-size: 12px;
                color: #495057;
                white-space: pre-wrap;
                max-height: 120px;
                overflow-y: auto;
            }}
        </style>
        
        <div class="chat-response-widget">
            <div class="widget-title">
                Chat Response <span class="status-badge {status_class}">{self.status.value.title()}</span>
            </div>
            
            <div class="status-line">
                <span class="status-label">Content:</span>
            </div>
            <div class="content-preview">{content_preview}</div>
            
            <div class="status-line">
                <span class="status-label">Usage:</span>
                <span class="status-value">{usage_text}</span>
            </div>
            
            <div class="status-line">
                <span class="status-label">Cost:</span>
                <span class="status-value">{cost_text}</span>
            </div>
            
            <div class="status-line">
                <span class="status-label">Timestamp:</span>
                <span class="status-value">{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </div>
        """
        
        display(HTML(html))

@dataclass
class SearchResponse(BaseResponse):
    """Search response data class."""
    query: str = ""
    results: List[DocumentResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, response_data: Dict[str, Any], original_query: str) -> 'SearchResponse':
        """Create SearchResponse from RPC response data.
        
        Expects schema.py format with camelCase fields:
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
            "providerInfo": {...},  # camelCase from endpoint
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
            provider_info=response_data.get('providerInfo')  # camelCase from endpoint
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
            "cost": self.cost,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat()
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

    def __repr__(self) -> str:
        """Return user-friendly search response display similar to pipeline result."""
        try:
            from IPython.display import display, HTML
            # In notebook environment, show rich display
            self._display_rich()
            return ""  # Return empty string to avoid double output
        except ImportError:
            # Not in notebook - provide comprehensive text representation
            lines = [
                f"Search Response [{self.status.value.title()}]",
                "",
            ]
            
            # Show query
            lines.append(f"Query:           {self.query}")
            lines.append("")
            
            # Show results count and top results
            if self.results:
                lines.append(f"Results Found:   {len(self.results)} documents")
                
                # Show top 3 results with scores
                lines.append(f"Top Results:")
                for i, result in enumerate(self.results[:3], 1):
                    content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                    lines.append(f"  {i}. Score {result.score:.3f}: {content_preview}")
                
                if len(self.results) > 3:
                    lines.append(f"  ... and {len(self.results) - 3} more results")
                    
            else:
                lines.append(f"Results Found:   No documents found")
            
            lines.append("")
            
            # Show cost if available
            if self.cost is not None:
                lines.append(f"Total Cost:      ${self.cost:.4f}")
            
            # Show timestamp
            lines.append(f"Timestamp:       {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(lines)
    
    def _display_rich(self) -> None:
        """Display rich HTML representation in Jupyter notebooks."""
        from IPython.display import display, HTML
        
        # Status badge styling
        status_class = "badge-ready" if self.status.value == "success" else "badge-not-ready"
        
        # Results preview
        results_preview = ""
        if self.results:
            results_preview = f"<div class='results-list'>"
            for i, result in enumerate(self.results[:3], 1):
                content = result.content[:150] + "..." if len(result.content) > 150 else result.content
                results_preview += f"""
                <div class='result-item'>
                    <div class='result-header'>
                        <span class='result-rank'>{i}.</span>
                        <span class='result-score'>Score: {result.score:.3f}</span>
                    </div>
                    <div class='result-content'>{content}</div>
                </div>
                """
            if len(self.results) > 3:
                results_preview += f"<div class='more-results'>... and {len(self.results) - 3} more results</div>"
            results_preview += "</div>"
        else:
            results_preview = "<div class='no-results'>No documents found</div>"
        
        # Cost info
        cost_text = f"${self.cost:.4f}" if self.cost is not None else "Free"
        
        html = f"""
        <style>
            .search-response-widget {{
                font-family: system-ui, -apple-system, sans-serif;
                padding: 12px 0;
                color: #333;
                line-height: 1.5;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background: #fafafa;
                padding: 16px;
                margin: 8px 0;
            }}
            .widget-title {{
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 12px;
                color: #333;
            }}
            .status-line {{
                display: flex;
                align-items: flex-start;
                margin: 6px 0;
                font-size: 13px;
            }}
            .status-label {{
                color: #666;
                min-width: 100px;
                margin-right: 12px;
                flex-shrink: 0;
            }}
            .status-value {{
                font-family: monospace;
                color: #333;
                word-break: break-word;
            }}
            .status-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                margin-left: 8px;
            }}
            .badge-ready {{
                background: #d4edda;
                color: #155724;
            }}
            .badge-not-ready {{
                background: #f8d7da;
                color: #721c24;
            }}
            .results-list {{
                margin-top: 8px;
            }}
            .result-item {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 8px;
                margin: 4px 0;
            }}
            .result-header {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 4px;
                font-size: 12px;
            }}
            .result-rank {{
                font-weight: 600;
                color: #495057;
            }}
            .result-score {{
                color: #6c757d;
                font-family: monospace;
            }}
            .result-content {{
                font-size: 12px;
                color: #495057;
                white-space: pre-wrap;
                max-height: 60px;
                overflow: hidden;
            }}
            .more-results {{
                font-size: 12px;
                color: #6c757d;
                font-style: italic;
                text-align: center;
                padding: 4px;
            }}
            .no-results {{
                font-size: 12px;
                color: #6c757d;
                font-style: italic;
                text-align: center;
                padding: 8px;
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
            }}
        </style>
        
        <div class="search-response-widget">
            <div class="widget-title">
                Search Response <span class="status-badge {status_class}">{self.status.value.title()}</span>
            </div>
            
            <div class="status-line">
                <span class="status-label">Query:</span>
                <span class="status-value">{self.query}</span>
            </div>
            
            <div class="status-line">
                <span class="status-label">Results:</span>
                <span class="status-value">{len(self.results)} documents found</span>
            </div>
            
            <div class="status-line">
                <span class="status-label">Cost:</span>
                <span class="status-value">{cost_text}</span>
            </div>
            
            <div class="status-line">
                <span class="status-label">Timestamp:</span>
                <span class="status-value">{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            
            {results_preview}
        </div>
        """
        
        display(HTML(html))

@dataclass
class HealthResponse(BaseResponse):
    """Health check response data class."""
    project_name: str = ""
    services: Dict[str, Any] = field(default_factory=dict)
    health_status: str = ""  # Separate from BaseResponse.status to avoid confusion
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthResponse':
        """Create HealthResponse from dictionary.
        
        Expects endpoint format with camelCase:
        {
            "status": "ok",
            "projectName": "my-project",  # camelCase from endpoint
            "services": {...}
        }
        """
        # Determine ResponseStatus from health status
        health_status = data.get('status', 'unknown')
        response_status = ResponseStatus.SUCCESS if health_status.lower() in ['ok', 'healthy', 'up'] else ResponseStatus.ERROR
        
        return cls(
            id=str(uuid.uuid4()),
            status=response_status,
            health_status=health_status,
            project_name=data.get('projectName', 'unknown'),  # camelCase from endpoint
            services=data.get('services', {}),
            provider_info=data  # Store full response as provider info
        )
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.status == ResponseStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "status": self.health_status,  # Use the actual health status from endpoint
            "project_name": self.project_name,
            "services": self.services,
            "response_status": self.status.value,  # Include BaseResponse status separately
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
        health_status="ok" if is_healthy else "error",
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