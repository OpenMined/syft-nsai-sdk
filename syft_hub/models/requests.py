"""
Request data classes for SyftBox services
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from ..core.types import ChatMessage, GenerationOptions, SearchOptions

class RequestMethod(Enum):
    """HTTP methods for requests."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class BaseRequest:
    """Base class for all requests."""
    user_email: str = ""
    transaction_token: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatRequest(BaseRequest):
    """Chat request data class."""
    model: str = ""
    messages: List[ChatMessage] = field(default_factory=list)
    options: Optional[GenerationOptions] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "userEmail": self.user_email,
            "model": self.model,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {})
                }
                for msg in self.messages
            ]
        }
        
        if self.options:
            options_dict = {}
            if self.options.max_tokens is not None:
                options_dict["maxTokens"] = self.options.max_tokens
            if self.options.temperature is not None:
                options_dict["temperature"] = self.options.temperature
            if self.options.top_p is not None:
                options_dict["topP"] = self.options.top_p
            if self.options.stop_sequences is not None:
                options_dict["stopSequences"] = self.options.stop_sequences
            
            if options_dict:
                data["options"] = options_dict
        
        if self.transaction_token:
            data["transactionToken"] = self.transaction_token
        
        if self.request_id:
            data["requestId"] = self.request_id
        
        return data


@dataclass
class SearchRequest(BaseRequest):
    """Search request data class."""
    query: str = ""
    options: Optional[SearchOptions] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "userEmail": self.user_email,
            "query": self.query,
        }
        
        if self.options:
            options_dict = {}
            if self.options.limit is not None:
                options_dict["limit"] = self.options.limit
            if self.options.similarity_threshold is not None:
                options_dict["similarityThreshold"] = self.options.similarity_threshold
            if self.options.include_metadata is not None:
                options_dict["includeMetadata"] = self.options.include_metadata
            if self.options.include_embeddings is not None:
                options_dict["includeEmbeddings"] = self.options.include_embeddings
            
            data["options"] = options_dict
        else:
            data["options"] = {"limit": 3}  # Default
        
        if self.transaction_token:
            data["transactionToken"] = self.transaction_token
        
        if self.request_id:
            data["requestId"] = self.request_id
        
        return data


@dataclass
class HealthCheckRequest(BaseRequest):
    """Health check request data class."""
    include_details: bool = False
    timeout: float = 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "userEmail": self.user_email,
            "includeDetails": self.include_details,
            "timeout": self.timeout
        }
        
        if self.request_id:
            data["requestId"] = self.request_id
        
        return data


@dataclass
class CustomRequest(BaseRequest):
    """Custom request for arbitrary endpoints."""
    endpoint: str = ""
    method: RequestMethod = RequestMethod.POST
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = self.payload.copy() if self.payload else {}
        
        # Always include user email
        data["userEmail"] = self.user_email
        
        if self.transaction_token:
            data["transactionToken"] = self.transaction_token
        
        if self.request_id:
            data["requestId"] = self.request_id
        
        return data


# Factory functions
def create_chat_request(user_email: str, model: str, message: str, **options) -> ChatRequest:
    """Create a simple chat request."""
    from .builders import ChatRequestBuilder
    
    builder = ChatRequestBuilder(user_email, model)
    builder.add_user_message(message)
    
    if options:
        builder.with_options(**options)
    
    return builder.build()


def create_search_request(user_email: str, query: str, **options) -> SearchRequest:
    """Create a simple search request."""
    from .builders import SearchRequestBuilder
    
    builder = SearchRequestBuilder(user_email, query)
    
    if options:
        for key, value in options.items():
            if key == "limit":
                builder.with_limit(value)
            elif key == "similarity_threshold":
                builder.with_threshold(value)
            elif key == "include_metadata":
                builder.with_metadata(value)
            elif key == "include_embeddings":
                builder.with_embeddings(value)
    
    return builder.build()


def create_conversation_request(user_email: str, model: str, messages: List[ChatMessage], **options) -> ChatRequest:
    """Create a chat request from existing conversation."""
    from ..core.types import GenerationOptions
    
    return ChatRequest(
        user_email=user_email,
        model=model,
        messages=messages,
        options=GenerationOptions(**options) if options else None
    )