"""
Request data classes for SyftBox services
"""
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, EmailStr, field_validator
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
    user_email: str = Field(..., description="User email address")
    transaction_token: Optional[str] = Field(None, description="Transaction token")
    request_id: Optional[str] = Field(None, description="Request ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Pydantic models for validation and serialization
class ChatMessageModel(BaseModel):
    """Pydantic model for chat messages."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional author name")
    
    @field_validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v


class GenerationOptionsModel(BaseModel):
    """Pydantic model for generation options."""
    max_tokens: Optional[int] = Field(None, ge=1, le=8000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences for generation")
    
    class Config:
        extra = "allow"  # Allow additional generation parameters


class ChatRequestModel(BaseModel):
    """Pydantic model for chat requests."""
    user_email: EmailStr = Field(..., description="User email address")
    model: str = Field(..., description="Model name or identifier")
    messages: List[ChatMessageModel] = Field(..., description="Conversation messages")
    options: Optional[GenerationOptionsModel] = Field(None, description="Generation options")
    transaction_token: Optional[str] = Field(None, description="Payment token for paid services")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_email": "user@example.com",
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "options": {
                    "max_tokens": 150,
                    "temperature": 0.7
                }
            }
        }


class SearchOptionsModel(BaseModel):
    """Pydantic model for search options."""
    limit: Optional[int] = Field(3, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: Optional[bool] = Field(None, description="Include document metadata")
    include_embeddings: Optional[bool] = Field(None, description="Include vector embeddings")
    
    class Config:
        extra = "allow"  # Allow searcher-specific extensions


class SearchRequestModel(BaseModel):
    """Pydantic model for search requests."""
    user_email: EmailStr = Field(..., description="User email address")
    query: str = Field(..., min_length=1, description="Search query")
    options: Optional[SearchOptionsModel] = Field(SearchOptionsModel(), description="Search options")
    transaction_token: Optional[str] = Field(None, description="Payment token for paid services")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_email": "user@example.com",
                "query": "machine learning tutorials",
                "options": {
                    "limit": 5,
                    "similarity_threshold": 0.7
                }
            }
        }


class HealthCheckRequestModel(BaseModel):
    """Pydantic model for health check requests."""
    user_email: EmailStr = Field(..., description="User email address")
    include_details: Optional[bool] = Field(False, description="Include detailed health information")
    timeout: Optional[float] = Field(5.0, ge=0.1, le=30.0, description="Request timeout in seconds")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


# Dataclass versions for internal use
@dataclass
class ChatRequest(BaseRequest):
    """Chat request data class."""
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    options: Optional[GenerationOptions] = Field(None, description="Generation options")
    
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
    query: str = Field(..., description="Search query")
    options: Optional[SearchOptions] = Field(None, description="Search options")

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
    endpoint: str = Field(..., description="API endpoint")
    method: RequestMethod = Field(RequestMethod.POST, description="HTTP method")
    payload: Optional[Dict[str, Any]] = Field(None, description="Request payload")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")

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


# Request builders for common patterns
class ChatRequestBuilder:
    """Builder for chat requests."""
    
    def __init__(self, user_email: str, model: str):
        self.user_email = user_email
        self.model = model
        self.messages: List[ChatMessage] = []
        self.options: Optional[GenerationOptions] = None
        self.transaction_token: Optional[str] = None
    
    def add_message(self, role: str, content: str, name: Optional[str] = None) -> 'ChatRequestBuilder':
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(role=role, content=content, name=name))
        return self
    
    def add_user_message(self, content: str) -> 'ChatRequestBuilder':
        """Add a user message."""
        return self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> 'ChatRequestBuilder':
        """Add an assistant message."""
        return self.add_message("assistant", content)
    
    def add_system_message(self, content: str) -> 'ChatRequestBuilder':
        """Add a system message."""
        return self.add_message("system", content)
    
    def with_options(self, **options) -> 'ChatRequestBuilder':
        """Set generation options."""
        self.options = GenerationOptions(**options)
        return self
    
    def with_token(self, token: str) -> 'ChatRequestBuilder':
        """Set transaction token."""
        self.transaction_token = token
        return self
    
    def build(self) -> ChatRequest:
        """Build the chat request."""
        return ChatRequest(
            user_email=self.user_email,
            model=self.model,
            messages=self.messages,
            options=self.options,
            transaction_token=self.transaction_token
        )


class SearchRequestBuilder:
    """Builder for search requests."""
    
    def __init__(self, user_email: str, query: str):
        self.user_email = user_email
        self.query = query
        self.options: Optional[SearchOptions] = None
        self.transaction_token: Optional[str] = None
    
    def with_limit(self, limit: int) -> 'SearchRequestBuilder':
        """Set result limit."""
        if self.options is None:
            self.options = SearchOptions()
        self.options.limit = limit
        return self
    
    def with_threshold(self, threshold: float) -> 'SearchRequestBuilder':
        """Set similarity threshold."""
        if self.options is None:
            self.options = SearchOptions()
        self.options.similarity_threshold = threshold
        return self
    
    def with_metadata(self, include: bool = True) -> 'SearchRequestBuilder':
        """Include metadata in results."""
        if self.options is None:
            self.options = SearchOptions()
        self.options.include_metadata = include
        return self
    
    def with_embeddings(self, include: bool = True) -> 'SearchRequestBuilder':
        """Include embeddings in results."""
        if self.options is None:
            self.options = SearchOptions()
        self.options.include_embeddings = include
        return self
    
    def with_token(self, token: str) -> 'SearchRequestBuilder':
        """Set transaction token."""
        self.transaction_token = token
        return self
    
    def build(self) -> SearchRequest:
        """Build the search request."""
        return SearchRequest(
            user_email=self.user_email,
            query=self.query,
            options=self.options,
            transaction_token=self.transaction_token
        )


# Validation functions
def validate_chat_request(request: Dict[str, Any]) -> ChatRequestModel:
    """Validate and parse chat request."""
    return ChatRequestModel(**request)


def validate_search_request(request: Dict[str, Any]) -> SearchRequestModel:
    """Validate and parse search request."""
    return SearchRequestModel(**request)


def validate_health_request(request: Dict[str, Any]) -> HealthCheckRequestModel:
    """Validate and parse health check request."""
    return HealthCheckRequestModel(**request)


# Factory functions
def create_chat_request(user_email: str, model: str, message: str, **options) -> ChatRequest:
    """Create a simple chat request."""
    builder = ChatRequestBuilder(user_email, model)
    builder.add_user_message(message)
    
    if options:
        builder.with_options(**options)
    
    return builder.build()


def create_search_request(user_email: str, query: str, **options) -> SearchRequest:
    """Create a simple search request."""
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
    return ChatRequest(
        user_email=user_email,
        model=model,
        messages=messages,
        options=GenerationOptions(**options) if options else None
    )