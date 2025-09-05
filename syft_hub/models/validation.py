"""
Pydantic validation models for API requests and responses
"""
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum

from ..core.types import ServiceType, HealthStatus
from ..utils.validator import (
    sanitize_string,
    sanitize_tags,
    validate_chat_message,
    validate_max_tokens,
    validate_temperature,
    validate_email,
    validate_service_name,
    validate_similarity_threshold,
    validate_search_query,
    validate_tags,
    validate_cost
)

# class ServiceTypeEnum(str, Enum):
#     """Service types for validation."""
#     CHAT = "chat"
#     SEARCH = "search"


# class HealthStatusEnum(str, Enum):
#     """Health status values for validation."""
#     ONLINE = "online"
#     OFFLINE = "offline"
#     TIMEOUT = "timeout"
#     UNKNOWN = "unknown"
#     NOT_APPLICABLE = "n/a"

class ChatMessageModel(BaseModel):
    """Pydantic model for chat messages."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional author name")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        # Use the existing validation from utils
        if not validate_chat_message(v):
            raise ValueError('Message content is invalid or too long')
        return v


class GenerationOptionsModel(BaseModel):
    """Pydantic model for generation options."""
    max_tokens: Optional[int] = Field(None, ge=1, le=100000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences for generation")
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None:
            if not validate_max_tokens(v):
                raise ValueError('Invalid max_tokens value')
        return v
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is not None:
            if not validate_temperature(v):
                raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    class Config:
        extra = "allow"


class ChatRequestModel(BaseModel):
    """Pydantic model for chat requests."""
    user_email: EmailStr = Field(..., description="User email address")
    model: str = Field(..., description="Model name or identifier")
    messages: List[ChatMessageModel] = Field(..., description="Conversation messages")
    options: Optional[GenerationOptionsModel] = Field(None, description="Generation options")
    transaction_token: Optional[str] = Field(None, description="Payment token for paid services")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    @field_validator('user_email', mode='before')
    @classmethod
    def validate_user_email(cls, v):
        email_str = str(v)
        if not validate_email(email_str):
            raise ValueError('Invalid email format')
        return email_str
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if not validate_service_name(v):
            raise ValueError('Invalid model name format')
        return v
    
    @field_validator('messages')
    @classmethod
    def validate_messages_not_empty(cls, v):
        if not v:
            raise ValueError('Messages cannot be empty')
        return v


class SearchOptionsModel(BaseModel):
    """Pydantic model for search options."""
    limit: Optional[int] = Field(3, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: Optional[bool] = Field(None, description="Include document metadata")
    include_embeddings: Optional[bool] = Field(None, description="Include vector embeddings")
    
    @field_validator('similarity_threshold')
    @classmethod
    def validate_threshold(cls, v):
        if v is not None:
            if not validate_similarity_threshold(v):
                raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v


class SearchRequestModel(BaseModel):
    """Pydantic model for search requests."""
    user_email: EmailStr = Field(..., description="User email address")
    query: str = Field(..., min_length=1, description="Search query")
    options: Optional[SearchOptionsModel] = Field(None, description="Search options")
    transaction_token: Optional[str] = Field(None, description="Payment token for paid services")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    @field_validator('user_email', mode='before')
    @classmethod
    def validate_user_email(cls, v):
        email_str = str(v)
        if not validate_email(email_str):
            raise ValueError('Invalid email format')
        return email_str
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not validate_search_query(v):
            raise ValueError('Invalid search query')
        return v


class HealthCheckRequestModel(BaseModel):
    """Pydantic model for health check requests."""
    user_email: EmailStr = Field(..., description="User email address")
    include_details: Optional[bool] = Field(False, description="Include detailed health information")
    timeout: Optional[float] = Field(5.0, ge=0.1, le=30.0, description="Request timeout in seconds")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    @field_validator('user_email', mode='before')
    @classmethod
    def validate_user_email(cls, v):
        email_str = str(v)
        if not validate_email(email_str):
            raise ValueError('Invalid email format')
        return email_str


# Response Models

class ChatUsageModel(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., ge=0, description="Tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Tokens in the completion")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    
    @field_validator('total_tokens')
    @classmethod
    def validate_total(cls, v, info):
        if hasattr(info, 'data') and 'prompt_tokens' in info.data and 'completion_tokens' in info.data:
            expected = info.data['prompt_tokens'] + info.data['completion_tokens']
            if v != expected:
                raise ValueError(f'Total tokens {v} != prompt + completion {expected}')
        return v


class ChatResponseModel(BaseModel):
    """Pydantic model for chat responses."""
    id: str = Field(..., description="Unique response ID")
    model: str = Field(..., description="Service that generated the response")
    message: ChatMessageModel = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(None, description="Why generation stopped")
    usage: ChatUsageModel = Field(..., description="Token usage information")
    cost: Optional[float] = Field(None, ge=0, description="Cost of the request")
    provider_info: Optional[Dict[str, Any]] = Field(None, description="Provider-specific information")


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


class HealthResponseModel(BaseModel):
    """Pydantic model for health check responses."""
    id: str = Field(..., description="Unique response ID")
    project_name: str = Field(..., description="Name of the project/service")
    status: str = Field(..., description="Overall health status")
    services: Dict[str, Any] = Field(..., description="Status of individual services")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    version: Optional[str] = Field(None, description="Service version")


# Filter and Utility Models

class ServiceFilterModel(BaseModel):
    """Pydantic model for service filter criteria."""
    datasite: Optional[str] = Field(None, description="Filter by datasite email")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    max_cost: Optional[float] = Field(None, description="Maximum cost filter")
    min_cost: Optional[float] = Field(None, description="Minimum cost filter")
    service_types: Optional[List[ServiceType]] = Field(None, description="Filter by service types")
    health_status: Optional[HealthStatus] = Field(None, description="Filter by health status")

    @field_validator('datasite')
    @classmethod
    def validate_datasite(cls, v):
        if v is not None:
            if not validate_email(v):
                raise ValueError('Datasite must be a valid email address')
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags_field(cls, v):
        if v is not None:
            if not validate_tags(v):
                raise ValueError('Invalid tags format')
        return v
    
    @field_validator('max_cost')
    @classmethod
    def validate_max_cost(cls, v):
        if v is not None:
            if not validate_cost(v):
                raise ValueError('Invalid max_cost value')
        return v
    
    @field_validator('min_cost')
    @classmethod
    def validate_min_cost(cls, v):
        if v is not None:
            if not validate_cost(v):
                raise ValueError('Invalid min_cost value')
        return v


class UserAccountModel(BaseModel):
    """User account information."""
    email: EmailStr = Field(..., description="User email address")
    balance: float = Field(ge=0.0, default=0.0, description="Account balance")
    password: str = Field(..., description="User password")
    organization: Optional[str] = Field(None, description="User organization")
    
    @field_validator('email', mode='before')
    @classmethod
    def validate_email_field(cls, v):
        email_str = str(v)
        if not validate_email(email_str):
            raise ValueError('Invalid email format')
        return email_str
    
    @field_validator('balance')
    @classmethod
    def validate_balance(cls, v):
        if not validate_cost(v):
            raise ValueError('Invalid balance value')
        return v


# Validation Functions

def validate_chat_request(data: Dict[str, Any]) -> ChatRequestModel:
    """Validate chat request data."""
    return ChatRequestModel(**data)


def validate_search_request(data: Dict[str, Any]) -> SearchRequestModel:
    """Validate search request data."""
    return SearchRequestModel(**data)


def validate_health_request(data: Dict[str, Any]) -> HealthCheckRequestModel:
    """Validate health check request data."""
    return HealthCheckRequestModel(**data)


def validate_service_filters(data: Dict[str, Any]) -> ServiceFilterModel:
    """Validate service filter criteria."""
    return ServiceFilterModel(**data)


# Integration helpers that combine validation with sanitization

def validate_and_sanitize_chat_request(data: Dict[str, Any]) -> ChatRequestModel:
    """Validate and sanitize chat request using both approaches."""
    # First sanitize using existing utils
    if 'messages' in data:
        for message in data['messages']:
            if 'content' in message:
                message['content'] = sanitize_string(message['content'], 50000)
    
    # Then validate with Pydantic
    return validate_chat_request(data)


def validate_and_sanitize_search_request(data: Dict[str, Any]) -> SearchRequestModel:
    """Validate and sanitize search request."""
    # Sanitize query
    if 'query' in data:
        data['query'] = sanitize_string(data['query'], 1000)
    
    # Validate with Pydantic
    return validate_search_request(data)


def validate_and_sanitize_service_filters(data: Dict[str, Any]) -> ServiceFilterModel:
    """Validate and sanitize service filter criteria."""
    # Sanitize tags if present
    if 'tags' in data and data['tags']:
        data['tags'] = sanitize_tags(data['tags'])
    
    # Validate with Pydantic
    return validate_service_filters(data)