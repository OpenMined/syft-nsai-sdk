"""
Request builders for common patterns
"""
from typing import List, Optional

from ..core.types import ChatMessage, GenerationOptions, SearchOptions
from .requests import ChatRequest, SearchRequest


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