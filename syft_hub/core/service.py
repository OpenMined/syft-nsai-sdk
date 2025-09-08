"""
Service class for object-oriented service interaction
"""
from typing import List, TYPE_CHECKING

from ..core.types import ServiceType
from ..models.service_info import ServiceInfo
from .exceptions import ServiceNotSupportedError

if TYPE_CHECKING:
    from ..main import Client

class Service:
    """Object-oriented interface for a loaded SyftBox service."""
    
    def __init__(self, service_info: ServiceInfo, client: 'Client'):
        self._service_info = service_info
        self._client = client
    
    # Properties
    @property
    def name(self) -> str:
        """Service name (without datasite prefix)."""
        return self._service_info.name
    
    @property
    def datasite(self) -> str:
        """Datasite email that owns this service."""
        return self._service_info.datasite
    
    @property
    def full_name(self) -> str:
        """Full service identifier: datasite/name."""
        return f"{self.datasite}/{self.name}"
    
    @property
    def cost(self) -> float:
        """Minimum cost per request for this service."""
        return self._service_info.min_pricing
    
    @property
    def supports_chat(self) -> bool:
        """Whether this service supports chat operations."""
        return self._service_info.supports_service(ServiceType.CHAT)
    
    @property
    def supports_search(self) -> bool:
        """Whether this service supports search operations."""
        return self._service_info.supports_service(ServiceType.SEARCH)
    
    @property
    def summary(self) -> str:
        """Brief description of the service."""
        return self._service_info.summary or ""
    
    @property
    def tags(self) -> List[str]:
        """Tags associated with this service."""
        return self._service_info.tags or []
    
    def __contains__(self, capability: str) -> bool:
        """Support 'chat' in service or 'search' in service syntax."""
        if capability == 'chat':
            return self.supports_chat
        elif capability == 'search':
            return self.supports_search
        return False
    
    def __repr__(self) -> str:
        capabilities = []
        if self.supports_chat:
            capabilities.append("chat")
        if self.supports_search:
            capabilities.append("search")
        caps_str = ", ".join(capabilities) if capabilities else "none"
        return f"Service('{self.full_name}', capabilities=[{caps_str}], cost=${self.cost})"
    
    # Service methods (always present, error if not supported)
    def chat(self, messages, **kwargs):
        """Chat with this service synchronously.
        
        Args:
            messages: Chat messages to send
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            ChatResponse from the service
            
        Raises:
            ServiceNotSupportedError: If service doesn't support chat
        """
        if not self.supports_chat:
            raise ServiceNotSupportedError(f"Service '{self.name}' doesn't support chat")
        return self._client.chat(self.full_name, messages, **kwargs)
    
    async def chat_async(self, messages, **kwargs):
        """Chat with this service asynchronously.
        
        Args:
            messages: Chat messages to send
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            ChatResponse from the service
            
        Raises:
            ServiceNotSupportedError: If service doesn't support chat
        """
        if not self.supports_chat:
            raise ServiceNotSupportedError(f"Service '{self.name}' doesn't support chat")
        return await self._client.chat_async(self.full_name, messages, **kwargs)
    
    def search(self, message, **kwargs):
        """Search with this service synchronously.
        
        Args:
            message: Search query
            **kwargs: Additional parameters (topK, similarity_threshold, etc.)
            
        Returns:
            SearchResponse from the service
            
        Raises:
            ServiceNotSupportedError: If service doesn't support search
        """
        if not self.supports_search:
            raise ServiceNotSupportedError(f"Service '{self.name}' doesn't support search")
        return self._client.search(self.full_name, message, **kwargs)
    
    async def search_async(self, message, **kwargs):
        """Search with this service asynchronously.
        
        Args:
            message: Search query
            **kwargs: Additional parameters (topK, similarity_threshold, etc.)
            
        Returns:
            SearchResponse from the service
            
        Raises:
            ServiceNotSupportedError: If service doesn't support search
        """
        if not self.supports_search:
            raise ServiceNotSupportedError(f"Service '{self.name}' doesn't support search")
        return await self._client.search_async(self.full_name, message, **kwargs)
    
    def show_example(self) -> str:
        """Show usage examples for this service.
        
        Returns:
            Formatted usage examples
        """
        examples = []
        examples.append(f"# Usage examples for {self.name}")
        examples.append(f"# Datasite: {self.datasite}")
        examples.append("")
        
        # Object-oriented examples (using Service object)
        examples.append("## Using Service object:")
        examples.append(f'service = client.load_service("{self.full_name}")')
        examples.append("")
        
        if self.supports_chat:
            examples.extend([
                "# Basic chat",
                'response = service.chat("Hello! How are you?")',
                "",
                "# Chat with parameters",
                'response = service.chat(',
                '    messages="Write a story",',
                '    temperature=0.7,',
                '    max_tokens=200',
                ')',
                ""
            ])
        
        if self.supports_search:
            examples.extend([
                "# Basic search",
                'results = service.search("machine learning")',
                "",
                "# Search with parameters", 
                'results = service.search(',
                '    message="latest AI research",',
                '    topK=10,',
                '    similarity_threshold=0.8',
                ')',
                ""
            ])
        
        # Direct client examples
        examples.append("## Using client directly:")
        
        if self.supports_chat:
            examples.extend([
                "# Basic chat",
                f'response = await client.chat(',
                f'    service_name="{self.full_name}",',
                f'    messages="Hello! How are you?"',
                f')',
                ""
            ])
        
        if self.supports_search:
            examples.extend([
                "# Basic search",
                f'results = await client.search(',
                f'    service_name="{self.full_name}",',
                f'    message="machine learning"',
                f')',
                ""
            ])
        
        # Add pricing info
        if self.cost > 0:
            examples.append(f"# Cost: ${self.cost} per request")
        else:
            examples.append("# Cost: Free")
        
        return "\n".join(examples)