"""
Chat service client for SyftBox models
"""
import json
import uuid
import logging
from typing import List, Optional, Dict, Any

from ..core.types import (
    ModelInfo, 
    ChatMessage, 
    ChatRequest, 
    ChatResponse, 
    ChatUsage, 
    GenerationOptions, 
    ServiceType
)
from ..core.exceptions import ServiceNotSupportedError, RPCError, ValidationError, raise_service_not_supported
from ..networking.rpc_client import SyftBoxRPCClient

logger = logging.getLogger(__name__)

# ======================
# UPDATED SERVICE CLASSES
# ======================

class ChatService:
    """Service client for chat/conversation models."""
    """Updated chat service with explicit parameter handling."""
    
    def __init__(self, model_info: ModelInfo, rpc_client: SyftBoxRPCClient):
        """Initialize chat service.
        
        Args:
            model_info: Information about the model
            rpc_client: RPC client for making calls
            
        Raises:
            ServiceNotSupportedError: If model doesn't support chat
        """
        self.model_info = model_info
        self.rpc_client = rpc_client

        # Validate that model supports chat
        if not model_info.supports_service(ServiceType.CHAT):
            raise_service_not_supported(model_info.name, "chat", model_info)
    
    async def send_message_with_params(self, params: Dict[str, Any]) -> ChatResponse:
        """Send message with explicit parameters dictionary.
        
        Args:
            params: Dictionary of parameters including 'prompt' and optional params
            
        Returns:
            Chat response
        """
        # Validate required parameters
        if "prompt" not in params:
            raise ValidationError("'prompt' parameter is required")
        
        # Extract standard parameters
        prompt = params.pop("prompt")
        temperature = params.pop("temperature", None)
        max_tokens = params.pop("max_tokens", None)
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build RPC payload with all parameters
        payload = {
            "user_email": self.rpc_client._accounting_credentials.get('email', ''),
            "model": self.model_info.name,
            "messages": messages
        }
        
        # Add generation options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["maxTokens"] = max_tokens
        
        # Add any additional model-specific parameters
        for key, value in params.items():
            options[key] = value
        
        if options:
            payload["options"] = options
        
        # Make RPC call
        response_data = await self.rpc_client.call_chat(self.model_info, payload)
        return self._parse_rpc_response(response_data)

    async def send_message(self, 
                          message: str,
                          model: Optional[str] = None,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          top_p: Optional[float] = None,
                          stop_sequences: Optional[List[str]] = None,
                          user_email: Optional[str] = None,
                          transaction_token: Optional[str] = None) -> ChatResponse:
        """Send a single message to the chat model.
        
        Args:
            message: The message to send
            model: Optional model name override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            stop_sequences: Stop sequences for generation
            user_email: Email of the user making the request
            transaction_token: Payment token for paid services
            
        Returns:
            Chat response from the model
        """
        messages = [ChatMessage(role="user", content=message)]
        return await self.send_conversation(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            user_email=user_email,
            transaction_token=transaction_token
        )
    
    async def send_conversation(self,
                               messages: List[ChatMessage],
                               model: Optional[str] = None,
                               max_tokens: Optional[int] = None,
                               temperature: Optional[float] = None,
                               top_p: Optional[float] = None,
                               stop_sequences: Optional[List[str]] = None,
                               user_email: Optional[str] = None,
                               transaction_token: Optional[str] = None) -> ChatResponse:
        """Send a conversation to the chat model.
        
        Args:
            messages: List of messages in the conversation
            model: Optional model name override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            stop_sequences: Stop sequences for generation
            user_email: Email of the user making the request
            transaction_token: Payment token for paid services
            
        Returns:
            Chat response from the model
        """
        # Build generation options
        options = None
        if any([max_tokens, temperature, top_p, stop_sequences]):
            options = GenerationOptions(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences
            )
        
        # Create request
        request = ChatRequest(
            messages=messages,
            model=model or self.model_info.name,
            options=options,
            user_email=user_email or self.rpc_client.from_email,
            transaction_token=transaction_token
        )
        
        # Build RPC payload
        payload = self._build_rpc_payload(request)
        
        try:
            # Make RPC call
            response_data = await self.rpc_client.call_chat(self.model_info, payload)
            
            # Parse response
            return self._parse_rpc_response(response_data)
            
        except Exception as e:
            logger.error(f"Chat request failed for model {self.model_info.name}: {e}")
            raise
    
    def _build_rpc_payload(self, request: ChatRequest) -> Dict[str, Any]:
        """Build RPC payload from chat request.
        
        Args:
            request: Chat request object
            
        Returns:
            Dictionary payload for RPC call
        """
        payload = {
            "userEmail": request.user_email,
            "model": request.model,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {})
                }
                for msg in request.messages
            ]
        }
        
        # Add generation options if provided
        if request.options:
            options = {}
            if request.options.max_tokens is not None:
                options["maxTokens"] = request.options.max_tokens
            if request.options.temperature is not None:
                options["temperature"] = request.options.temperature
            if request.options.top_p is not None:
                options["topP"] = request.options.top_p
            if request.options.stop_sequences is not None:
                options["stopSequences"] = request.options.stop_sequences
            
            if options:
                payload["options"] = options
        
        # Add transaction token if provided
        if request.transaction_token:
            payload["transactionToken"] = request.transaction_token
            logger.debug(f"Added transaction token to chat request for {self.model_info.name}")
        
        return payload
    
    def _parse_rpc_response(self, response_data: Dict[str, Any]) -> ChatResponse:
        """Parse RPC response into ChatResponse object.
        
        Args:
            response_data: Raw response data from RPC call
            
        Returns:
            Parsed ChatResponse object
        """
        try:
            # Handle different response formats
            if "message" in response_data:
                # Standard format
                message_data = response_data["message"]
                message = ChatMessage(
                    role=message_data.get("role", "assistant"),
                    content=message_data.get("content", ""),
                    name=message_data.get("name")
                )
                
                # Parse usage information
                usage_data = response_data.get("usage", {})
                usage = ChatUsage(
                    prompt_tokens=usage_data.get("promptTokens", 0),
                    completion_tokens=usage_data.get("completionTokens", 0),
                    total_tokens=usage_data.get("totalTokens", 0)
                )
                
                return ChatResponse(
                    id=response_data.get("id", str(uuid.uuid4())),
                    model=response_data.get("model", self.model_info.name),
                    message=message,
                    usage=usage,
                    cost=response_data.get("cost"),
                    provider_info=response_data.get("providerInfo")
                )
            
            elif "content" in response_data:
                # Simple format - just content
                message = ChatMessage(
                    role="assistant",
                    content=response_data["content"]
                )
                
                usage = ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                
                return ChatResponse(
                    id=str(uuid.uuid4()),
                    model=self.model_info.name,
                    message=message,
                    usage=usage,
                    cost=response_data.get("cost"),
                    provider_info=response_data.get("providerInfo")
                )
            
            else:
                # Fallback - treat whole response as content
                message = ChatMessage(
                    role="assistant",
                    content=str(response_data)
                )
                
                usage = ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                
                return ChatResponse(
                    id=str(uuid.uuid4()),
                    model=self.model_info.name,
                    message=message,
                    usage=usage
                )
                
        except Exception as e:
            raise RPCError(f"Failed to parse chat response: {e}")
    
    @property
    def pricing(self) -> float:
        """Get pricing for chat service."""
        chat_service = self.model_info.get_service_info(ServiceType.CHAT)
        return chat_service.pricing if chat_service else 0.0
    
    @property
    def charge_type(self) -> str:
        """Get charge type for chat service."""
        chat_service = self.model_info.get_service_info(ServiceType.CHAT)
        return chat_service.charge_type.value if chat_service else "per_request"
    
    def estimate_cost(self, message_count: int = 1) -> float:
        """Estimate cost for a chat request.
        
        Args:
            message_count: Number of messages in conversation
            
        Returns:
            Estimated cost
        """
        if self.charge_type == "per_request":
            return self.pricing
        elif self.charge_type == "per_token":
            # Rough estimation - actual cost depends on token count
            estimated_tokens = message_count * 50  # Very rough estimate
            return self.pricing * estimated_tokens
        else:
            return self.pricing


class ConversationManager:
    """Helper class for managing multi-turn conversations."""
    
    def __init__(self, chat_service: ChatService):
        """Initialize conversation manager.
        
        Args:
            chat_service: Chat service to use
        """
        self.chat_service = chat_service
        self.messages: List[ChatMessage] = []
        self.system_message: Optional[str] = None
    
    def set_system_message(self, message: str):
        """Set or update the system message.
        
        Args:
            message: System message content
        """
        self.system_message = message
    
    def add_message(self, role: str, content: str, name: Optional[str] = None):
        """Add a message to the conversation.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            name: Optional message author name
        """
        message = ChatMessage(role=role, content=content, name=name)
        self.messages.append(message)
    
    async def send_message(self, 
                          message: str,
                          **kwargs) -> ChatResponse:
        """Send a message and update conversation history.
        
        Args:
            message: User message to send
            **kwargs: Additional arguments for chat service
            
        Returns:
            Chat response
        """
        # Build complete conversation
        conversation = []
        
        # Add system message if set
        if self.system_message:
            conversation.append(ChatMessage(role="system", content=self.system_message))
        
        # Add conversation history
        conversation.extend(self.messages)
        
        # Add new user message
        user_message = ChatMessage(role="user", content=message)
        conversation.append(user_message)
        
        # Send conversation
        response = await self.chat_service.send_conversation(
            messages=conversation,
            **kwargs
        )
        
        # Update conversation history
        self.messages.append(user_message)
        self.messages.append(response.message)
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        assistant_messages = [msg for msg in self.messages if msg.role == "assistant"]
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "has_system_message": self.system_message is not None,
            "estimated_tokens": sum(len(msg.content.split()) for msg in self.messages) * 1.3  # Rough estimate
        }