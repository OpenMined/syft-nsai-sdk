"""
Chat service client for SyftBox models
"""
import json
import uuid
import logging
from typing import List, Optional, Dict, Any

from typer import prompt

from ..core.types import (
    ModelInfo, 
    ChatMessage, 
    ChatRequest, 
    ChatResponse, 
    ChatUsage, 
    GenerationOptions,
    PricingChargeType, 
    ServiceType
)
from ..core.exceptions import ServiceNotSupportedError, RPCError, ValidationError, raise_service_not_supported
from ..networking.rpc_client import SyftBoxRPCClient

logger = logging.getLogger(__name__)

class ChatService:
    """Service client for chat/conversation models."""
    
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
        params = params.copy()
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
    
    def _parse_rpc_response(self, response_data: Dict[str, Any]) -> ChatResponse:
        """Parse RPC response into ChatResponse object.
        
        Handles the actual SyftBox response format:
        {
        "request_id": "...",
        "data": {
            "message": {
            "body": {
                "cost": 0.3,
                "message": {"content": "...", "role": "assistant"},
                "usage": {"completionTokens": 113, ...},
                "model": "claude-sonnet-3.5"
            }
            }
        }
        }
        
        Args:
            response_data: Raw response data from RPC call
            
        Returns:
            Parsed ChatResponse object
        """
        
        try:
            # Extract the actual response body from SyftBox nested structure
            if "data" in response_data and "message" in response_data["data"]:
                message_data = response_data["data"]["message"]
                
                if "body" in message_data and isinstance(message_data["body"], dict):
                    # This is the actual response content
                    body = message_data["body"]
                    
                    # Extract message content
                    if "message" in body and isinstance(body["message"], dict):
                        msg_content = body["message"]
                        message = ChatMessage(
                            role=msg_content.get("role", "assistant"),
                            content=msg_content.get("content", ""),
                            name=msg_content.get("name")
                        )
                    else:
                        # Fallback if message structure is different
                        message = ChatMessage(
                            role="assistant",
                            content=str(body.get("content", body))
                        )
                    
                    # Extract usage information
                    usage_data = body.get("usage", {})
                    usage = ChatUsage(
                        prompt_tokens=usage_data.get("promptTokens", 0),
                        completion_tokens=usage_data.get("completionTokens", 0),
                        total_tokens=usage_data.get("totalTokens", 0)
                    )
                    
                    return ChatResponse(
                        id=body.get("id", str(uuid.uuid4())),
                        model=body.get("model", self.model_info.name),
                        message=message,
                        usage=usage,
                        cost=body.get("cost"),
                        provider_info=body.get("providerInfo")
                    )
            
            # Handle legacy/direct response formats (backwards compatibility)
            if "message" in response_data:
                # Direct format
                message_data = response_data["message"]
                message = ChatMessage(
                    role=message_data.get("role", "assistant"),
                    content=message_data.get("content", ""),
                    name=message_data.get("name")
                )
                
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
                # Simple content format
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
                # Last resort - treat whole response as string content
                # But log this so we can debug new formats
                logger.warning(f"Unexpected response format, using fallback parsing: {response_data}")
                
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
            logger.error(f"Failed to parse chat response: {e}")
            logger.error(f"Response data: {response_data}")
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
        """Estimate cost for a chat request."""
        chat_service_info = self.model_info.get_service_info(ServiceType.CHAT)
        if not chat_service_info:
            return 0.0
        
        if chat_service_info.charge_type == PricingChargeType.PER_REQUEST:
            return chat_service_info.pricing * message_count
        elif chat_service_info.charge_type == PricingChargeType.PER_TOKEN:
            # Use actual per-token pricing from model info
            estimated_tokens = message_count * 50  # Still need rough token estimate
            return chat_service_info.pricing * estimated_tokens
        else:
            return chat_service_info.pricing


class ConversationManager:
    """Helper class for managing multi-turn conversations."""
    
    def __init__(self, chat_service: ChatService):
        """Initialize conversation manager.
        
        Args:
            chat_service: Chat service to use
        """
        self.chat_service = chat_service
        # self.messages: List[ChatMessage] = []
        self.messages: List[Dict[str, str]] = []
        self.system_message: Optional[str] = None
        self.max_exchanges = 2
    
    def set_system_message(self, message: str):
        """Set or update the system message.
        
        Args:
            message: System message content
        """
        self.system_message = message
    
    def add_message1(self, role: str, content: str, name: Optional[str] = None):
        """Add a message to the conversation.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            name: Optional message author name
        """
        message = ChatMessage(role=role, content=content, name=name)
        self.messages.append(message)

    def add_message(self, role: str, content: str, name: Optional[str] = None):
        msg = {"role": role, "content": content}
        if name:
            msg["name"] = name
        self.messages.append(msg)
    
    async def send_message(self, message: str, **kwargs) -> ChatResponse:
        """Send a message and update conversation history."""
        # Build context and send message
        context_prompt = self._build_context_prompt(message)
        params = {"prompt": context_prompt, **kwargs}
        response = await self.chat_service.send_message_with_params(params)
        
        # Update conversation history
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": response.message.content})
        
        # Auto-trim to keep only recent exchanges
        self._auto_trim()
        
        return response
    
    def _auto_trim(self):
        """Automatically trim conversation to keep only recent exchanges."""
        max_messages = self.max_exchanges * 2  # Each exchange = user + assistant
        if len(self.messages) > max_messages:
            # Keep only the most recent exchanges
            self.messages = self.messages[-max_messages:]
    
    def set_max_exchanges(self, max_exchanges: int):
        """Set maximum number of exchanges to retain."""
        self.max_exchanges = max_exchanges
        self._auto_trim()  # Trim immediately if needed
    
    def _build_context_prompt1(self, new_message: str) -> str:
        """Build a context-aware prompt from conversation history."""
        parts = []
        
        # Add system message if set
        if self.system_message:
            parts.append(f"system: {self.system_message}")
        
        # Add conversation history
        for msg in self.messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        
        # Add new user message and assistant prompt
        parts.append(f"user: {new_message}")
        parts.append("assistant:")
        
        return "\n".join(parts)
    
    def _build_context_prompt(self, new_message: str) -> str:
        """Build a context-aware prompt from conversation history."""
        parts = []
        
        if self.system_message:
            parts.append(f"system: {self.system_message}")
        
        # Add conversation history
        for msg in self.messages:
            # Ensure content is properly cleaned
            content = str(msg["content"]).strip()
            parts.append(f"{msg['role']}: {content}")
        
        parts.append(f"user: {new_message.strip()}")
        parts.append("assistant:")
        
        context = "\n".join(parts)
        
        # Debug: print context length
        print(f"Context length: {len(context)} characters")
        
        return context
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        user_messages = [msg for msg in self.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.messages if msg["role"] == "assistant"]
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "has_system_message": self.system_message is not None,
            "estimated_tokens": sum(len(msg["content"].split()) for msg in self.messages) * 1.3
        }