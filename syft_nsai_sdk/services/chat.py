"""
Chat service client for SyftBox services
"""
import json
import uuid
import logging
from typing import List, Optional, Dict, Any

from ..models.service_info import ServiceInfo
from ..core.types import (
    ChatMessage,
    ChatUsage, 
    GenerationOptions,
    PricingChargeType, 
    ServiceType
)
from ..core.exceptions import (
    ServiceNotSupportedError, 
    RPCError, 
    ValidationError, 
    raise_service_not_supported
)
from ..clients.rpc_client import SyftBoxRPCClient
from ..models.responses import ChatResponse
from ..utils.estimator import CostEstimator

logger = logging.getLogger(__name__)

class ChatService:
    """Service client for chat/conversation services."""
    
    def __init__(self, service_info: ServiceInfo, rpc_client: SyftBoxRPCClient):
        """Initialize chat service.
        
        Args:
            service_info: Information about the service
            rpc_client: RPC client for making calls
            
        Raises:
            ServiceNotSupportedError: If service doesn't support chat
        """
        self.service_info = service_info
        self.rpc_client = rpc_client

        # Validate that service supports chat
        if not service_info.supports_service(ServiceType.CHAT):
            raise_service_not_supported(service_info.name, "chat", service_info)

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
                "service": "claude-sonnet-3.5"
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
                        model=body.get("model", self.service_info.name),
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
                    model=response_data.get("model", self.service_info.name),
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
                    model=self.service_info.name,
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
                    model=self.service_info.name,
                    message=message,
                    usage=usage
                )
                
        except Exception as e:
            logger.error(f"Failed to parse chat response: {e}")
            logger.error(f"Response data: {response_data}")
            raise RPCError(f"Failed to parse chat response: {e}")
    
    async def chat_with_params(self, params: Dict[str, Any]) -> ChatResponse:
        """Send message with explicit parameters dictionary.
        
        Args:
            params: Dictionary of parameters including 'prompt' and optional params
            
        Returns:
            Chat response
        """
        # Validate required parameters
        if "messages" not in params:
            raise ValidationError("'messages' parameter is required")
        
        # Extract standard parameters
        params = params.copy()
        messages = params.pop("messages")
        temperature = params.pop("temperature", None)
        max_tokens = params.pop("max_tokens", None)
        
        # Build messages
        # messages = [{"role": "user", "content": messages}]
        
        # Build RPC payload with all parameters
        account_email = self.rpc_client.accounting_client.get_email()
        payload = {
            "user_email": account_email,
            "model": self.service_info.name,
            "messages": messages
        }
        
        # Add generation options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["maxTokens"] = max_tokens
        
        # Add any additional service-specific parameters
        for key, value in params.items():
            options[key] = value
        
        if options:
            payload["options"] = options

        # Make RPC call
        response_data = await self.rpc_client.call_chat(self.service_info, payload)
        return self._parse_rpc_response(response_data)
    
    def estimate_cost1(self, message_count: int = 1) -> float:
        """Estimate cost for a chat request."""
        chat_service_info = self.service_info.get_service_info(ServiceType.CHAT)
        if not chat_service_info:
            return 0.0
        
        if chat_service_info.charge_type == PricingChargeType.PER_REQUEST:
            return chat_service_info.pricing * message_count
        else:
            return chat_service_info.pricing
        
    def estimate_cost(self, message_count: int = 1) -> float:
        return CostEstimator.estimate_chat_cost(self.service_info, message_count)

    @property
    def pricing(self) -> float:
        """Get pricing for chat service."""
        chat_service = self.service_info.get_service_info(ServiceType.CHAT)
        return chat_service.pricing if chat_service else 0.0
    
    @property
    def charge_type(self) -> str:
        """Get charge type for chat service."""
        chat_service = self.service_info.get_service_info(ServiceType.CHAT)
        return chat_service.charge_type.value if chat_service else "per_request"