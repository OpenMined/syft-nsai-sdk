"""
Chat service client for SyftBox services
"""
import logging
from typing import Dict, Any

from ..core.exceptions import RPCError, ValidationError, ServiceNotSupportedError
from ..core.types import ServiceType
from ..clients.rpc_client import SyftBoxRPCClient
from ..models.responses import ChatResponse
from ..models.service_info import ServiceInfo
from ..utils.estimator import CostEstimator

logger = logging.getLogger(__name__)

class ChatService:
    """Service client for chat services."""
    
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
            raise ServiceNotSupportedError(service_info.name, "chat", service_info)

    def _parse_rpc_chat_response(self, response_data: Dict[str, Any]) -> ChatResponse:
        """Parse RPC response into ChatResponse object.
        
        Handles the actual SyftBox response format for chat:
        {
            "data": {
                "message": {
                    "body": {
                        "id": "uuid-string",
                        "model": "claude-sonnet-3.5",
                        "message": {"content": "...", "role": "assistant"},
                        "finishReason": "stop",  # camelCase
                        "usage": {"promptTokens": 113, "completionTokens": 45, "totalTokens": 158},
                        "cost": 0.3,
                        "providerInfo": {...},  # camelCase
                        "logprobs": {...}
                    }
                }
            }
        }
        
        Args:
            response_data: Raw response data from RPC call matching schema.py format
            
        Returns:
            Parsed ChatResponse object
        """
        
        try:
            # Extract the actual response body from SyftBox nested structure
            if "data" in response_data and "message" in response_data["data"]:
                message_data = response_data["data"]["message"]
                
                if "body" in message_data and isinstance(message_data["body"], dict):
                    # Extract the body and convert to schema.py format
                    body = message_data["body"]
                    return ChatResponse.from_dict(body)
            
            # If not nested format, try direct parsing
            return ChatResponse.from_dict(response_data)
                
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
        return self._parse_rpc_chat_response(response_data)
        
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