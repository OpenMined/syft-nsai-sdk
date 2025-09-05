"""
Search service client for SyftBox services
"""
import logging
from typing import Dict, Any

from ..core.types import ServiceType
from ..core.exceptions import RPCError, ValidationError, ServiceNotSupportedError
from ..clients.rpc_client import SyftBoxRPCClient
from ..models.responses import SearchResponse
from ..models.service_info import ServiceInfo
from ..utils.estimator import CostEstimator

logger = logging.getLogger(__name__)

class SearchService:
    """Service client for document search services."""
    
    def __init__(self, service_info: ServiceInfo, rpc_client: SyftBoxRPCClient):
        """Initialize search service.
        
        Args:
            service_info: Information about the service
            rpc_client: RPC client for making calls
            
        Raises:
            ServiceNotSupportedError: If service doesn't support search
        """
        self.service_info = service_info
        self.rpc_client = rpc_client
        
        # Validate that service supports search
        if not service_info.supports_service(ServiceType.SEARCH):
            raise ServiceNotSupportedError(service_info.name, "search", service_info)

    def _parse_rpc_search_response(self, response_data: Dict[str, Any], original_query: str) -> SearchResponse:
        """Parse RPC response into SearchResponse object.
        
        Handles the actual SyftBox response format for search:
        {
            "data": {
                "message": {
                    "body": {
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
                        "providerInfo": {...},  # camelCase
                        "cost": 0.1
                    }
                }
            }
        }
        
        Args:
            response_data: Raw response data from RPC call matching schema.py format
            original_query: The original search query
            
        Returns:
            Parsed SearchResponse object
        """
        
        try:
            # Extract the actual response body from SyftBox nested structure
            if "data" in response_data and "message" in response_data["data"]:
                message_data = response_data["data"]["message"]
                
                if "body" in message_data and isinstance(message_data["body"], dict):
                    # Extract the body and convert to schema.py format
                    body = message_data["body"]
                    return SearchResponse.from_dict(body, original_query)
            
            # If not nested format, try direct parsing
            return SearchResponse.from_dict(response_data, original_query)
                
        except Exception as e:
            logger.error(f"Failed to parse search response: {e}")
            logger.error(f"Response data: {response_data}")
            raise RPCError(f"Failed to parse search response: {e}")
    
    async def search_with_params(self, params: Dict[str, Any]) -> SearchResponse:
        """Search with explicit parameters dictionary.
        
        Args:
            params: Dictionary of parameters including 'query' and optional params
            
        Returns:
            Search response
        """
        # Validate required parameters
        if "message" not in params:
            raise ValidationError("'message' parameter is required")
        
        # Extract standard parameters (make copy to avoid mutating input)
        params = params.copy()
        message = params.pop("message")
        topK = params.pop("topK", 3)
        similarity_threshold = params.pop("similarity_threshold", None)
        
        # Build RPC payload with consistent authentication
        account_email = self.rpc_client.accounting_client.get_email()
        payload = {
            "user_email": account_email,
            "query": message,
            "options": {"limit": topK}
        }
        
        if similarity_threshold is not None:
            payload["options"]["similarityThreshold"] = similarity_threshold
        
        # Add any additional service-specific parameters
        for key, value in params.items():
            payload["options"][key] = value
        
        # Make RPC call
        response_data = await self.rpc_client.call_search(self.service_info, payload)
        return self._parse_rpc_search_response(response_data, message)
        
    def estimate_cost(self, query_count: int = 1, result_limit: int = 3) -> float:
        return CostEstimator.estimate_search_cost(self.service_info, query_count, result_limit)
        
    @property
    def pricing(self) -> float:
        """Get pricing for search service."""
        search_service = self.service_info.get_service_info(ServiceType.SEARCH)
        return search_service.pricing if search_service else 0.0
    
    @property
    def charge_type(self) -> str:
        """Get charge type for search service."""
        search_service = self.service_info.get_service_info(ServiceType.SEARCH)
        return search_service.charge_type.value if search_service else "per_request"