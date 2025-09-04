"""
Search service client for SyftBox services
"""
import uuid
import logging
from typing import List, Optional, Dict, Any

from ..models.service_info import ServiceInfo
from ..core.types import (
    PricingChargeType,
    SearchOptions, 
    DocumentResult, 
    ServiceType
)
from ..core.exceptions import (
    ServiceNotSupportedError, 
    RPCError, 
    ValidationError, 
    raise_service_not_supported
)
from ..clients.rpc_client import SyftBoxRPCClient
from ..models.responses import SearchResponse
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
            raise_service_not_supported(service_info.name, "search", service_info)

    def _parse_rpc_response(self, response_data: Dict[str, Any], original_query: str) -> SearchResponse:
        """Parse RPC response into SearchResponse object.
        
        Handles the actual SyftBox response format for search:
        {
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
            "provider_info": {...},
            "cost": 0.1
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

    def _parse_rpc_response1(self, response_data: Dict[str, Any], original_query: str) -> SearchResponse:
        """Parse RPC response into SearchResponse object.
        
        Handles the actual SyftBox response format for search:
        {
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
            "provider_info": {...},
            "cost": 0.1
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
                    # This is the actual response content
                    body = message_data["body"]
                    
                    # Extract results
                    results = []
                    results_data = body.get("results", [])
                    
                    if isinstance(results_data, list):
                        for result_data in results_data:
                            if isinstance(result_data, dict):
                                result = DocumentResult(
                                    id=result_data.get("id", str(uuid.uuid4())),
                                    score=float(result_data.get("score", 0.0)),
                                    content=result_data.get("content", ""),
                                    metadata=result_data.get("metadata"),
                                    embedding=result_data.get("embedding")
                                )
                                results.append(result)
                            else:
                                # Handle string results
                                result = DocumentResult(
                                    id=str(uuid.uuid4()),
                                    score=1.0,
                                    content=str(result_data),
                                    metadata=None
                                )
                                results.append(result)
                    
                    return SearchResponse(
                        id=body.get("id", str(uuid.uuid4())),
                        query=body.get("query", original_query),
                        results=results,
                        cost=body.get("cost"),
                        provider_info=body.get("providerInfo")
                    )
            
            # Handle legacy/direct formats (backwards compatibility)
            if "results" in response_data:
                # Direct format
                results_data = response_data["results"]
                results = []
                
                for result_data in results_data:
                    result = DocumentResult(
                        id=result_data.get("id", str(uuid.uuid4())),
                        score=float(result_data.get("score", 0.0)),
                        content=result_data.get("content", ""),
                        metadata=result_data.get("metadata"),
                        embedding=result_data.get("embedding")
                    )
                    results.append(result)
                
                return SearchResponse(
                    id=response_data.get("id", str(uuid.uuid4())),
                    query=response_data.get("query", original_query),
                    results=results,
                    cost=response_data.get("cost"),
                    provider_info=response_data.get("providerInfo")
                )
            
            elif isinstance(response_data, list):
                # Array of results
                results = []
                
                for result_data in response_data:
                    if isinstance(result_data, dict):
                        result = DocumentResult(
                            id=result_data.get("id", str(uuid.uuid4())),
                            score=float(result_data.get("score", 1.0)),
                            content=result_data.get("content", str(result_data)),
                            metadata=result_data.get("metadata")
                        )
                        results.append(result)
                    else:
                        # Simple string result
                        result = DocumentResult(
                            id=str(uuid.uuid4()),
                            score=1.0,
                            content=str(result_data),
                            metadata=None
                        )
                        results.append(result)
                
                return SearchResponse(
                    id=str(uuid.uuid4()),
                    query=original_query,
                    results=results
                )
            
            else:
                # Last resort - treat as single result
                logger.warning(f"Unexpected search response format, using fallback parsing: {response_data}")
                
                result = DocumentResult(
                    id=str(uuid.uuid4()),
                    score=1.0,
                    content=str(response_data),
                    metadata=None
                )
                
                return SearchResponse(
                    id=str(uuid.uuid4()),
                    query=original_query,
                    results=[result]
                )
                
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
        logger.info(f"Search response data: {response_data}")
        return self._parse_rpc_response(response_data, message)
    
    def estimate_cost1(self, query_count: int = 1, result_limit: int = 3) -> float:
        """Estimate cost for search requests."""
        search_service_info = self.service_info.get_service_info(ServiceType.SEARCH)
        if not search_service_info:
            return 0.0
            
        if search_service_info.charge_type == PricingChargeType.PER_REQUEST:
            return search_service_info.pricing * query_count
        else:
            return search_service_info.pricing
        
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