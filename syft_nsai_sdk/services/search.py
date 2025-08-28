"""
Search service client for SyftBox models
"""
import uuid
import logging
from typing import List, Optional, Dict, Any

from ..core.types import (
    ModelInfo,
    PricingChargeType, 
    SearchRequest, 
    SearchResponse, 
    SearchOptions, 
    DocumentResult, 
    ServiceType
)
from ..core.exceptions import ServiceNotSupportedError, RPCError, ValidationError, raise_service_not_supported
from ..networking.rpc_client import SyftBoxRPCClient

logger = logging.getLogger(__name__)

class SearchService:
    """Service client for document search models."""
    
    def __init__(self, model_info: ModelInfo, rpc_client: SyftBoxRPCClient):
        """Initialize search service.
        
        Args:
            model_info: Information about the model
            rpc_client: RPC client for making calls
            
        Raises:
            ServiceNotSupportedError: If model doesn't support search
        """
        self.model_info = model_info
        self.rpc_client = rpc_client
        
        # Validate that model supports search
        if not model_info.supports_service(ServiceType.SEARCH):
            raise_service_not_supported(model_info.name, "search", model_info)
    
    async def search_with_params(self, params: Dict[str, Any]) -> SearchResponse:
        """Search with explicit parameters dictionary.
        
        Args:
            params: Dictionary of parameters including 'query' and optional params
            
        Returns:
            Search response
        """
        # Validate required parameters
        if "query" not in params:
            raise ValidationError("'query' parameter is required")
        
        # Extract standard parameters (make copy to avoid mutating input)
        params = params.copy()
        query = params.pop("query")
        limit = params.pop("limit", 3)
        similarity_threshold = params.pop("similarity_threshold", None)
        
        # Build RPC payload with consistent authentication
        payload = {
            "userEmail": self.rpc_client._accounting_credentials.get('email', ''),
            "query": query,
            "options": {"limit": limit}
        }
        
        if similarity_threshold is not None:
            payload["options"]["similarityThreshold"] = similarity_threshold
        
        # Add any additional model-specific parameters
        for key, value in params.items():
            payload["options"][key] = value
        
        # Make RPC call
        response_data = await self.rpc_client.call_search(self.model_info, payload)
        return self._parse_rpc_response(response_data, query)
    
    def _parse_rpc_response(self, response_data: Dict[str, Any], original_query: str) -> SearchResponse:
        """Parse RPC response into SearchResponse object.
        
        Handles the actual SyftBox response format for search:
        {
        "request_id": "...",
        "data": {
            "message": {
            "body": {
                "results": [
                {"id": "...", "content": "...", "score": 0.95, "metadata": {...}},
                ...
                ],
                "cost": 0.1
            }
            }
        }
        }
        
        Args:
            response_data: Raw response data from RPC call
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
    
    @property
    def pricing(self) -> float:
        """Get pricing for search service."""
        search_service = self.model_info.get_service_info(ServiceType.SEARCH)
        return search_service.pricing if search_service else 0.0
    
    @property
    def charge_type(self) -> str:
        """Get charge type for search service."""
        search_service = self.model_info.get_service_info(ServiceType.SEARCH)
        return search_service.charge_type.value if search_service else "per_request"
    
    def estimate_cost(self, query_count: int = 1, result_limit: int = 3) -> float:
        """Estimate cost for search requests."""
        search_service_info = self.model_info.get_service_info(ServiceType.SEARCH)
        if not search_service_info:
            return 0.0
            
        if search_service_info.charge_type == PricingChargeType.PER_REQUEST:
            return search_service_info.pricing * query_count
        elif search_service_info.charge_type == PricingChargeType.PER_TOKEN:
            # Use actual per-token pricing from model info
            estimated_tokens = query_count * (20 + result_limit * 100)  # Still rough estimate
            return search_service_info.pricing * estimated_tokens
        else:
            return search_service_info.pricing


class BatchSearchService:
    """Helper class for batch search operations."""
    
    def __init__(self, search_service: SearchService):
        """Initialize batch search service.
        
        Args:
            search_service: Search service to use
        """
        self.search_service = search_service
    
    async def search_multiple_queries(self,
                                     queries: List[str],
                                     **kwargs) -> List[SearchResponse]:
        """Search multiple queries sequentially.
        
        Args:
            queries: List of search queries
            **kwargs: Additional search arguments
            
        Returns:
            List of search responses
        """
        responses = []
        
        for query in queries:
            try:
                response = await self.search_service.search(query, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch search failed for query '{query}': {e}")
                # Create empty response for failed query
                empty_response = SearchResponse(
                    id=str(uuid.uuid4()),
                    query=query,
                    results=[],
                    cost=0.0,
                    provider_info={"error": str(e)}
                )
                responses.append(empty_response)
        
        return responses
    
    async def search_with_fallback(self,
                                  query: str,
                                  fallback_services: List[SearchService],
                                  **kwargs) -> SearchResponse:
        """Search with fallback to other services if primary fails.
        
        Args:
            query: Search query
            fallback_services: List of fallback search services
            **kwargs: Additional search arguments
            
        Returns:
            Search response from first successful service
        """
        # Try primary service first
        try:
            return await self.search_service.search(query, **kwargs)
        except Exception as primary_error:
            logger.warning(f"Primary search failed: {primary_error}")
        
        # Try fallback services
        for i, fallback_service in enumerate(fallback_services):
            try:
                logger.info(f"Trying fallback service {i+1}/{len(fallback_services)}")
                response = await fallback_service.search(query, **kwargs)
                
                # Add fallback info to provider info
                if response.provider_info is None:
                    response.provider_info = {}
                response.provider_info["fallback_used"] = True
                response.provider_info["fallback_service"] = fallback_service.model_info.name
                
                return response
                
            except Exception as fallback_error:
                logger.warning(f"Fallback service {i+1} failed: {fallback_error}")
                continue
        
        # All services failed
        raise RPCError(f"All search services failed for query: {query}")
    
    def aggregate_results(self, 
                         responses: List[SearchResponse],
                         max_results: int = 10,
                         remove_duplicates: bool = True) -> SearchResponse:
        """Aggregate results from multiple search responses.
        
        Args:
            responses: List of search responses to aggregate
            max_results: Maximum number of results to return
            remove_duplicates: Whether to remove duplicate results
            
        Returns:
            Aggregated search response
        """
        if not responses:
            return SearchResponse(
                id=str(uuid.uuid4()),
                query="",
                results=[]
            )
        
        # Combine all results
        all_results = []
        total_cost = 0.0
        queries = []
        
        for response in responses:
            all_results.extend(response.results)
            total_cost += response.cost or 0.0
            queries.append(response.query)
        
        # Remove duplicates if requested
        if remove_duplicates:
            seen_content = set()
            unique_results = []
            
            for result in all_results:
                content_hash = hash(result.content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
            
            all_results = unique_results
        
        # Sort by score (descending) and limit results
        all_results.sort(key=lambda r: r.score, reverse=True)
        final_results = all_results[:max_results]
        
        return SearchResponse(
            id=str(uuid.uuid4()),
            query=" | ".join(queries),
            results=final_results,
            cost=total_cost,
            provider_info={
                "aggregated_from": len(responses),
                "total_results_found": len(all_results),
                "duplicates_removed": len(all_results) - len(unique_results) if remove_duplicates else 0
            }
        )