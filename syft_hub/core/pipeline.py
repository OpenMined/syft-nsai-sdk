"""
Pipeline implementation for SyftBox NSAI SDK
Supports both inline and object-oriented RAG/FedRAG workflows
"""
import asyncio
import logging
from typing import List, Dict, Optional, Union, TYPE_CHECKING

from .types import ServiceType, ServiceSpec
from .exceptions import ValidationError, ServiceNotFoundError, ServiceNotSupportedError
from ..services.chat import ChatService
from ..services.search import SearchService
from ..models.pipeline import PipelineResult
from ..utils.estimator import CostEstimator

if TYPE_CHECKING:
    from ..main import Client
    from .service import Service 

logger = logging.getLogger(__name__)

class Pipeline:
    """Pipeline for structured RAG/FedRAG workflows.
    
    Provides a streamlined way to combine multiple search services (data sources)
    with chat services (synthesizers) to create powerful RAG/FedRAG applications.
    """
    
    def __init__(
            self, 
            client: 'Client', 
            data_sources: Optional[List[Union[str, Dict, 'Service']]] = None,
            synthesizer: Optional[Union[str, Dict, 'Service']] = None,
            context_format: str = "simple"
        ):
        """Initialize the pipeline with data sources and synthesizer.
        
        Args:
            client: SyftBox client instance
            data_sources: List of search services for data retrieval. Each item can be:
                - str: Service name like "alice@example.com/docs" 
                - dict: Service with params like {"name": "service", "topK": 10}
                - Service: Loaded service object from client.load_service()
            synthesizer: Chat service for response generation. Can be:
                - str: Service name like "ai@openai.com/gpt-4"
                - dict: Service with params like {"name": "service", "temperature": 0.7}
                - Service: Loaded service object
            context_format: Format for injecting search context (default: "simple")
                - "simple": Clean format with ## headers for each source document
                - "frontend": Compact [filename] format matching web application
        """
        self.client = client
        self.data_sources: List[ServiceSpec] = []
        self.synthesizer: Optional[ServiceSpec] = None
        self.context_format = context_format
            
        # Handle inline initialization
        if data_sources:
            for source in data_sources:
                if isinstance(source, str):
                    self.data_sources.append(ServiceSpec(name=source, params={}))
                elif hasattr(source, 'full_name'):  # Service object
                    self.data_sources.append(ServiceSpec(name=source.full_name, params={}))
                elif isinstance(source, dict):
                    name = source.pop('name')
                    self.data_sources.append(ServiceSpec(name=name, params=source))
                else:
                    raise ValidationError(f"Invalid data source format: {source}. Expected str (service name), dict (service with params), or Service object.")

        if synthesizer:
            if isinstance(synthesizer, str):
                self.synthesizer = ServiceSpec(name=synthesizer, params={})
            elif hasattr(synthesizer, 'full_name'):  # Service object
                self.synthesizer = ServiceSpec(name=synthesizer.full_name, params={})
            elif isinstance(synthesizer, dict):
                name = synthesizer.pop('name')
                self.synthesizer = ServiceSpec(name=name, params=synthesizer)
            else:
                raise ValidationError(f"Invalid synthesizer format: {synthesizer}. Expected str (service name), dict (service with params), or Service object.")
    
    def add_source(self, service_name: str, **params) -> 'Pipeline':
        """Add a data source service with parameters"""
        self.data_sources.append(ServiceSpec(name=service_name, params=params))
        return self
    
    def set_synthesizer(self, service_name: str, **params) -> 'Pipeline':
        """Set the synthesizer service with parameters"""
        self.synthesizer = ServiceSpec(name=service_name, params=params)
        return self
    
    def validate(self) -> bool:
        """Check that all services exist, are reachable, and support required operations"""
        if not self.data_sources:
            raise ValidationError("No data sources configured")
        
        if not self.synthesizer:
            raise ValidationError("No synthesizer configured")
        
        # Validate data sources
        for source_spec in self.data_sources:
            try:
                service = self.client.get_service(source_spec.name)
                if not service.supports_service(ServiceType.SEARCH):
                    raise ServiceNotSupportedError(service.name, "search", service)
            except ServiceNotFoundError:
                raise ValidationError(f"Data source service '{source_spec.name}' not found")
        
        # Validate synthesizer
        try:
            service = self.client.get_service(self.synthesizer.name)
            if not service.supports_service(ServiceType.CHAT):
                raise ServiceNotSupportedError(service.name, "chat", service)
        except ServiceNotFoundError:
            raise ValidationError(f"Synthesizer service '{self.synthesizer.name}' not found")
        
        return True
    
    def estimate_cost(self, message_count: int = 1) -> float:
        """Estimate total cost for pipeline execution"""
        
        # Prepare data sources for cost estimation
        data_sources = []
        for source_spec in self.data_sources:
            try:
                service = self.client.get_service(source_spec.name)
                data_sources.append((service, source_spec.params))
            except ServiceNotFoundError:
                logger.warning(f"Service '{source_spec.name}' not found during cost estimation")
                continue
        
        # Get synthesizer service
        synthesizer_service = None
        if self.synthesizer:
            try:
                synthesizer_service = self.client.get_service(self.synthesizer.name)
            except ServiceNotFoundError:
                logger.warning(f"Synthesizer service '{self.synthesizer.name}' not found during cost estimation")
        
        if not data_sources or not synthesizer_service:
            return 0.0
        
        # Estimate cost
        return CostEstimator.estimate_pipeline_cost(
            data_sources=data_sources,
            synthesizer_service=synthesizer_service,
            message_count=message_count
        )
    
    def run(self, messages: List[Dict[str, str]]) -> PipelineResult:
        """Execute the pipeline synchronously"""
        return asyncio.run(self.run_async(messages))
    
    async def run_async(self, messages: List[Dict[str, str]]) -> PipelineResult:
        """Execute the pipeline asynchronously with parallel search execution"""
        # Validate pipeline first
        self.validate()
        
        # Extract search query from messages
        if not messages:
            raise ValidationError("No messages provided")
        
        # Use the last user message as the search query
        search_query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                search_query = msg.get("content")
                break
        
        if not search_query:
            raise ValidationError("No user message found for search query")
        
        # Execute searches in parallel
        search_tasks = []
        for source_spec in self.data_sources:
            task = self._execute_search(source_spec, search_query)
            search_tasks.append(task)
        
        # Wait for all searches to complete
        search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process search results and handle errors
        all_search_results = []
        total_cost = 0.0
        
        for i, result in enumerate(search_results_list):
            if isinstance(result, Exception):
                logger.warning(f"Search failed for source {self.data_sources[i].name}: {result}")
                continue
            
            search_response, cost = result
            all_search_results.extend(search_response.results)
            total_cost += cost
        
        if not all_search_results:
            raise ValidationError("All data source searches failed")
        
        # Remove duplicate results
        unique_results = self.client._remove_duplicate_results(all_search_results)
        
        # Format search context for synthesizer
        context = self.client._format_search_context(unique_results, self.context_format)
        
        # Prepare messages with context
        enhanced_messages = self._prepare_enhanced_messages(messages, context)
        
        # Execute synthesis
        synthesizer_cost, chat_response = await self._execute_synthesis(enhanced_messages)
        total_cost += synthesizer_cost
        
        return PipelineResult(
            response=chat_response,
            search_results=unique_results,
            cost=total_cost
        )
    
    async def _execute_search(self, source_spec: ServiceSpec, query: str):
        """Execute search on a single data source"""
        try:
            service = self.client.get_service(source_spec.name)
            search_service = SearchService(service, self.client.rpc_client)
            
            # Build search parameters
            search_params = {
                "message": query,
                **source_spec.params
            }
            
            # Execute search
            response = await search_service.search_with_params(search_params)

            # Estimate cost
            topK = source_spec.params.get('topK', len(response.results))
            cost = CostEstimator.estimate_search_cost(service, query_count=1, result_limit=topK)
            
            return response, cost
            
        except Exception as e:
            logger.error(f"Search failed for {source_spec.name}: {e}")
            raise
    
    async def _execute_synthesis(self, messages: List[Dict[str, str]]):
        """Execute synthesis with the enhanced messages"""
        try:
            service = self.client.get_service(self.synthesizer.name)
            chat_service = ChatService(service, self.client.rpc_client)
            
            # Build chat parameters
            chat_params = {
                "messages": messages,
                **self.synthesizer.params
            }
            
            # Execute chat
            response = await chat_service.chat_with_params(chat_params)
            
            # Estimate cost
            cost = CostEstimator.estimate_chat_cost(service, message_count=len(messages))
            
            return cost, response
            
        except Exception as e:
            logger.error(f"Synthesis failed for {self.synthesizer.name}: {e}")
            raise
    
    def _prepare_enhanced_messages(self, original_messages: List[Dict[str, str]], context: str) -> List[Dict[str, str]]:
        """Prepare messages with search context injected"""
        if not context.strip():
            return original_messages
        
        # Find the last user message and enhance it with context
        enhanced_messages = []
        context_injected = False
        
        for msg in original_messages:
            if msg.get("role") == "user" and not context_injected:
                # Inject context before the user's message
                enhanced_content = f"Context:\n{context}\n\nUser Question: {msg.get('content', '')}"
                enhanced_messages.append({
                    "role": "user",
                    "content": enhanced_content
                })
                context_injected = True
            else:
                enhanced_messages.append(msg)
        
        # If no user message found, add context as system message
        if not context_injected:
            enhanced_messages.insert(0, {
                "role": "system", 
                "content": f"Use this context to answer questions:\n{context}"
            })
        
        return enhanced_messages