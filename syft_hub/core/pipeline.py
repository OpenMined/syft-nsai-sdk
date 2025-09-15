"""
Pipeline implementation for SyftBox NSAI SDK
Supports both inline and object-oriented RAG/FedRAG workflows
"""
import asyncio
import logging
from typing import List, Dict, Optional, Union, TYPE_CHECKING

from .types import ServiceType, ServiceSpec
from .exceptions import ValidationError, ServiceNotFoundError, ServiceNotSupportedError
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
    
    def __repr__(self) -> str:
        """Display pipeline configuration and usage examples."""
        try:
            from IPython.display import display, HTML
            # In notebook environment, show HTML widget
            self._show_html_widget()
            return ""  # Return empty string to avoid double output
        except ImportError:
            # Not in notebook - provide comprehensive text representation
            return self._get_text_representation()
    
    def _get_text_representation(self) -> str:
        """Get text representation of the pipeline."""
        # Determine pipeline status
        status = "Configured" if self.data_sources and self.synthesizer else "Incomplete"
        status_text = f" [{status}]"
        
        lines = [
            f"RAG Pipeline{status_text}",
            "",
            f"Data Sources:     {len(self.data_sources)} sources",
        ]
        
        # List data sources
        if self.data_sources:
            for i, source in enumerate(self.data_sources, 1):
                params_str = ""
                if source.params:
                    key_params = [f"{k}={v}" for k, v in list(source.params.items())[:2]]
                    params_str = f" ({', '.join(key_params)})"
                lines.append(f"  {i}. {source.name}{params_str}")
        else:
            lines.append("  None configured")
        
        lines.append("")
        
        # Synthesizer info
        synthesizer_info = f"Synthesizer:      {self.synthesizer.name if self.synthesizer else 'None configured'}"
        if self.synthesizer and self.synthesizer.params:
            key_params = [f"{k}={v}" for k, v in list(self.synthesizer.params.items())[:2]]
            synthesizer_info += f" ({', '.join(key_params)})"
        lines.append(synthesizer_info)
        
        lines.extend([
            f"Context Format:   {self.context_format}",
            "",
        ])
        
        # Cost estimation
        try:
            estimated_cost = self.estimate_cost()
            cost_str = f"${estimated_cost:.4f}" if estimated_cost > 0 else "Free"
            lines.append(f"Estimated Cost:   {cost_str} per execution")
        except:
            lines.append("Estimated Cost:   Unable to calculate")
        
        lines.extend([
            "",
            "Usage examples:",
        ])
        
        if status == "Configured":
            lines.extend([
                "  # Execute pipeline",
                "  result = pipeline.run([",
                "      {'role': 'user', 'content': 'Your question here'}",
                "  ])",
                "",
                "  # Execute asynchronously", 
                "  result = await pipeline.run_async([",
                "      {'role': 'user', 'content': 'Your question here'}",
                "  ])",
                "",
                "  # Access results",
                "  print(result.response.content)  # Synthesized response",
                "  print(result.search_results)    # Source documents", 
                "  print(f'Cost: ${result.cost}')  # Execution cost"
            ])
        else:
            lines.extend([
                "  # Add data sources first",
                "  pipeline.add_source('alice@example.com/docs')",
                "  pipeline.set_synthesizer('ai@openai.com/gpt-4')",
                "",
                "  # Or create configured pipeline directly",
                "  pipeline = client.pipeline(",
                "      data_sources=['alice@example.com/docs'],",
                "      synthesizer='ai@openai.com/gpt-4'",
                "  )"
            ])
        
        return "\n".join(lines)
    
    def _show_html_widget(self) -> None:
        """Show HTML widget in notebook environment."""
        try:
            from IPython.display import display, HTML
        except ImportError:
            return
        
        # Determine pipeline status and styling
        is_configured = bool(self.data_sources and self.synthesizer)
        status_text = "Configured" if is_configured else "Incomplete"
        status_class = "configured" if is_configured else "incomplete"
        
        # Build data sources HTML
        sources_html = ""
        if self.data_sources:
            for i, source in enumerate(self.data_sources, 1):
                params_display = ""
                if source.params:
                    params_list = [f"{k}: {v}" for k, v in source.params.items()]
                    params_display = f"<div class='pipeline-params'>{', '.join(params_list[:3])}</div>"
                
                sources_html += f"""
                <div class="pipeline-source">
                    <div class="pipeline-source-name">{i}. {source.name}</div>
                    {params_display}
                </div>
                """
        else:
            sources_html = '<div class="pipeline-empty">No data sources configured</div>'
        
        # Build synthesizer HTML
        synthesizer_html = ""
        if self.synthesizer:
            params_display = ""
            if self.synthesizer.params:
                params_list = [f"{k}: {v}" for k, v in self.synthesizer.params.items()]
                params_display = f"<div class='pipeline-params'>{', '.join(params_list[:3])}</div>"
            
            synthesizer_html = f"""
            <div class="pipeline-synthesizer">
                <div class="pipeline-synthesizer-name">{self.synthesizer.name}</div>
                {params_display}
            </div>
            """
        else:
            synthesizer_html = '<div class="pipeline-empty">No synthesizer configured</div>'
        
        # Cost estimation
        try:
            estimated_cost = self.estimate_cost()
            cost_display = f"${estimated_cost:.4f}" if estimated_cost > 0 else "Free"
        except:
            cost_display = "Unable to calculate"
        
        # Usage examples based on configuration
        if is_configured:
            usage_examples = """
            <div class="pipeline-command-code">
# Execute pipeline
result = pipeline.run([
    {'role': 'user', 'content': 'Your question here'}
])

# Access results  
print(result.response.content)  # Synthesized response
print(result.search_results)    # Source documents
print(f'Cost: ${result.cost}')  # Execution cost
            </div>
            """
        else:
            usage_examples = """
            <div class="pipeline-command-code">
# Add components
pipeline.add_source('alice@example.com/docs')
pipeline.set_synthesizer('ai@openai.com/gpt-4')

# Or create configured pipeline directly
pipeline = client.pipeline(
    data_sources=['alice@example.com/docs'],
    synthesizer='ai@openai.com/gpt-4'
)
            </div>
            """
        
        html = f'''
        <style>
            .pipeline-widget {{
                font-family: system-ui, -apple-system, sans-serif;
                padding: 16px 0;
                color: #333;
                line-height: 1.5;
                max-width: 900px;
            }}
            .pipeline-title {{
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .pipeline-status-badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 500;
            }}
            .pipeline-status-badge.configured {{
                background: #d4edda;
                color: #155724;
            }}
            .pipeline-status-badge.incomplete {{
                background: #fff3cd;
                color: #856404;
            }}
            .pipeline-status-line {{
                display: flex;
                align-items: center;
                margin: 6px 0;
                font-size: 13px;
            }}
            .pipeline-status-label {{
                color: #666;
                min-width: 140px;
                margin-right: 12px;
                font-weight: 500;
            }}
            .pipeline-status-value {{
                font-family: monospace;
                color: #333;
                font-size: 12px;
            }}
            .pipeline-source, .pipeline-synthesizer {{
                margin: 4px 0;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
                border-left: 3px solid #007bff;
            }}
            .pipeline-source-name, .pipeline-synthesizer-name {{
                font-weight: 500;
                color: #495057;
            }}
            .pipeline-params {{
                font-size: 11px;
                color: #6c757d;
                margin-top: 2px;
            }}
            .pipeline-empty {{
                color: #6c757d;
                font-style: italic;
                font-size: 12px;
            }}
            .pipeline-docs-section {{
                margin-top: 20px;
                padding: 16px;
                border-top: 1px solid #e0e0e0;
                font-size: 12px;
                color: #666;
            }}
            .pipeline-section-header {{
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 12px;
                color: #495057;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .pipeline-command-code {{
                font-family: Monaco, 'Courier New', monospace;
                background: #f5f5f5;
                padding: 16px;
                border-radius: 4px;
                color: #0066cc;
                margin: 8px 0;
                border-left: 3px solid #007bff;
                white-space: pre-wrap;
                line-height: 1.4;
                font-size: 12px;
            }}
        </style>
        <div class="pipeline-widget">
            <div class="pipeline-title">
                RAG Pipeline <span class="pipeline-status-badge {status_class}">{status_text}</span>
            </div>
            
            <div class="pipeline-status-line">
                <span class="pipeline-status-label">Data Sources:</span>
                <span class="pipeline-status-value">{len(self.data_sources)} configured</span>
            </div>
            <div style="margin-left: 140px; margin-bottom: 12px;">
                {sources_html}
            </div>
            
            <div class="pipeline-status-line">
                <span class="pipeline-status-label">Synthesizer:</span>
                <span class="pipeline-status-value">{"Configured" if self.synthesizer else "Not configured"}</span>
            </div>
            <div style="margin-left: 140px; margin-bottom: 12px;">
                {synthesizer_html}
            </div>
            
            <div class="pipeline-status-line">
                <span class="pipeline-status-label">Context Format:</span>
                <span class="pipeline-status-value">{self.context_format}</span>
            </div>
            
            <div class="pipeline-status-line">
                <span class="pipeline-status-label">Estimated Cost:</span>
                <span class="pipeline-status-value" style="color: {'#28a745' if 'Free' in cost_display else '#dc3545'}; font-weight: 600;">{cost_display} per execution</span>
            </div>
            
            <div class="pipeline-docs-section">
                <div class="pipeline-section-header">Usage Examples</div>
                {usage_examples}
            </div>
        </div>
        '''
        
        display(HTML(html))
    
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
                service = self.client.load_service(source_spec.name)
                if not service.supports_search:
                    raise ServiceNotSupportedError(service.name, "search", service._service_info)
            except ServiceNotFoundError:
                raise ValidationError(f"Data source service '{source_spec.name}' not found")
        
        # Validate synthesizer
        try:
            service = self.client.load_service(self.synthesizer.name)
            if not service.supports_chat:
                raise ServiceNotSupportedError(service.name, "chat", service._service_info)
        except ServiceNotFoundError:
            raise ValidationError(f"Synthesizer service '{self.synthesizer.name}' not found")
        
        return True
    
    def estimate_cost(self, message_count: int = 1) -> float:
        """Estimate total cost for pipeline execution"""
        
        # Prepare data sources for cost estimation
        data_sources = []
        for source_spec in self.data_sources:
            try:
                service = self.client.load_service(source_spec.name)
                data_sources.append((service._service_info, source_spec.params))
            except ServiceNotFoundError:
                logger.warning(f"Service '{source_spec.name}' not found during cost estimation")
                continue
        
        # Get synthesizer service
        synthesizer_service = None
        if self.synthesizer:
            try:
                service = self.client.load_service(self.synthesizer.name)
                synthesizer_service = service._service_info
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
    
    def run(self, messages: List[Dict[str, str]], continue_without_results: bool = False) -> PipelineResult:
        """Execute the pipeline synchronously
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            continue_without_results: If True, automatically continue synthesis even if no search results found.
                                    If False (default), prompts user when no results found.
        """
        from ..utils.async_utils import detect_async_context, run_async_in_thread
        
        if detect_async_context():
            # In Jupyter or other async context, use run_async_in_thread
            return run_async_in_thread(self.run_async(messages, continue_without_results))
        else:
            # In regular sync context, use asyncio.run
            return asyncio.run(self.run_async(messages, continue_without_results))
    
    async def run_async(self, messages: List[Dict[str, str]], continue_without_results: bool = False) -> PipelineResult:
        """Execute the pipeline asynchronously with parallel search execution
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            continue_without_results: If True, automatically continue synthesis even if no search results found.
                                    If False (default), prompts user when no results found.
        """
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
        
        # Print search progress header
        print(f"\n📊 Searching {len(self.data_sources)} data source(s)...")
        
        for i, result in enumerate(search_results_list):
            source_name = self.data_sources[i].name
            
            if isinstance(result, Exception):
                logger.warning(f"Search failed for source {source_name}: {result}")
                print(f"❌ {source_name}: Search failed - {result}")
                continue
            
            search_response, cost = result
            num_results = len(search_response.results)
            
            # Print summary for this source
            if num_results > 0:
                print(f"✅ {source_name}: Found {num_results} result(s)")
                # Show top result preview
                if search_response.results:
                    top_result = search_response.results[0]
                    preview = top_result.content[:100] + "..." if len(top_result.content) > 100 else top_result.content
            else:
                print(f"⚠️  {source_name}: No results found")
            
            all_search_results.extend(search_response.results)
            total_cost += cost
        
        if not all_search_results:
            if not continue_without_results:
                # Interactive prompt
                print("\n⚠️  No search results found from data sources")
                print("Options: Continue with predictions without sources or cancel")
                
                try:
                    response = input("Continue without search results? (y/n): ").lower().strip()
                    if response not in ['y', 'yes']:
                        print("Pipeline cancelled.")
                        raise ValidationError("Pipeline cancelled by user - no search results available")
                except (EOFError, KeyboardInterrupt):
                    print("\nPipeline cancelled.")
                    raise ValidationError("Pipeline cancelled by user - no search results available")
            
            print("Continuing with synthesis without search context...")
            logger.warning("All data source searches failed or returned empty results")
        
        # Remove duplicate results
        unique_results = self.client.remove_duplicate_results(all_search_results)
        
        # Print search summary
        print(f"📋 Search Summary: {len(unique_results)} result(s)")
        
        # Format search context for synthesizer
        context = self.client.format_search_context(unique_results, self.context_format)
        
        # Prepare messages with context
        enhanced_messages = self._prepare_enhanced_messages(messages, context)
        
        # Print synthesis start
        print(f"\n🤖 Synthesizing response with {self.synthesizer.name}...")
        
        # Execute synthesis
        synthesizer_cost, chat_response = await self._execute_synthesis(enhanced_messages)
        total_cost += synthesizer_cost
        
        # Print synthesis result
        if chat_response and chat_response.message:
            # Show preview of the response
            content = chat_response.message.content
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"✅ Response generated ({len(content)} chars)")
        else:
            print(f"⚠️  No response generated")
        
        # Print cost summary
        
        return PipelineResult(
            query=search_query,
            response=chat_response,
            search_results=unique_results,
            cost=total_cost
        )
    
    async def _execute_search(self, source_spec: ServiceSpec, query: str):
        """Execute search on a single data source"""
        try:
            # Get service info but use the service's own search method
            # This ensures the service uses its own properly initialized context
            service = self.client.load_service(source_spec.name)
            
            # Use the service's search_async method directly
            # This avoids creating a new SearchService with potentially mismatched event loop
            response = await service.search_async(
                message=query,
                **source_spec.params
            )

            # Estimate cost
            topK = source_spec.params.get('topK', len(response.results))
            cost = CostEstimator.estimate_search_cost(service._service_info, query_count=1, result_limit=topK)
            
            return response, cost
            
        except Exception as e:
            logger.error(f"Search failed for {source_spec.name}: {e}")
            raise
    
    async def _execute_synthesis(self, messages: List[Dict[str, str]]):
        """Execute synthesis with the enhanced messages"""
        try:
            # Load service and use its chat_async method directly
            service = self.client.load_service(self.synthesizer.name)
            
            # Debug: Log what we're sending
            logger.debug(f"Sending {len(messages)} messages to synthesizer")
            logger.debug(f"Synthesizer params: {self.synthesizer.params}")
            
            # Execute chat using the service's method
            response = await service.chat_async(
                messages=messages,
                **self.synthesizer.params
            )
            
            # Debug: Log what we received
            logger.debug(f"Received response type: {type(response)}")
            if response:
                logger.debug(f"Response has message: {hasattr(response, 'message')}")
                if hasattr(response, 'message') and response.message:
                    logger.debug(f"Message content length: {len(response.message.content) if response.message.content else 0}")
            
            # Estimate cost
            cost = CostEstimator.estimate_chat_cost(service._service_info, message_count=len(messages))
            
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