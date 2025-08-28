"""
Main SyftBox NSAI SDK client
"""
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dotenv import load_dotenv

from .core.config import (
    SyftBoxConfig, 
    get_config, 
    is_syftbox_available, 
    get_installation_instructions
)
from .core.types import (
    ModelInfo, 
    ServiceType, 
    HealthStatus, 
    QualityPreference,
    ChatMessage, 
    ChatResponse, 
    SearchResponse, 
    FilterDict,
    DocumentResult,
)
from .core.exceptions import (
    AuthenticationError, 
    PaymentError, 
    SyftBoxNotFoundError, 
    ModelNotFoundError, 
    ServiceNotSupportedError, 
    ValidationError,
    raise_model_not_found, 
    raise_service_not_supported
)
from .discovery.scanner import ModelScanner, FastScanner
from .discovery.parser import MetadataParser
from .discovery.filters import ModelFilter, FilterCriteria, FilterBuilder
from .networking.rpc_client import SyftBoxRPCClient
from .services.chat import ChatService, ConversationManager
from .services.search import SearchService, BatchSearchService
from .services.health import check_model_health, batch_health_check, HealthMonitor
from .utils.formatting import format_models_table, format_model_details

from syft_accounting_sdk import UserClient, ServiceException

logger = logging.getLogger(__name__)
load_dotenv()

class SyftBoxClient:
    """Main client for discovering and using SyftBox AI models."""
    
    def __init__(self, 
                syftbox_config_path: Optional[Path] = None,
                user_email: str = "guest@syft.org",
                cache_server_url: Optional[str] = None,
                accounting_credentials: Optional[Dict[str, str]] = None,
                auto_setup_accounting: bool = True,
                auto_health_check_threshold: int = 10):
        """Initialize SyftBox client.
        
        Args:
            syftbox_config_path: Custom path to SyftBox config file
            user_email: Override user email for requests
            cache_server_url: Override cache server URL
            accounting_credentials: Manual accounting credentials
            auto_setup_accounting: Whether to prompt for accounting setup when needed
            auto_health_check_threshold: Max models for auto health checking
        """
        # Check SyftBox availability
        if not is_syftbox_available():
            raise SyftBoxNotFoundError(get_installation_instructions())
        
        # Load configuration
        self.config = get_config(syftbox_config_path)
        
        # Set up RPC client
        from_email = user_email
        server_url = cache_server_url or self.config.cache_server_url
        
        self.rpc_client = SyftBoxRPCClient(
            cache_server_url=server_url,
            from_email=from_email,
            accounting_service_url=self._get_accounting_service_url(),
            accounting_credentials=accounting_credentials or self._load_accounting_credentials()
        )
        
        # Set up model scanner
        self.scanner = FastScanner(self.config)
        self.parser = MetadataParser()
        
        # Configuration
        self.auto_health_check_threshold = auto_health_check_threshold
        self.auto_setup_accounting = auto_setup_accounting
        self._accounting_configured = None  # Cache accounting status
        
        # Optional health monitor
        self._health_monitor: Optional[HealthMonitor] = None

        # Try to load accounting config
        if accounting_credentials:
            self._configure_accounting(accounting_credentials)
        else:
            self._load_existing_accounting_config()

        # Load user email from config if not provided
        if from_email:
            logger.info(f"SyftBoxClient initialized for {from_email}")
        else:
            logger.info("SyftBoxClient initialized in guest mode (no user email provided)")
    
    async def close(self):
        """Close client and cleanup resources."""
        await self.rpc_client.close()
        if self._health_monitor:
            await self._health_monitor.stop_monitoring()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # Model Discovery Methods
    def discover_models(self,
                    service_type: Optional[str] = None,
                    owner: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    max_cost: Optional[float] = None,
                    free_only: bool = False,  # NEW PARAMETER
                    health_check: str = "auto",
                    include_disabled: bool = False,
                    **filter_kwargs) -> List[ModelInfo]:
        """Discover available models with filtering and optional health checking.
        
        Args:
            service_type: Filter by service type (chat, search)
            owner: Filter by owner email
            tags: Filter by tags (any match)
            max_cost: Maximum cost per request
            free_only: Only return free models (cost = 0)  # NEW PARAMETER DOCS
            health_check: Health checking mode ("auto", "always", "never")
            include_disabled: Include models with disabled services
            **filter_kwargs: Additional filter criteria
            
        Returns:
            List of discovered and filtered models
        """
        # Scan for metadata files
        metadata_paths = self.scanner.scan_with_cache()
        
        # Parse models from metadata
        models = []
        for metadata_path in metadata_paths:
            try:
                model_info = self.parser.parse_model_from_files(metadata_path)
                models.append(model_info)
            except Exception as e:
                logger.debug(f"Failed to parse {metadata_path}: {e}")
                continue

        # CONVERT STRING TO ENUM
        service_type_enum = None
        if service_type:
            try:
                service_type_enum = ServiceType(service_type.lower())
            except ValueError:
                logger.error(f"❌ Invalid service_type: {service_type}")
                return []
        
        # Apply filters
        filter_criteria = FilterCriteria(
            service_type=service_type_enum,
            owner=owner,
            has_any_tags=tags,
            max_cost=max_cost,
            free_only=free_only,  # NEW PARAMETER USAGE
            enabled_only=not include_disabled,
            **filter_kwargs
        )
        
        model_filter = ModelFilter(filter_criteria)
        filtered_models = model_filter.filter_models(models)
        
        # Determine if we should do health checking
        should_health_check = self._should_do_health_check(
            health_check, len(filtered_models)
        )
        
        if should_health_check:
            filtered_models = asyncio.run(self._add_health_status(filtered_models))
        
        logger.info(f"Discovered {len(filtered_models)} models (health_check={should_health_check})")
        return filtered_models
    
    def find_model(self, model_name: str, owner: Optional[str] = None) -> Optional[ModelInfo]:
        """Find a specific model by name.
        
        Args:
            model_name: Name of the model to find
            owner: Optional owner email to narrow search
            
        Returns:
            ModelInfo if found, None otherwise
        """
        models = self.discover_models(name=model_name, owner=owner, health_check="never")
        
        # Find exact match
        for model in models:
            if model.name == model_name:
                if owner is None or model.owner == owner:
                    return model
        
        return None
    
    def find_models_by_owner(self, owner_email: str) -> List[ModelInfo]:
        """Find all models by a specific owner.
        
        Args:
            owner_email: Email of the model owner
            
        Returns:
            List of models owned by the user
        """
        return self.discover_models(owner=owner_email, health_check="never")
    
    def find_models_by_tags(self, tags: List[str], match_all: bool = False) -> List[ModelInfo]:
        """Find models by tags.
        
        Args:
            tags: List of tags to match
            match_all: If True, model must have ALL tags; if False, ANY tag
            
        Returns:
            List of matching models
        """
        if match_all:
            return self.discover_models(has_all_tags=tags, health_check="never")
        else:
            return self.discover_models(has_any_tags=tags, health_check="never")
    
    # Display Methods 
    def list_models(self, 
                   service_type: Optional[ServiceType] = None,
                   health_check: str = "auto",
                   format: str = "table") -> str:
        """List available models in a user-friendly format.
        
        Args:
            service_type: Optional service type filter
            health_check: Health checking mode ("auto", "always", "never")
            format: Output format ("table", "json", "summary")
            
        Returns:
            Formatted string with model information
        """
        models = self.discover_models(
            service_type=service_type,
            health_check=health_check
        )
        
        if format == "table":
            return format_models_table(models)
        elif format == "json":
            import json
            model_dicts = [self._model_to_dict(model) for model in models]
            return json.dumps(model_dicts, indent=2)
        elif format == "summary":
            return self._format_models_summary(models)
        else:
            return [self._model_to_dict(model) for model in models]
    
    def show_model_details(self, model_name: str, owner: Optional[str] = None) -> str:
        """Show detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            owner: Optional owner to narrow search
            
        Returns:
            Formatted model details
        """
        model = self.find_model(model_name, owner)
        if not model:
            return f"Model '{model_name}' not found"
        
        return format_model_details(model)
    
    # Model Selection Methods
    def find_best_chat_model(self,
                            preference: QualityPreference = QualityPreference.BALANCED,
                            max_cost: Optional[float] = None,
                            tags: Optional[List[str]] = None,
                            **criteria) -> Optional[ModelInfo]:
        """Find the best chat model based on preferences.
        
        Args:
            preference: Quality preference (cheapest, balanced, premium, fastest)
            max_cost: Maximum cost per request
            tags: Preferred tags
            **criteria: Additional filter criteria
            
        Returns:
            Best matching chat model, or None if none found
        """
        models = self.discover_models(
            service_type=ServiceType.CHAT,
            max_cost=max_cost,
            tags=tags,
            health_check="auto",
            **criteria
        )
        
        return self._select_best_model(models, preference)
    
    def find_best_search_model(self,
                              preference: QualityPreference = QualityPreference.BALANCED,
                              max_cost: Optional[float] = None,
                              tags: Optional[List[str]] = None,
                              **criteria) -> Optional[ModelInfo]:
        """Find the best search model based on preferences.
        
        Args:
            preference: Quality preference (cheapest, balanced, premium, fastest)
            max_cost: Maximum cost per request
            tags: Preferred tags
            **criteria: Additional filter criteria
            
        Returns:
            Best matching search model, or None if none found
        """
        models = self.discover_models(
            service_type=ServiceType.SEARCH,
            max_cost=max_cost,
            tags=tags,
            health_check="auto",
            **criteria
        )
        
        return self._select_best_model(models, preference)

    # Service Usage Methods
    async def chat(self,
                   model_name: str,
                   prompt: str,
                   owner: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   auto_pay: bool = True,
                   **kwargs) -> ChatResponse:
        """Chat with a specific model.
        
        Args:
            model_name: Name of the model to use (REQUIRED)
            prompt: Message to send
            owner: Owner email (required if model name is ambiguous)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            auto_pay: Automatically handle payment for paid models
            **kwargs: Additional model-specific parameters
            
        Returns:
            Chat response from the specified model
            
        Example:
            response = await client.chat(
                model_name="public-tinnyllama",
                owner="irina@openmined.org", 
                prompt="Hello! Testing the API",
                temperature=0.7
            )
        """
        # Find the specific model
        model = self.find_model(model_name, owner)
        if not model:
            if owner:
                raise ModelNotFoundError(f"Model '{model_name}' not found for owner '{owner}'")
            else:
                # Show available models with same name
                similar_models = [m for m in self.discover_models() if m.name == model_name]
                if len(similar_models) > 1:
                    owners = [m.owner for m in similar_models]
                    raise ValidationError(
                        f"Multiple models named '{model_name}' found. "
                        f"Please specify owner. Available owners: {', '.join(owners)}"
                    )
                else:
                    raise ModelNotFoundError(f"Model '{model_name}' not found")
        
        # Check if model supports chat
        if not model.supports_service(ServiceType.CHAT):
            raise_service_not_supported(model.name, "chat", model)
            # raise ValidationError(f"Model '{model_name}' does not support chat service")
        
        # Build request parameters
        chat_params = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Remove None values
        chat_params = {k: v for k, v in chat_params.items() if v is not None}
        
        # Create service and make request
        from .services.chat import ChatService
        chat_service = ChatService(model, self.rpc_client)
        
        return await chat_service.send_message_with_params(chat_params)
    
    async def search(self,
                    model_name: str, 
                    query: str,
                    owner: Optional[str] = None,
                    limit: Optional[int] = None,
                    similarity_threshold: Optional[float] = None,
                    **kwargs) -> SearchResponse:
        """Search with a specific model.
        
        Args:
            model_name: Name of the model to use (REQUIRED)
            query: Search query
            owner: Owner email (required if model name is ambiguous)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            **kwargs: Additional model-specific parameters
            
        Returns:
            Search response from the specified model
            
        Example:
            results = await client.search(
                model_name="the-city",
                query="latest news",
                owner="speters@thecity.nyc"
            )
        """
        # Find the specific model
        model = self.find_model(model_name, owner)
        if not model:
            if owner:
                raise ModelNotFoundError(f"Model '{model_name}' not found for owner '{owner}'")
            else:
                raise ModelNotFoundError(f"Model '{model_name}' not found")
        
        # Check if model supports search
        if not model.supports_service(ServiceType.SEARCH):
            raise ValidationError(f"Model '{model_name}' does not support search service")
        
        # Build request parameters
        search_params = {
            "query": query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            **kwargs
        }
        
        # Remove None values
        search_params = {k: v for k, v in search_params.items() if v is not None}
        
        # Create service and make request
        from .services.search import SearchService
        search_service = SearchService(model, self.rpc_client)
        
        return await search_service.search_with_params(search_params)

    # Service Factory Methods
    async def chat_with_best(self, 
                            prompt: str,
                            max_cost: Optional[float] = None,
                            tags: Optional[List[str]] = None,
                            preference: str = "balanced",
                            **chat_params) -> ChatResponse:
        """Find best chat model and use it.
        
        Args:
            prompt: Message to send
            max_cost: Maximum cost willing to pay
            tags: Preferred model tags
            preference: Selection preference (cheapest, balanced, premium)
            **chat_params: Parameters for the chat request
            
        Returns:
            Chat response with model info included
            
        Example:
            # This is explicit about model selection happening
            response = await client.chat_with_best(
                prompt="Hello!",
                max_cost=0.10,
                tags=["opensource"],
                temperature=0.7
            )
            print(f"Used model: {response.model}")
        """
        # Find best model with explicit criteria
        best_model = self.find_best_chat_model(
            max_cost=max_cost,
            tags=tags,
            preference=preference
        )
        
        if not best_model:
            raise ModelNotFoundError("No suitable chat models found with specified criteria")
        
        print(f"🤖 Selected model: {best_model.name} by {best_model.owner}")
        
        # Use the selected model
        return await self.chat(
            model_name=best_model.name,
            owner=best_model.owner,
            prompt=prompt,
            **chat_params
        )

    async def search_with_best(self,
                              query: str,
                              max_cost: Optional[float] = None,
                              tags: Optional[List[str]] = None,
                              preference: str = "balanced",
                              **search_params) -> SearchResponse:
        """Find best search model and use it.
        
        Args:
            query: Search query
            max_cost: Maximum cost willing to pay
            tags: Preferred model tags
            preference: Selection preference
            **search_params: Parameters for the search request
            
        Returns:
            Search response with model info included
        """
        best_model = self.find_best_search_model(
            max_cost=max_cost,
            tags=tags,
            preference=preference
        )
        
        if not best_model:
            raise ModelNotFoundError("No suitable search models found with specified criteria")
        
        print(f"🔍 Selected model: {best_model.name} by {best_model.owner}")
        
        return await self.search(
            model_name=best_model.name,
            owner=best_model.owner,
            query=query,
            **search_params
        )
    
    # Model Parameters
    def get_model_parameters(self, model_name: str, owner: Optional[str] = None) -> Dict[str, Any]:
        """Get available parameters for a specific model."""
        model = self.find_model(model_name, owner)
        if not model:
            raise ModelNotFoundError(f"Model '{model_name}' not found")
        
        parameters = {}
        
        if model.endpoints and "components" in model.endpoints:
            schemas = model.endpoints["components"].get("schemas", {})
            
            # Extract chat parameters
            chat_request = schemas.get("ChatRequest", {})
            if chat_request:
                parameters["chat"] = self._extract_request_parameters(chat_request, schemas)
            
            # Extract search parameters  
            search_request = schemas.get("SearchRequest", {})
            if search_request:
                parameters["search"] = self._extract_request_parameters(search_request, schemas)
        
        return parameters

    def _extract_request_parameters(self, request_schema: Dict[str, Any], all_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from request schema."""
        parameters = {}
        properties = request_schema.get("properties", {})
        required_fields = set(request_schema.get("required", []))
        
        # Skip system fields
        skip_fields = {"userEmail", "model", "messages", "query", "transactionToken"}
        
        for field_name, field_info in properties.items():
            if field_name in skip_fields:
                continue
                
            param_info = {
                "required": field_name in required_fields,
                "description": field_info.get("description", "")
            }
            
            # Handle direct type
            if "type" in field_info:
                param_info["type"] = field_info["type"]
                self._add_constraints(param_info, field_info)
            
            # Handle $ref
            elif "$ref" in field_info:
                ref_name = field_info["$ref"].split("/")[-1]
                if ref_name in all_schemas:
                    nested = self._extract_schema_properties(all_schemas[ref_name])
                    param_info.update(nested)
            
            # Handle anyOf (optional references)
            elif "anyOf" in field_info:
                for option in field_info["anyOf"]:
                    if "$ref" in option:
                        ref_name = option["$ref"].split("/")[-1]
                        if ref_name in all_schemas:
                            nested = self._extract_schema_properties(all_schemas[ref_name])
                            param_info.update(nested)
                            break
                    elif "type" in option and option["type"] != "null":
                        param_info["type"] = option["type"]
                        self._add_constraints(param_info, option)
            
            parameters[field_name] = param_info
        
        return parameters

    def _extract_schema_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties from nested schema."""
        result = {"type": "object", "properties": {}}
        properties = schema.get("properties", {})
        
        for prop_name, prop_info in properties.items():
            if prop_name == "extensions":
                continue
                
            prop_data = {
                "description": prop_info.get("description", ""),
                "required": False
            }
            
            if "type" in prop_info:
                prop_data["type"] = prop_info["type"]
                self._add_constraints(prop_data, prop_info)
            elif "anyOf" in prop_info:
                for option in prop_info["anyOf"]:
                    if "type" in option and option["type"] != "null":
                        prop_data["type"] = option["type"]
                        self._add_constraints(prop_data, option)
                        break
            
            result["properties"][prop_name] = prop_data
        
        return result

    def _add_constraints(self, param_info: Dict[str, Any], field_info: Dict[str, Any]):
        """Add validation constraints to parameter info."""
        for constraint in ["minimum", "maximum", "enum", "format"]:
            if constraint in field_info:
                param_info[constraint] = field_info[constraint]

    def show_model_usage(self, model_name: str, owner: Optional[str] = None) -> str:
        """Show usage examples for a specific model.
        
        Args:
            model_name: Name of the model
            owner: Owner email if needed
            
        Returns:
            Formatted usage examples
        """
        model = self.find_model(model_name, owner)
        if not model:
            raise ModelNotFoundError(f"Model '{model_name}' not found")
        
        examples = []
        examples.append(f"# Usage examples for {model.name}")
        examples.append(f"# Owner: {model.owner}")
        examples.append("")
        
        if model.supports_service(ServiceType.CHAT):
            examples.extend([
                "# Basic chat",
                f'response = await client.chat(',
                f'    model_name="{model.name}",',
                f'    owner="{model.owner}",',
                f'    prompt="Hello! How are you?"',
                f')',
                "",
                "# Chat with parameters",
                f'response = await client.chat(',
                f'    model_name="{model.name}",',
                f'    owner="{model.owner}",',
                f'    prompt="Write a story",',
                f'    temperature=0.7,',
                f'    max_tokens=200',
                f')',
                ""
            ])
        
        if model.supports_service(ServiceType.SEARCH):
            examples.extend([
                "# Basic search",
                f'results = await client.search(',
                f'    model_name="{model.name}",',
                f'    owner="{model.owner}",',
                f'    query="machine learning"',
                f')',
                "",
                "# Search with parameters", 
                f'results = await client.search(',
                f'    model_name="{model.name}",',
                f'    owner="{model.owner}",',
                f'    query="latest AI research",',
                f'    limit=10,',
                f'    similarity_threshold=0.8',
                f')',
                ""
            ])
        
        # Add pricing info
        if model.min_pricing > 0:
            examples.append(f"# Cost: ${model.min_pricing} per request")
        else:
            examples.append("# Cost: Free")
        
        return "\n".join(examples)
    
    # RAG Workflow
    async def chat_with_search_context(self,
                                    search_models: Union[str, List[str], List[Dict[str, str]]],
                                    chat_model: str,
                                    prompt: str,
                                    chat_owner: Optional[str] = None,
                                    max_search_results: int = 3,
                                    search_similarity_threshold: Optional[float] = None,
                                    context_format: str = "frontend",  # "frontend" or "simple"
                                    **chat_kwargs) -> ChatResponse:
        """Perform search across multiple models then chat with context injection.
        
        This method replicates the frontend pattern where users can:
        1. Chat only (if no search_models provided)
        2. Search + Chat (if search_models provided - becomes RAG workflow)
        
        Args:
            search_models: Search models to query (OPTIONAL). Can be:
                        - None or [] for chat-only
                        - Single model name: "model-name"
                        - List of names: ["model1", "model2"] 
                        - List with owners: [{"name": "model1", "owner": "user@email.com"}]
            chat_model: Name of chat model for final response (REQUIRED)
            prompt: User's question/prompt
            chat_owner: Owner of chat model (if ambiguous)
            max_search_results: Max results per search model (ignored if no search)
            search_similarity_threshold: Minimum similarity score (ignored if no search)
            context_format: How to format context ("frontend" matches web app, "simple" is cleaner)
            **chat_kwargs: Additional parameters for chat request (temperature, max_tokens, etc.)
            
        Returns:
            ChatResponse with or without search context
            
        Example:
            # Chat only (like frontend with no data sources selected)
            response = await client.chat_with_search_context(
                search_models=[],  # No search - just chat
                chat_model="gpt-assistant", 
                prompt="What is machine learning?"
            )
            
            # RAG workflow (like frontend with data sources selected)
            response = await client.chat_with_search_context(
                search_models=["legal-docs", "company-policies"],
                chat_model="gpt-assistant", 
                prompt="What are our remote work policies?",
                max_search_results=5,
                temperature=0.7
            )
        """
        logger.info(f"🔄 Starting RAG workflow: search → chat")
        
        # Normalize search models to list of dicts
        search_model_specs = self._normalize_model_specs(search_models)
        
        if not search_model_specs:
            # No search models - do chat only (like frontend with no data sources)
            logger.info(f"💬 Chat-only mode: no search models specified")
            
            # Find and validate chat model
            chat_model_info = self.find_model(chat_model, chat_owner)
            if not chat_model_info:
                raise ModelNotFoundError(f"Chat model '{chat_model}' not found" + 
                                    (f" for owner '{chat_owner}'" if chat_owner else ""))
            
            if not chat_model_info.supports_service(ServiceType.CHAT):
                raise ValidationError(f"Model '{chat_model}' does not support chat service")
            
            # Direct chat without search context
            chat_service = ChatService(chat_model_info, self.rpc_client)
            
            # Build simple messages (just system + user)
            messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "You are a helpful AI assistant. Use your general knowledge to provide "
                        "comprehensive and helpful responses to user questions."
                    )
                ),
                ChatMessage(role="user", content=prompt)
            ]
            
            chat_response = await chat_service.send_conversation(
                messages=messages,
                **chat_kwargs
            )
            
            # Add metadata indicating this was chat-only
            if chat_response.provider_info is None:
                chat_response.provider_info = {}
            
            chat_response.provider_info.update({
                "rag_workflow": False,
                "chat_only": True,
                "search_models_used": [],
                "search_results_count": 0,
                "context_injected": False
            })
            
            logger.info(f"✅ Chat-only completed - Cost: ${chat_response.cost:.4f}")
            return chat_response
        
        # Step 1: Search across specified models (RAG mode)
        logger.info(f"🔍 RAG mode: searching {len(search_model_specs)} models for context")
        search_responses = []
        
        for model_spec in search_model_specs:
            model_name = model_spec["name"]
            owner = model_spec.get("owner")
            
            try:
                search_response = await self.search(
                    model_name=model_name,
                    query=prompt,  # Use the prompt as search query
                    owner=owner,
                    limit=max_search_results,
                    similarity_threshold=search_similarity_threshold
                )
                search_responses.append(search_response)
                logger.debug(f"✅ Search completed: {model_name} returned {len(search_response.results)} results")
                
            except Exception as e:
                logger.warning(f"⚠️ Search failed for {model_name}: {e}")
                # Continue with other models - graceful degradation
                continue
        
        # Step 2: Aggregate and format search results
        all_results = []
        total_search_cost = 0.0
        
        for response in search_responses:
            all_results.extend(response.results)
            total_search_cost += response.cost or 0.0
        
        logger.info(f"📊 Aggregated {len(all_results)} total results from {len(search_responses)} successful searches")
        
        # Step 3: Format context for chat injection
        if all_results:
            context_message = self._format_search_context(all_results, context_format)
            logger.debug(f"📝 Formatted context: {len(context_message)} characters")
        else:
            context_message = None
            logger.info("⚠️ No search results found - proceeding with chat only")
        
        # Step 4: Build enhanced message sequence
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a helpful AI assistant that can answer questions using both provided sources "
                    "and your general knowledge.\n\n"
                    "When sources are provided, use them as your primary information and supplement with "
                    "your general knowledge when helpful. If you don't know the answer from the sources, "
                    "you can still use your general knowledge to provide a helpful response.\n\n"
                    "When no sources are provided, rely on your general knowledge to answer the question comprehensively."
                )
            )
        ]
        
        # Add context if we have search results
        if context_message:
            messages.append(
                ChatMessage(
                    role="system", 
                    content=f"Here is relevant source context to help answer the user's question:\n\n{context_message}"
                )
            )
        
        # Add user prompt
        messages.append(ChatMessage(role="user", content=prompt))
        
        # Step 5: Send to chat model
        logger.info(f"💬 Sending enhanced prompt to chat model: {chat_model}")
        
        # Find and validate chat model
        chat_model_info = self.find_model(chat_model, chat_owner)
        if not chat_model_info:
            raise ModelNotFoundError(f"Chat model '{chat_model}' not found" + 
                                (f" for owner '{chat_owner}'" if chat_owner else ""))
        
        if not chat_model_info.supports_service(ServiceType.CHAT):
            raise ValidationError(f"Model '{chat_model}' does not support chat service")
        
        # Create chat service and send enhanced conversation
        chat_service = ChatService(chat_model_info, self.rpc_client)
        
        try:
            chat_response = await chat_service.send_conversation(
                messages=messages,
                **chat_kwargs
            )
            
            # Enhance response with RAG metadata
            if chat_response.provider_info is None:
                chat_response.provider_info = {}
            
            chat_response.provider_info.update({
                "rag_workflow": True,
                "search_models_used": [spec["name"] for spec in search_model_specs],
                "search_results_count": len(all_results),
                "total_search_cost": total_search_cost,
                "context_injected": context_message is not None
            })
            
            # Add combined cost
            chat_response.cost = (chat_response.cost or 0.0) + total_search_cost
            
            logger.info(f"✅ RAG workflow completed - Total cost: ${chat_response.cost:.4f}")
            return chat_response
            
        except Exception as e:
            logger.error(f"❌ Chat request failed in RAG workflow: {e}")
            raise

    async def search_multiple_models(self,
                                    model_names: Union[List[str], List[Dict[str, str]]],
                                    query: str,
                                    limit_per_model: int = 3,
                                    total_limit: Optional[int] = None,
                                    similarity_threshold: Optional[float] = None,
                                    remove_duplicates: bool = True,
                                    sort_by_score: bool = True,
                                    **search_kwargs) -> SearchResponse:
        """Search across multiple models and aggregate results.
        
        Args:
            model_names: List of model names or model specs with owners
            query: Search query
            limit_per_model: Max results per individual model
            total_limit: Max results in final aggregated response (None = no limit)
            similarity_threshold: Minimum similarity score
            remove_duplicates: Remove duplicate content based on content hash
            sort_by_score: Sort final results by similarity score (descending)
            **search_kwargs: Additional parameters for search requests
            
        Returns:
            SearchResponse with aggregated results from all models
            
        Example:
            # Search multiple data sources
            results = await client.search_multiple_models(
                model_names=["legal-docs", "company-wiki", "slack-archive"],
                query="vacation policy changes",
                limit_per_model=5,
                total_limit=10
            )
        """
        logger.info(f"🔍 Multi-model search across {len(model_names)} models")
        
        # Normalize model specs
        model_specs = self._normalize_model_specs(model_names)
        
        all_results = []
        total_cost = 0.0
        successful_models = []
        failed_models = []
        
        # Search each model
        for model_spec in model_specs:
            model_name = model_spec["name"]
            owner = model_spec.get("owner")
            
            try:
                response = await self.search(
                    model_name=model_name,
                    query=query,
                    owner=owner,
                    limit=limit_per_model,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
                
                all_results.extend(response.results)
                total_cost += response.cost or 0.0
                successful_models.append(model_name)
                
                logger.debug(f"✅ {model_name}: {len(response.results)} results")
                
            except Exception as e:
                logger.warning(f"⚠️ Search failed for {model_name}: {e}")
                failed_models.append({"model": model_name, "error": str(e)})
                continue
        
        # Remove duplicates if requested
        if remove_duplicates:
            all_results = self._remove_duplicate_results(all_results)
            logger.debug(f"🔄 Deduplication: {len(all_results)} unique results remain")
        
        # Sort by score if requested
        if sort_by_score:
            all_results.sort(key=lambda r: r.score, reverse=True)
        
        # Apply total limit
        if total_limit and len(all_results) > total_limit:
            all_results = all_results[:total_limit]
            logger.debug(f"✂️ Limited to top {total_limit} results")
        
        # Create aggregated response
        aggregated_response = SearchResponse(
            id=f"multi-search-{hash(query) % 10000}",
            query=query,
            results=all_results,
            cost=total_cost,
            provider_info={
                "multi_model_search": True,
                "successful_models": successful_models,
                "failed_models": failed_models,
                "total_models_searched": len(successful_models),
                "deduplication_applied": remove_duplicates,
                "sorted_by_score": sort_by_score
            }
        )
        
        logger.info(f"📊 Multi-search completed: {len(all_results)} results from {len(successful_models)}/{len(model_specs)} models")
        
        return aggregated_response
    
    async def search_then_chat(self,
                            search_model: str,
                            chat_model: str,
                            prompt: str,
                            search_owner: Optional[str] = None,
                            chat_owner: Optional[str] = None,
                            **kwargs) -> ChatResponse:
        """Simplified single-model search then chat workflow.
        
        Args:
            search_model: Name of model to search
            chat_model: Name of model to chat with
            prompt: User prompt (used for both search and chat)
            search_owner: Owner of search model
            chat_owner: Owner of chat model
            **kwargs: Additional parameters for both search and chat
            
        Returns:
            ChatResponse with search context
            
        Example:
            response = await client.search_then_chat(
                search_model="company-docs",
                chat_model="assistant-gpt",
                prompt="How do I submit expenses?"
            )
        """
        return await self.chat_with_search_context(
            search_models=[{"name": search_model, "owner": search_owner} if search_owner else search_model],
            chat_model=chat_model,
            prompt=prompt,
            chat_owner=chat_owner,
            **kwargs
        )

    # Private methods to support RAG coordination
    def _normalize_model_specs(self, models: Union[str, List[str], List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Normalize various model specification formats to list of dicts.
        
        Args:
            models: Models in various formats
            
        Returns:
            List of model specs with 'name' and optional 'owner' keys
        """
        if isinstance(models, str):
            # Single model name
            return [{"name": models}]
        
        elif isinstance(models, list):
            normalized = []
            for model in models:
                if isinstance(model, str):
                    # List of model names
                    normalized.append({"name": model})
                elif isinstance(model, dict):
                    # List of model specs
                    if "name" not in model:
                        raise ValidationError(f"Model spec missing 'name' key: {model}")
                    normalized.append(model)
                else:
                    raise ValidationError(f"Invalid model specification: {model}")
            return normalized
        
        else:
            raise ValidationError(f"Invalid models format: {type(models)}")

    def _format_search_context(self, results: List[DocumentResult], format_type: str = "frontend") -> str:
        """Format search results as context for chat injection.
        
        Args:
            results: Search results to format
            format_type: "frontend" (matches web app) or "simple"
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        if format_type == "frontend":
            # Match the exact frontend pattern: [filename]\nContent
            formatted_parts = []
            for result in results:
                filename = result.metadata.get("filename", "unknown") if result.metadata else "unknown"
                formatted_parts.append(f"[{filename}]\n{result.content}")
            
            return "\n\n".join(formatted_parts)
        
        elif format_type == "simple":
            # Cleaner format for direct SDK usage
            formatted_parts = []
            for i, result in enumerate(results, 1):
                source = result.metadata.get("filename", f"Source {i}") if result.metadata else f"Source {i}"
                formatted_parts.append(f"## {source}\n{result.content}")
            
            return "\n\n".join(formatted_parts)
        
        else:
            raise ValidationError(f"Unknown context format: {format_type}")

    def _remove_duplicate_results(self, results: List[DocumentResult]) -> List[DocumentResult]:
        """Remove duplicate results based on content similarity.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated list of results
        """
        if not results:
            return results
        
        # Simple deduplication based on content hash
        seen_hashes = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content.strip().lower())
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(result)
        
        return unique_results

    def get_rag_cost_estimate(self,
                            search_models: Union[str, List[str], List[Dict[str, str]]],
                            chat_model: str,
                            chat_owner: Optional[str] = None) -> Dict[str, Any]:
        """Get cost estimate for RAG workflow before execution.
        
        Provides transparent cost breakdown so users can make informed decisions
        about which models to use (matching the frontend's cost preview).
        
        Args:
            search_models: Search models to estimate
            chat_model: Chat model to estimate
            chat_owner: Owner of chat model
            
        Returns:
            Dictionary with cost breakdown and model details
            
        Example:
            # Preview costs before RAG workflow
            estimate = client.get_rag_cost_estimate(
                search_models=["docs-1", "docs-2"],
                chat_model="premium-chat"
            )
            print(f"Total cost: ${estimate['total_cost']}")
            print(f"Models: {estimate['model_summary']}")
        """
        search_model_specs = self._normalize_model_specs(search_models)
        
        breakdown = {
            "search_models": [],
            "chat_model": None,
            "search_cost": 0.0,
            "chat_cost": 0.0, 
            "total_cost": 0.0,
            "model_summary": {
                "search_count": len(search_model_specs),
                "search_names": [spec["name"] for spec in search_model_specs],
                "chat_name": chat_model
            }
        }
        
        # Get search model costs
        for spec in search_model_specs:
            model = self.find_model(spec["name"], spec.get("owner"))
            if model:
                search_service = model.get_service_info(ServiceType.SEARCH)
                if search_service:
                    model_cost = search_service.pricing
                    breakdown["search_cost"] += model_cost
                    breakdown["search_models"].append({
                        "name": model.name,
                        "owner": model.owner,
                        "cost": model_cost,
                        "charge_type": search_service.charge_type.value
                    })
        
        # Get chat model cost
        chat_model_info = self.find_model(chat_model, chat_owner)
        if chat_model_info:
            chat_service = chat_model_info.get_service_info(ServiceType.CHAT)
            if chat_service:
                breakdown["chat_cost"] = chat_service.pricing
                breakdown["chat_model"] = {
                    "name": chat_model_info.name,
                    "owner": chat_model_info.owner,
                    "cost": chat_service.pricing,
                    "charge_type": chat_service.charge_type.value
                }
        
        breakdown["total_cost"] = breakdown["search_cost"] + breakdown["chat_cost"]
        
        return breakdown

    def preview_rag_workflow(self,
                            search_models: Union[str, List[str], List[Dict[str, str]]],
                            chat_model: str,
                            chat_owner: Optional[str] = None) -> str:
        """Preview RAG workflow with model details and costs.
        
        Provides a human-readable preview of the RAG workflow showing exactly
        which models will be used and their costs (transparency like frontend).
        
        Args:
            search_models: Search models for the workflow
            chat_model: Chat model for the workflow  
            chat_owner: Owner of chat model
            
        Returns:
            Formatted preview string
            
        Example:
            preview = client.preview_rag_workflow(
                search_models=["legal-docs", "hr-policies"],
                chat_model="gpt-assistant"
            )
            print(preview)
        """
        estimate = self.get_rag_cost_estimate(search_models, chat_model, chat_owner)
        
        lines = [
            "📋 RAG Workflow Preview",
            "=" * 30,
            "",
            f"🔍 Search Phase ({len(estimate['search_models'])} models):"
        ]
        
        if estimate["search_models"]:
            for model in estimate["search_models"]:
                cost_str = f"${model['cost']:.4f}" if model['cost'] > 0 else "Free"
                lines.append(f"  • {model['name']} by {model['owner']} - {cost_str}")
            lines.append(f"  Subtotal: ${estimate['search_cost']:.4f}")
        else:
            lines.append("  • No valid search models found")
        
        lines.extend([
            "",
            "💬 Chat Phase:"
        ])
        
        if estimate["chat_model"]:
            chat = estimate["chat_model"]
            cost_str = f"${chat['cost']:.4f}" if chat['cost'] > 0 else "Free"
            lines.append(f"  • {chat['name']} by {chat['owner']} - {cost_str}")
        else:
            lines.append("  • Chat model not found")
        
        lines.extend([
            "",
            f"💰 Total Estimated Cost: ${estimate['total_cost']:.4f}",
            "",
            "To execute: client.chat_with_search_context(...)"
        ])
        
        return "\n".join(lines)
    
    def get_chat_service(self, model_name: str) -> ChatService:
        """Get a chat service for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ChatService instance
        """
        model = self.find_model(model_name)
        if not model:
            raise_model_not_found(model_name)
        
        return ChatService(model, self.rpc_client)
    
    def get_search_service(self, model_name: str) -> SearchService:
        """Get a search service for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            SearchService instance
        """
        model = self.find_model(model_name)
        if not model:
            raise_model_not_found(model_name)
        
        return SearchService(model, self.rpc_client)
    
    def create_conversation(self, model_name: str, owner: Optional[str] = None) -> ConversationManager:
        """Create a conversation manager for a chat model.
        
        Args:
            model_name: Name of the chat model
            owner: Owner email (required if model name is ambiguous)
            
        Returns:
            ConversationManager instance
        """
        # Find and validate model
        model = self.find_model(model_name, owner)
        if not model:
            if owner:
                raise ModelNotFoundError(f"Model '{model_name}' not found for owner '{owner}'")
            else:
                raise ModelNotFoundError(f"Model '{model_name}' not found")
        
        if not model.supports_service(ServiceType.CHAT):
            raise ValidationError(f"Model '{model_name}' does not support chat service")
        
        # Create chat service and conversation manager
        chat_service = ChatService(model, self.rpc_client)
        return ConversationManager(chat_service)
    
    # Health Monitoring Methods
    async def check_model_health(self, model_name: str, timeout: float = 2.0) -> HealthStatus:
        """Check health of a specific model.
        
        Args:
            model_name: Name of the model to check
            timeout: Timeout for health check
            
        Returns:
            Health status of the model
        """
        model = self.find_model(model_name)
        if not model:
            raise_model_not_found(model_name)
        
        return await check_model_health(model, self.rpc_client, timeout)
    
    async def check_all_models_health(self, 
                                     service_type: Optional[ServiceType] = None,
                                     timeout: float = 2.0) -> Dict[str, HealthStatus]:
        """Check health of all discovered models.
        
        Args:
            service_type: Optional service type filter
            timeout: Timeout per health check
            
        Returns:
            Dictionary mapping model names to health status
        """
        models = self.discover_models(service_type=service_type, health_check="never")
        return await batch_health_check(models, self.rpc_client, timeout)
    
    def start_health_monitoring(self, 
                               models: Optional[List[str]] = None,
                               check_interval: float = 30.0) -> HealthMonitor:
        """Start continuous health monitoring.
        
        Args:
            models: Optional list of model names to monitor (default: all chat/search models)
            check_interval: Seconds between health checks
            
        Returns:
            HealthMonitor instance
        """
        if self._health_monitor:
            logger.warning("Health monitoring already running")
            return self._health_monitor
        
        self._health_monitor = HealthMonitor(self.rpc_client, check_interval)
        
        # Add models to monitor
        if models:
            for model_name in models:
                model = self.find_model(model_name)
                if model:
                    self._health_monitor.add_model(model)
        else:
            # Monitor all enabled chat/search models
            all_models = self.discover_models(health_check="never")
            for model in all_models:
                if model.supports_service(ServiceType.CHAT) or model.supports_service(ServiceType.SEARCH):
                    self._health_monitor.add_model(model)
        
        # Start monitoring
        asyncio.create_task(self._health_monitor.start_monitoring())
        
        return self._health_monitor
    
    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        if self._health_monitor:
            await self._health_monitor.stop_monitoring()
            self._health_monitor = None
    
    # Accounting Integration Methods
    def _get_accounting_service_url(self) -> Optional[str]:
        """Get accounting service URL from configuration."""
        # This could come from SyftBox config or environment variable
        return os.getenv('SYFTBOX_ACCOUNTING_URL')
    
    def _load_accounting_credentials(self) -> Optional[Dict[str, str]]:
        """Load accounting credentials from secure storage."""
        # In practice, this might load from keyring, config file, or prompt user
        email = os.getenv("SYFTBOX_ACCOUNTING_EMAIL")
        password = os.getenv("SYFTBOX_ACCOUNTING_PASSWORD") 
        
        if email and password:
            return {"email": email, "password": password}
        return None
    
    def _load_existing_accounting_config(self):
        """Try to load accounting config from various sources."""
        import json
        import os
        
        # Try environment variables
        email = os.getenv("SYFTBOX_ACCOUNTING_EMAIL")
        password = os.getenv("SYFTBOX_ACCOUNTING_PASSWORD")
        service_url = os.getenv('SYFTBOX_ACCOUNTING_URL')

        if email and password:
            self._configure_accounting({
                "email": email,
                "password": password,
                "service_url": service_url
            })
            return
        
        # Try separate accounting config file
        accounting_config_path = Path.home() / ".syftbox" / "accounting.json"
        if accounting_config_path.exists():
            try:
                with open(accounting_config_path, 'r') as f:
                    config = json.load(f)
                
                if "email" in config and "password" in config:
                    self._configure_accounting({
                        "email": config["email"],
                        "password": config["password"],
                        "service_url": config.get("service_url", "")
                    })
                    return
            except Exception as e:
                logger.debug(f"Could not read accounting config file: {e}")
        
        logger.info("No existing accounting configuration found")
        self._accounting_configured = False
    
    def _configure_accounting(self, config: Dict[str, str]):
        """Configure accounting with provided credentials."""
        # Store credentials in RPC client
        self.rpc_client.configure_accounting(
            service_url=config["service_url"],
            email=config["email"],
            password=config["password"]
        )
        self._accounting_configured = True
        logger.info(f"Accounting configured for {config['email']}")
    
    def is_accounting_configured(self) -> bool:
        """Check if accounting is properly configured."""
        if self._accounting_configured is not None:
            return self._accounting_configured
        
        try:
            # Test accounting client
            return self.rpc_client.has_accounting_client()
        except Exception:
            self._accounting_configured = False
            return False
    
    async def setup_accounting(self, email: str, password: str, service_url: Optional[str] = None, organization: Optional[str] = None):
        """Setup accounting credentials.
        
        Args:
            email: Accounting service email
            password: Accounting service password  
            organization: Optional organization
        """
        """Simplified setup with better error handling."""

        credentials = {"email": email, "password": password}
        if organization:
            credentials["organization"] = organization

        # Get service URL from environment if not provided
        if service_url is None:
            # service_url = service_url
            service_url = self._get_accounting_service_url()
        
        # Validate service URL is available
        if not service_url:
            raise ValueError(
                "Accounting service URL is required. Please either:\n"
                "1. Set SYFTBOX_ACCOUNTING_URL in your .env file, or\n"
                "2. Pass service_url parameter to this method\n"
                "Example: await client.setup_accounting(email, password, 'https://your-service.com')"
            )
        
        logger.info(f"Using syftbox accounting service URL: {service_url}")
        
        try:
            # Create client
            user_client = UserClient(url=service_url, email=email, password=password)
            
            # Make a raw request to see what we get
            response = user_client._session.get(f"{user_client.url}/user/my-info")
            # response = user_client.get_user_info()
            
            # Check response
            if response.status_code == 401:
                raise AuthenticationError("Invalid email or password")
            elif response.status_code == 404:
                raise AuthenticationError("Service not found - check the URL")
            elif not response.ok:
                raise AuthenticationError(f"Service error: HTTP {response.status_code}")
            
            # Check content type
            if 'json' not in response.headers.get('content-type', '').lower():
                raise AuthenticationError("Service returned non-JSON response")
            
            # Try to parse JSON
            try:
                data = response.json()
                if 'user' not in data:
                    raise AuthenticationError("Invalid response format")
            except ValueError:
                raise AuthenticationError("Service returned invalid JSON")
            
            # If we get here, everything looks good
            self._configure_accounting({
                "email": email,
                "password": password,
                "service_url": service_url
            })

            # Store credentials
            self.rpc_client._accounting_credentials = credentials
            self.rpc_client._accounting_client = None  # Reset to recreate with new creds
            
            await self._save_accounting_config(service_url, email, password)
            logger.info("Accounting setup successful")
            
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Setup failed: {e}")
    
    async def _save_accounting_config(self, service_url: str, email: str, password: str):
        """Save accounting config to file."""
        import json
        import os
        from datetime import datetime
        
        try:
            config_dir = Path.home() / ".syftbox"
            config_dir.mkdir(exist_ok=True)
            
            accounting_config_path = config_dir / "accounting.json"
            config = {
                "service_url": service_url,
                "email": email,
                "password": password,
                "created_at": datetime.now().isoformat()
            }
            
            with open(accounting_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(accounting_config_path, 0o600)
            
        except Exception as e:
            logger.warning(f"Could not save accounting config: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balance."""
        try:
            if not self.is_accounting_configured():
                return {"error": "Accounting not configured"}
            
            balance = await self.rpc_client.get_account_balance()
            email = self.rpc_client.get_accounting_email()
            
            return {
                "email": email,
                "balance": balance,
                "currency": "USD"
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
    
    def show_accounting_status(self) -> str:
        """Show current accounting configuration status."""
        if not self.is_accounting_configured():
            return (
                "❌ Accounting not configured\n"
                "   Use client.setup_accounting() to configure payment services\n"
                "   Currently limited to free models only"
            )
        
        try:
            # Get account info
            import asyncio
            account_info = asyncio.run(self.get_account_info())
            
            if "error" in account_info:
                return (
                    f"⚠️  Accounting configured but connection failed\n"
                    f"   Error: {account_info['error']}\n"
                    f"   May need to reconfigure credentials"
                )
            
            return (
                f"✅ Accounting configured\n"
                f"   Email: {account_info['email']}\n" 
                f"   Balance: ${account_info['balance']}\n"
                f"   Can use both free and paid models"
            )
        except Exception as e:
            return (
                f"⚠️  Accounting configured but connection failed\n"
                f"   Error: {e}\n"
                f"   May need to reconfigure credentials"
            )
    
    async def _ensure_payment_setup(self, model: ModelInfo) -> Optional[str]:
        """Ensure payment is set up for a paid model.
        
        Args:
            model: Model that requires payment
            
        Returns:
            Transaction token if payment required, None if free
        """
        # Check if model requires payment
        service_info = None
        if model.supports_service(ServiceType.CHAT):
            service_info = model.get_service_info(ServiceType.CHAT)
        elif model.supports_service(ServiceType.SEARCH):
            service_info = model.get_service_info(ServiceType.SEARCH)
        
        if not service_info or service_info.pricing == 0:
            return None  # Free model
        
        # Model requires payment - ensure accounting is set up
        if not self.is_accounting_configured():
            if self.auto_setup_accounting:
                print(f"\n💰 Payment Required")
                print(f"Model '{model.name}' costs ${service_info.pricing} per request")
                print(f"Owner: {model.owner}")
                print(f"\nAccounting setup required for paid models.")
                
                try:
                    response = input("Would you like to set up accounting now? (y/n): ").lower().strip()
                    if response in ['y', 'yes']:
                        # Interactive setup would go here
                        print("Please use client.setup_accounting(email, password) to configure.")
                        return None
                    else:
                        print("Payment setup skipped.")
                        return None
                except (EOFError, KeyboardInterrupt):
                    print("\nPayment setup cancelled.")
                    return None
            else:
                raise PaymentError(
                    f"Model '{model.name}' requires payment (${service_info.pricing}) "
                    "but accounting is not configured"
                )
        
        # Create transaction token
        try:
            token = await self.rpc_client.create_transaction_token(model.owner)
            logger.info(f"💰 Payment authorized: ${service_info.pricing} to {model.owner}")
            return token
        except Exception as e:
            raise PaymentError(f"Failed to create payment token: {e}")
    
    # Updated Service Usage Methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered models.
        
        Returns:
            Dictionary with model statistics
        """
        models = self.discover_models(health_check="never", include_disabled=True)
        
        # Count by service type
        chat_models = [m for m in models if m.supports_service(ServiceType.CHAT)]
        search_models = [m for m in models if m.supports_service(ServiceType.SEARCH)]
        
        # Count by pricing
        free_models = [m for m in models if m.min_pricing == 0]
        paid_models = [m for m in models if m.min_pricing > 0]
        
        # Count by owner
        owners = {}
        for model in models:
            owners[model.owner] = owners.get(model.owner, 0) + 1
        
        return {
            "total_models": len(models),
            "enabled_models": len([m for m in models if m.has_enabled_services]),
            "disabled_models": len([m for m in models if not m.has_enabled_services]),
            "chat_models": len(chat_models),
            "search_models": len(search_models),
            "free_models": len(free_models),
            "paid_models": len(paid_models),
            "total_owners": len(owners),
            "avg_models_per_owner": len(models) / len(owners) if owners else 0,
            "top_owners": sorted(owners.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def clear_cache(self):
        """Clear the model discovery cache."""
        self.scanner.clear_cache()
    
    # Private Helper Methods
    def _should_do_health_check(self, health_check: str, model_count: int) -> bool:
        """Determine if health checking should be performed."""
        if health_check == "always":
            return True
        elif health_check == "never":
            return False
        elif health_check == "auto":
            return model_count <= self.auto_health_check_threshold
        else:
            raise ValueError(f"Invalid health_check value: {health_check}")
    
    async def _add_health_status(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Add health status to models."""
        health_status = await batch_health_check(models, self.rpc_client, timeout=2.0)
        
        for model in models:
            model.health_status = health_status.get(model.name, HealthStatus.UNKNOWN)
        
        return models
    
    def _select_best_model(self, models: List[ModelInfo], 
                          preference: QualityPreference) -> Optional[ModelInfo]:
        """Select the best model based on preference."""
        if not models:
            return None
        
        if preference == QualityPreference.CHEAPEST:
            return min(models, key=lambda m: m.min_pricing)
        elif preference == QualityPreference.PREMIUM:
            return max(models, key=lambda m: m.min_pricing)
        elif preference == QualityPreference.FASTEST:
            # Prefer models with health status ONLINE, then by pricing
            online_models = [m for m in models if m.health_status == HealthStatus.ONLINE]
            if online_models:
                return min(online_models, key=lambda m: m.min_pricing)
            return models[0]  # Fallback to first model
        elif preference == QualityPreference.BALANCED:
            # Balance between cost and quality indicators
            def score_model(model: ModelInfo) -> float:
                score = 0.0
                
                # Lower cost is better (inverse scoring)
                if model.min_pricing == 0:
                    score += 1.0  # Free is great
                else:
                    score += max(0, 1.0 - (model.min_pricing / 1.0))  # Diminishing returns
                
                # Health status bonus
                if model.health_status == HealthStatus.ONLINE:
                    score += 0.5
                
                # Quality tags bonus
                quality_tags = {'premium', 'gpt4', 'claude', 'high-quality', 'enterprise'}
                tag_matches = len(set(model.tags).intersection(quality_tags))
                score += tag_matches * 0.1
                
                # Multiple services bonus
                score += len(model.enabled_service_types) * 0.1
                
                return score
            
            return max(models, key=score_model)
        else:
            return models[0]
    
    def _model_to_dict(self, model: ModelInfo) -> Dict[str, Any]:
        """Convert ModelInfo to dictionary for JSON serialization."""
        return {
            "name": model.name,
            "owner": model.owner,
            "summary": model.summary,
            "description": model.description,
            "tags": model.tags,
            "services": [
                {
                    "type": service.type.value,
                    "enabled": service.enabled,
                    "pricing": service.pricing,
                    "charge_type": service.charge_type.value
                }
                for service in model.services
            ],
            "config_status": model.config_status.value,
            "health_status": model.health_status.value if model.health_status else None,
            "delegate_email": model.delegate_email,
            "min_pricing": model.min_pricing,
            "max_pricing": model.max_pricing
        }
    
    def _format_models_summary(self, models: List[ModelInfo]) -> str:
        """Format models as a summary."""
        if not models:
            return "No models found."
        
        lines = [f"Found {len(models)} models:\n"]
        
        # Group by owner
        by_owner = {}
        for model in models:
            if model.owner not in by_owner:
                by_owner[model.owner] = []
            by_owner[model.owner].append(model)
        
        for owner, owner_models in sorted(by_owner.items()):
            lines.append(f"📧 {owner} ({len(owner_models)} models)")
            
            for model in sorted(owner_models, key=lambda m: m.name):
                services = ", ".join([s.type.value for s in model.services if s.enabled])
                pricing = f"${model.min_pricing}" if model.min_pricing > 0 else "Free"
                health = ""
                if model.health_status:
                    if model.health_status == HealthStatus.ONLINE:
                        health = " ✅"
                    elif model.health_status == HealthStatus.OFFLINE:
                        health = " ❌"
                    elif model.health_status == HealthStatus.TIMEOUT:
                        health = " ⏱️"
                
                lines.append(f"  • {model.name} ({services}) - {pricing}{health}")
            
            lines.append("")  # Empty line between owners
        
        return "\n".join(lines)