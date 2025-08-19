"""
Main SyftBox NSAI SDK client
"""
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from .core.config import SyftBoxConfig, get_config, is_syftbox_available, get_installation_instructions
from .core.types import (
    ModelInfo, 
    ServiceType, 
    HealthStatus, 
    QualityPreference,
    ChatMessage, 
    ChatResponse, 
    SearchResponse, 
    FilterDict
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
                 user_email: Optional[str] = None,
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
        from_email = "guest@syft.org" or user_email or self.config.email
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
        
        logger.info(f"SyftBoxClient initialized for {from_email}")
    
    async def close(self):
        """Close client and cleanup resources."""
        await self.rpc_client.close()
        if self._health_monitor:
            await self._health_monitor.stop_monitoring()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ======================
    # MODEL DISCOVERY (UNCHANGED - WORKS WELL)
    # ======================
    # Model Discovery Methods
    def discover_models(self,
                       service_type: Optional[str] = None,
                       owner: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       max_cost: Optional[float] = None,
                       health_check: str = "auto",
                       include_disabled: bool = False,
                       **filter_kwargs) -> List[ModelInfo]:
        """Discover available models with filtering and optional health checking.
        
        Args:
            service_type: Filter by service type (chat, search)
            owner: Filter by owner email
            tags: Filter by tags (any match)
            max_cost: Maximum cost per request
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
                print(f"❌ Invalid service_type: {service_type}")
                return []
        
        # Apply filters
        filter_criteria = FilterCriteria(
            service_type=service_type_enum,
            owner=owner,
            has_any_tags=tags,
            max_cost=max_cost,
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

    # ======================
    # EXPLICIT MODEL USAGE - NEW DESIGN
    # ======================
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

    # ======================
    # CONVENIENCE METHODS FOR DISCOVERY + USAGE
    # ======================
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
    
    # ======================
    # MODEL-SPECIFIC PARAMETER HELPERS
    # ======================

    def get_model_parameters(self, model_name: str, owner: Optional[str] = None) -> Dict[str, Any]:
        """Get available parameters for a specific model.
        
        Args:
            model_name: Name of the model
            owner: Owner email if needed
            
        Returns:
            Dictionary of available parameters and their descriptions
        """
        model = self.find_model(model_name, owner)
        if not model:
            raise ModelNotFoundError(f"Model '{model_name}' not found")
        
        # Parse parameters from OpenAPI schema or RPC schema
        parameters = {}
        
        if model.endpoints:
            # Extract from OpenAPI spec
            chat_endpoint = model.endpoints.get("paths", {}).get("/chat", {})
            if "requestBody" in chat_endpoint:
                # Parse OpenAPI schema to extract parameters
                # This would need full implementation
                parameters = self._extract_parameters_from_openapi(chat_endpoint)
        
        if model.rpc_schema:
            # Extract from RPC schema
            chat_rpc = model.rpc_schema.get("/chat", {})
            if chat_rpc:
                parameters.update(self._extract_parameters_from_rpc(chat_rpc))
        
        return parameters

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
    
    # ======================
    # UPDATED SERVICE CLASSES
    # ======================

    
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
    
    def create_conversation(self, model_name: str) -> ConversationManager:
        """Create a conversation manager for a chat model.
        
        Args:
            model_name: Name of the chat model
            
        Returns:
            ConversationManager instance
        """
        chat_service = self.get_chat_service(model_name)
        return ConversationManager(chat_service)
    
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
            # raise ValueError(f"Unknown format: {format}")
    
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
        
        # 1. Try environment variables
        email = os.getenv("SYFTBOX_ACCOUNTING_EMAIL")
        password = os.getenv("SYFTBOX_ACCOUNTING_PASSWORD")
        service_url = os.getenv('SYFTBOX_ACCOUNTING_URL')
        logger.info(f"Using email: {email}, service URL: {service_url}")
        if email and password:
            self._configure_accounting({
                "email": email,
                "password": password,
                "service_url": service_url
            })
            return
        
        # 2. Try separate accounting config file
        accounting_config_path = Path.home() / ".syftbox" / "accounting.json"
        if accounting_config_path.exists():
            try:
                with open(accounting_config_path, 'r') as f:
                    config = json.load(f)
                
                if "email" in config and "password" in config:
                    self._configure_accounting({
                        "email": config["email"],
                        "password": config["password"],
                        "service_url": config.get("service_url", "https://accounting.syftbox.net")
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
    
    async def setup_accounting(self, email: str, password: str, service_url: str = None, organization: Optional[str] = None):
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
            service_url = service_url
            # service_url = self._get_accounting_service_url()
        
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
    
    async def setup_accounting1(self, email: str, password: str, service_url: str = None):
        """Setup accounting credentials.
        
        Args:
            email: Accounting service email
            password: Accounting service password  
            service_url: Accounting service URL
        """
        from syft_accounting_sdk import UserClient, ServiceException
        
        try:
            # Test credentials by creating client
            user_client = UserClient(url=service_url, email=email, password=password)
            # Test with a simple call
            user_client.get_user_info()
            
            # Configure if successful
            self._configure_accounting({
                "email": email,
                "password": password,
                "service_url": service_url
            })
            
            # Save credentials for future use
            await self._save_accounting_config(service_url, email, password)
            
            logger.info("Accounting credentials configured successfully")
        except ServiceException as e:
            raise AuthenticationError(f"Invalid accounting credentials: {e}")
    
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

# ======================
# CONVENIENCE FUNCTIONS (UPDATED)
# ======================

# Convenience functions for quick usage
async def quick_chat(message: str, max_cost: float = 1.0) -> str:
    """Quick chat function for simple use cases.
    
    Args:
        message: Message to send
        max_cost: Maximum cost willing to pay
        
    Returns:
        Response content as string
    """
    async with SyftBoxClient() as client:
        response = await client.chat(message, max_cost=max_cost)
        return response.message.content


async def quick_search(query: str, max_cost: float = 1.0, limit: int = 3) -> List[str]:
    """Quick search function for simple use cases.
    
    Args:
        query: Search query
        max_cost: Maximum cost willing to pay
        limit: Maximum results to return
        
    Returns:
        List of result contents as strings
    """
    async with SyftBoxClient() as client:
        response = await client.search(query, max_cost=max_cost, limit=limit)
        return [result.content for result in response.results]


def list_available_models(service_type: Optional[ServiceType] = None) -> str:
    """List available models (convenience function).
    
    Args:
        service_type: Optional service type filter
        
    Returns:
        Formatted table of available models
    """
    client = SyftBoxClient()
    return client.list_models(service_type=service_type)

async def chat_with_model(model_name: str, 
                         prompt: str,
                         owner: Optional[str] = None,
                         **params) -> str:
    """Convenience function for quick chat with specific model.
    
    Args:
        model_name: Name of the model to use
        prompt: Message to send  
        owner: Owner email
        **params: Additional parameters
        
    Returns:
        Response content as string
    """
    async with SyftBoxClient() as client:
        response = await client.chat(
            model_name=model_name,
            owner=owner,
            prompt=prompt,
            **params
        )
        return response.message.content


async def search_with_model(model_name: str,
                           query: str, 
                           owner: Optional[str] = None,
                           **params) -> List[str]:
    """Convenience function for quick search with specific model.
    
    Args:
        model_name: Name of the model to use
        query: Search query
        owner: Owner email
        **params: Additional parameters
        
    Returns:
        List of result contents
    """
    async with SyftBoxClient() as client:
        response = await client.search(
            model_name=model_name,
            owner=owner,
            query=query,
            **params
        )
        return [result.content for result in response.results]