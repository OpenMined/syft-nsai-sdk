"""
Main SyftBox NSAI SDK client
"""
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dotenv import load_dotenv

from .core.decorators import require_account
from .core.config import ConfigManager
from .core.types import (
    ServiceInfo, 
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
    ServiceNotFoundError, 
    ServiceNotSupportedError,
    SyftBoxNotRunningError, 
    ValidationError,
    raise_service_not_found, 
    raise_service_not_supported
)
from .discovery.scanner import ServiceScanner, FastScanner
from .discovery.parser import MetadataParser
from .discovery.filters import ServiceFilter, FilterCriteria, FilterBuilder
from .clients.rpc_client import SyftBoxRPCClient
from .clients.accounting_client import AccountingClient
from .services.chat import ChatService, ConversationManager
from .services.search import SearchService, BatchSearchService
from .services.health import check_service_health, batch_health_check, HealthMonitor
from .services.services_list import ServicesList
from .utils.formatting import format_services_table, format_service_details

from syft_accounting_sdk import UserClient, ServiceException

logger = logging.getLogger(__name__)
load_dotenv()

class Client:
    """Main client for discovering and using SyftBox AI services."""
    
    def __init__(self, 
                syftbox_config_path: Optional[Path] = None,
                # user_email: str = "guest@syft.org",
                cache_server_url: Optional[str] = None,
                accounting_client: Optional[AccountingClient] = None,
                auto_setup_accounting: bool = True,
                auto_health_check_threshold: int = 10
            ):
        """Initialize SyftBox client.
        
        Args:
            syftbox_config_path: Custom path to SyftBox config file
            user_email: Override user email for requests
            cache_server_url: Override cache server URL
            auto_setup_accounting: Whether to prompt for accounting setup when needed
            auto_health_check_threshold: Max services for auto health checking
        """
        # Check SyftBox availability
        # Create config manager with custom path if provided
        self.config_manager = ConfigManager(syftbox_config_path)
        
        # Check if installed first
        # if not self.config_manager.is_syftbox_installed():
        #     raise SyftBoxNotFoundError(self.config_manager.get_installation_instructions())

        # Then check if running  
        if not self.config_manager.is_syftbox_running():
            raise SyftBoxNotFoundError(self.config_manager.get_startup_instructions())
        
        # Store config for later use
        self.config = self.config_manager.config
        
        # Initialize account state
        self._account_configured = False
        
        # Load configuration
        # self.config = get_config(syftbox_config_path)

        # Set up accounting client - check for existing credentials
        if accounting_client:
            self.accounting_client = accounting_client
            if self.accounting_client.is_configured():
                self._account_configured = True
        else:
            self.accounting_client = self._setup_default_accounting()
        
        # Set up RPC client
        # from_email = user_email
        server_url = cache_server_url or self.config.cache_server_url
        
        self.rpc_client = SyftBoxRPCClient(
            cache_server_url=server_url,
            # from_email=from_email,
            accounting_client=self.accounting_client,
        )
        
        # Set up service scanner
        self.scanner = FastScanner(self.config)
        self.parser = MetadataParser()
        
        # Configuration
        self.auto_health_check_threshold = auto_health_check_threshold
        self.auto_setup_accounting = auto_setup_accounting
        
        # Optional health monitor
        self._health_monitor: Optional[HealthMonitor] = None

        # Load user email from config if not provided (from_email and self._account_configured)
        logger.info(f"Client initialized for {self.config.email}")
        # if self._account_configured:
        #     logger.info(f"Client initialized for {self.accounting_client.get_email()}")
        # # if from_email:
        # #     logger.info(f"Client initialized for {self.accounting_client.get_email()}")
        # else:
        #     logger.info("Client initialized in guest mode (no user account provided)")

    # def _setup_default_accounting1(self) -> AccountingClient:
    #     """Check for existing accounting credentials and guide user if none exist."""
    #     # Try loading from environment variables
    #     try:
    #         client = AccountingClient.from_environment()
    #         self._account_configured = True
    #         logger.info("Found existing accounting credentials in environment")
    #         return client
    #     except AuthenticationError:
    #         pass
        
    #     # Try loading from saved config file
    #     try:
    #         client = AccountingClient.load_from_config()
    #         self._account_configured = True
    #         logger.info("Found existing accounting credentials in config file")
    #         return client
    #     except AuthenticationError:
    #         pass
        
    #     # No credentials found - inform user
    #     self._show_setup_message()
    #     return AccountingClient()
    
    def _setup_default_accounting(self) -> AccountingClient:
        client, is_configured = AccountingClient.setup_accounting_discovery()
        
        if is_configured:
            self._account_configured = True
            # logger.info(await client.get_account_info())  # Validate credentials
            logger.info(f"Found existing accounting credentials for {client.get_email()}")
        else:
            self._show_setup_message()
        
        return client
    
    def _show_setup_message(self):
        """Display account setup instructions to user."""
        print("\n" + "="*60)
        print("NO ACTIVE ACCOUNT FOUND!")
        print("="*60)
        print("You are currently limited to SyftBox free services.")
        print("New users receive $20 in free credits upon first connection to the accounting service.")
        print("To use SyftBox paid services, you need to set up an account.")
        print("")
        print("Please run:")
        print("  await client.register_accounting(email, password) to create an account.")
        print("  await client.connect_accounting(email, password) to connect to an existing account.")
        print("")
        print("Or set environment variables:")
        print("  SYFTBOX_ACCOUNTING_EMAIL=your_email@example.com")
        print("  SYFTBOX_ACCOUNTING_PASSWORD=your_password")
        print("  SYFTBOX_ACCOUNTING_URL=https://service.url")
        print("="*60)
    
    async def close(self):
        """Close client and cleanup resources."""
        await self.rpc_client.close()
        if self._health_monitor:
            await self._health_monitor.stop_monitoring()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # Service Discovery Methods
    def list_services(self,
                    service_type: Optional[str] = None,
                    datasite: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    max_cost: Optional[float] = None,
                    free_only: bool = False,
                    health_check: str = "auto",
                    **filter_kwargs) -> List[ServiceInfo]:
        """Discover available services with filtering and optional health checking.
        
        Args:
            service_type: Filter by service type (chat, search)
            datasite: Filter by datasite email
            tags: Filter by tags (any match)
            max_cost: Maximum cost per request
            free_only: Only return free services (cost = 0)
            health_check: Health checking mode ("auto", "always", "never")
            **filter_kwargs: Additional filter criteria
            
        Returns:
            List of discovered and filtered services
        """
        # Scan for metadata files
        metadata_paths = self.scanner.scan_with_cache()
        
        # Parse services from metadata
        services = []
        for metadata_path in metadata_paths:
            try:
                service_info = self.parser.parse_service_from_files(metadata_path)
                services.append(service_info)
            except Exception as e:
                logger.debug(f"Failed to parse {metadata_path}: {e}")
                continue

        # Convert string to enum
        service_type_enum = None
        if service_type:
            try:
                service_type_enum = ServiceType(service_type.lower())
            except ValueError:
                logger.error(f"Invalid service_type: {service_type}")
                return []
        
        # Apply filters
        filter_criteria = FilterCriteria(
            service_type=service_type_enum,
            datasite=datasite,
            has_any_tags=tags,
            max_cost=max_cost,
            free_only=free_only,
            enabled_only=True,
            **filter_kwargs
        )
        
        service_filter = ServiceFilter(filter_criteria)
        filtered_services = service_filter.filter_services(services)
        
        # Determine if we should do health checking
        should_health_check = self._should_do_health_check(
            health_check, len(filtered_services)
        )
        
        if should_health_check:
            # Check if we're in Jupyter to avoid asyncio.run() issues
            try:
                import IPython
                ipython = IPython.get_ipython()
                if ipython is not None and hasattr(ipython, 'kernel'):
                    # We're in Jupyter, skip health check to avoid asyncio issues
                    logger.info("Skipping health check in Jupyter environment to avoid asyncio issues")
                else:
                    # Not in Jupyter, safe to use asyncio.run()
                    filtered_services = asyncio.run(self._add_health_status(filtered_services))
            except ImportError:
                # IPython not available, safe to use asyncio.run()
                filtered_services = asyncio.run(self._add_health_status(filtered_services))
            except Exception as e:
                # Any other error, log and continue without health check
                logger.warning(f"Health check failed: {e}. Continuing without health status.")
        
        logger.info(f"Discovered {len(filtered_services)} services (health_check={should_health_check})")
        return ServicesList(filtered_services, self)
    
    def get_service(self, service_name: str, datasite: Optional[str] = None) -> Optional[ServiceInfo]:
        """Find a specific service by name.
        
        Args:
            service_name: Name of the service to find
            datasite: Optional datasite email to narrow search
            
        Returns:
            ServiceInfo if found, None otherwise
        """
        services = self.list_services(name=service_name, datasite=datasite, health_check="never")
        
        # Find exact match
        for service in services:
            if service.name == service_name:
                if datasite is None or service.datasite == datasite:
                    return service
        
        return None
    
    # Display Methods 
    def format_services(self, 
                   service_type: Optional[ServiceType] = None,
                   health_check: str = "auto",
                   format: str = "table") -> str:
        """List available services in a user-friendly format.
        
        Args:
            service_type: Optional service type filter
            health_check: Health checking mode ("auto", "always", "never")
            format: Output format ("table", "json", "summary")
            
        Returns:
            Formatted string with service information
        """
        services = self.list_services(
            service_type=service_type,
            health_check=health_check
        )
        
        if format == "table":
            return format_services_table(services)
        elif format == "json":
            import json
            service_dicts = [self._service_to_dict(service) for service in services]
            return json.dumps(service_dicts, indent=2)
        elif format == "summary":
            return self._format_services_summary(services)
        else:
            return [self._service_to_dict(service) for service in services]
    
    def show_service_details(self, service_name: str, datasite: Optional[str] = None) -> str:
        """Show detailed information about a specific service.
        
        Args:
            service_name: Name of the service
            datasite: Optional datasite to narrow search
            
        Returns:
            Formatted service details
        """
        service = self.get_service(service_name, datasite)
        if not service:
            return f"Service '{service_name}' not found"
        
        return format_service_details(service)
    
    # Service Usage Methods
    @require_account
    def chat(self,
                   service_name: str,
                   prompt: str,
                   datasite: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   auto_pay: bool = True,
                   **kwargs) -> ChatResponse:
        """Chat with a specific service.
        
        Args:
            service_name: Name of the service to use (REQUIRED)
            prompt: Message to send
            datasite: Datasite email (required if service name is ambiguous)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            auto_pay: Automatically handle payment for paid services
            **kwargs: Additional service-specific parameters
            
        Returns:
            Chat response from the specified service
            
        Example:
            response = await client.chat(
                service_name="public-tinnyllama",
                datasite="irina@openmined.org", 
                prompt="Hello! Testing the API",
                temperature=0.7
            )
        """
        # Find the specific service
        service = self.get_service(service_name, datasite)
        if not service:
            if datasite:
                raise ServiceNotFoundError(f"Service '{service_name}' not found for datasite '{datasite}'")
            else:
                # Show available services with same name
                similar_services = [m for m in self.list_services() if m.name == service_name]
                if len(similar_services) > 1:
                    datasites = [m.datasite for m in similar_services]
                    raise ValidationError(
                        f"Multiple services named '{service_name}' found. "
                        f"Please specify datasite. Available datasites: {', '.join(datasites)}"
                    )
                else:
                    raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        # Check if service supports chat
        if not service.supports_service(ServiceType.CHAT):
            raise_service_not_supported(service.name, "chat", service)
            # raise ValidationError(f"Service '{service_name}' does not support chat service")
        
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
        chat_service = ChatService(service, self.rpc_client)

        return asyncio.run(chat_service.chat_with_params(chat_params))

    @require_account
    def search(self,
                    service_name: str, 
                    query: str,
                    datasite: Optional[str] = None,
                    limit: Optional[int] = None,
                    similarity_threshold: Optional[float] = None,
                    **kwargs) -> SearchResponse:
        """Search with a specific service.
        
        Args:
            service_name: Name of the service to use (REQUIRED)
            query: Search query
            datasite: Datasite email (required if service name is ambiguous)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            **kwargs: Additional service-specific parameters
            
        Returns:
            Search response from the specified service
            
        Example:
            results = await client.search(
                service_name="the-city",
                query="latest news",
                datasite="speters@thecity.nyc"
            )
        """
        # Find the specific service
        service = self.get_service(service_name, datasite)
        if not service:
            if datasite:
                raise ServiceNotFoundError(f"Service '{service_name}' not found for datasite '{datasite}'")
            else:
                raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        # Check if service supports search
        if not service.supports_service(ServiceType.SEARCH):
            raise ValidationError(f"Service '{service_name}' does not support search service")
        
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
        search_service = SearchService(service, self.rpc_client)

        return asyncio.run(search_service.search_with_params(search_params))

    @require_account
    async def chat_async(self,
                   service_name: str,
                   prompt: str,
                   datasite: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   auto_pay: bool = True,
                   **kwargs) -> ChatResponse:
        """Chat with a specific service.
        
        Args:
            service_name: Name of the service to use (REQUIRED)
            prompt: Message to send
            datasite: Datasite email (required if service name is ambiguous)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            auto_pay: Automatically handle payment for paid services
            **kwargs: Additional service-specific parameters
            
        Returns:
            Chat response from the specified service
            
        Example:
            response = await client.chat(
                service_name="public-tinnyllama",
                datasite="irina@openmined.org", 
                prompt="Hello! Testing the API",
                temperature=0.7
            )
        """
        # Find the specific service
        service = self.get_service(service_name, datasite)
        if not service:
            if datasite:
                raise ServiceNotFoundError(f"Service '{service_name}' not found for datasite '{datasite}'")
            else:
                # Show available services with same name
                similar_services = [m for m in self.list_services() if m.name == service_name]
                if len(similar_services) > 1:
                    datasites = [m.datasite for m in similar_services]
                    raise ValidationError(
                        f"Multiple services named '{service_name}' found. "
                        f"Please specify datasite. Available datasites: {', '.join(datasites)}"
                    )
                else:
                    raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        # Check if service supports chat
        if not service.supports_service(ServiceType.CHAT):
            raise_service_not_supported(service.name, "chat", service)
            # raise ValidationError(f"Service '{service_name}' does not support chat service")
        
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
        chat_service = ChatService(service, self.rpc_client)
        
        return await chat_service.chat_with_params(chat_params)
    
    @require_account
    async def search_async(self,
                    service_name: str, 
                    query: str,
                    datasite: Optional[str] = None,
                    limit: Optional[int] = None,
                    similarity_threshold: Optional[float] = None,
                    **kwargs) -> SearchResponse:
        """Search with a specific service.
        
        Args:
            service_name: Name of the service to use (REQUIRED)
            query: Search query
            datasite: Datasite email (required if service name is ambiguous)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            **kwargs: Additional service-specific parameters
            
        Returns:
            Search response from the specified service
            
        Example:
            results = await client.search(
                service_name="the-city",
                query="latest news",
                datasite="speters@thecity.nyc"
            )
        """
        # Find the specific service
        service = self.get_service(service_name, datasite)
        if not service:
            if datasite:
                raise ServiceNotFoundError(f"Service '{service_name}' not found for datasite '{datasite}'")
            else:
                raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        # Check if service supports search
        if not service.supports_service(ServiceType.SEARCH):
            raise ValidationError(f"Service '{service_name}' does not support search service")
        
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
        search_service = SearchService(service, self.rpc_client)
        
        return await search_service.search_with_params(search_params)
    
    # Service Parameters
    def get_service_parameters(self, service_name: str, datasite: Optional[str] = None) -> Dict[str, Any]:
        """Get available parameters for a specific service."""
        service = self.get_service(service_name, datasite)
        if not service:
            raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        parameters = {}
        
        if service.endpoints and "components" in service.endpoints:
            schemas = service.endpoints["components"].get("schemas", {})
            
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
        skip_fields = {"userEmail", "service", "messages", "query", "transactionToken"}
        
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

    def show_service_usage(self, service_name: str, datasite: Optional[str] = None) -> str:
        """Show usage examples for a specific service.
        
        Args:
            service_name: Name of the service
            datasite: Datasite email if needed
            
        Returns:
            Formatted usage examples
        """
        service = self.get_service(service_name, datasite)
        if not service:
            raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        examples = []
        examples.append(f"# Usage examples for {service.name}")
        examples.append(f"# Datasite: {service.datasite}")
        examples.append("")
        
        if service.supports_service(ServiceType.CHAT):
            examples.extend([
                "# Basic chat",
                f'response = await client.chat(',
                f'    service_name="{service.name}",',
                f'    datasite="{service.datasite}",',
                f'    prompt="Hello! How are you?"',
                f')',
                "",
                "# Chat with parameters",
                f'response = await client.chat(',
                f'    service_name="{service.name}",',
                f'    datasite="{service.datasite}",',
                f'    prompt="Write a story",',
                f'    temperature=0.7,',
                f'    max_tokens=200',
                f')',
                ""
            ])
        
        if service.supports_service(ServiceType.SEARCH):
            examples.extend([
                "# Basic search",
                f'results = await client.search(',
                f'    service_name="{service.name}",',
                f'    datasite="{service.datasite}",',
                f'    query="machine learning"',
                f')',
                "",
                "# Search with parameters", 
                f'results = await client.search(',
                f'    service_name="{service.name}",',
                f'    datasite="{service.datasite}",',
                f'    query="latest AI research",',
                f'    limit=10,',
                f'    similarity_threshold=0.8',
                f')',
                ""
            ])
        
        # Add pricing info
        if service.min_pricing > 0:
            examples.append(f"# Cost: ${service.min_pricing} per request")
        else:
            examples.append("# Cost: Free")
        
        return "\n".join(examples)
    
    # RAG Workflow
    @require_account
    async def chat_with_search_context(self,
                                    search_services: Union[str, List[str], List[Dict[str, str]]],
                                    chat_service: str,
                                    prompt: str,
                                    chat_datasite: Optional[str] = None,
                                    max_search_results: int = 3,
                                    search_similarity_threshold: Optional[float] = None,
                                    context_format: str = "frontend",  # "frontend" or "simple"
                                    **chat_kwargs) -> ChatResponse:
        """Perform search across multiple services then chat with context injection.
        
        This method replicates the frontend pattern where users can:
        1. Chat only (if no search_services provided)
        2. Search + Chat (if search_services provided - becomes RAG workflow)
        
        Args:
            search_services: Search services to query (OPTIONAL). Can be:
                        - None or [] for chat-only
                        - Single service name: "service-name"
                        - List of names: ["service1", "service2"] 
                        - List with datasites: [{"name": "service1", "datasite": "user@email.com"}]
            chat_service: Name of chat service for final response (REQUIRED)
            prompt: User's question/prompt
            chat_datasite: Datasite of chat service (if ambiguous)
            max_search_results: Max results per search service (ignored if no search)
            search_similarity_threshold: Minimum similarity score (ignored if no search)
            context_format: How to format context ("frontend" matches web app, "simple" is cleaner)
            **chat_kwargs: Additional parameters for chat request (temperature, max_tokens, etc.)
            
        Returns:
            ChatResponse with or without search context
            
        Example:
            # Chat only (like frontend with no data sources selected)
            response = await client.chat_with_search_context(
                search_services=[],  # No search - just chat
                chat_service="gpt-assistant", 
                prompt="What is machine learning?"
            )
            
            # RAG workflow (like frontend with data sources selected)
            response = await client.chat_with_search_context(
                search_services=["legal-docs", "company-policies"],
                chat_service="gpt-assistant", 
                prompt="What are our remote work policies?",
                max_search_results=5,
                temperature=0.7
            )
        """
        logger.info(f"Starting RAG workflow: search → chat")
        
        # Normalize search services to list of dicts
        search_service_specs = self._normalize_service_specs(search_services)
        
        if not search_service_specs:
            # No search services - do chat only (like frontend with no data sources)
            logger.info(f"Chat-only mode: no search services specified")
            
            # Find and validate chat service
            chat_service_info = self.get_service(chat_service, chat_datasite)
            if not chat_service_info:
                raise ServiceNotFoundError(f"Chat service '{chat_service}' not found" + 
                                    (f" for datasite '{chat_datasite}'" if chat_datasite else ""))
            
            if not chat_service_info.supports_service(ServiceType.CHAT):
                raise ValidationError(f"Service '{chat_service}' does not support chat service")
            
            # Direct chat without search context
            chat_service = ChatService(chat_service_info, self.rpc_client)
            
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
                "search_services_used": [],
                "search_results_count": 0,
                "context_injected": False
            })
            
            logger.info(f"✅ Chat-only completed - Cost: ${chat_response.cost:.4f}")
            return chat_response
        
        # Step 1: Search across specified services (RAG mode)
        logger.info(f"🔍 RAG mode: searching {len(search_service_specs)} services for context")
        search_responses = []
        
        for service_spec in search_service_specs:
            service_name = service_spec["name"]
            datasite = service_spec.get("datasite")
            
            try:
                search_response = await self.search(
                    service_name=service_name,
                    query=prompt,  # Use the prompt as search query
                    datasite=datasite,
                    limit=max_search_results,
                    similarity_threshold=search_similarity_threshold
                )
                search_responses.append(search_response)
                logger.debug(f"✅ Search completed: {service_name} returned {len(search_response.results)} results")
                
            except Exception as e:
                logger.warning(f"⚠️ Search failed for {service_name}: {e}")
                # Continue with other services - graceful degradation
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
        
        # Step 5: Send to chat service
        logger.info(f"Sending enhanced prompt to chat service: {chat_service}")
        
        # Find and validate chat service
        chat_service_info = self.get_service(chat_service, chat_datasite)
        if not chat_service_info:
            raise ServiceNotFoundError(f"Chat service '{chat_service}' not found" + 
                                (f" for datasite '{chat_datasite}'" if chat_datasite else ""))
        
        if not chat_service_info.supports_service(ServiceType.CHAT):
            raise ValidationError(f"Service '{chat_service}' does not support chat service")
        
        # Create chat service and send enhanced conversation
        chat_service = ChatService(chat_service_info, self.rpc_client)
        
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
                "search_services_used": [spec["name"] for spec in search_service_specs],
                "search_results_count": len(all_results),
                "total_search_cost": total_search_cost,
                "context_injected": context_message is not None
            })
            
            # Add combined cost
            chat_response.cost = (chat_response.cost or 0.0) + total_search_cost
            
            logger.info(f"RAG workflow completed - Total cost: ${chat_response.cost:.4f}")
            return chat_response
            
        except Exception as e:
            logger.error(f"Chat request failed in RAG workflow: {e}")
            raise

    @require_account
    async def search_multiple_services(self,
                                    service_names: Union[List[str], List[Dict[str, str]]],
                                    query: str,
                                    limit_per_service: int = 3,
                                    total_limit: Optional[int] = None,
                                    similarity_threshold: Optional[float] = None,
                                    remove_duplicates: bool = True,
                                    sort_by_score: bool = True,
                                    **search_kwargs) -> SearchResponse:
        """Search across multiple services and aggregate results.
        
        Args:
            service_names: List of service names or service specs with datasites
            query: Search query
            limit_per_service: Max results per individual service
            total_limit: Max results in final aggregated response (None = no limit)
            similarity_threshold: Minimum similarity score
            remove_duplicates: Remove duplicate content based on content hash
            sort_by_score: Sort final results by similarity score (descending)
            **search_kwargs: Additional parameters for search requests
            
        Returns:
            SearchResponse with aggregated results from all services
            
        Example:
            # Search multiple data sources
            results = await client.search_multiple_services(
                service_names=["legal-docs", "company-wiki", "slack-archive"],
                query="vacation policy changes",
                limit_per_service=5,
                total_limit=10
            )
        """
        logger.info(f"Multi-service search across {len(service_names)} services")
        
        # Normalize service specs
        service_specs = self._normalize_service_specs(service_names)
        
        all_results = []
        total_cost = 0.0
        successful_services = []
        failed_services = []
        
        # Search each service
        for service_spec in service_specs:
            service_name = service_spec["name"]
            datasite = service_spec.get("datasite")
            
            try:
                response = await self.search(
                    service_name=service_name,
                    query=query,
                    datasite=datasite,
                    limit=limit_per_service,
                    similarity_threshold=similarity_threshold,
                    **search_kwargs
                )
                
                all_results.extend(response.results)
                total_cost += response.cost or 0.0
                successful_services.append(service_name)
                
                logger.debug(f"✅ {service_name}: {len(response.results)} results")
                
            except Exception as e:
                logger.warning(f"⚠️ Search failed for {service_name}: {e}")
                failed_services.append({"service": service_name, "error": str(e)})
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
                "multi_service_search": True,
                "successful_services": successful_services,
                "failed_services": failed_services,
                "total_services_searched": len(successful_services),
                "deduplication_applied": remove_duplicates,
                "sorted_by_score": sort_by_score
            }
        )
        
        logger.info(f"Multi-search completed: {len(all_results)} results from {len(successful_services)}/{len(service_specs)} services")
        
        return aggregated_response
    
    @require_account
    async def search_then_chat(self,
                            search_service: str,
                            chat_service: str,
                            prompt: str,
                            search_datasite: Optional[str] = None,
                            chat_datasite: Optional[str] = None,
                            **kwargs) -> ChatResponse:
        """Simplified single-service search then chat workflow.
        
        Args:
            search_service: Name of service to search
            chat_service: Name of service to chat with
            prompt: User prompt (used for both search and chat)
            search_datasite: Datasite of search service
            chat_datasite: Datasite of chat service
            **kwargs: Additional parameters for both search and chat
            
        Returns:
            ChatResponse with search context
            
        Example:
            response = await client.search_then_chat(
                search_service="company-docs",
                chat_service="assistant-gpt",
                prompt="How do I submit expenses?"
            )
        """
        return await self.chat_with_search_context(
            search_services=[{"name": search_service, "datasite": search_datasite} if search_datasite else search_service],
            chat_service=chat_service,
            prompt=prompt,
            chat_datasite=chat_datasite,
            **kwargs
        )

    # Private methods to support RAG coordination
    def _normalize_service_specs(self, services: Union[str, List[str], List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Normalize various service specification formats to list of dicts.
        
        Args:
            services: Services in various formats
            
        Returns:
            List of service specs with 'name' and optional 'datasite' keys
        """
        if isinstance(services, str):
            # Single service name
            return [{"name": services}]
        
        elif isinstance(services, list):
            normalized = []
            for service in services:
                if isinstance(service, str):
                    # List of service names
                    normalized.append({"name": service})
                elif isinstance(service, dict):
                    # List of service specs
                    if "name" not in service:
                        raise ValidationError(f"Service spec missing 'name' key: {service}")
                    normalized.append(service)
                else:
                    raise ValidationError(f"Invalid service specification: {service}")
            return normalized
        
        else:
            raise ValidationError(f"Invalid services format: {type(services)}")

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
                            search_services: Union[str, List[str], List[Dict[str, str]]],
                            chat_service: str,
                            chat_datasite: Optional[str] = None) -> Dict[str, Any]:
        """Get cost estimate for RAG workflow before execution.
        
        Provides transparent cost breakdown so users can make informed decisions
        about which services to use (matching the frontend's cost preview).
        
        Args:
            search_services: Search services to estimate
            chat_service: Chat service to estimate
            chat_datasite: Datasite of chat service
            
        Returns:
            Dictionary with cost breakdown and service details
            
        Example:
            # Preview costs before RAG workflow
            estimate = client.get_rag_cost_estimate(
                search_services=["docs-1", "docs-2"],
                chat_service="paid-chat"
            )
            print(f"Total cost: ${estimate['total_cost']}")
            print(f"Services: {estimate['service_summary']}")
        """
        search_service_specs = self._normalize_service_specs(search_services)
        
        breakdown = {
            "search_services": [],
            "chat_service": None,
            "search_cost": 0.0,
            "chat_cost": 0.0, 
            "total_cost": 0.0,
            "service_summary": {
                "search_count": len(search_service_specs),
                "search_names": [spec["name"] for spec in search_service_specs],
                "chat_name": chat_service
            }
        }
        
        # Get search service costs
        for spec in search_service_specs:
            service = self.get_service(spec["name"], spec.get("datasite"))
            if service:
                search_service = service.get_service_info(ServiceType.SEARCH)
                if search_service:
                    service_cost = search_service.pricing
                    breakdown["search_cost"] += service_cost
                    breakdown["search_services"].append({
                        "name": service.name,
                        "datasite": service.datasite,
                        "cost": service_cost,
                        "charge_type": search_service.charge_type.value
                    })
        
        # Get chat service cost
        chat_service_info = self.get_service(chat_service, chat_datasite)
        if chat_service_info:
            chat_service = chat_service_info.get_service_info(ServiceType.CHAT)
            if chat_service:
                breakdown["chat_cost"] = chat_service.pricing
                breakdown["chat_service"] = {
                    "name": chat_service_info.name,
                    "datasite": chat_service_info.datasite,
                    "cost": chat_service.pricing,
                    "charge_type": chat_service.charge_type.value
                }
        
        breakdown["total_cost"] = breakdown["search_cost"] + breakdown["chat_cost"]
        
        return breakdown

    def preview_rag_workflow(self,
                            search_services: Union[str, List[str], List[Dict[str, str]]],
                            chat_service: str,
                            chat_datasite: Optional[str] = None) -> str:
        """Preview RAG workflow with service details and costs.
        
        Provides a human-readable preview of the RAG workflow showing exactly
        which services will be used and their costs (transparency like frontend).
        
        Args:
            search_services: Search services for the workflow
            chat_service: Chat service for the workflow  
            chat_datasite: Datasite of chat service
            
        Returns:
            Formatted preview string
            
        Example:
            preview = client.preview_rag_workflow(
                search_services=["legal-docs", "hr-policies"],
                chat_service="gpt-assistant"
            )
            print(preview)
        """
        estimate = self.get_rag_cost_estimate(search_services, chat_service, chat_datasite)
        
        lines = [
            "RAG Workflow Preview",
            "=" * 30,
            "",
            f"Search Phase ({len(estimate['search_services'])} services):"
        ]
        
        if estimate["search_services"]:
            for service in estimate["search_services"]:
                cost_str = f"${service['cost']:.4f}" if service['cost'] > 0 else "Free"
                lines.append(f"  • {service['name']} by {service['datasite']} - {cost_str}")
            lines.append(f"  Subtotal: ${estimate['search_cost']:.4f}")
        else:
            lines.append("  • No valid search services found")
        
        lines.extend([
            "",
            "Chat Phase:"
        ])
        
        if estimate["chat_service"]:
            chat = estimate["chat_service"]
            cost_str = f"${chat['cost']:.4f}" if chat['cost'] > 0 else "Free"
            lines.append(f"  • {chat['name']} by {chat['datasite']} - {cost_str}")
        else:
            lines.append("  • Chat service not found")
        
        lines.extend([
            "",
            f"Total Estimated Cost: ${estimate['total_cost']:.4f}",
            "",
            "To execute: client.chat_with_search_context(...)"
        ])
        
        return "\n".join(lines)
    
    def get_chat_service(self, service_name: str) -> ChatService:
        """Get a chat service for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            ChatService instance
        """
        service = self.get_service(service_name)
        if not service:
            raise_service_not_found(service_name)
        
        return ChatService(service, self.rpc_client)
    
    def get_search_service(self, service_name: str) -> SearchService:
        """Get a search service for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            SearchService instance
        """
        service = self.get_service(service_name)
        if not service:
            raise_service_not_found(service_name)
        
        return SearchService(service, self.rpc_client)
    
    def create_conversation(self, service_name: str, datasite: Optional[str] = None) -> ConversationManager:
        """Create a conversation manager for a chat service.
        
        Args:
            service_name: Name of the chat service
            datasite: Datasite email (required if service name is ambiguous)
            
        Returns:
            ConversationManager instance
        """
        # Find and validate service
        service = self.get_service(service_name, datasite)
        if not service:
            if datasite:
                raise ServiceNotFoundError(f"Service '{service_name}' not found for datasite '{datasite}'")
            else:
                raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        if not service.supports_service(ServiceType.CHAT):
            raise ValidationError(f"Service '{service_name}' does not support chat service")
        
        # Create chat service and conversation manager
        chat_service = ChatService(service, self.rpc_client)
        return ConversationManager(chat_service)
    
    # Health Monitoring Methods
    async def check_service_health(self, service_name: str, timeout: float = 2.0) -> HealthStatus:
        """Check health of a specific service.
        
        Args:
            service_name: Name of the service to check
            timeout: Timeout for health check
            
        Returns:
            Health status of the service
        """
        service = self.get_service(service_name)
        if not service:
            raise_service_not_found(service_name)
        
        return await check_service_health(service, self.rpc_client, timeout)
    
    async def check_all_services_health(self, 
                                     service_type: Optional[ServiceType] = None,
                                     timeout: float = 2.0) -> Dict[str, HealthStatus]:
        """Check health of all discovered services.
        
        Args:
            service_type: Optional service type filter
            timeout: Timeout per health check
            
        Returns:
            Dictionary mapping service names to health status
        """
        services = self.list_services(service_type=service_type, health_check="never")
        return await batch_health_check(services, self.rpc_client, timeout)
    
    def start_health_monitoring(self, 
                               services: Optional[List[str]] = None,
                               check_interval: float = 30.0) -> HealthMonitor:
        """Start continuous health monitoring.
        
        Args:
            services: Optional list of service names to monitor (default: all chat/search services)
            check_interval: Seconds between health checks
            
        Returns:
            HealthMonitor instance
        """
        if self._health_monitor:
            logger.warning("Health monitoring already running")
            return self._health_monitor
        
        self._health_monitor = HealthMonitor(self.rpc_client, check_interval)
        
        # Add services to monitor
        if services:
            for service_name in services:
                service = self.get_service(service_name)
                if service:
                    self._health_monitor.add_service(service)
        else:
            # Monitor all enabled chat/search services
            all_services = self.list_services(health_check="never")
            for service in all_services:
                if service.supports_service(ServiceType.CHAT) or service.supports_service(ServiceType.SEARCH):
                    self._health_monitor.add_service(service)
        
        # Start monitoring
        asyncio.create_task(self._health_monitor.start_monitoring())
        
        return self._health_monitor
    
    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        if self._health_monitor:
            await self._health_monitor.stop_monitoring()
            self._health_monitor = None
    
    # Accounting Integration Methods
    def register_accounting(self, email: str, password: str, organization: Optional[str] = None):
        """
        Register a new accounting user.
        """
        try:
            asyncio.run(self.accounting_client.create_accounting_user(email, password, organization))
            self.accounting_client.save_credentials()

            # logger.info("Accounting setup successful")
            self.connect_accounting(email, password, self.accounting_client.accounting_url)
            logger.info("Accounting setup completed and connected successful")

        except Exception as e:
            raise AuthenticationError(f"Accounting setup failed: {e}")

    def connect_accounting(self, email: str, password: str, accounting_url: Optional[str] = None, save_config: bool = False):
        """Setup accounting credentials.
        
        Args:
            email: Accounting service email
            password: Accounting service password  
            accounting_url: Accounting service URL (uses env var if not provided)
            save_config: Whether to save config to file (requires explicit user consent)
        """
        # Get service URL from environment if not provided
        if accounting_url is None:
            accounting_url = os.getenv('SYFTBOX_ACCOUNTING_URL')
        
        if not accounting_url:
            raise ValueError(
                "Accounting service URL is required. Please either:\n"
                "1. Set SYFTBOX_ACCOUNTING_URL in your .env file, or\n"
                "2. Pass accounting_url parameter to this method"
            )
        
        try:
            # Configure the accounting client
            self.accounting_client.configure(accounting_url, email, password)
            
            # Test the connection
            # await self.accounting_client.get_account_info()
            
            # Save config if explicitly requested
            if save_config:
                self.accounting_client.save_credentials()

            logger.info(f"Accounting setup successful for {self.accounting_client.get_email()}")

        except Exception as e:
            raise AuthenticationError(f"Accounting setup failed: {e}")
        
    async def register_accounting_async(self, email: str, password: str, organization: Optional[str] = None):
        """
        Register a new accounting user.
        """
        try:
            await self.accounting_client.create_accounting_user(email, password, organization)
            self.accounting_client.save_credentials()

            # logger.info("Accounting setup successful")
            await self.connect_accounting_async(email, password, self.accounting_client.accounting_url)
            logger.info("Accounting setup completed and connected successful")

        except Exception as e:
            raise AuthenticationError(f"Accounting setup failed: {e}")

    async def connect_accounting_async(self, email: str, password: str, accounting_url: Optional[str] = None, save_config: bool = False):
        """Setup accounting credentials.
        
        Args:
            email: Accounting service email
            password: Accounting service password  
            accounting_url: Accounting service URL (uses env var if not provided)
            save_config: Whether to save config to file (requires explicit user consent)
        """
        # Get service URL from environment if not provided
        if accounting_url is None:
            accounting_url = os.getenv('SYFTBOX_ACCOUNTING_URL')
        
        if not accounting_url:
            raise ValueError(
                "Accounting service URL is required. Please either:\n"
                "1. Set SYFTBOX_ACCOUNTING_URL in your .env file, or\n"
                "2. Pass accounting_url parameter to this method"
            )
        
        try:
            # Configure the accounting client
            self.accounting_client.configure(accounting_url, email, password)
            
            # Test the connection
            # await self.accounting_client.get_account_info()
            
            # Save config if explicitly requested
            if save_config:
                self.accounting_client.save_credentials()

            logger.info(f"Accounting setup successful for {self.accounting_client.get_email()}")

        except Exception as e:
            raise AuthenticationError(f"Accounting setup failed: {e}")
        
    def is_accounting_configured(self) -> bool:
        """Check if accounting is properly configured."""
        return self.accounting_client.is_configured()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balance."""
        if not self.is_accounting_configured():
            return {"error": "Accounting not configured"}
        
        try:
            return await self.accounting_client.get_account_info()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
    
    def show_accounting_status(self) -> str:
        """Show current accounting configuration status."""
        if not self.is_accounting_configured():
            return (
                "Accounting not configured\n"
                "   Use client.setup_accounting() to configure payment services\n"
                "   Currently limited to free services only"
            )
        
        try:
            import asyncio
            account_info = asyncio.run(self.get_account_info())
            
            if "error" in account_info:
                return (
                    f"Accounting configured but connection failed\n"
                    f"   Error: {account_info['error']}\n"
                    f"   May need to reconfigure credentials"
                )
            
            return (
                f"Accounting configured\n"
                f"   Email: {account_info['email']}\n" 
                f"   Balance: ${account_info['balance']}\n"
                f"   Can use both free and paid services"
            )
        except Exception as e:
            return (
                f"Accounting configured but connection failed\n"
                f"   Error: {e}\n"
                f"   May need to reconfigure credentials"
            )
        
    async def _ensure_payment_setup(self, service: ServiceInfo) -> Optional[str]:
        """Ensure payment is set up for a paid service.
        
        Args:
            service: Service that requires payment
            
        Returns:
            Transaction token if payment required, None if free
        """
        # Check if service requires payment
        service_info = None
        if service.supports_service(ServiceType.CHAT):
            service_info = service.get_service_info(ServiceType.CHAT)
        elif service.supports_service(ServiceType.SEARCH):
            service_info = service.get_service_info(ServiceType.SEARCH)
        
        if not service_info or service_info.pricing == 0:
            return None  # Free service
        
        # Service requires payment - ensure accounting is set up
        if not self.is_accounting_configured():
            if self.auto_setup_accounting:
                print(f"\n💰 Payment Required")
                print(f"Service '{service.name}' costs ${service_info.pricing} per request")
                print(f"Datasite: {service.datasite}")
                print(f"\nAccounting setup required for paid services.")
                
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
                    f"Service '{service.name}' requires payment (${service_info.pricing}) "
                    "but accounting is not configured"
                )
        
        # Create transaction token
        try:
            token = await self.rpc_client.create_transaction_token(service.datasite)
            logger.info(f"Payment authorized: ${service_info.pricing} to {service.datasite}")
            return token
        except Exception as e:
            raise PaymentError(f"Failed to create payment token: {e}")

    # Updated Service Usage Methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered services.
        
        Returns:
            Dictionary with service statistics
        """
        services = self.list_services(health_check="never")
        
        # Count by service type
        chat_services = [m for m in services if m.supports_service(ServiceType.CHAT)]
        search_services = [m for m in services if m.supports_service(ServiceType.SEARCH)]
        
        # Count by pricing
        free_services = [m for m in services if m.min_pricing == 0]
        paid_services = [m for m in services if m.min_pricing > 0]
        
        # Count by datasite
        datasites = {}
        for service in services:
            datasites[service.datasite] = datasites.get(service.datasite, 0) + 1
        
        return {
            "total_services": len(services),
            "enabled_services": len([m for m in services if m.has_enabled_services]),
            "disabled_services": len([m for m in services if not m.has_enabled_services]),
            "chat_services": len(chat_services),
            "search_services": len(search_services),
            "free_services": len(free_services),
            "paid_services": len(paid_services),
            "total_datasites": len(datasites),
            "avg_services_per_datasite": len(services) / len(datasites) if datasites else 0,
            "top_datasites": sorted(datasites.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def clear_cache(self):
        """Clear the service discovery cache."""
        self.scanner.clear_cache()
    
    # Private Helper Methods
    def _should_do_health_check(self, health_check: str, service_count: int) -> bool:
        """Determine if health checking should be performed."""
        if health_check == "always":
            return True
        elif health_check == "never":
            return False
        elif health_check == "auto":
            return service_count <= self.auto_health_check_threshold
        else:
            raise ValueError(f"Invalid health_check value: {health_check}")
    
    async def _add_health_status(self, services: List[ServiceInfo]) -> List[ServiceInfo]:
        """Add health status to services."""
        health_status = await batch_health_check(services, self.rpc_client, timeout=2.0)
        
        for service in services:
            service.health_status = health_status.get(service.name, HealthStatus.UNKNOWN)
        
        return services
    
    def _select_best_service(self, services: List[ServiceInfo], 
                          preference: QualityPreference) -> Optional[ServiceInfo]:
        """Select the best service based on preference."""
        if not services:
            return None
        
        if preference == QualityPreference.CHEAPEST:
            return min(services, key=lambda m: m.min_pricing)
        elif preference == QualityPreference.PREMIUM:
            return max(services, key=lambda m: m.min_pricing)
        elif preference == QualityPreference.FASTEST:
            # Prefer services with health status ONLINE, then by pricing
            online_services = [m for m in services if m.health_status == HealthStatus.ONLINE]
            if online_services:
                return min(online_services, key=lambda m: m.min_pricing)
            return services[0]  # Fallback to first service
        elif preference == QualityPreference.BALANCED:
            # Balance between cost and quality indicators
            def score_service(service: ServiceInfo) -> float:
                score = 0.0
                
                # Lower cost is better (inverse scoring)
                if service.min_pricing == 0:
                    score += 1.0  # Free is great
                else:
                    score += max(0, 1.0 - (service.min_pricing / 1.0))  # Diminishing returns
                
                # Health status bonus
                if service.health_status == HealthStatus.ONLINE:
                    score += 0.5
                
                # Quality tags bonus
                quality_tags = {'paid', 'gpt4', 'claude', 'high-quality', 'enterprise'}
                tag_matches = len(set(service.tags).intersection(quality_tags))
                score += tag_matches * 0.1
                
                # Multiple services bonus
                score += len(service.enabled_service_types) * 0.1
                
                return score
            
            return max(services, key=score_service)
        else:
            return services[0]
    
    def _service_to_dict(self, service: ServiceInfo) -> Dict[str, Any]:
        """Convert ServiceInfo to dictionary for JSON serialization."""
        return {
            "name": service.name,
            "datasite": service.datasite,
            "summary": service.summary,
            "description": service.description,
            "tags": service.tags,
            "services": [
                {
                    "type": service.type.value,
                    "enabled": service.enabled,
                    "pricing": service.pricing,
                    "charge_type": service.charge_type.value
                }
                for service in service.services
            ],
            "config_status": service.config_status.value,
            "health_status": service.health_status.value if service.health_status else None,
            "delegate_email": service.delegate_email,
            "min_pricing": service.min_pricing,
            "max_pricing": service.max_pricing
        }
    
    def _format_services_summary(self, services: List[ServiceInfo]) -> str:
        """Format services as a summary."""
        if not services:
            return "No services found."
        
        lines = [f"Found {len(services)} services:\n"]
        
        # Group by datasite
        by_datasite = {}
        for service in services:
            if service.datasite not in by_datasite:
                by_datasite[service.datasite] = []
            by_datasite[service.datasite].append(service)
        
        for datasite, datasite_services in sorted(by_datasite.items()):
            lines.append(f"📧 {datasite} ({len(datasite_services)} services)")
            
            for service in sorted(datasite_services, key=lambda m: m.name):
                services = ", ".join([s.type.value for s in service.services if s.enabled])
                pricing = f"${service.min_pricing}" if service.min_pricing > 0 else "Free"
                health = ""
                if service.health_status:
                    if service.health_status == HealthStatus.ONLINE:
                        health = " ✅"
                    elif service.health_status == HealthStatus.OFFLINE:
                        health = " ❌"
                    elif service.health_status == HealthStatus.TIMEOUT:
                        health = " ⏱️"
                
                lines.append(f"  • {service.name} ({services}) - {pricing}{health}")
            
            lines.append("")  # Empty line between datasites
        
        return "\n".join(lines)