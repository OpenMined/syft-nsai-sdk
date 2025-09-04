"""
Main SyftBox NSAI SDK client
"""
import os
import asyncio
import asyncio
import nest_asyncio
import logging

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

from .core import Service
from .core import Pipeline
from .core.decorators import require_account
from .core.config import ConfigManager
from .core.types import (
    ServiceInfo,
    ServiceSpec, 
    ServiceType,
    HealthStatus, 
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
    SyftBoxNotRunningError, 
    ServiceNotSupportedError,
    ServiceNotFoundError,
    ValidationError,
    raise_service_not_found, 
    raise_service_not_supported
)
from .discovery.scanner import ServiceScanner, FastScanner
from .discovery.parser import MetadataParser
from .discovery.filters import ServiceFilter, FilterCriteria, FilterBuilder
from .clients.rpc_client import SyftBoxRPCClient
from .clients.accounting_client import AccountingClient
from .services.chat import ChatService
from .services.search import SearchService
from .services.health import check_service_health, batch_health_check, HealthMonitor
from .models.services_list import ServicesList
from .utils.formatting import format_services_table, format_service_details

logger = logging.getLogger(__name__)

nest_asyncio.apply()
load_dotenv()

class Client:
    """Main client for discovering and using SyftBox AI services."""
    
    def __init__(self, 
                syftbox_config_path: Optional[Path] = None,
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
            raise SyftBoxNotRunningError(self.config_manager.get_startup_instructions())
        
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
            # Inlined _setup_default_accounting logic
            client, is_configured = AccountingClient.setup_accounting_discovery()
            
            if is_configured:
                self._account_configured = True
                logger.info(f"Found existing accounting credentials for {client.get_email()}")
            else:
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

            self.accounting_client = client
        
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
                filtered_services = asyncio.run(self._add_health_status(filtered_services))
                # filtered_services = asyncio.run(self._add_health_status(filtered_services))
                # import IPython
                # ipython = IPython.get_ipython()
                # if ipython is not None and hasattr(ipython, 'kernel'):
                #     # We're in Jupyter, skip health check to avoid asyncio issues
                #     logger.info("Skipping health check in Jupyter environment to avoid asyncio issues")
                # else:
                #     # Not in Jupyter, safe to use asyncio.run()
                #     filtered_services = asyncio.run(self._add_health_status(filtered_services))
            except ImportError:
                # IPython not available, safe to use asyncio.run()
                filtered_services = asyncio.run(self._add_health_status(filtered_services))
            except Exception as e:
                # Any other error, log and continue without health check
                logger.warning(f"Health check failed: {e}. Continuing without health status.")
        
        logger.info(f"Discovered {len(filtered_services)} services (health_check={should_health_check})")
        return ServicesList(filtered_services, self)
    
    def get_service1(self, service_name: str, datasite: Optional[str] = None) -> Optional[ServiceInfo]:
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
    
    def get_service2(self, service_name: str) -> Optional[ServiceInfo]:
        """Find a specific service by name.
        
        Args:
            service_name: Name of the service to find
            
        Returns:
            ServiceInfo if found, None otherwise
        """
        if "/" in service_name:
            datasite, name = service_name.split("/", 1)
            # services = self.list_services(name=service_name, datasite=datasite, health_check="never")
        
        # Find exact match
        # for service in services:
        #     if service.name == service_name:
        #         if datasite is None or service.datasite == datasite:
        #             return service
        
        # Try direct lookup
        service = self._lookup_service_direct(name, datasite)
        if service:
            return service
        
        # Handle not found with helpful errors
        if datasite:
            raise ServiceNotFoundError(f"Service '{service_name}' not found for datasite '{datasite}'")
        else:
            # Check for ambiguous names
            similar_services = self._find_services_by_name(service_name)  # Minimal scan
            if len(similar_services) > 1:
                datasites = [s.datasite for s in similar_services]
                raise ValidationError(
                    f"Multiple services named '{service_name}' found. "
                    f"Please specify datasite. Available datasites: {', '.join(datasites)}"
                )
            else:
                raise ServiceNotFoundError(f"Service '{service_name}' not found!")
    
    def get_service(self, service_name: str) -> ServiceInfo:
        datasite, name = service_name.split("/", 1)
        metadata_path = self.scanner.get_service_path(datasite, name)
        
        if not metadata_path:
            raise ServiceNotFoundError(f"'{service_name}'")
        
        return self.parser.parse_service_from_files(metadata_path)

    async def get_service_async(self, service_name: str) -> ServiceInfo:
        datasite, name = service_name.split("/", 1)
        metadata_path = self.scanner.get_service_path(datasite, name)
        
        if not metadata_path:
            raise ServiceNotFoundError(f"'{service_name}'")
        
        return self.parser.parse_service_from_files(metadata_path)

    # Service Usage Methods
    @require_account
    def chat1(self,
            service_name: str,
            messages: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
        ) -> ChatResponse:
        """Chat with a specific service.
        
        Args:
            service_name: Name of the service to use
            prompt: Message to send
            datasite: Datasite email (required if service name is ambiguous)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional service-specific parameters
            
        Returns:
            Chat response from the specified service
        """
        # Find the specific service
        [datasite, name] = service_name.split("/")
        service = self.get_service(service_name)
        if not service:
            if datasite:
                # raise ServiceNotFoundError(f"Service '{service_name}' not found!")
                raise ServiceNotFoundError(f"Service '{service_name}' not found for datasite '{datasite}'")

            else:
                # Show available services with same name
                similar_services = [m for m in self.list_services() if m.name == name]
                if len(similar_services) > 1:
                    datasites = [m.datasite for m in similar_services]
                    raise ValidationError(
                        f"Multiple services named '{service_name}' found. "
                        f"Please specify datasite. Available datasites: {', '.join(datasites)}"
                    )
                else:
                    raise ServiceNotFoundError(f"Service '{service_name}' not found!")
        
        # Check if service supports chat
        if not service.supports_service(ServiceType.CHAT):
            raise_service_not_supported(service.name, "chat", service)
            # raise ValidationError(f"Service '{service_name}' does not support chat service")
        
        # Build request parameters
        chat_params = {
            "messages": messages,
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
    def chat(self,
            service_name: str,
            messages: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
        ) -> ChatResponse:
        """Chat with a specific service.
        
        Args:
            service_name: Datasite/Name of the service to use
            messages: Message to send
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional service-specific parameters
            
        Returns:
            Chat response from the specified service
        """
        
        # Find the specific service
        service = self.get_service(service_name)
        logger.info(f"Using service: {service.name} from datasite: {service.datasite}") 
        
        # Validate service supports chat
        if not service.supports_service(ServiceType.CHAT):
            raise_service_not_supported(service.name, "chat", service)
        
        # Build request parameters
        chat_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Remove None values
        chat_params = {k: v for k, v in chat_params.items() if v is not None}
        
        # Execute chat
        chat_service = ChatService(service, self.rpc_client)
        return asyncio.run(chat_service.chat_with_params(chat_params))

    @require_account
    def search(self,
                    service_name: str, 
                    message: str,
                    topK: Optional[int] = None,
                    similarity_threshold: Optional[float] = None,
                    **kwargs) -> SearchResponse:
        """Search with a specific service.
        
        Args:
            service_name: Datasite/Name of the service to use
            message: Search message
            topK: Maximum number of results
            similarity_threshold: Minimum similarity score
            **kwargs: Additional service-specific parameters
            
        Returns:
            Search response from the specified service
        """

        # Find the specific service
        service = self.get_service(service_name)
        logger.info(f"Using service: {service.name} from datasite: {service.datasite}") 
        
        # Validate service supports search
        if not service.supports_service(ServiceType.SEARCH):
            raise_service_not_supported(service.name, "search", service)
        
        # Build request parameters
        search_params = {
            "message": message,
            "topK": topK,
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
                   **kwargs) -> ChatResponse:
        """Chat with a specific service.
        
        Args:
            service_name: Name of the service to use (REQUIRED)
            prompt: Message to send
            datasite: Datasite email (required if service name is ambiguous)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
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
    def get_parameters(self, service_name: str, datasite: Optional[str] = None) -> Dict[str, Any]:
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
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balance."""
        if not self.is_accounting_configured():
            return {"error": "Accounting not configured"}
        
        try:
            return asyncio.run(self.accounting_client.get_account_info())
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
        
    async def get_account_info_async(self) -> Dict[str, Any]:
        """Get account information and balance."""
        if not self.is_accounting_configured():
            return {"error": "Accounting not configured"}
        
        try:
            return await self.accounting_client.get_account_info()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}

    async def show_accounting_status_async(self) -> str:
        """Show current accounting configuration status."""
        if not self.is_accounting_configured():
            return (
                "Accounting not configured\n"
                "   Use client.setup_accounting() to configure payment services\n"
                "   Currently limited to free services only"
            )
        
        try:
            account_info = await self.get_account_info_async()

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
        
    async def _ensure_payment_setup1(self, service: ServiceInfo) -> Optional[str]:
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
        
        # Early return for free services - skip all accounting logic entirely
        if not service_info or service_info.pricing == 0:
            return None  # Free service
        
        # Service requires payment - ensure accounting is set up
        if not self.is_accounting_configured():
            if self.auto_setup_accounting:
                print(f"\nPayment Required")
                print(f"Service '{service.name}' costs ${service_info.pricing} per request")
                print(f"Datasite: {service.datasite}")
                print(f"\nAccounting setup required for paid services.")
                
                try:
                    response = input("Would you like to set up accounting now? (y/n): ").lower().strip()
                    if response in ['y', 'yes']:
                        # Interactive setup would go here
                        print("Please use below to configure:\n")
                        print("     await client.register_accounting_async(email, password) to configure.\n")
                        print("Or:\n")
                        print("     client.register_accounting(email, password) to configure.")
                        return None
                    else:
                        print("Payment setup skipped.")
                        return None
                except (EOFError, KeyboardInterrupt):
                    print("\nPayment setup cancelled.")
                    return None
            else:
                from .core.exceptions import PaymentError
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
            from .core.exceptions import PaymentError
            raise PaymentError(f"Failed to create payment token: {e}")

    # Updated Service Usage Methods
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

    # Service loader
    def load(self, service_name: str) -> Service:
        """Load a service by name and return Service object.
        
        Args:
            service_name: Full service name in format 'datasite/service_name'
            
        Returns:
            Service object for object-oriented interaction
            
        Example:
            service = client.load("alice@example.com/gpt-assistant")
            response = service.chat(messages=[{"role": "user", "content": "Hello"}])
        """
        service_info = self.get_service(service_name)
        return Service(service_info, self)

    # Private methods to support RAG coordination
    def _normalize_service_specs(self, services: Union[str, Service, List[Union[str, Service, Dict[str, Any]]]]) -> List[ServiceSpec]:
        """Normalize various service specification formats including Service objects.
        
        Handles:
        - Strings: "alice@example.com/docs"
        - Service objects: service_obj
        - Dicts with strings: {"name": "alice@example.com/docs", "topK": 5}
        - Dicts with Service objects: {"name": service_obj, "topK": 5}
        
        Args:
            services: Services in various formats
            
        Returns:
            List of ServiceSpec objects
        """
        if isinstance(services, str):
            # Single string service name
            return [ServiceSpec(name=services, params={})]
        
        elif hasattr(services, 'full_name'):  # Service object
            # Single Service object
            return [ServiceSpec(name=services.full_name, params={})]
        
        elif isinstance(services, list):
            normalized = []
            for service in services:
                if isinstance(service, str):
                    # String service name
                    normalized.append(ServiceSpec(name=service, params={}))
                    
                elif hasattr(service, 'full_name'):  # Service object
                    # Service object
                    normalized.append(ServiceSpec(name=service.full_name, params={}))
                    
                elif isinstance(service, dict):
                    # Dictionary format - could contain string or Service object
                    if "name" not in service:
                        raise ValidationError(f"Service spec missing 'name' key: {service}")
                    
                    name_value = service["name"]
                    params = {k: v for k, v in service.items() if k != "name"}
                    
                    if isinstance(name_value, str):
                        # Dict with string name
                        normalized.append(ServiceSpec(name=name_value, params=params))
                    elif hasattr(name_value, 'full_name'):  # Service object
                        # Dict with Service object
                        normalized.append(ServiceSpec(name=name_value.full_name, params=params))
                    else:
                        raise ValidationError(f"Service 'name' must be string or Service object, got {type(name_value)}")
                        
                else:
                    raise ValidationError(f"Invalid service specification: {service} (type: {type(service)})")
            
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
    
    # RAG pipeline methods
    def create_pipeline(self) -> Pipeline:
        """Create a new pipeline for RAG workflows"""
        return Pipeline(client=self)

    def pipeline(self, data_sources: Optional[List[Union[str, Dict]]] = None, 
                synthesizer: Optional[Union[str, Dict]] = None, 
                context_format: Optional[str] = None) -> Pipeline:
        """Create and configure a pipeline in one call (inline approach)
        
        Args:
            data_sources: List of search services to use as data sources
            synthesizer: Chat service to use for synthesis
            context_format: Format for search context ("simple" or "frontend")
        
        Returns:
            Configured Pipeline ready for execution
        
        Example:
            result = client.pipeline(
                data_sources=["alice@example.com/docs", "bob@example.com/wiki"],
                synthesizer="ai@openai.com/gpt-4"
            ).run(messages=[{"role": "user", "content": "What is Python?"}])
        """
        return Pipeline(
            client=self, 
            data_sources=data_sources, 
            synthesizer=synthesizer,
            context_format=context_format or "simple"
        )
    
    def pipeline_mixed(self, data_sources=None, synthesizer=None, context_format=None):
        """Create and configure a pipeline with mixed input types (COMPLEX VERSION).
        
        Handles multiple input formats:
        - Strings: "alice@example.com/docs"
        - Service objects: service_obj
        - Dicts with strings: {"name": "alice@example.com/docs", "topK": 5}
        - Dicts with Service objects: {"name": service_obj, "topK": 10}
        
        Args:
            data_sources: Mixed list of strings, Service objects, or dicts
            synthesizer: String, Service object, or dict
            context_format: Format for search context ("simple" or "frontend")
            
        Returns:
            Configured Pipeline ready for execution
            
        Example:
            docs = client.load("alice@example.com/docs")
            result = client.pipeline_mixed(
                data_sources=[
                    "alice@example.com/docs",              # String
                    docs,                                  # Service object  
                    {"name": "bob@example.com/wiki"},      # Dict with string
                    {"name": docs, "topK": 10}             # Dict with Service object
                ],
                synthesizer=docs  # Service object
            ).run(messages=[{"role": "user", "content": "What is Python?"}])
        """
        normalized_sources = []
        if data_sources:
            specs = self._normalize_service_specs(data_sources)
            # Convert ServiceSpec objects to dict format for Pipeline
            normalized_sources = []
            for spec in specs:
                source_dict = {"name": spec.name}
                source_dict.update(spec.params)
                normalized_sources.append(source_dict)
        
        normalized_synthesizer = None
        if synthesizer:
            specs = self._normalize_service_specs([synthesizer])
            spec = specs[0]
            normalized_synthesizer = {"name": spec.name, **spec.params}
        
        return Pipeline(
            client=self,
            data_sources=normalized_sources,
            synthesizer=normalized_synthesizer,
            context_format=context_format or "simple"
        )
    
    def pipeline_from_services(self, data_sources: List[Service], synthesizer: Service, context_format: str = "simple"):
        """Create pipeline from Service objects (object-oriented approach).
        
        Creates a basic pipeline using Service objects without additional parameters.
        For complex parameter configuration, use create_pipeline() with add_source() method.
        
        Args:
            data_sources: List of Service objects to use as data sources
            synthesizer: Service object to use for synthesis
            context_format: Format for search context ("simple" or "frontend")
            
        Returns:
            Configured Pipeline ready for execution
            
        Example:
            # Basic usage
            docs = client.load("alice@example.com/docs")
            wiki = client.load("bob@example.com/wiki")
            gpt = client.load("ai@openai.com/gpt-4")
            
            pipeline = client.pipeline_from_services(
                data_sources=[docs, wiki],
                synthesizer=gpt
            )
            result = pipeline.run(messages=[{"role": "user", "content": "What is Python?"}])
            
            # For parameters, use the method-based approach:
            pipeline = client.create_pipeline()
            pipeline.add_source(docs, topK=5)
            pipeline.add_source(wiki, topK=8)
            pipeline.set_synthesizer(gpt, temperature=0.7)
        """
        # Convert Service objects to the format Pipeline expects
        data_source_specs = [{"name": service.full_name} for service in data_sources]
        synthesizer_spec = {"name": synthesizer.full_name}
        
        return Pipeline(
            client=self, 
            data_sources=data_source_specs, 
            synthesizer=synthesizer_spec, 
            context_format=context_format
        )