"""
Filtering system for discovered services
"""
import re
from typing import List, Callable, Any, Optional, Set
from dataclasses import dataclass

from ..core.types import ServiceType, ServiceStatus, HealthStatus
from ..models.service_info import ServiceInfo

@dataclass
class FilterCriteria:
    """Criteria for filtering services."""
    
    # Basic filters
    name: Optional[str] = None
    name_pattern: Optional[str] = None  # Regex pattern
    datasite: Optional[str] = None
    datasite_pattern: Optional[str] = None  # Regex pattern
    
    # Service filters
    service_type: Optional[ServiceType] = None
    service_types: Optional[List[ServiceType]] = None
    has_all_services: Optional[List[ServiceType]] = None
    has_any_services: Optional[List[ServiceType]] = None
    
    # Tag filters
    tags: Optional[List[str]] = None
    has_all_tags: Optional[List[str]] = None
    has_any_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    
    # Pricing filters
    max_cost: Optional[float] = None
    min_cost: Optional[float] = None
    free_only: Optional[bool] = None
    paid_only: Optional[bool] = None
    
    # Status filters
    status: Optional[ServiceStatus] = None
    health_status: Optional[HealthStatus] = None
    enabled_only: bool = True
    
    # Advanced filters
    has_delegate: Optional[bool] = None
    delegate_email: Optional[str] = None


class ServiceFilter:
    """Flexible filtering system for services."""
    
    def __init__(self, criteria: Optional[FilterCriteria] = None):
        """Initialize filter with criteria.
        
        Args:
            criteria: Filter criteria to apply
        """
        self.criteria = criteria or FilterCriteria()
        self._custom_filters: List[Callable[[ServiceInfo], bool]] = []
    
    def add_custom_filter(self, filter_func: Callable[[ServiceInfo], bool]):
        """Add a custom filter function.
        
        Args:
            filter_func: Function that takes ServiceInfo and returns bool
        """
        self._custom_filters.append(filter_func)
    
    def filter_services(self, services: List[ServiceInfo]) -> List[ServiceInfo]:
        """Apply all filters to a list of services.
        
        Args:
            services: List of services to filter
            
        Returns:
            Filtered list of services
        """
        filtered_services = []
        
        for service in services:
            if self._passes_all_filters(service):
                filtered_services.append(service)
        
        return filtered_services
    
    def _passes_all_filters(self, service: ServiceInfo) -> bool:
        """Check if a service passes all filter criteria.
        
        Args:
            service: Service to check
            
        Returns:
            True if service passes all filters
        """
        # Basic name filters
        if not self._check_name_filters(service):
            return False
        
        # Datasite filters
        if not self._check_datasite_filters(service):
            return False
        
        # Service filters
        if not self._check_service_filters(service):
            return False
        
        # Tag filters
        if not self._check_tag_filters(service):
            return False
        
        # Pricing filters
        if not self._check_pricing_filters(service):
            return False
        
        # Status filters
        if not self._check_status_filters(service):
            return False
        
        # Advanced filters
        if not self._check_advanced_filters(service):
            return False
        
        # Custom filters
        for custom_filter in self._custom_filters:
            if not custom_filter(service):
                return False
        
        return True
    
    def _check_name_filters(self, service: ServiceInfo) -> bool:
        """Check name-based filters."""
        if self.criteria.name and service.name != self.criteria.name:
            return False
        
        if self.criteria.name_pattern:
            if not re.search(self.criteria.name_pattern, service.name, re.IGNORECASE):
                return False
        
        return True
    
    def _check_datasite_filters(self, service: ServiceInfo) -> bool:
        """Check datasite-based filters."""
        if self.criteria.datasite and service.datasite != self.criteria.datasite:
            return False
        
        if self.criteria.datasite_pattern:
            if not re.search(self.criteria.datasite_pattern, service.datasite, re.IGNORECASE):
                return False
        
        return True
    
    def _check_service_filters(self, service: ServiceInfo) -> bool:
        """Check service-based filters."""
        # Enabled only filter
        if self.criteria.enabled_only and not service.has_enabled_services:
            return False
        
        # Single service type filter
        if self.criteria.service_type:
            if not service.supports_service(self.criteria.service_type):
                return False
        
        # Multiple service types (OR logic)
        if self.criteria.service_types:
            if not any(service.supports_service(st) for st in self.criteria.service_types):
                return False
        
        # Has all services (AND logic)
        if self.criteria.has_all_services:
            if not all(service.supports_service(st) for st in self.criteria.has_all_services):
                return False
        
        # Has any services (OR logic)
        if self.criteria.has_any_services:
            if not any(service.supports_service(st) for st in self.criteria.has_any_services):
                return False
        
        return True
    
    # def _check_service_filters(self, service: ServiceInfo) -> bool:
    #     """Check service-based filters."""
    #     # This single check handles all enabled/disabled logic.
    #     if self.criteria.enabled_only and not service.has_enabled_services:
    #         return False

    #     # Single service type filter
    #     if self.criteria.service_type:
    #         # Check if the service supports the service and if it's enabled (as a secondary check)
    #         service = service.get_service_info(self.criteria.service_type)
    #         if service is None or not service.enabled:
    #             return False

    #     return True

    def _check_tag_filters(self, service: ServiceInfo) -> bool:
        """Check tag-based filters."""
        service_tags = set(tag.lower() for tag in service.tags)
        
        # Simple tags filter (backward compatibility)
        if self.criteria.tags:
            filter_tags = set(tag.lower() for tag in self.criteria.tags)
            if not filter_tags.intersection(service_tags):
                return False
        
        # Has all tags (AND logic)
        if self.criteria.has_all_tags:
            required_tags = set(tag.lower() for tag in self.criteria.has_all_tags)
            if not required_tags.issubset(service_tags):
                return False
        
        # Has any tags (OR logic)
        if self.criteria.has_any_tags:
            any_tags = set(tag.lower() for tag in self.criteria.has_any_tags)
            if not any_tags.intersection(service_tags):
                return False
        
        # Exclude tags
        if self.criteria.exclude_tags:
            exclude_tags = set(tag.lower() for tag in self.criteria.exclude_tags)
            if exclude_tags.intersection(service_tags):
                return False
        
        return True

    # def _check_tag_filters(self, service: ServiceInfo) -> bool:
    #     """Check tag-based filters."""
    #     service_tags = set(tag.lower() for tag in service.tags)
        
    #     # Has any tags (OR logic)
    #     # This single check handles the "tags" filter and is the correct approach.
    #     if self.criteria.has_any_tags:
    #         any_tags = set(tag.lower() for tag in self.criteria.has_any_tags)
    #         if not any_tags.intersection(service_tags):
    #             return False
        
    #     # The rest of the checks are fine as they are.
    #     # Has all tags (AND logic)
    #     if self.criteria.has_all_tags:
    #         required_tags = set(tag.lower() for tag in self.criteria.has_all_tags)
    #         if not required_tags.issubset(service_tags):
    #             return False
        
    #     # Exclude tags
    #     if self.criteria.exclude_tags:
    #         exclude_tags = set(tag.lower() for tag in self.criteria.exclude_tags)
    #         if exclude_tags.intersection(service_tags):
    #             return False
        
    #     return True

    def _check_pricing_filters(self, service: ServiceInfo) -> bool:
        """Check pricing-based filters."""
        min_pricing = service.min_pricing
        max_pricing = service.max_pricing
        
        # Free only
        if self.criteria.free_only and max_pricing > 0:
            return False
        
        # Paid only
        if self.criteria.paid_only and max_pricing == 0:
            return False
        
        # Max cost filter (use minimum pricing of service)
        if self.criteria.max_cost is not None and min_pricing > self.criteria.max_cost:
            return False
        
        # Min cost filter (use minimum pricing of service)
        if self.criteria.min_cost is not None and min_pricing < self.criteria.min_cost:
            return False
        
        return True
    
    def _check_status_filters(self, service: ServiceInfo) -> bool:
        """Check status-based filters."""
        if self.criteria.status and service.config_status != self.criteria.status:
            return False
        
        if self.criteria.health_status and service.health_status != self.criteria.health_status:
            return False
        
        return True
    
    def _check_advanced_filters(self, service: ServiceInfo) -> bool:
        """Check advanced filters."""
        if self.criteria.has_delegate is not None:
            has_delegate = service.delegate_email is not None
            if self.criteria.has_delegate != has_delegate:
                return False
        
        if self.criteria.delegate_email and service.delegate_email != self.criteria.delegate_email:
            return False
        
        return True


# Convenience filter builders
class FilterBuilder:
    """Builder pattern for creating service filters."""
    
    def __init__(self):
        self.criteria = FilterCriteria()
    
    def by_name(self, name: str) -> 'FilterBuilder':
        """Filter by exact service name."""
        self.criteria.name = name
        return self
    
    def by_name_pattern(self, pattern: str) -> 'FilterBuilder':
        """Filter by service name regex pattern."""
        self.criteria.name_pattern = pattern
        return self
    
    def by_datasite(self, datasite: str) -> 'FilterBuilder':
        """Filter by exact datasite email."""
        self.criteria.datasite = datasite
        return self
    
    def by_datasite_pattern(self, pattern: str) -> 'FilterBuilder':
        """Filter by datasite email regex pattern."""
        self.criteria.datasite_pattern = pattern
        return self
    
    def by_service_type(self, service_type: ServiceType) -> 'FilterBuilder':
        """Filter by single service type."""
        self.criteria.service_type = service_type
        return self
    
    def by_service_types(self, service_types: List[ServiceType]) -> 'FilterBuilder':
        """Filter by multiple service types (OR logic)."""
        self.criteria.service_types = service_types
        return self
    
    def requires_all_services(self, service_types: List[ServiceType]) -> 'FilterBuilder':
        """Require all specified services (AND logic)."""
        self.criteria.has_all_services = service_types
        return self
    
    def by_tags(self, tags: List[str], match_all: bool = False) -> 'FilterBuilder':
        """Filter by tags.
        
        Args:
            tags: List of tags to match
            match_all: If True, service must have ALL tags; if False, ANY tag
        """
        if match_all:
            self.criteria.has_all_tags = tags
        else:
            self.criteria.has_any_tags = tags
        return self
    
    def exclude_tags(self, tags: List[str]) -> 'FilterBuilder':
        """Exclude services with any of the specified tags."""
        self.criteria.exclude_tags = tags
        return self
    
    def by_max_cost(self, max_cost: float) -> 'FilterBuilder':
        """Filter by maximum cost."""
        self.criteria.max_cost = max_cost
        return self
    
    def by_min_cost(self, min_cost: float) -> 'FilterBuilder':
        """Filter by minimum cost."""
        self.criteria.min_cost = min_cost
        return self
    
    def free_only(self) -> 'FilterBuilder':
        """Only include free services."""
        self.criteria.free_only = True
        return self
    
    def paid_only(self) -> 'FilterBuilder':
        """Only include paid services."""
        self.criteria.paid_only = True
        return self
    
    def by_status(self, status: ServiceStatus) -> 'FilterBuilder':
        """Filter by configuration status."""
        self.criteria.status = status
        return self
    
    def by_health_status(self, health_status: HealthStatus) -> 'FilterBuilder':
        """Filter by health status."""
        self.criteria.health_status = health_status
        return self
    
    # def include_disabled(self) -> 'FilterBuilder':
    #     """Include services with disabled services."""
    #     self.criteria.enabled_only = False
    #     return self
    
    def with_delegate(self) -> 'FilterBuilder':
        """Only include services with delegates."""
        self.criteria.has_delegate = True
        return self
    
    def without_delegate(self) -> 'FilterBuilder':
        """Only include services without delegates."""
        self.criteria.has_delegate = False
        return self
    
    def by_delegate(self, delegate_email: str) -> 'FilterBuilder':
        """Filter by specific delegate email."""
        self.criteria.delegate_email = delegate_email
        return self
    
    def build(self) -> ServiceFilter:
        """Build the filter."""
        return ServiceFilter(self.criteria)


# Predefined common filters
def create_chat_services_filter(max_cost: Optional[float] = None) -> ServiceFilter:
    """Create filter for chat services."""
    builder = FilterBuilder().by_service_type(ServiceType.CHAT)
    if max_cost is not None:
        builder = builder.by_max_cost(max_cost)
    return builder.build()


def create_search_services_filter(max_cost: Optional[float] = None) -> ServiceFilter:
    """Create filter for search services."""
    builder = FilterBuilder().by_service_type(ServiceType.SEARCH)
    if max_cost is not None:
        builder = builder.by_max_cost(max_cost)
    return builder.build()


def create_free_services_filter() -> ServiceFilter:
    """Create filter for free services."""
    return FilterBuilder().free_only().build()


def create_paid_services_filter() -> ServiceFilter:
    """Create filter for paid services."""
    return FilterBuilder().paid_only().build()


def create_healthy_services_filter() -> ServiceFilter:
    """Create filter for healthy services."""
    return FilterBuilder().by_health_status(HealthStatus.ONLINE).build()


def create_datasite_services_filter(datasite: str) -> ServiceFilter:
    """Create filter for specific datasite."""
    return FilterBuilder().by_datasite(datasite).build()


def create_tag_services_filter(tags: List[str], match_all: bool = False) -> ServiceFilter:
    """Create filter for specific tags."""
    return FilterBuilder().by_tags(tags, match_all).build()


def combine_filters(filters: List[ServiceFilter]) -> ServiceFilter:
    """Combine multiple filters into one (AND logic).
    
    Args:
        filters: List of filters to combine
        
    Returns:
        Combined filter
    """
    combined_filter = ServiceFilter()
    
    def combined_check(service: ServiceInfo) -> bool:
        return all(f._passes_all_filters(service) for f in filters)
    
    combined_filter.add_custom_filter(combined_check)
    return combined_filter