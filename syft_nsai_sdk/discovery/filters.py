"""
Filtering system for discovered models
"""
from typing import List, Callable, Any, Optional, Set
import re
from dataclasses import dataclass

from ..core.types import ModelInfo, ServiceType, ModelStatus, HealthStatus


@dataclass
class FilterCriteria:
    """Criteria for filtering models."""
    
    # Basic filters
    name: Optional[str] = None
    name_pattern: Optional[str] = None  # Regex pattern
    owner: Optional[str] = None
    owner_pattern: Optional[str] = None  # Regex pattern
    
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
    status: Optional[ModelStatus] = None
    health_status: Optional[HealthStatus] = None
    enabled_only: bool = True
    
    # Advanced filters
    has_delegate: Optional[bool] = None
    delegate_email: Optional[str] = None


class ModelFilter:
    """Flexible filtering system for models."""
    
    def __init__(self, criteria: Optional[FilterCriteria] = None):
        """Initialize filter with criteria.
        
        Args:
            criteria: Filter criteria to apply
        """
        self.criteria = criteria or FilterCriteria()
        self._custom_filters: List[Callable[[ModelInfo], bool]] = []
    
    def add_custom_filter(self, filter_func: Callable[[ModelInfo], bool]):
        """Add a custom filter function.
        
        Args:
            filter_func: Function that takes ModelInfo and returns bool
        """
        self._custom_filters.append(filter_func)
    
    def filter_models(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Apply all filters to a list of models.
        
        Args:
            models: List of models to filter
            
        Returns:
            Filtered list of models
        """
        filtered_models = []
        
        for model in models:
            if self._passes_all_filters(model):
                filtered_models.append(model)
        
        return filtered_models
    
    def _passes_all_filters(self, model: ModelInfo) -> bool:
        """Check if a model passes all filter criteria.
        
        Args:
            model: Model to check
            
        Returns:
            True if model passes all filters
        """
        # Basic name filters
        if not self._check_name_filters(model):
            return False
        
        # Owner filters
        if not self._check_owner_filters(model):
            return False
        
        # Service filters
        if not self._check_service_filters(model):
            return False
        
        # Tag filters
        if not self._check_tag_filters(model):
            return False
        
        # Pricing filters
        if not self._check_pricing_filters(model):
            return False
        
        # Status filters
        if not self._check_status_filters(model):
            return False
        
        # Advanced filters
        if not self._check_advanced_filters(model):
            return False
        
        # Custom filters
        for custom_filter in self._custom_filters:
            if not custom_filter(model):
                return False
        
        return True
    
    def _check_name_filters(self, model: ModelInfo) -> bool:
        """Check name-based filters."""
        if self.criteria.name and model.name != self.criteria.name:
            return False
        
        if self.criteria.name_pattern:
            if not re.search(self.criteria.name_pattern, model.name, re.IGNORECASE):
                return False
        
        return True
    
    def _check_owner_filters(self, model: ModelInfo) -> bool:
        """Check owner-based filters."""
        if self.criteria.owner and model.owner != self.criteria.owner:
            return False
        
        if self.criteria.owner_pattern:
            if not re.search(self.criteria.owner_pattern, model.owner, re.IGNORECASE):
                return False
        
        return True
    
    def _check_service_filters(self, model: ModelInfo) -> bool:
        """Check service-based filters."""
        # Enabled only filter
        if self.criteria.enabled_only and not model.has_enabled_services:
            return False
        
        # Single service type filter
        if self.criteria.service_type:
            if not model.supports_service(self.criteria.service_type):
                return False
        
        # Multiple service types (OR logic)
        if self.criteria.service_types:
            if not any(model.supports_service(st) for st in self.criteria.service_types):
                return False
        
        # Has all services (AND logic)
        if self.criteria.has_all_services:
            if not all(model.supports_service(st) for st in self.criteria.has_all_services):
                return False
        
        # Has any services (OR logic)
        if self.criteria.has_any_services:
            if not any(model.supports_service(st) for st in self.criteria.has_any_services):
                return False
        
        return True
    
    # def _check_service_filters(self, model: ModelInfo) -> bool:
    #     """Check service-based filters."""
    #     # This single check handles all enabled/disabled logic.
    #     if self.criteria.enabled_only and not model.has_enabled_services:
    #         return False

    #     # Single service type filter
    #     if self.criteria.service_type:
    #         # Check if the model supports the service and if it's enabled (as a secondary check)
    #         service = model.get_service_info(self.criteria.service_type)
    #         if service is None or not service.enabled:
    #             return False

    #     return True

    def _check_tag_filters(self, model: ModelInfo) -> bool:
        """Check tag-based filters."""
        model_tags = set(tag.lower() for tag in model.tags)
        
        # Simple tags filter (backward compatibility)
        if self.criteria.tags:
            filter_tags = set(tag.lower() for tag in self.criteria.tags)
            if not filter_tags.intersection(model_tags):
                return False
        
        # Has all tags (AND logic)
        if self.criteria.has_all_tags:
            required_tags = set(tag.lower() for tag in self.criteria.has_all_tags)
            if not required_tags.issubset(model_tags):
                return False
        
        # Has any tags (OR logic)
        if self.criteria.has_any_tags:
            any_tags = set(tag.lower() for tag in self.criteria.has_any_tags)
            if not any_tags.intersection(model_tags):
                return False
        
        # Exclude tags
        if self.criteria.exclude_tags:
            exclude_tags = set(tag.lower() for tag in self.criteria.exclude_tags)
            if exclude_tags.intersection(model_tags):
                return False
        
        return True

    # def _check_tag_filters(self, model: ModelInfo) -> bool:
    #     """Check tag-based filters."""
    #     model_tags = set(tag.lower() for tag in model.tags)
        
    #     # Has any tags (OR logic)
    #     # This single check handles the "tags" filter and is the correct approach.
    #     if self.criteria.has_any_tags:
    #         any_tags = set(tag.lower() for tag in self.criteria.has_any_tags)
    #         if not any_tags.intersection(model_tags):
    #             return False
        
    #     # The rest of the checks are fine as they are.
    #     # Has all tags (AND logic)
    #     if self.criteria.has_all_tags:
    #         required_tags = set(tag.lower() for tag in self.criteria.has_all_tags)
    #         if not required_tags.issubset(model_tags):
    #             return False
        
    #     # Exclude tags
    #     if self.criteria.exclude_tags:
    #         exclude_tags = set(tag.lower() for tag in self.criteria.exclude_tags)
    #         if exclude_tags.intersection(model_tags):
    #             return False
        
    #     return True

    def _check_pricing_filters(self, model: ModelInfo) -> bool:
        """Check pricing-based filters."""
        min_pricing = model.min_pricing
        max_pricing = model.max_pricing
        
        # Free only
        if self.criteria.free_only and max_pricing > 0:
            return False
        
        # Paid only
        if self.criteria.paid_only and max_pricing == 0:
            return False
        
        # Max cost filter (use minimum pricing of model)
        if self.criteria.max_cost is not None and min_pricing > self.criteria.max_cost:
            return False
        
        # Min cost filter (use minimum pricing of model)
        if self.criteria.min_cost is not None and min_pricing < self.criteria.min_cost:
            return False
        
        return True
    
    def _check_status_filters(self, model: ModelInfo) -> bool:
        """Check status-based filters."""
        if self.criteria.status and model.config_status != self.criteria.status:
            return False
        
        if self.criteria.health_status and model.health_status != self.criteria.health_status:
            return False
        
        return True
    
    def _check_advanced_filters(self, model: ModelInfo) -> bool:
        """Check advanced filters."""
        if self.criteria.has_delegate is not None:
            has_delegate = model.delegate_email is not None
            if self.criteria.has_delegate != has_delegate:
                return False
        
        if self.criteria.delegate_email and model.delegate_email != self.criteria.delegate_email:
            return False
        
        return True


# Convenience filter builders
class FilterBuilder:
    """Builder pattern for creating model filters."""
    
    def __init__(self):
        self.criteria = FilterCriteria()
    
    def by_name(self, name: str) -> 'FilterBuilder':
        """Filter by exact model name."""
        self.criteria.name = name
        return self
    
    def by_name_pattern(self, pattern: str) -> 'FilterBuilder':
        """Filter by model name regex pattern."""
        self.criteria.name_pattern = pattern
        return self
    
    def by_owner(self, owner: str) -> 'FilterBuilder':
        """Filter by exact owner email."""
        self.criteria.owner = owner
        return self
    
    def by_owner_pattern(self, pattern: str) -> 'FilterBuilder':
        """Filter by owner email regex pattern."""
        self.criteria.owner_pattern = pattern
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
            match_all: If True, model must have ALL tags; if False, ANY tag
        """
        if match_all:
            self.criteria.has_all_tags = tags
        else:
            self.criteria.has_any_tags = tags
        return self
    
    def exclude_tags(self, tags: List[str]) -> 'FilterBuilder':
        """Exclude models with any of the specified tags."""
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
        """Only include free models."""
        self.criteria.free_only = True
        return self
    
    def paid_only(self) -> 'FilterBuilder':
        """Only include paid models."""
        self.criteria.paid_only = True
        return self
    
    def by_status(self, status: ModelStatus) -> 'FilterBuilder':
        """Filter by configuration status."""
        self.criteria.status = status
        return self
    
    def by_health_status(self, health_status: HealthStatus) -> 'FilterBuilder':
        """Filter by health status."""
        self.criteria.health_status = health_status
        return self
    
    def include_disabled(self) -> 'FilterBuilder':
        """Include models with disabled services."""
        self.criteria.enabled_only = False
        return self
    
    def with_delegate(self) -> 'FilterBuilder':
        """Only include models with delegates."""
        self.criteria.has_delegate = True
        return self
    
    def without_delegate(self) -> 'FilterBuilder':
        """Only include models without delegates."""
        self.criteria.has_delegate = False
        return self
    
    def by_delegate(self, delegate_email: str) -> 'FilterBuilder':
        """Filter by specific delegate email."""
        self.criteria.delegate_email = delegate_email
        return self
    
    def build(self) -> ModelFilter:
        """Build the filter."""
        return ModelFilter(self.criteria)


# Predefined common filters
def create_chat_models_filter(max_cost: Optional[float] = None) -> ModelFilter:
    """Create filter for chat models."""
    builder = FilterBuilder().by_service_type(ServiceType.CHAT)
    if max_cost is not None:
        builder = builder.by_max_cost(max_cost)
    return builder.build()


def create_search_models_filter(max_cost: Optional[float] = None) -> ModelFilter:
    """Create filter for search models."""
    builder = FilterBuilder().by_service_type(ServiceType.SEARCH)
    if max_cost is not None:
        builder = builder.by_max_cost(max_cost)
    return builder.build()


def create_free_models_filter() -> ModelFilter:
    """Create filter for free models."""
    return FilterBuilder().free_only().build()


def create_premium_models_filter() -> ModelFilter:
    """Create filter for premium/paid models."""
    return FilterBuilder().paid_only().build()


def create_healthy_models_filter() -> ModelFilter:
    """Create filter for healthy models."""
    return FilterBuilder().by_health_status(HealthStatus.ONLINE).build()


def create_owner_models_filter(owner: str) -> ModelFilter:
    """Create filter for specific owner."""
    return FilterBuilder().by_owner(owner).build()


def create_tag_models_filter(tags: List[str], match_all: bool = False) -> ModelFilter:
    """Create filter for specific tags."""
    return FilterBuilder().by_tags(tags, match_all).build()


def combine_filters(filters: List[ModelFilter]) -> ModelFilter:
    """Combine multiple filters into one (AND logic).
    
    Args:
        filters: List of filters to combine
        
    Returns:
        Combined filter
    """
    combined_filter = ModelFilter()
    
    def combined_check(model: ModelInfo) -> bool:
        return all(f._passes_all_filters(model) for f in filters)
    
    combined_filter.add_custom_filter(combined_check)
    return combined_filter