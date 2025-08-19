"""
ModelInfo data class and related utilities
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..core.types import ServiceInfo, ServiceType, ModelStatus, HealthStatus


@dataclass
class ModelInfo:
    """Complete information about a discovered SyftBox model."""
    
    # Basic metadata
    name: str
    owner: str
    summary: str
    description: str
    tags: List[str] = field(default_factory=list)
    
    # Service configuration
    services: List[ServiceInfo] = field(default_factory=list)
    
    # Status information
    config_status: ModelStatus = ModelStatus.DISABLED
    health_status: Optional[HealthStatus] = None
    
    # Delegation information
    delegate_email: Optional[str] = None
    delegate_control_types: Optional[List[str]] = None
    
    # Technical details
    endpoints: Dict[str, Any] = field(default_factory=dict)
    rpc_schema: Dict[str, Any] = field(default_factory=dict)
    code_hash: Optional[str] = None
    version: Optional[str] = None
    
    # File system paths
    metadata_path: Optional[Path] = None
    rpc_schema_path: Optional[Path] = None
    
    # Timestamps
    publish_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    discovered_at: Optional[datetime] = None
    
    # Computed service URLs (populated at runtime)
    service_urls: Dict[ServiceType, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set discovery timestamp
        if self.discovered_at is None:
            self.discovered_at = datetime.now()
        
        # Parse string dates if needed
        if isinstance(self.publish_date, str):
            try:
                self.publish_date = datetime.fromisoformat(self.publish_date.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                self.publish_date = None
    
    # Service-related properties
    
    @property
    def has_enabled_services(self) -> bool:
        """Check if model has any enabled services."""
        return any(service.enabled for service in self.services)
    
    @property
    def enabled_service_types(self) -> List[ServiceType]:
        """Get list of enabled service types."""
        return [service.type for service in self.services if service.enabled]
    
    @property
    def disabled_service_types(self) -> List[ServiceType]:
        """Get list of disabled service types."""
        return [service.type for service in self.services if not service.enabled]
    
    @property
    def all_service_types(self) -> List[ServiceType]:
        """Get list of all service types (enabled and disabled)."""
        return [service.type for service in self.services]
    
    def get_service_info(self, service_type: ServiceType) -> Optional[ServiceInfo]:
        """Get service information for a specific service type."""
        for service in self.services:
            if service.type == service_type:
                return service
        return None
    
    def supports_service(self, service_type: ServiceType) -> bool:
        """Check if model supports and has enabled a specific service type."""
        service = self.get_service_info(service_type)
        return service is not None and service.enabled
    
    def has_service(self, service_type: ServiceType) -> bool:
        """Check if model has a service type (regardless of enabled status)."""
        return any(service.type == service_type for service in self.services)
    
    # Pricing-related properties
    
    @property
    def min_pricing(self) -> float:
        """Get minimum pricing across all enabled services."""
        enabled_services = [s for s in self.services if s.enabled]
        if not enabled_services:
            return 0.0
        return min(service.pricing for service in enabled_services)
    
    @property
    def max_pricing(self) -> float:
        """Get maximum pricing across all enabled services."""
        enabled_services = [s for s in self.services if s.enabled]
        if not enabled_services:
            return 0.0
        return max(service.pricing for service in enabled_services)
    
    @property
    def avg_pricing(self) -> float:
        """Get average pricing across all enabled services."""
        enabled_services = [s for s in self.services if s.enabled]
        if not enabled_services:
            return 0.0
        return sum(service.pricing for service in enabled_services) / len(enabled_services)
    
    @property
    def is_free(self) -> bool:
        """Check if all enabled services are free."""
        return self.max_pricing == 0.0
    
    @property
    def is_paid(self) -> bool:
        """Check if any enabled services require payment."""
        return self.max_pricing > 0.0
    
    def get_pricing_for_service(self, service_type: ServiceType) -> Optional[float]:
        """Get pricing for a specific service type."""
        service = self.get_service_info(service_type)
        return service.pricing if service else None
    
    # Status-related properties
    
    @property
    def is_healthy(self) -> bool:
        """Check if model is healthy (online)."""
        return self.health_status == HealthStatus.ONLINE
    
    @property
    def is_available(self) -> bool:
        """Check if model is available (has enabled services and is healthy or health unknown)."""
        return (self.has_enabled_services and 
                (self.health_status is None or 
                 self.health_status in [HealthStatus.ONLINE, HealthStatus.UNKNOWN]))
    
    @property
    def is_active(self) -> bool:
        """Check if model is active (enabled services and active config status)."""
        return (self.has_enabled_services and 
                self.config_status == ModelStatus.ACTIVE)
    
    # Delegate-related properties
    
    @property
    def has_delegate(self) -> bool:
        """Check if model has a delegate."""
        return self.delegate_email is not None
    
    @property
    def is_delegated(self) -> bool:
        """Alias for has_delegate."""
        return self.has_delegate
    
    def can_delegate_control(self, control_type: str) -> bool:
        """Check if delegate can perform specific control type."""
        if not self.has_delegate or not self.delegate_control_types:
            return False
        return control_type in self.delegate_control_types
    
    # Metadata-related properties
    
    @property
    def has_metadata_file(self) -> bool:
        """Check if model has an accessible metadata file."""
        return self.metadata_path is not None and self.metadata_path.exists()
    
    @property
    def has_rpc_schema(self) -> bool:
        """Check if model has an RPC schema."""
        return bool(self.rpc_schema) or (
            self.rpc_schema_path is not None and self.rpc_schema_path.exists()
        )
    
    @property
    def has_endpoints_documented(self) -> bool:
        """Check if model has documented endpoints."""
        return bool(self.endpoints)
    
    # Tag-related methods
    
    def has_tag(self, tag: str) -> bool:
        """Check if model has a specific tag (case-insensitive)."""
        return tag.lower() in [t.lower() for t in self.tags]
    
    def has_any_tags(self, tags: List[str]) -> bool:
        """Check if model has any of the specified tags."""
        model_tags = [t.lower() for t in self.tags]
        return any(tag.lower() in model_tags for tag in tags)
    
    def has_all_tags(self, tags: List[str]) -> bool:
        """Check if model has all of the specified tags."""
        model_tags = [t.lower() for t in self.tags]
        return all(tag.lower() in model_tags for tag in tags)
    
    def get_matching_tags(self, tags: List[str]) -> List[str]:
        """Get list of tags that match the provided tags."""
        model_tags_lower = {t.lower(): t for t in self.tags}
        return [model_tags_lower[tag.lower()] for tag in tags 
                if tag.lower() in model_tags_lower]
    
    # Utility methods
    
    def get_service_summary(self) -> Dict[str, Any]:
        """Get summary of services."""
        enabled = [s for s in self.services if s.enabled]
        disabled = [s for s in self.services if not s.enabled]
        
        return {
            'total_services': len(self.services),
            'enabled_services': len(enabled),
            'disabled_services': len(disabled),
            'enabled_types': [s.type.value for s in enabled],
            'disabled_types': [s.type.value for s in disabled],
            'min_price': self.min_pricing,
            'max_price': self.max_pricing,
            'avg_price': self.avg_pricing,
            'is_free': self.is_free
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of model status."""
        return {
            'config_status': self.config_status.value,
            'health_status': self.health_status.value if self.health_status else None,
            'is_available': self.is_available,
            'is_healthy': self.is_healthy,
            'is_active': self.is_active,
            'has_delegate': self.has_delegate,
            'delegate_email': self.delegate_email
        }
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get summary of metadata and file information."""
        return {
            'name': self.name,
            'owner': self.owner,
            'summary': self.summary,
            'tags': self.tags,
            'version': self.version,
            'code_hash': self.code_hash,
            'publish_date': self.publish_date.isoformat() if self.publish_date else None,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            'has_metadata_file': self.has_metadata_file,
            'has_rpc_schema': self.has_rpc_schema,
            'has_endpoints_documented': self.has_endpoints_documented
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelInfo to dictionary for serialization."""
        return {
            # Basic info
            'name': self.name,
            'owner': self.owner,
            'summary': self.summary,
            'description': self.description,
            'tags': self.tags,
            
            # Services
            'services': [
                {
                    'type': service.type.value,
                    'enabled': service.enabled,
                    'pricing': service.pricing,
                    'charge_type': service.charge_type.value
                }
                for service in self.services
            ],
            
            # Status
            'config_status': self.config_status.value,
            'health_status': self.health_status.value if self.health_status else None,
            
            # Delegate info
            'delegate_email': self.delegate_email,
            'delegate_control_types': self.delegate_control_types,
            
            # Technical details
            'endpoints': self.endpoints,
            'rpc_schema': self.rpc_schema,
            'code_hash': self.code_hash,
            'version': self.version,
            
            # Paths (as strings)
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'rpc_schema_path': str(self.rpc_schema_path) if self.rpc_schema_path else None,
            
            # Timestamps
            'publish_date': self.publish_date.isoformat() if self.publish_date else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            
            # Computed properties
            'min_pricing': self.min_pricing,
            'max_pricing': self.max_pricing,
            'is_free': self.is_free,
            'is_available': self.is_available,
            'has_enabled_services': self.has_enabled_services,
            'enabled_service_types': [st.value for st in self.enabled_service_types]
        }
    
    def __repr__(self) -> str:
        """String representation of ModelInfo."""
        service_types = ', '.join([s.type.value for s in self.services if s.enabled])
        return (f"ModelInfo(name='{self.name}', owner='{self.owner}', "
                f"services=[{service_types}], status={self.config_status.value})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        health_indicator = ""
        if self.health_status:
            indicators = {
                HealthStatus.ONLINE: "✅",
                HealthStatus.OFFLINE: "❌",
                HealthStatus.TIMEOUT: "⏱️",
                HealthStatus.UNKNOWN: "❓"
            }
            health_indicator = f" {indicators.get(self.health_status, '❓')}"
        
        pricing = f"${self.min_pricing}" if self.min_pricing > 0 else "Free"
        
        return f"{self.name} by {self.owner} ({pricing}){health_indicator}"
    
    def __eq__(self, other) -> bool:
        """Check equality based on name and owner."""
        if not isinstance(other, ModelInfo):
            return False
        return self.name == other.name and self.owner == other.owner
    
    def __hash__(self) -> int:
        """Hash based on name and owner."""
        return hash((self.name, self.owner))


# Utility functions for working with ModelInfo objects

def group_models_by_owner(models: List[ModelInfo]) -> Dict[str, List[ModelInfo]]:
    """Group models by owner email."""
    groups = {}
    for model in models:
        if model.owner not in groups:
            groups[model.owner] = []
        groups[model.owner].append(model)
    return groups


def group_models_by_service_type(models: List[ModelInfo]) -> Dict[ServiceType, List[ModelInfo]]:
    """Group models by service type."""
    groups = {}
    for model in models:
        for service_type in model.enabled_service_types:
            if service_type not in groups:
                groups[service_type] = []
            groups[service_type].append(model)
    return groups


def group_models_by_status(models: List[ModelInfo]) -> Dict[str, List[ModelInfo]]:
    """Group models by availability status."""
    groups = {
        'available': [],
        'unavailable': [],
        'unknown': []
    }
    
    for model in models:
        if model.is_available:
            groups['available'].append(model)
        elif model.health_status == HealthStatus.OFFLINE:
            groups['unavailable'].append(model)
        else:
            groups['unknown'].append(model)
    
    return groups


def sort_models_by_preference(models: List[ModelInfo], 
                            preference: str = "balanced") -> List[ModelInfo]:
    """Sort models by preference (cheapest, premium, balanced)."""
    if preference == "cheapest":
        return sorted(models, key=lambda m: m.min_pricing)
    elif preference == "premium":
        return sorted(models, key=lambda m: m.max_pricing, reverse=True)
    elif preference == "balanced":
        def score(model):
            # Balance cost (lower is better) and quality indicators
            cost_score = 1.0 / (model.min_pricing + 0.01)
            
            # Quality indicators
            quality_score = 0
            quality_tags = {'premium', 'gpt4', 'claude', 'enterprise', 'high-quality'}
            quality_score += len(set(model.tags).intersection(quality_tags)) * 0.5
            
            # Health bonus
            if model.health_status == HealthStatus.ONLINE:
                quality_score += 1.0
            
            # Service variety bonus
            quality_score += len(model.enabled_service_types) * 0.2
            
            return cost_score + quality_score
        
        return sorted(models, key=score, reverse=True)
    else:
        return models


def filter_healthy_models(models: List[ModelInfo]) -> List[ModelInfo]:
    """Filter models to only include healthy ones."""
    return [model for model in models if model.is_healthy]


def filter_available_models(models: List[ModelInfo]) -> List[ModelInfo]:
    """Filter models to only include available ones."""
    return [model for model in models if model.is_available]


def get_model_statistics(models: List[ModelInfo]) -> Dict[str, Any]:
    """Get comprehensive statistics about a list of models."""
    if not models:
        return {}
    
    # Basic counts
    total = len(models)
    enabled = len([m for m in models if m.has_enabled_services])
    healthy = len([m for m in models if m.is_healthy])
    free = len([m for m in models if m.is_free])
    paid = len([m for m in models if m.is_paid])
    
    # Service type counts
    service_counts = {}
    for service_type in ServiceType:
        service_counts[service_type.value] = len([
            m for m in models if m.supports_service(service_type)
        ])
    
    # Owner statistics
    owners = list(set(m.owner for m in models))
    models_per_owner = {}
    for owner in owners:
        models_per_owner[owner] = len([m for m in models if m.owner == owner])
    
    # Pricing statistics
    paid_models = [m for m in models if m.is_paid]
    pricing_stats = {}
    if paid_models:
        prices = [m.min_pricing for m in paid_models]
        pricing_stats = {
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': sum(prices) / len(prices),
            'median_price': sorted(prices)[len(prices) // 2]
        }
    
    # Tag statistics
    all_tags = []
    for model in models:
        all_tags.extend(model.tags)
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'total_models': total,
        'enabled_models': enabled,
        'healthy_models': healthy,
        'free_models': free,
        'paid_models': paid,
        'service_counts': service_counts,
        'total_owners': len(owners),
        'avg_models_per_owner': total / len(owners) if owners else 0,
        'top_owners': sorted(models_per_owner.items(), 
                           key=lambda x: x[1], reverse=True)[:5],
        'pricing_stats': pricing_stats,
        'total_tags': len(set(all_tags)),
        'top_tags': top_tags
    }