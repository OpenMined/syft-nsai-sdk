"""
ServiceInfo data class and related utilities
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..core.types import ServiceItem, ServiceType, ServiceStatus, HealthStatus


@dataclass
class ServiceInfo:
    """Complete information about a discovered SyftBox service."""
    
    # Basic metadata
    name: str = ""
    datasite: str = ""
    summary: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Service configuration
    services: List[ServiceItem] = field(default_factory=list)
    
    # Status information
    config_status: ServiceStatus = ServiceStatus.DISABLED
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
        """Check if service has any enabled services."""
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
    
    def get_service_info(self, service_type: ServiceType) -> Optional[ServiceItem]:
        """Get service information for a specific service type."""
        for service in self.services:
            if service.type == service_type:
                return service
        return None
    
    def supports_service(self, service_type: ServiceType) -> bool:
        """Check if service supports and has enabled a specific service type."""
        service = self.get_service_info(service_type)
        return service is not None and service.enabled
    
    def has_service(self, service_type: ServiceType) -> bool:
        """Check if service has a service type (regardless of enabled status)."""
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
        """Check if service is healthy (online)."""
        return self.health_status == HealthStatus.ONLINE
    
    @property
    def is_available(self) -> bool:
        """Check if service is available (has enabled services and is healthy or health unknown)."""
        return (self.has_enabled_services and 
                (self.health_status is None or 
                 self.health_status in [HealthStatus.ONLINE, HealthStatus.UNKNOWN]))
    
    @property
    def is_active(self) -> bool:
        """Check if service is active (enabled services and active config status)."""
        return (self.has_enabled_services and 
                self.config_status == ServiceStatus.ACTIVE)
    
    # Delegate-related properties
    @property
    def has_delegate(self) -> bool:
        """Check if service has a delegate."""
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
        """Check if service has an accessible metadata file."""
        return self.metadata_path is not None and self.metadata_path.exists()
    
    @property
    def has_rpc_schema(self) -> bool:
        """Check if service has an RPC schema."""
        return bool(self.rpc_schema) or (
            self.rpc_schema_path is not None and self.rpc_schema_path.exists()
        )
    
    @property
    def has_endpoints_documented(self) -> bool:
        """Check if service has documented endpoints."""
        return bool(self.endpoints)
    
    # Tag-related methods
    def has_tag(self, tag: str) -> bool:
        """Check if service has a specific tag (case-insensitive)."""
        return tag.lower() in [t.lower() for t in self.tags]
    
    def has_any_tags(self, tags: List[str]) -> bool:
        """Check if service has any of the specified tags."""
        service_tags = [t.lower() for t in self.tags]
        return any(tag.lower() in service_tags for tag in tags)
    
    def has_all_tags(self, tags: List[str]) -> bool:
        """Check if service has all of the specified tags."""
        service_tags = [t.lower() for t in self.tags]
        return all(tag.lower() in service_tags for tag in tags)
    
    def get_matching_tags(self, tags: List[str]) -> List[str]:
        """Get list of tags that match the provided tags."""
        service_tags_lower = {t.lower(): t for t in self.tags}
        return [service_tags_lower[tag.lower()] for tag in tags 
                if tag.lower() in service_tags_lower]
    
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
        """Get summary of service status."""
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
            'datasite': self.datasite,
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
        """Convert ServiceInfo to dictionary for serialization."""
        return {
            # Basic info
            'name': self.name,
            'datasite': self.datasite,
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
        """String representation of ServiceInfo in client's __repr__ format."""
        # Get basic service info
        service_name = self.name
        datasite = self.datasite
        summary = self.summary
        status = self.config_status.value
        
        # Get enabled services
        enabled_services = []
        total_cost = 0
        for service_item in self.services:
            if service_item.enabled:
                enabled_services.append(service_item.type.value.title())
                total_cost += service_item.pricing
        
        services_str = ", ".join(enabled_services) if enabled_services else "None"
        
        # Health status
        health_str = ""
        if self.health_status:
            health_map = {
                HealthStatus.ONLINE: "Online",
                HealthStatus.OFFLINE: "Offline", 
                HealthStatus.TIMEOUT: "Timeout",
                HealthStatus.UNKNOWN: "Unknown"
            }
            health_str = f" [{health_map.get(self.health_status, 'Unknown')}]"
        
        # Pricing info
        pricing_str = "Free" if total_cost == 0 else f"${total_cost:.2f}/request"
        
        # Tags (limit to 3-4 for display)
        tags_display = ""
        if self.tags:
            display_tags = self.tags[:4]
            tags_display = ", ".join(display_tags)
            if len(self.tags) > 4:
                tags_display += f" (+{len(self.tags) - 4} more)"
        
        # Build lines in client's __repr__ format (exactly matching spacing)
        lines = [
            f"{service_name} Service [{status}]{health_str}",
            "",
            f"Datasite:         {datasite}",
            f"Summary:          {summary}",
            f"Services:         {services_str}",
            f"Pricing:          {pricing_str}",
        ]
        
        if tags_display:
            lines.append(f"Tags:             {tags_display}")
        
        lines.extend([
            "",
            "Available operations:",
            f"  client.chat('{datasite}/{service_name}', messages=[...])   — Chat with service{'✅' if self.is_healthy else '❌' if self.health_status and not self.is_healthy else ''}",
            f"  client.search('{datasite}/{service_name}', 'message')     — Search with service{'✅' if self.is_healthy else '❌' if self.health_status and not self.is_healthy else ''}",
        ])
        
        return "\n".join(lines)
    
    def show(self) -> None:
        """Display service information as an HTML widget in notebooks."""
        try:
            from IPython.display import display, HTML
        except ImportError:
            # Fallback to text representation if not in a notebook
            print(self.__repr__())
            return
        
        # Get basic service info
        service_name = self.name
        datasite = self.datasite
        summary = self.summary
        description = self.description if self.description != self.summary else ""
        status = self.config_status.value
        
        # Get enabled services
        enabled_services = []
        total_cost = 0
        for service_item in self.services:
            if service_item.enabled:
                enabled_services.append(service_item.type.value.title())
                total_cost += service_item.pricing
        
        services_str = ", ".join(enabled_services) if enabled_services else "None"
        
        # Health status with styling
        health_class = ""
        health_text = ""
        if self.health_status:
            health_map = {
                HealthStatus.ONLINE: ("online", "Online"),
                HealthStatus.OFFLINE: ("offline", "Offline"), 
                HealthStatus.TIMEOUT: ("timeout", "Timeout"),
                HealthStatus.UNKNOWN: ("unknown", "Unknown")
            }
            health_class, health_text = health_map.get(self.health_status, ("unknown", "Unknown"))
            health_text = f" [{health_text}]"
        
        # Pricing info
        pricing_str = "Free" if total_cost == 0 else f"${total_cost:.2f}/request"
        pricing_class = "free" if total_cost == 0 else "paid"
        
        # Tags (limit for display)
        tags_display = ""
        if self.tags:
            display_tags = self.tags[:4]
            tags_display = ", ".join(display_tags)
            if len(self.tags) > 4:
                tags_display += f" (+{len(self.tags) - 4} more)"
        
        # Build HTML widget with styling similar to client's show()
        html = f'''
        <style>
            .serviceinfo-widget {{
                font-family: system-ui, -apple-system, sans-serif;
                padding: 12px 0;
                color: #333;
                line-height: 1.5;
            }}
            .serviceinfo-title {{
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 12px;
                color: #333;
            }}
            .serviceinfo-status-line {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }}
            .serviceinfo-status {{
                font-size: 12px;
                padding: 2px 6px;
                border-radius: 3px;
                background: #e8f5e8;
                color: #2d5a2d;
            }}
            .serviceinfo-status.online {{
                background: #e8f5e8;
                color: #2d5a2d;
            }}
            .serviceinfo-status.offline {{
                background: #ffe6e6;
                color: #cc0000;
            }}
            .serviceinfo-status.timeout {{
                background: #fff3cd;
                color: #856404;
            }}
            .serviceinfo-status.unknown {{
                background: #e2e3e5;
                color: #6c757d;
            }}
            .serviceinfo-info {{
                display: grid;
                grid-template-columns: 100px 1fr;
                gap: 8px 12px;
                margin: 12px 0;
                font-size: 12px;
            }}
            .serviceinfo-label {{
                font-weight: 500;
                color: #666;
            }}
            .serviceinfo-value {{
                color: #333;
            }}
            .serviceinfo-pricing {{
                font-weight: 500;
            }}
            .serviceinfo-pricing.free {{
                color: #2d5a2d;
            }}
            .serviceinfo-pricing.paid {{
                color: #d63384;
            }}
            .serviceinfo-operations {{
                margin-top: 16px;
                padding-top: 12px;
                border-top: 1px solid #e0e0e0;
            }}
            .serviceinfo-operations-title {{
                font-size: 12px;
                font-weight: 500;
                margin-bottom: 8px;
                color: #666;
            }}
            .serviceinfo-command-code {{
                font-family: Monaco, 'Courier New', monospace;
                background: #f5f5f5;
                padding: 1px 4px;
                border-radius: 2px;
                color: #0066cc;
                font-size: 11px;
            }}
            .command-code {{
                font-family: Monaco, 'Courier New', monospace;
                background: #f5f5f5;
                padding: 1px 4px;
                border-radius: 2px;
                color: #0066cc;
                font-size: 11px;
            }}
        </style>
        <div class="serviceinfo-widget">
            <div class="serviceinfo-title">{service_name} Service</div>
            <div class="serviceinfo-status-line">
                <span>Status: {status}</span>
                {f'<span class="serviceinfo-status {health_class}">Health: {health_text.strip("[] ")}</span>' if health_text else ''}
            </div>
            <div class="serviceinfo-info">
                <div class="serviceinfo-label">Datasite:</div>
                <div class="serviceinfo-value">{datasite}</div>
                <div class="serviceinfo-label">Summary:</div>
                <div class="serviceinfo-value">{summary}</div>
                <div class="serviceinfo-label">Services:</div>
                <div class="serviceinfo-value">{services_str}</div>
                <div class="serviceinfo-label">Pricing:</div>
                <div class="serviceinfo-value serviceinfo-pricing {pricing_class}">{pricing_str}</div>
                {f'<div class="serviceinfo-label">Tags:</div><div class="serviceinfo-value">{tags_display}</div>' if tags_display else ''}
            </div>
            {f'<div style="margin: 12px 0; padding: 12px; background: #fff; border-radius: 4px; border: 1px solid #e9ecef; font-size: 13px; line-height: 1.6; color: #555;"><strong>Description:</strong><br>{description}</div>' if description else ''}
            <div class="serviceinfo-operations">
                <div class="serviceinfo-operations-title">Available operations:</div>
                <div style="line-height: 1.8;">
                    <span class="serviceinfo-command-code">client.chat('{datasite}/{service_name}', messages=[])</span> — Chat with service {"✅" if self.is_healthy else "❌" if self.health_status and not self.is_healthy else ""}<br>
                    <span class="serviceinfo-command-code">client.search('{datasite}/{service_name}', 'message')</span> — Search with service {"✅" if self.is_healthy else "❌" if self.health_status and not self.is_healthy else ""}
                </div>
            </div>
        </div>
        '''
        
        display(HTML(html))
    
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
        
        return f"{self.name} by {self.datasite} ({pricing}){health_indicator}"
    
    def __eq__(self, other) -> bool:
        """Check equality based on name and datasite."""
        if not isinstance(other, ServiceInfo):
            return False
        return self.name == other.name and self.datasite == other.datasite
    
    def __hash__(self) -> int:
        """Hash based on name and datasite."""
        return hash((self.name, self.datasite))


# Utility functions for working with ServiceInfo objects
def group_services_by_datasite(services: List[ServiceInfo]) -> Dict[str, List[ServiceInfo]]:
    """Group services by datasite email."""
    groups = {}
    for service in services:
        if service.datasite not in groups:
            groups[service.datasite] = []
        groups[service.datasite].append(service)
    return groups


def group_services_by_service_type(services: List[ServiceInfo]) -> Dict[ServiceType, List[ServiceInfo]]:
    """Group services by service type."""
    groups = {}
    for service in services:
        for service_type in service.enabled_service_types:
            if service_type not in groups:
                groups[service_type] = []
            groups[service_type].append(service)
    return groups


def group_services_by_status(services: List[ServiceInfo]) -> Dict[str, List[ServiceInfo]]:
    """Group services by availability status."""
    groups = {
        'available': [],
        'unavailable': [],
        'unknown': []
    }
    
    for service in services:
        if service.is_available:
            groups['available'].append(service)
        elif service.health_status == HealthStatus.OFFLINE:
            groups['unavailable'].append(service)
        else:
            groups['unknown'].append(service)
    
    return groups


def sort_services_by_preference(
        services: List[ServiceInfo], 
        preference: str = "balanced"
    ) -> List[ServiceInfo]:
    """Sort services by preference (cheapest, paid, balanced)."""
    if preference == "cheapest":
        return sorted(services, key=lambda m: m.min_pricing)
    elif preference == "paid":
        return sorted(services, key=lambda m: m.max_pricing, reverse=True)
    elif preference == "balanced":
        def score(service):
            # Balance cost (lower is better) and quality indicators
            cost_score = 1.0 / (service.min_pricing + 0.01)
            
            # Quality indicators
            quality_score = 0
            quality_tags = {'paid', 'gpt4', 'claude', 'enterprise', 'high-quality'}
            quality_score += len(set(service.tags).intersection(quality_tags)) * 0.5
            
            # Health bonus
            if service.health_status == HealthStatus.ONLINE:
                quality_score += 1.0
            
            # Service variety bonus
            quality_score += len(service.enabled_service_types) * 0.2
            
            return cost_score + quality_score
        
        return sorted(services, key=score, reverse=True)
    else:
        return services


def filter_healthy_services(services: List[ServiceInfo]) -> List[ServiceInfo]:
    """Filter services to only include healthy ones."""
    return [service for service in services if service.is_healthy]


def filter_available_services(services: List[ServiceInfo]) -> List[ServiceInfo]:
    """Filter services to only include available ones."""
    return [service for service in services if service.is_available]


def get_service_statistics(services: List[ServiceInfo]) -> Dict[str, Any]:
    """Get comprehensive statistics about a list of services."""
    if not services:
        return {}
    
    # Basic counts
    total = len(services)
    enabled = len([m for m in services if m.has_enabled_services])
    healthy = len([m for m in services if m.is_healthy])
    free = len([m for m in services if m.is_free])
    paid = len([m for m in services if m.is_paid])
    
    # Service type counts
    service_counts = {}
    for service_type in ServiceType:
        service_counts[service_type.value] = len([
            m for m in services if m.supports_service(service_type)
        ])
    
    # Datasite statistics
    datasites = list(set(m.datasite for m in services))
    services_per_datasite = {}
    for datasite in datasites:
        services_per_datasite[datasite] = len([m for m in services if m.datasite == datasite])
    
    # Pricing statistics
    paid_services = [m for m in services if m.is_paid]
    pricing_stats = {}
    if paid_services:
        prices = [m.min_pricing for m in paid_services]
        pricing_stats = {
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': sum(prices) / len(prices),
            'median_price': sorted(prices)[len(prices) // 2]
        }
    
    # Tag statistics
    all_tags = []
    for service in services:
        all_tags.extend(service.tags)
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'total_services': total,
        'enabled_services': enabled,
        'healthy_services': healthy,
        'free_services': free,
        'paid_services': paid,
        'service_counts': service_counts,
        'total_datasites': len(datasites),
        'avg_services_per_datasite': total / len(datasites) if datasites else 0,
        'top_datasites': sorted(services_per_datasite.items(), 
                           key=lambda x: x[1], reverse=True)[:5],
        'pricing_stats': pricing_stats,
        'total_tags': len(set(all_tags)),
        'top_tags': top_tags
    }