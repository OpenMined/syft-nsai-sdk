"""
Formatting utilities for displaying model information
"""
from typing import List, Optional
from datetime import datetime

from ..core.types import ModelInfo, ServiceType, HealthStatus


def format_models_table(models: List[ModelInfo]) -> str:
    """Format models as a table.
    
    Args:
        models: List of models to display
        
    Returns:
        Formatted table string
    """
    if not models:
        return "No models found."
    
    # Calculate column widths
    name_width = max(len("Name"), max(len(model.name) for model in models), 15)
    owner_width = max(len("Owner"), max(len(model.owner) for model in models), 15)
    services_width = max(len("Services"), max(len(_format_services(model)) for model in models), 10)
    summary_width = max(len("Summary"), max(len(model.summary[:30]) for model in models), 20)
    status_width = max(len("Status"), max(len(_format_status(model)) for model in models), 10)
    
    # Ensure minimum widths and maximum widths for readability
    name_width = min(max(name_width, 15), 25)
    owner_width = min(max(owner_width, 15), 30)
    services_width = min(max(services_width, 10), 15)
    summary_width = min(max(summary_width, 20), 40)
    status_width = min(max(status_width, 10), 15)
    
    # Build table
    lines = []
    
    # Header
    header = f"┌─{'─' * name_width}─┬─{'─' * owner_width}─┬─{'─' * services_width}─┬─{'─' * summary_width}─┬─{'─' * status_width}─┐"
    lines.append(header)
    
    header_row = (f"│ {'Name':<{name_width}} │ {'Owner':<{owner_width}} │ "
                  f"{'Services':<{services_width}} │ {'Summary':<{summary_width}} │ {'Status':<{status_width}} │")
    lines.append(header_row)
    
    separator = f"├─{'─' * name_width}─┼─{'─' * owner_width}─┼─{'─' * services_width}─┼─{'─' * summary_width}─┼─{'─' * status_width}─┤"
    lines.append(separator)
    
    # Data rows
    for model in models:
        name = _truncate(model.name, name_width)
        owner = _truncate(model.owner, owner_width)
        services = _truncate(_format_services(model), services_width)
        summary = _truncate(model.summary, summary_width)
        status = _truncate(_format_status(model), status_width)
        
        row = (f"│ {name:<{name_width}} │ {owner:<{owner_width}} │ "
               f"{services:<{services_width}} │ {summary:<{summary_width}} │ {status:<{status_width}} │")
        lines.append(row)
    
    # Footer
    footer = f"└─{'─' * name_width}─┴─{'─' * owner_width}─┴─{'─' * services_width}─┴─{'─' * summary_width}─┴─{'─' * status_width}─┘"
    lines.append(footer)
    
    # Add summary info
    total_models = len(models)
    health_checked = len([m for m in models if m.health_status is not None])
    
    if health_checked > 0:
        online = len([m for m in models if m.health_status == HealthStatus.ONLINE])
        lines.append(f"\nFound {total_models} models (health checks: {online}/{health_checked} online)")
    else:
        lines.append(f"\nFound {total_models} models")
    
    return "\n".join(lines)


def format_model_details(model: ModelInfo) -> str:
    """Format detailed information about a single model.
    
    Args:
        model: Model to display details for
        
    Returns:
        Formatted details string
    """
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append(f"Model: {model.name}")
    lines.append("=" * 60)
    
    # Basic info
    lines.append(f"Owner: {model.owner}")
    lines.append(f"Summary: {model.summary}")
    if model.description != model.summary:
        lines.append(f"Description: {model.description}")
    
    # Status
    lines.append(f"Config Status: {model.config_status.value}")
    if model.health_status:
        lines.append(f"Health Status: {_format_health_status(model.health_status)}")
    
    # Services
    lines.append("\nServices:")
    if model.services:
        for service in model.services:
            status = "✅ Enabled" if service.enabled else "❌ Disabled"
            pricing = f"${service.pricing}/{service.charge_type.value}" if service.pricing > 0 else "Free"
            lines.append(f"  • {service.type.value.title()}: {status} ({pricing})")
    else:
        lines.append("  No services defined")
    
    # Tags
    if model.tags:
        lines.append(f"\nTags: {', '.join(model.tags)}")
    
    # Delegate info
    if model.delegate_email:
        lines.append(f"\nDelegate: {model.delegate_email}")
    
    # Pricing summary
    if model.has_enabled_services:
        if model.min_pricing == model.max_pricing:
            if model.min_pricing == 0:
                lines.append("\nPricing: Free")
            else:
                lines.append(f"\nPricing: ${model.min_pricing}")
        else:
            lines.append(f"\nPricing: ${model.min_pricing} - ${model.max_pricing}")
    
    # File paths (for debugging)
    if model.metadata_path:
        lines.append(f"\nMetadata: {model.metadata_path}")
    if model.rpc_schema_path:
        lines.append(f"RPC Schema: {model.rpc_schema_path}")
    
    return "\n".join(lines)


def format_search_results(query: str, results: List[dict], max_content_length: int = 100) -> str:
    """Format search results for display.
    
    Args:
        query: Original search query
        results: List of search results
        max_content_length: Maximum length of content to show
        
    Returns:
        Formatted search results
    """
    lines = []
    
    lines.append(f"Search Results for: \"{query}\"")
    lines.append("=" * (len(query) + 20))
    
    if not results:
        lines.append("No results found.")
        return "\n".join(lines)
    
    for i, result in enumerate(results, 1):
        lines.append(f"\n{i}. Score: {result.get('score', 'N/A')}")
        
        # Content
        content = result.get('content', '')
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        lines.append(f"   {content}")
        
        # Metadata
        if result.get('metadata'):
            metadata = result['metadata']
            if isinstance(metadata, dict):
                if 'filename' in metadata:
                    lines.append(f"   Source: {metadata['filename']}")
                if 'url' in metadata:
                    lines.append(f"   URL: {metadata['url']}")
    
    return "\n".join(lines)


def format_chat_conversation(messages: List[dict]) -> str:
    """Format a chat conversation for display.
    
    Args:
        messages: List of chat messages
        
    Returns:
        Formatted conversation
    """
    lines = []
    
    for message in messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        timestamp = message.get('timestamp')
        
        # Format timestamp if available
        time_str = ""
        if timestamp:
            if isinstance(timestamp, datetime):
                time_str = f" ({timestamp.strftime('%H:%M:%S')})"
            else:
                time_str = f" ({timestamp})"
        
        # Format message
        if role == 'user':
            lines.append(f"👤 User{time_str}:")
            lines.append(f"   {content}")
        elif role == 'assistant':
            lines.append(f"🤖 Assistant{time_str}:")
            lines.append(f"   {content}")
        elif role == 'system':
            lines.append(f"⚙️  System{time_str}:")
            lines.append(f"   {content}")
        else:
            lines.append(f"❓ {role.title()}{time_str}:")
            lines.append(f"   {content}")
        
        lines.append("")  # Empty line between messages
    
    return "\n".join(lines)


def format_health_summary(health_status: dict) -> str:
    """Format health status summary.
    
    Args:
        health_status: Dictionary mapping model names to health status
        
    Returns:
        Formatted health summary
    """
    if not health_status:
        return "No health data available."
    
    lines = []
    lines.append("Health Status Summary")
    lines.append("=" * 30)
    
    # Count by status
    status_counts = {}
    for status in health_status.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Overall stats
    total = len(health_status)
    online = status_counts.get(HealthStatus.ONLINE, 0)
    offline = status_counts.get(HealthStatus.OFFLINE, 0)
    timeout = status_counts.get(HealthStatus.TIMEOUT, 0)
    unknown = status_counts.get(HealthStatus.UNKNOWN, 0)
    
    lines.append(f"Total Models: {total}")
    lines.append(f"Online: {online} ✅")
    lines.append(f"Offline: {offline} ❌")
    lines.append(f"Timeout: {timeout} ⏱️")
    lines.append(f"Unknown: {unknown} ❓")
    
    # Detailed list
    lines.append("\nDetailed Status:")
    lines.append("-" * 30)
    
    for model_name, status in sorted(health_status.items()):
        status_str = _format_health_status(status)
        lines.append(f"{model_name}: {status_str}")
    
    return "\n".join(lines)


def format_statistics(stats: dict) -> str:
    """Format model statistics for display.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        Formatted statistics
    """
    lines = []
    lines.append("Model Statistics")
    lines.append("=" * 20)
    
    lines.append(f"Total Models: {stats.get('total_models', 0)}")
    lines.append(f"Enabled Models: {stats.get('enabled_models', 0)}")
    lines.append(f"Disabled Models: {stats.get('disabled_models', 0)}")
    lines.append(f"Chat Models: {stats.get('chat_models', 0)}")
    lines.append(f"Search Models: {stats.get('search_models', 0)}")
    lines.append(f"Free Models: {stats.get('free_models', 0)}")
    lines.append(f"Paid Models: {stats.get('paid_models', 0)}")
    lines.append(f"Total Owners: {stats.get('total_owners', 0)}")
    
    avg_models = stats.get('avg_models_per_owner', 0)
    lines.append(f"Avg Models per Owner: {avg_models:.1f}")
    
    # Top owners
    top_owners = stats.get('top_owners', [])
    if top_owners:
        lines.append("\nTop Model Owners:")
        for owner, count in top_owners:
            lines.append(f"  {owner}: {count} models")
    
    return "\n".join(lines)


# Private helper functions

def _truncate(text: str, max_length: int) -> str:
    """Truncate text to fit in column."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def _format_services(model: ModelInfo) -> str:
    """Format services list for table display."""
    enabled_services = [s.type.value for s in model.services if s.enabled]
    if not enabled_services:
        return "none"
    return ",".join(enabled_services)


def _format_status(model: ModelInfo) -> str:
    """Format status column for table display."""
    base_status = model.config_status.value
    
    if not model.has_enabled_services:
        return "Disabled"
    
    if model.health_status is None:
        return base_status
    
    if model.health_status == HealthStatus.ONLINE:
        return f"{base_status} ✅"
    elif model.health_status == HealthStatus.OFFLINE:
        return f"{base_status} ❌"
    elif model.health_status == HealthStatus.TIMEOUT:
        return f"{base_status} ⏱️"
    else:
        return f"{base_status} ❓"


def _format_health_status(status: HealthStatus) -> str:
    """Format health status with emoji."""
    status_map = {
        HealthStatus.ONLINE: "Online ✅",
        HealthStatus.OFFLINE: "Offline ❌", 
        HealthStatus.TIMEOUT: "Timeout ⏱️",
        HealthStatus.UNKNOWN: "Unknown ❓",
        HealthStatus.NOT_APPLICABLE: "N/A ➖"
    }
    return status_map.get(status, f"{status.value} ❓")