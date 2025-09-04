"""
CLI-specific display utilities and formatting
"""
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
import time

from ..core.types import ServiceType, HealthStatus
from ..models.service_info import ServiceInfo
from ..utils.formatting import _format_health_status


console = Console()


def create_services_table(services: List[ServiceInfo], show_health: bool = True) -> Table:
    """Create a rich table for displaying services.
    
    Args:
        services: List of services to display
        show_health: Whether to include health status
        
    Returns:
        Rich Table object
    """
    table = Table(title="Available Services", show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Datasite", style="blue", width=25)
    table.add_column("Services", style="green", width=12)
    table.add_column("Tags", style="yellow", width=15)
    table.add_column("Pricing", style="bright_green", width=10)
    
    if show_health:
        table.add_column("Status", style="bold", width=12)
    
    # Add rows
    for service in services:
        # Format services
        enabled_services = [s.type.value for s in service.services if s.enabled]
        services_str = ", ".join(enabled_services) if enabled_services else "none"
        
        # Format tags
        tags_str = ", ".join(service.tags[:3]) if service.tags else "none"
        if len(service.tags) > 3:
            tags_str += f" (+{len(service.tags)-3})"
        
        # Format pricing
        if service.min_pricing == 0:
            pricing_str = "[green]Free[/green]"
        elif service.min_pricing == service.max_pricing:
            pricing_str = f"${service.min_pricing}"
        else:
            pricing_str = f"${service.min_pricing}-${service.max_pricing}"
        
        # Format status
        status_str = ""
        if show_health:
            if service.health_status == HealthStatus.ONLINE:
                status_str = "[green]Online ✅[/green]"
            elif service.health_status == HealthStatus.OFFLINE:
                status_str = "[red]Offline ❌[/red]"
            elif service.health_status == HealthStatus.TIMEOUT:
                status_str = "[yellow]Timeout ⏱️[/yellow]"
            elif service.health_status == HealthStatus.UNKNOWN:
                status_str = "[dim]Unknown ❓[/dim]"
            else:
                status_str = service.config_status.value
        
        # Add row
        row = [
            service.name,
            service.datasite,
            services_str,
            tags_str,
            pricing_str
        ]
        
        if show_health:
            row.append(status_str)
        
        table.add_row(*row)
    
    return table


def create_service_detail_panel(service: ServiceInfo) -> Panel:
    """Create a detailed panel for a single service.
    
    Args:
        service: Service to display
        
    Returns:
        Rich Panel object
    """
    # Build content
    content_lines = []
    
    # Basic info
    content_lines.extend([
        f"[bold]Datasite:[/bold] {service.datasite}",
        f"[bold]Summary:[/bold] {service.summary}",
    ])
    
    if service.description != service.summary:
        content_lines.append(f"[bold]Description:[/bold] {service.description}")
    
    # Status
    content_lines.append(f"[bold]Config Status:[/bold] {service.config_status.value}")
    if service.health_status:
        health_display = _format_health_status(service.health_status)
        content_lines.append(f"[bold]Health Status:[/bold] {health_display}")
    
    # Services
    content_lines.append("\n[bold]Services:[/bold]")
    if service.services:
        for service in service.services:
            status_icon = "✅" if service.enabled else "❌"
            pricing = f"${service.pricing}/{service.charge_type.value}" if service.pricing > 0 else "Free"
            content_lines.append(f"  {status_icon} {service.type.value.title()}: {pricing}")
    else:
        content_lines.append("  No services defined")
    
    # Tags
    if service.tags:
        tags_display = ", ".join(f"[yellow]{tag}[/yellow]" for tag in service.tags)
        content_lines.append(f"\n[bold]Tags:[/bold] {tags_display}")
    
    # Delegate
    if service.delegate_email:
        content_lines.append(f"\n[bold]Delegate:[/bold] {service.delegate_email}")
    
    # Pricing summary
    if service.has_enabled_services:
        if service.min_pricing == service.max_pricing:
            if service.min_pricing == 0:
                content_lines.append(f"\n[bold]Pricing:[/bold] [green]Free[/green]")
            else:
                content_lines.append(f"\n[bold]Pricing:[/bold] ${service.min_pricing}")
        else:
            content_lines.append(f"\n[bold]Pricing:[/bold] ${service.min_pricing} - ${service.max_pricing}")
    
    content = "\n".join(content_lines)
    
    return Panel(
        content,
        title=f"[bold cyan]{service.name}[/bold cyan]",
        border_style="blue",
        padding=(1, 2)
    )


def create_health_summary_table(health_status: Dict[str, HealthStatus]) -> Table:
    """Create a table showing health status summary.
    
    Args:
        health_status: Dictionary mapping service names to health status
        
    Returns:
        Rich Table object
    """
    table = Table(title="Service Health Status", show_header=True, header_style="bold magenta")
    
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Health", justify="center")
    
    for service_name, status in sorted(health_status.items()):
        if status == HealthStatus.ONLINE:
            health_icon = "✅"
            status_style = "green"
        elif status == HealthStatus.OFFLINE:
            health_icon = "❌"
            status_style = "red"
        elif status == HealthStatus.TIMEOUT:
            health_icon = "⏱️"
            status_style = "yellow"
        else:
            health_icon = "❓"
            status_style = "dim"
        
        table.add_row(
            service_name,
            f"[{status_style}]{status.value.title()}[/{status_style}]",
            health_icon
        )
    
    return table


def create_statistics_panel(stats: Dict[str, Any]) -> Panel:
    """Create a panel showing service statistics.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        Rich Panel object
    """
    content_lines = [
        f"[bold]Total Services:[/bold] {stats.get('total_services', 0)}",
        f"[bold]Enabled Services:[/bold] {stats.get('enabled_services', 0)}",
        f"[bold]Disabled Services:[/bold] {stats.get('disabled_services', 0)}",
        "",
        f"[bold]Service Types:[/bold]",
        f"  Chat Services: {stats.get('chat_services', 0)}",
        f"  Search Services: {stats.get('search_services', 0)}",
        "",
        f"[bold]Pricing:[/bold]",
        f"  Free Services: [green]{stats.get('free_services', 0)}[/green]",
        f"  Paid Services: [yellow]{stats.get('paid_services', 0)}[/yellow]",
        "",
        f"[bold]Ownership:[/bold]",
        f"  Total Owners: {stats.get('total_datasites', 0)}",
        f"  Avg Services/Datasite: {stats.get('avg_services_per_datasite', 0):.1f}",
    ]
    
    # Top datasites
    top_datasites = stats.get('top_datasites', [])
    if top_datasites:
        content_lines.extend([
            "",
            "[bold]Top Owners:[/bold]"
        ])
        for datasite, count in top_datasites[:5]:
            content_lines.append(f"  {datasite}: {count} services")
    
    content = "\n".join(content_lines)
    
    return Panel(
        content,
        title="[bold cyan]Service Statistics[/bold cyan]",
        border_style="green",
        padding=(1, 2)
    )


def create_search_results_panel(query: str, results: List[Dict], cost: Optional[float] = None) -> Panel:
    """Create a panel showing search results.
    
    Args:
        query: Original search query
        results: List of search results
        cost: Optional cost of the search
        
    Returns:
        Rich Panel object
    """
    title = f"Search Results: '{query}'"
    if cost is not None and cost > 0:
        title += f" (${cost})"
    
    if not results:
        content = "[yellow]No results found.[/yellow]"
    else:
        content_lines = []
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            content_lines.append(f"[bold]{i}. Score: {score:.3f}[/bold]")
            
            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."
            content_lines.append(content)
            
            # Add source if available
            if isinstance(metadata, dict) and 'filename' in metadata:
                content_lines.append(f"[dim]Source: {metadata['filename']}[/dim]")
            
            if i < len(results):
                content_lines.append("")  # Blank line between results
        
        content = "\n".join(content_lines)
    
    return Panel(
        content,
        title=title,
        border_style="cyan",
        padding=(1, 2)
    )


def create_chat_response_panel(response_content: str, service: str, cost: Optional[float] = None, 
                              tokens_used: Optional[int] = None) -> Panel:
    """Create a panel showing chat response.
    
    Args:
        response_content: The response text
        service: Service that generated the response
        cost: Optional cost of the request
        tokens_used: Optional number of tokens used
        
    Returns:
        Rich Panel object
    """
    title_parts = [f"Response from {service}"]
    
    if cost is not None and cost > 0:
        title_parts.append(f"${cost}")
    
    if tokens_used is not None and tokens_used > 0:
        title_parts.append(f"{tokens_used} tokens")
    
    title = " • ".join(title_parts)
    
    return Panel(
        response_content,
        title=title,
        border_style="green",
        padding=(1, 2)
    )


def create_progress_context(description: str):
    """Create a progress context manager for long operations.
    
    Args:
        description: Description of the operation
        
    Returns:
        Context manager for progress display
    """
    return console.status(f"[bold blue]{description}...")


def show_error_panel(error_message: str, details: Optional[str] = None, 
                    suggestions: Optional[List[str]] = None):
    """Display an error panel with optional details and suggestions.
    
    Args:
        error_message: Main error message
        details: Optional detailed error information
        suggestions: Optional list of suggestions to fix the error
    """
    content_lines = [f"[bold red]Error:[/bold red] {error_message}"]
    
    if details:
        content_lines.extend(["", f"[dim]Details: {details}[/dim]"])
    
    if suggestions:
        content_lines.extend(["", "[bold yellow]Suggestions:[/bold yellow]"])
        for suggestion in suggestions:
            content_lines.append(f"• {suggestion}")
    
    content = "\n".join(content_lines)
    
    panel = Panel(
        content,
        title="[bold red]Error[/bold red]",
        border_style="red",
        padding=(1, 2)
    )
    
    console.print(panel)


def show_warning_panel(warning_message: str, details: Optional[str] = None):
    """Display a warning panel.
    
    Args:
        warning_message: Warning message
        details: Optional detailed information
    """
    content_lines = [f"[bold yellow]Warning:[/bold yellow] {warning_message}"]
    
    if details:
        content_lines.extend(["", f"[dim]{details}[/dim]"])
    
    content = "\n".join(content_lines)
    
    panel = Panel(
        content,
        title="[bold yellow]Warning[/bold yellow]",
        border_style="yellow",
        padding=(1, 2)
    )
    
    console.print(panel)


def show_success_panel(success_message: str, details: Optional[str] = None):
    """Display a success panel.
    
    Args:
        success_message: Success message
        details: Optional detailed information
    """
    content_lines = [f"[bold green]Success:[/bold green] {success_message}"]
    
    if details:
        content_lines.extend(["", f"[dim]{details}[/dim]"])
    
    content = "\n".join(content_lines)
    
    panel = Panel(
        content,
        title="[bold green]Success[/bold green]",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print(panel)


def create_tree_view(services: List[ServiceInfo]) -> Tree:
    """Create a tree view of services grouped by datasite.
    
    Args:
        services: List of services to display
        
    Returns:
        Rich Tree object
    """
    tree = Tree("📦 [bold blue]SyftBox Services[/bold blue]")
    
    # Group services by datasite
    by_datasite = {}
    for service in services:
        if service.datasite not in by_datasite:
            by_datasite[service.datasite] = []
        by_datasite[service.datasite].append(service)
    
    # Add datasite branches
    for datasite, datasite_services in sorted(by_datasite.items()):
        datasite_branch = tree.add(f"👤 [cyan]{datasite}[/cyan] ({len(datasite_services)} services)")
        
        for service in sorted(datasite_services, key=lambda m: m.name):
            # Service info
            services = [s.type.value for s in service.services if s.enabled]
            services_str = ", ".join(services) if services else "none"
            
            pricing = "Free" if service.min_pricing == 0 else f"${service.min_pricing}"
            
            health_icon = ""
            if service.health_status == HealthStatus.ONLINE:
                health_icon = " ✅"
            elif service.health_status == HealthStatus.OFFLINE:
                health_icon = " ❌"
            elif service.health_status == HealthStatus.TIMEOUT:
                health_icon = " ⏱️"
            
            service_info = f"🤖 [yellow]{service.name}[/yellow] • {services_str} • {pricing}{health_icon}"
            service_branch = datasite_branch.add(service_info)
            
            # Add tags if any
            if service.tags:
                tags_str = ", ".join(f"#{tag}" for tag in service.tags[:5])
                if len(service.tags) > 5:
                    tags_str += f" (+{len(service.tags)-5} more)"
                service_branch.add(f"🏷️  {tags_str}")
    
    return tree


def create_comparison_table(services: List[ServiceInfo], criteria: List[str]) -> Table:
    """Create a comparison table for multiple services.
    
    Args:
        services: Services to compare
        criteria: List of criteria to compare ('name', 'datasite', 'pricing', 'services', 'health')
        
    Returns:
        Rich Table object
    """
    table = Table(title="Service Comparison", show_header=True, header_style="bold magenta")
    
    # Add columns based on criteria
    column_map = {
        'name': ('Name', 'cyan'),
        'datasite': ('Datasite', 'blue'),
        'pricing': ('Pricing', 'green'),
        'services': ('Services', 'yellow'),
        'health': ('Health', 'bold'),
        'tags': ('Tags', 'dim'),
    }
    
    for criterion in criteria:
        if criterion in column_map:
            col_name, col_style = column_map[criterion]
            table.add_column(col_name, style=col_style)
    
    # Add rows
    for service in services:
        row = []
        
        for criterion in criteria:
            if criterion == 'name':
                row.append(service.name)
            elif criterion == 'datasite':
                row.append(service.datasite)
            elif criterion == 'pricing':
                if service.min_pricing == 0:
                    row.append("[green]Free[/green]")
                else:
                    row.append(f"${service.min_pricing}")
            elif criterion == 'services':
                services = [s.type.value for s in service.services if s.enabled]
                row.append(", ".join(services) if services else "none")
            elif criterion == 'health':
                if service.health_status == HealthStatus.ONLINE:
                    row.append("[green]Online ✅[/green]")
                elif service.health_status == HealthStatus.OFFLINE:
                    row.append("[red]Offline ❌[/red]")
                elif service.health_status == HealthStatus.TIMEOUT:
                    row.append("[yellow]Timeout ⏱️[/yellow]")
                else:
                    row.append("[dim]Unknown[/dim]")
            elif criterion == 'tags':
                tags_str = ", ".join(service.tags[:3]) if service.tags else "none"
                if len(service.tags) > 3:
                    tags_str += f" (+{len(service.tags)-3})"
                row.append(tags_str)
        
        table.add_row(*row)
    
    return table


def create_live_health_monitor(services: List[str]) -> Live:
    """Create a live updating health monitor display.
    
    Args:
        services: List of service names to monitor
        
    Returns:
        Rich Live object for updating display
    """
    # Create initial table
    table = Table(title="Live Health Monitor", show_header=True, header_style="bold magenta")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Last Check", style="dim")
    
    # Add placeholder rows
    for service_name in services:
        table.add_row(service_name, "[dim]Checking...[/dim]", "[dim]N/A[/dim]")
    
    return Live(table, refresh_per_second=1)


def update_health_monitor_table(table: Table, service_name: str, status: HealthStatus, 
                              last_check: str) -> None:
    """Update a specific row in the health monitor table.
    
    Args:
        table: Table to update
        service_name: Name of the service
        status: Current health status
        last_check: Timestamp of last check
    """
    # This would need more complex implementation to update specific rows
    # For now, it's a placeholder for the live monitoring functionality
    pass