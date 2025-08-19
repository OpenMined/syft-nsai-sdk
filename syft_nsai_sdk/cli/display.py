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

from ..core.types import ModelInfo, ServiceType, HealthStatus
from ..utils.formatting import _format_health_status


console = Console()


def create_models_table(models: List[ModelInfo], show_health: bool = True) -> Table:
    """Create a rich table for displaying models.
    
    Args:
        models: List of models to display
        show_health: Whether to include health status
        
    Returns:
        Rich Table object
    """
    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Owner", style="blue", width=25)
    table.add_column("Services", style="green", width=12)
    table.add_column("Tags", style="yellow", width=15)
    table.add_column("Pricing", style="bright_green", width=10)
    
    if show_health:
        table.add_column("Status", style="bold", width=12)
    
    # Add rows
    for model in models:
        # Format services
        enabled_services = [s.type.value for s in model.services if s.enabled]
        services_str = ", ".join(enabled_services) if enabled_services else "none"
        
        # Format tags
        tags_str = ", ".join(model.tags[:3]) if model.tags else "none"
        if len(model.tags) > 3:
            tags_str += f" (+{len(model.tags)-3})"
        
        # Format pricing
        if model.min_pricing == 0:
            pricing_str = "[green]Free[/green]"
        elif model.min_pricing == model.max_pricing:
            pricing_str = f"${model.min_pricing}"
        else:
            pricing_str = f"${model.min_pricing}-${model.max_pricing}"
        
        # Format status
        status_str = ""
        if show_health:
            if model.health_status == HealthStatus.ONLINE:
                status_str = "[green]Online ✅[/green]"
            elif model.health_status == HealthStatus.OFFLINE:
                status_str = "[red]Offline ❌[/red]"
            elif model.health_status == HealthStatus.TIMEOUT:
                status_str = "[yellow]Timeout ⏱️[/yellow]"
            elif model.health_status == HealthStatus.UNKNOWN:
                status_str = "[dim]Unknown ❓[/dim]"
            else:
                status_str = model.config_status.value
        
        # Add row
        row = [
            model.name,
            model.owner,
            services_str,
            tags_str,
            pricing_str
        ]
        
        if show_health:
            row.append(status_str)
        
        table.add_row(*row)
    
    return table


def create_model_detail_panel(model: ModelInfo) -> Panel:
    """Create a detailed panel for a single model.
    
    Args:
        model: Model to display
        
    Returns:
        Rich Panel object
    """
    # Build content
    content_lines = []
    
    # Basic info
    content_lines.extend([
        f"[bold]Owner:[/bold] {model.owner}",
        f"[bold]Summary:[/bold] {model.summary}",
    ])
    
    if model.description != model.summary:
        content_lines.append(f"[bold]Description:[/bold] {model.description}")
    
    # Status
    content_lines.append(f"[bold]Config Status:[/bold] {model.config_status.value}")
    if model.health_status:
        health_display = _format_health_status(model.health_status)
        content_lines.append(f"[bold]Health Status:[/bold] {health_display}")
    
    # Services
    content_lines.append("\n[bold]Services:[/bold]")
    if model.services:
        for service in model.services:
            status_icon = "✅" if service.enabled else "❌"
            pricing = f"${service.pricing}/{service.charge_type.value}" if service.pricing > 0 else "Free"
            content_lines.append(f"  {status_icon} {service.type.value.title()}: {pricing}")
    else:
        content_lines.append("  No services defined")
    
    # Tags
    if model.tags:
        tags_display = ", ".join(f"[yellow]{tag}[/yellow]" for tag in model.tags)
        content_lines.append(f"\n[bold]Tags:[/bold] {tags_display}")
    
    # Delegate
    if model.delegate_email:
        content_lines.append(f"\n[bold]Delegate:[/bold] {model.delegate_email}")
    
    # Pricing summary
    if model.has_enabled_services:
        if model.min_pricing == model.max_pricing:
            if model.min_pricing == 0:
                content_lines.append(f"\n[bold]Pricing:[/bold] [green]Free[/green]")
            else:
                content_lines.append(f"\n[bold]Pricing:[/bold] ${model.min_pricing}")
        else:
            content_lines.append(f"\n[bold]Pricing:[/bold] ${model.min_pricing} - ${model.max_pricing}")
    
    content = "\n".join(content_lines)
    
    return Panel(
        content,
        title=f"[bold cyan]{model.name}[/bold cyan]",
        border_style="blue",
        padding=(1, 2)
    )


def create_health_summary_table(health_status: Dict[str, HealthStatus]) -> Table:
    """Create a table showing health status summary.
    
    Args:
        health_status: Dictionary mapping model names to health status
        
    Returns:
        Rich Table object
    """
    table = Table(title="Model Health Status", show_header=True, header_style="bold magenta")
    
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Health", justify="center")
    
    for model_name, status in sorted(health_status.items()):
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
            model_name,
            f"[{status_style}]{status.value.title()}[/{status_style}]",
            health_icon
        )
    
    return table


def create_statistics_panel(stats: Dict[str, Any]) -> Panel:
    """Create a panel showing model statistics.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        Rich Panel object
    """
    content_lines = [
        f"[bold]Total Models:[/bold] {stats.get('total_models', 0)}",
        f"[bold]Enabled Models:[/bold] {stats.get('enabled_models', 0)}",
        f"[bold]Disabled Models:[/bold] {stats.get('disabled_models', 0)}",
        "",
        f"[bold]Service Types:[/bold]",
        f"  Chat Models: {stats.get('chat_models', 0)}",
        f"  Search Models: {stats.get('search_models', 0)}",
        "",
        f"[bold]Pricing:[/bold]",
        f"  Free Models: [green]{stats.get('free_models', 0)}[/green]",
        f"  Paid Models: [yellow]{stats.get('paid_models', 0)}[/yellow]",
        "",
        f"[bold]Ownership:[/bold]",
        f"  Total Owners: {stats.get('total_owners', 0)}",
        f"  Avg Models/Owner: {stats.get('avg_models_per_owner', 0):.1f}",
    ]
    
    # Top owners
    top_owners = stats.get('top_owners', [])
    if top_owners:
        content_lines.extend([
            "",
            "[bold]Top Owners:[/bold]"
        ])
        for owner, count in top_owners[:5]:
            content_lines.append(f"  {owner}: {count} models")
    
    content = "\n".join(content_lines)
    
    return Panel(
        content,
        title="[bold cyan]Model Statistics[/bold cyan]",
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


def create_chat_response_panel(response_content: str, model: str, cost: Optional[float] = None, 
                              tokens_used: Optional[int] = None) -> Panel:
    """Create a panel showing chat response.
    
    Args:
        response_content: The response text
        model: Model that generated the response
        cost: Optional cost of the request
        tokens_used: Optional number of tokens used
        
    Returns:
        Rich Panel object
    """
    title_parts = [f"Response from {model}"]
    
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


def create_tree_view(models: List[ModelInfo]) -> Tree:
    """Create a tree view of models grouped by owner.
    
    Args:
        models: List of models to display
        
    Returns:
        Rich Tree object
    """
    tree = Tree("📦 [bold blue]SyftBox Models[/bold blue]")
    
    # Group models by owner
    by_owner = {}
    for model in models:
        if model.owner not in by_owner:
            by_owner[model.owner] = []
        by_owner[model.owner].append(model)
    
    # Add owner branches
    for owner, owner_models in sorted(by_owner.items()):
        owner_branch = tree.add(f"👤 [cyan]{owner}[/cyan] ({len(owner_models)} models)")
        
        for model in sorted(owner_models, key=lambda m: m.name):
            # Model info
            services = [s.type.value for s in model.services if s.enabled]
            services_str = ", ".join(services) if services else "none"
            
            pricing = "Free" if model.min_pricing == 0 else f"${model.min_pricing}"
            
            health_icon = ""
            if model.health_status == HealthStatus.ONLINE:
                health_icon = " ✅"
            elif model.health_status == HealthStatus.OFFLINE:
                health_icon = " ❌"
            elif model.health_status == HealthStatus.TIMEOUT:
                health_icon = " ⏱️"
            
            model_info = f"🤖 [yellow]{model.name}[/yellow] • {services_str} • {pricing}{health_icon}"
            model_branch = owner_branch.add(model_info)
            
            # Add tags if any
            if model.tags:
                tags_str = ", ".join(f"#{tag}" for tag in model.tags[:5])
                if len(model.tags) > 5:
                    tags_str += f" (+{len(model.tags)-5} more)"
                model_branch.add(f"🏷️  {tags_str}")
    
    return tree


def create_comparison_table(models: List[ModelInfo], criteria: List[str]) -> Table:
    """Create a comparison table for multiple models.
    
    Args:
        models: Models to compare
        criteria: List of criteria to compare ('name', 'owner', 'pricing', 'services', 'health')
        
    Returns:
        Rich Table object
    """
    table = Table(title="Model Comparison", show_header=True, header_style="bold magenta")
    
    # Add columns based on criteria
    column_map = {
        'name': ('Name', 'cyan'),
        'owner': ('Owner', 'blue'),
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
    for model in models:
        row = []
        
        for criterion in criteria:
            if criterion == 'name':
                row.append(model.name)
            elif criterion == 'owner':
                row.append(model.owner)
            elif criterion == 'pricing':
                if model.min_pricing == 0:
                    row.append("[green]Free[/green]")
                else:
                    row.append(f"${model.min_pricing}")
            elif criterion == 'services':
                services = [s.type.value for s in model.services if s.enabled]
                row.append(", ".join(services) if services else "none")
            elif criterion == 'health':
                if model.health_status == HealthStatus.ONLINE:
                    row.append("[green]Online ✅[/green]")
                elif model.health_status == HealthStatus.OFFLINE:
                    row.append("[red]Offline ❌[/red]")
                elif model.health_status == HealthStatus.TIMEOUT:
                    row.append("[yellow]Timeout ⏱️[/yellow]")
                else:
                    row.append("[dim]Unknown[/dim]")
            elif criterion == 'tags':
                tags_str = ", ".join(model.tags[:3]) if model.tags else "none"
                if len(model.tags) > 3:
                    tags_str += f" (+{len(model.tags)-3})"
                row.append(tags_str)
        
        table.add_row(*row)
    
    return table


def create_live_health_monitor(models: List[str]) -> Live:
    """Create a live updating health monitor display.
    
    Args:
        models: List of model names to monitor
        
    Returns:
        Rich Live object for updating display
    """
    # Create initial table
    table = Table(title="Live Health Monitor", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Last Check", style="dim")
    
    # Add placeholder rows
    for model_name in models:
        table.add_row(model_name, "[dim]Checking...[/dim]", "[dim]N/A[/dim]")
    
    return Live(table, refresh_per_second=1)


def update_health_monitor_table(table: Table, model_name: str, status: HealthStatus, 
                              last_check: str) -> None:
    """Update a specific row in the health monitor table.
    
    Args:
        table: Table to update
        model_name: Name of the model
        status: Current health status
        last_check: Timestamp of last check
    """
    # This would need more complex implementation to update specific rows
    # For now, it's a placeholder for the live monitoring functionality
    pass