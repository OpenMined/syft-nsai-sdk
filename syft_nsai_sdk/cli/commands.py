"""
CLI commands for SyftBox NSAI SDK
"""
import asyncio
import json
import sys
import os
from dotenv import load_dotenv
from typing import Optional, List
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from ..main import SyftBoxClient
from ..core.types import ServiceType, HealthStatus
from ..core.exceptions import (
    SyftBoxNotFoundError, ModelNotFoundError, PaymentError,
    AuthenticationError, NetworkError
)

# Create console and app
console = Console()
app = typer.Typer(
    name="syftbox-sdk",
    help="SyftBox NSAI SDK - Discover and use AI models across the SyftBox network",
    add_completion=False,
    rich_markup_mode="rich"
)

# Global client instance
_client: Optional[SyftBoxClient] = None

def get_client() -> SyftBoxClient:
    """Get or create client instance."""
    global _client
    if _client is None:
        try:
            _client = SyftBoxClient()
        except SyftBoxNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print("\n[yellow]SyftBox installation required:[/yellow]")
            console.print("1. Quick install: [cyan]curl -LsSf https://install.syftbox.openmined.org | sh[/cyan]")
            console.print("2. Setup: [cyan]syftbox setup[/cyan]")
            console.print("3. Retry this command")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error initializing SDK:[/red] {e}")
            raise typer.Exit(1)
    return _client

@app.command()
def list_models(
    service: Optional[str] = typer.Option(None, "--service", "-s", help="Filter by service type (chat, search)"),
    owner: Optional[str] = typer.Option(None, "--owner", "-o", help="Filter by owner email"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Filter by tags (comma-separated)"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", "-c", help="Maximum cost per request"),
    health_check: str = typer.Option("auto", "--health-check", "-h", help="Health check mode (auto, always, never)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, summary)"),
    free_only: bool = typer.Option(False, "--free-only", help="Show only free models"),
    paid_only: bool = typer.Option(False, "--paid-only", help="Show only paid models"),
):
    """List available models with optional filtering."""
    client = get_client()
    
    with console.status("[bold blue]Discovering models..."):
        try:
            # Parse service type
            service_type = None
            if service:
                try:
                    service_type = ServiceType(service.lower())
                except ValueError:
                    console.print(f"[red]Invalid service type:[/red] {service}")
                    console.print("Valid options: chat, search")
                    raise typer.Exit(1)
            
            # Parse tags
            tag_list = None
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",")]
            
            # Discover models
            kwargs = {
                "service_type": service_type,
                "owner": owner,
                "tags": tag_list,
                "max_cost": max_cost,
                "health_check": health_check,
                "free_only": free_only,
                "paid_only": paid_only,
            }
            
            models = client.discover_models(**{k: v for k, v in kwargs.items() if v is not None})
            
        except Exception as e:
            console.print(f"[red]Error discovering models:[/red] {e}")
            raise typer.Exit(1)
    
    if not models:
        console.print("[yellow]No models found matching the criteria.[/yellow]")
        return
    
    # Display results
    if format == "table":
        console.print(client.list_models(service_type=service_type, health_check=health_check))
    elif format == "json":
        model_dicts = [client._model_to_dict(model) for model in models]
        print(json.dumps(model_dicts, indent=2))
    elif format == "summary":
        console.print(client._format_models_summary(models))
    else:
        console.print(f"[red]Invalid format:[/red] {format}")
        raise typer.Exit(1)

@app.command()
def model_info(
    name: str = typer.Argument(..., help="Model name to show info for"),
    owner: Optional[str] = typer.Option(None, "--owner", "-o", help="Model owner (if ambiguous)"),
):
    """Show detailed information about a specific model."""
    client = get_client()
    
    with console.status(f"[bold blue]Looking up model '{name}'..."):
        model = client.find_model(name, owner)
    
    if not model:
        console.print(f"[red]Model not found:[/red] {name}")
        if owner:
            console.print(f"Searched for models owned by: {owner}")
        
        # Suggest similar models
        all_models = client.discover_models(health_check="never")
        similar = [m for m in all_models if name.lower() in m.name.lower()]
        if similar:
            console.print(f"\n[yellow]Did you mean one of these?[/yellow]")
            for m in similar[:5]:
                console.print(f"  • {m.name} (by {m.owner})")
        
        raise typer.Exit(1)
    
    console.print(client.show_model_details(name, owner))

@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use"),
    max_cost: float = typer.Option(1.0, "--max-cost", "-c", help="Maximum cost willing to pay"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum tokens to generate"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Sampling temperature (0.0-1.0)"),
    preference: str = typer.Option("balanced", "--preference", "-p", help="Model selection preference"),
):
    """Send a chat message to an AI model."""
    client = get_client()
    
    async def run_chat():
        with console.status("[bold blue]Sending message..."):
            try:
                response = await client.chat(
                    message=message,
                    model_name=model,
                    max_cost=max_cost,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    preference=preference
                )
                
                # Display response
                console.print(f"\n[bold green]Response from {response.model}:[/bold green]")
                console.print(Panel(response.message.content, border_style="green"))
                
                # Show cost if applicable
                if response.cost and response.cost > 0:
                    console.print(f"[dim]Cost: ${response.cost}[/dim]")
                
                # Show usage stats
                if response.usage.total_tokens > 0:
                    console.print(f"[dim]Tokens used: {response.usage.total_tokens}[/dim]")
                
            except ModelNotFoundError as e:
                console.print(f"[red]Model not found:[/red] {e}")
                raise typer.Exit(1)
            except PaymentError as e:
                console.print(f"[red]Payment error:[/red] {e}")
                console.print("\n[yellow]Try:[/yellow]")
                console.print("• Use --max-cost to increase cost limit")
                console.print("• Run [cyan]syftbox-sdk setup-accounting[/cyan] to configure payments")
                console.print("• Use [cyan]syftbox-sdk list-models --free-only[/cyan] to find free models")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]Chat failed:[/red] {e}")
                raise typer.Exit(1)
    
    asyncio.run(run_chat())

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use"),
    max_cost: float = typer.Option(1.0, "--max-cost", "-c", help="Maximum cost willing to pay"),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum number of results"),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Minimum similarity score"),
):
    """Search documents using an AI model."""
    client = get_client()
    
    async def run_search():
        with console.status("[bold blue]Searching..."):
            try:
                response = await client.search(
                    query=query,
                    model_name=model,
                    max_cost=max_cost,
                    limit=limit,
                    similarity_threshold=threshold
                )
                
                # Display results
                console.print(f"\n[bold green]Search Results for: '{query}'[/bold green]")
                
                if not response.results:
                    console.print("[yellow]No results found.[/yellow]")
                    return
                
                for i, result in enumerate(response.results, 1):
                    console.print(f"\n[bold]{i}. Score: {result.score:.3f}[/bold]")
                    console.print(Panel(result.content[:200] + "..." if len(result.content) > 200 else result.content))
                    
                    if result.metadata and "filename" in result.metadata:
                        console.print(f"[dim]Source: {result.metadata['filename']}[/dim]")
                
                # Show cost if applicable
                if response.cost and response.cost > 0:
                    console.print(f"\n[dim]Cost: ${response.cost}[/dim]")
                
            except ModelNotFoundError as e:
                console.print(f"[red]Model not found:[/red] {e}")
                raise typer.Exit(1)
            except PaymentError as e:
                console.print(f"[red]Payment error:[/red] {e}")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]Search failed:[/red] {e}")
                raise typer.Exit(1)
    
    asyncio.run(run_search())

@app.command()
def health_check(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Check specific model"),
    all_models: bool = typer.Option(False, "--all", help="Check all models"),
    service: Optional[str] = typer.Option(None, "--service", "-s", help="Filter by service type"),
    timeout: float = typer.Option(2.0, "--timeout", "-t", help="Health check timeout"),
):
    """Check health status of models."""
    client = get_client()
    
    async def run_health_check():
        if model:
            # Check specific model
            with console.status(f"[bold blue]Checking health of '{model}'..."):
                try:
                    status = await client.check_model_health(model, timeout)
                    icon = "✅" if status == HealthStatus.ONLINE else "❌" if status == HealthStatus.OFFLINE else "⏱️" if status == HealthStatus.TIMEOUT else "❓"
                    console.print(f"{model}: {status.value.title()} {icon}")
                except ModelNotFoundError:
                    console.print(f"[red]Model not found:[/red] {model}")
                    raise typer.Exit(1)
        elif all_models:
            # Check all models
            service_type = ServiceType(service) if service else None
            
            with console.status("[bold blue]Checking health of all models..."):
                try:
                    health_status = await client.check_all_models_health(service_type, timeout)
                    
                    if not health_status:
                        console.print("[yellow]No models found to check.[/yellow]")
                        return
                    
                    # Create health summary table
                    table = Table(title="Model Health Status")
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
                    
                    console.print(table)
                    
                    # Summary
                    total = len(health_status)
                    online = sum(1 for s in health_status.values() if s == HealthStatus.ONLINE)
                    offline = sum(1 for s in health_status.values() if s == HealthStatus.OFFLINE)
                    timeout = sum(1 for s in health_status.values() if s == HealthStatus.TIMEOUT)
                    
                    console.print(f"\n[bold]Summary:[/bold] {online} online, {offline} offline, {timeout} timeout out of {total} total")
                    
                except Exception as e:
                    console.print(f"[red]Health check failed:[/red] {e}")
                    raise typer.Exit(1)
        else:
            console.print("[red]Error:[/red] Must specify either --model or --all")
            raise typer.Exit(1)
    
    asyncio.run(run_health_check())

@app.command()
def account_status():
    """Show accounting service status."""
    client = get_client()
    
    status = client.show_accounting_status()
    console.print(status)

@app.command()
def account_balance():
    """Show current account balance."""
    client = get_client()
    
    async def show_balance():
        try:
            account_info = await client.get_account_info()
            
            if "error" in account_info:
                console.print(f"[red]Error:[/red] {account_info['error']}")
                console.print("\n[yellow]Try running:[/yellow] [cyan]syftbox-sdk setup-accounting[/cyan]")
                raise typer.Exit(1)
            
            console.print(f"[bold]Account Balance[/bold]")
            console.print(f"Email: {account_info['email']}")
            console.print(f"Balance: [green]${account_info['balance']:.2f}[/green]")
            
        except AuthenticationError as e:
            console.print(f"[red]Authentication error:[/red] {e}")
            console.print("\n[yellow]Try running:[/yellow] [cyan]syftbox-sdk setup-accounting[/cyan]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error getting balance:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(show_balance())

@app.command()
def setup_accounting():
    """Setup accounting service credentials interactively."""
    # TODO: Implement interactive setup
    client = get_client()
    
    async def setup():
        try:
            load_dotenv()
            console.print("[bold blue]Setting up SyftBox Accounting Service[/bold blue]")
            console.print("This will configure payment credentials for using paid models.\n")
            
            # Get service URL
            default_url = os.getenv("SYFTBOX_ACCOUNTING_URL", "")
            service_url = typer.prompt(f"Accounting service URL", default=default_url)
            
            # Check if user has existing account
            has_account = typer.confirm("Do you have an existing accounting service account?")
            
            if has_account:
                # Existing account
                email = typer.prompt("Email")
                password = typer.prompt("Password", hide_input=True)
                
                with console.status("[bold blue]Verifying credentials..."):
                    await client.setup_accounting(email, password, service_url)
                
                console.print("[green]✅ Accounting configured successfully![/green]")
                
                # Show balance
                account_info = await client.get_account_info()
                console.print(f"Balance: ${account_info['balance']:.2f}")
                
            else:
                # Create new account
                console.print("\n[bold]Creating new account:[/bold]")
                email = typer.prompt("Email")
                organization = typer.prompt("Organization (optional)", default="")
                password = typer.prompt("Password (leave empty for auto-generated)", hide_input=True, default="")
                
                # TODO: This would need implementation in the main client
                console.print("[yellow]Account creation via CLI not yet implemented.[/yellow]")
                console.print("Please create an account at the accounting service web interface first.")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Setup cancelled.[/yellow]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Setup failed:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(setup())

@app.command()
def stats():
    """Show statistics about discovered models."""
    client = get_client()
    
    with console.status("[bold blue]Gathering statistics..."):
        try:
            stats = client.get_statistics()
        except Exception as e:
            console.print(f"[red]Error gathering stats:[/red] {e}")
            raise typer.Exit(1)
    
    # Create stats table
    table = Table(title="SyftBox Model Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    
    table.add_row("Total Models", str(stats["total_models"]))
    table.add_row("Enabled Models", str(stats["enabled_models"]))
    table.add_row("Disabled Models", str(stats["disabled_models"]))
    table.add_row("Chat Models", str(stats["chat_models"]))
    table.add_row("Search Models", str(stats["search_models"]))
    table.add_row("Free Models", str(stats["free_models"]))
    table.add_row("Paid Models", str(stats["paid_models"]))
    table.add_row("Total Owners", str(stats["total_owners"]))
    table.add_row("Avg Models/Owner", f"{stats['avg_models_per_owner']:.1f}")
    
    console.print(table)
    
    # Top owners
    if stats["top_owners"]:
        console.print("\n[bold]Top Model Owners:[/bold]")
        for owner, count in stats["top_owners"]:
            console.print(f"  {owner}: {count} models")

@app.command()
def version():
    """Show SDK version information."""
    from .. import __version__, __author__
    
    console.print(f"[bold]SyftBox NSAI SDK[/bold]")
    console.print(f"Version: {__version__}")
    console.print(f"Author: {__author__}")
    console.print(f"Python: {sys.version.split()[0]}")

# Error handling
@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode (less output)"),
):
    """SyftBox NSAI SDK CLI."""
    if debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    if quiet:
        console.quiet = True

def cli_main():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()