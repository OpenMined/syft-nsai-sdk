#!/usr/bin/env python3
"""
SyftBox SDK CLI

Command-line interface for the SyftBox SDK.
Also serves as the FastAPI app when imported.
"""

import typer
from typing import Optional
from syft_nsai_sdk import find_models, get_models, get_model, display_models, status, is_authenticated

# For FastAPI compatibility
try:
    from app import app as fastapi_app
except ImportError:
    fastapi_app = None

# CLI App
cli_app = typer.Typer(help="SyftBox SDK - Discover and interact with SyftBox models")

@cli_app.command()
def show_status():
    """Check SDK status and configuration."""
    status()

@cli_app.command()
def list_models(
    owner: Optional[str] = typer.Option(None, "--owner", "-o", help="Filter by owner email"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Filter by name"),
):
    """List available models."""
    owners = [owner] if owner else None
    tags = [tag] if tag else None
    
    models = find_models(name=name, tags=tags, owners=owners)
    
    if not models:
        typer.echo("No models found matching the criteria.")
        return
    
    typer.echo(f"Found {len(models)} model(s):")
    display_models(models)

@cli_app.command()
def chat(
    model_name: str = typer.Argument(..., help="Model name to chat with"),
    prompt: str = typer.Argument(..., help="Chat prompt"),
    owner: Optional[str] = typer.Option(None, "--owner", "-o", help="Model owner email"),
    temperature: float = typer.Option(0.7, "--temperature", "-temp", help="Sampling temperature"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-max", help="Maximum tokens"),
):
    """Chat with a specific model."""
    model = get_model(model_name, owner=owner)
    
    if not model:
        typer.echo(f"❌ Model '{model_name}' not found", err=True)
        if owner:
            typer.echo(f"   (searched for owner: {owner})", err=True)
        raise typer.Exit(1)
    
    if "chat" not in model.services:
        typer.echo(f"❌ Model '{model_name}' does not have chat service enabled", err=True)
        typer.echo(f"   Available services: {', '.join(model.services)}", err=True)
        raise typer.Exit(1)
    
    try:
        typer.echo(f"💬 Chatting with {model.name} by {model.owner}...")
        response = model.chat(prompt, temperature=temperature, max_tokens=max_tokens)
        typer.echo(f"\n🤖 Response: {response}")
    except Exception as e:
        typer.echo(f"❌ Chat failed: {e}", err=True)
        raise typer.Exit(1)

@cli_app.command()
def search(
    model_name: str = typer.Argument(..., help="Model name to search"),
    query: str = typer.Argument(..., help="Search query"),
    owner: Optional[str] = typer.Option(None, "--owner", "-o", help="Model owner email"),
):
    """Search using a specific model."""
    model = get_model(model_name, owner=owner)
    
    if not model:
        typer.echo(f"❌ Model '{model_name}' not found", err=True)
        if owner:
            typer.echo(f"   (searched for owner: {owner})", err=True)
        raise typer.Exit(1)
    
    if "search" not in model.services:
        typer.echo(f"❌ Model '{model_name}' does not have search service enabled", err=True)
        typer.echo(f"   Available services: {', '.join(model.services)}", err=True)
        raise typer.Exit(1)
    
    try:
        typer.echo(f"🔍 Searching with {model.name} by {model.owner}...")
        results = model.search(query)
        typer.echo(f"\n📋 Results:\n{results}")
    except Exception as e:
        typer.echo(f"❌ Search failed: {e}", err=True)
        raise typer.Exit(1)

@cli_app.command()
def demo():
    """Run an interactive demo of the SDK."""
    typer.echo("🚀 Starting SyftBox SDK Demo...")
    
    # Check status
    status()
    
    if not is_authenticated:
        typer.echo("\n❌ SDK not configured. Please set up SyftBox first:")
        typer.echo("   1. Install: curl -fsSL https://syftbox.net/install.sh | sh")
        typer.echo("   2. Login: syftbox login")
        typer.echo("   3. Start: syftbox")
        return
    
    # Show available models
    models = get_models()
    typer.echo(f"\n📦 Found {len(models)} models:")
    display_models(models)
    
    if not models:
        typer.echo("💡 No models found. Make sure SyftBox is synced and contains published models.")
        return
    
    # Interactive model selection
    model_names = [f"{m.name} (by {m.owner})" for m in models]
    typer.echo("\n🤖 Available models:")
    for i, name in enumerate(model_names):
        typer.echo(f"  {i+1}. {name}")
    
    try:
        choice = typer.prompt("Select a model number", type=int)
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            typer.echo(f"\n✅ Selected: {selected_model.name}")
            typer.echo(f"   Services: {', '.join(selected_model.services)}")
            
            if "chat" in selected_model.services:
                prompt = typer.prompt("\n💬 Enter a chat prompt")
                typer.echo("🤖 Processing...")
                response = selected_model.chat(prompt)
                typer.echo(f"\n🎯 Response: {response}")
            else:
                typer.echo("❌ Selected model doesn't have chat service")
        else:
            typer.echo("❌ Invalid selection")
    except KeyboardInterrupt:
        typer.echo("\n👋 Demo cancelled")
    except Exception as e:
        typer.echo(f"❌ Demo failed: {e}")

# For CLI usage
if __name__ == "__main__":
    cli_app()

# For FastAPI usage (when imported)
app = fastapi_app if fastapi_app else None