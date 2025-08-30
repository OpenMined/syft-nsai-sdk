"""
Custom ModelsList class that provides a show() method for displaying models in a widget.
"""

from typing import List, Optional, Any
from .models_widget import get_models_widget_html
import tempfile
import webbrowser
import os
from pathlib import Path


class ModelsList:
    """A custom list-like class that wraps ModelInfo objects and provides a show() method."""
    
    def __init__(self, models: List[Any], client=None):
        """Initialize with a list of models and optional client reference.
        
        Args:
            models: List of ModelInfo objects
            client: Optional reference to the SyftBoxClient for context
        """
        self._models = models
        self._client = client
        self._widget_html = None
    
    def __len__(self):
        """Return the number of models."""
        return len(self._models)
    
    def __getitem__(self, index):
        """Get a model by index."""
        return self._models[index]
    
    def __iter__(self):
        """Iterate over models."""
        return iter(self._models)
    
    def __contains__(self, item):
        """Check if a model is in the list."""
        return item in self._models
    
    def __repr__(self):
        """String representation."""
        return f"ModelsList({len(self._models)} models)"
    
    def __str__(self):
        """Human-readable string representation."""
        return f"ModelsList with {len(self._models)} models"
    
    def _is_jupyter_notebook(self) -> bool:
        """Detect if we're running in a Jupyter notebook environment."""
        try:
            # Check for IPython kernel
            import IPython
            ipython = IPython.get_ipython()
            if ipython is not None:
                # Check if we're in a notebook (not in terminal)
                if hasattr(ipython, 'kernel') and ipython.kernel is not None:
                    return True
                # Alternative check for notebook environment
                if 'ipykernel' in str(type(ipython)).lower():
                    return True
        except ImportError:
            pass
        
        # Check environment variables
        if os.environ.get('JUPYTER_RUNTIME_DIR'):
            return True
        
        # Check for Jupyter-related environment variables
        jupyter_vars = ['JUPYTER_KERNEL_ID', 'JPY_PARENT_PID', 'JUPYTER_TOKEN']
        if any(os.environ.get(var) for var in jupyter_vars):
            return True
        
        return False
    
    def _display_in_notebook(self, html: str) -> None:
        """Display HTML widget directly in Jupyter notebook."""
        try:
            # Try to import and use IPython display
            import IPython.display as display
            display.display(display.HTML(html))
            print("✅ Models widget displayed in notebook")
        except ImportError:
            # Fallback: save and show file path
            print("⚠️ IPython not available, saving to file instead")
            self._save_and_open_file(html)
        except Exception as e:
            # Fallback: save and show file path
            print(f"⚠️ Could not display in notebook: {e}")
            self._save_and_open_file(html)
    
    def _save_and_open_file(self, html: str, output_path: Optional[str] = None) -> str:
        """Save HTML to file and optionally open in browser."""
        if output_path:
            file_path = Path(output_path)
        else:
            file_path = Path("syftbox_models_widget.html")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"✅ Models widget saved to: {file_path.absolute()}")
        
        # Try to open in browser
        try:
            webbrowser.open(f"file://{file_path.absolute()}")
            print(f"🌐 Opened widget in browser")
        except Exception as e:
            print(f"⚠️ Could not auto-open browser: {e}")
            print(f"Please open {file_path} manually in your browser")
        
        return str(file_path)
    
    def show_models(self, 
             page: int = 1,
             items_per_page: int = 50,
             current_user_email: str = "",
             auto_open: bool = True,
             save_to_file: bool = False,
             output_path: Optional[str] = None,
             open_in_browser: bool = False) -> str:
        """Display the models in an interactive HTML widget.
        
        This method focuses on display and presentation of already-discovered models.
        For filtering and discovery, use discover_models() with appropriate parameters.
        
        This method automatically detects the environment:
        - In Jupyter notebooks: displays widget directly in the notebook (default)
        - In other environments: saves to file and opens in browser
        
        Args:
            page: Starting page number
            items_per_page: Models per page
            current_user_email: Current user's email for context
            auto_open: Automatically open in browser (for non-notebook environments)
            save_to_file: Force save to file even in notebooks
            output_path: Custom output path for HTML file
            open_in_browser: Force open in browser even in Jupyter notebooks (default: False)
            
        Returns:
            Path to the generated HTML file (or empty string if displayed in notebook)
        """
        # Convert models to widget-compatible format
        widget_models = []
        for model in self._models:
            try:
                widget_models.append({
                    "name": model.name,
                    "owner": model.owner,
                    "summary": model.summary,
                    "description": getattr(model, 'description', ''),
                    "tags": model.tags,
                    "services": [
                        {
                            "type": service.type.value,
                            "enabled": service.enabled,
                            "pricing": service.pricing,
                            "charge_type": service.charge_type.value
                        }
                        for service in model.services
                    ],
                    "config_status": model.config_status.value,
                    "health_status": model.health_status.value if model.health_status else None,
                    "min_pricing": model.min_pricing,
                    "max_pricing": model.max_pricing
                })
            except Exception as e:
                # Skip models that can't be converted
                continue
        
        # Generate widget HTML (no filtering parameters needed)
        html = get_models_widget_html(
            models=widget_models,
            page=page,
            items_per_page=items_per_page,
            current_user_email=current_user_email
        )
        
        # Check if we should force browser opening or if we're in a Jupyter notebook
        if self._is_jupyter_notebook() and not save_to_file and not open_in_browser:
            # Display directly in notebook (default Jupyter behavior)
            self._display_in_notebook(html)
            return ""  # Return empty string since we're displaying in notebook
        else:
            # Save to file and optionally open in browser
            return self._save_and_open_file(html, output_path)
    
    def show(self, **kwargs):
        """Alias for show_models() method for backward compatibility."""
        return self.show_models(**kwargs)
    
    def display(self, **kwargs):
        """Alias for show() method."""
        return self.show(**kwargs)
    
    def to_widget(self, **kwargs):
        """Generate widget HTML without displaying.
        
        This method is mainly for advanced users who want to customize the display.
        For most users, use show() instead.
        """
        # Convert models to widget-compatible format
        widget_models = []
        for model in self._models:
            try:
                widget_models.append({
                    "name": model.name,
                    "owner": model.owner,
                    "summary": model.summary,
                    "description": getattr(model, 'description', ''),
                    "tags": model.tags,
                    "services": [
                        {
                            "type": service.type.value,
                            "enabled": service.enabled,
                            "pricing": service.pricing,
                            "charge_type": service.charge_type.value
                        }
                        for service in model.services
                    ],
                    "config_status": model.config_status.value,
                    "health_status": model.health_status.value if model.health_status else None,
                    "min_pricing": model.min_pricing,
                    "max_pricing": model.max_pricing
                })
            except Exception as e:
                continue
        
        return get_models_widget_html(
            models=widget_models,
            **kwargs
        )
    
    # List-like methods
    def append(self, model):
        """Add a model to the list."""
        self._models.append(model)
    
    def extend(self, models):
        """Extend the list with more models."""
        self._models.extend(models)
    
    def insert(self, index, model):
        """Insert a model at a specific index."""
        self._models.insert(index, model)
    
    def remove(self, model):
        """Remove a model from the list."""
        self._models.remove(model)
    
    def pop(self, index=-1):
        """Remove and return a model at the specified index."""
        return self._models.pop(index)
    
    def clear(self):
        """Clear all models from the list."""
        self._models.clear()
    
    def index(self, model):
        """Return the index of a model."""
        return self._models.index(model)
    
    def count(self, model):
        """Return the number of occurrences of a model."""
        return self._models.count(model)
    
    def sort(self, key=None, reverse=False):
        """Sort the models list."""
        self._models.sort(key=key, reverse=reverse)
    
    def reverse(self):
        """Reverse the models list."""
        self._models.reverse()
    
    def copy(self):
        """Create a shallow copy of the models list."""
        return ModelsList(self._models.copy(), self._client)
    
    # Additional utility methods
    def filter(self, **kwargs):
        """Filter models by criteria and return a new ModelsList."""
        filtered = []
        for model in self._models:
            # Apply filters
            if 'name' in kwargs and kwargs['name'].lower() not in model.name.lower():
                continue
            if 'owner' in kwargs and kwargs['owner'].lower() not in model.owner.lower():
                continue
            if 'tags' in kwargs:
                if not any(tag.lower() in [t.lower() for t in model.tags] for tag in kwargs['tags']):
                    continue
            if 'service_type' in kwargs:
                if not model.supports_service(kwargs['service_type']):
                    continue
            if 'max_cost' in kwargs and model.min_pricing > kwargs['max_cost']:
                continue
            if kwargs.get('free_only', False) and model.min_pricing > 0:
                continue
            
            filtered.append(model)
        
        return ModelsList(filtered, self._client)
    
    def search(self, query: str):
        """Search models by query string."""
        query_lower = query.lower()
        results = []
        
        for model in self._models:
            searchable_content = [
                model.name,
                model.owner,
                model.summary,
                getattr(model, 'description', ''),
                ' '.join(model.tags)
            ]
            
            if any(query_lower in content.lower() for content in searchable_content):
                results.append(model)
        
        return ModelsList(results, self._client)
    
    def get_by_owner(self, owner: str):
        """Get models by specific owner."""
        return ModelsList([m for m in self._models if m.owner == owner], self._client)
    
    def get_by_service(self, service_type: str):
        """Get models that support a specific service."""
        return ModelsList([m for m in self._models if m.supports_service(service_type)], self._client)
    
    def get_free_models(self):
        """Get only free models."""
        return ModelsList([m for m in self._models if m.min_pricing == 0], self._client)
    
    def get_paid_models(self):
        """Get only paid models."""
        return ModelsList([m for m in self._models if m.min_pricing > 0], self._client)
    
    def summary(self):
        """Print a summary of the models."""
        if not self._models:
            print("No models found.")
            return
        
        print(f"Found {len(self._models)} models:")
        print("-" * 50)
        
        # Group by owner
        by_owner = {}
        for model in self._models:
            if model.owner not in by_owner:
                by_owner[model.owner] = []
            by_owner[model.owner].append(model)
        
        for owner, models in sorted(by_owner.items()):
            print(f"\n📧 {owner} ({len(models)} models):")
            for model in sorted(models, key=lambda m: m.name):
                services = ", ".join([s.type.value for s in model.services if s.enabled])
                pricing = f"${model.min_pricing}" if model.min_pricing > 0 else "Free"
                health = ""
                if hasattr(model, 'health_status') and model.health_status:
                    if model.health_status.value == 'online':
                        health = " ✅"
                    elif model.health_status.value == 'offline':
                        health = " ❌"
                    elif model.health_status.value == 'timeout':
                        health = " ⏱️"
                
                print(f"  • {model.name} ({services}) - {pricing}{health}")
    
    def to_dict(self):
        """Convert models to list of dictionaries."""
        return [
            {
                "name": model.name,
                "owner": model.owner,
                "summary": model.summary,
                "description": getattr(model, 'description', ''),
                "tags": model.tags,
                "services": [
                    {
                        "type": service.type.value,
                        "enabled": service.enabled,
                        "pricing": service.pricing,
                        "charge_type": service.charge_type.value
                    }
                    for service in model.services
                ],
                "config_status": model.config_status.value,
                "health_status": model.health_status.value if hasattr(model, 'health_status') and model.health_status else None,
                "min_pricing": model.min_pricing,
                "max_pricing": model.max_pricing
            }
            for model in self._models
        ]
    
    def to_json(self):
        """Convert models to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)
