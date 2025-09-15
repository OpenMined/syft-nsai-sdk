"""
Custom ServicesList class that provides a show() method for displaying services in a widget.
"""
import webbrowser
import os

from typing import List, Optional, Any
from pathlib import Path

from ..views.services_widget import get_services_widget_html

class ServicesList:
    """A custom list-like class that wraps ServiceInfo objects and provides a show() method."""

    def __init__(self, services: List[Any], client=None):
        """Initialize with a list of services and optional client reference.

        Args:
            services: List of ServiceInfo objects
            client: Optional reference to the Client for context
        """
        self._services = services
        self._client = client
        self._widget_html = None
    
    def __len__(self):
        """Return the number of services."""
        return len(self._services)

    def __getitem__(self, index):
        """Get a service by index."""
        return self._services[index]
    
    def __iter__(self):
        """Iterate over services."""
        return iter(self._services)
    
    def __contains__(self, item):
        """Check if a service is in the list."""
        return item in self._services

    def __repr__(self):
        """String representation."""
        return f"ServicesList({len(self._services)} services)"

    def __str__(self):
        """Human-readable string representation."""
        return f"ServicesList with {len(self._services)} services"

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
            print("Services widget displayed in notebook")
        except ImportError:
            # Fallback: save and show file path
            print("IPython not available, saving to file instead")
            self._save_and_open_file(html)
        except Exception as e:
            # Fallback: save and show file path
            print(f"Could not display in notebook: {e}")
            self._save_and_open_file(html)
    
    def _save_and_open_file(self, html: str, output_path: Optional[str] = None) -> str:
        """Save HTML to file and optionally open in browser."""
        if output_path:
            file_path = Path(output_path)
        else:
            file_path = Path("syftbox_services_widget.html")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"Services widget saved to: {file_path.absolute()}")
        
        # Try to open in browser
        try:
            webbrowser.open(f"file://{file_path.absolute()}")
            print(f"🌐 Opened widget in browser")
        except Exception as e:
            print(f"⚠️ Could not auto-open browser: {e}")
            print(f"Please open {file_path} manually in your browser")
        
        return str(file_path)
    
    def show_services(self, **kwargs) -> None:
        """Display services using the enhanced widget (alias for show method)."""
        self.show(**kwargs)
    
    def show(self, 
             page: int = 1,
             items_per_page: int = 50,
             current_user_email: str = "",
             save_to_file: bool = False,
             output_path: Optional[str] = None,
             open_in_browser: bool = False) -> str:
        """Display the services in an interactive HTML widget.
        
        This method focuses on display and presentation of already-discovered services.
        For filtering and discovery, use discover_services() with appropriate parameters.
        
        This method automatically detects the environment:
        - In Jupyter notebooks: displays widget directly in the notebook (default)
        - In other environments: saves to file and opens in browser
        
        Args:
            page: Starting page number
            items_per_page: Services per page
            current_user_email: Current user's email for context
            auto_open: Automatically open in browser (for non-notebook environments)
            save_to_file: Force save to file even in notebooks
            output_path: Custom output path for HTML file
            open_in_browser: Force open in browser even in Jupyter notebooks (default: False)
            
        Returns:
            Path to the generated HTML file (or empty string if displayed in notebook)
        """
        # Convert services to widget-compatible format
        widget_services = []
        for _service in self._services:
            try:
                widget_services.append({
                    "name": _service.name,
                    "datasite": _service.datasite,
                    "summary": _service.summary,
                    "description": getattr(_service, 'description', ''),
                    "tags": _service.tags,
                    "services": [
                        {
                            "type": service.type.value,
                            "enabled": service.enabled,
                            "pricing": service.pricing,
                            "charge_type": service.charge_type.value
                        }
                        for service in _service.services
                    ],
                    "config_status": _service.config_status.value,
                    "health_status": _service.health_status.value if _service.health_status else None,
                    "min_pricing": _service.min_pricing,
                    "max_pricing": _service.max_pricing
                })
            # except Exception as e:
            #     # Skip services that can't be converted
            #     continue
            except Exception as e:
                print(f"ERROR processing service {_service.name}: {e}")
                print(f"Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate widget HTML (no filtering parameters needed)
        html = get_services_widget_html(
            services=widget_services,
            page=page,
            items_per_page=items_per_page,
            current_user_email=current_user_email
        )
        
        # Check if we should force browser opening or if we're in a Jupyter notebook
        if self._is_jupyter_notebook() and not save_to_file and not open_in_browser:
            # Display directly in notebook (default Jupyter behavior)
            self._display_in_notebook(html)
        else:
            # Save to file and optionally open in browser
            self._save_and_open_file(html, output_path)
    
    def to_widget(self, **kwargs):
        """Generate widget HTML without displaying.
        
        This method is mainly for advanced users who want to customize the display.
        For most users, use show() instead.
        """
        # Convert services to widget-compatible format
        widget_services = []
        for _service in self._services:
            try:
                widget_services.append({
                    "name": _service.name,
                    "datasite": _service.datasite,
                    "summary": _service.summary,
                    "description": getattr(_service, 'description', ''),
                    "tags": _service.tags,
                    "services": [
                        {
                            "type": service.type.value,
                            "enabled": service.enabled,
                            "pricing": service.pricing,
                            "charge_type": service.charge_type.value
                        }
                        for service in _service.services
                    ],
                    "config_status": _service.config_status.value,
                    "health_status": _service.health_status.value if _service.health_status else None,
                    "min_pricing": _service.min_pricing,
                    "max_pricing": _service.max_pricing
                })
            except Exception as e:
                continue

        return get_services_widget_html(
            services=widget_services,
            **kwargs
        )
    
    # List-like methods
    def append(self, service):
        """Add a service to the list."""
        self._services.append(service)

    def extend(self, services):
        """Extend the list with more services."""
        self._services.extend(services)

    def insert(self, index, service):
        """Insert a service at a specific index."""
        self._services.insert(index, service)

    def remove(self, service):
        """Remove a service from the list."""
        self._services.remove(service)

    def pop(self, index=-1):
        """Remove and return a service at the specified index."""
        return self._services.pop(index)

    def clear(self):
        """Clear all services from the list."""
        self._services.clear()

    def index(self, service):
        """Return the index of a service."""
        return self._services.index(service)

    def count(self, service):
        """Return the number of occurrences of a service."""
        return self._services.count(service)

    def sort(self, key=None, reverse=False):
        """Sort the services list."""
        self._services.sort(key=key, reverse=reverse)
    
    def reverse(self):
        """Reverse the services list."""
        self._services.reverse()

    def copy(self):
        """Create a shallow copy of the services list."""
        return ServicesList(self._services.copy(), self._client)

    # Additional utility methods
    def filter(self, **kwargs):
        """Filter services by criteria and return a new ServicesList."""
        filtered = []
        for _service in self._services:
            # Apply filters
            if 'name' in kwargs and kwargs['name'].lower() not in _service.name.lower():
                continue
            if 'datasite' in kwargs and kwargs['datasite'].lower() not in _service.datasite.lower():
                continue
            if 'tags' in kwargs:
                if not any(tag.lower() in [t.lower() for t in _service.tags] for tag in kwargs['tags']):
                    continue
            if 'service_type' in kwargs:
                if not _service.supports_service(kwargs['service_type']):
                    continue
            if 'max_cost' in kwargs and _service.min_pricing > kwargs['max_cost']:
                continue
            if kwargs.get('free_only', False) and _service.min_pricing > 0:
                continue

            filtered.append(_service)
        
        return ServicesList(filtered, self._client)
    
    def search(self, query: str):
        """Search services by query string."""
        query_lower = query.lower()
        results = []

        for _service in self._services:
            searchable_content = [
                _service.name,
                _service.datasite,
                _service.summary,
                getattr(_service, 'description', ''),
                ' '.join(_service.tags)
            ]
            
            if any(query_lower in content.lower() for content in searchable_content):
                results.append(_service)
        
        return ServicesList(results, self._client)
    
    def get_by_datasite(self, datasite: str):
        """Get services by specific datasite."""
        return ServicesList([_service for _service in self._services if _service.datasite == datasite], self._client)

    def get_by_service(self, service_type: str):
        """Get services that support a specific service."""
        return ServicesList([_service for _service in self._services if _service.supports_service(service_type)], self._client)
    
    def get_free_services(self):
        """Get only free services."""
        return ServicesList([_service for _service in self._services if _service.min_pricing == 0], self._client)

    def get_paid_services(self):
        """Get only paid services."""
        return ServicesList([_service for _service in self._services if _service.min_pricing > 0], self._client)

    def summary(self):
        """Print a summary of the services in a clean, client-repr style."""
        if not self._services:
            print("No services found.")
            return

        total = len(self._services)
        chat_count = len([s for s in self._services if s.supports_service('chat')])
        search_count = len([s for s in self._services if s.supports_service('search')])
        free_count = len([s for s in self._services if s.min_pricing == 0])
        paid_count = len([s for s in self._services if s.min_pricing > 0])
        online_count = len([s for s in self._services if hasattr(s, 'health_status') and s.health_status and s.health_status.value == 'online'])

        print(f"SyftBox Services Summary")
        print(f"")
        print(f"Services:         {total} services found")
        if chat_count > 0:
            print(f"Chat services:    {chat_count}")
        if search_count > 0:
            print(f"Search services:  {search_count}")
        if free_count > 0:
            print(f"Free services:    {free_count}")
        if paid_count > 0:
            print(f"Paid services:    {paid_count}")
        if online_count > 0:
            print(f"Online services:  {online_count}")
        print(f"")

        # Group by datasite
        by_datasite = {}
        for _service in self._services:
            if _service.datasite not in by_datasite:
                by_datasite[_service.datasite] = []
            by_datasite[_service.datasite].append(_service)

        print("Available services:")
        for datasite, _services in sorted(by_datasite.items()):
            print(f"  📧 {datasite} ({len(_services)} services)")
            for _service in sorted(_services, key=lambda s: s.name):
                services = ", ".join([s.type.value for s in _service.services if s.enabled])
                pricing = f"${_service.min_pricing}" if _service.min_pricing > 0 else "Free"
                health = ""
                if hasattr(_service, 'health_status') and _service.health_status:
                    if _service.health_status.value == 'online':
                        health = " ✅"
                    elif _service.health_status.value == 'offline':
                        health = " ❌"
                    elif _service.health_status.value == 'timeout':
                        health = " ⏱️"

                print(f"    • {_service.name} ({services}) - {pricing}{health}")
        
        print(f"")
        print("Common operations:")
        print(f"  services.show()                           — Show interactive widget")
        print(f"  client.chat('datasite/service', 'msg')   — Chat with a service")
        print(f"  client.search('datasite/service', 'msg') — Search with a service")
    
    def to_dict(self):
        """Convert services to list of dictionaries."""
        return [
            {
                "name": _service.name,
                "datasite": _service.datasite,
                "summary": _service.summary,
                "description": getattr(_service, 'description', ''),
                "tags": _service.tags,
                "services": [
                    {
                        "type": service.type.value,
                        "enabled": service.enabled,
                        "pricing": service.pricing,
                        "charge_type": service.charge_type.value
                    }
                    for service in _service.services
                ],
                "config_status": _service.config_status.value,
                "health_status": _service.health_status.value if hasattr(_service, 'health_status') and _service.health_status else None,
                "min_pricing": _service.min_pricing,
                "max_pricing": _service.max_pricing
            }
            for _service in self._services
        ]
    
    def to_json(self):
        """Convert services to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)
