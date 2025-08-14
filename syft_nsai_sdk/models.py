"""
Model objects and interaction logic for the SyftBox SDK.
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Any

from .config import SyftBoxConfig
from .types import ModelMetadata


class ModelObject:
    """
    Represents a discovered model with interaction capabilities.
    """
    
    def __init__(self, metadata: ModelMetadata, config: SyftBoxConfig):
        self.metadata = metadata
        self.config = config
        
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def owner(self) -> str:
        return self.metadata.owner
    
    @property
    def tags(self) -> List[str]:
        return self.metadata.tags
    
    @property
    def services(self) -> List[str]:
        return [service.type for service in self.metadata.services if service.enabled]
    
    @property
    def summary(self) -> str:
        return self.metadata.summary
    
    @property
    def status(self) -> str:
        """Return availability status based on enabled services."""
        enabled_services = [s for s in self.metadata.services if s.enabled]
        return "available" if enabled_services else "unavailable"
    
    def _join_urls(self, base_url: str, endpoint: str) -> str:
        """Join base URL and endpoint properly."""
        clean_base = base_url.rstrip("/")
        clean_endpoint = endpoint.lstrip("/")
        return f"{clean_base}/{clean_endpoint}"
    
    async def _make_rpc_request(self, service_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make an RPC request using SyftBox protocol."""
        if not self.config.email:
            raise RuntimeError("Not authenticated. Config missing email.")
        
        server_url = self.config.server_url
        syft_from = self.config.email
        
        if service_type == "search":
            query = payload.get("query", payload.get("prompt", ""))
            encoded_query = requests.utils.quote(query)
            syft_url = f"syft://{self.owner}/app_data/{self.name}/rpc/search?query=\"{encoded_query}\""
            method = "POST"
            body = None
        elif service_type == "chat":
            syft_url = f"syft://{self.owner}/app_data/{self.name}/rpc/chat"
            method = "POST"
            body = payload
        else:
            raise ValueError(f"Unsupported service type: {service_type}")
        
        encoded_syft_url = requests.utils.quote(syft_url)
        endpoint = f"/api/v1/send/msg?x-syft-url={encoded_syft_url}&x-syft-from={syft_from}"
        
        # Prepare headers with authentication
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*"
        }
        
        # Add authorization header if refresh_token is available
        if self.config.refresh_token:
            headers["Authorization"] = f"Bearer {self.config.refresh_token}"
        
        async with aiohttp.ClientSession() as session:
            request_kwargs = {"headers": headers}
            
            if body:
                request_kwargs["json"] = body
            
            full_url = self._join_urls(server_url, endpoint)
            
            async with session.request(method, full_url, **request_kwargs) as response:
                if response.status not in [200, 202]:
                    error_text = await response.text()
                    raise RuntimeError(f"RPC request failed with status {response.status}: {error_text}")
                
                data = await response.json()
                poll_url = data.get("data", {}).get("poll_url")
                
                if not poll_url:
                    raise RuntimeError("No poll URL returned from RPC request")
                
                return await self._poll_for_response(session, poll_url, headers)
    
    async def _poll_for_response(self, session: "aiohttp.ClientSession", poll_url: str, headers: Dict[str, str], max_attempts: int = 20) -> Dict[str, Any]:
        """Poll for RPC response."""
        server_url = self.config.server_url
        
        for attempt in range(max_attempts):
            try:
                full_poll_url = self._join_urls(server_url, poll_url)
                async with session.get(full_poll_url, headers=headers) as response:
                    try:
                        data = await response.json()
                    except Exception:
                        raise RuntimeError("Invalid response from SyftBox server")
                    
                    status_code = data.get("data", {}).get("message", {}).get("status_code")
                    
                    if response.status == 200 and status_code == 200:
                        return data.get("data", {}).get("message", {}).get("body", {})
                    elif response.status == 202 or status_code == 202:
                        await asyncio.sleep(2)
                        continue
                    else:
                        error_body = data.get("data", {}).get("message", {}).get("body")
                        raise RuntimeError(f"RPC request failed: {error_body}")
                        
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"RPC polling failed after {max_attempts} attempts: {e}")
                await asyncio.sleep(2)
        
        raise RuntimeError("RPC request timed out after maximum attempts")
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results into a readable string."""
        if not results:
            return "No results found."
        
        formatted = []
        for result in results:
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            formatted.append(f"[{filename}]\n{content}")
        
        return "\n\n".join(formatted)
    
    def predict(self, prompt: str, service: str = "chat", **kwargs) -> str:
        """
        Make a prediction using the specified service.
        
        Args:
            prompt: The input prompt
            service: Service type ('chat' or 'search')
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        """
        if service not in self.services:
            available_services = ", ".join(self.services)
            raise ValueError(f"Service '{service}' not available. Available services: {available_services}")
        
        try:
            # Always use thread approach to avoid event loop conflicts
            import concurrent.futures
            import asyncio
            
            def run_in_new_thread():
                # Create a completely new event loop in a thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self._predict_async(prompt, service, **kwargs))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_thread)
                return future.result(timeout=60)
                
        except Exception as e:
            raise RuntimeError(f"Failed to call service '{service}': {e}")
    
    async def _predict_async(self, prompt: str, service: str, **kwargs) -> str:
        """Async version of predict for internal use."""
        if service == "search":
            payload = {"query": prompt, **kwargs}
            result = await self._make_rpc_request("search", payload)
            
            if isinstance(result, dict) and "results" in result:
                return self._format_search_results(result["results"])
            elif isinstance(result, list):
                return self._format_search_results(result)
            return str(result)
        
        elif service == "chat":
            messages = kwargs.get("messages", [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ])
            
            payload = {
                "model": kwargs.get("model", "tinyllama:latest"),
                "messages": messages,
                **{k: v for k, v in kwargs.items() if k not in ["messages", "model"]}
            }
            
            result = await self._make_rpc_request("chat", payload)
            
            if isinstance(result, dict):
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"]
                elif "content" in result:
                    return result["content"]
            return str(result)
        
        else:
            raise ValueError(f"Unsupported service type: {service}")
    
    def chat(self, prompt: str, **kwargs) -> str:
        """Convenience method for chat service."""
        return self.predict(prompt, service="chat", **kwargs)
    
    def search(self, query: str, **kwargs) -> str:
        """Convenience method for search service."""
        return self.predict(query, service="search", **kwargs)
    
    def __repr__(self) -> str:
        return f"ModelObject(name='{self.name}', owner='{self.owner}', services={self.services})"