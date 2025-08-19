"""
Health check utilities for SyftBox models
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import time
import logging

from ..core.types import ModelInfo, HealthStatus
from ..core.exceptions import HealthCheckError, NetworkError, RPCError
from ..networking.rpc_client import SyftBoxRPCClient

logger = logging.getLogger(__name__)


async def check_model_health(model_info: ModelInfo, 
                            rpc_client: SyftBoxRPCClient,
                            timeout: float = 2.0) -> HealthStatus:
    """Check health of a single model.
    
    Args:
        model_info: Model to check
        rpc_client: RPC client for making calls
        timeout: Timeout in seconds for health check
        
    Returns:
        Health status of the model
    """
    try:
        # Create a temporary client with shorter timeout for health checks
        health_client = SyftBoxRPCClient(
            cache_server_url=rpc_client.cache_server_url,
            from_email=rpc_client.from_email,
            timeout=timeout,
            max_poll_attempts=3,  # Fewer attempts for health checks
            poll_interval=0.5  # Faster polling for health checks
        )
        
        try:
            response = await health_client.call_health(model_info)
            
            # Parse health response
            if isinstance(response, dict):
                status = response.get("status", "unknown").lower()
                if status == "ok" or status == "healthy":
                    return HealthStatus.ONLINE
                elif status == "error" or status == "unhealthy":
                    return HealthStatus.OFFLINE
                else:
                    return HealthStatus.UNKNOWN
            else:
                # Any response is considered healthy
                return HealthStatus.ONLINE
                
        finally:
            await health_client.close()
    
    except asyncio.TimeoutError:
        return HealthStatus.TIMEOUT
    except (NetworkError, RPCError) as e:
        logger.debug(f"Health check failed for {model_info.name}: {e}")
        return HealthStatus.OFFLINE
    except Exception as e:
        logger.warning(f"Unexpected error in health check for {model_info.name}: {e}")
        return HealthStatus.UNKNOWN


async def batch_health_check(models: List[ModelInfo],
                            rpc_client: SyftBoxRPCClient,
                            timeout: float = 2.0,
                            max_concurrent: int = 10) -> Dict[str, HealthStatus]:
    """Check health of multiple models concurrently.
    
    Args:
        models: List of models to check
        rpc_client: RPC client for making calls
        timeout: Timeout per health check
        max_concurrent: Maximum concurrent health checks
        
    Returns:
        Dictionary mapping model names to health status
    """
    if not models:
        return {}
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def check_single_model(model: ModelInfo) -> Tuple[str, HealthStatus]:
        async with semaphore:
            health = await check_model_health(model, rpc_client, timeout)
            return model.name, health
    
    # Start all health checks concurrently
    tasks = [check_single_model(model) for model in models]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    
    logger.info(f"Batch health check completed in {end_time - start_time:.2f}s for {len(models)} models")
    
    # Process results
    health_status = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Health check task failed: {result}")
            continue
        
        model_name, status = result
        health_status[model_name] = status
    
    return health_status


class HealthMonitor:
    """Continuous health monitoring for models."""
    
    def __init__(self, rpc_client: SyftBoxRPCClient, check_interval: float = 30.0):
        """Initialize health monitor.
        
        Args:
            rpc_client: RPC client for health checks
            check_interval: Seconds between health checks
        """
        self.rpc_client = rpc_client
        self.check_interval = check_interval
        self.monitored_models: List[ModelInfo] = []
        self.health_status: Dict[str, HealthStatus] = {}
        self.last_check_time: Optional[float] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._callbacks: List[callable] = []
    
    def add_model(self, model_info: ModelInfo):
        """Add a model to monitoring.
        
        Args:
            model_info: Model to monitor
        """
        if model_info not in self.monitored_models:
            self.monitored_models.append(model_info)
            logger.info(f"Added {model_info.name} to health monitoring")
    
    def remove_model(self, model_name: str):
        """Remove a model from monitoring.
        
        Args:
            model_name: Name of model to remove
        """
        self.monitored_models = [
            model for model in self.monitored_models 
            if model.name != model_name
        ]
        
        if model_name in self.health_status:
            del self.health_status[model_name]
        
        logger.info(f"Removed {model_name} from health monitoring")
    
    def add_callback(self, callback: callable):
        """Add callback for health status changes.
        
        Args:
            callback: Function to call when health status changes
                     Signature: callback(model_name: str, old_status: HealthStatus, new_status: HealthStatus)
        """
        self._callbacks.append(callback)
    
    async def check_all_models(self) -> Dict[str, HealthStatus]:
        """Check health of all monitored models.
        
        Returns:
            Current health status of all models
        """
        if not self.monitored_models:
            return {}
        
        new_status = await batch_health_check(
            self.monitored_models,
            self.rpc_client,
            timeout=2.0
        )
        
        # Check for status changes and trigger callbacks
        for model_name, new_health in new_status.items():
            old_health = self.health_status.get(model_name)
            
            if old_health != new_health:
                logger.info(f"Health status changed for {model_name}: {old_health} -> {new_health}")
                
                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        callback(model_name, old_health, new_health)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")
        
        # Update stored status
        self.health_status.update(new_status)
        self.last_check_time = time.time()
        
        return self.health_status
    
    def get_model_health(self, model_name: str) -> Optional[HealthStatus]:
        """Get current health status of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Current health status, or None if not monitored
        """
        return self.health_status.get(model_name)
    
    def get_healthy_models(self) -> List[str]:
        """Get list of currently healthy model names.
        
        Returns:
            List of model names that are online
        """
        return [
            model_name for model_name, status in self.health_status.items()
            if status == HealthStatus.ONLINE
        ]
    
    def get_unhealthy_models(self) -> List[str]:
        """Get list of currently unhealthy model names.
        
        Returns:
            List of model names that are offline or having issues
        """
        return [
            model_name for model_name, status in self.health_status.items()
            if status in [HealthStatus.OFFLINE, HealthStatus.TIMEOUT, HealthStatus.UNKNOWN]
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health status.
        
        Returns:
            Dictionary with health statistics
        """
        if not self.health_status:
            return {
                "total_models": 0,
                "healthy": 0,
                "unhealthy": 0,
                "unknown": 0,
                "last_check": None
            }
        
        status_counts = {}
        for status in self.health_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_models": len(self.health_status),
            "healthy": status_counts.get(HealthStatus.ONLINE, 0),
            "unhealthy": (
                status_counts.get(HealthStatus.OFFLINE, 0) +
                status_counts.get(HealthStatus.TIMEOUT, 0)
            ),
            "unknown": status_counts.get(HealthStatus.UNKNOWN, 0),
            "last_check": self.last_check_time,
            "status_breakdown": {
                status.value: count for status, count in status_counts.items()
            }
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_task is not None:
            logger.warning("Health monitoring already running")
            return
        
        logger.info(f"Starting health monitoring with {self.check_interval}s interval")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if self._monitoring_task is None:
            return
        
        logger.info("Stopping health monitoring")
        self._monitoring_task.cancel()
        
        try:
            await self._monitoring_task
        except asyncio.CancelledError:
            pass
        
        self._monitoring_task = None
    
    async def _monitoring_loop(self):
        """Internal monitoring loop."""
        try:
            while True:
                try:
                    await self.check_all_models()
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
                
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("Health monitoring cancelled")
            raise


def format_health_status(status: HealthStatus) -> str:
    """Format health status for display.
    
    Args:
        status: Health status to format
        
    Returns:
        Formatted status string with emoji
    """
    status_icons = {
        HealthStatus.ONLINE: "✅",
        HealthStatus.OFFLINE: "❌",
        HealthStatus.TIMEOUT: "⏱️",
        HealthStatus.UNKNOWN: "❓",
        HealthStatus.NOT_APPLICABLE: "➖"
    }
    
    icon = status_icons.get(status, "❓")
    return f"{status.value.title()} {icon}"


async def get_model_response_time(model_info: ModelInfo, 
                                 rpc_client: SyftBoxRPCClient) -> Optional[float]:
    """Measure response time for a model's health endpoint.
    
    Args:
        model_info: Model to test
        rpc_client: RPC client for making calls
        
    Returns:
        Response time in seconds, or None if failed
    """
    try:
        start_time = time.time()
        await check_model_health(model_info, rpc_client, timeout=10.0)
        end_time = time.time()
        return end_time - start_time
    except Exception:
        return None