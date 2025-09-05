"""
Health check utilities for SyftBox services
"""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..core.types import HealthStatus
from ..core.exceptions import HealthCheckError, NetworkError, RPCError
from ..clients.rpc_client import SyftBoxRPCClient
from ..models.service_info import ServiceInfo

logger = logging.getLogger(__name__)

class HealthMonitor:
    """Continuous health monitoring for services."""
    
    def __init__(self, rpc_client: SyftBoxRPCClient, check_interval: float = 30.0):
        """Initialize health monitor.
        
        Args:
            rpc_client: RPC client for health checks
            check_interval: Seconds between health checks
        """
        self.rpc_client = rpc_client
        self.check_interval = check_interval
        self.monitored_services: List[ServiceInfo] = []
        self.health_status: Dict[str, HealthStatus] = {}
        self.last_check_time: Optional[float] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._callbacks: List[callable] = []
    
    def add_service(self, service_info: ServiceInfo):
        """Add a service to monitoring.
        
        Args:
            service_info: Service to monitor
        """
        if service_info not in self.monitored_services:
            self.monitored_services.append(service_info)
            logger.info(f"Added {service_info.name} to health monitoring")
    
    def remove_service(self, service_name: str):
        """Remove a service from monitoring.
        
        Args:
            service_name: Name of service to remove
        """
        self.monitored_services = [
            service for service in self.monitored_services 
            if service.name != service_name
        ]
        
        if service_name in self.health_status:
            del self.health_status[service_name]
        
        logger.info(f"Removed {service_name} from health monitoring")
    
    def add_callback(self, callback: callable):
        """Add callback for health status changes.
        
        Args:
            callback: Function to call when health status changes
                     Signature: callback(service_name: str, old_status: HealthStatus, new_status: HealthStatus)
        """
        self._callbacks.append(callback)
    
    async def check_all_services(self) -> Dict[str, HealthStatus]:
        """Check health of all monitored services.
        
        Returns:
            Current health status of all services
        """
        if not self.monitored_services:
            return {}
        
        new_status = await batch_health_check(
            self.monitored_services,
            self.rpc_client,
            timeout=2.0
        )
        
        # Check for status changes and trigger callbacks
        for service_name, new_health in new_status.items():
            old_health = self.health_status.get(service_name)
            
            if old_health != new_health:
                logger.info(f"Health status changed for {service_name}: {old_health} -> {new_health}")
                
                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        callback(service_name, old_health, new_health)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")
        
        # Update stored status
        self.health_status.update(new_status)
        self.last_check_time = time.time()
        
        return self.health_status
    
    def get_service_health(self, service_name: str) -> Optional[HealthStatus]:
        """Get current health status of a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Current health status, or None if not monitored
        """
        return self.health_status.get(service_name)
    
    def get_healthy_services(self) -> List[str]:
        """Get list of currently healthy service names.
        
        Returns:
            List of service names that are online
        """
        return [
            service_name for service_name, status in self.health_status.items()
            if status == HealthStatus.ONLINE
        ]
    
    def get_unhealthy_services(self) -> List[str]:
        """Get list of currently unhealthy service names.
        
        Returns:
            List of service names that are offline or having issues
        """
        return [
            service_name for service_name, status in self.health_status.items()
            if status in [HealthStatus.OFFLINE, HealthStatus.TIMEOUT, HealthStatus.UNKNOWN]
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health status.
        
        Returns:
            Dictionary with health statistics
        """
        if not self.health_status:
            return {
                "total_services": 0,
                "healthy": 0,
                "unhealthy": 0,
                "unknown": 0,
                "last_check": None
            }
        
        status_counts = {}
        for status in self.health_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_services": len(self.health_status),
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
                    await self.check_all_services()
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
                
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("Health monitoring cancelled")
            raise

async def check_service_health(
        service_info: ServiceInfo,
        rpc_client: SyftBoxRPCClient,
        timeout: float = 2.0
    ) -> HealthStatus:
    """Check health of a single service.
    
    Args:
        service_info: Service to check
        rpc_client: RPC client for making calls
        timeout: Timeout in seconds for health check
        
    Returns:
        Health status of the service
    """
    try:
        # Create a temporary client with shorter timeout for health checks
        health_client = SyftBoxRPCClient(
            cache_server_url=rpc_client.base_url,
            timeout=timeout,
            max_poll_attempts=3,  # Fewer attempts for health checks
            poll_interval=0.5  # Faster polling for health checks
        )
        
        try:
            response = await health_client.call_health(service_info)
            
            # Parse health response
            if isinstance(response, dict):
                data = response.get("data", {})
                message = data.get("message", {})
                body = message.get("body", {})
                
                # Check if body is a dict (successful response) or string (error response)
                if isinstance(body, dict):
                    status = body.get("status", "unknown").lower()
                    # logger.info(f"Parsed status for {service_info.name}: {status}")
                    
                    if status == "ok" or status == "healthy":
                        return HealthStatus.ONLINE
                    elif status == "error" or status == "unhealthy":
                        return HealthStatus.OFFLINE
                    else:
                        return HealthStatus.UNKNOWN
                else:
                    # body is a string (error message), service is having issues
                    # logger.warning(f"Service {service_info.name} returned error: {body}")
                    return HealthStatus.OFFLINE
                
        finally:
            await health_client.close()
    
    except asyncio.TimeoutError:
        return HealthStatus.TIMEOUT
    except (NetworkError, RPCError) as e:
        logger.debug(f"Health check failed for {service_info.name}: {e}")
        return HealthStatus.OFFLINE
    except Exception as e:
        logger.warning(f"Unexpected error in health check for {service_info.name}: {e}")
        return HealthStatus.UNKNOWN
    
async def batch_health_check(
        services: List[ServiceInfo],
        rpc_client: SyftBoxRPCClient,
        timeout: float = 2.0,
        max_concurrent: int = 10
    ) -> Dict[str, HealthStatus]:
    """Check health of multiple services concurrently.
    
    Args:
        services: List of services to check
        rpc_client: RPC client for making calls
        timeout: Timeout per health check
        max_concurrent: Maximum concurrent health checks
        
    Returns:
        Dictionary mapping service names to health status
    """
    if not services:
        return {}
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def check_single_service(service: ServiceInfo) -> Tuple[str, HealthStatus]:
        async with semaphore:
            health = await check_service_health(service, rpc_client, timeout)
            return service.name, health
    
    # Start all health checks concurrently
    tasks = [check_single_service(service) for service in services]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    
    logger.info(f"Batch health check completed in {end_time - start_time:.2f}s for {len(services)} services")
    
    # Process results
    health_status = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Health check task failed: {result}")
            continue
        
        service_name, status = result
        health_status[service_name] = status
    
    return health_status

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

async def get_service_response_time(service_info: ServiceInfo, 
                                 rpc_client: SyftBoxRPCClient) -> Optional[float]:
    """Measure response time for a service's health endpoint.
    
    Args:
        service_info: Service to test
        rpc_client: RPC client for making calls
        
    Returns:
        Response time in seconds, or None if failed
    """
    try:
        start_time = time.time()
        await check_service_health(service_info, rpc_client, timeout=10.0)
        end_time = time.time()
        return end_time - start_time
    except Exception:
        return None