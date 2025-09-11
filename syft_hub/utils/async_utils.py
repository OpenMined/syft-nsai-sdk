"""
Async utilities for SyftBox SDK - handles event loop management and context detection
"""
import asyncio
import concurrent.futures
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)

def detect_async_context() -> bool:
    """
    Detect if we're currently in an async context with a running event loop.
    
    Returns:
        bool: True if there's a running event loop, False otherwise
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False

def run_async_in_thread(coro: Coroutine) -> Any:
    """
    Run an async coroutine in a separate thread with its own event loop.
    This avoids conflicts with existing event loops and is safe to use
    from any synchronous context.
    
    Args:
        coro: The coroutine to execute
        
    Returns:
        The result of the coroutine execution
        
    Raises:
        Any exception raised by the coroutine
    """
    def run_in_thread():
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up the loop properly
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Run the loop once more to handle cancellations
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logger.debug(f"Error during loop cleanup: {e}")
            finally:
                try:
                    loop.close()
                except Exception as e:
                    logger.debug(f"Error closing loop: {e}")
    
    # Use ThreadPoolExecutor for better resource management
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result()

# Convenience function for one-off async execution in sync contexts
def run_async(coro: Coroutine) -> Any:
    """
    Convenience function to run a single coroutine synchronously.
    Alias for run_async_in_thread() with a shorter name.
    
    Args:
        coro: The coroutine to execute
        
    Returns:
        The result of the coroutine execution
    """
    return run_async_in_thread(coro)