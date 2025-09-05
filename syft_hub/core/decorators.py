"""
Decorators for SyftBox SDK
"""
import asyncio
import functools

from typing import Callable, Any

from ..utils.async_utils import detect_async_context, run_async_in_thread
from .exceptions import AuthenticationError


def require_account(func: Callable) -> Callable:
    """Decorator that requires account setup before service operations.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that checks account status
    """
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs) -> Any:
        if not getattr(self, '_account_configured', False):
            raise AuthenticationError(
                "Account setup required before using services. "
                "Please run: await client.setup_accounting(email, password)"
            )
        return await func(self, *args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs) -> Any:
        if not getattr(self, '_account_configured', False):
            raise AuthenticationError(
                "Account setup required before using services. "
                "Please run: await client.setup_accounting(email, password)"
            )
        return func(self, *args, **kwargs)
    
    # Return appropriate wrapper based on whether function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
    
def make_sync_wrapper(async_method):
    """
    Decorator factory to create thread-safe synchronous wrappers for async methods.
    
    Args:
        async_method: The async method to wrap
        
    Returns:
        A synchronous wrapper function
        
    Example:
        @make_sync_wrapper
        async def some_async_method(self, arg1, arg2):
            return await some_operation(arg1, arg2)
            
        # Creates: some_async_method_sync() that can be called synchronously
    """
    def sync_wrapper(self, *args, **kwargs):
        return run_async_in_thread(async_method(self, *args, **kwargs))
    
    # Preserve metadata
    sync_wrapper.__name__ = f"{async_method.__name__}_sync"
    sync_wrapper.__doc__ = f"Synchronous wrapper for {async_method.__name__}.\n\n{async_method.__doc__ or ''}"
    
    return sync_wrapper

def smart_async_wrapper(async_method):
    """
    Decorator that creates a "smart" method that adapts to the execution context.
    
    In async contexts: returns the awaitable coroutine
    In sync contexts: executes synchronously and returns the result
    
    Args:
        async_method: The async method to wrap
        
    Returns:
        A smart wrapper that adapts to the execution context
    """
    def smart_wrapper(self, *args, **kwargs):
        if detect_async_context():
            # Return coroutine - caller must await it
            return async_method(self, *args, **kwargs)
        else:
            # Execute synchronously using thread pool
            return run_async_in_thread(async_method(self, *args, **kwargs))
    
    # Preserve metadata
    smart_wrapper.__name__ = async_method.__name__.replace('_async', '')
    smart_wrapper.__doc__ = f"""Smart wrapper for {async_method.__name__} that adapts to execution context.
    
    In async contexts: returns awaitable coroutine
    In sync contexts: returns the actual result
    
    {async_method.__doc__ or ''}
    """
    
    return smart_wrapper