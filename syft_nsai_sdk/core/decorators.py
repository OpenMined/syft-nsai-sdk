"""
Decorators for SyftBox SDK
"""
import functools
from typing import Callable, Any
from .exceptions import AuthenticationError


def require_account(func: Callable) -> Callable:
    """Decorator that requires account setup before model operations.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that checks account status
    """
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs) -> Any:
        if not getattr(self, '_account_configured', False):
            raise AuthenticationError(
                "Account setup required before using models. "
                "Please run: await client.setup_accounting(email, password)"
            )
        return await func(self, *args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs) -> Any:
        if not getattr(self, '_account_configured', False):
            raise AuthenticationError(
                "Account setup required before using models. "
                "Please run: await client.setup_accounting(email, password)"
            )
        return func(self, *args, **kwargs)
    
    # Return appropriate wrapper based on whether function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Import asyncio at module level for the decorator
import asyncio