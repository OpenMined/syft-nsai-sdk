"""
SyftBox authentication client for generating and managing auth tokens
"""
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from ..core.config import SyftBoxConfig
from ..core.exceptions import AuthenticationError
from .request_client import HTTPClient

logger = logging.getLogger(__name__)

class SyftBoxAuthClient:
    """Client for handling SyftBox authentication using config.json refresh tokens."""
    
    def __init__(self, config: Optional[SyftBoxConfig] = None):
        """Initialize SyftBox auth client.
        
        Args:
            config: SyftBox configuration object with refresh token
        """
        self.config = config
        self._auth_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._http_client: Optional[HTTPClient] = None
    
    def __dir__(self):
        """Control what appears in autocomplete suggestions."""
        return [
            'get_user_email',
            'get_auth_token',
            'is_authenticated',
            'clear_session',
        ]
    
    @property
    def http_client(self) -> HTTPClient:
        """Get HTTP client for auth requests."""
        if self._http_client is None:
            self._http_client = HTTPClient()
        return self._http_client
    
    async def close(self):
        """Close the auth client and cleanup resources."""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None
    
    def is_authenticated(self) -> bool:
        """Check if we have valid SyftBox authentication.
        
        Returns:
            True if config exists with valid refresh token, False otherwise
        """
        result = (
            self.config is not None and 
            self.config.email is not None and
            self.config.refresh_token is not None and
            len(self.config.refresh_token.strip()) > 0  # Check for non-empty token
        )
        
        if not result and self.config is not None:
            logger.info("SyftBox config found but no valid refresh token - using guest mode")
        
        return result
    
    def get_user_email(self) -> str:
        """Get authenticated user email or fallback to guest.
        
        Returns:
            User email from config or guest@syftbox.net
        """
        if self.is_authenticated():
            return self.config.email
        return "guest@syftbox.net"
    
    async def get_auth_token(self) -> Optional[str]:
        """Get valid auth token, refreshing if necessary.
        
        Returns:
            Auth token string if authentication available, None for guest mode
        """
        if not self.is_authenticated():
            logger.debug("No SyftBox authentication available, using guest mode")
            return None
        
        # Check if we have a valid cached token
        if self._is_token_valid():
            logger.debug("Using cached auth token")
            return self._auth_token
        
        # Generate new token from refresh token
        try:
            await self._refresh_auth_token()
            return self._auth_token
        except Exception as e:
            logger.warning(f"Failed to refresh SyftBox auth token: {e}. Falling back to guest mode.")
            # Clear invalid token and continue in guest mode
            self._clear_cached_token()
            return None
    
    def clear_session(self):
        """Clear cached auth token (force re-authentication on next request)."""
        self._clear_cached_token()
        logger.info("SyftBox auth session cleared")
    
    async def _refresh_auth_token(self):
        """Refresh auth token using the refresh token from config.
        
        Raises:
            AuthenticationError: If token refresh fails
        """
        if not self.config or not self.config.refresh_token:
            raise AuthenticationError("No refresh token available")
        
        refresh_url = f"{self.config.server_url}/auth/refresh"
        
        # Prepare request payload
        refresh_payload = {
            "refreshToken": self.config.refresh_token
        }
        
        try:
            logger.debug(f"Refreshing auth token from {refresh_url}")
            
            response = await self.http_client.post(
                refresh_url,
                json=refresh_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            if response.status_code == 200:
                token_data = response.json()
                
                # Extract token and expiration
                auth_token = token_data.get("accessToken")
                expires_in = token_data.get("expiresIn", 3600)  # Default 1 hour
                
                if not auth_token:
                    raise AuthenticationError("No access token in refresh response")
                
                # Cache the token with expiration
                self._auth_token = auth_token
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # 1 min buffer
                
                logger.info(f"Successfully refreshed auth token for {self.config.email}")
                
            elif response.status_code == 401:
                raise AuthenticationError("Refresh token expired or invalid")
            else:
                response_text = ""
                try:
                    error_data = response.json()
                    response_text = error_data.get("message", f"HTTP {response.status_code}")
                except:
                    response_text = f"HTTP {response.status_code}"
                
                raise AuthenticationError(f"Token refresh failed: {response_text}")
        
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Network error during token refresh: {e}")
    
    def _is_token_valid(self) -> bool:
        """Check if cached token is still valid.
        
        Returns:
            True if token exists and not expired, False otherwise
        """
        if not self._auth_token or not self._token_expires_at:
            return False
        
        return datetime.now() < self._token_expires_at
    
    def _clear_cached_token(self):
        """Clear cached token data."""
        self._auth_token = None
        self._token_expires_at = None
    
    @classmethod
    def setup_auth_discovery(cls, config: Optional[SyftBoxConfig] = None) -> tuple['SyftBoxAuthClient', bool]:
        """Try to auto-discover SyftBox authentication and return client + success status.
        
        Args:
            config: Optional SyftBox config object
            
        Returns:
            Tuple of (SyftBoxAuthClient, is_authenticated)
        """
        auth_client = cls(config)
        
        if auth_client.is_authenticated():
            logger.info(f"Found SyftBox authentication for {auth_client.get_user_email()}")
            return auth_client, True
        else:
            logger.info("No SyftBox authentication found, will use guest mode")
            return auth_client, False
    
    def show_status(self) -> str:
        """Get authentication status as formatted string.
        
        Returns:
            Status string for display
        """
        if self.is_authenticated():
            email = self.config.email
            server = self.config.server_url
            token_status = "Valid" if self._is_token_valid() else "Needs refresh"
            
            return (
                f"SyftBox Authentication: Configured\n"
                f"  User: {email}\n"
                f"  Server: {server}\n"
                f"  Token: {token_status}\n"
                f"  Mode: Authenticated requests"
            )
        else:
            return (
                f"SyftBox Authentication: Not configured\n"
                f"  User: guest@syftbox.net\n"
                f"  Mode: Guest requests only\n"
                f"  Note: Install and setup SyftBox for authenticated access"
            )
    
    def __repr__(self) -> str:
        """Return text representation of auth status."""
        return self.show_status()
