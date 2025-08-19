"""
Authentication and transaction token management for SyftBox
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from syft_accounting_sdk import UserClient, ServiceException

from ..core.exceptions import AuthenticationError, PaymentError
from ..core.types import TransactionToken

logger = logging.getLogger(__name__)


class AccountingAuth:
    """Handles authentication with the SyftBox accounting service."""
    
    def __init__(self, service_url: str, email: str, password: str):
        """Initialize accounting authentication.
        
        Args:
            service_url: URL of the accounting service
            email: User email
            password: User password
        """
        self.service_url = service_url
        self.email = email
        self.password = password
        self._client: Optional[UserClient] = None
        self._last_validated: Optional[datetime] = None
        self._validation_cache_minutes = 5  # Cache validation for 5 minutes
    
    @property
    def client(self) -> UserClient:
        """Get or create authenticated client.
        
        Returns:
            UserClient instance
            
        Raises:
            AuthenticationError: If authentication fails
        """
        if self._client is None or self._needs_revalidation():
            self._create_client()
        
        return self._client
    
    def _create_client(self):
        """Create and validate accounting client."""
        try:
            self._client = UserClient(
                url=self.service_url,
                email=self.email,
                password=self.password
            )
            
            # Validate by getting user info
            user_info = self._client.get_user_info()
            logger.debug(f"Authenticated with accounting service: {user_info.email}")
            
            self._last_validated = datetime.now()
            
        except ServiceException as e:
            self._client = None
            self._last_validated = None
            raise AuthenticationError(f"Authentication failed: {e}")
        except Exception as e:
            self._client = None
            self._last_validated = None
            raise AuthenticationError(f"Unexpected authentication error: {e}")
    
    def _needs_revalidation(self) -> bool:
        """Check if client needs revalidation."""
        if self._last_validated is None:
            return True
        
        expiry = self._last_validated + timedelta(minutes=self._validation_cache_minutes)
        return datetime.now() > expiry
    
    def validate_credentials(self) -> bool:
        """Validate current credentials.
        
        Returns:
            True if credentials are valid
        """
        try:
            _ = self.client  # This will trigger validation
            return True
        except AuthenticationError:
            return False
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get user account information.
        
        Returns:
            User information dictionary
        """
        try:
            user_info = self.client.get_user_info()
            return {
                "email": user_info.email,
                "organization": user_info.organization,
                "balance": user_info.balance
            }
        except ServiceException as e:
            raise AuthenticationError(f"Failed to get user info: {e}")
    
    def get_balance(self) -> float:
        """Get current account balance.
        
        Returns:
            Account balance
        """
        try:
            user_info = self.client.get_user_info()
            return user_info.balance
        except ServiceException as e:
            raise PaymentError(f"Failed to get balance: {e}")
    
    def create_transaction_token(self, recipient_email: str) -> str:
        """Create transaction token for payment.
        
        Args:
            recipient_email: Email of the payment recipient
            
        Returns:
            Transaction token string
        """
        try:
            token = self.client.create_transaction_token(recipientEmail=recipient_email)
            logger.debug(f"Created transaction token for {recipient_email}")
            return token
        except ServiceException as e:
            raise PaymentError(f"Failed to create transaction token: {e}")


class TokenCache:
    """Cache for transaction tokens to avoid recreating them unnecessarily."""
    
    def __init__(self, cache_duration_minutes: int = 30):
        """Initialize token cache.
        
        Args:
            cache_duration_minutes: How long to cache tokens
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_duration = timedelta(minutes=cache_duration_minutes)
    
    def get_token(self, recipient_email: str) -> Optional[str]:
        """Get cached token for recipient.
        
        Args:
            recipient_email: Recipient email
            
        Returns:
            Cached token if valid, None otherwise
        """
        if recipient_email not in self._cache:
            return None
        
        cache_entry = self._cache[recipient_email]
        created_at = cache_entry["created_at"]
        
        # Check if token is still valid
        if datetime.now() - created_at < self._cache_duration:
            logger.debug(f"Using cached token for {recipient_email}")
            return cache_entry["token"]
        else:
            # Token expired, remove from cache
            del self._cache[recipient_email]
            return None
    
    def store_token(self, recipient_email: str, token: str):
        """Store token in cache.
        
        Args:
            recipient_email: Recipient email
            token: Transaction token
        """
        self._cache[recipient_email] = {
            "token": token,
            "created_at": datetime.now(),
            "recipient": recipient_email
        }
        logger.debug(f"Cached token for {recipient_email}")
    
    def clear_cache(self):
        """Clear all cached tokens."""
        self._cache.clear()
        logger.debug("Token cache cleared")
    
    def clear_expired(self):
        """Remove expired tokens from cache."""
        now = datetime.now()
        expired_recipients = []
        
        for recipient, cache_entry in self._cache.items():
            if now - cache_entry["created_at"] >= self._cache_duration:
                expired_recipients.append(recipient)
        
        for recipient in expired_recipients:
            del self._cache[recipient]
        
        if expired_recipients:
            logger.debug(f"Cleared {len(expired_recipients)} expired tokens")


class CredentialManager:
    """Manages storage and retrieval of authentication credentials."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize credential manager.
        
        Args:
            storage_dir: Directory to store credentials (default: ~/.syftbox)
        """
        self.storage_dir = storage_dir or Path.home() / ".syftbox"
        self.storage_dir.mkdir(exist_ok=True)
        self.credentials_file = self.storage_dir / "accounting.json"
    
    def save_credentials(self, service_url: str, email: str, password: str, 
                        organization: Optional[str] = None):
        """Save credentials to secure storage.
        
        Args:
            service_url: Accounting service URL
            email: User email
            password: User password
            organization: Optional organization name
        """
        credentials = {
            "service_url": service_url,
            "email": email,
            "password": password,
            "organization": organization,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }
        
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(self.credentials_file, 0o600)
            
            logger.info(f"Credentials saved for {email}")
            
        except Exception as e:
            raise AuthenticationError(f"Failed to save credentials: {e}")
    
    def load_credentials(self) -> Optional[Dict[str, Any]]:
        """Load credentials from storage.
        
        Returns:
            Credentials dictionary if found, None otherwise
        """
        if not self.credentials_file.exists():
            return None
        
        try:
            with open(self.credentials_file, 'r') as f:
                credentials = json.load(f)
            
            # Update last used timestamp
            credentials["last_used"] = datetime.now().isoformat()
            
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return None
    
    def delete_credentials(self):
        """Delete stored credentials."""
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
                logger.info("Credentials deleted")
        except Exception as e:
            logger.error(f"Failed to delete credentials: {e}")
    
    def credentials_exist(self) -> bool:
        """Check if credentials are stored.
        
        Returns:
            True if credentials file exists
        """
        return self.credentials_file.exists()


class AuthManager:
    """High-level authentication manager combining all auth components."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize authentication manager.
        
        Args:
            storage_dir: Directory for credential storage
        """
        self.credential_manager = CredentialManager(storage_dir)
        self.token_cache = TokenCache()
        self._auth: Optional[AccountingAuth] = None
    
    def configure_from_credentials(self, service_url: str, email: str, password: str) -> bool:
        """Configure authentication with provided credentials.
        
        Args:
            service_url: Accounting service URL
            email: User email
            password: User password
            
        Returns:
            True if configuration successful
        """
        try:
            self._auth = AccountingAuth(service_url, email, password)
            
            # Validate credentials
            if self._auth.validate_credentials():
                logger.info(f"Authentication configured for {email}")
                return True
            else:
                self._auth = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to configure authentication: {e}")
            self._auth = None
            return False
    
    def configure_from_storage(self) -> bool:
        """Configure authentication from stored credentials.
        
        Returns:
            True if configuration successful
        """
        credentials = self.credential_manager.load_credentials()
        if not credentials:
            return False
        
        return self.configure_from_credentials(
            credentials["service_url"],
            credentials["email"],
            credentials["password"]
        )
    
    def configure_from_environment(self) -> bool:
        """Configure authentication from environment variables.
        
        Returns:
            True if configuration successful
        """
        email = os.getenv("SYFTBOX_ACCOUNTING_EMAIL")
        password = os.getenv("SYFTBOX_ACCOUNTING_PASSWORD")
        service_url = os.getenv('SYFTBOX_ACCOUNTING_URL')
        
        if not email or not password:
            return False
        
        return self.configure_from_credentials(service_url, email, password)
    
    def auto_configure(self) -> bool:
        """Try to configure authentication from various sources.
        
        Returns:
            True if any configuration method succeeded
        """
        # Try environment variables first
        if self.configure_from_environment():
            return True
        
        # Try stored credentials
        if self.configure_from_storage():
            return True
        
        return False
    
    def is_configured(self) -> bool:
        """Check if authentication is configured.
        
        Returns:
            True if authentication is ready
        """
        return self._auth is not None and self._auth.validate_credentials()
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get user account information.
        
        Returns:
            User information dictionary
        """
        if not self._auth:
            raise AuthenticationError("Authentication not configured")
        
        return self._auth.get_user_info()
    
    def get_balance(self) -> float:
        """Get current account balance.
        
        Returns:
            Account balance
        """
        if not self._auth:
            raise AuthenticationError("Authentication not configured")
        
        return self._auth.get_balance()
    
    def create_transaction_token(self, recipient_email: str, use_cache: bool = True) -> TransactionToken:
        """Create transaction token for payment.
        
        Args:
            recipient_email: Email of the payment recipient
            use_cache: Whether to use cached tokens
            
        Returns:
            TransactionToken object
        """
        if not self._auth:
            raise AuthenticationError("Authentication not configured")
        
        # Try cache first if enabled
        if use_cache:
            cached_token = self.token_cache.get_token(recipient_email)
            if cached_token:
                return TransactionToken(token=cached_token, recipient_email=recipient_email)
        
        # Create new token
        token = self._auth.create_transaction_token(recipient_email)
        
        # Cache the token
        if use_cache:
            self.token_cache.store_token(recipient_email, token)
        
        return TransactionToken(token=token, recipient_email=recipient_email)
    
    def save_current_credentials(self, organization: Optional[str] = None):
        """Save current authentication credentials to storage.
        
        Args:
            organization: Optional organization name
        """
        if not self._auth:
            raise AuthenticationError("No credentials to save")
        
        self.credential_manager.save_credentials(
            self._auth.service_url,
            self._auth.email,
            self._auth.password,
            organization
        )
    
    def clear_credentials(self):
        """Clear stored credentials and reset authentication."""
        self.credential_manager.delete_credentials()
        self.token_cache.clear_cache()
        self._auth = None
        logger.info("Authentication cleared")
    
    def get_auth_status(self) -> Dict[str, Any]:
        """Get detailed authentication status.
        
        Returns:
            Status information dictionary
        """
        if not self._auth:
            return {
                "configured": False,
                "credentials_stored": self.credential_manager.credentials_exist(),
                "error": "No authentication configured"
            }
        
        try:
            user_info = self._auth.get_user_info()
            return {
                "configured": True,
                "valid": True,
                "email": user_info["email"],
                "organization": user_info.get("organization"),
                "balance": user_info["balance"],
                "service_url": self._auth.service_url,
                "credentials_stored": self.credential_manager.credentials_exist()
            }
        except Exception as e:
            return {
                "configured": True,
                "valid": False,
                "error": str(e),
                "credentials_stored": self.credential_manager.credentials_exist()
            }