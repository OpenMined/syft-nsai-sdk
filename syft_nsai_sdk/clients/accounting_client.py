"""
SyftBox Accounting Client for managing payments and transactions
"""
import logging
from typing import Any, Dict, Optional
from syft_accounting_sdk import UserClient, ServiceException

from ..core.exceptions import PaymentError, AuthenticationError

logger = logging.getLogger(__name__)


class AccountingClient:
    """Client for handling accounting operations."""
    
    def __init__(self, 
                 service_url: Optional[str] = None,
                 credentials: Optional[Dict[str, str]] = None):
        """Initialize accounting client.
        
        Args:
            service_url: URL of the accounting service
            credentials: Dict with 'email', 'password', and optionally 'service_url'
        """
        self.service_url = service_url
        self._credentials = credentials
        self._client = None
    
    def configure(self, service_url: str, email: str, password: str):
        """Configure accounting client.
        
        Args:
            service_url: Accounting service URL
            email: User email
            password: User password
        """
        self.service_url = service_url
        self._credentials = {
            "service_url": service_url,
            "email": email,
            "password": password
        }
        self._client = None  # Reset to recreate with new credentials
    
    def is_configured(self) -> bool:
        """Check if accounting client is configured."""
        return self._credentials is not None
    
    def get_email(self) -> Optional[str]:
        """Get accounting email."""
        return self._credentials["email"] if self._credentials else None
    
    @property
    def client(self) -> UserClient:
        """Get or create accounting client."""
        if self._client is None:
            # Try to get service URL from multiple sources
            service_url = self.service_url
            
            # Fallback to credentials if service_url not set
            if not service_url and self._credentials:
                service_url = self._credentials.get("service_url")
            
            if not service_url:
                raise AuthenticationError("No accounting service URL configured")
            
            if not self._credentials:
                raise AuthenticationError("No accounting credentials provided")
            
            try:
                self._client = UserClient(
                    url=service_url,
                    email=self._credentials["email"],
                    password=self._credentials["password"]
                )
            except ServiceException as e:
                raise AuthenticationError(f"Failed to create accounting client: {e}")
        
        return self._client
    
    async def create_transaction_token(self, recipient_email: str) -> str:
        """Create a transaction token for paying a model owner.
        
        Args:
            recipient_email: Email of the model owner to pay
            
        Returns:
            Transaction token string
        """
        try:
            token = self.client.create_transaction_token(
                recipientEmail=recipient_email
            )
            return token
        except ServiceException as e:
            raise PaymentError(f"Failed to create transaction token: {e}")
    
    async def get_account_balance(self) -> float:
        """Get current account balance.
        
        Returns:
            Account balance
        """
        try:
            user_info = self.client.get_user_info()
            return user_info.balance
        except ServiceException as e:
            raise PaymentError(f"Failed to get account balance: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get complete account information.
        
        Returns:
            Dictionary with account details
        """
        try:
            user_info = self.client.get_user_info()
            return {
                "email": self.get_email(),
                "balance": user_info.balance,
                "currency": "USD",  # This might come from user_info in future
                "account_id": getattr(user_info, 'id', None),
                "created_at": getattr(user_info, 'created_at', None)
            }
        except ServiceException as e:
            raise PaymentError(f"Failed to get account info: {e}")
    
    async def validate_credentials(self) -> bool:
        """Test if current credentials are valid.
        
        Returns:
            True if credentials work, False otherwise
        """
        try:
            self.client.get_user_info()
            return True
        except ServiceException:
            return False
    
    def save_credentials(self, config_path: Optional[str] = None):
        """Save credentials to a config file.
        
        WARNING: This saves sensitive credentials to disk. Only call this method
        if you have explicit user consent to save credentials.
        
        Args:
            config_path: Path to save config (default: ~/.syftbox/accounting.json)
        """
        import json
        import os
        from pathlib import Path
        from datetime import datetime
        
        if not self._credentials:
            raise ValueError("No credentials to save")
        
        if config_path is None:
            config_dir = Path.home() / ".syftbox"
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "accounting.json"
        
        try:
            config = {
                "service_url": self.service_url,
                "email": self._credentials["email"],
                "password": self._credentials["password"],
                "created_at": datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(config_path, 0o600)
            
        except Exception as e:
            raise PaymentError(f"Failed to save credentials: {e}")
    
    @classmethod
    def load_from_config(cls, config_path: Optional[str] = None) -> 'AccountingClient':
        """Load accounting client from saved config.
        
        Args:
            config_path: Path to config file (default: ~/.syftbox/accounting.json)
            
        Returns:
            Configured AccountingClient
        """
        import json
        from pathlib import Path
        
        if config_path is None:
            config_path = Path.home() / ".syftbox" / "accounting.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            client = cls()
            client.configure(
                service_url=config["service_url"],
                email=config["email"],
                password=config["password"]
            )
            return client
            
        except FileNotFoundError:
            raise AuthenticationError("No accounting config file found")
        except Exception as e:
            raise AuthenticationError(f"Failed to load config: {e}")
    
    @classmethod
    def from_environment(cls) -> 'AccountingClient':
        """Create accounting client from environment variables.
        
        Looks for:
        - SYFTBOX_ACCOUNTING_URL
        - SYFTBOX_ACCOUNTING_EMAIL  
        - SYFTBOX_ACCOUNTING_PASSWORD
        
        Returns:
            Configured AccountingClient
        """
        import os
        
        service_url = os.getenv("SYFTBOX_ACCOUNTING_URL")
        email = os.getenv("SYFTBOX_ACCOUNTING_EMAIL")
        password = os.getenv("SYFTBOX_ACCOUNTING_PASSWORD")
        
        if not all([service_url, email, password]):
            missing = []
            if not service_url: missing.append("SYFTBOX_ACCOUNTING_URL")
            if not email: missing.append("SYFTBOX_ACCOUNTING_EMAIL") 
            if not password: missing.append("SYFTBOX_ACCOUNTING_PASSWORD")
            
            raise AuthenticationError(f"Missing environment variables: {', '.join(missing)}")
        
        client = cls()
        client.configure(service_url, email, password)
        return client