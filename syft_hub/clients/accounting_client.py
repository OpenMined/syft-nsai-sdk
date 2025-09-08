"""
SyftBox Accounting Client for managing payments and transactions
"""
import json
import os
import logging

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from syft_accounting_sdk import UserClient, ServiceException

from ..models.validation import UserAccountModel
from ..core.types import APIException
from ..core.exceptions import PaymentError, AuthenticationError

logger = logging.getLogger(__name__)


class AccountingClient:
    """Client for handling accounting operations."""
    
    def __init__(self, 
                 accounting_url: Optional[str] = "https://syftaccounting.centralus.cloudapp.azure.com/",
                 credentials: Optional[Dict[str, str]] = None
        ):
        """Initialize accounting client.
        
        Args:
            accounting_url: URL of the accounting service
            credentials: Dict with 'email', 'password', and optionally 'accounting_url'
        """
        self.accounting_url = accounting_url
        self._credentials = credentials
        self._client = None

    def create_accounting_user(
        self,
        email: str,
        password: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> UserAccountModel:
        """Create a user account on the service."""
        try:
            user, user_pwd = UserClient.create_user(
                url=self.accounting_url,
                organization=organization,
                email=email,
                password=password,
            )

            # Initialize _credentials if None
            if self._credentials is None:
                self._credentials = {}

            self._credentials['email'] = email
            self._credentials['password'] = user_pwd
            self._credentials['organization'] = organization

            return UserAccountModel(
                email=email,
                password=user_pwd,
                organization=organization,
                balance=getattr(user, 'balance', 0.0)  # Default to 0.0 if not available
            )
        except ServiceException as e:
            logger.error(
                f"Failed to create user account: {e.message} with {e.status_code}"
            )
            if e.status_code == 409:
                raise APIException(
                    f"User account already exists: {e.message}",
                    status_code=e.status_code,
                )
            else:
                raise APIException(
                    f"Failed to create user account: {e.message} with {e.status_code}",
                    status_code=e.status_code,
                )
        # return user, user_pwd
        # Create UserAccountModel object with required fields
        # return UserAccountModel(
        #     email=email,
        #     password=user_pwd,
        #     organization=organization,
        #     balance=getattr(user, 'balance', 0.0)  # Default to 0.0 if not available
        # )

    # def add_or_update_credentials(
    #     self,
    #     email: str,
    #     organization: Optional[str] = None,
    #     password: Optional[str] = None,
    # ) -> UserAccountModel:
    #     """Add or update user account credentials to the repository."""

    #     try:
    #         credentials = self.repository.add_or_update_credentials(
    #             email=email,
    #             password=password,
    #             accounting_url=self.accounting_config.url,
    #             organization=organization,
    #         )
    #     except Exception as e:
    #         logger.error(f"Failed to add or update credentials: {e}")
    #         raise APIException(
    #             f"Failed to add or update credentials: {e}",
    #             status_code=500,
    #         )

    #     try:
    #         user_info = self.client.get_user_info()
    #     except ServiceException as e:
    #         raise APIException(
    #             f"Failed to get user info: {e}",
    #             status_code=e.status_code,
    #         )

    #     return UserAccountModel(
    #         email=credentials.email,
    #         organization=credentials.organization,
    #         balance=user_info.balance,
    #         password=credentials.password,
    #     )

    def configure(self, accounting_url: str, email: str, password: str):
        """Configure accounting client.
        
        Args:
            accounting_url: Accounting service URL
            email: User email
            password: User password
        """
        self.accounting_url = accounting_url
        self._credentials = {
            "accounting_url": accounting_url,
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
            accounting_url = self.accounting_url
            
            # Fallback to credentials if accounting_url not set
            if not accounting_url and self._credentials:
                accounting_url = self._credentials.get("accounting_url")
            
            if not accounting_url:
                raise AuthenticationError("No accounting service URL configured")
            
            if not self._credentials:
                raise AuthenticationError("No accounting credentials provided")
            
            try:
                self._client = UserClient(
                    url=accounting_url,
                    email=self._credentials["email"],
                    password=self._credentials["password"]
                )
            except ServiceException as e:
                raise AuthenticationError(f"Failed to create accounting client: {e}")
        
        return self._client
    
    async def create_transaction_token(self, recipient_email: str) -> str:
        """Create a transaction token for paying a service datasite.
        
        Args:
            recipient_email: Email of the service datasite to pay
            
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
        
        if not self._credentials:
            raise ValueError("No credentials to save")
        
        if config_path is None:
            config_dir = Path.home() / ".syftbox"
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "accounting.json"
        
        try:
            config = {
                "service_url": self.accounting_url,
                "email": self._credentials["email"],
                "password": self._credentials["password"],
                "created_at": datetime.now().isoformat(),
                "organization": self._credentials["organization"],
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(config_path, 0o600)
            
        except Exception as e:
            raise PaymentError(f"Failed to save credentials: {e}")
    
    @classmethod
    def setup_accounting_discovery(cls) -> tuple['AccountingClient', bool]:
        """Try to auto-discover credentials and return client + success status."""
        
        # Try environment variables
        try:
            client = cls.from_environment()
            return client, True  # Successfully configured
        except AuthenticationError:
            pass
        
        # Try config file
        try:
            client = cls.load_from_config()
            return client, True  # Successfully configured
        except AuthenticationError:
            pass
        
        # No credentials found
        return cls(), False  # Not configured

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
                accounting_url=config["service_url"],
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
        
        accounting_url = os.getenv("SYFTBOX_ACCOUNTING_URL")
        email = os.getenv("SYFTBOX_ACCOUNTING_EMAIL")
        password = os.getenv("SYFTBOX_ACCOUNTING_PASSWORD")
        
        if not all([accounting_url, email, password]):
            missing = []
            if not accounting_url: missing.append("SYFTBOX_ACCOUNTING_URL")
            if not email: missing.append("SYFTBOX_ACCOUNTING_EMAIL") 
            if not password: missing.append("SYFTBOX_ACCOUNTING_PASSWORD")
            
            raise AuthenticationError(f"Missing environment variables: {', '.join(missing)}")
        
        client = cls()
        client.configure(accounting_url, email, password)
        return client