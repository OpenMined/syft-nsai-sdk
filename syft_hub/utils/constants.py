"""
SyftBox constants and configuration values
"""

# Directory structure constants
SYFTBOX_DIR = ".syftbox"
CONFIG_FILENAME = "config.json"
DATASITES_DIR = "datasites"
APP_DATA_DIR = "app_data"
PUBLIC_DIR = "public"
ROUTERS_DIR = "routers"
RPC_DIR = "rpc"

# File names
METADATA_FILENAME = "metadata.json"
RPC_SCHEMA_FILENAME = "rpc.schema.json"
OPENAPI_FILENAME = "openapi.json"

# URL schemes and endpoints
SYFT_SCHEME = "syft"
HTTP_SCHEMES = {"http", "https"}

# API endpoints
API_VERSION = "v1"
SEND_MESSAGE_ENDPOINT = f"/api/{API_VERSION}/send/msg"
HEALTH_ENDPOINT = "/health"
OPENAPI_ENDPOINT = "/openapi.json"

# Default ports and addresses
DEFAULT_APP_PORT = 7938
DEFAULT_HOST = "127.0.0.1"

# Network timeouts (seconds)
DEFAULT_SOCKET_TIMEOUT = 1
DEFAULT_HTTP_TIMEOUT = 30

# Configuration validation
REQUIRED_CONFIG_FIELDS = ["data_dir", "email", "server_url"]
OPTIONAL_CONFIG_FIELDS = ["refresh_token"]

# Process names to check for SyftBox
SYFTBOX_PROCESS_NAMES = ["syftbox"]

# Email validation
EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# Installation URLs
DESKTOP_RELEASES_URL = "https://github.com/OpenMined/SyftUI/releases"
QUICK_INSTALL_URL = "https://syftbox.net/install.sh"
LEGACY_INSTALL_URL = "https://install.syftbox.openmined.org"

# Documentation URLs
CLI_DOCS_URL = "https://github.com/OpenMined/syftbox"
DESKTOP_DOCS_URL = "https://github.com/OpenMined/SyftUI"

# Default cache server
DEFAULT_CACHE_SERVER = "https://syftbox.net"