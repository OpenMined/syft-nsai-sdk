# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2025-09-25

### Fixed
- Resolved dependency conflicts with Google Colab and other environments
- Relaxed ipython constraint from >=8.18.1 to >=7.34.0 for Colab compatibility  
- Relaxed pandas constraint from >=2.3.1 to >=2.0.0 for broader compatibility
- Added cryptography version constraint (<44.0.0) to avoid conflicts
- Removed duplicate pydantic and typer entries
- Removed asyncio dependency (part of Python standard library)

### Changed
- Improved dependency compatibility across different Python environments

## [0.1.4] - 2025-09-25

### Changed
- Made syft-accounting-sdk an optional dependency to enable PyPI publication
- Accounting features now require installing with `pip install syft-hub[accounting]`
- Improved error handling when accounting SDK is not available

### Fixed
- Resolved PyPI publication issue with git dependencies

## [0.1.3] - 2025-09-25

### Added
- Added missing `SyftBoxAuthClient` for authentication management
- Authentication client now provides user email and token handling with guest mode fallback

### Fixed
- Fixed import error for missing `auth_client.py` module
- Resolved module import issues that prevented package initialization

### Changed
- Improved handling of free requests
- Enhanced real-time health checks for chat/search operations
- Better error handling for unavailable services

## [0.1.2] - 2025-09-21

### Changed
- Improved handling of free requests
- Enhanced real-time health checks to chat/search operations

### Fixed
- Fixed unavailability log messages

## [0.1.0] - Initial Release

### Added
- Initial release of SyftBox NSAI SDK
- Client for discovering and using AI services across SyftBox network
- Support for chat and search services
- Built-in payment handling via accounting client
- RAG coordination capabilities
- Service discovery and health monitoring
- Conversation management with context retention
- CLI interface for service management
