"""
Type definitions and data classes for the SyftBox SDK.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ServiceInfo:
    """Information about a service within a model."""
    type: str
    pricing: float
    charge_type: str
    enabled: bool


@dataclass
class ModelMetadata:
    """Model metadata loaded from metadata.json."""
    name: str
    owner: str
    description: str
    summary: str
    tags: List[str]
    services: List[ServiceInfo]
    version: str
    publish_date: str
    code_hash: str
    delegate_email: Optional[str] = None