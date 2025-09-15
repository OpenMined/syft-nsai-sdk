"""
Parser for SyftBox service metadata and RPC schema files
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.types import ServiceItem, ServiceType, PricingChargeType, ServiceStatus
from ..core.exceptions import MetadataParsingError
from ..discovery.filesystem import SyftURLBuilder
from ..models.service_info import ServiceInfo

logger = logging.getLogger(__name__)

class MetadataParser:
    """Parser for service metadata.json and rpc.schema.json files."""
    
    @staticmethod
    def parse_metadata(metadata_path: Path) -> Dict[str, Any]:
        """Parse metadata.json file."""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise MetadataParsingError(str(metadata_path), f"Invalid JSON: {e}")
        except Exception as e:
            raise MetadataParsingError(str(metadata_path), str(e))
    
    @staticmethod
    def parse_rpc_schema(schema_path: Path) -> Dict[str, Any]:
        """Parse rpc.schema.json file."""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise MetadataParsingError(str(schema_path), f"Invalid JSON in RPC schema: {e}")
        except FileNotFoundError:
            # RPC schema is optional
            return {}
        except Exception as e:
            raise MetadataParsingError(str(schema_path), f"Error reading RPC schema: {e}")
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """Validate that metadata contains required fields."""
        required_fields = [
            "project_name",
            "author", 
            "services"
        ]
        
        for field in required_fields:
            if field not in metadata:
                return False
        
        # Validate services array
        services = metadata.get("services", [])
        if not isinstance(services, list):
            return False
        
        for service in services:
            if not isinstance(service, dict):
                return False
            if "type" not in service or "enabled" not in service:
                return False
        
        return True
    
    @staticmethod
    def extract_service_info_from_metadata_path(metadata_path: Path) -> Dict[str, str]:
        """Extract service information from metadata file path.
        
        Args:
            metadata_path: Path to metadata.json file
            
        Returns:
            Dictionary with datasite and service_name, or empty dict if extraction fails
        """
        try:
            # Expected structure: datasites/{datasite}/public/routers/{service}/metadata.json
            path_parts = metadata_path.parts
            
            # Find datasites directory index
            if "datasites" not in path_parts:
                return {}
            
            datasites_idx = path_parts.index("datasites")
            
            # Validate path structure
            if (len(path_parts) < datasites_idx + 6 or 
                path_parts[datasites_idx + 2] != "public" or 
                path_parts[datasites_idx + 3] != "routers" or 
                path_parts[datasites_idx + 5] != "metadata.json"):
                return {}
            
            datasite = path_parts[datasites_idx + 1]
            service_name = path_parts[datasites_idx + 4]
            
            # Basic validation
            if '@' not in datasite or not service_name:
                return {}
            
            return {
                "datasite": datasite,
                "service_name": service_name,
                "datasites_path": Path(*path_parts[:datasites_idx + 1])
            }
            
        except (ValueError, IndexError):
            return {}
    
    @classmethod
    def parse_services(cls, services_data: List[Dict[str, Any]]) -> List[ServiceItem]:
        """Parse services array from metadata."""
        services = []
        
        for service_data in services_data:
            try:
                # Parse service type
                service_type_str = service_data.get("type", "").lower()
                try:
                    service_type = ServiceType(service_type_str)
                except ValueError:
                    # Skip unknown service types
                    continue
                
                # Parse other fields
                enabled = service_data.get("enabled", False)
                pricing = float(service_data.get("pricing", 0.0))
                
                # Parse charge type
                charge_type_str = service_data.get("charge_type", "per_request").lower()
                try:
                    charge_type = PricingChargeType(charge_type_str)
                except ValueError:
                    charge_type = PricingChargeType.PER_REQUEST
                
                service = ServiceItem(
                    type=service_type,
                    enabled=enabled,
                    pricing=pricing,
                    charge_type=charge_type
                )
                services.append(service)
                
            except (ValueError, TypeError) as e:
                # Skip invalid service entries
                continue
        
        return services
    
    @classmethod
    def create_service_info(cls, 
                         metadata_path: Path,
                         metadata: Dict[str, Any],
                         rpc_schema: Optional[Dict[str, Any]] = None) -> ServiceInfo:
        """Create ServiceInfo from parsed metadata."""
        
        # Extract basic information
        name = metadata.get("project_name", "")
        datasite = metadata.get("author", "")
        summary = metadata.get("summary", "")
        description = metadata.get("description", "")
        tags = metadata.get("tags", [])
        
        # Parse services
        services_data = metadata.get("services", [])
        services = cls.parse_services(services_data)
        
        # Determine config status
        has_enabled_services = any(service.enabled for service in services)
        config_status = ServiceStatus.ACTIVE if has_enabled_services else ServiceStatus.DISABLED
        
        # Extract optional fields
        delegate_email = metadata.get("delegate_email")
        endpoints = metadata.get("documented_endpoints", {})
        
        # Extract or derive publish date
        publish_date = None
        if "publish_date" in metadata:
            # If publish_date is provided in metadata, use it
            try:
                if isinstance(metadata["publish_date"], str):
                    publish_date = datetime.fromisoformat(metadata["publish_date"].replace('Z', '+00:00'))
                else:
                    publish_date = metadata["publish_date"]
            except (ValueError, TypeError):
                publish_date = None
        
        # If no publish_date in metadata, use file modification time as fallback
        if publish_date is None and metadata_path.exists():
            try:
                publish_date = datetime.fromtimestamp(metadata_path.stat().st_mtime)
            except (OSError, ValueError):
                publish_date = None
        
        # Find RPC schema path using centralized path builder
        rpc_schema_path = None
        service_info = cls.extract_service_info_from_metadata_path(metadata_path)
        
        if service_info:
            datasites_path = service_info["datasites_path"]
            datasite_email = service_info["datasite"]
            service_name = service_info["service_name"]
            
            potential_rpc_path = SyftURLBuilder.build_rpc_schema_path(
                datasites_path, datasite_email, service_name
            )
            
            if potential_rpc_path.exists():
                rpc_schema_path = potential_rpc_path
        
        return ServiceInfo(
            name=name,
            datasite=datasite,
            summary=summary,
            description=description,
            tags=tags,
            services=services,
            config_status=config_status,
            health_status=None,
            delegate_email=delegate_email,
            endpoints=endpoints,
            rpc_schema=rpc_schema or {},
            metadata_path=metadata_path,
            rpc_schema_path=rpc_schema_path,
            publish_date=publish_date
        )
    
    @classmethod
    def parse_service_from_files(cls, metadata_path: Path, strict_path_validation: bool = True) -> ServiceInfo:
        """Parse a complete ServiceInfo from metadata file and optional RPC schema.
        
        Args:
            metadata_path: Path to metadata.json file
            strict_path_validation: If True, raises error if path extraction fails.
                                  If False, falls back to old method.
        """
        
        # Parse metadata
        metadata = cls.parse_metadata(metadata_path)
        
        # Validate metadata
        if not cls.validate_metadata(metadata):
            raise MetadataParsingError(
                str(metadata_path), 
                "Metadata validation failed - missing required fields"
            )
        
        # Extract service info from path
        service_info = cls.extract_service_info_from_metadata_path(metadata_path)
        
        if not service_info:
            error_msg = f"Failed to extract service info from path: {metadata_path}"
            if strict_path_validation:
                raise MetadataParsingError(str(metadata_path), error_msg)
            # else:
            #     # Log warning and use fallback
            #     logger.warning(f"{error_msg} - falling back to metadata fields")
        
        # Try to parse RPC schema using centralized path building
        rpc_schema = {}
        
        if service_info:
            datasites_path = service_info["datasites_path"]
            datasite = service_info["datasite"]
            service_name = service_info["service_name"]

            # Try multiple locations for RPC schema
            potential_rpc_paths = [
                # Primary location: datasites/{datasite}/app_data/{service}/rpc/rpc.schema.json
                SyftURLBuilder.build_rpc_schema_path(datasites_path, datasite, service_name),
                # Fallback: same directory as metadata
                metadata_path.parent / "rpc.schema.json",
                # Fallback: subdirectory of metadata location
                metadata_path.parent / "rpc" / "rpc.schema.json",
            ]
            
            for rpc_path in potential_rpc_paths:
                if rpc_path.exists():
                    try:
                        rpc_schema = cls.parse_rpc_schema(rpc_path)
                        break
                    except MetadataParsingError:
                        # Continue trying other paths
                        continue
        
        # elif not strict_path_validation:
        #     # TODO: Remove this fallback code once centralized path building is proven stable in production
        #     # Only use fallback if strict validation is disabled
        #     datasite = metadata.get("author", "")
        #     service_name = metadata.get("project_name", "")
            
        #     logger.warning(f"Using fallback extraction - datasite: {datasite}, service: {service_name}")
            
        #     if datasite and service_name:
        #         try:
        #             # Legacy path traversal - will be removed in future version
        #             datasites_root = metadata_path.parent.parent.parent.parent
        #             potential_rpc_paths = [
        #                 datasites_root / datasite / "app_data" / service_name / "rpc" / "rpc.schema.json",
        #                 metadata_path.parent / "rpc.schema.json",
        #                 metadata_path.parent / "rpc" / "rpc.schema.json",
        #             ]
                    
        #             for rpc_path in potential_rpc_paths:
        #                 if rpc_path.exists():
        #                     try:
        #                         rpc_schema = cls.parse_rpc_schema(rpc_path)
        #                         logger.info(f"Found RPC schema at: {rpc_path} (fallback method)")
        #                         break
        #                     except MetadataParsingError:
        #                         continue
        #         except (IndexError, AttributeError) as e:
        #             logger.warning(f"Fallback path extraction failed: {e}")
        
        # Create service info
        return cls.create_service_info(metadata_path, metadata, rpc_schema)

class SchemaValidator:
    """Validates parsed metadata and RPC schemas."""
    
    @staticmethod
    def validate_service_types(services: List[Dict[str, Any]]) -> List[str]:
        """Validate service types and return list of warnings."""
        warnings = []
        
        supported_types = {service_type.value for service_type in ServiceType}
        
        for i, service in enumerate(services):
            service_type = service.get("type", "").lower()
            if service_type not in supported_types:
                warnings.append(
                    f"Unknown service type '{service_type}' at index {i}. "
                    f"Supported types: {', '.join(supported_types)}"
                )
        
        return warnings
    
    @staticmethod
    def validate_pricing(services: List[Dict[str, Any]]) -> List[str]:
        """Validate pricing information and return list of warnings."""
        warnings = []
        
        for i, service in enumerate(services):
            pricing = service.get("pricing")
            if pricing is not None:
                try:
                    price_float = float(pricing)
                    if price_float < 0:
                        warnings.append(f"Negative pricing at service index {i}: {pricing}")
                except (ValueError, TypeError):
                    warnings.append(f"Invalid pricing format at service index {i}: {pricing}")
        
        return warnings
    
    @staticmethod
    def validate_rpc_schema(rpc_schema: Dict[str, Any]) -> List[str]:
        """Validate RPC schema format and return list of warnings."""
        warnings = []
        
        if not rpc_schema:
            return warnings
        
        # Check for common RPC endpoints
        expected_endpoints = ["/health", "/chat", "/search"]
        available_endpoints = list(rpc_schema.keys())
        
        missing_common = [ep for ep in expected_endpoints if ep not in available_endpoints]
        if missing_common:
            warnings.append(f"Missing common RPC endpoints: {', '.join(missing_common)}")
        
        return warnings
    
    @classmethod
    def validate_service_info(cls, service_info: ServiceInfo) -> List[str]:
        """Validate complete ServiceInfo and return list of warnings."""
        warnings = []
        
        # Basic validation
        if not service_info.name:
            warnings.append("Missing service name")
        if not service_info.datasite or '@' not in service_info.datasite:
            warnings.append("Missing or invalid datasite email")
        if not service_info.services:
            warnings.append("No services defined")
        
        # Service validation
        services_data = [
            {
                "type": service.type.value,
                "pricing": service.pricing,
                "enabled": service.enabled
            }
            for service in service_info.services
        ]
        
        warnings.extend(cls.validate_service_types(services_data))
        warnings.extend(cls.validate_pricing(services_data))
        warnings.extend(cls.validate_rpc_schema(service_info.rpc_schema))
        
        return warnings