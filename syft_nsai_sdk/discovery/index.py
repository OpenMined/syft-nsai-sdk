"""
Model indexing and caching system for fast lookups
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
import threading
import logging

from ..core.config import SyftBoxConfig
from ..core.types import ModelInfo, ServiceType
from .scanner import ModelScanner
from .parser import MetadataParser

logger = logging.getLogger(__name__)


class ModelIndex:
    """In-memory index of discovered models for fast searching and filtering."""
    
    def __init__(self, syftbox_config: SyftBoxConfig, cache_ttl: int = 300):
        """Initialize model index.
        
        Args:
            syftbox_config: SyftBox configuration
            cache_ttl: Cache time-to-live in seconds
        """
        self.config = syftbox_config
        self.scanner = ModelScanner(syftbox_config)
        self.parser = MetadataParser()
        self.cache_ttl = cache_ttl
        
        # Index data structures
        self._models: Dict[str, ModelInfo] = {}  # name -> ModelInfo
        self._by_owner: Dict[str, List[str]] = defaultdict(list)  # owner -> [model_names]
        self._by_service: Dict[ServiceType, List[str]] = defaultdict(list)  # service -> [model_names]
        self._by_tag: Dict[str, List[str]] = defaultdict(list)  # tag -> [model_names]
        self._by_pricing: Dict[str, List[str]] = defaultdict(list)  # pricing_tier -> [model_names]
        
        # Cache metadata
        self._last_updated: Optional[float] = None
        self._is_building: bool = False
        self._build_lock = threading.Lock()
    
    def build_index(self, force_refresh: bool = False) -> None:
        """Build or refresh the model index.
        
        Args:
            force_refresh: Force rebuild even if cache is valid
        """
        with self._build_lock:
            # Check if rebuild is needed
            if not force_refresh and self._is_cache_valid():
                logger.debug("Model index cache is valid, skipping rebuild")
                return
            
            if self._is_building:
                logger.debug("Index build already in progress")
                return
            
            self._is_building = True
            
            try:
                logger.info("Building model index...")
                start_time = time.time()
                
                # Clear existing index
                self._clear_index()
                
                # Scan for metadata files
                metadata_paths = self.scanner.scan_all_datasites()
                logger.debug(f"Found {len(metadata_paths)} metadata files")
                
                # Parse and index models
                indexed_count = 0
                for metadata_path in metadata_paths:
                    try:
                        model_info = self.parser.parse_model_from_files(metadata_path)
                        self._add_model_to_index(model_info)
                        indexed_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to index {metadata_path}: {e}")
                        continue
                
                self._last_updated = time.time()
                build_time = self._last_updated - start_time
                
                logger.info(f"Model index built: {indexed_count} models in {build_time:.2f}s")
                
            finally:
                self._is_building = False
    
    def refresh_index(self) -> None:
        """Force refresh of the index."""
        self.build_index(force_refresh=True)
    
    def _is_cache_valid(self) -> bool:
        """Check if the current cache is still valid."""
        if self._last_updated is None:
            return False
        
        age = time.time() - self._last_updated
        return age < self.cache_ttl
    
    def _clear_index(self) -> None:
        """Clear all index data structures."""
        self._models.clear()
        self._by_owner.clear()
        self._by_service.clear()
        self._by_tag.clear()
        self._by_pricing.clear()
    
    def _add_model_to_index(self, model_info: ModelInfo) -> None:
        """Add a model to all relevant indexes.
        
        Args:
            model_info: Model to add to index
        """
        name = model_info.name
        
        # Main model index
        self._models[name] = model_info
        
        # Owner index
        self._by_owner[model_info.owner].append(name)
        
        # Service index
        for service in model_info.services:
            if service.enabled:
                self._by_service[service.type].append(name)
        
        # Tag index
        for tag in model_info.tags:
            self._by_tag[tag.lower()].append(name)
        
        # Pricing tier index
        pricing_tier = self._get_pricing_tier(model_info.min_pricing)
        self._by_pricing[pricing_tier].append(name)
    
    def _get_pricing_tier(self, price: float) -> str:
        """Categorize pricing into tiers."""
        if price == 0:
            return "free"
        elif price <= 0.01:
            return "budget"
        elif price <= 0.10:
            return "standard"
        else:
            return "premium"
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get model by exact name.
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo if found, None otherwise
        """
        self._ensure_index_built()
        return self._models.get(name)
    
    def get_models_by_owner(self, owner: str) -> List[ModelInfo]:
        """Get all models by a specific owner.
        
        Args:
            owner: Owner email
            
        Returns:
            List of models owned by the user
        """
        self._ensure_index_built()
        model_names = self._by_owner.get(owner, [])
        return [self._models[name] for name in model_names]
    
    def get_models_by_service(self, service_type: ServiceType) -> List[ModelInfo]:
        """Get all models supporting a specific service.
        
        Args:
            service_type: Service type to filter by
            
        Returns:
            List of models supporting the service
        """
        self._ensure_index_built()
        model_names = self._by_service.get(service_type, [])
        return [self._models[name] for name in model_names]
    
    def get_models_by_tags(self, tags: List[str], match_all: bool = False) -> List[ModelInfo]:
        """Get models by tags.
        
        Args:
            tags: List of tags to match
            match_all: If True, model must have ALL tags; if False, ANY tag
            
        Returns:
            List of matching models
        """
        self._ensure_index_built()
        
        if not tags:
            return list(self._models.values())
        
        # Normalize tags
        normalized_tags = [tag.lower() for tag in tags]
        
        if match_all:
            # Must have ALL tags (intersection)
            matching_names = None
            for tag in normalized_tags:
                tag_models = set(self._by_tag.get(tag, []))
                if matching_names is None:
                    matching_names = tag_models
                else:
                    matching_names = matching_names.intersection(tag_models)
            
            return [self._models[name] for name in (matching_names or [])]
        else:
            # Must have ANY tag (union)
            matching_names = set()
            for tag in normalized_tags:
                matching_names.update(self._by_tag.get(tag, []))
            
            return [self._models[name] for name in matching_names]
    
    def get_models_by_pricing_tier(self, tier: str) -> List[ModelInfo]:
        """Get models by pricing tier.
        
        Args:
            tier: Pricing tier ("free", "budget", "standard", "premium")
            
        Returns:
            List of models in the pricing tier
        """
        self._ensure_index_built()
        model_names = self._by_pricing.get(tier, [])
        return [self._models[name] for name in model_names]
    
    def search_models(self, 
                     name_pattern: Optional[str] = None,
                     owner: Optional[str] = None,
                     service_type: Optional[ServiceType] = None,
                     tags: Optional[List[str]] = None,
                     max_cost: Optional[float] = None,
                     free_only: bool = False,
                     **kwargs) -> List[ModelInfo]:
        """Search models with multiple criteria.
        
        Args:
            name_pattern: Pattern to match in model names (case-insensitive)
            owner: Filter by owner email
            service_type: Filter by service type
            tags: Filter by tags
            max_cost: Maximum cost per request
            free_only: Only include free models
            **kwargs: Additional filter criteria
            
        Returns:
            List of matching models
        """
        self._ensure_index_built()
        
        # Start with all models
        candidates = list(self._models.values())
        
        # Apply filters progressively
        if owner:
            candidates = [m for m in candidates if m.owner == owner]
        
        if service_type:
            candidates = [m for m in candidates if m.supports_service(service_type)]
        
        if tags:
            # Use tag index for efficiency
            tag_models = set()
            for model in self.get_models_by_tags(tags, match_all=False):
                tag_models.add(model.name)
            candidates = [m for m in candidates if m.name in tag_models]
        
        if free_only:
            candidates = [m for m in candidates if m.min_pricing == 0]
        elif max_cost is not None:
            candidates = [m for m in candidates if m.min_pricing <= max_cost]
        
        if name_pattern:
            pattern_lower = name_pattern.lower()
            candidates = [m for m in candidates if pattern_lower in m.name.lower()]
        
        return candidates
    
    def get_all_models(self) -> List[ModelInfo]:
        """Get all indexed models.
        
        Returns:
            List of all models
        """
        self._ensure_index_built()
        return list(self._models.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Dictionary with statistics
        """
        self._ensure_index_built()
        
        total_models = len(self._models)
        
        # Count by service type
        service_counts = {}
        for service_type in ServiceType:
            service_counts[service_type.value] = len(self._by_service[service_type])
        
        # Count by pricing tier
        pricing_counts = {}
        for tier in ["free", "budget", "standard", "premium"]:
            pricing_counts[tier] = len(self._by_pricing[tier])
        
        # Top owners
        top_owners = sorted(
            [(owner, len(models)) for owner, models in self._by_owner.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top tags
        tag_counts = {tag: len(models) for tag, models in self._by_tag.items()}
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_models": total_models,
            "total_owners": len(self._by_owner),
            "total_tags": len(self._by_tag),
            "last_updated": self._last_updated,
            "cache_age_seconds": time.time() - self._last_updated if self._last_updated else None,
            "service_counts": service_counts,
            "pricing_counts": pricing_counts,
            "top_owners": top_owners,
            "top_tags": top_tags,
        }
    
    def _ensure_index_built(self) -> None:
        """Ensure the index is built before use."""
        if self._last_updated is None or not self._is_cache_valid():
            self.build_index()
    
    def list_owners(self) -> List[str]:
        """Get list of all model owners.
        
        Returns:
            Sorted list of owner emails
        """
        self._ensure_index_built()
        return sorted(self._by_owner.keys())
    
    def list_tags(self) -> List[str]:
        """Get list of all tags.
        
        Returns:
            Sorted list of unique tags
        """
        self._ensure_index_built()
        return sorted(self._by_tag.keys())
    
    def get_tag_popularity(self) -> Dict[str, int]:
        """Get tag usage statistics.
        
        Returns:
            Dictionary mapping tags to usage count
        """
        self._ensure_index_built()
        return {tag: len(models) for tag, models in self._by_tag.items()}
    
    def find_similar_models(self, model: ModelInfo, limit: int = 5) -> List[ModelInfo]:
        """Find models similar to the given model.
        
        Args:
            model: Reference model
            limit: Maximum number of similar models to return
            
        Returns:
            List of similar models, sorted by similarity
        """
        self._ensure_index_built()
        
        candidates = [m for m in self._models.values() if m.name != model.name]
        
        # Calculate similarity scores
        scored_models = []
        for candidate in candidates:
            score = self._calculate_similarity(model, candidate)
            if score > 0:
                scored_models.append((score, candidate))
        
        # Sort by similarity and return top results
        scored_models.sort(key=lambda x: x[0], reverse=True)
        return [model for score, model in scored_models[:limit]]
    
    def _calculate_similarity(self, model1: ModelInfo, model2: ModelInfo) -> float:
        """Calculate similarity score between two models.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        score = 0.0
        
        # Same owner bonus
        if model1.owner == model2.owner:
            score += 0.3
        
        # Service overlap
        services1 = set(s.type for s in model1.services if s.enabled)
        services2 = set(s.type for s in model2.services if s.enabled)
        
        if services1 and services2:
            service_overlap = len(services1.intersection(services2)) / len(services1.union(services2))
            score += 0.4 * service_overlap
        
        # Tag overlap
        tags1 = set(tag.lower() for tag in model1.tags)
        tags2 = set(tag.lower() for tag in model2.tags)
        
        if tags1 and tags2:
            tag_overlap = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
            score += 0.3 * tag_overlap
        
        return score


class PersistentModelIndex(ModelIndex):
    """Model index with disk persistence for faster startup."""
    
    def __init__(self, syftbox_config: SyftBoxConfig, 
                 cache_ttl: int = 300,
                 cache_file: Optional[Path] = None):
        """Initialize persistent model index.
        
        Args:
            syftbox_config: SyftBox configuration
            cache_ttl: Cache time-to-live in seconds
            cache_file: Path to cache file (defaults to ~/.syftbox/model_index.json)
        """
        super().__init__(syftbox_config, cache_ttl)
        
        if cache_file is None:
            cache_file = syftbox_config.data_dir / ".syftbox" / "model_index.json"
        
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    def build_index(self, force_refresh: bool = False) -> None:
        """Build index, trying to load from cache first."""
        with self._build_lock:
            # Try loading from cache first
            if not force_refresh and self._load_from_cache():
                logger.debug("Loaded model index from cache")
                return
            
            # Fall back to full rebuild
            super().build_index(force_refresh)
            
            # Save to cache
            self._save_to_cache()
    
    def _load_from_cache(self) -> bool:
        """Load index from cache file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.cache_file.exists():
                return False
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check cache validity
            last_updated = cache_data.get("last_updated")
            if not last_updated:
                return False
            
            age = time.time() - last_updated
            if age >= self.cache_ttl:
                logger.debug("Cache file expired")
                return False
            
            # Restore index data
            self._models = {}
            models_data = cache_data.get("models", {})
            
            for name, model_data in models_data.items():
                # This is a simplified restoration - in practice you'd need
                # to properly deserialize ModelInfo objects
                # For now, we'll skip cache loading and always rebuild
                pass
            
            return False  # Skip cache loading for now
            
        except Exception as e:
            logger.debug(f"Failed to load index cache: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """Save current index to cache file."""
        try:
            # Prepare cache data
            cache_data = {
                "last_updated": self._last_updated,
                "models": {},  # Simplified - would need proper serialization
                "metadata": {
                    "version": "1.0",
                    "total_models": len(self._models),
                    "cache_ttl": self.cache_ttl,
                }
            }
            
            # Save to file
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Saved model index cache to {self.cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save index cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear the cache file."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.debug("Cleared model index cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")