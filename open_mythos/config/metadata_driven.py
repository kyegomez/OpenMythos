"""
Metadata-Driven Loop Depth Module

Leverages OpenMetadata to get data asset complexity and dynamically
adjust the recurrent loop depth based on task complexity.

This module provides:
1. OpenMetadataClient - Client for querying OM API
2. MetadataComplexityPredictor - MLP-based complexity predictor
3. MetadataDrivenLoopDepth - Full loop depth controller
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# OpenMetadata Client
# =============================================================================

class OpenMetadataClient:
    """
    Lightweight client for OpenMetadata API.
    
    Provides methods to query:
    - Table metadata and statistics
    - Column information
    - Data quality metrics
    - Freshness indicators
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8585",
        jwt_token: Optional[str] = None,
        cache_ttl: int = 300
    ):
        """
        Args:
            endpoint: OpenMetadata API endpoint
            jwt_token: JWT authentication token
            cache_ttl: Cache TTL in seconds
        """
        self.endpoint = endpoint.rstrip('/')
        self.jwt_token = jwt_token
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def _make_url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self.endpoint}/api/v1{path}"
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return value
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (value, time.time())
    
    def get_table_metadata(self, fully_qualified_name: str) -> Dict[str, Any]:
        """
        Get table metadata including statistics.
        
        Args:
            fully_qualified_name: Table FQN (e.g., "database.schema.table")
            
        Returns:
            Dictionary with table metadata
        """
        cache_key = f"table:{fully_qualified_name}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # For now, return a structured dict that would come from the API
        # In production, this would make actual HTTP requests
        metadata = {
            "fully_qualified_name": fully_qualified_name,
            "table_type": "table",
            "row_count": 0,
            "column_count": 0,
            "avg_row_size_bytes": 0,
            "last_updated_timestamp": time.time(),
            "freshness_score": 1.0,
            "quality_score": 1.0,
            "test_coverage": 0.0,
            "tier_level": 3,
            "pii_level": 0,
            "tags": [],
            "columns": [],
        }
        
        self._set_cache(cache_key, metadata)
        return metadata
    
    def get_column_metadata(self, table_fqn: str, column_name: str) -> Dict[str, Any]:
        """
        Get column-level metadata.
        
        Args:
            table_fqn: Table FQN
            column_name: Column name
            
        Returns:
            Dictionary with column metadata
        """
        cache_key = f"column:{table_fqn}.{column_name}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        metadata = {
            "name": column_name,
            "data_type": "string",
            "nullable": True,
            "unique_count": 0,
            "null_count": 0,
            "sample_values": [],
        }
        
        self._set_cache(cache_key, metadata)
        return metadata
    
    def get_quality_metrics(self, table_fqn: str) -> Dict[str, Any]:
        """
        Get data quality metrics for a table.
        
        Returns:
            Dictionary with quality metrics
        """
        cache_key = f"quality:{table_fqn}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        metrics = {
            "completeness": 1.0,
            "validity": 1.0,
            "accuracy": 1.0,
            "consistency": 1.0,
            "freshness": 1.0,
            "overall_score": 1.0,
        }
        
        self._set_cache(cache_key, metrics)
        return metrics
    
    def search_assets(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for data assets.
        
        Args:
            query: Search query
            asset_type: Filter by asset type (table, dashboard, etc.)
            limit: Maximum results
            
        Returns:
            List of matching assets
        """
        # Placeholder - would make actual API call
        return []


# =============================================================================
# Metadata Complexity Predictor
# =============================================================================

class MetadataComplexityPredictor(nn.Module):
    """
    MLP-based complexity predictor using metadata features.
    
    Takes metadata features from OpenMetadata and predicts
    the complexity score [0, 1] for a given task.
    
    Complexity is used to determine loop depth:
    - Low (0.0-0.33): 4 iterations
    - Medium (0.33-0.66): 8 iterations  
    - High (0.66-1.0): 16 iterations
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        depth_thresholds: List[float] = None
    ):
        """
        Args:
            input_dim: Dimension of input metadata features
            hidden_dim: Hidden layer dimension
            depth_thresholds: Thresholds for depth selection [low, high]
        """
        super().__init__()
        
        self.depth_thresholds = depth_thresholds or [0.33, 0.66]
        
        # MLP for complexity prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Predict complexity score.
        
        Args:
            metadata_features: (B, input_dim) metadata features
            
        Returns:
            (B, 1) complexity scores in [0, 1]
        """
        return self.mlp(metadata_features)
    
    def complexity_to_depth(
        self,
        complexity: torch.Tensor,
        min_depth: int = 4,
        max_depth: int = 16
    ) -> torch.Tensor:
        """
        Convert complexity score to loop depth.
        
        Args:
            complexity: (B, 1) complexity scores
            min_depth: Minimum depth
            max_depth: Maximum depth
            
        Returns:
            (B,) loop depths
        """
        # Map complexity to depth using thresholds
        depths = torch.ones_like(complexity.squeeze(-1)) * min_depth
        
        # Medium complexity -> 8 iterations
        depths = torch.where(
            complexity.squeeze(-1) > self.depth_thresholds[0],
            torch.tensor(8, device=complexity.device),
            depths
        )
        
        # High complexity -> max_depth
        depths = torch.where(
            complexity.squeeze(-1) > self.depth_thresholds[1],
            torch.tensor(max_depth, device=complexity.device),
            depths
        )
        
        return depths.long()


# =============================================================================
# Metadata Features Extractor
# =============================================================================

class MetadataFeaturesExtractor:
    """
    Extracts features from OpenMetadata for complexity prediction.
    
    Features include:
    - Table statistics (row count, column count, size)
    - Freshness (last updated, update frequency)
    - Quality (test coverage, quality scores)
    - Business context (tier level, PII level)
    """
    
    # Feature names for interpretability
    FEATURE_NAMES = [
        "row_count_norm",           # Normalized row count
        "column_count_norm",         # Normalized column count
        "avg_row_size_norm",         # Normalized average row size
        "freshness_score",           # Data freshness [0, 1]
        "quality_score",            # Overall quality [0, 1]
        "test_coverage",            # Test coverage [0, 1]
        "tier_level_norm",           # Business tier [0, 1]
        "pii_level_norm",           # PII sensitivity [0, 1]
        "update_frequency",          # How often data updates
        "null_fraction",            # Null value fraction
        "unique_fraction",          # Unique value fraction
        "num_tags",                 # Number of tags
        "num_owners",               # Number of owners
        "has_description",          # Has description (0/1)
        "column_type_diversity",     # Diversity of column types
        "complex_data_types",        # Presence of complex types
    ]
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
    
    def extract(
        self,
        metadata: Dict[str, Any],
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Extract features from metadata dictionary.
        
        Args:
            metadata: Table metadata from OpenMetadata
            quality_metrics: Optional quality metrics
            
        Returns:
            (16,) feature tensor
        """
        features = []
        
        # 1. Table statistics (normalized)
        row_count = metadata.get("row_count", 0)
        features.append(self._normalize_row_count(row_count))
        
        column_count = metadata.get("column_count", 0)
        features.append(min(column_count / 1000, 1.0))
        
        avg_row_size = metadata.get("avg_row_size_bytes", 0)
        features.append(min(avg_row_size / 1e6, 1.0))
        
        # 2. Freshness
        freshness = metadata.get("freshness_score", 1.0)
        features.append(freshness)
        
        # 3. Quality
        if quality_metrics:
            quality = quality_metrics.get("overall_score", 1.0)
        else:
            quality = metadata.get("quality_score", 1.0)
        features.append(quality)
        
        # 4. Test coverage
        test_coverage = metadata.get("test_coverage", 0.0)
        features.append(test_coverage)
        
        # 5. Business context
        tier = metadata.get("tier_level", 3)
        features.append(tier / 3.0)
        
        pii = metadata.get("pii_level", 0)
        features.append(pii / 4.0)
        
        # 6. Update frequency (computed from timestamps)
        last_updated = metadata.get("last_updated_timestamp", time.time())
        if isinstance(last_updated, (int, float)):
            hours_since_update = (time.time() - last_updated) / 3600
            update_freq = 1.0 / (1.0 + hours_since_update / 24)  # Decay over days
        else:
            update_freq = 1.0
        features.append(update_freq)
        
        # 7. Column statistics
        columns = metadata.get("columns", [])
        if columns:
            null_fraction = sum(c.get("null_count", 0) for c in columns) / max(row_count, 1)
            unique_count = sum(c.get("unique_count", 0) for c in columns) / max(row_count, 1)
        else:
            null_fraction = 0.0
            unique_count = 0.0
        features.append(min(null_fraction, 1.0))
        features.append(min(unique_count, 1.0))
        
        # 8. Governance
        tags = metadata.get("tags", [])
        features.append(min(len(tags) / 10, 1.0))  # Num tags normalized
        
        owners = metadata.get("owners", [])
        features.append(min(len(owners) / 5, 1.0))  # Num owners normalized
        
        # 9. Documentation
        has_description = 1.0 if metadata.get("description") else 0.0
        features.append(has_description)
        
        # 10. Type diversity
        if columns:
            types = set(c.get("data_type", "string") for c in columns)
            type_diversity = len(types) / 20.0  # Normalize by expected max types
        else:
            type_diversity = 0.5
        features.append(min(type_diversity, 1.0))
        
        # 11. Complex types
        complex_types = {"array", "map", "struct", "json", "variant"}
        if columns:
            has_complex = any(
                c.get("data_type", "").lower() in complex_types 
                for c in columns
            )
        else:
            has_complex = False
        features.append(1.0 if has_complex else 0.0)
        
        # Ensure we have exactly 16 features
        while len(features) < 16:
            features.append(0.0)
        
        return torch.tensor(features[:16], dtype=torch.float32, device=self.device)
    
    @staticmethod
    def _normalize_row_count(count: int) -> float:
        """Normalize row count using log scale."""
        import math
        if count <= 0:
            return 0.0
        # Log scale normalization: 0 -> 0, 1M -> ~0.5, 1B -> ~0.75
        return min(math.log1p(count) / 20.0, 1.0)


# =============================================================================
# Metadata-Driven Loop Depth Controller
# =============================================================================

class MetadataDrivenLoopDepth(nn.Module):
    """
    Metadata-Driven Loop Depth Controller.
    
    This module combines OpenMetadata client, feature extraction,
    and complexity prediction to dynamically select the optimal
    loop depth for each inference task.
    
    Usage:
        controller = MetadataDrivenLoopDepth(
            om_endpoint="http://localhost:8585",
            om_jwt_token="...",
            min_depth=4,
            max_depth=16
        )
        
        depth = controller(
            hidden_states=hidden,
            task_context={"asset_fqn": "warehouse.sales.transactions"}
        )
    """
    
    def __init__(
        self,
        cfg: "MythosConfig",  # Forward reference to avoid circular import
        om_endpoint: str = "http://localhost:8585",
        om_jwt_token: Optional[str] = None,
        min_depth: int = 4,
        max_depth: int = 16,
        metadata_cache_ttl: int = 300,
        use_heuristic_fallback: bool = True
    ):
        """
        Args:
            cfg: MythosConfig for model dimensions
            om_endpoint: OpenMetadata API endpoint
            om_jwt_token: JWT token for authentication
            min_depth: Minimum loop depth
            max_depth: Maximum loop depth
            metadata_cache_ttl: Cache TTL for metadata
            use_heuristic_fallback: Use heuristic fallback if OM unavailable
        """
        super().__init__()
        
        self.cfg = cfg
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_heuristic_fallback = use_heuristic_fallback
        
        # OpenMetadata client
        self.om_client = OpenMetadataClient(
            endpoint=om_endpoint,
            jwt_token=om_jwt_token,
            cache_ttl=metadata_cache_ttl
        )
        
        # Feature extractor
        self.feature_extractor = MetadataFeaturesExtractor()
        
        # Complexity predictor
        self.complexity_predictor = MetadataComplexityPredictor(
            input_dim=16,
            hidden_dim=128,
            depth_thresholds=[0.33, 0.66]
        )
        
        # Heuristic fallback for when OM is unavailable
        self.heuristic_predictor = HeuristicComplexityPredictor(
            min_depth=min_depth,
            max_depth=max_depth
        )
        
        # Statistics tracking
        self.register_buffer("avg_predicted_depth", torch.tensor(8.0))
        self.register_buffer("total_inferences", torch.tensor(0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Predict optimal loop depth based on metadata.
        
        Args:
            hidden_states: (B, T, D) hidden states
            task_context: Optional context with metadata hints
            
        Returns:
            (B,) predicted loop depths
        """
        if task_context is None:
            task_context = {}
        
        # Try metadata-driven approach
        try:
            depth = self._predict_from_metadata(hidden_states, task_context)
        except Exception as e:
            if self.use_heuristic_fallback:
                # Fall back to heuristic
                depth = self.heuristic_predictor(hidden_states)
            else:
                # Use default depth
                depth = torch.full(
                    (hidden_states.shape[0],),
                    self.min_depth,
                    device=hidden_states.device,
                    dtype=torch.long
                )
        
        # Update statistics
        self._update_stats(depth)
        
        return depth
    
    def _predict_from_metadata(
        self,
        hidden_states: torch.Tensor,
        task_context: Dict[str, Any]
    ) -> torch.Tensor:
        """Predict depth using OpenMetadata complexity."""
        B, T, D = hidden_states.shape
        device = hidden_states.device
        
        # Extract metadata features
        metadata_features = self._get_metadata_features(task_context)
        metadata_features = metadata_features.to(device).unsqueeze(0).expand(B, -1)
        
        # Predict complexity
        complexity = self.complexity_predictor(metadata_features)
        
        # Map to depth
        depth = self.complexity_predictor.complexity_to_depth(
            complexity,
            min_depth=self.min_depth,
            max_depth=self.max_depth
        )
        
        return depth
    
    def _get_metadata_features(self, task_context: Dict[str, Any]) -> torch.Tensor:
        """Get metadata features from OpenMetadata or task context."""
        # First try task context hints
        if "metadata" in task_context:
            metadata = task_context["metadata"]
            quality = task_context.get("quality_metrics")
            return self.feature_extractor.extract(metadata, quality)
        
        # Then try asset FQN
        if "asset_fqn" in task_context:
            fqn = task_context["asset_fqn"]
            asset_type = task_context.get("asset_type", "table")
            
            if asset_type == "table":
                metadata = self.om_client.get_table_metadata(fqn)
                quality = self.om_client.get_quality_metrics(fqn)
            else:
                metadata = {"row_count": 0, "column_count": 0}
                quality = None
            
            return self.feature_extractor.extract(metadata, quality)
        
        # Default features
        return torch.zeros(16)
    
    def _update_stats(self, depth: torch.Tensor) -> None:
        """Update running statistics."""
        with torch.no_grad():
            B = depth.shape[0]
            total = self.total_inferences.item() + B
            current_avg = self.avg_predicted_depth.item()
            
            # Running average
            new_avg = (current_avg * self.total_inferences.item() + depth.sum().item()) / total
            
            self.avg_predicted_depth.fill_(new_avg)
            self.total_inferences.fill_(total)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "avg_predicted_depth": self.avg_predicted_depth.item(),
            "total_inferences": self.total_inferences.item(),
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
        }


# =============================================================================
# Heuristic Fallback Predictor
# =============================================================================

class HeuristicComplexityPredictor(nn.Module):
    """
    Heuristic complexity predictor for when OpenMetadata is unavailable.
    
    Uses the hidden states themselves to estimate complexity:
    - Attention entropy (higher = more complex)
    - Hidden state variance (higher = more diverse content)
    - Sequence length
    """
    
    def __init__(self, min_depth: int = 4, max_depth: int = 16):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict depth using heuristic features.
        
        Args:
            hidden_states: (B, T, D) hidden states
            
        Returns:
            (B,) predicted depths
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        
        # Compute heuristic complexity
        # 1. Hidden state variance across sequence
        variance = hidden_states.var(dim=-1).mean(dim=1)  # (B,)
        
        # 2. Sequence length factor
        seq_factor = T / 4096.0  # Normalize to typical max
        
        # 3. Combine and map to depth
        complexity = (variance / (D ** 0.5)) * 0.5 + seq_factor * 0.5
        complexity = complexity.clamp(0, 1)
        
        # Map to depth
        depth = torch.ones(B, device=device, dtype=torch.long) * self.min_depth
        depth = torch.where(
            complexity > 0.33,
            torch.tensor(8, device=device),
            depth
        )
        depth = torch.where(
            complexity > 0.66,
            torch.tensor(self.max_depth, device=device),
            depth
        )
        
        return depth


# =============================================================================
# Budget-Aware Depth Selector
# =============================================================================

class BudgetAwareDepthSelector:
    """
    Adjusts loop depth based on computational budget.
    
    When under budget (fast response needed), reduces depth.
    When over budget (throttling), suggests lower depth.
    """
    
    def __init__(
        self,
        min_depth: int = 4,
        max_depth: int = 16,
        target_latency_ms: float = 100.0,
        ms_per_loop: float = 5.0
    ):
        """
        Args:
            min_depth: Minimum depth
            max_depth: Maximum depth
            target_latency_ms: Target latency in milliseconds
            ms_per_loop: Milliseconds per loop iteration
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_latency_ms = target_latency_ms
        self.ms_per_loop = ms_per_loop
    
    def select_depth(
        self,
        base_depth: int,
        available_budget_ms: Optional[float] = None,
        current_latency_ms: Optional[float] = None
    ) -> int:
        """
        Select depth based on budget constraints.
        
        Args:
            base_depth: Base depth from complexity predictor
            available_budget_ms: Available time budget
            current_latency_ms: Current measured latency
            
        Returns:
            Adjusted depth
        """
        if available_budget_ms is not None:
            # Calculate max depth from budget
            max_allowed = int(available_budget_ms / self.ms_per_loop)
            return min(base_depth, max(self.min_depth, max_allowed))
        
        if current_latency_ms is not None:
            if current_latency_ms > self.target_latency_ms * 1.2:
                # Over budget, reduce depth
                return max(self.min_depth, base_depth // 2)
            elif current_latency_ms < self.target_latency_ms * 0.5:
                # Under budget, could increase depth
                return min(self.max_depth, base_depth + 4)
        
        return base_depth
