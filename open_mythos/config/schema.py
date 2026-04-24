"""
OpenMythos Configuration Schema

Schema-First configuration using Pydantic models with JSON Schema generation.
Supports YAML/JSON configuration files with full validation.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union
from typing_extensions import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Pydantic Models for Enhanced Configuration
# =============================================================================


class ModelConfig(BaseModel):
    """
    Core model architecture configuration.
    
    Attributes:
        vocab_size: Token vocabulary size
        dim: Model hidden dimension (must be divisible by 64)
        n_heads: Number of query attention heads
        n_kv_heads: Number of key/value heads for GQA
        max_seq_len: Maximum sequence length for RoPE precomputation
        max_output_tokens: Maximum tokens to generate per forward pass
        dropout: Dropout rate (0.0 to disable)
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Core architecture
    vocab_size: int = Field(default=32000, ge=1000, le=1000000, description="Token vocabulary size")
    dim: int = Field(default=2048, ge=128, le=16384, description="Model hidden dimension")
    n_heads: int = Field(default=16, ge=1, le=128, description="Number of query attention heads")
    n_kv_heads: int = Field(default=4, ge=1, le=128, description="Number of key/value heads (GQA)")
    max_seq_len: int = Field(default=4096, ge=128, le=1048576, description="Maximum sequence length")
    max_output_tokens: int = Field(default=4096, ge=1, le=1048576, description="Max generation tokens")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")
    
    # Prelude and Coda layers
    prelude_layers: int = Field(default=2, ge=0, le=32, description="Standard transformer layers before loop")
    coda_layers: int = Field(default=2, ge=0, le=32, description="Standard transformer layers after loop")
    
    # RoPE configuration
    rope_theta: float = Field(default=500000.0, ge=1.0, le=10000000.0, description="RoPE base frequency")
    
    # LoRA configuration
    lora_rank: int = Field(default=16, ge=1, le=256, description="LoRA adapter rank")
    
    @field_validator('dim')
    @classmethod
    def validate_dim_divisible_by_64(cls, v: int) -> int:
        if v % 64 != 0:
            raise ValueError(f'dim must be divisible by 64, got {v}')
        return v
    
    @field_validator('n_kv_heads')
    @classmethod
    def validate_kv_heads(cls, v: int, info) -> int:
        n_heads = info.data.get('n_heads', 16)
        if v > n_heads:
            raise ValueError(f'n_kv_heads ({v}) cannot exceed n_heads ({n_heads})')
        return v


class AttentionConfig(BaseModel):
    """
    Attention mechanism configuration.
    
    Supports two attention types:
    - "gqa": Grouped Query Attention
    - "mla": Multi-Latent Attention (DeepSeek style)
    """
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    attn_type: Literal["gqa", "mla"] = Field(
        default="mla", 
        description="Attention type: 'gqa' for Grouped Query Attention, 'mla' for Multi-Latent Attention"
    )
    
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = Field(
        default=512, 
        ge=64, 
        le=2048, 
        description="[MLA] Compressed KV latent dimension stored in cache"
    )
    q_lora_rank: int = Field(
        default=1536, 
        ge=64, 
        le=4096, 
        description="[MLA] Compressed Q latent dimension"
    )
    qk_rope_head_dim: int = Field(
        default=64, 
        ge=8, 
        le=256, 
        description="[MLA] Per-head dims that receive RoPE"
    )
    qk_nope_head_dim: int = Field(
        default=128, 
        ge=8, 
        le=256, 
        description="[MLA] Per-head dims without positional encoding"
    )
    v_head_dim: int = Field(
        default=128, 
        ge=8, 
        le=256, 
        description="[MLA] Per-head value dimension"
    )
    
    @model_validator(mode='after')
    def validate_mla_params(self) -> Self:
        if self.attn_type == "mla":
            # MLA requires specific head dimension relationships
            if self.qk_rope_head_dim + self.qk_nope_head_dim != self.qk_rope_head_dim * 2:
                pass  # Allow any combination
        return self


class LoopConfig(BaseModel):
    """
    Recurrent loop configuration.
    
    Controls the recurrent depth adaptation mechanism including:
    - Minimum/maximum loop depth
    - Curriculum learning phases
    - ACT (Adaptive Computation Time) halting
    """
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    # Loop depth bounds
    min_depth: int = Field(default=4, ge=1, le=32, description="Minimum recurrent loop depth")
    max_depth: int = Field(default=16, ge=4, le=64, description="Maximum recurrent loop depth")
    max_loop_iters: int = Field(default=16, ge=1, le=64, description="Default recurrent depth at inference")
    
    # Curriculum learning
    curriculum_enabled: bool = Field(default=True, description="Enable curriculum learning")
    curriculum_phases: List[Dict[str, Any]] = Field(
        default=[
            {"name": "warmup", "duration_steps": 2000, "depth": 4},
            {"name": "transition", "duration_steps": 5000, "depth": 10},
            {"name": "mixed", "duration_steps": 10000, "depth_range": [4, 16]},
            {"name": "adaptive", "duration_steps": -1, "mode": "complexity_aware"}
        ],
        description="Curriculum learning phases"
    )
    
    # ACT configuration
    act_enabled: bool = Field(default=True, description="Enable ACT halting")
    act_threshold: float = Field(default=0.99, ge=0.5, le=1.0, description="ACT halting threshold")
    act_halting_type: Literal["exponential", "linear"] = Field(
        default="exponential", 
        description="ACT halting weight type"
    )
    
    # Complexity-aware depth selection
    complexity_thresholds: List[float] = Field(
        default=[0.33, 0.66],
        description="Thresholds for complexity-based depth selection"
    )
    
    @model_validator(mode='after')
    def validate_depth_bounds(self) -> Self:
        if self.min_depth > self.max_depth:
            raise ValueError(f'min_depth ({self.min_depth}) cannot exceed max_depth ({self.max_depth})')
        if self.max_loop_iters > self.max_depth:
            raise ValueError(f'max_loop_iters ({self.max_loop_iters}) cannot exceed max_depth ({self.max_depth})')
        return self


class MoEConfig(BaseModel):
    """
    Mixture of Experts configuration.
    
    Controls the MoE FFN routing inside the recurrent block.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    # Expert counts
    n_experts: int = Field(default=64, ge=1, le=256, description="Total number of routed experts")
    n_shared_experts: int = Field(default=2, ge=0, le=16, description="Number of always-active shared experts")
    n_experts_per_tok: int = Field(default=4, ge=1, le=16, description="Top-K experts selected per token")
    expert_dim: int = Field(default=512, ge=64, le=4096, description="Hidden dimension inside each expert")
    
    # Routing
    capacity_factor: float = Field(
        default=1.25, 
        ge=1.0, 
        le=4.0, 
        description="Token capacity factor for load balancing"
    )
    expert_specialization: bool = Field(default=True, description="Enable task-conditioned expert specialization")
    capacity_aware_routing: bool = Field(default=True, description="Enable capacity-aware routing")
    
    # Expert groups (for specialization)
    expert_groups: List[Dict[str, Any]] = Field(
        default=[
            {"id": 0, "name": "syntax", "experts": "0-15", "domain": "syntax_morphology"},
            {"id": 1, "name": "knowledge", "experts": "16-31", "domain": "factual_named_entities"},
            {"id": 2, "name": "reasoning", "experts": "32-47", "domain": "logic_deduction"},
            {"id": 3, "name": "math_code", "experts": "48-63", "domain": "math_code_formal"}
        ],
        description="Expert specialization groups"
    )
    
    @field_validator('n_experts_per_tok')
    @classmethod
    def validate_topk(cls, v: int, info) -> int:
        n_experts = info.data.get('n_experts', 64)
        if v > n_experts:
            raise ValueError(f'n_experts_per_tok ({v}) cannot exceed n_experts ({n_experts})')
        return v


class DataContractConfig(BaseModel):
    """
    Data Contract configuration for inference SLA and quality guarantees.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    enabled: bool = Field(default=False, description="Enable inference data contracts")
    contract_path: Optional[str] = Field(default=None, description="Path to contract YAML file")
    
    # SLA constraints
    max_latency_ms: float = Field(default=100.0, ge=1.0, le=60000.0, description="Max latency in ms")
    max_memory_mb: int = Field(default=4096, ge=256, le=1048576, description="Max memory in MB")
    min_throughput_tokens_per_sec: float = Field(
        default=50.0, 
        ge=1.0, 
        le=100000.0, 
        description="Minimum throughput"
    )
    
    # Quality standards
    min_accuracy: float = Field(default=0.85, ge=0.0, le=1.0, description="Minimum accuracy requirement")
    max_confidence_variance: float = Field(default=0.1, ge=0.0, le=1.0, description="Max confidence variance")
    min_attention_coverage: float = Field(default=0.7, ge=0.0, le=1.0, description="Min attention coverage")
    
    # Monitoring
    alert_on_latency_p95: bool = Field(default=True, description="Alert on P95 latency violation")
    alert_on_quality_drop: bool = Field(default=True, description="Alert on quality drop")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


class OpenMetadataIntegrationConfig(BaseModel):
    """
    OpenMetadata integration configuration.
    
    Enables metadata-driven features:
    - Metadata-driven loop depth selection
    - Inference lineage tracking
    - Quality-aware routing
    - MCP server for AI assistants
    """
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    enabled: bool = Field(default=False, description="Enable OpenMetadata integration")
    endpoint: str = Field(default="http://localhost:8585", description="OpenMetadata API endpoint")
    jwt_token: Optional[str] = Field(default=None, description="JWT authentication token")
    
    # Feature flags
    enable_metadata_driven_depth: bool = Field(
        default=False, 
        description="Use OpenMetadata complexity for loop depth"
    )
    enable_inference_lineage: bool = Field(
        default=False, 
        description="Track inference lineage in OpenMetadata"
    )
    enable_quality_aware_routing: bool = Field(
        default=False, 
        description="Use data quality for MoE routing"
    )
    enable_mcp_server: bool = Field(
        default=False, 
        description="Enable MCP server for AI assistants"
    )
    enable_data_contract: bool = Field(
        default=False, 
        description="Enable inference data contracts"
    )
    
    # Cache configuration
    metadata_cache_ttl_seconds: int = Field(
        default=300, 
        ge=0, 
        le=86400, 
        description="Metadata cache TTL in seconds"
    )
    
    # MCP server config
    mcp_port: int = Field(default=8080, ge=1024, le=65535, description="MCP server port")


class P0EnhancementsConfig(BaseModel):
    """P0 Enhancements: Multi-scale loop depth and curriculum learning."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    enabled: bool = Field(default=True, description="Enable P0 enhancements")
    multiscale_loop_enabled: bool = Field(
        default=True, 
        description="Enable dynamic multi-scale loop depth"
    )
    curriculum_enabled: bool = Field(
        default=True, 
        description="Enable curriculum learning scheduler"
    )


class P1EnhancementsConfig(BaseModel):
    """P1 Enhancements: Flash MLA, consistency regularization, capacity routing."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    enabled: bool = Field(default=True, description="Enable P1 enhancements")
    flash_mla_enabled: bool = Field(
        default=True, 
        description="Enable Flash MLA with tile-based computation"
    )
    consistency_regularization_enabled: bool = Field(
        default=True, 
        description="Enable loop consistency regularization"
    )
    capacity_aware_routing_enabled: bool = Field(
        default=True, 
        description="Enable capacity-aware MoE routing"
    )


class P2EnhancementsConfig(BaseModel):
    """P2 Enhancements: Cross-layer KV, speculative decoding, expert specialization."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    enabled: bool = Field(default=True, description="Enable P2 enhancements")
    cross_layer_kv_enabled: bool = Field(
        default=True, 
        description="Enable cross-layer KV sharing"
    )
    cross_layer_kv_share_every: int = Field(
        default=3, 
        ge=1, 
        le=12, 
        description="Share KV every N layers"
    )
    speculative_decoding_enabled: bool = Field(
        default=True, 
        description="Enable speculative decoding"
    )
    speculative_k_tokens: int = Field(
        default=4, 
        ge=1, 
        le=16, 
        description="Number of tokens to speculate"
    )
    expert_specialization_enabled: bool = Field(
        default=True, 
        description="Enable task-conditioned expert specialization"
    )


class P3EnhancementsConfig(BaseModel):
    """P3 Enhancements: Hierarchical loops, meta-learned depth."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    enabled: bool = Field(default=True, description="Enable P3 enhancements")
    hierarchical_loops_enabled: bool = Field(
        default=True, 
        description="Enable hierarchical recurrent blocks"
    )
    n_outer_loops: int = Field(default=4, ge=1, le=16, description="Number of outer loop iterations")
    n_inner_loops: int = Field(default=4, ge=1, le=16, description="Number of inner loops per outer")
    meta_learned_depth_enabled: bool = Field(
        default=True, 
        description="Enable LSTM-based meta-learned depth prediction"
    )
    lstm_hidden: int = Field(default=512, ge=64, le=2048, description="LSTM hidden dimension")


class EnhancementsConfig(BaseModel):
    """All P0-P3 enhancements configuration."""
    
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    p0: P0EnhancementsConfig = Field(default_factory=P0EnhancementsConfig)
    p1: P1EnhancementsConfig = Field(default_factory=P1EnhancementsConfig)
    p2: P2EnhancementsConfig = Field(default_factory=P2EnhancementsConfig)
    p3: P3EnhancementsConfig = Field(default_factory=P3EnhancementsConfig)
    
    # Master switch
    all_enabled: bool = Field(default=True, description="Enable all enhancements")


class MythosEnhancedConfig(BaseModel):
    """
    Complete OpenMythos Enhanced Configuration.
    
    This is the primary configuration class for OpenMythos with all
    enhancements (P0-P3) and OpenMetadata integration support.
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore"  # Allow extra fields for forward compatibility
    )
    
    # Version for schema migration
    version: str = Field(default="1.0.0", description="Configuration version")
    
    # Core configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    
    # Integration
    data_contract: DataContractConfig = Field(default_factory=DataContractConfig)
    openmetadata: OpenMetadataIntegrationConfig = Field(default_factory=OpenMetadataIntegrationConfig)
    
    # Enhancements
    enhancements: EnhancementsConfig = Field(default_factory=EnhancementsConfig)
    
    # Metadata for tracking
    name: Optional[str] = Field(default=None, description="Configuration name")
    description: Optional[str] = Field(default=None, description="Configuration description")
    
    @model_validator(mode='after')
    def validate_enhancement_dependencies(self) -> Self:
        """Validate that enhancement dependencies are satisfied."""
        # P3 requires P0 to be enabled
        if self.enhancements.p3.enabled and not self.enhancements.p0.enabled:
            raise ValueError("P3 enhancements require P0 enhancements to be enabled")
        
        # OpenMetadata features require OpenMetadata to be enabled
        if self.openmetadata.enabled:
            if self.openmetadata.enable_metadata_driven_depth and not self.enhancements.p0.enabled:
                raise ValueError("Metadata-driven depth requires P0 enhancements")
        
        return self
    
    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get the effective configuration as a dictionary.
        Used for serialization and schema generation.
        """
        return self.model_dump(exclude_none=True)
    
    def to_yaml(self, path: Optional[str] = None) -> str:
        """Serialize to YAML format."""
        import yaml
        data = self.model_dump(exclude_none=True, serialize_as_any=True)
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path:
            with open(path, 'w') as f:
                f.write(yaml_str)
        return yaml_str
    
    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Serialize to JSON format."""
        import json
        data = self.model_dump(exclude_none=True, serialize_as_any=True)
        json_str = json.dumps(data, indent=indent)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_json(cls, path: str) -> Self:
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def generate_json_schema(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate JSON Schema for this configuration.
        
        Returns the JSON Schema dictionary and optionally writes to file.
        """
        schema = self.model_json_schema()
        
        # Add metadata
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        schema["title"] = "OpenMythos Configuration Schema"
        schema["description"] = "Configuration schema for OpenMythos Enhanced with P0-P3 enhancements"
        
        if path:
            import json
            with open(path, 'w') as f:
                json.dump(schema, f, indent=2)
        
        return schema


# =============================================================================
# Backward Compatibility: Legacy MythosConfig
# =============================================================================


@dataclass
class MythosConfig:
    """
    Legacy dataclass-based configuration for backward compatibility.
    
    This class wraps MythosEnhancedConfig to provide backward compatibility
    with existing code using the dataclass-based MythosConfig.
    
    Example:
        # New way (recommended)
        config = MythosEnhancedConfig.from_yaml("config.yaml")
        
        # Legacy way (backward compatible)
        config = MythosConfig(
            dim=2048,
            n_heads=16,
            n_experts=64
        )
    """
    
    # Core fields (matching original dataclass)
    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2
    
    # Attention
    attn_type: str = "mla"
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4
    expert_dim: int = 512
    
    # ACT
    act_threshold: float = 0.99
    rope_theta: float = 500000.0
    lora_rank: int = 16
    max_output_tokens: int = 4096
    dropout: float = 0.0
    
    # Enhancement flags (P0-P3)
    enable_multiscale_loop: bool = True
    enable_curriculum: bool = True
    enable_p1_flash_mla: bool = True
    enable_p1_consistency_regularization: bool = True
    enable_p1_capacity_aware_routing: bool = True
    enable_p2_cross_layer_kv: bool = True
    enable_p2_speculative_decoding: bool = True
    enable_p2_expert_specialization: bool = True
    enable_p3_hierarchical: bool = True
    enable_p3_meta_learned_depth: bool = True
    
    # OpenMetadata integration
    om_endpoint: Optional[str] = None
    om_jwt_token: Optional[str] = None
    enable_metadata_driven_depth: bool = False
    enable_inference_lineage: bool = False
    enable_quality_aware_routing: bool = False
    enable_mcp_server: bool = False
    
    @classmethod
    def from_enhanced(cls, enhanced: MythosEnhancedConfig) -> "MythosConfig":
        """Create legacy config from enhanced config."""
        return cls(
            vocab_size=enhanced.model.vocab_size,
            dim=enhanced.model.dim,
            n_heads=enhanced.model.n_heads,
            n_kv_heads=enhanced.model.n_kv_heads,
            max_seq_len=enhanced.model.max_seq_len,
            max_loop_iters=enhanced.loop.max_loop_iters,
            prelude_layers=enhanced.model.prelude_layers,
            coda_layers=enhanced.model.coda_layers,
            attn_type=enhanced.attention.attn_type,
            kv_lora_rank=enhanced.attention.kv_lora_rank,
            q_lora_rank=enhanced.attention.q_lora_rank,
            qk_rope_head_dim=enhanced.attention.qk_rope_head_dim,
            qk_nope_head_dim=enhanced.attention.qk_nope_head_dim,
            v_head_dim=enhanced.attention.v_head_dim,
            n_experts=enhanced.moe.n_experts,
            n_shared_experts=enhanced.moe.n_shared_experts,
            n_experts_per_tok=enhanced.moe.n_experts_per_tok,
            expert_dim=enhanced.moe.expert_dim,
            act_threshold=enhanced.loop.act_threshold,
            rope_theta=enhanced.model.rope_theta,
            lora_rank=enhanced.model.lora_rank,
            max_output_tokens=enhanced.model.max_output_tokens,
            dropout=enhanced.model.dropout,
            enable_multiscale_loop=enhanced.enhancements.p0.multiscale_loop_enabled,
            enable_curriculum=enhanced.enhancements.p0.curriculum_enabled,
            enable_p1_flash_mla=enhanced.enhancements.p1.flash_mla_enabled,
            enable_p1_consistency_regularization=enhanced.enhancements.p1.consistency_regularization_enabled,
            enable_p1_capacity_aware_routing=enhanced.enhancements.p1.capacity_aware_routing_enabled,
            enable_p2_cross_layer_kv=enhanced.enhancements.p2.cross_layer_kv_enabled,
            enable_p2_speculative_decoding=enhanced.enhancements.p2.speculative_decoding_enabled,
            enable_p2_expert_specialization=enhanced.enhancements.p2.expert_specialization_enabled,
            enable_p3_hierarchical=enhanced.enhancements.p3.hierarchical_loops_enabled,
            enable_p3_meta_learned_depth=enhanced.enhancements.p3.meta_learned_depth_enabled,
            om_endpoint=enhanced.openmetadata.endpoint if enhanced.openmetadata.enabled else None,
            om_jwt_token=enhanced.openmetadata.jwt_token,
            enable_metadata_driven_depth=enhanced.openmetadata.enable_metadata_driven_depth,
            enable_inference_lineage=enhanced.openmetadata.enable_inference_lineage,
            enable_quality_aware_routing=enhanced.openmetadata.enable_quality_aware_routing,
            enable_mcp_server=enhanced.openmetadata.enable_mcp_server,
        )
    
    def to_enhanced(self) -> MythosEnhancedConfig:
        """Convert to enhanced configuration."""
        return MythosEnhancedConfig(
            model=ModelConfig(
                vocab_size=self.vocab_size,
                dim=self.dim,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                max_seq_len=self.max_seq_len,
                max_output_tokens=self.max_output_tokens,
                dropout=self.dropout,
                prelude_layers=self.prelude_layers,
                coda_layers=self.coda_layers,
                rope_theta=self.rope_theta,
                lora_rank=self.lora_rank,
            ),
            attention=AttentionConfig(
                attn_type=self.attn_type,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                qk_nope_head_dim=self.qk_nope_head_dim,
                v_head_dim=self.v_head_dim,
            ),
            loop=LoopConfig(
                min_depth=4,
                max_depth=self.max_loop_iters,
                max_loop_iters=self.max_loop_iters,
                curriculum_enabled=self.enable_curriculum,
                act_enabled=True,
                act_threshold=self.act_threshold,
            ),
            moe=MoEConfig(
                n_experts=self.n_experts,
                n_shared_experts=self.n_shared_experts,
                n_experts_per_tok=self.n_experts_per_tok,
                expert_dim=self.expert_dim,
                capacity_aware_routing=self.enable_p1_capacity_aware_routing,
                expert_specialization=self.enable_p2_expert_specialization,
            ),
            enhancements=EnhancementsConfig(
                p0=P0EnhancementsConfig(
                    enabled=True,
                    multiscale_loop_enabled=self.enable_multiscale_loop,
                    curriculum_enabled=self.enable_curriculum,
                ),
                p1=P1EnhancementsConfig(
                    enabled=True,
                    flash_mla_enabled=self.enable_p1_flash_mla,
                    consistency_regularization_enabled=self.enable_p1_consistency_regularization,
                    capacity_aware_routing_enabled=self.enable_p1_capacity_aware_routing,
                ),
                p2=P2EnhancementsConfig(
                    enabled=True,
                    cross_layer_kv_enabled=self.enable_p2_cross_layer_kv,
                    speculative_decoding_enabled=self.enable_p2_speculative_decoding,
                    expert_specialization_enabled=self.enable_p2_expert_specialization,
                ),
                p3=P3EnhancementsConfig(
                    enabled=True,
                    hierarchical_loops_enabled=self.enable_p3_hierarchical,
                    meta_learned_depth_enabled=self.enable_p3_meta_learned_depth,
                ),
            ),
            openmetadata=OpenMetadataIntegrationConfig(
                enabled=self.om_endpoint is not None,
                endpoint=self.om_endpoint or "http://localhost:8585",
                jwt_token=self.om_jwt_token,
                enable_metadata_driven_depth=self.enable_metadata_driven_depth,
                enable_inference_lineage=self.enable_inference_lineage,
                enable_quality_aware_routing=self.enable_quality_aware_routing,
                enable_mcp_server=self.enable_mcp_server,
            ),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "MythosConfig":
        """Load from YAML file (legacy interface)."""
        enhanced = MythosEnhancedConfig.from_yaml(path)
        return cls.from_enhanced(enhanced)
    
    @classmethod
    def from_json(cls, path: str) -> "MythosConfig":
        """Load from JSON file (legacy interface)."""
        enhanced = MythosEnhancedConfig.from_json(path)
        return cls.from_enhanced(enhanced)
