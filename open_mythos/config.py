from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class AttentionConfig:
    """Attention-related hyperparameters.

    Fields:
        attn_type        -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        n_heads          -- number of query attention heads
        n_kv_heads       -- number of key/value heads (GQA; ignored by MLA)
        kv_lora_rank     -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank      -- [MLA] compressed Q latent dimension
        qk_rope_head_dim -- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim -- [MLA] per-head dims without positional encoding
        v_head_dim       -- [MLA] per-head value dimension
    """

    attn_type: Literal["gqa", "mla"] = "mla"
    n_heads: int = 16
    n_kv_heads: int = 4
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128

    def __post_init__(self) -> None:
        if self.attn_type not in ("gqa", "mla"):
            raise ValueError(
                f"attn_type must be 'gqa' or 'mla', got {self.attn_type!r}"
            )


@dataclass(frozen=True)
class MoEConfig:
    """MoE FFN hyperparameters used inside the recurrent block.

    Fields:
        n_experts          -- total number of routed expert FFNs
        n_shared_experts   -- number of always-active shared experts
        n_experts_per_tok  -- top-K experts selected per token by the router
        expert_dim         -- hidden dimension inside each fine-grained expert
        moe_balance_alpha  -- DeepSeekMoE §3.3 balance-loss weight (0 = disabled,
                              opt-in via MoEFFN.forward(return_balance_loss=True))
    """

    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4
    expert_dim: int = 512
    moe_balance_alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.n_experts_per_tok > self.n_experts:
            raise ValueError(
                f"n_experts_per_tok ({self.n_experts_per_tok}) must be "
                f"<= n_experts ({self.n_experts})"
            )


@dataclass(frozen=True)
class RecurrenceConfig:
    """Recurrent block hyperparameters (ACT halting + depth LoRA).

    Fields:
        max_loop_iters -- default recurrent loop depth T at inference
        act_threshold  -- ACT halting threshold (cumulative probability to stop looping)
        lora_rank      -- rank of the per-loop depth-wise LoRA adapter
    """

    max_loop_iters: int = 16
    act_threshold: float = 0.99
    lora_rank: int = 16

    def __post_init__(self) -> None:
        if not (0.0 < self.act_threshold <= 1.0):
            raise ValueError(
                f"act_threshold ({self.act_threshold}) must be in (0.0, 1.0]"
            )
        if self.lora_rank <= 0:
            raise ValueError(
                f"lora_rank ({self.lora_rank}) must be > 0"
            )


@dataclass(frozen=True)
class QuantConfig:
    """Weight-quantization hyperparameters (BitNet b1.58 W1.58 track).

    Fields:
        mode       -- "none" for full precision, "w158" for ternary weights
                      (absmean scale + straight-through QAT)
        group_size -- activation-quant group width reserved for the A8 stage;
                      must be > 0 whenever mode != "none"
    """

    mode: Literal["none", "w158"] = "none"
    group_size: int = 128

    def __post_init__(self) -> None:
        if self.mode not in ("none", "w158"):
            raise ValueError(
                f"quant mode must be 'none' or 'w158', got {self.mode!r}"
            )
        if self.mode != "none" and self.group_size <= 0:
            raise ValueError(
                f"quant group_size ({self.group_size}) must be > 0 "
                f"when mode != 'none'"
            )


# Field-name partitions used by the flat-kwargs interceptor on MythosConfig.
# Defined at module level (instead of as class-level attrs) so the __init__
# logic doesn't accidentally shadow them as instance attributes.
_ATTENTION_FIELDS = frozenset((
    "attn_type",
    "n_heads",
    "n_kv_heads",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
))
_MOE_FIELDS = frozenset((
    "n_experts",
    "n_shared_experts",
    "n_experts_per_tok",
    "expert_dim",
    "moe_balance_alpha",
))
_RECURRENCE_FIELDS = frozenset((
    "max_loop_iters",
    "act_threshold",
    "lora_rank",
))
_QUANT_FIELDS = frozenset(("quant_mode", "quant_group_size"))
_QUANT_FIELD_MAP = {"quant_mode": "mode", "quant_group_size": "group_size"}
_NESTED_FIELDS = frozenset(("attention", "moe", "recurrence", "quant"))


@dataclass(frozen=True, init=False)
class MythosConfig:
    """Top-level hyperparameter configuration for OpenMythos.

    Composes ``AttentionConfig`` + ``MoEConfig`` + ``RecurrenceConfig`` +
    ``QuantConfig`` around a small set of truly-core fields.  Cross-group
    invariants (e.g. ``dim % n_heads``) are validated in ``__post_init__``
    after each sub-config has already validated its own local invariants.

    Two construction styles are supported:

    - **Nested**: pass ``attention=AttentionConfig(...)``, etc.
    - **Flat (legacy)**: pass every sub-config field as a top-level kwarg
      (``MythosConfig(n_heads=4, n_experts=8, ...)``).  The custom
      ``__init__`` intercepts flat kwargs and routes them into the
      appropriate sub-config.  This keeps pre-Stage-4.4 callers working
      without modification.

    Flat and nested kwargs cannot be mixed in a single call (``TypeError``).

    Property accessors (``self.n_heads``, ``self.attn_type``, ...) expose
    sub-config fields at the top level so legacy read sites keep working.
    """

    # Core fields ---------------------------------------------------------
    vocab_size: int = 32000
    dim: int = 2048
    max_seq_len: int = 4096
    prelude_layers: int = 2
    coda_layers: int = 2
    rope_theta: float = 500000.0
    max_output_tokens: int = 4096
    dropout: float = 0.0

    # Nested sub-configs --------------------------------------------------
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    recurrence: RecurrenceConfig = field(default_factory=RecurrenceConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)

    # Defaults for the core (non-nested) fields.  Kept in sync with the
    # dataclass field defaults above; used by ``__init__`` to fill in
    # omitted kwargs before ``object.__setattr__``.
    _CORE_DEFAULTS = {
        "vocab_size": 32000,
        "dim": 2048,
        "max_seq_len": 4096,
        "prelude_layers": 2,
        "coda_layers": 2,
        "rope_theta": 500000.0,
        "max_output_tokens": 4096,
        "dropout": 0.0,
    }

    def __init__(self, **kwargs: Any) -> None:
        has_flat = any(
            k in (
                _ATTENTION_FIELDS | _MOE_FIELDS | _RECURRENCE_FIELDS | _QUANT_FIELDS
            )
            for k in kwargs
        )
        has_nested = any(k in _NESTED_FIELDS for k in kwargs)
        if has_flat and has_nested:
            raise TypeError(
                "MythosConfig: cannot mix flat and nested kwargs "
                "(e.g. attention=... with n_heads=...)"
            )

        if has_flat:
            attn_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs) if k in _ATTENTION_FIELDS
            }
            moe_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs) if k in _MOE_FIELDS
            }
            rec_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs) if k in _RECURRENCE_FIELDS
            }
            quant_kwargs = {
                _QUANT_FIELD_MAP[k]: kwargs.pop(k)
                for k in list(kwargs)
                if k in _QUANT_FIELDS
            }
            kwargs["attention"] = AttentionConfig(**attn_kwargs)
            kwargs["moe"] = MoEConfig(**moe_kwargs)
            kwargs["recurrence"] = RecurrenceConfig(**rec_kwargs)
            kwargs["quant"] = QuantConfig(**quant_kwargs)

        # Fill in any unspecified core fields with their defaults.  Nested
        # sub-configs default to a fresh instance of their sub-dataclass so
        # the cross-group checks below always have something to read.
        for k, v in self._CORE_DEFAULTS.items():
            kwargs.setdefault(k, v)
        kwargs.setdefault("attention", AttentionConfig())
        kwargs.setdefault("moe", MoEConfig())
        kwargs.setdefault("recurrence", RecurrenceConfig())
        kwargs.setdefault("quant", QuantConfig())

        # Reject any stray kwargs that don't correspond to a known field so
        # typos fail loudly rather than being silently dropped.
        allowed = set(self._CORE_DEFAULTS) | _NESTED_FIELDS
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(
                f"MythosConfig: unknown field(s) {sorted(unknown)}"
            )

        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

        self.__post_init__()

    def __post_init__(self) -> None:
        # Sub-configs have already validated their own local invariants in
        # their own __post_init__.  Only cross-group checks go here.
        if self.dim % self.attention.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by "
                f"n_heads ({self.attention.n_heads})"
            )
        if (
            self.attention.attn_type == "gqa"
            and self.attention.n_heads % self.attention.n_kv_heads != 0
        ):
            raise ValueError(
                f"n_heads ({self.attention.n_heads}) must be divisible by "
                f"n_kv_heads ({self.attention.n_kv_heads})"
            )
        if self.recurrence.lora_rank > self.dim:
            raise ValueError(
                f"lora_rank ({self.recurrence.lora_rank}) must be <= "
                f"dim ({self.dim})"
            )

    # --- Flat-accessor properties (backward compatibility) --------------
    @property
    def attn_type(self) -> str:
        return self.attention.attn_type

    @property
    def n_heads(self) -> int:
        return self.attention.n_heads

    @property
    def n_kv_heads(self) -> int:
        return self.attention.n_kv_heads

    @property
    def kv_lora_rank(self) -> int:
        return self.attention.kv_lora_rank

    @property
    def q_lora_rank(self) -> int:
        return self.attention.q_lora_rank

    @property
    def qk_rope_head_dim(self) -> int:
        return self.attention.qk_rope_head_dim

    @property
    def qk_nope_head_dim(self) -> int:
        return self.attention.qk_nope_head_dim

    @property
    def v_head_dim(self) -> int:
        return self.attention.v_head_dim

    @property
    def n_experts(self) -> int:
        return self.moe.n_experts

    @property
    def n_shared_experts(self) -> int:
        return self.moe.n_shared_experts

    @property
    def n_experts_per_tok(self) -> int:
        return self.moe.n_experts_per_tok

    @property
    def expert_dim(self) -> int:
        return self.moe.expert_dim

    @property
    def moe_balance_alpha(self) -> float:
        return self.moe.moe_balance_alpha

    @property
    def max_loop_iters(self) -> int:
        return self.recurrence.max_loop_iters

    @property
    def act_threshold(self) -> float:
        return self.recurrence.act_threshold

    @property
    def lora_rank(self) -> int:
        return self.recurrence.lora_rank

    @property
    def quant_mode(self) -> str:
        return self.quant.mode

    @property
    def quant_group_size(self) -> int:
        return self.quant.group_size

    # --- Legacy flat constructor shim -----------------------------------
    @classmethod
    def from_flat(cls, **kwargs: Any) -> "MythosConfig":
        """Legacy constructor accepting all fields as flat kwargs.

        Equivalent to calling ``MythosConfig(**kwargs)`` directly; kept as
        an explicit-intent alias for code that wants to signal it is using
        the pre-Stage-4.4 flat API.
        """
        return cls(**kwargs)
