"""
LoRA (Low-Rank Adaptation) for OpenMythos.

Enables parameter-efficient fine-tuning of OpenMythos models by adding
low-rank adapters to attention and FFN layers. Only trains ~0.1-1% of
total parameters while maintaining full model quality.

Usage:
    from open_mythos.lora import LoRAConfig, apply_lora, get_lora_params

    config = LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
    model = apply_lora(model, config)

    # Only train LoRA parameters
    trainable = get_lora_params(model)
    # trainable = ~0.5% of total params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adaptation.

    Args:
        rank: Low-rank dimension (default: 16)
        alpha: Scaling factor (alpha/rank) (default: 32)
        dropout: LoRA dropout probability (default: 0.05)
        target_modules: Which linear layers to adapt (default: attention projections)
        bias: Whether to train bias terms ("none", "all", "lora_only")
        modules_to_save: Modules to fully train (not LoRA, but saved with adapter)
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"  # "none" | "all" | "lora_only"
    modules_to_save: Optional[List[str]] = None


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adapter.

    Wraps an existing linear layer with low-rank decomposition:
        W' = W + (alpha/rank) * B @ A

    Where:
        W: Original frozen weights (out_features × in_features)
        A: Low-rank projection (rank × in_features)
        B: Low-rank projection (out_features × rank)
        alpha: Scaling factor

    Args:
        original: The original linear layer to adapt
        rank: Low-rank dimension
        alpha: Scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        # LoRA decomposition: W + (alpha/rank) * B @ A
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize A with Kaiming, B with zeros (so LoRA starts at zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original output + LoRA adaptation."""
        # Original path (frozen)
        original_out = self.original(x)

        # LoRA path (trainable)
        lora_out = self.lora_A(x)
        lora_out = self.lora_dropout(lora_out)
        lora_out = self.lora_B(lora_out) * self.scaling

        return original_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into original linear layer (for inference)."""
        # W_merged = W_original + (alpha/rank) * B @ A
        merged_weight = self.original.weight.data + (
            self.lora_B.weight @ self.lora_A.weight
        ) * self.scaling

        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = merged_weight
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data.clone()

        return merged

    def __repr__(self) -> str:
        return (
            f"LoRALinear(in={self.original.in_features}, "
            f"out={self.original.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.4f})"
        )


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """
    Apply LoRA adapters to a model.

    Replaces target linear layers with LoRA-wrapped versions.
    Only the LoRA parameters (A, B) are trainable.

    Args:
        model: OpenMythos model
        config: LoRA configuration

    Returns:
        Model with LoRA adapters applied (modifies in-place)
    """
    target_modules = set(config.target_modules)
    adapted_count = 0

    for name, module in model.named_modules():
        # Check if this module should be adapted
        should_adapt = any(target in name for target in target_modules)

        if not should_adapt:
            continue

        if not isinstance(module, nn.Linear):
            continue

        # Find parent module and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name

        # Replace with LoRA version
        lora_layer = LoRALinear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, attr_name, lora_layer)
        adapted_count += 1

    logger.info(
        f"Applied LoRA to {adapted_count} layers "
        f"(rank={config.rank}, alpha={config.alpha})"
    )

    # Handle modules_to_save (fully trainable)
    if config.modules_to_save:
        for name, module in model.named_modules():
            if any(target in name for target in config.modules_to_save):
                for param in module.parameters():
                    param.requires_grad = True
                logger.info(f"Marked {name} as fully trainable")

    return model


def get_lora_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Get all trainable LoRA parameters."""
    lora_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params[name] = param
    return lora_params


def get_lora_param_stats(model: nn.Module) -> Dict[str, int]:
    """Get statistics about LoRA parameters vs total parameters."""
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_ratio": trainable_params / max(total_params, 1) * 100,
    }


def print_lora_summary(model: nn.Module):
    """Print a summary of LoRA adaptation."""
    stats = get_lora_param_stats(model)

    # Count LoRA layers
    lora_layers = sum(1 for m in model.modules() if isinstance(m, LoRALinear))

    print("=" * 50)
    print("LoRA Adaptation Summary")
    print("=" * 50)
    print(f"LoRA layers:           {lora_layers}")
    print(f"Total parameters:      {stats['total_params']:,}")
    print(f"Trainable parameters:  {stats['trainable_params']:,}")
    print(f"Frozen parameters:     {stats['frozen_params']:,}")
    print(f"Trainable ratio:       {stats['trainable_ratio']:.2f}%")
    print("=" * 50)


def save_lora_adapter(
    model: nn.Module,
    path: str,
    config: Optional[LoRAConfig] = None,
):
    """
    Save only the LoRA adapter weights (not the full model).

    This creates a small file (~1-100MB) instead of the full model (~GBs).
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state_dict[name] = param.data.cpu()

    # Also save LoRA config if provided
    metadata = {}
    if config:
        metadata["rank"] = config.rank
        metadata["alpha"] = config.alpha
        metadata["target_modules"] = config.target_modules

    save_dict = {
        "lora_weights": lora_state_dict,
        "metadata": metadata,
    }

    torch.save(save_dict, path)
    logger.info(
        f"Saved LoRA adapter to {path} "
        f"({sum(p.numel() for p in lora_state_dict.values()):,} parameters)"
    )


def load_lora_adapter(
    model: nn.Module,
    path: str,
):
    """
    Load LoRA adapter weights into a model.

    The model must already have LoRA applied with matching configuration.
    """
    save_dict = torch.load(path, map_location="cpu", weights_only=True)
    lora_state_dict = save_dict["lora_weights"]

    # Load weights
    model_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_dict:
            model_dict[name] = param
        else:
            logger.warning(f"LoRA parameter {name} not found in model")

    model.load_state_dict(model_dict, strict=False)
    logger.info(f"Loaded LoRA adapter from {path}")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA adapters into the base model weights.

    After merging, the model can be used for inference without LoRA overhead.
    The merged model has the same architecture as the original.
    """
    merged_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            merged_linear = module.merge_weights()

            # Find parent and replace
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr_name = name

            setattr(parent, attr_name, merged_linear)
            merged_count += 1

    logger.info(f"Merged {merged_count} LoRA layers")
    return model
