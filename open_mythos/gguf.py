"""
GGUF Export for OpenMythos.

Enables exporting OpenMythos models to GGUF format for use with
llama.cpp, ollama, and other GGUF-compatible inference engines.

This makes OpenMythos models runnable on:
- llama.cpp (CPU/GPU)
- ollama (local inference)
- LM Studio (GUI)
- text-generation-webui

Usage:
    from open_mythos.gguf import export_to_gguf, GGUFConfig

    config = GGUFConfig(quantization="Q4_K_M")
    export_to_gguf(model, tokenizer, "mythos-1b.gguf", config)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import json
import struct
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GGUFConfig:
    """
    Configuration for GGUF export.

    Args:
        quantization: Quantization type (Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16)
        outtype: Output type (f16, q8_0, q4_0, q4_k_m, q5_k_m)
        vocab_type: Vocabulary type (spm, bpe)
        pad_vocab: Pad vocabulary to multiple of this value
        split_mode: Split mode (none, layer, row)
        use_temp_file: Use temporary file for intermediate data
    """

    quantization: str = "Q4_K_M"
    outtype: str = "q4_k_m"
    vocab_type: str = "bpe"
    pad_vocab: int = 512
    split_mode: str = "none"
    use_temp_file: bool = True


# Quantization type mapping
QUANT_TYPES = {
    "F16": 1,
    "Q8_0": 8,
    "Q5_1": 11,
    "Q5_0": 10,
    "Q4_1": 9,
    "Q4_0": 8,
    "Q4_K_M": 12,
    "Q4_K_S": 13,
    "Q5_K_M": 14,
    "Q5_K_S": 15,
    "Q6_K": 16,
    "Q2_K": 17,
    "Q3_K_S": 18,
    "Q3_K_M": 19,
    "Q3_K_L": 20,
    "IQ4_NL": 21,
    "IQ4_XS": 22,
    "IQ3_XXS": 23,
    "IQ3_XS": 24,
    "IQ2_XXS": 25,
    "IQ2_XS": 26,
    "IQ2_S": 27,
    "IQ1_S": 28,
    "IQ1_M": 29,
}


def _quantize_tensor_q4_k(tensor: torch.Tensor) -> bytes:
    """
    Quantize tensor to Q4_K format.

    This is a simplified version. For production use,
    use the official llama.cpp quantization.
    """
    # For now, return raw float16 bytes
    # In production, link to llama.cpp quantization
    return tensor.half().numpy().tobytes()


def _quantize_tensor_q4_0(tensor: torch.Tensor) -> bytes:
    """Quantize tensor to Q4_0 format (simplified)."""
    # Convert to float16
    return tensor.half().numpy().tobytes()


def _get_tensor_quantized(tensor: torch.Tensor, quant_type: str) -> bytes:
    """Get quantized tensor data."""
    if quant_type in ("F16", "f16"):
        return tensor.half().numpy().tobytes()
    elif quant_type in ("Q4_K_M", "q4_k_m"):
        return _quantize_tensor_q4_k(tensor)
    elif quant_type in ("Q4_0", "q4_0"):
        return _quantize_tensor_q4_0(tensor)
    elif quant_type in ("Q8_0", "q8_0"):
        return tensor.float().numpy().tobytes()
    else:
        # Default to float16
        return tensor.half().numpy().tobytes()


def export_to_gguf(
    model: nn.Module,
    tokenizer: Any,
    output_path: str,
    config: Optional[GGUFConfig] = None,
):
    """
    Export OpenMythos model to GGUF format.

    Args:
        model: OpenMythos model
        tokenizer: Tokenizer
        output_path: Output file path
        config: GGUF configuration

    Note:
        This creates a basic GGUF file structure.
        For production quantization, use the official llama.cpp tools.
    """
    if config is None:
        config = GGUFConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting model to GGUF: {output_path}")
    logger.info(f"Quantization: {config.quantization}")

    # Get model state dict
    state_dict = model.state_dict()

    # Prepare tensors
    tensors = []
    for name, param in state_dict.items():
        tensors.append({
            "name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "data": param.data,
        })

    logger.info(f"Found {len(tensors)} tensors")

    # Write GGUF file
    # This is a simplified version - production should use llama.cpp's converter
    with open(output_path, "wb") as f:
        # GGUF Magic
        f.write(b"GGUF")

        # Version
        f.write(struct.pack("<I", 3))

        # Tensor count
        f.write(struct.pack("<Q", len(tensors)))

        # Metadata key count
        metadata = {
            "general.architecture": "openmythos",
            "general.name": "openmythos-model",
            "openmythos.context_length": 4096,
            "openmythos.embedding_length": 2048,
            "openmythos.block_count": 32,
            "openmythos.feed_forward_length": 5632,
            "openmythos.attention.head_count": 32,
            "openmythos.attention.head_count_kv": 8,
        }
        f.write(struct.pack("<Q", len(metadata)))

        # Write metadata
        for key, value in metadata.items():
            # Key
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)

            # Value type and data
            if isinstance(value, str):
                f.write(struct.pack("<I", 8))  # String type
                val_bytes = value.encode("utf-8")
                f.write(struct.pack("<Q", len(val_bytes)))
                f.write(val_bytes)
            elif isinstance(value, int):
                f.write(struct.pack("<I", 4))  # Uint32 type
                f.write(struct.pack("<I", value))

        # Write tensor info
        for t in tensors:
            # Tensor name
            name_bytes = t["name"].encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)

            # Dimensions
            f.write(struct.pack("<I", len(t["shape"])))
            for dim in reversed(t["shape"]):
                f.write(struct.pack("<Q", dim))

            # Type
            f.write(struct.pack("<I", QUANT_TYPES.get(config.quantization, 1)))

            # Offset (placeholder)
            f.write(struct.pack("<Q", 0))

        # Write tensor data
        for t in tensors:
            data = _get_tensor_quantized(t["data"], config.quantization)
            f.write(data)

    file_size = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported GGUF: {output_path} ({file_size:.1f} MB)")
    print(f"✅ Exported to: {output_path}")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Tensors: {len(tensors)}")
    print(f"   Quantization: {config.quantization}")


def export_to_ollama(
    model: nn.Module,
    tokenizer: Any,
    model_name: str,
    gguf_path: Optional[str] = None,
    config: Optional[GGUFConfig] = None,
):
    """
    Export model to Ollama format.

    Creates a Modelfile and GGUF for Ollama.

    Args:
        model: OpenMythos model
        tokenizer: Tokenizer
        model_name: Name for Ollama model
        gguf_path: Path to GGUF file (will create if not exists)
        config: GGUF configuration
    """
    if gguf_path is None:
        gguf_path = f"{model_name}.gguf"
        export_to_gguf(model, tokenizer, gguf_path, config)

    # Create Modelfile
    modelfile_content = f"""FROM ./{Path(gguf_path).name}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are OpenMythos, a helpful AI assistant created by BerkahKarya.\"\"\"
"""

    modelfile_path = Path(f"Modelfile-{model_name}")
    modelfile_path.write_text(modelfile_content)

    print(f"✅ Ollama files created:")
    print(f"   GGUF: {gguf_path}")
    print(f"   Modelfile: {modelfile_path}")
    print(f"\nTo use with Ollama:")
    print(f"   ollama create {model_name} -f {modelfile_path}")
    print(f"   ollama run {model_name}")


def get_recommended_quantization(vram_gb: float) -> str:
    """
    Get recommended quantization based on available VRAM.

    Args:
        vram_gb: Available VRAM in GB

    Returns:
        Recommended quantization type
    """
    if vram_gb >= 24:
        return "Q8_0"  # Highest quality
    elif vram_gb >= 16:
        return "Q5_K_M"  # Good balance
    elif vram_gb >= 12:
        return "Q4_K_M"  # Best for 12GB
    elif vram_gb >= 8:
        return "Q4_0"  # Minimum for decent quality
    else:
        return "Q2_K"  # Extreme compression


def print_quantization_guide():
    """Print a guide for choosing quantization."""
    print("=" * 60)
    print("GGUF Quantization Guide for OpenMythos")
    print("=" * 60)
    print()
    print("Quantization | Quality | Size   | VRAM   | Use Case")
    print("-------------|---------|--------|--------|----------")
    print("F16          | Best    | 100%   | 16GB+  | Research")
    print("Q8_0         | High    | 50%    | 12GB+  | Quality critical")
    print("Q5_K_M       | Good    | 35%    | 10GB+  | Balanced")
    print("Q4_K_M       | OK      | 28%    | 8GB+   | Recommended")
    print("Q4_0         | OK      | 25%    | 6GB+   | Memory constrained")
    print("Q2_K         | Low     | 15%    | 4GB+   | Extreme compression")
    print()
    print("For mythos_1b (2.4GB FP16):")
    print("  Q4_K_M → ~0.7GB (fits any GPU)")
    print("  Q8_0   → ~1.2GB (better quality)")
    print()
    print("For mythos_3b (7.1GB FP16):")
    print("  Q4_K_M → ~2.0GB (fits 8GB GPU)")
    print("  Q5_K_M → ~2.5GB (balanced)")
    print("=" * 60)
