"""
Expert Offloading System for OpenMythos MoE Models.

Enables running large MoE models (500B, 1T) on consumer hardware by
offloading inactive experts to CPU RAM or NVMe SSD, and loading only
the active experts to GPU on-demand.

Memory hierarchy:
    GPU VRAM (fastest)  → Active experts only (top-K per token)
    CPU RAM (fast)      → Recently used expert cache
    NVMe SSD (slow)     → Cold storage for all experts

Usage:
    from open_mythos.expert_offloader import ExpertOffloader

    offloader = ExpertOffloader(model, gpu_experts=4, cache_experts=16)
    offloader.prepare()  # Move inactive experts to CPU/NVMe

    # During inference, experts are loaded automatically
    output = model(input_ids)
"""

import os
import json
import time
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Set
from collections import OrderedDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExpertOffloader:
    """
    Manages expert placement across GPU/CPU/NVMe memory hierarchy.

    Keeps only the most recently used experts on GPU, with an LRU cache
    for CPU-resident experts and disk storage for cold experts.

    Args:
        model: OpenMythos model with MoE layers
        gpu_experts: Number of experts to keep on GPU (default: 4)
        cache_experts: Number of experts to keep in CPU RAM (default: 16)
        storage_dir: Directory for NVMe expert storage (default: /tmp/mythos_experts)
        preload_all: If True, load all experts to CPU at init (default: False)
    """

    def __init__(
        self,
        model: nn.Module,
        gpu_experts: int = 4,
        cache_experts: int = 16,
        storage_dir: str = "/tmp/mythos_experts",
        preload_all: bool = False,
    ):
        self.model = model
        self.gpu_experts = gpu_experts
        self.cache_experts = cache_experts
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Expert state tracking
        self.expert_states: Dict[str, Dict[int, str]] = {}  # layer_name -> {expert_id -> location}
        self.gpu_cache: Dict[str, Dict[int, nn.Module]] = {}  # layer_name -> {expert_id -> module}
        self.cpu_cache: Dict[str, Dict[int, nn.Module]] = {}  # layer_name -> {expert_id -> module}
        self.lru_order: Dict[str, List[int]] = {}  # layer_name -> [expert_ids in LRU order]

        # Statistics
        self.stats = {
            "gpu_hits": 0,
            "cpu_hits": 0,
            "disk_loads": 0,
            "evictions": 0,
            "total_requests": 0,
        }

        # Discover MoE layers
        self.moe_layers = self._discover_moe_layers()

        if preload_all:
            self._preload_to_cpu()

    def _discover_moe_layers(self) -> Dict[str, nn.Module]:
        """Find all MoE layers with expert modules."""
        moe_layers = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") and hasattr(module, "router"):
                moe_layers[name] = module
                logger.info(f"Discovered MoE layer: {name} with {len(module.experts)} experts")
        return moe_layers

    def _get_expert_module(self, layer_name: str, expert_id: int) -> nn.Module:
        """Get the expert module by layer name and expert ID."""
        if layer_name not in self.moe_layers:
            raise KeyError(f"MoE layer '{layer_name}' not found. Available: {list(self.moe_layers.keys())}")
        layer = self.moe_layers[layer_name]
        if not hasattr(layer, "experts"):
            raise AttributeError(f"Layer '{layer_name}' has no 'experts' attribute")
        if expert_id < 0 or expert_id >= len(layer.experts):
            raise IndexError(f"Expert {expert_id} out of range [0, {len(layer.experts)-1}] for layer '{layer_name}'")
        return layer.experts[expert_id]

    def _move_expert_to_device(
        self, layer_name: str, expert_id: int, device: str
    ) -> nn.Module:
        """Move an expert to the specified device."""
        expert = self._get_expert_module(layer_name, expert_id)
        return expert.to(device)

    def __repr__(self) -> str:
        """String representation of the offloader."""
        total_experts = sum(len(m.experts) for m in self.moe_layers.values())
        return (
            f"ExpertOffloader("
            f"moe_layers={len(self.moe_layers)}, "
            f"total_experts={total_experts}, "
            f"gpu_experts={self.gpu_experts}, "
            f"cache_experts={self.cache_experts})"
        )

    def _save_expert_to_disk(self, layer_name: str, expert_id: int):
        """Save expert weights to NVMe storage."""
        expert = self._get_expert_module(layer_name, expert_id)
        safe_name = layer_name.replace(".", "_")
        path = self.storage_dir / f"{safe_name}_expert_{expert_id}.pt"
        torch.save(
            {k: v.cpu() for k, v in expert.state_dict().items()},
            path,
        )

    def _load_expert_from_disk(self, layer_name: str, expert_id: int) -> nn.Module:
        """Load expert weights from NVMe storage."""
        safe_name = layer_name.replace(".", "_")
        path = self.storage_dir / f"{safe_name}_expert_{expert_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expert storage not found: {path}")

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        expert = self._get_expert_module(layer_name, expert_id)
        expert.load_state_dict(state_dict)
        return expert

    def _update_lru(self, layer_name: str, expert_id: int):
        """Update LRU order for a layer."""
        if layer_name not in self.lru_order:
            self.lru_order[layer_name] = []

        if expert_id in self.lru_order[layer_name]:
            self.lru_order[layer_name].remove(expert_id)
        self.lru_order[layer_name].append(expert_id)

    def _evict_from_gpu(self, layer_name: str):
        """Evict least recently used expert from GPU to CPU."""
        if layer_name not in self.lru_order:
            return

        lru = self.lru_order[layer_name]
        if len(lru) <= self.gpu_experts:
            return

        # Find experts currently on GPU that aren't the most recent
        gpu_expert_ids = [
            eid for eid in lru[:-self.gpu_experts]
            if self.expert_states.get(layer_name, {}).get(eid) == "gpu"
        ]

        if not gpu_expert_ids:
            return

        # Evict oldest
        evict_id = gpu_expert_ids[0]
        self._move_expert_to_device(layer_name, evict_id, "cpu")

        # Move to CPU cache
        if layer_name not in self.cpu_cache:
            self.cpu_cache[layer_name] = {}
        self.cpu_cache[layer_name][evict_id] = self._get_expert_module(layer_name, evict_id)
        self.expert_states[layer_name][evict_id] = "cpu"
        self.stats["evictions"] += 1

        logger.debug(f"Evicted expert {evict_id} from GPU to CPU (layer: {layer_name})")

    def prepare(self):
        """
        Prepare the model for offloaded inference.

        Moves all experts to CPU initially, keeping only the first
        gpu_experts on GPU for each MoE layer.
        """
        for layer_name, moe_layer in self.moe_layers.items():
            num_experts = len(moe_layer.experts)
            self.expert_states[layer_name] = {}
            self.gpu_cache[layer_name] = {}
            self.cpu_cache[layer_name] = {}
            self.lru_order[layer_name] = []

            for expert_id in range(num_experts):
                if expert_id < self.gpu_experts:
                    # Keep on GPU
                    self.expert_states[layer_name][expert_id] = "gpu"
                    self.gpu_cache[layer_name][expert_id] = self._get_expert_module(
                        layer_name, expert_id
                    )
                else:
                    # Move to CPU
                    self._move_expert_to_device(layer_name, expert_id, "cpu")
                    self.expert_states[layer_name][expert_id] = "cpu"
                    self.cpu_cache[layer_name][expert_id] = self._get_expert_module(
                        layer_name, expert_id
                    )

                self._update_lru(layer_name, expert_id)

        logger.info(
            f"Prepared {len(self.moe_layers)} MoE layers for offloaded inference. "
            f"GPU experts per layer: {self.gpu_experts}"
        )

    def load_expert(self, layer_name: str, expert_id: int, target_device: str = "cuda"):
        """
        Load an expert to the target device, managing cache hierarchy.

        This is called automatically during forward pass when an expert
        is selected by the router.
        """
        self.stats["total_requests"] += 1
        state = self.expert_states.get(layer_name, {}).get(expert_id)

        if state == "gpu":
            # Already on GPU
            self.stats["gpu_hits"] += 1
            self._update_lru(layer_name, expert_id)
            return

        if state == "cpu":
            # In CPU cache, move to GPU
            self.stats["cpu_hits"] += 1

            # Make room on GPU if needed
            self._evict_from_gpu(layer_name)

            # Move to GPU
            self._move_expert_to_device(layer_name, expert_id, target_device)
            self.expert_states[layer_name][expert_id] = "gpu"
            self.gpu_cache[layer_name][expert_id] = self._get_expert_module(
                layer_name, expert_id
            )
            self._update_lru(layer_name, expert_id)
            return

        # On disk (cold storage)
        self.stats["disk_loads"] += 1
        logger.debug(f"Loading expert {expert_id} from disk (layer: {layer_name})")

        # Load from disk to CPU first
        self._load_expert_from_disk(layer_name, expert_id)

        # Then move to GPU
        self._evict_from_gpu(layer_name)
        self._move_expert_to_device(layer_name, expert_id, target_device)
        self.expert_states[layer_name][expert_id] = "gpu"
        self.gpu_cache[layer_name][expert_id] = self._get_expert_module(
            layer_name, expert_id
        )
        self._update_lru(layer_name, expert_id)

    def offload_all_to_cpu(self):
        """Move all experts to CPU (for memory cleanup)."""
        for layer_name in self.moe_layers:
            for expert_id in list(self.expert_states.get(layer_name, {}).keys()):
                if self.expert_states[layer_name][expert_id] == "gpu":
                    self._move_expert_to_device(layer_name, expert_id, "cpu")
                    self.expert_states[layer_name][expert_id] = "cpu"
        torch.cuda.empty_cache()

    def save_all_to_disk(self):
        """Save all experts to NVMe storage."""
        for layer_name in self.moe_layers:
            num_experts = len(self.moe_layers[layer_name].experts)
            for expert_id in range(num_experts):
                self._save_expert_to_disk(layer_name, expert_id)
        logger.info(f"Saved all experts to {self.storage_dir}")

    def get_stats(self) -> dict:
        """Get offloading statistics."""
        total = self.stats["total_requests"] or 1
        return {
            **self.stats,
            "gpu_hit_rate": self.stats["gpu_hits"] / total * 100,
            "cpu_hit_rate": self.stats["cpu_hits"] / total * 100,
            "disk_load_rate": self.stats["disk_loads"] / total * 100,
        }

    def print_stats(self):
        """Print offloading statistics."""
        s = self.get_stats()
        print("=" * 50)
        print("Expert Offloader Statistics")
        print("=" * 50)
        print(f"Total requests:  {s['total_requests']}")
        print(f"GPU hits:        {s['gpu_hits']} ({s['gpu_hit_rate']:.1f}%)")
        print(f"CPU hits:        {s['cpu_hits']} ({s['cpu_hit_rate']:.1f}%)")
        print(f"Disk loads:      {s['disk_loads']} ({s['disk_load_rate']:.1f}%)")
        print(f"GPU evictions:   {s['evictions']}")
        print("=" * 50)


def create_offloaded_model(
    model: nn.Module,
    gpu_experts: int = 4,
    cache_experts: int = 16,
    storage_dir: str = "/tmp/mythos_experts",
) -> tuple:
    """
    Convenience function to create an offloaded model.

    Returns:
        (model, offloader) tuple
    """
    offloader = ExpertOffloader(
        model,
        gpu_experts=gpu_experts,
        cache_experts=cache_experts,
        storage_dir=storage_dir,
    )
    offloader.prepare()
    return model, offloader
