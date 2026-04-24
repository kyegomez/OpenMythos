"""
Phase 3: 内容类型路由器
========================

循环内每轮的内容类型路由器。

分析当前 hidden state 对应的主要模态，
决定本轮循环应注入哪类检索上下文。
"""

from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Modality Types
# ============================================================================


class ModalityType(str, Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    MIXED = "mixed"


# ============================================================================
# Content Type Router
# ============================================================================


class ContentTypeRouter(nn.Module):
    """
    循环内每轮的内容类型路由器。

    分析当前 hidden state 对应的主要模态，
    决定本轮循环应注入哪类检索上下文。

    Architecture:
        hidden_state (B, T, D)
            → mean pooling (B, D)
            → MLP classifier
            → 4-class softmax (TEXT, IMAGE, TABLE, EQUATION)

    The routing decision influences:
        1. Which retrieval context to inject
        2. How to interpret the retrieval results
        3. Weighting of multi-modal retrieval
    """

    NUM_MODALITIES = 4
    MODALITY_NAMES = ["TEXT", "IMAGE", "TABLE", "EQUATION"]

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Hidden dimension
            hidden_dim: Router hidden dimension (default: dim // 2)
            dropout: Dropout probability
        """
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim // 2

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.NUM_MODALITIES),
        )

        # Modality embeddings for conditioning
        self.modality_embeddings = nn.Embedding(
            self.NUM_MODALITIES,
            dim,
        )

    def forward(
        self,
        h: torch.Tensor,
        return_probs: bool = False,
    ) -> dict:
        """
        分析 hidden state 的模态分布。

        Args:
            h: Hidden state (B, T, dim)
            return_probs: 是否返回详细概率

        Returns:
            dict with:
                - modality: 预测的模态类型 (str or list[str])
                - probs: (B, NUM_MODALITIES) 各模态概率
                - routing_weights: (B, NUM_MODALITIES) 用于加权融合
                - modality_embedding: (B, dim) 模态条件嵌入
        """
        B, T, D = h.shape

        # Mean pooling over sequence
        pooled = h.mean(dim=1)  # (B, D)

        # Classifier
        logits = self.classifier(pooled)  # (B, NUM_MODALITIES)
        probs = F.softmax(logits, dim=-1)  # (B, NUM_MODALITIES)

        # Greedy modality selection
        modality_idx = probs.argmax(dim=-1)  # (B,)
        modalities = [self.MODALITY_NAMES[i] for i in modality_idx]

        # Get modality embeddings for conditioning
        modality_embedding = self.modality_embeddings(modality_idx)  # (B, dim)

        result = {
            "modality": modalities[0] if B == 1 else modalities,
            "probs": probs,
            "routing_weights": probs,  # Can be used for weighted combination
            "modality_embedding": modality_embedding,
        }

        return result

    def get_modality_context(
        self,
        modality: str,
        batch_size: int = 1,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        获取指定模态的条件嵌入。

        Args:
            modality: 模态名称
            batch_size: 批次大小
            device: 设备

        Returns:
            (batch_size, dim) 模态嵌入
        """
        idx = self.MODALITY_NAMES.index(modality.upper())
        embedding = self.modality_embeddings.weight[idx]  # (dim,)
        return embedding.unsqueeze(0).expand(batch_size, -1)  # (B, dim)


class ModalityAwareFusion(nn.Module):
    """
    模态感知融合模块。

    根据内容模态自适应地融合不同模态的信息。
    用于将检索到的多模态上下文与当前 hidden state 融合。
    """

    def __init__(
        self,
        dim: int,
        num_modalities: int = 4,
    ):
        """
        Args:
            dim: Hidden dimension
            num_modalities: 模态数量
        """
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities

        # Modality-specific projection
        self.modality_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            for name in ["TEXT", "IMAGE", "TABLE", "EQUATION"]
        })

        # Cross-modality attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,           # (B, T, D) current hidden state
        retrieval_context: dict,    # Dict of {modality: tensor}
        current_modality: str,
    ) -> torch.Tensor:
        """
        融合检索上下文。

        Args:
            h: 当前 hidden state (B, T, D)
            retrieval_context: 检索到的各模态上下文
            current_modality: 当前主要模态

        Returns:
            融合后的 hidden state (B, T, D)
        """
        B, T, D = h.shape

        # Project current modality-specific
        mod_proj = self.modality_proj.get(
            current_modality.upper(),
            self.modality_proj["TEXT"],
        )
        h_proj = mod_proj(h)  # (B, T, D)

        # Collect available retrieval modalities
        retrieved_mods = []
        retrieved_tensors = []
        for mod_name, mod_tensor in retrieval_context.items():
            if mod_tensor is not None:
                retrieved_mods.append(mod_name)
                # Ensure same sequence length
                if mod_tensor.shape[1] != T:
                    mod_tensor = F.interpolate(
                        mod_tensor.permute(0, 2, 1),  # (B, D, T')
                        size=T,
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1)  # (B, T, D)
                retrieved_tensors.append(mod_proj(mod_tensor))

        if not retrieved_tensors:
            return h

        # Stack and cross-attend
        retrieved_stack = torch.stack(retrieved_tensors, dim=2)  # (B, T, num_mods, D)
        retrieved_flat = retrieved_stack.view(B * T, len(retrieved_mods), D)  # (B*T, num_mods, D)
        h_flat = h_proj.view(B * T, 1, D)  # (B*T, 1, D)

        # Cross attention: h attends to retrieval modalities
        attn_out, _ = self.cross_attn(h_flat, retrieved_flat, retrieved_flat)
        attn_out = attn_out.view(B, T, D)

        # Gating
        gate_input = torch.cat([h_proj, attn_out], dim=-1)
        gate = self.gate(gate_input)

        # Gated fusion
        fused = gate * h_proj + (1 - gate) * attn_out

        return fused
