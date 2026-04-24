"""
Phase 3: RAG 增强循环块
========================

检索增强的循环推理块。

每轮循环:
1. 路由网络决定当前内容类型
2. 根据内容类型检索多模态 KG
3. 将检索上下文注入 hidden state
4. 标准的 Transformer + LTI + ACT 更新
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.main_p0 import (
    TransformerBlock,
    LTIInjection,
    LoRAAdapter,
    ACTHalting,
    RMSNorm,
    loop_index_embedding,
)


# ============================================================================
# RAG Enhanced Recurrent Block
# ============================================================================


class RAGEnhancedRecurrentBlock(nn.Module):
    """
    检索增强的循环推理块。

    每轮循环包含:
    1. 内容类型路由 (ContentTypeRouter)
    2. 多模态 KG 检索
    3. 检索上下文注入
    4. Transformer 更新
    5. LTI 注入
    6. ACT halting

    关键: 检索上下文丰富了循环推理的外部知识
    """

    def __init__(
        self,
        cfg: Any,  # MythosConfig
        kg: Any,   # MultimodalKnowledgeGraph
        content_router: Any,  # ContentTypeRouter
        retrieval_top_k: int = 5,
        retrieval_depth: int = 2,
        use_moe: bool = True,
    ):
        """
        Args:
            cfg: MythosConfig
            kg: MultimodalKnowledgeGraph 实例
            content_router: ContentTypeRouter 实例
            retrieval_top_k: 每轮检索返回数量
            retrieval_depth: 图遍历深度
            use_moe: 是否在 TransformerBlock 中使用 MoE
        """
        super().__init__()
        self.cfg = cfg
        self.kg = kg
        self.router = content_router
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_depth = retrieval_depth
        self.use_moe = use_moe

        # 标准组件
        self.transformer_block = TransformerBlock(cfg, use_moe=use_moe)
        self.lti_injection = LTIInjection(cfg.dim)
        self.lora_adapter = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.act_halting = ACTHalting(cfg.dim)
        self.norm = RMSNorm(cfg.dim)

        # 检索上下文融合
        self.retrieval_proj = nn.Linear(cfg.dim, cfg.dim)
        self.retrieval_norm = RMSNorm(cfg.dim)

        # 检索缓存 (避免重复检索)
        self._retrieval_cache = {}
        self._cache_ttl = 100  # Cache TTL in steps

        self.loop_dim = cfg.dim // 8

    def forward(
        self,
        h: torch.Tensor,           # 当前 hidden state (B, T, dim)
        e: torch.Tensor,            # 编码输入 (B, T, dim)
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        loop_idx: int = 0,
        enable_retrieval: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        RAG 增强的循环推理。

        Args:
            h: hidden state (B, T, dim)
            e: 编码输入 (B, T, dim)
            freqs_cis: RoPE 频率
            mask: 因果 mask
            kv_cache: KV 缓存
            loop_idx: 当前循环步索引
            enable_retrieval: 是否启用检索

        Returns:
            (h_out, info)
            - h_out: 更新后的 hidden state
            - info: dict with retrieval_context, modality, halting_prob
        """
        B, T, D = h.shape

        # ===== Step 1: 内容类型路由 =====
        routing_info = self.router(h)
        current_modality = routing_info["modality"]
        routing_weights = routing_info["routing_weights"]  # (B, num_modalities)

        # ===== Step 2: 多模态检索 =====
        retrieval_context = None
        retrieved_entities = []

        if enable_retrieval and self.kg is not None:
            retrieval_context, retrieved_entities = self._retrieve(
                h, current_modality, loop_idx
            )

        # ===== Step 3: 融合检索上下文 =====
        if retrieval_context is not None:
            # 将检索上下文与 hidden state 融合
            h_with_context = self._fuse_retrieval(h, retrieval_context)
        else:
            h_with_context = h

        # ===== Step 4: Loop index embedding =====
        h_loop = loop_index_embedding(h_with_context, loop_idx, self.loop_dim)
        combined = self.norm(h_loop + e)

        # ===== Step 5: Transformer 更新 =====
        cache_key = f"rag_loop_{loop_idx}"
        trans_out = self.transformer_block(
            combined, freqs_cis, mask, kv_cache, cache_key
        )

        # LoRA 深度适配
        trans_out = trans_out + self.lora_adapter(trans_out, loop_idx)

        # ===== Step 6: LTI 注入 =====
        h_new = self.lti_injection(h, e, trans_out)

        # ===== Step 7: ACT halting =====
        p = self.act_halting(h_new)

        info = {
            "modality": current_modality,
            "routing_probs": routing_weights,
            "retrieved_entities": retrieved_entities,
            "halting_prob": p,
            "retrieval_context": retrieval_context,
        }

        return h_new, info

    def _retrieve(
        self,
        h: torch.Tensor,
        modality: str,
        loop_idx: int,
    ) -> tuple[Optional[torch.Tensor], list[dict]]:
        """
        执行多模态检索。

        Returns:
            (retrieval_context, retrieved_entities)
        """
        B, T, D = h.shape

        # 生成查询 (使用 hidden state 的 mean 作为查询)
        query_emb = h.mean(dim=1)  # (B, D)

        # 构建缓存 key
        cache_key = self._get_cache_key(query_emb, modality, loop_idx)

        # 检查缓存
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        # 执行检索
        try:
            # 将 query_emb 转换为文本查询
            # 简化: 使用向量直接检索
            # 生产版本: 可以训练一个解码器将 hidden state 转换为文本

            # 获取查询向量
            query_np = query_emb[0].detach().cpu().numpy()

            # 检索
            results = self.kg.retrieve(
                query="",  # 使用空查询，仅依赖向量
                query_type=modality.lower() if isinstance(modality, str) else "text",
                top_k=self.retrieval_top_k,
                depth=self.retrieval_depth,
            )

            # 解析检索结果
            retrieved_entities = [
                {
                    "id": r.get("id", ""),
                    "type": r.get("type", "text"),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ]

            # 编码检索上下文
            if retrieved_entities:
                retrieval_context = self._encode_retrieval(retrieved_entities, B, T)
            else:
                retrieval_context = None

            # 缓存
            if len(self._retrieval_cache) < self._cache_ttl:
                self._retrieval_cache[cache_key] = (retrieval_context, retrieved_entities)

            return retrieval_context, retrieved_entities

        except Exception as e:
            # 检索失败，返回 None
            return None, []

    def _get_cache_key(
        self,
        query_emb: torch.Tensor,
        modality: str,
        loop_idx: int,
    ) -> str:
        """生成缓存 key"""
        # 使用 query_emb 的 hash 和 modality, loop_idx
        emb_hash = hash(query_emb.detach().cpu().numpy().tobytes()[:100])
        return f"{emb_hash}_{modality}_{loop_idx}"

    def _encode_retrieval(
        self,
        retrieved: list[dict],
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        将检索结果编码为上下文向量。

        Args:
            retrieved: 检索结果列表
            batch_size: 批次大小
            seq_len: 序列长度

        Returns:
            (batch_size, seq_len, dim) 检索上下文张量
        """
        device = next(self.parameters()).device

        if not retrieved:
            return torch.zeros(batch_size, seq_len, self.cfg.dim, device=device)

        # 简单策略: 使用检索结果的加权平均
        # 生产版本: 可以使用 cross-attention 机制

        # 计算加权嵌入
        total_score = sum(r.get("score", 0.0) for r in retrieved) + 1e-8

        context_emb = torch.zeros(self.cfg.dim, device=device)
        for r in retrieved:
            score = r.get("score", 0.0) / total_score
            # 简化: 使用随机初始化或平均
            # 生产版本: 需要从 KG 获取实际嵌入
            context_emb = context_emb + score * torch.randn(self.cfg.dim, device=device)

        # 扩展到序列维度
        context = context_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        context = context.expand(batch_size, seq_len, -1)  # (B, T, dim)

        return context

    def _fuse_retrieval(
        self,
        h: torch.Tensor,
        retrieval_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        融合检索上下文与 hidden state。

        使用门控机制:
            h_fused = gate * h + (1 - gate) * retrieval_context
        """
        # 投影检索上下文
        retrieval_proj = self.retrieval_proj(retrieval_context)
        retrieval_proj = self.retrieval_norm(retrieval_proj)

        # 门控
        gate_input = torch.cat([h, retrieval_proj], dim=-1)
        gate = torch.sigmoid(gate_input)

        # 融合
        fused = gate * h + (1 - gate) * retrieval_proj

        return fused

    def clear_cache(self):
        """清空检索缓存"""
        self._retrieval_cache.clear()


# ============================================================================
# Simplified RAG Block (for when KG is not available)
# ============================================================================


class SimplifiedRAGBlock(nn.Module):
    """
    简化的 RAG 循环块 (当 KG 不可用时使用)。

    不依赖外部 KG，仅使用可学习的检索嵌入进行上下文注入。
    """

    def __init__(
        self,
        cfg: Any,
        num_modalities: int = 4,
        retrieval_dim: int = 512,
    ):
        """
        Args:
            cfg: MythosConfig
            num_modalities: 模态数量
            retrieval_dim: 检索嵌入维度
        """
        super().__init__()
        self.cfg = cfg
        self.num_modalities = num_modalities

        # 标准组件
        self.transformer_block = TransformerBlock(cfg, use_moe=True)
        self.lti_injection = LTIInjection(cfg.dim)
        self.lora_adapter = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.act_halting = ACTHalting(cfg.dim)
        self.norm = RMSNorm(cfg.dim)

        # 可学习的检索嵌入 (模拟检索上下文)
        self.retrieval_embeddings = nn.Parameter(
            torch.randn(num_modalities, retrieval_dim) * 0.02
        )

        # 检索上下文融合
        self.retrieval_proj = nn.Linear(retrieval_dim, cfg.dim)
        self.gate = nn.Sequential(
            nn.Linear(cfg.dim * 2, cfg.dim),
            nn.Sigmoid(),
        )

        self.loop_dim = cfg.dim // 8

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        loop_idx: int = 0,
        modality_idx: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """
        简化的 RAG 循环。

        Args:
            h: hidden state (B, T, dim)
            e: 编码输入 (B, T, dim)
            freqs_cis: RoPE 频率
            mask: 因果 mask
            kv_cache: KV 缓存
            loop_idx: 当前循环步索引
            modality_idx: 模态索引 (0=TEXT, 1=IMAGE, 2=TABLE, 3=EQUATION)

        Returns:
            (h_out, info)
        """
        B, T, D = h.shape

        # 获取检索嵌入
        retrieval_emb = self.retrieval_embeddings[modality_idx % self.num_modalities]
        retrieval_emb = retrieval_emb.unsqueeze(0).unsqueeze(1)  # (1, 1, retrieval_dim)
        retrieval_emb = retrieval_emb.expand(B, T, -1)  # (B, T, retrieval_dim)

        # 投影到 dim
        retrieval_proj = self.retrieval_proj(retrieval_emb)  # (B, T, dim)

        # 门控融合
        gate_input = torch.cat([h, retrieval_proj], dim=-1)
        gate = self.gate(gate_input)
        h_with_retrieval = gate * h + (1 - gate) * retrieval_proj

        # Loop index embedding
        h_loop = loop_index_embedding(h_with_retrieval, loop_idx, self.loop_dim)
        combined = self.norm(h_loop + e)

        # Transformer
        cache_key = f"simple_rag_loop_{loop_idx}"
        trans_out = self.transformer_block(
            combined, freqs_cis, mask, kv_cache, cache_key
        )

        # LoRA
        trans_out = trans_out + self.lora_adapter(trans_out, loop_idx)

        # LTI
        h_new = self.lti_injection(h, e, trans_out)

        # ACT
        p = self.act_halting(h_new)

        info = {
            "modality_idx": modality_idx,
            "halting_prob": p,
        }

        return h_new, info
