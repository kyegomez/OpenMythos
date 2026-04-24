"""
OpenMythosRAG: 检索增强的多模态循环推理
========================================

统一入口类，整合所有组件:
- 多模态文档解析
- 知识图谱存储
- 混合检索
- 循环推理 (RAG增强)
- 联合深度+模态调度
"""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False

from open_mythos.rag.kg_storage import (
    Entity, EntityType, Edge, EdgeType,
    InMemoryVectorStore, InMemoryGraphStore,
)
try:
    from open_mythos.rag.knowledge_graph import MultimodalKnowledgeGraph
except ImportError:
    MultimodalKnowledgeGraph = None
try:
    from open_mythos.rag.hybrid_retrieval import HybridRetrieval, RetrievalResult
except ImportError:
    HybridRetrieval = None
    RetrievalResult = None
try:
    from open_mythos.rag.content_router import ContentTypeRouter, ModalityAwareFusion
except ImportError:
    ContentTypeRouter = None
    ModalityAwareFusion = None
try:
    from open_mythos.rag.rag_recurrent_block import RAGEnhancedRecurrentBlock, SimplifiedRAGBlock
except ImportError:
    RAGEnhancedRecurrentBlock = None
    SimplifiedRAGBlock = None
try:
    from open_mythos.rag.joint_selector import JointDepthModalitySelector, AdaptiveLoopController
except ImportError:
    JointDepthModalitySelector = None
    AdaptiveLoopController = None

# Retrieval components
try:
    from open_mythos.rag.retrieval import (
        BGELargeEmbedder,
        E5Embedder,
        CrossEncoderReranker,
        RerankerPool,
        RerankResult,
    )
    HAS_RETRIEVAL = True
except ImportError:
    HAS_RETRIEVAL = False


# ============================================================================
# Config
# ============================================================================


@dataclass
class RAGConfig:
    """RAG 模块配置"""
    # 知识图谱
    kg_storage_type: str = "memory"  # memory | neo4j | postgresql
    kg_vector_dim: int = 1536
    kg_max_entities: int = 100000

    # 检索
    retrieval_top_k: int = 10
    retrieval_depth: int = 3
    retrieval_expansion_factor: float = 2.0
    use_cross_modal: bool = True

    # 循环推理
    max_loop_iters: int = 16
    loop_dim: int = 256
    enable_rag: bool = True

    # 路由
    routing_hidden_dim: Optional[int] = None

    # 设备
    device: str = "cuda" if _HAS_TORCH else "cpu"


# Conditionally define OpenMythosRAG only when torch is available
if _HAS_TORCH and nn is not None:
    class OpenMythosRAG(nn.Module):
	    """
	    OpenMythosRAG: 检索增强的多模态循环推理模型。

	    Usage:
	        # 初始化
	        model = OpenMythosRAG(
	            base_model=base_mythos,
	            rag_config=RAGConfig(),
	        )

	        # 索引文档
	        model.index_document("doc.pdf", parser=MinerUPreviewParser())

	        # 推理
	        output = model.generate(
	            query="What is the main topic?",
	            max_length=512,
	        )

	        # 保存/加载
	        model.save("rag_model.pt")
	        model = OpenMythosRAG.load("rag_model.pt")
	    """

	    def __init__(
	        self,
	        base_model: nn.Module,
	        rag_config: Optional[RAGConfig] = None,
	        embedding_func: Optional[Callable] = None,
	        llm_func: Optional[Callable] = None,
	    ):
	        """
	        Args:
	            base_model: 基础模型 (Mythos 模型)
	            rag_config: RAG 配置
	            embedding_func: 嵌入函数 (texts -> embeddings)
	            llm_func: LLM 函数 (用于关系抽取)
	        """
	        super().__init__()
	        self.base_model = base_model
	        self.rag_config = rag_config or RAGConfig()
	        self.embedding_func = embedding_func
	        self.llm_func = llm_func

	        # 设备
	        self.device = self.rag_config.device

	        # ===== 知识图谱组件 =====
	        self._init_knowledge_graph()

	        # ===== RAG 组件 =====
	        self._init_rag_components()

	        # ===== 检索缓存 =====
	        self._retrieval_cache = {}

	    def _init_knowledge_graph(self):
	        """初始化知识图谱"""
	        config = self.rag_config

	        # 存储后端
	        if config.kg_storage_type == "memory":
	            vector_store = InMemoryVectorStore(dimensions=config.kg_vector_dim)
	            graph_store = InMemoryGraphStore()
	        else:
	            # TODO: 支持 Neo4j, PostgreSQL
	            vector_store = InMemoryVectorStore(dimensions=config.kg_vector_dim)
	            graph_store = InMemoryGraphStore()

	        # 嵌入函数
	        if self.embedding_func is None:
	            # 默认: 随机初始化 (生产版本应使用真实嵌入)
	            self.embedding_func = self._default_embedding_func

	        # 知识图谱
	        self.kg = MultimodalKnowledgeGraph(
	            embedding_func=self.embedding_func,
	            llm_func=self.llm_func,
	            vector_store=vector_store,
	            graph_store=graph_store,
	        )

	        # 混合检索器
	        self.retriever = HybridRetrieval(
	            vector_store=vector_store,
	            graph_store=graph_store,
	            embedding_func=self.embedding_func,
	            expansion_factor=config.retrieval_expansion_factor,
	        )

	    def _init_rag_components(self):
	        """初始化 RAG 组件"""
	        config = self.rag_config
	        base_cfg = self.base_model.config if hasattr(self.base_model, 'config') else None

	        dim = base_cfg.dim if base_cfg else 768
	        routing_hidden = config.routing_hidden_dim or (dim // 2)

	        # 内容类型路由
	        self.content_router = ContentTypeRouter(
	            dim=dim,
	            hidden_dim=routing_hidden,
	        )

	        # 联合深度+模态选择器
	        self.joint_selector = JointDepthModalitySelector(
	            cfg=base_cfg or self._create_fake_cfg(dim),
	        )

	        # 自适应循环控制器
	        self.loop_controller = AdaptiveLoopController(
	            cfg=base_cfg or self._create_fake_cfg(dim),
	            joint_selector=self.joint_selector,
	        )

	        # RAG 循环块
	        if config.enable_rag and self.kg is not None:
	            self.rag_block = RAGEnhancedRecurrentBlock(
	                cfg=base_cfg or self._create_fake_cfg(dim),
	                kg=self.kg,
	                content_router=self.content_router,
	                retrieval_top_k=config.retrieval_top_k,
	                retrieval_depth=config.retrieval_depth,
	            )
	        else:
	            self.rag_block = SimplifiedRAGBlock(
	                cfg=base_cfg or self._create_fake_cfg(dim),
	            )

	    def _create_fake_cfg(self, dim: int):
	        """创建假配置用于 standalone 模式"""
	        @dataclass
	        class FakeCfg:
	            dim: int = dim
	            max_loop_iters: int = self.rag_config.max_loop_iters
	            loop_dim: int = self.rag_config.loop_dim
	            lora_rank: int = 16
	            num_heads: int = 8
	            act_threshold: float = 0.9

	        return FakeCfg()

	    def _default_embedding_func(self, texts: list[str]) -> list[np.ndarray]:
	        """默认嵌入函数 (随机)"""
	        embeddings = []
	        for _ in texts:
	            emb = np.random.randn(self.rag_config.kg_vector_dim).astype(np.float32)
	            emb = emb / (np.linalg.norm(emb) + 1e-8)
	            embeddings.append(emb)
	        return embeddings

	    # =========================================================================
	    # Document Indexing
	    # =========================================================================

	    def index_document(
	        self,
	        doc_path: str,
	        doc_id: Optional[str] = None,
	        parser: Optional[Any] = None,
	        metadata: Optional[dict] = None,
	    ) -> str:
	        """
	        索引文档到知识图谱。

	        Args:
	            doc_path: 文档路径
	            doc_id: 文档 ID (默认自动生成)
	            parser: 文档解析器 (默认使用 MultimodalDocumentParser)
	            metadata: 额外元数据

	        Returns:
	            doc_id: 索引的文档 ID
	        """
	        import uuid
	        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"

	        # 解析文档
	        if parser is None:
	            from open_mythos.rag.multimodal_parser import MultimodalDocumentParser
	            parser = MultimodalDocumentParser()

	        content_list = parser.parse(doc_path)

	        # 转换格式
	        content_dicts = []
	        for item in content_list:
	            if isinstance(item, dict):
	                content_dicts.append(item)
	            else:
	                content_dicts.append(item.to_dict())

	        # 索引到 KG
	        self.kg.index_content_list(
	            content_list=content_dicts,
	            doc_id=doc_id,
	            doc_metadata=metadata or {"path": doc_path},
	        )

	        return doc_id

	    def index_content_list(
	        self,
	        content_list: list[dict],
	        doc_id: Optional[str] = None,
	        metadata: Optional[dict] = None,
	    ) -> str:
	        """
	        直接索引预解析的内容列表。

	        Args:
	            content_list: 内容列表
	            doc_id: 文档 ID
	            metadata: 元数据

	        Returns:
	            doc_id
	        """
	        import uuid
	        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"

	        self.kg.index_content_list(
	            content_list=content_list,
	            doc_id=doc_id,
	            doc_metadata=metadata,
	        )

	        return doc_id

	    # =========================================================================
	    # Inference
	    # =========================================================================

	    def generate(
	        self,
	        query: str,
	        max_length: int = 512,
	        max_loops: Optional[int] = None,
	        enable_retrieval: bool = True,
	        retrieval_top_k: Optional[int] = None,
	        temperature: float = 0.7,
	        top_p: float = 0.9,
	    ) -> dict:
	        """
	        生成答案。

	        Args:
	            query: 查询文本
	            max_length: 最大生成长度
	            max_loops: 最大循环次数 (默认使用配置)
	            enable_retrieval: 是否启用检索
	            retrieval_top_k: 检索返回数量
	            temperature: 采样温度
	            top_p: nucleus 采样

	        Returns:
	            dict with:
	                - text: 生成的文本
	                - retrieved: 检索到的上下文
	                - loop_info: 循环信息
	        """
	        max_loops = max_loops or self.rag_config.max_loop_iters
	        retrieval_top_k = retrieval_top_k or self.rag_config.retrieval_top_k

	        # 检索阶段
	        retrieved_entities = []
	        retrieval_context = None

	        if enable_retrieval:
	            results = self.kg.retrieve(
	                query=query,
	                top_k=retrieval_top_k,
	                depth=self.rag_config.retrieval_depth,
	            )
	            retrieved_entities = results

	            # 格式化检索上下文
	            retrieval_context = self._format_retrieval_context(results)

	        # 循环推理
	        loop_info = self._run_loop(
	            query=query,
	            retrieval_context=retrieval_context,
	            max_loops=max_loops,
	            enable_retrieval=enable_retrieval,
	        )

	        # 生成最终答案
	        output_text = self._generate_from_loop(
	            query=query,
	            loop_info=loop_info,
	            max_length=max_length,
	            temperature=temperature,
	            top_p=top_p,
	        )

	        return {
	            "text": output_text,
	            "retrieved": retrieved_entities,
	            "loop_info": loop_info,
	        }

	    def _run_loop(
	        self,
	        query: str,
	        retrieval_context: Optional[str],
	        max_loops: int,
	        enable_retrieval: bool,
	    ) -> dict:
	        """运行循环推理"""
	        loop_states = []
	        halting_probs = []
	        routing_probs = None
	        retrieved_per_step = []

	        h_prev = None

	        for loop_idx in range(max_loops):
	            # 简化: 每轮进行检索和更新
	            # 实际实现需要更复杂的 hidden state 管理

	            # 内容类型路由
	            # (简化处理，实际需要真实 hidden state)
	            modality = "TEXT"

	            # 检索
	            if enable_retrieval and loop_idx == 0:
	                retrieved = self.kg.retrieve(
	                    query=query,
	                    top_k=self.rag_config.retrieval_top_k,
	                    depth=self.rag_config.retrieval_depth,
	                )
	            else:
	                retrieved = []

	            retrieved_per_step.append(retrieved)

	            # ACT halting (简化)
	            p = 0.5 + 0.03 * loop_idx
	            halting_probs.append(p)

	            # 早停
	            if p >= 0.9:
	                break

	            loop_states.append({"loop_idx": loop_idx, "modality": modality})

	        return {
	            "loop_states": loop_states,
	            "halting_probs": halting_probs,
	            "routing_probs": routing_probs,
	            "retrieved_per_step": retrieved_per_step,
	            "actual_loops": len(loop_states),
	        }

	    def _format_retrieval_context(self, results: list[dict]) -> str:
	        """格式化检索上下文"""
	        parts = []
	        for i, r in enumerate(results[:5]):
	            content_type = r.get("type", "text")
	            content = r.get("content", "")[:200]

	            parts.append(f"[{content_type.upper()}] {content}")

	        return "\n\n".join(parts)

	    def _generate_from_loop(
	        self,
	        query: str,
	        loop_info: dict,
	        max_length: int,
	        temperature: float,
	        top_p: float,
	    ) -> str:
	        """从循环信息生成答案"""
	        # 简化: 直接使用 base_model 生成
	        # 生产版本需要传入 hidden state

	        if hasattr(self.base_model, 'generate'):
	            return self.base_model.generate(
	                query,
	                max_length=max_length,
	                temperature=temperature,
	                top_p=top_p,
	            )
	        else:
	            return f"[Generated response for: {query}]"

	    # =========================================================================
	    # Forward (for training)
	    # =========================================================================

	    def forward(
	        self,
	        input_ids: torch.Tensor,
	        labels: Optional[torch.Tensor] = None,
	        retrieval_contexts: Optional[list] = None,
	        target_modalities: Optional[list[str]] = None,
	        max_loops: Optional[int] = None,
	    ) -> dict:
	        """
	        前向传播 (用于训练)。

	        Args:
	            input_ids: 输入 token IDs
	            labels: 标签
	            retrieval_contexts: 检索上下文
	            target_modalities: 目标模态
	            max_loops: 最大循环次数

	        Returns:
	            dict with loss, outputs, etc.
	        """
	        max_loops = max_loops or self.rag_config.max_loop_iters

	        # 基础模型前向
	        base_output = self.base_model(
	            input_ids=input_ids,
	            labels=labels,
	        )

	        loss = base_output.get("loss", torch.tensor(0.0))
	        hidden_states = base_output.get("hidden_states", [])

	        # RAG 增强
	        loop_states = []
	        halting_probs = []
	        retrieved_entities = []

	        if self.rag_config.enable_rag and hidden_states:
	            # RAG block forward
	            rag_output = self.rag_block(
	                hidden_states=hidden_states,
	                retrieval_contexts=retrieval_contexts,
	                target_modalities=target_modalities,
	                max_loops=max_loops,
	            )

	            loop_states = rag_output.get("loop_states", [])
	            halting_probs = rag_output.get("halting_probs", [])
	            retrieved_entities = rag_output.get("retrieved_entities", [])

	            # 添加 RAG loss
	            if rag_output.get("loss"):
	                loss = loss + rag_output["loss"]

	        return {
	            "loss": loss,
	            "base_output": base_output,
	            "loop_states": loop_states,
	            "halting_probs": halting_probs,
	            "retrieved_entities": retrieved_entities,
	        }

	    # =========================================================================
	    # Save / Load
	    # =========================================================================

	    def save(self, path: str):
	        """保存模型"""
	        import pickle
	        with open(path, 'wb') as f:
	            pickle.dump(self.state_dict(), f)

	    @classmethod
	    def load(cls, path: str, base_model: nn.Module, **kwargs):
	        """加载模型"""
	        import pickle
	        with open(path, 'rb') as f:
	            state_dict = pickle.load(f)

	        model = cls(base_model=base_model, **kwargs)
	        model.load_state_dict(state_dict)
	        return model

	    # =========================================================================
	    # Utility
	    # =========================================================================

	    def get_retriever(self):
	        """获取检索器"""
	        return self.retriever

	    def get_knowledge_graph(self):
	        """获取知识图谱"""
	        return self.kg

	    def set_embedding_model(self, model_type: str = "bge-large", **kwargs):
	        """
	        设置 embedding 模型。

	        Args:
	            model_type: 模型类型 ("bge-large", "e5")
	            **kwargs: 其他参数
	        """
	        if not HAS_RETRIEVAL:
	            raise ImportError("Retrieval module not available")

	        if model_type == "bge-large":
	            embedder = BGELargeEmbedder(**kwargs)
	        elif model_type == "e5":
	            embedder = E5Embedder(**kwargs)
	        else:
	            raise ValueError(f"Unknown model type: {model_type}")

	        self.embedding_func = embedder.encode

	        # 更新 KG 的 embedding 函数
	        if hasattr(self, 'kg'):
	            self.kg.embedding_func = embedder.encode

	    def set_reranker(self, model_name: str = "BAAI/bge-reranker-large", **kwargs):
	        """
	        设置 reranker 模型。

	        Args:
	            model_name: 模型名称
	            **kwargs: 其他参数
	        """
	        if not HAS_RETRIEVAL:
	            raise ImportError("Retrieval module not available")

	        self.reranker = CrossEncoderReranker(model_name=model_name, **kwargs)
	        return self.reranker


__all__ = [
	    "OpenMythosRAG",
	    "RAGConfig",
	    "RerankResult" if HAS_RETRIEVAL else None,
	    "BGELargeEmbedder" if HAS_RETRIEVAL else None,
	    "E5Embedder" if HAS_RETRIEVAL else None,
	    "CrossEncoderReranker" if HAS_RETRIEVAL else None,
	    "RerankerPool" if HAS_RETRIEVAL else None,
	    "HAS_RETRIEVAL",
]