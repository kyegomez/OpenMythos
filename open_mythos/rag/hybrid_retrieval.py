"""
Phase 2: 混合检索引擎
=======================

结合向量相似度搜索和知识图谱遍历的融合检索。

核心算法:
1. 向量检索获取初始候选
2. 图遍历扩展候选集
3. 多跳关系加权
4. 模态感知排序
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class RetrievalResult:
    """
    检索结果。

    Attributes:
        id: 实体 ID
        type: 实体类型 (text/image/table/equation)
        content: 文本内容
        score: 综合分数
        rank: 排序名次
        metadata: 额外元数据
        path: 从查询到该实体的路径 (用于可解释性)
    """
    id: str
    type: str
    content: str
    score: float
    rank: int = 0
    metadata: dict = field(default_factory=dict)
    path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
            "path": self.path,
        }


# ============================================================================
# Hybrid Retrieval
# ============================================================================


class HybridRetrieval:
    """
    混合检索引擎。

    结合多种检索策略:
    - 向量相似度搜索
    - 知识图谱遍历
    - 跨模态关联
    - 文档结构感知

    Usage:
        retriever = HybridRetrieval(
            vector_store=vector_db,
            graph_store=kg,
            embedding_func=embed,
        )

        results = retriever.retrieve(
            query="explain the chart",
            query_modality="image",
            top_k=10,
            expand_depth=2,
        )
    """

    def __init__(
        self,
        vector_store: Any,
        graph_store: Any,
        embedding_func: Any,  # EmbeddingFunc
        cross_modal_mapping: Optional[dict[str, list[str]]] = None,
        default_top_k: int = 10,
        expansion_factor: float = 2.0,
    ):
        """
        Args:
            vector_store: 向量存储后端
            graph_store: 图存储后端
            embedding_func: 嵌入函数
            cross_modal_mapping: 跨模态映射
                e.g., {"image": ["text", "table"]} 表示图像可以关联到文本和表格
            default_top_k: 默认返回数量
            expansion_factor: 扩展因子 (扩展 top_k × expansion_factor)
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_func = embedding_func
        self.cross_modal_mapping = cross_modal_mapping or {
            "image": ["text", "table"],
            "table": ["text", "equation"],
            "equation": ["text", "table"],
            "text": ["image", "table", "equation"],
        }
        self.default_top_k = default_top_k
        self.expansion_factor = expansion_factor

    def retrieve(
        self,
        query: str,
        query_modality: str = "text",
        top_k: Optional[int] = None,
        expand_depth: int = 2,
        rerank: bool = True,
        use_cross_modal: bool = True,
    ) -> list[RetrievalResult]:
        """
        执行混合检索。

        Args:
            query: 查询文本
            query_modality: 查询的模态类型
            top_k: 返回数量
            expand_depth: 图扩展深度
            rerank: 是否重排序
            use_cross_modal: 是否使用跨模态扩展

        Returns:
            检索结果列表，按分数降序排列
        """
        top_k = top_k or self.default_top_k
        k_expanded = int(top_k * self.expansion_factor)

        # 1. 向量检索
        seed_results = self._vector_search(query, k_expanded)

        if not seed_results:
            return []

        seed_ids = [r["id"] for r in seed_results]

        # 2. 图扩展
        if expand_depth > 0:
            expanded = self._graph_expand(seed_ids, expand_depth)
        else:
            expanded = seed_results

        # 3. 跨模态扩展
        if use_cross_modal:
            cross_modal_results = self._cross_modal_expand(
                seed_ids, query_modality
            )
            expanded = self._merge_results(expanded, cross_modal_results)

        # 4. 重排序
        if rerank:
            expanded = self._rerank(query, expanded)

        # 5. 截断并返回
        results = expanded[:top_k]

        # 设置排名
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        """向量相似度搜索"""
        # 获取查询嵌入
        query_emb = self.embedding_func([query])
        if isinstance(query_emb, list):
            query_emb = query_emb[0]
        query_emb = np.array(query_emb)

        # 搜索
        results = self.vector_store.search(query_emb, top_k)

        # 补充信息
        enriched = []
        for r in results:
            entity_data = {
                "id": r["id"],
                "score": r["score"],
                "modality": None,
                "content": "",
                "metadata": {},
            }

            if self.graph_store:
                entity = self.graph_store.get_node(r["id"])
                if entity:
                    entity_data["modality"] = entity.type.value
                    entity_data["content"] = entity.content
                    entity_data["metadata"] = entity.metadata

            enriched.append(entity_data)

        return enriched

    def _graph_expand(
        self,
        seed_ids: list[str],
        depth: int,
    ) -> list[dict]:
        """图遍历扩展"""
        from open_mythos.rag.kg_storage import EdgeType

        visited = {eid: 0 for eid in seed_ids}  # id -> distance
        frontier = list(seed_ids)

        for d in range(1, depth + 1):
            next_frontier = []
            for eid in frontier:
                neighbors = self.graph_store.get_neighbors(
                    eid,
                    edge_types=None,
                    direction="both",
                )
                for neighbor_id, edge in neighbors:
                    if neighbor_id not in visited:
                        visited[neighbor_id] = d
                        next_frontier.append(neighbor_id)
            frontier = next_frontier

        # 获取扩展实体的信息
        expanded = []
        for eid, distance in visited.items():
            entity_data = {"id": eid, "distance": distance}

            if self.graph_store:
                entity = self.graph_store.get_node(eid)
                if entity:
                    entity_data["modality"] = entity.type.value
                    entity_data["content"] = entity.content
                    entity_data["metadata"] = entity.metadata

            # 距离衰减分数
            entity_data["score"] = 1.0 / (1.0 + distance * 0.5)

            expanded.append(entity_data)

        return expanded

    def _cross_modal_expand(
        self,
        seed_ids: list[str],
        query_modality: str,
    ) -> list[dict]:
        """跨模态扩展"""
        if query_modality == "text":
            return []

        # 获取可关联的模态
        related_modalities = self.cross_modal_mapping.get(query_modality, [])
        if not related_modalities:
            return []

        # 找到相关模态的实体
        expanded = []
        for eid in seed_ids:
            neighbors = self.graph_store.get_neighbors(
                eid,
                edge_types=[EdgeType.CROSS_MODAL],
                direction="both",
            )
            for neighbor_id, edge in neighbors:
                entity = self.graph_store.get_node(neighbor_id)
                if entity and entity.type.value in related_modalities:
                    expanded.append({
                        "id": neighbor_id,
                        "score": edge.weight * 0.8,  # 跨模态衰减
                        "modality": entity.type.value,
                        "content": entity.content,
                        "metadata": entity.metadata,
                    })

        return expanded

    def _merge_results(
        self,
        results_a: list[dict],
        results_b: list[dict],
    ) -> list[dict]:
        """合并两个结果列表（去重）"""
        seen = {}
        for r in results_a:
            seen[r["id"]] = r

        for r in results_b:
            if r["id"] not in seen:
                seen[r["id"]] = r
            else:
                # 取较高分数
                seen[r["id"]]["score"] = max(seen[r["id"]]["score"], r["score"])

        return list(seen.values())

    def _rerank(
        self,
        query: str,
        results: list[dict],
    ) -> list[RetrievalResult]:
        """重排序"""
        if not results:
            return []

        # 简单重排序：综合分数 + 模态匹配度 + 内容相关性
        query_emb = self.embedding_func([query])
        if isinstance(query_emb, list):
            query_emb = query_emb[0]
        query_emb = np.array(query_emb)

        for r in results:
            # 向量相似度已在 score 中
            # 额外调整
            modality_bonus = 1.0 if r.get("modality") else 1.0
            content_length_penalty = min(len(r.get("content", "")) / 1000, 1.0)

            r["final_score"] = r["score"] * modality_bonus * (1 + 0.1 * content_length_penalty)

        # 排序
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        return [
            RetrievalResult(
                id=r["id"],
                type=r.get("modality", "text"),
                content=r.get("content", ""),
                score=r.get("final_score", r.get("score", 0)),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------

    def retrieve_by_modality(
        self,
        query: str,
        modality: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """按模态检索"""
        return self.retrieve(
            query=query,
            query_modality=modality,
            top_k=top_k,
            use_cross_modal=False,
        )

    def get_context_for_loop(
        self,
        retrieval_results: list[RetrievalResult],
        max_tokens: int = 4096,
    ) -> str:
        """
        将检索结果格式化为循环推理的上下文。

        Args:
            retrieval_results: 检索结果
            max_tokens: 最大 token 数（估算）

        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        current_tokens = 0

        for result in retrieval_results:
            # 估算 token 数 (粗略: 1 token ≈ 4 chars)
            est_tokens = len(result.content) / 4
            if current_tokens + est_tokens > max_tokens:
                break

            # 格式化
            if result.type == "text":
                part = f"[TEXT] {result.content}"
            elif result.type == "image":
                caption = result.metadata.get("caption", [])
                caption_text = caption[0] if caption else "Image"
                part = f"[IMAGE] {caption_text}"
            elif result.type == "table":
                part = f"[TABLE]\n{result.content}"
            elif result.type == "equation":
                latex = result.metadata.get("latex", result.content)
                part = f"[EQUATION] {latex}"
            else:
                part = f"[{result.type.upper()}] {result.content}"

            context_parts.append(part)
            current_tokens += est_tokens

        return "\n\n".join(context_parts)

    def batch_retrieve(
        self,
        queries: list[str],
        query_modalities: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[list[RetrievalResult]]:
        """
        批量检索。

        Args:
            queries: 查询列表
            query_modalities: 对应的模态列表
            top_k: 每查询返回数量

        Returns:
            每查询的检索结果列表
        """
        if query_modalities is None:
            query_modalities = ["text"] * len(queries)

        results = []
        for query, modality in zip(queries, query_modalities):
            r = self.retrieve(query, modality, top_k)
            results.append(r)

        return results
