"""
Phase 2: 多模态知识图谱
========================

支持多种模态 (text, image, table, equation) 的统一知识图谱索引。

核心功能:
- 实体创建与存储
- 跨模态关系抽取
- 图结构维护
- 文档层次结构保持
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
import asyncio
import json
import os
import uuid

import numpy as np
import torch


# ============================================================================
# Protocols / Type Hints
# ============================================================================


class EmbeddingFunc(Protocol):
    """嵌入函数协议"""
    def __call__(self, texts: list[str]) -> list[np.ndarray]: ...


class LLMFunc(Protocol):
    """LLM 函数协议"""
    async def __call__(self, prompt: str, **kwargs) -> str: ...


# ============================================================================
# Multimodal Knowledge Graph
# ============================================================================


class MultimodalKnowledgeGraph:
    """
    多模态知识图谱。

    统一管理 text, image, table, equation 四类实体的:
    - 向量存储
    - 图结构存储
    - 跨模态关系抽取

    Usage:
        # 初始化
        kg = MultimodalKnowledgeGraph(
            embedding_func=embedding_model.encode,
            llm_func=gpt4_complete,
            vector_store=InMemoryVectorStore(),
            graph_store=InMemoryGraphStore(),
        )

        # 索引内容
        content_list = parser.parse("document.pdf")
        kg.index_content_list(content_list, doc_id="doc_001")

        # 检索
        results = kg.retrieve("query text", top_k=10, depth=2)
    """

    def __init__(
        self,
        embedding_func: EmbeddingFunc,
        llm_func: Optional[LLMFunc] = None,
        vector_store: Optional[Any] = None,
        graph_store: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """
        Args:
            embedding_func: 文本嵌入函数 (texts -> embeddings)
            llm_func: LLM 调用函数 (用于关系抽取)
            vector_store: 向量存储后端
            graph_store: 图存储后端
            device: 计算设备
            batch_size: 批处理大小
        """
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.device = device
        self.batch_size = batch_size

        # 存储后端
        self.vector_store = vector_store
        self.graph_store = graph_store

        # 用于跟踪文档结构
        self._doc_pages: dict[str, list[str]] = {}  # doc_id -> [page_ids]
        self._page_entities: dict[str, list[str]] = {}  # page_id -> [entity_ids]

        # 嵌入缓存 (避免重复计算)
        self._embedding_cache: dict[str, np.ndarray] = {}

    # --------------------------------------------------------------------------
    # Content Indexing
    # --------------------------------------------------------------------------

    def index_content_list(
        self,
        content_list: list[dict],
        doc_id: str,
        doc_metadata: Optional[dict] = None,
    ) -> list[str]:
        """
        将预解析的 content_list 索引到知识图谱。

        Args:
            content_list: [
                {"type": "text"|"image"|"table"|"equation", ...},
                ...
            ]
            doc_id: 文档唯一标识
            doc_metadata: 文档级别元数据

        Returns:
            索引的实体 ID 列表
        """
        from open_mythos.rag.kg_storage import Entity, EntityType, Edge, EdgeType

        entity_ids = []

        # 创建文档节点
        doc_entity_id = f"{doc_id}_doc"
        if self.graph_store:
            doc_entity = Entity(
                id=doc_entity_id,
                type=EntityType.DOCUMENT,
                content=f"Document: {doc_id}",
                metadata=doc_metadata or {},
            )
            self.graph_store.upsert_node(doc_entity_id, doc_entity)

        # 创建页面节点
        page_ids = sorted(set(item.get("page_idx", 0) for item in content_list))
        self._doc_pages[doc_id] = []

        for page_idx in page_ids:
            page_id = f"{doc_id}_page_{page_idx}"
            self._doc_pages[doc_id].append(page_id)

            if self.graph_store:
                page_entity = Entity(
                    id=page_id,
                    type=EntityType.PAGE,
                    content=f"Page {page_idx}",
                    metadata={"doc_id": doc_id, "page_idx": page_idx},
                )
                self.graph_store.upsert_node(page_id, page_entity)

                # BELONGS_TO: page -> document
                self.graph_store.upsert_edge(Edge(
                    source_id=page_id,
                    target_id=doc_entity_id,
                    type=EdgeType.BELONGS_TO,
                    weight=1.0,
                ))

        # 处理每个内容项
        for item in content_list:
            entity_id = self._index_item(item, doc_id)
            if entity_id:
                entity_ids.append(entity_id)

        # 跨实体关系抽取
        if self.llm_func and len(entity_ids) > 1:
            asyncio.create_task(self._extract_relations_async(entity_ids))

        return entity_ids

    def _index_item(self, item: dict, doc_id: str) -> Optional[str]:
        """索引单个内容项"""
        from open_mythos.rag.kg_storage import Entity, EntityType, Edge, EdgeType

        content_type = item.get("type", "text")
        page_idx = item.get("page_idx", 0)
        entity_id = f"{doc_id}_{content_type}_{page_idx}_{uuid.uuid4().hex[:8]}"

        # 获取文本内容用于嵌入
        text_for_embedding = self._get_item_text(item)
        if not text_for_embedding:
            return None

        # 计算或获取嵌入
        if text_for_embedding in self._embedding_cache:
            embedding = self._embedding_cache[text_for_embedding]
        else:
            embeddings = self.embedding_func([text_for_embedding])
            embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
            self._embedding_cache[text_for_embedding] = embedding

        # 创建实体
        entity = Entity(
            id=entity_id,
            type=EntityType(content_type),
            content=text_for_embedding,
            embedding=embedding,
            metadata={
                "doc_id": doc_id,
                "page_idx": page_idx,
                "raw_data": self._get_raw_data(item),
            },
        )

        # 存储
        if self.vector_store:
            self.vector_store.upsert({entity_id: embedding})
            if hasattr(self.vector_store, '_metadata'):
                self.vector_store.set_metadata(entity_id, {"type": content_type})

        if self.graph_store:
            self.graph_store.upsert_node(entity_id, entity)

            # BELONGS_TO: entity -> page
            page_id = f"{doc_id}_page_{page_idx}"
            self.graph_store.upsert_edge(Edge(
                source_id=entity_id,
                target_id=page_id,
                type=EdgeType.BELONGS_TO,
                weight=1.0,
            ))

            # 跟踪页面实体
            if page_id not in self._page_entities:
                self._page_entities[page_id] = []
            self._page_entities[page_id].append(entity_id)

        return entity_id

    def _get_item_text(self, item: dict) -> str:
        """提取用于嵌入的文本"""
        item_type = item.get("type", "text")

        if item_type == "text":
            return item.get("text", "")
        elif item_type == "image":
            caption = item.get("caption", [])
            if isinstance(caption, list) and caption:
                return f"Image: {caption[0]}"
            return "Image content"
        elif item_type == "table":
            return f"Table: {item.get('markdown', '')}"
        elif item_type == "equation":
            latex = item.get("latex", "")
            text = item.get("text", "")
            return f"Equation: {latex} {text}".strip()
        else:
            return item.get("text", item.get("content", ""))

    def _get_raw_data(self, item: dict) -> dict:
        """提取原始数据（用于检索结果）"""
        item_type = item.get("type", "text")
        result = {"type": item_type}

        if item_type == "text":
            result["text"] = item.get("text", "")
        elif item_type == "image":
            result["img_path"] = item.get("img_path", "")
            result["caption"] = item.get("caption", [])
        elif item_type == "table":
            result["markdown"] = item.get("markdown", "")
            result["caption"] = item.get("table_caption", [])
        elif item_type == "equation":
            result["latex"] = item.get("latex", "")
            result["text"] = item.get("text", "")

        return result

    # --------------------------------------------------------------------------
    # Relation Extraction
    # --------------------------------------------------------------------------

    async def _extract_relations_async(self, entity_ids: list[str]):
        """异步抽取跨实体关系"""
        if not self.llm_func or len(entity_ids) < 2:
            return

        try:
            # 获取实体内容
            entities = []
            for eid in entity_ids:
                if self.graph_store:
                    entity = self.graph_store.get_node(eid)
                    if entity:
                        entities.append({
                            "id": eid,
                            "type": entity.type.value,
                            "content": entity.content[:200],  # 截断
                        })

            if len(entities) < 2:
                return

            # 构建关系抽取提示
            prompt = self._build_relation_extraction_prompt(entities)

            # 调用 LLM
            response = await self.llm_func(prompt)

            # 解析关系
            relations = self._parse_relations(response)

            # 存储关系
            for rel in relations:
                self._add_relation(
                    source_id=rel["source"],
                    target_id=rel["target"],
                    rel_type=rel["type"],
                    weight=rel.get("weight", 0.8),
                )

        except Exception as e:
            print(f"Relation extraction failed: {e}")

    def _build_relation_extraction_prompt(self, entities: list[dict]) -> str:
        """构建关系抽取提示"""
        entity_str = "\n".join([
            f"- {e['id']}: [{e['type']}] {e['content']}"
            for e in entities[:20]  # 限制数量
        ])

        return f"""Given the following entities extracted from a document, identify meaningful relationships between them.

Entities:
{entity_str}

Identify relationships between entities. For each relationship, specify:
- source: ID of the source entity
- target: ID of the target entity
- type: one of [semantic_sim, cross_modal, coreference, causal, parallel, derives_from]
- weight: confidence score 0-1

Output as JSON array:
[
  {{"source": "id1", "target": "id2", "type": "semantic_sim", "weight": 0.9}},
  ...
]

Only output valid JSON, no explanation."""

    def _parse_relations(self, response: str) -> list[dict]:
        """解析 LLM 返回的关系"""
        try:
            # 尝试提取 JSON
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def _add_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 0.8,
    ):
        """添加关系"""
        from open_mythos.rag.kg_storage import Edge, EdgeType

        if self.graph_store:
            self.graph_store.upsert_edge(Edge(
                source_id=source_id,
                target_id=target_id,
                type=EdgeType(rel_type),
                weight=weight,
            ))

    # --------------------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_type: Optional[str] = None,
        top_k: int = 10,
        depth: int = 2,
    ) -> list[dict]:
        """
        检索相关实体。

        Args:
            query: 查询文本
            query_type: 查询类型 (text|image|table|equation|mixed)
            top_k: 返回数量
            depth: 图遍历深度

        Returns:
            检索结果列表
        """
        # 向量检索
        query_emb = self._get_embedding(query)
        seed_results = self._vector_search(query_emb, top_k * 2, query_type)

        if not seed_results:
            return []

        seed_ids = [r["id"] for r in seed_results]

        # 图扩展
        if depth > 0:
            expanded = self._graph_expand(seed_ids, depth)
        else:
            expanded = seed_results

        # 模态过滤
        if query_type and query_type != "mixed":
            expanded = [r for r in expanded if r.get("type") == query_type]

        # 融合排序
        reranked = self._rerank(query_emb, seed_results, expanded, top_k)

        return reranked

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        embeddings = self.embedding_func([text])
        embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
        self._embedding_cache[text] = embedding
        return embedding

    def _vector_search(
        self,
        query_emb: np.ndarray,
        top_k: int,
        query_type: Optional[str] = None,
    ) -> list[dict]:
        """向量相似度搜索"""
        if not self.vector_store:
            return []

        filters = {"type": query_type} if query_type and query_type != "mixed" else None
        results = self.vector_store.search(query_emb, top_k, filters)

        # 补充实体信息
        enriched = []
        for r in results:
            entity_id = r["id"]
            entity_data = {
                "id": entity_id,
                "score": r["score"],
            }

            if self.graph_store:
                entity = self.graph_store.get_node(entity_id)
                if entity:
                    entity_data["type"] = entity.type.value
                    entity_data["content"] = entity.content
                    entity_data["metadata"] = entity.metadata

            enriched.append(entity_data)

        return enriched

    def _graph_expand(
        self,
        seed_ids: list[str],
        depth: int,
    ) -> list[dict]:
        """从种子实体出发，遍历 KG 获取关联实体"""
        from open_mythos.rag.kg_storage import EdgeType

        if not self.graph_store:
            return []

        visited = set(seed_ids)
        frontier = list(seed_ids)

        for _ in range(depth):
            next_frontier = []
            for eid in frontier:
                neighbors = self.graph_store.get_neighbors(
                    eid,
                    edge_types=None,
                    direction="both",
                )
                for neighbor_id, edge in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.append(neighbor_id)
            frontier = next_frontier

        # 获取扩展实体的详细信息
        expanded = []
        for eid in visited:
            entity_data = {"id": eid}

            if self.graph_store:
                entity = self.graph_store.get_node(eid)
                if entity:
                    entity_data["type"] = entity.type.value
                    entity_data["content"] = entity.content
                    entity_data["metadata"] = entity.metadata

            expanded.append(entity_data)

        return expanded

    def _rerank(
        self,
        query_emb: np.ndarray,
        seed_results: list[dict],
        expanded: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        融合排序。

        策略:
        1. 种子结果获得 boost (×1.5)
        2. 图扩展结果根据距离衰减
        3. 综合排序
        """
        seed_ids = {r["id"] for r in seed_results}

        # 计算综合分数
        scored = []
        for item in expanded:
            entity_id = item["id"]

            # 基础分数（向量相似度）
            base_score = item.get("score", 0.0)

            # 种子 boost
            if entity_id in seed_ids:
                base_score *= 1.5

            # 图距离衰减
            depth_bonus = 0.0
            if "metadata" in item:
                # 可以根据 metadata 中的关系类型调整
                pass

            total_score = base_score + depth_bonus
            scored.append((entity_id, total_score, item))

        # 排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k
        return [item for _, _, item in scored[:top_k]]

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------

    def clear(self):
        """清空所有数据"""
        if self.vector_store:
            # 无法直接清空，遍历删除
            pass
        if self.graph_store:
            # 同上
            pass
        self._embedding_cache.clear()
        self._doc_pages.clear()
        self._page_entities.clear()

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "num_documents": len(self._doc_pages),
            "num_entities": len(self._embedding_cache),
            "embedding_cache_size": len(self._embedding_cache),
        }
