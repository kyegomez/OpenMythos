"""
CrossEncoder Reranker
======================

使用 CrossEncoder 对检索结果进行重排序。

支持模型:
- BAAI/bge-reranker-large
- cross-encoder/ms-marco-MiniLM-L-6-v2
- cross-encoder/ms-marco-MiniLM-L-12-v2

Usage:
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-large")

    # 重排序
    results = reranker.rerank(
        query="what is AI",
        documents=["AI is artificial intelligence", "Python is a programming language"],
        top_k=5,
    )
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Rerank Result
# ============================================================================


@dataclass
class RerankResult:
    """重排序结果"""
    index: int          # 原始文档索引
    doc_id: str         # 文档 ID
    content: str        # 文档内容
    score: float        # 重新计算的分数
    original_rank: int  # 原始排名

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "original_rank": self.original_rank,
        }


# ============================================================================
# CrossEncoder Reranker
# ============================================================================


class CrossEncoderReranker:
    """
    CrossEncoder 重排序器。

    使用交叉编码器对 query-document 对进行打分，实现更精确的排序。

    支持模型:
    - BAAI/bge-reranker-large (中文最强)
    - BAAI/bge-reranker-base
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (英文快速)
    - cross-encoder/ms-marco-MiniLM-L-12-v2
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 64,
        use_fp16: bool = True,
        normalize_scores: bool = True,
    ):
        """
        Args:
            model_name: 模型名称
            device: 设备 (cuda/cpu)
            max_length: 最大序列长度
            batch_size: 批处理大小
            use_fp16: 是否使用 FP16
            normalize_scores: 是否归一化分数
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and (self.device == "cuda")
        self.normalize_scores = normalize_scores
        self._model = None

    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装，请运行: pip install sentence-transformers"
            )

        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
            tokenizer_args={"fast": True},
            model_args={"torch_dtype": torch.float16} if self.use_fp16 else {},
        )

    def _get_model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[str],
        doc_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        return_scores: bool = False,
    ) -> list[RerankResult] | list[tuple[RerankResult, float]]:
        """
        对文档进行重排序。

        Args:
            query: 查询文本
            documents: 文档列表
            doc_ids: 文档 ID 列表 (可选)
            top_k: 返回前 k 个结果 (None = 返回全部)
            return_scores: 是否同时返回原始分数

        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []

        model = self._get_model()

        # 构建 query-document 对
        pairs = [[query, doc] for doc in documents]

        # 批量预测
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            apply_softmax=True,
        )

        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()

        # 如果是单个值，转为列表
        if isinstance(scores, (int, float)):
            scores = [scores]

        # 归一化分数
        if self.normalize_scores and scores:
            min_s = min(scores)
            max_s = max(scores)
            if max_s > min_s:
                scores = [(s - min_s) / (max_s - min_s) for s in scores]
            else:
                scores = [0.5] * len(scores)

        # 构建结果
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        results = [
            RerankResult(
                index=i,
                doc_id=doc_ids[i],
                content=documents[i],
                score=scores[i],
                original_rank=i + 1,
            )
            for i in range(len(documents))
        ]

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 截取 top_k
        if top_k is not None and top_k < len(results):
            results = results[:top_k]

        if return_scores:
            return [(r, r.score) for r in results]
        return results

    def rerank_with_scores(
        self,
        query: str,
        documents: list[str],
        doc_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float, int]]:
        """
        重排序并返回 (doc_id, score, original_rank)。

        简化接口。
        """
        results = self.rerank(query, documents, doc_ids, top_k, return_scores=False)
        return [(r.doc_id, r.score, r.original_rank) for r in results]


# ============================================================================
# Hybrid Reranker (Combine multiple rerankers)
# ============================================================================


class HybridReranker:
    """
    混合重排序器。

    将多个 CrossEncoder 的结果进行融合。
    """

    def __init__(
        self,
        rerankers: list[CrossEncoderReranker],
        fusion_method: str = "rrf",  # "rrf" (Reciprocal Rank Fusion) or "average"
        k: int = 60,  # RRF parameter
    ):
        """
        Args:
            rerankers: 重排序器列表
            fusion_method: 融合方法 ("rrf" 或 "average")
            k: RRF 参数
        """
        self.rerankers = rerankers
        self.fusion_method = fusion_method
        self.k = k

    def rerank(
        self,
        query: str,
        documents: list[str],
        doc_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """
        融合多个 reranker 的结果。
        """
        if not self.rerankers:
            raise ValueError("No rerankers provided")

        if len(self.rerankers) == 1:
            return self.rerankers[0].rerank(query, documents, doc_ids, top_k)

        # 收集每个 reranker 的排序
        all_ranks: list[dict[str, int]] = []
        for reranker in self.rerankers:
            results = reranker.rerank(query, documents, doc_ids, return_scores=False)
            rank_dict = {r.doc_id: i for i, r in enumerate(results)}
            all_ranks.append(rank_dict)

        # 融合
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        fused_scores: dict[str, float] = {doc_id: 0.0 for doc_id in doc_ids}

        if self.fusion_method == "rrf":
            for ranks in all_ranks:
                for doc_id, rank in ranks.items():
                    fused_scores[doc_id] += 1.0 / (self.k + rank + 1)
        else:  # average
            for ranks in all_ranks:
                for doc_id in doc_ids:
                    if doc_id in ranks:
                        fused_scores[doc_id] += ranks[doc_id] / len(self.rerankers)

        # 排序
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        if top_k is not None:
            sorted_doc_ids = sorted_doc_ids[:top_k]

        # 构建结果
        doc_id_to_content = {doc_ids[i]: documents[i] for i in range(len(documents))}
        return [
            RerankResult(
                index=doc_ids.index(doc_id),
                doc_id=doc_id,
                content=doc_id_to_content.get(doc_id, ""),
                score=fused_scores[doc_id],
                original_rank=doc_ids.index(doc_id) + 1,
            )
            for doc_id in sorted_doc_ids
        ]


# ============================================================================
# CrossEncoder Pool (Manager)
# ============================================================================


class RerankerPool:
    """
    重排序器池。

    管理多个重排序器，支持动态选择。
    """

    def __init__(self):
        self._rerankers: dict[str, CrossEncoderReranker] = {}

    def register(
        self,
        name: str,
        model_name: str = "BAAI/bge-reranker-large",
        device: Optional[str] = None,
        **kwargs,
    ) -> CrossEncoderReranker:
        """注册一个重排序器"""
        reranker = CrossEncoderReranker(
            model_name=model_name,
            device=device,
            **kwargs,
        )
        self._rerankers[name] = reranker
        return reranker

    def get(self, name: str) -> Optional[CrossEncoderReranker]:
        """获取重排序器"""
        return self._rerankers.get(name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        reranker_name: str = "default",
        doc_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """使用指定的重排序器"""
        reranker = self.get(reranker_name)
        if reranker is None:
            raise KeyError(f"Reranker '{reranker_name}' not found")
        return reranker.rerank(query, documents, doc_ids, top_k)

    def list_rerankers(self) -> list[str]:
        """列出所有已注册的重排序器"""
        return list(self._rerankers.keys())