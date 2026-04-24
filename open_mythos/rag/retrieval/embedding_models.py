"""
Embedding Model Integration
============================

支持多种 embedding 模型:
- BGE-large (BAAI)
- E5 (Microsoft)

Usage:
    # BGE-large
    embedder = BGELargeEmbedder(device="cuda", normalize=True)
    embeddings = embedder.encode(["hello world"])

    # E5
    embedder = E5Embedder(device="cuda", normalize=True)
    embeddings = embedder.encode(["hello world"])
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Embedding Model Base
# ============================================================================


class EmbeddingModel(ABC):
    """Embedding 模型基类"""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    @abstractmethod
    def _load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """编码单条文本"""
        pass

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        """
        批量编码文本。

        Args:
            texts: 文本列表

        Returns:
            embeddings 列表
        """
        if self._model is None:
            self._load_model()

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _encode_batch(self, batch: list[str]) -> list[np.ndarray]:
        """批量编码"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"


# ============================================================================
# BGE-large Embedder
# ============================================================================


class BGELargeEmbedder(EmbeddingModel):
    """
    BAAI BGE-large embedding 模型。

    支持:
    - bgem3 (最新版本，支持稠密+稀疏+多向量)
    - bge-large (传统版本)

    Usage:
        embedder = BGELargeEmbedder()
        embeddings = embedder.encode(["hello", "world"])
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: Optional[str] = None,
        normalize: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
        use_fp16: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            normalize=normalize,
            max_length=max_length,
            batch_size=batch_size,
        )
        self.use_fp16 = use_fp16

    def _load_model(self):
        """加载 BGE 模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装，请运行: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            prompts=None,
            default_response_length=256,
        )

        if self.use_fp16 and self.device == "cuda":
            self._model = self._model.half()

    def _encode_batch(self, batch: list[str]) -> list[np.ndarray]:
        """批量编码"""
        embeddings = self._model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            convert_to_tensor=False,
            prompt_name=None,
            prompt=None,
        )

        return [emb.astype(np.float32) for emb in embeddings]

    def encode_single(self, text: str) -> np.ndarray:
        """编码单条文本"""
        if self._model is None:
            self._load_model()

        emb = self._model.encode(
            [text],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )[0]

        return emb.astype(np.float32)


# ============================================================================
# E5 Embedder
# ============================================================================


class E5Embedder(EmbeddingModel):
    """
    Microsoft E5 embedding 模型。

    支持:
    - e5-large-v2
    - e5-base-v2
    - e5-small-v2

    注意: E5 模型需要特定的前缀:
    - query: "query: " + text
    - passage: "passage: " + text

    Usage:
        embedder = E5Embedder(model_name="intfloat/e5-large-v2")

        # 查询编码
        query_emb = embedder.encode_query("what is AI")

        # 文档编码
        doc_emb = embedder.encode_passage(["document text"])
    """

    E5_QUERY_PREFIX = "query: "
    E5_PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
        use_fp16: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            normalize=normalize,
            max_length=max_length,
            batch_size=batch_size,
        )
        self.use_fp16 = use_fp16

    def _load_model(self):
        """加载 E5 模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装，请运行: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        if self.use_fp16 and self.device == "cuda":
            self._model = self._model.half()

    def _encode_batch(self, batch: list[str], prefix: str = "") -> list[np.ndarray]:
        """批量编码"""
        if prefix:
            batch = [prefix + text for text in batch]

        embeddings = self._model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return [emb.astype(np.float32) for emb in embeddings]

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询文本。

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        if self._model is None:
            self._load_model()

        emb = self._model.encode(
            [self.E5_QUERY_PREFIX + query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )[0]

        return emb.astype(np.float32)

    def encode_queries(self, queries: list[str]) -> list[np.ndarray]:
        """
        批量编码查询文本。

        Args:
            queries: 查询列表

        Returns:
            查询向量列表
        """
        return self._encode_batch(queries, prefix=self.E5_QUERY_PREFIX)

    def encode_passage(self, passage: str) -> np.ndarray:
        """
        编码单个文档。

        Args:
            passage: 文档文本

        Returns:
            文档向量
        """
        if self._model is None:
            self._load_model()

        emb = self._model.encode(
            [self.E5_PASSAGE_PREFIX + passage],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )[0]

        return emb.astype(np.float32)

    def encode_passages(self, passages: list[str]) -> list[np.ndarray]:
        """
        批量编码文档。

        Args:
            passages: 文档列表

        Returns:
            文档向量列表
        """
        return self._encode_batch(passages, prefix=self.E5_PASSAGE_PREFIX)

    def encode(self, texts: list[str], mode: str = "passage") -> list[np.ndarray]:
        """
        通用编码接口。

        Args:
            texts: 文本列表
            mode: "query" 或 "passage"

        Returns:
            embeddings 列表
        """
        if mode == "query":
            return self.encode_queries(texts)
        else:
            return self.encode_passages(texts)


# ============================================================================
# Embedding Model Factory
# ============================================================================


def create_embedding_model(
    model_type: str,
    device: Optional[str] = None,
    **kwargs,
) -> EmbeddingModel:
    """
    创建 embedding 模型。

    Args:
        model_type: 模型类型 ("bge-large", "e5", "bge-m3")
        device: 设备
        **kwargs: 其他参数

    Returns:
        EmbeddingModel 实例
    """
    model_type = model_type.lower()

    if model_type in ("bge-large", "bge", "bgem3"):
        return BGELargeEmbedder(
            model_name=kwargs.get("model_name", "BAAI/bge-large-zh-v1.5"),
            device=device,
            normalize=kwargs.get("normalize", True),
            max_length=kwargs.get("max_length", 512),
            batch_size=kwargs.get("batch_size", 32),
        )
    elif model_type in ("e5", "e5-large", "e5-base"):
        return E5Embedder(
            model_name=kwargs.get("model_name", "intfloat/e5-large-v2"),
            device=device,
            normalize=kwargs.get("normalize", True),
            max_length=kwargs.get("max_length", 512),
            batch_size=kwargs.get("batch_size", 32),
        )
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


# ============================================================================
# Utility Functions
# ============================================================================


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean Pooling - 考虑 attention mask 的平均池化。

    用于获取句子级别的嵌入。
    """
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)