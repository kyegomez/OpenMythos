# OpenMythos × RAG-Anything 融合增强方案

## 1. 背景与目标

### 1.1 两个系统的核心能力

| 系统 | 核心能力 | 定位 |
|------|---------|------|
| **OpenMythos** | 循环 transformer + 自适应循环深度 + ACT  halting | 推理引擎 |
| **RAG-Anything** | 多模态文档解析 + 多模态 KG + 混合检索 | 文档理解 + 检索 |

### 1.2 融合目标

将 RAG-Anything 的**多模态文档理解**和**知识图谱检索**能力，与 OpenMythos 的**循环推理引擎**深度融合，构建：

> **一个能处理多模态文档、理解文档结构、构建多模态知识图谱、并通过循环推理引擎进行深度推理的统一系统**

### 1.3 预期收益

| 维度 | 当前 OpenMythos | 融合后 |
|------|----------------|--------|
| 输入类型 | 纯文本 token | PDF/Office/图像/表格/公式 |
| 知识表示 | 隐式存在于权重 | 显式多模态知识图谱 |
| 推理依据 | 训练知识 | 检索增强 + 训练知识 |
| 循环深度 | 复杂度/课程自适应 | 内容类型 + 复杂度联合自适应 |

---

## 2. 系统架构

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    输入文档 (PDF/Office/图像)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────────┐
   │ MinerU   │   │ Docling  │   │ PaddleOCR    │
   │ 解析器   │   │ 解析器   │   │ 解析器        │
   └────┬─────┘   └────┬─────┘   └──────┬───────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
        ┌────────────────────────────────┐
        │   内容分类路由器 (Content Router)  │
        │   text → image → table → equation │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │ 文本分析  │    │ 视觉分析  │    │ 表格/公式分析 │
  │ 管道     │    │ 管道     │    │ 管道         │
  └────┬─────┘    └────┬─────┘    └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
        ┌────────────────────────────────┐
        │   多模态知识图谱 (Multimodal KG)  │
        │   实体: text/image/table/equation │
        │   关系: 语义关联 + 跨模态关联     │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   混合检索引擎 (Hybrid Retrieval) │
        │   向量相似度 + KG 图遍历 + 融合   │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │     OpenMythos 推理引擎          │
        │  ┌──────────────────────────┐  │
        │  │  循环 (T 次)              │  │
        │  │  ├─ 内容类型路由 (每轮)   │  │
        │  │  ├─ KG 上下文注入         │  │
        │  │  ├─ 复杂度感知深度调整     │  │
        │  │  └─ ACT halting           │  │
        │  └──────────────────────────┘  │
        └────────────────┬────────────────┘
                         │
                         ▼
                 ┌──────────────┐
                 │   结构化输出   │
                 └──────────────┘
```

### 2.2 核心模块交互

```
循环迭代 t:
    h_t = f(h_{t-1}, e, retrieval_context, content_type)

    每轮循环内:
    1. content_router 分析当前内容类型
    2. KG_RETRIEVE(content_type, h_t) → top-k 实体
    3. multimodal_context = CONCAT(top-k 实体)
    4. h_t = RECURRENT_BLOCK(h_t, multimodal_context)
    5. complexity_score = COMPLEXITY_NET(h_t)
    6. if P(halting) > threshold: 退出循环
```

---

## 3. 详细增强设计

### 3.1 多模态文档解析管道

**新增模块**: `MultimodalDocumentParser`

```python
class MultimodalDocumentParser:
    """
    支持多模态文档的统一解析管道。

    输入: PDF/Office/图像文件
    输出: content_list = [
        {"type": "text",      "text": "...", "page_idx": 0},
        {"type": "image",     "img_path": "...", "caption": "...", "page_idx": 1},
        {"type": "table",     "markdown": "...", "page_idx": 2},
        {"type": "equation",  "latex": "...", "text": "...", "page_idx": 3},
    ]
    """
    def __init__(
        self,
        parser_type: str = "mineru",  # mineru | docling | paddleocr
        enable_image: bool = True,
        enable_table: bool = True,
        enable_equation: bool = True,
        vlm_model: str = "gpt-4o",     # 视觉语言模型用于图像描述
    ):
        self.parser_type = parser_type
        self.vlm_model = vlm_model

    def parse(self, doc_path: str) -> list[dict]:
        """
        解析文档，返回 content_list。
        支持格式: PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, 图片
        """
        if doc_path.endswith(".pdf"):
            return self._parse_pdf(doc_path)
        elif doc_path.endswith((".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")):
            return self._parse_office(doc_path)
        elif doc_path.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            return self._parse_image(doc_path)
        else:
            return self._parse_text(doc_path)

    def _parse_pdf(self, path: str) -> list[dict]:
        # MinerU 高保真 PDF 解析
        # 返回: [{"type": "text"|"image"|"table"|"equation", ...}]
        ...

    def _parse_image(self, path: str) -> list[dict]:
        # VLM 生成图像描述
        # 返回: [{"type": "image", "img_path": path, "caption": "..."}]
        ...
```

**增强点**:
- 与现有 `OpenMythos` 输入无缝对接
- 支持直接插入预解析 content_list（绕过解析阶段）

---

### 3.2 多模态知识图谱

**新增模块**: `MultimodalKnowledgeGraph`

```python
class MultimodalKnowledgeGraph:
    """
    多模态知识图谱，实体节点支持多种模态。

    节点类型:
    - TextEntity: {"type": "text", "content": str, "embedding": np.array}
    - ImageEntity: {"type": "image", "img_path": str, "caption": str, "embedding": np.array}
    - TableEntity: {"type": "table", "markdown": str, "embedding": np.array}
    - EquationEntity: {"type": "equation", "latex": str, "text": str, "embedding": np.array}

    边类型:
    - SEMANTIC_SIM: 语义相似（text ↔ text, image ↔ image）
    - CROSS_MODAL: 跨模态关联（text ↔ image, table ↔ text）
    - BELONGS_TO: 文档结构（element → page → document）
    - COREFERENCE: 指代关联（同一实体的多次出现）
    """

    def __init__(
        self,
        embedding_func: EmbeddingFunc,  # from lightrag
        llm_func: LLMFunc,
        vector_storage: VectorStore,
        graph_storage: GraphStore,
    ):
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.vector_store = vector_storage
        self.graph_store = graph_storage

    def index_content_list(self, content_list: list[dict], doc_id: str):
        """
        将预解析的 content_list 索引到 KG。
        1. 为每个元素创建实体节点
        2. 提取实体间关系创建边
        3. 存储向量 + 图结构
        """
        for item in content_list:
            entity_id = f"{doc_id}_{item['type']}_{item.get('page_idx', 0)}"

            # 创建实体
            entity = self._create_entity(item, entity_id)

            # 存储向量
            self.vector_store.upsert({entity_id: entity["embedding"]})

            # 存储图节点
            self.graph_store.upsert_node(entity_id, entity)

            # 建立文档结构关系
            page_id = f"{doc_id}_page_{item.get('page_idx', 0)}"
            self.graph_store.upsert_edge(
                entity_id, page_id, {"type": "BELONGS_TO", "weight": 1.0}
            )

        # 跨实体关系抽取（使用 LLM）
        self._extract_cross_entity_relations(content_list, doc_id)

    def _create_entity(self, item: dict, entity_id: str) -> dict:
        """根据内容类型创建对应实体"""
        if item["type"] == "text":
            return {
                "id": entity_id,
                "type": "text",
                "content": item["text"],
                "embedding": self.embedding_func(item["text"]),
            }
        elif item["type"] == "image":
            # VLM 生成 caption
            caption = self._vlm_caption(item["img_path"])
            return {
                "id": entity_id,
                "type": "image",
                "img_path": item["img_path"],
                "caption": caption,
                "embedding": self.embedding_func(caption),
            }
        elif item["type"] == "table":
            return {
                "id": entity_id,
                "type": "table",
                "markdown": item["markdown"],
                "embedding": self.embedding_func(item["markdown"]),
            }
        elif item["type"] == "equation":
            return {
                "id": entity_id,
                "type": "equation",
                "latex": item["latex"],
                "text": item.get("text", ""),
                "embedding": self.embedding_func(item.get("text", item["latex"])),
            }

    def retrieve(
        self,
        query: str,
        query_type: str = "text",  # text | image | table | equation
        top_k: int = 10,
        depth: int = 2,  # KG 遍历深度
    ) -> list[dict]:
        """
        混合检索：向量相似度 + KG 图遍历

        1. 向量检索 top-k 相关实体
        2. 以这些实体为种子，遍历 KG 获取 N 跳关联实体
        3. 融合排序返回
        """
        # 向量检索
        query_emb = self.embedding_func(query)
        seed_entities = self.vector_store.search(query_emb, top_k)

        # KG 扩展
        expanded = self._graph_expand(
            [e["id"] for e in seed_entities], depth=depth
        )

        # 模态过滤（如果指定了 query_type）
        if query_type != "mixed":
            expanded = [e for e in expanded if e["type"] == query_type]

        # 加权融合排序
        return self._rerank(query_emb, seed_entities, expanded, top_k)

    def _graph_expand(self, entity_ids: list[str], depth: int) -> list[dict]:
        """从种子实体出发，遍历 KG 获取关联实体"""
        visited = set(entity_ids)
        frontier = list(entity_ids)

        for _ in range(depth):
            next_frontier = []
            for eid in frontier:
                neighbors = self.graph_store.get_neighbors(eid)
                for neighbor_id, edge_data in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.append(neighbor_id)
            frontier = next_frontier

        return [self.graph_store.get_node(eid) for eid in visited]
```

---

### 3.3 内容类型路由网络

**新增模块**: `ContentTypeRouter`

```python
class ContentTypeRouter(nn.Module):
    """
    循环内每轮的内容类型路由器。

    模仿 RAG-Anything 的内容分类思想：
    - 分析当前 hidden state 对应的主要模态
    - 决定本轮循环应注入哪类检索上下文

    模态路由决策:
    - hidden state 的均值池化 → MLP → 4类 softmax
    - 4类: TEXT, IMAGE, TABLE, EQUATION
    """

    def __init__(self, dim: int, num_modalities: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_modalities),
        )
        self.modality_names = ["TEXT", "IMAGE", "TABLE", "EQUATION"]

    def forward(self, h: torch.Tensor) -> dict:
        """
        Args:
            h: hidden state (B, T, dim)

        Returns:
            dict with:
                - modality: str, 预测的模态类型
                - probs: (B, num_modalities), 各模态概率
                - routing_weights: (B, num_modalities), 用于加权检索
        """
        pooled = h.mean(dim=1)  # (B, dim)
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        # 贪心选择主要模态
        modality_idx = probs.argmax(dim=-1)
        modalities = [self.modality_names[i] for i in modality_idx]

        return {
            "modality": modalities if probs.shape[0] > 1 else modalities[0],
            "probs": probs,
            "routing_weights": probs,  # 可用于加权融合多模态检索结果
        }
```

---

### 3.4 检索增强循环推理引擎

**新增模块**: `RAGEnhancedRecurrentBlock`

```python
class RAGEnhancedRecurrentBlock(nn.Module):
    """
    检索增强的循环推理块。

    每轮循环:
    1. 路由网络决定当前内容类型
    2. 根据内容类型检索多模态 KG
    3. 将检索上下文注入 hidden state
    4. 标准的 Transformer + LTI + ACT 更新

    关键: 检索上下文丰富了循环推理的外部知识
    """

    def __init__(
        self,
        cfg: MythosConfig,
        kg: MultimodalKnowledgeGraph,
        content_router: ContentTypeRouter,
        retrieval_top_k: int = 5,
        retrieval_depth: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        self.kg = kg
        self.router = content_router
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_depth = retrieval_depth

        # 标准组件（复用现有）
        self.transformer_block = TransformerBlock(cfg, use_moe=True)
        self.lti_injection = LTIInjection(cfg.dim)
        self.lora_adapter = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.act_halting = ACTHalting(cfg.dim)
        self.norm = RMSNorm(cfg.dim)

        # 检索上下文融合
        self.retrieval_proj = nn.Linear(cfg.dim, cfg.dim)
        self.retrieval_norm = RMSNorm(cfg.dim)

    def forward(
        self,
        h: torch.Tensor,           # 当前 hidden state (B, T, dim)
        e: torch.Tensor,            # 编码输入 (B, T, dim)
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        loop_idx: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            h_out: 更新后的 hidden state
            info: dict with retrieval_context, modality, halting_prob
        """
        B, T, D = h.shape

        # ===== Step 1: 内容类型路由 =====
        routing_info = self.router(h)
        current_modality = routing_info["modality"]

        # ===== Step 2: 多模态检索 =====
        # 将 hidden state 转换为 query text（可用投影 + decode）
        query_text = self._hidden_to_query(h, current_modality)

        # 检索
        retrieved = self.kg.retrieve(
            query=query_text,
            query_type=current_modality,
            top_k=self.retrieval_top_k,
            depth=self.retrieval_depth,
        )

        # ===== Step 3: 检索上下文编码 =====
        retrieval_context = self._encode_retrieval(retrieved)  # (B, T, D)
        retrieval_context = self.retrieval_proj(retrieval_context)

        # ===== Step 4: 融合检索上下文 =====
        h_with_context = self.retrieval_norm(h + retrieval_context)

        # ===== Step 5: 标准 Transformer 更新 =====
        combined = self.norm(h_with_context + e)
        cache_key = f"rag_loop_{loop_idx}"
        trans_out = self.transformer_block(combined, freqs_cis, mask, kv_cache, cache_key)

        # LoRA 深度适配
        trans_out = trans_out + self.lora_adapter(trans_out, loop_idx)

        # ===== Step 6: LTI 注入 =====
        h_new = self.lti_injection(h, e, trans_out)

        # ===== Step 7: ACT halting =====
        p = self.act_halting(h_new)

        info = {
            "modality": current_modality,
            "routing_probs": routing_info["probs"],
            "retrieved_entities": [r["id"] for r in retrieved],
            "halting_prob": p,
        }

        return h_new, info

    def _hidden_to_query(self, h: torch.Tensor, modality: str) -> str:
        """
        将 hidden state 转换为检索查询文本。
        简化版本：取 hidden state 均值，投影到词汇表维度，取 top token。
        生产版本可使用小型解码器。
        """
        # 简化：返回空字符串，使用向量直接检索
        # 生产版本应实现: h → text decoder → query string
        return ""

    def _encode_retrieval(self, retrieved: list[dict]) -> torch.Tensor:
        """
        将检索结果编码为向量。
        简化版本: 直接使用预存的 embedding 均值。
        生产版本: 可使用多模态 encoder 融合不同模态的 embedding。
        """
        if not retrieved:
            return torch.zeros(1, 1, self.cfg.dim, device=next(self.parameters()).device)

        # 平均池化所有检索实体的 embedding
        embeddings = [torch.tensor(r["embedding"]) for r in retrieved]
        pooled = torch.stack(embeddings).mean(dim=0)
        return pooled.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
```

---

### 3.5 自适应深度 + 内容类型联合调度

**增强现有**: `ComplexityAwareLoopDepth` → `JointDepthModalitySelector`

```python
class JointDepthModalitySelector(nn.Module):
    """
    联合深度 + 模态选择器。

    输入: hidden state h_t
    输出:
        - loop_depth: 推荐的循环深度 (4/8/16)
        - primary_modality: 主要模态 (TEXT/IMAGE/TABLE/EQUATION)
        - confidence: 置信度

    决策逻辑:
    1. 复杂度网络预测复杂度分数
    2. 内容路由预测主要模态
    3. 两者联合决定:
       - 简单 + TEXT → depth=4
       - 简单 + IMAGE → depth=6  (图像理解稍慢)
       - 复杂 + TABLE → depth=16 (表格推理复杂)
       - 复杂 + EQUATION → depth=12 (公式较结构化)
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.complexity_net = ComplexityAwareLoopDepth(cfg)
        self.modality_router = ContentTypeRouter(cfg.dim)

        # 深度决策表 (可学习)
        self.depth_table = nn.Parameter(
            torch.randn(4, 4, 3) * 0.02  # complexity_level × modality × depth_options
        )

    def forward(self, h: torch.Tensor) -> dict:
        """
        Returns:
            {
                "depth": int,  # 4/8/12/16
                "modality": str,
                "confidence": float,
                "complexity": float,
            }
        """
        # 1. 复杂度评估
        complexity = self.complexity_net(h)  # (B,)

        # 2. 模态路由
        routing = self.modality_router(h)
        modality_probs = routing["probs"]
        modality_idx = modality_probs.argmax(dim=-1)

        # 3. 联合决策
        batch_size = h.shape[0]
        depths = []
        for b in range(batch_size):
            c_level = self._complexity_to_level(complexity[b])
            m_idx = modality_idx[b].item()
            depth_candidate = self.depth_table[c_level, m_idx].mean().item()
            depth = self._round_to_depth(depth_candidate)
            depths.append(depth)

        return {
            "depth": depths if batch_size > 1 else depths[0],
            "modality": routing["modality"],
            "confidence": modality_probs.max(dim=-1).values,
            "complexity": complexity,
        }

    def _complexity_to_level(self, complexity: torch.Tensor) -> int:
        """复杂度分数 → 等级 0-3"""
        if complexity < 0.25:
            return 0
        elif complexity < 0.5:
            return 1
        elif complexity < 0.75:
            return 2
        else:
            return 3

    def _round_to_depth(self, depth: float) -> int:
        """连续深度 → 离散选项"""
        options = [4, 8, 12, 16]
        return min(options, key=lambda x: abs(x - depth))
```

---

### 3.6 完整融合模型

**新增类**: `OpenMythosRAG`

```python
class OpenMythosRAG(OpenMythos):
    """
    OpenMythos 与 RAG-Anything 的完全融合模型。

    融合了:
    - RAG-Anything 的多模态文档解析管道
    - RAG-Anything 的多模态知识图谱
    - OpenMythos 的循环推理引擎
    - JointDepthModalitySelector 的自适应调度

    使用方式:
        # 初始化
        model = OpenMythosRAG(cfg, kg=multimodal_kg)

        # 文档 indexing
        content_list = model.parse_document("report.pdf")
        model.index(content_list, doc_id="report_2024")

        # RAG 增强推理
        output = model(input_ids, use_rag=True, max_loops=16)
    """

    def __init__(self, cfg: MythosConfig, kg: Optional[MultimodalKnowledgeGraph] = None):
        super().__init__(cfg)

        # RAG 组件
        self.document_parser = MultimodalDocumentParser()
        self.kg = kg or MultimodalKnowledgeGraph(...)
        self.content_router = ContentTypeRouter(cfg.dim)
        self.joint_selector = JointDepthModalitySelector(cfg)
        self.rag_recurrent_block = RAGEnhancedRecurrentBlock(
            cfg, self.kg, self.content_router
        )

        # 控制开关
        self.cfg.enable_rag_enhancement = False

    def parse_document(self, doc_path: str) -> list[dict]:
        """解析文档，返回 content_list"""
        return self.document_parser.parse(doc_path)

    def index(self, content_list: list[dict], doc_id: str):
        """将 content_list 索引到知识图谱"""
        self.kg.index_content_list(content_list, doc_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_rag: bool = False,
        n_loops: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        融合 RAG 的前向传播。

        Args:
            input_ids: 输入 token (B, T)
            use_rag: 是否启用 RAG 增强
            n_loops: 循环深度（None 则自适应）
        """
        # Embedding
        x = self.embedding(input_ids)
        freqs_cis = self.freqs_cis[: x.shape[1]]

        # Prelude
        h = x
        for layer in self.prelude_layers:
            h = layer(h, freqs_cis, mask=None)

        # 循环推理（启用 RAG 则使用 RAG 增强块）
        if use_rag and self.cfg.enable_rag_enhancement:
            h = self._rag_loop(h, x, freqs_cis, n_loops)
        else:
            h = self._standard_loop(h, x, freqs_cis, n_loops)

        # Coda
        for layer in self.coda_layers:
            h = layer(h, freqs_cis, mask=None)

        # LM head
        return self.lm_head(h)

    def _rag_loop(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        n_loops: Optional[int],
    ) -> torch.Tensor:
        """
        RAG 增强的循环推理。
        每轮循环包含: 内容路由 → KG 检索 → 上下文注入 → Transformer 更新
        """
        max_loops = n_loops or self.cfg.max_loop_iters
        halted = torch.zeros(h.shape[0], h.shape[1], device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(h.shape[0], h.shape[1], device=h.device)

        loop_outputs = []

        for t in range(max_loops):
            # 自适应深度选择
            if self.joint_selector is not None:
                decision = self.joint_selector(h)
                current_modality = decision["modality"]
            else:
                current_modality = "TEXT"

            # RAG 增强循环块
            h_new, info = self.rag_recurrent_block(
                h, e, freqs_cis, loop_idx=t
            )

            # ACT
            p = info["halting_prob"]
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()

            h = h + weight.unsqueeze(-1) * (h_new - h)
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            loop_outputs.append({
                "step": t,
                "modality": current_modality,
                "retrieved": info["retrieved_entities"],
                "halting_prob": p,
            })

            if halted.all():
                break

        # 可选：保存 loop_outputs 用于调试
        self.last_loop_outputs = loop_outputs

        return h
```

---

## 4. 训练策略

### 4.1 两阶段训练

```
阶段 1: 文档理解预训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标: 学会解析和索引多模态文档

数据:
  - 文档 PDF: arXiv 论文、财报、PPT
  - 内容类型标注: text / image / table / equation
  - KG 关系标注: 跨模态关联标注

损失:
  L = L_entity_extraction + L_relation_prediction + L_retrieval

训练:
  - 冻住 OpenMythos 主干
  - 只训练: DocumentParser, KGIndexer, ContentRouter
  - 30K steps, lr=1e-4


阶段 2: RAG-增强循环推理微调
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标: 学会在循环中有效利用检索上下文

数据:
  - 需要外部知识的问答对
  - 涉及图表/公式的多模态问题

损失:
  L = L_cross_entropy
    + λ1 * L_retrieval_quality      (检索结果与答案相关性)
    + λ2 * L_loop_efficiency        (鼓励早停，节省计算)
    + λ3 * L_modality_routing       (正确路由到相关模态)

训练:
  - 解冻 RAGEnhancedRecurrentBlock
  - 保持 JointDepthModalitySelector 轻量更新
  - 60K steps, lr=3e-5
```

### 4.2 检索质量信号

```python
class RetrievalQualityLoss(nn.Module):
    """
    信号来源:
    1. 直接监督: 检索结果包含正确答案片段 → 正信号
    2. 间接监督: 最终答案正确 → 回传信用分配到检索步骤
    3. 对比学习: 正确检索 vs 随机采样负例
    """

    def forward(
        self,
        retrieved_entities: list[dict],
        answer: str,
        final_loss: torch.Tensor,
        loop_step: int,
    ) -> torch.Tensor:
        # 1. 直接监督
        hit = any(
            self._text_overlap(r["content"], answer) > 0.5
            for r in retrieved_entities
        )

        # 2. 间接监督 (REINFORCE-style)
        retrieval_reward = 1.0 if hit else 0.0

        # 3. 衰减因子: 越早的检索步骤权重越高（更直接影响结果）
        decay = 0.9 ** (max_loops - loop_step)

        return final_loss - decay * retrieval_reward
```

---

## 5. 实施路线图

### Phase 1: 基础设施 (2 周)

| 任务 | 负责 | 产出 |
|------|------|------|
| MultimodalDocumentParser | 复用 MinerU | 支持 PDF/Office 解析 |
| content_list 数据结构设计 | 新增 | 统一内容表示格式 |
| 基本 KG 存储接口 | 复用 LightRAG | Neo4j / 向量存储 |

### Phase 2: 知识图谱 (2 周)

| 任务 | 负责 | 产出 |
|------|------|------|
| MultimodalKG 实体/边模型 | 新增 | 支持 4 种实体类型 |
| 跨模态关系抽取 | LLM API | text↔image, table↔text 等关系 |
| 混合检索实现 | 新增 | 向量 + KG 融合 |

### Phase 3: 循环融合 (3 周)

| 任务 | 负责 | 产出 |
|------|------|------|
| ContentTypeRouter | 新增 | 模态分类网络 |
| RAGEnhancedRecurrentBlock | 新增 | 检索增强循环块 |
| JointDepthModalitySelector | 新增 | 联合调度器 |
| OpenMythosRAG 整合 | 新增 | 统一入口类 |

### Phase 4: 训练与优化 (2 周)

| 任务 | 负责 | 产出 |
|------|------|------|
| 两阶段训练脚本 | 新增 | 预训练 + 微调 |
| 检索质量损失函数 | 新增 | RAG 信号回传 |
| 推理优化 (批处理检索) | 优化 | 减少检索延迟 |

### 总工期: ~9 周

---

## 6. 评估基准

### 6.1 多模态文档理解

| 基准 | 描述 | 指标 |
|------|------|------|
| MMLU (多模态) | 科学图表/公式理解 | Accuracy |
| DocVQA | 文档视觉问答 | ANLS |
| ChartQA | 图表问答 | Accuracy |
| MathVista | 数学视觉推理 | Accuracy |

### 6.2 RAG 质量

| 基准 | 描述 | 指标 |
|------|------|------|
| HotpotQA | 多跳问答 | EM / F1 |
| HybridQA | 表+文混合问答 | EM / F1 |
| MultimodalQA | 多模态问答 | Accuracy |
| RAGAS | 检索相关性 | Faithfulness / Relevance |

### 6.3 循环效率

| 指标 | 描述 |
|------|------|
| 平均循环深度 | 实际使用深度 vs 最大深度 |
| 模态路由准确率 | 路由决策 vs 人工标注 |
| 检索召回率 | 检索结果包含答案的比例 |

---

## 7. 关键设计决策

### 7.1 为什么不直接用 VLM 做多模态理解？

| 方案 | 优点 | 缺点 |
|------|------|------|
| VLM 端到端 | 质量高 | 慢，依赖外部 API |
| 专用分析器 + KG | 快，可控 | 解析质量有上限 |
| **本文: 混合** | 平衡 | 中等 |

RAG-Anything 的实践证明：专用分析器（表格解析器、公式解析器）效果往往优于通用 VLM，因为针对性强。VLM 仅用于图像 captioning 等模糊场景。

### 7.2 循环内每轮都检索是否太慢？

优化策略：
1. **检索缓存**：相同 query 的检索结果缓存 N 分钟
2. **轻量路由**：ContentTypeRouter 是单层 MLP，推理极快
3. **异步检索**：与 Transformer 计算并行
4. **选择性检索**：简单样本（低复杂度）在 Prelude 后直接输出，无需循环检索

### 7.3 KG 存储选型

| 存储 | 向量 | 图结构 | 适用场景 |
|------|------|--------|---------|
| Neo4j + ChromaDB | ✅ | ✅ | 原生图查询 |
| PostgreSQL + pgvector | ✅ | ✅ | 统一 SQL |
| MongoDB + 插件 | ✅ | ❌ | 灵活文档 |
| **OpenSearch** | ✅ | ✅ | 大规模 + LightRAG 原生支持 |

推荐: **OpenSearch** 或 **PostgreSQL**（已有 LightRAG 支持）

---

## 8. 文件结构

```
/tmp/openmythos/
├── open_mythos/
│   ├── main.py                          # 原版 OpenMythos
│   ├── main_p0.py                       # P0 增强版 (含 curriculum)
│   ├── main_rag.py                      # 新增: RAG 融合版 ← NEW
│   │
│   ├── rag/                             # 新增: RAG 相关模块
│   │   ├── __init__.py
│   │   ├── multimodal_parser.py         # 多模态文档解析
│   │   ├── knowledge_graph.py           # 多模态 KG
│   │   ├── content_router.py            # 内容类型路由
│   │   ├── hybrid_retrieval.py          # 混合检索
│   │   ├── rag_recurrent_block.py       # RAG 增强循环块
│   │   └── joint_selector.py            # 联合深度+模态调度
│   │
│   └── main_enhanced.py                 # 整合所有增强 (P0+P1+P2+P3+RAG)
│
├── training/
│   ├── enhanced.py                      # P0~P3 综合训练
│   ├── rag_finetune.py                  # 新增: RAG 微调脚本
│   └── parse_pretrain.py                # 新增: 文档理解预训练
│
├── tests/
│   ├── test_rag_pipeline.py             # 新增: RAG 管道测试
│   └── test_multimodal_kg.py            # 新增: KG 测试
│
└── docs/
    └── rag_anything_enhancement_proposal.md  # 本文档
```

---

## 9. 总结

| 模块 | 新增/复用 | 代码量估计 |
|------|-----------|-----------|
| MultimodalDocumentParser | 复用 MinerU | ~200 行 |
| MultimodalKnowledgeGraph | 新增 | ~400 行 |
| ContentTypeRouter | 新增 | ~100 行 |
| HybridRetrieval | 新增 | ~200 行 |
| RAGEnhancedRecurrentBlock | 新增 | ~300 行 |
| JointDepthModalitySelector | 增强现有 | ~100 行 |
| OpenMythosRAG | 新增 | ~150 行 |
| 训练脚本 × 2 | 新增 | ~400 行 |
| **合计** | | **~1850 行** |

核心创新：
1. **循环内实时检索**：每轮循环都从多模态 KG 中检索相关内容
2. **内容类型感知**：循环深度与内容模态联合自适应
3. **统一多模态知识图谱**：文本、图像、表格、公式统一表示
4. **检索信号回传**：将检索质量信号注入训练损失
