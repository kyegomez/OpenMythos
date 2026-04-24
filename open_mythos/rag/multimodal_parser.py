"""
Phase 1: 多模态文档解析管道
============================

支持 PDF, Office文档, 图像 等多模态文档的统一解析。

核心设计:
- MinerU: 高保真 PDF 解析 (默认)
- Docling: 轻量文档解析
- PaddleOCR: 快速 OCR 识别

输出: content_list = [
    {"type": "text", "text": "...", "page_idx": 0},
    {"type": "image", "img_path": "...", "caption": "...", "page_idx": 1},
    {"type": "table", "markdown": "...", "page_idx": 2},
    {"type": "equation", "latex": "...", "text": "...", "page_idx": 3},
]
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import asyncio
import os
import subprocess
import tempfile

import torch
import numpy as np


# ============================================================================
# Data Structures
# ============================================================================


class ContentType(str, Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CUSTOM = "custom"


class ParserType(str, Enum):
    """解析器类型"""
    MINERU = "mineru"      # 高保真 PDF 解析 (默认)
    DOCLING = "docling"    # 轻量解析
    PADDLEOCR = "paddleocr"  # OCR 识别
    DIRECT = "direct"      # 直接插入预解析内容


@dataclass
class ContentItem:
    """
    统一的内容项表示。

    Attributes:
        type: 内容类型 (text/image/table/equation/custom)
        page_idx: 页码 (从 0 开始)
        # Text fields
        text: 文本内容
        # Image fields
        img_path: 图像路径 (绝对路径)
        caption: 图像描述
        footnote: 图像脚注
        # Table fields
        markdown: 表格的 Markdown 表示
        table_caption: 表格标题
        table_footnote: 表格脚注
        # Equation fields
        latex: LaTeX 公式
        equation_text: 公式的文本描述
        # Custom fields
        content: 自定义内容
        metadata: 额外元数据
    """
    type: ContentType
    page_idx: int = 0

    # Text
    text: Optional[str] = None

    # Image
    img_path: Optional[str] = None
    caption: Optional[list[str]] = field(default_factory=list)
    footnote: Optional[list[str]] = field(default_factory=list)

    # Table
    markdown: Optional[str] = None
    table_caption: Optional[list[str]] = field(default_factory=list)
    table_footnote: Optional[list[str]] = field(default_factory=list)

    # Equation
    latex: Optional[str] = None
    equation_text: Optional[str] = None

    # Custom
    content: Optional[dict] = None
    metadata: Optional[dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典格式"""
        result = {
            "type": self.type.value,
            "page_idx": self.page_idx,
        }
        if self.text is not None:
            result["text"] = self.text
        if self.img_path is not None:
            result["img_path"] = self.img_path
        if self.caption:
            result["caption"] = self.caption
        if self.footnote:
            result["footnote"] = self.footnote
        if self.markdown is not None:
            result["markdown"] = self.markdown
        if self.table_caption:
            result["table_caption"] = self.table_caption
        if self.table_footnote:
            result["table_footnote"] = self.table_footnote
        if self.latex is not None:
            result["latex"] = self.latex
        if self.equation_text is not None:
            result["equation_text"] = self.equation_text
        if self.content is not None:
            result["content"] = self.content
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ContentItem":
        """从字典创建"""
        return cls(
            type=ContentType(data["type"]),
            page_idx=data.get("page_idx", 0),
            text=data.get("text"),
            img_path=data.get("img_path"),
            caption=data.get("caption", []),
            footnote=data.get("footnote", []),
            markdown=data.get("markdown"),
            table_caption=data.get("table_caption", []),
            table_footnote=data.get("table_footnote", []),
            latex=data.get("latex"),
            equation_text=data.get("equation_text"),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Base Parser Interface
# ============================================================================


class BaseDocumentParser:
    """文档解析器基类"""

    def parse(self, doc_path: str) -> list[ContentItem]:
        """
        解析文档，返回 content_list。

        Args:
            doc_path: 文档路径 (PDF/Office/图像)

        Returns:
            list of ContentItem
        """
        raise NotImplementedError

    async def parse_async(self, doc_path: str) -> list[ContentItem]:
        """异步解析"""
        return self.parse(doc_path)


# ============================================================================
# MinerU Parser
# ============================================================================


class MinerUPreviewParser(BaseDocumentParser):
    """
    MinerU (preview) 文档解析器。

    MinerU 是一个高保真的 PDF 文档解析工具，能够:
    - 精确识别文档结构 (标题、正文、页眉、页脚)
    - 提取图像、表格、公式
    - 保持文档层次结构

    安装: pip install magic-pdf
    注意: MinerU 2.0 不再使用 magic-pdf.json 配置
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        parse_method: str = "auto",  # auto | ocr | txt
    ):
        """
        Args:
            model_path: MinerU 模型路径 (默认自动下载)
            device: cuda | cpu
            parse_method: 解析方法
                - auto: 自动选择最优方法
                - ocr: 优先 OCR
                - txt: 纯文本提取
        """
        self.model_path = model_path
        self.device = device
        self.parse_method = parse_method
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化"""
        if self._initialized:
            return

        try:
            from magic_pdf.data.utils import pdf_extract
            from magic_pdf.model.pdf_extract_v2 import PdfExtractor
            self._pdf_extract = pdf_extract
            self._extractor = PdfExtractor(self.model_path, self.device)
            self._initialized = True
        except ImportError:
            raise ImportError(
                "MinerU is not installed. Install with: pip install magic-pdf\n"
                "Or use alternative parsers: Docling or PaddleOCR"
            )

    def parse(self, doc_path: str) -> list[ContentItem]:
        """解析 PDF 文档"""
        path = Path(doc_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(doc_path)
        else:
            raise ValueError(f"MinerU parser only supports PDF, got: {path.suffix}")

    def _parse_pdf(self, pdf_path: str) -> list[ContentItem]:
        """解析 PDF"""
        self._ensure_initialized()

        try:
            # MinerU 2.0 解析方式
            import json
            from magic_pdf.data.read_api import (
                get_pdf_lines_and_images,
                extract_images_from_pdf,
            )

            # 提取文本行和图像信息
            pdf_info = get_pdf_lines_and_images(pdf_path, self.parse_method)

            content_list = []

            # 处理文本
            for page_idx, page in enumerate(pdf_info.get("pages", [])):
                # 文本块
                for block in page.get("text_blocks", []):
                    content_list.append(ContentItem(
                        type=ContentType.TEXT,
                        page_idx=page_idx,
                        text=block.get("text", ""),
                        metadata={
                            "bbox": block.get("bbox"),
                            "type": block.get("type", "text"),
                        }
                    ))

                # 表格块
                for block in page.get("table_blocks", []):
                    content_list.append(ContentItem(
                        type=ContentType.TABLE,
                        page_idx=page_idx,
                        markdown=block.get("markdown", ""),
                        table_caption=block.get("caption", []),
                        metadata={"bbox": block.get("bbox")}
                    ))

                # 公式块
                for block in page.get("equation_blocks", []):
                    content_list.append(ContentItem(
                        type=ContentType.EQUATION,
                        page_idx=page_idx,
                        latex=block.get("latex", ""),
                        equation_text=block.get("text", ""),
                        metadata={"bbox": block.get("bbox")}
                    ))

                # 图像
                for img in page.get("image_blocks", []):
                    img_path = img.get("path", "")
                    if img_path and os.path.exists(img_path):
                        content_list.append(ContentItem(
                            type=ContentType.IMAGE,
                            page_idx=page_idx,
                            img_path=img_path,
                            caption=img.get("caption", []),
                            footnote=img.get("footnote", []),
                            metadata={"bbox": img.get("bbox")}
                        ))

            return content_list

        except Exception as e:
            # Fallback: 使用基础 PDF 解析
            return self._parse_pdf_fallback(pdf_path)

    def _parse_pdf_fallback(self, pdf_path: str) -> list[ContentItem]:
        """回退方案：使用基础 PDF 解析"""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            content_list = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text()

                if text.strip():
                    content_list.append(ContentItem(
                        type=ContentType.TEXT,
                        page_idx=page_idx,
                        text=text,
                    ))

                # 提取图像
                image_list = page.get_images()
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        tmp.write(image_bytes)
                        img_path = tmp.name

                    content_list.append(ContentItem(
                        type=ContentType.IMAGE,
                        page_idx=page_idx,
                        img_path=img_path,
                        caption=[],
                        footnote=[],
                    ))

            doc.close()
            return content_list

        except ImportError:
            raise ImportError(
                "Neither MinerU nor PyMuPDF is available. "
                "Install with: pip install pymupdf"
            )


# ============================================================================
# Docling Parser
# ============================================================================


class DoclingParser(BaseDocumentParser):
    """
    Docling 文档解析器。

    Docling 是一个轻量级的文档解析工具，支持:
    - PDF (含扫描件 OCR)
    - Office 文档 (DOCX, XLSX, PPTX)
    - 图像

    安装: pip install docling
    """

    def __init__(
        self,
        enable_table_vlm: bool = False,
        enable_equation_vlm: bool = False,
    ):
        """
        Args:
            enable_table_vlm: 使用 VLM 增强表格解析
            enable_equation_vlm: 使用 VLM 增强公式解析
        """
        self.enable_table_vlm = enable_table_vlm
        self.enable_equation_vlm = enable_equation_vlm
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化"""
        if self._initialized:
            return

        try:
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
                WordFormatOption,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.backend.pypdf_backend import PyPdfDocumentBackend

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        backend=PyPdfDocumentBackend,
                    ),
                    InputFormat.DOCX: WordFormatOption(),
                    InputFormat.XLSX: WordFormatOption(),
                    InputFormat.PPTX: WordFormatOption(),
                }
            )
            self._initialized = True

        except ImportError:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling"
            )

    def parse(self, doc_path: str) -> list[ContentItem]:
        """解析文档"""
        self._ensure_initialized()

        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(doc_path)
        elif suffix in [".docx", ".doc"]:
            return self._parse_docx(doc_path)
        elif suffix in [".xlsx", ".xls"]:
            return self._parse_xlsx(doc_path)
        elif suffix in [".pptx", ".ppt"]:
            return self._parse_pptx(doc_path)
        elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            return self._parse_image(doc_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _parse_pdf(self, pdf_path: str) -> list[ContentItem]:
        """解析 PDF"""
        result = self._converter.convert(pdf_path)
        return self._dl_to_content_list(result)

    def _parse_docx(self, docx_path: str) -> list[ContentItem]:
        """解析 DOCX"""
        result = self._converter.convert(docx_path)
        return self._dl_to_content_list(result)

    def _parse_xlsx(self, xlsx_path: str) -> list[ContentItem]:
        """解析 XLSX"""
        result = self._converter.convert(xlsx_path)
        return self._dl_to_content_list(result)

    def _parse_pptx(self, pptx_path: str) -> list[ContentItem]:
        """解析 PPTX"""
        result = self._converter.convert(pptx_path)
        return self._dl_to_content_list(result)

    def _parse_image(self, img_path: str) -> list[ContentItem]:
        """解析图像"""
        result = self._converter.convert(img_path)
        return self._dl_to_content_list(result)

    def _dl_to_content_list(self, dl_result) -> list[ContentItem]:
        """将 Docling 结果转换为 ContentItem 列表"""
        content_list = []

        # 遍历文档结构
        for element in dl_result.document.iterate_items():
            page_idx = element.meta.page or 0

            # 根据元素类型转换
            if element.category == "text":
                content_list.append(ContentItem(
                    type=ContentType.TEXT,
                    page_idx=page_idx,
                    text=element.text,
                ))
            elif element.category == "table":
                content_list.append(ContentItem(
                    type=ContentType.TABLE,
                    page_idx=page_idx,
                    markdown=element.export_to_markdown(),
                ))
            elif element.category == "picture":
                if hasattr(element, "image_path") and element.image_path:
                    content_list.append(ContentItem(
                        type=ContentType.IMAGE,
                        page_idx=page_idx,
                        img_path=element.image_path,
                        caption=getattr(element, "caption", []),
                    ))
            elif element.category == "formula":
                content_list.append(ContentItem(
                    type=ContentType.EQUATION,
                    page_idx=page_idx,
                    latex=getattr(element, "latex", ""),
                    equation_text=element.text,
                ))

        return content_list


# ============================================================================
# PaddleOCR Parser
# ============================================================================


class PaddleOCRParser(BaseDocumentParser):
    """
    PaddleOCR 文档解析器。

    适合快速 OCR 场景，支持:
    - 印刷文本识别
    - 表格识别
    - 多语言

    安装:
        pip install paddlepaddle paddleocr
        # 或仅 CPU 版本:
        # pip install paddlepaddle
    """

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = "en",
        use_gpu: bool = torch.cuda.is_available(),
    ):
        """
        Args:
            use_angle_cls: 使用方向分类器
            lang: 语言 (en, ch, japan, korean, etc.)
            use_gpu: 使用 GPU
        """
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化"""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            self._initialized = True

        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install with:\n"
                "pip install paddlepaddle paddleocr"
            )

    def parse(self, doc_path: str) -> list[ContentItem]:
        """解析文档"""
        self._ensure_initialized()

        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(doc_path)
        elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
            return self._parse_image(doc_path)
        else:
            raise ValueError(f"PaddleOCR parser supports PDF and images, got: {suffix}")

    def _parse_pdf(self, pdf_path: str) -> list[ContentItem]:
        """解析 PDF (逐页 OCR)"""
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF parsing. Install with: pip install pymupdf")

        doc = fitz.open(pdf_path)
        content_list = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            # 渲染为图像
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # 保存临时图像
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                img_path = tmp.name

            # OCR
            items = self._ocr_image(img_path)
            for item in items:
                content_list.append(ContentItem(
                    type=ContentType.TEXT,
                    page_idx=page_idx,
                    text=item["text"],
                    metadata={"bbox": item.get("bbox")},
                ))

            # 清理临时文件
            os.unlink(img_path)

        doc.close()
        return content_list

    def _parse_image(self, img_path: str) -> list[ContentItem]:
        """解析图像"""
        items = self._ocr_image(img_path)
        return [
            ContentItem(
                type=ContentType.TEXT,
                page_idx=0,
                text=item["text"],
                metadata={"bbox": item.get("bbox")},
            )
            for item in items
        ]

    def _ocr_image(self, img_path: str) -> list[dict]:
        """OCR 单张图像"""
        result = self._ocr.ocr(img_path, cls=self.use_angle_cls)

        items = []
        if result and result[0]:
            for line in result[0]:
                bbox, (text, confidence) = line
                items.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence,
                })

        return items


# ============================================================================
# Office Parser (LibreOffice)
# ============================================================================


class LibreOfficeParser(BaseDocumentParser):
    """
    LibreOffice 文档解析器。

    将 Office 文档转换为 PDF，然后使用 PDF 解析器处理。
    需要系统安装 LibreOffice。

    安装:
        # Ubuntu/Debian: sudo apt-get install libreoffice
        # macOS: brew install --cask libreoffice
        # Windows: 下载安装包
    """

    def __init__(
        self,
        pdf_parser: Optional[BaseDocumentParser] = None,
        libreoffice_path: Optional[str] = None,
    ):
        """
        Args:
            pdf_parser: 用于解析转换后 PDF 的解析器
            libreoffice_path: LibreOffice 可执行文件路径
        """
        self.pdf_parser = pdf_parser or PaddleOCRParser()
        self.lo_path = libreoffice_path or self._find_libreoffice()

    def _find_libreoffice(self) -> str:
        """查找 LibreOffice 可执行文件"""
        import shutil

        # 尝试常见路径
        candidates = [
            "/usr/bin/libreoffice",
            "/usr/bin/soffice",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        # 尝试从 PATH 查找
        path = shutil.which("libreoffice") or shutil.which("soffice")
        if path:
            return path

        raise RuntimeError(
            "LibreOffice is not installed or not found in PATH.\n"
            "Install from: https://www.libreoffice.org/download/download/"
        )

    def parse(self, doc_path: str) -> list[ContentItem]:
        """解析 Office 文档"""
        path = Path(doc_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        suffix = path.suffix.lower()
        if suffix not in [".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]:
            raise ValueError(f"Unsupported Office format: {suffix}")

        # 转换为 PDF
        pdf_path = self._convert_to_pdf(doc_path)

        try:
            # 解析 PDF
            return self.pdf_parser.parse(pdf_path)
        finally:
            # 清理临时 PDF
            if pdf_path.endswith(".pdf") and ".ragtmp" in pdf_path:
                os.unlink(pdf_path)

    def _convert_to_pdf(self, doc_path: str) -> str:
        """将 Office 文档转换为 PDF"""
        doc_path = os.path.abspath(doc_path)
        output_dir = tempfile.mkdtemp(prefix="ragtmp_")
        output_base = os.path.join(output_dir, Path(doc_path).stem)

        cmd = [
            self.lo_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", output_dir,
            doc_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed: {result.stderr}"
            )

        pdf_path = output_base + ".pdf"
        if not os.path.exists(pdf_path):
            raise RuntimeError(
                f"PDF was not created at {pdf_path}"
            )

        return pdf_path


# ============================================================================
# Unified Multimodal Document Parser
# ============================================================================


class MultimodalDocumentParser:
    """
    统一的多模态文档解析器。

    自动选择最佳解析器，支持多种文档格式。

    Usage:
        parser = MultimodalDocumentParser(
            parser_type="mineru",  # mineru | docling | paddleocr
            enable_image=True,
            enable_table=True,
            enable_equation=True,
        )

        content_list = parser.parse("document.pdf")
    """

    def __init__(
        self,
        parser_type: str = "auto",
        enable_image: bool = True,
        enable_table: bool = True,
        enable_equation: bool = True,
        vlm_model: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
    ):
        """
        Args:
            parser_type: 解析器类型
                - auto: 自动选择 (优先 MinerU, 其次 Docling, 最后 PaddleOCR)
                - mineru: MinerU (高保真 PDF)
                - docling: Docling (轻量, 支持 Office)
                - paddleocr: PaddleOCR (快速 OCR)
            enable_image: 启用图像处理
            enable_table: 启用表格处理
            enable_equation: 启用公式处理
            vlm_model: VLM 模型名 (用于图像描述生成)
            vlm_api_key: VLM API key
            vlm_base_url: VLM API base URL
        """
        self.parser_type = ParserType(parser_type)
        self.enable_image = enable_image
        self.enable_table = enable_table
        self.enable_equation = enable_equation
        self.vlm_model = vlm_model
        self.vlm_api_key = vlm_api_key
        self.vlm_base_url = vlm_base_url

        # 初始化解析器
        self._parsers: dict[str, BaseDocumentParser] = {}
        self._office_parser: Optional[LibreOfficeParser] = None

    def parse(self, doc_path: str) -> list[ContentItem]:
        """
        解析文档，返回 content_list。

        Args:
            doc_path: 文档路径

        Returns:
            list of ContentItem
        """
        path = Path(doc_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        suffix = path.suffix.lower()

        # 图像文件
        if suffix in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]:
            if self.enable_image:
                return self._parse_image(path)
            else:
                return []

        # Office 文档
        if suffix in [".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]:
            return self._parse_office(path)

        # PDF
        if suffix == ".pdf":
            return self._parse_pdf(path)

        # 文本文件
        if suffix in [".txt", ".md", ".csv"]:
            return self._parse_text(path)

        raise ValueError(f"Unsupported file format: {suffix}")

    def _get_parser(self, parser_type: Optional[str] = None) -> BaseDocumentParser:
        """获取指定类型的解析器"""
        ptype = parser_type or self.parser_type.value

        if ptype not in self._parsers:
            if ptype == "mineru" or ptype == ParserType.MINERU.value:
                self._parsers[ptype] = MinerUPreviewParser()
            elif ptype == "docling" or ptype == ParserType.DOCLING.value:
                self._parsers[ptype] = DoclingParser()
            elif ptype == "paddleocr" or ptype == ParserType.PADDLEOCR.value:
                self._parsers[ptype] = PaddleOCRParser()
            else:
                raise ValueError(f"Unknown parser type: {ptype}")

        return self._parsers[ptype]

    def _parse_pdf(self, pdf_path: Path) -> list[ContentItem]:
        """解析 PDF"""
        # 尝试自动选择最佳解析器
        for parser_name in ["mineru", "docling", "paddleocr"]:
            try:
                parser = self._get_parser(parser_name)
                return parser.parse(str(pdf_path))
            except ImportError:
                continue
            except Exception as e:
                # 尝试下一个解析器
                continue

        raise RuntimeError(
            "No suitable PDF parser available. "
            "Install at least one: magic-pdf, docling, or pymupdf+paddleocr"
        )

    def _parse_office(self, office_path: Path) -> list[ContentItem]:
        """解析 Office 文档"""
        if self._office_parser is None:
            self._office_parser = LibreOfficeParser()

        return self._office_parser.parse(str(office_path))

    def _parse_image(self, img_path: Path) -> list[ContentItem]:
        """解析图像"""
        content_list = []

        # OCR 提取文本
        try:
            ocr_parser = PaddleOCRParser()
            text_items = ocr_parser.parse(str(img_path))
            for item in text_items:
                item.page_idx = 0
                content_list.append(item)
        except ImportError:
            pass

        # VLM 生成描述
        if self.enable_image and self.vlm_model:
            caption = self._vlm_caption(str(img_path))
            if caption:
                content_list.append(ContentItem(
                    type=ContentType.IMAGE,
                    page_idx=0,
                    img_path=str(img_path),
                    caption=[caption],
                ))

        if not content_list:
            # 至少返回一个图像项
            content_list.append(ContentItem(
                type=ContentType.IMAGE,
                page_idx=0,
                img_path=str(img_path),
            ))

        return content_list

    def _parse_text(self, text_path: Path) -> list[ContentItem]:
        """解析文本文件"""
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        return [ContentItem(
            type=ContentType.TEXT,
            page_idx=0,
            text=text,
        )]

    def _vlm_caption(self, img_path: str) -> Optional[str]:
        """使用 VLM 生成图像描述"""
        if not self.vlm_model:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.vlm_api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self.vlm_base_url or os.environ.get("OPENAI_API_BASE"),
            )

            with open(img_path, "rb") as f:
                response = client.chat.completions.create(
                    model=self.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(img_path)}"}},
                                {"type": "text", "text": "Describe this image briefly in one sentence."},
                            ],
                        }
                    ],
                    max_tokens=256,
                )

            return response.choices[0].message.content

        except Exception:
            return None

    def _image_to_base64(self, img_path: str) -> str:
        """将图像转换为 base64"""
        import base64

        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def from_content_list(
        content_list: list[dict],
    ) -> list[ContentItem]:
        """
        从字典列表创建 ContentItem 列表。

        用于直接插入预解析内容，绕过解析阶段。

        Args:
            content_list: [
                {"type": "text", "text": "...", "page_idx": 0},
                {"type": "image", "img_path": "...", "caption": ["..."], "page_idx": 1},
                ...
            ]

        Returns:
            list of ContentItem
        """
        return [ContentItem.from_dict(item) for item in content_list]
