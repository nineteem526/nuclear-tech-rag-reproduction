from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf

from rag_system.core.config import PdfParserConfig
from rag_system.core.exceptions import (
    PdfInvalidTypeError,
    PdfNoExtractableTextError,
    PdfNotFoundError,
    PdfParseError,
    PdfPasswordRequiredError,
)
from rag_system.domain.documents import (
    BlockType,
    BoundingBox,
    DocumentBlock,
    ParsedDocument,
)


_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)[\.、\s]+\S+")
_CHINESE_HEADING = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百零〇0-9]+[章节篇]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）)"
)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[)）])\s+")


@dataclass(frozen=True)
class _RawBlock:
    page_number: int
    content: str
    bbox: tuple[float, float, float, float]
    max_font_size: float
    font_sizes: tuple[float, ...]
    is_bold: bool
    line_count: int


class PdfParser:
    """Extract ordered text blocks from text-based PDFs using PyMuPDF."""

    def __init__(self, config: PdfParserConfig | None = None) -> None:
        self.config = config or PdfParserConfig()

    def parse(
        self,
        path: str | Path,
        *,
        document_id: str | None = None,
    ) -> ParsedDocument:
        pdf_path = Path(path).expanduser()
        if not pdf_path.exists() or not pdf_path.is_file():
            raise PdfNotFoundError(str(pdf_path))
        if pdf_path.suffix.lower() != ".pdf":
            raise PdfInvalidTypeError(str(pdf_path))

        resolved_path = pdf_path.resolve()
        file_hash = self._sha256_file(resolved_path)
        stable_document_id = document_id or self._default_document_id(resolved_path)

        try:
            document = pymupdf.open(resolved_path)
        except Exception as exc:
            raise PdfParseError(str(resolved_path), str(exc)) from exc

        try:
            if document.needs_pass:
                raise PdfPasswordRequiredError(str(resolved_path))
            if document.page_count < 1:
                raise PdfNoExtractableTextError(str(resolved_path))

            page_count = document.page_count
            raw_blocks: list[_RawBlock] = []
            blank_pages: list[int] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_blocks = self._extract_page_blocks(page, page_index + 1)
                if not page_blocks:
                    blank_pages.append(page_index + 1)
                raw_blocks.extend(page_blocks)
        except (PdfPasswordRequiredError, PdfNoExtractableTextError):
            raise
        except Exception as exc:
            raise PdfParseError(str(resolved_path), str(exc)) from exc
        finally:
            document.close()

        if not raw_blocks:
            raise PdfNoExtractableTextError(str(resolved_path))

        body_font_size = self._estimate_body_font_size(raw_blocks)
        blocks = self._build_document_blocks(
            raw_blocks,
            document_id=stable_document_id,
            document_version=file_hash,
            body_font_size=body_font_size,
        )
        warnings = []
        if blank_pages:
            warnings.append(
                "Pages with no extractable text: "
                + ", ".join(str(page) for page in blank_pages)
            )

        return ParsedDocument(
            document_id=stable_document_id,
            document_version=file_hash,
            file_name=resolved_path.name,
            source_path=str(resolved_path),
            file_hash=file_hash,
            page_count=page_count,
            blocks=blocks,
            warnings=warnings,
            metadata={
                "parser": "pymupdf",
                "pymupdf_version": pymupdf.VersionBind,
                "body_font_size": body_font_size,
            },
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as pdf_file:
            for data in iter(lambda: pdf_file.read(1024 * 1024), b""):
                digest.update(data)
        return digest.hexdigest()

    @staticmethod
    def _default_document_id(path: Path) -> str:
        source_key = path.as_uri().lower()
        return f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, source_key).hex}"

    @staticmethod
    def _extract_page_blocks(page: pymupdf.Page, page_number: int) -> list[_RawBlock]:
        page_dict: dict[str, Any] = page.get_text("dict", sort=True)
        extracted: list[_RawBlock] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: list[str] = []
            font_sizes: list[float] = []
            bold = False
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(str(span.get("text", "")) for span in spans)
                line_text = re.sub(r"[\t\u00a0 ]+", " ", line_text).strip()
                if line_text:
                    lines.append(line_text)
                for span in spans:
                    span_text = str(span.get("text", ""))
                    if not span_text.strip():
                        continue
                    font_sizes.append(float(span.get("size", 0.0)))
                    flags = int(span.get("flags", 0))
                    font_name = str(span.get("font", "")).lower()
                    bold = bold or bool(flags & 16) or "bold" in font_name

            content = "\n".join(lines).strip()
            if not content or not font_sizes:
                continue
            bbox_values = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            bbox = tuple(float(value) for value in bbox_values)
            extracted.append(
                _RawBlock(
                    page_number=page_number,
                    content=content,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    max_font_size=max(font_sizes),
                    font_sizes=tuple(font_sizes),
                    is_bold=bold,
                    line_count=len(lines),
                )
            )
        return extracted

    @staticmethod
    def _estimate_body_font_size(blocks: list[_RawBlock]) -> float:
        sizes = [size for block in blocks for size in block.font_sizes if size > 0]
        return float(median(sizes)) if sizes else 10.0

    def _build_document_blocks(
        self,
        raw_blocks: list[_RawBlock],
        *,
        document_id: str,
        document_version: str,
        body_font_size: float,
    ) -> list[DocumentBlock]:
        section_stack: list[str] = []
        result: list[DocumentBlock] = []

        for block_index, raw in enumerate(raw_blocks):
            block_type, heading_level = self._classify_block(raw, body_font_size)
            if block_type == BlockType.TITLE and heading_level is not None:
                section_stack = section_stack[: max(0, heading_level - 1)]
                section_stack.append(raw.content)

            block_id_seed = (
                f"{document_version}:{raw.page_number}:{block_index}:{raw.content}"
            )
            block_id = "blk_" + hashlib.sha256(
                block_id_seed.encode("utf-8")
            ).hexdigest()[:24]
            result.append(
                DocumentBlock(
                    block_id=block_id,
                    document_id=document_id,
                    document_version=document_version,
                    block_index=block_index,
                    page_number=raw.page_number,
                    content=raw.content,
                    block_type=block_type,
                    heading_level=heading_level,
                    section_path=list(section_stack),
                    bbox=BoundingBox(
                        x0=raw.bbox[0],
                        y0=raw.bbox[1],
                        x1=raw.bbox[2],
                        y1=raw.bbox[3],
                    ),
                    font_size=raw.max_font_size,
                    is_bold=raw.is_bold,
                    metadata={"line_count": raw.line_count},
                )
            )
        return result

    def _classify_block(
        self,
        block: _RawBlock,
        body_font_size: float,
    ) -> tuple[BlockType, int | None]:
        text = block.content.replace("\n", " ").strip()
        is_short = len(text) <= self.config.heading_max_chars
        size_ratio = block.max_font_size / max(body_font_size, 0.1)
        numbered_match = _NUMBERED_HEADING.match(text)
        chinese_match = _CHINESE_HEADING.match(text)
        looks_like_heading = is_short and (
            size_ratio >= self.config.heading_size_ratio
            or (
                block.is_bold
                and size_ratio >= self.config.bold_heading_size_ratio
            )
            or numbered_match is not None
            or chinese_match is not None
        )

        if looks_like_heading:
            if numbered_match:
                level = min(6, numbered_match.group(1).count(".") + 1)
            elif chinese_match:
                level = 1 if text.startswith("第") or "、" in text else 2
            elif size_ratio >= 1.70:
                level = 1
            elif size_ratio >= 1.40:
                level = 2
            elif size_ratio >= 1.20:
                level = 3
            else:
                level = 4
            return BlockType.TITLE, level

        if _LIST_PREFIX.match(text):
            return BlockType.LIST, None
        return BlockType.PARAGRAPH, None

