from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BlockType(StrEnum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST = "list"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


class DocumentBlock(BaseModel):
    """A readable-order text block with complete source provenance."""

    model_config = ConfigDict(frozen=True)

    block_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    block_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    content: str = Field(min_length=1)
    block_type: BlockType = BlockType.UNKNOWN
    heading_level: int | None = Field(default=None, ge=1, le=6)
    section_path: list[str] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    font_size: float | None = Field(default=None, gt=0)
    is_bold: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Result of parsing one immutable PDF file version."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    file_type: str = "pdf"
    source_path: str = Field(min_length=1)
    file_hash: str = Field(min_length=64, max_length=64)
    page_count: int = Field(ge=1)
    blocks: list[DocumentBlock]
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """The single source unit passed to later indexing and retrieval stages."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    file_type: str = "pdf"
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    token_count: int = Field(ge=1)
    content_hash: str = Field(min_length=64, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

