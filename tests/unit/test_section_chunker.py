from rag_system.chunking.section_chunker import SectionAwareChunker
from rag_system.chunking.tokenizer import ApproximateTokenCounter
from rag_system.core.config import ChunkingConfig
from rag_system.domain.documents import (
    BlockType,
    DocumentBlock,
    ParsedDocument,
)


def make_block(
    index: int,
    content: str,
    *,
    page: int = 1,
    section: list[str] | None = None,
    block_type: BlockType = BlockType.PARAGRAPH,
) -> DocumentBlock:
    return DocumentBlock(
        block_id=f"block_{index}",
        document_id="doc_test",
        document_version="a" * 64,
        block_index=index,
        page_number=page,
        content=content,
        block_type=block_type,
        section_path=section or [],
    )


def make_document(blocks: list[DocumentBlock]) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc_test",
        document_version="a" * 64,
        file_name="test.pdf",
        source_path="test.pdf",
        file_hash="a" * 64,
        page_count=max(block.page_number for block in blocks),
        blocks=blocks,
    )


def test_small_blocks_merge_and_keep_page_range() -> None:
    section = ["1 系统说明"]
    document = make_document(
        [
            make_block(0, "1 系统说明", section=section, block_type=BlockType.TITLE),
            make_block(1, "主泵负责冷却剂循环。", section=section),
            make_block(2, "设备编号为 RCP-001。", page=2, section=section),
        ]
    )
    chunker = SectionAwareChunker(
        ChunkingConfig(target_tokens=40, overlap_tokens=5, min_chunk_tokens=5)
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].section_path == section
    assert chunks[0].metadata["source_block_ids"] == [
        "block_0",
        "block_1",
        "block_2",
    ]


def test_oversized_block_splits_and_adds_overlap() -> None:
    text = "。".join(f"第{i}项要求保持系统安全" for i in range(30)) + "。"
    document = make_document(
        [make_block(0, text, section=["2 安全要求"])]
    )
    counter = ApproximateTokenCounter()
    chunker = SectionAwareChunker(
        ChunkingConfig(target_tokens=30, overlap_tokens=6, min_chunk_tokens=5),
        token_counter=counter,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 30 for chunk in chunks)
    assert any(
        chunk.metadata["overlap_token_count"] > 0 for chunk in chunks[1:]
    )


def test_overlap_never_crosses_section_boundary() -> None:
    document = make_document(
        [
            make_block(0, "甲章节中的内容足够形成一个独立块。", section=["甲章节"]),
            make_block(1, "乙章节必须从一个全新的块开始。", section=["乙章节"]),
        ]
    )
    chunker = SectionAwareChunker(
        ChunkingConfig(target_tokens=20, overlap_tokens=5, min_chunk_tokens=1)
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].section_path == ["甲章节"]
    assert chunks[1].section_path == ["乙章节"]
    assert chunks[1].metadata["overlap_token_count"] == 0
    assert "甲章节" not in chunks[1].content


def test_chunk_ids_are_deterministic() -> None:
    document = make_document(
        [make_block(0, "稳定的输入应产生稳定的 Chunk ID。", section=["测试"])]
    )
    chunker = SectionAwareChunker(
        ChunkingConfig(target_tokens=32, overlap_tokens=4, min_chunk_tokens=1)
    )

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_technical_identifier_is_counted_as_one_token() -> None:
    counter = ApproximateTokenCounter()

    # Four Chinese characters + two protected identifiers + punctuation.
    assert counter.count("设备 RCP-001 触发 ALM-204。") == 7

