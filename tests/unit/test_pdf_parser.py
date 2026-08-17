from pathlib import Path

import pytest

from rag_system.core.exceptions import (
    ErrorCode,
    PdfNoExtractableTextError,
    PdfNotFoundError,
    PdfParseError,
)
from rag_system.domain.documents import BlockType
from rag_system.parsing.pdf import PdfParser


def test_parse_preserves_pages_sections_and_provenance(
    structured_pdf: Path,
) -> None:
    document = PdfParser().parse(structured_pdf, document_id="doc_test")

    assert document.document_id == "doc_test"
    assert document.page_count == 2
    assert len(document.file_hash) == 64
    assert [block.page_number for block in document.blocks] == [1, 1, 2, 2]

    first_heading = document.blocks[0]
    second_heading = document.blocks[2]
    assert first_heading.block_type == BlockType.TITLE
    assert first_heading.heading_level == 1
    assert first_heading.section_path == ["1 Reactor Cooling System"]
    assert second_heading.block_type == BlockType.TITLE
    assert second_heading.heading_level == 2
    assert second_heading.section_path == [
        "1 Reactor Cooling System",
        "1.1 Pump Requirements",
    ]
    assert document.blocks[3].section_path == second_heading.section_path
    assert all(block.bbox is not None for block in document.blocks)


def test_parse_is_deterministic_for_same_file(structured_pdf: Path) -> None:
    parser = PdfParser()
    first = parser.parse(structured_pdf)
    second = parser.parse(structured_pdf)

    assert first.document_id == second.document_id
    assert first.document_version == second.document_version
    assert [block.block_id for block in first.blocks] == [
        block.block_id for block in second.blocks
    ]


def test_missing_pdf_has_stable_error_code(tmp_path: Path) -> None:
    with pytest.raises(PdfNotFoundError) as captured:
        PdfParser().parse(tmp_path / "missing.pdf")

    assert captured.value.code == ErrorCode.PDF_NOT_FOUND


def test_pdf_without_text_requires_future_ocr(empty_pdf: Path) -> None:
    with pytest.raises(PdfNoExtractableTextError) as captured:
        PdfParser().parse(empty_pdf)

    assert captured.value.code == ErrorCode.PDF_NO_EXTRACTABLE_TEXT


def test_corrupt_pdf_is_classified(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(PdfParseError) as captured:
        PdfParser().parse(path)

    assert captured.value.code == ErrorCode.PDF_PARSE_FAILED

