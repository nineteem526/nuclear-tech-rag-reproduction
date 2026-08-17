from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from rag_system.chunking.section_chunker import SectionAwareChunker
from rag_system.core.config import ChunkingConfig
from rag_system.core.exceptions import RagPipelineError
from rag_system.parsing.pdf import PdfParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-pdf",
        description="Parse a text-based PDF into DocumentBlocks and Chunks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    parse_chunk = subparsers.add_parser("parse-chunk")
    parse_chunk.add_argument("pdf", type=Path)
    parse_chunk.add_argument("--document-id")
    parse_chunk.add_argument("--target-tokens", type=int, default=512)
    parse_chunk.add_argument("--overlap-tokens", type=int, default=80)
    parse_chunk.add_argument("--min-chunk-tokens", type=int, default=80)
    parse_chunk.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = PdfParser().parse(args.pdf, document_id=args.document_id)
        chunker = SectionAwareChunker(
            ChunkingConfig(
                target_tokens=args.target_tokens,
                overlap_tokens=args.overlap_tokens,
                min_chunk_tokens=args.min_chunk_tokens,
            )
        )
        chunks = chunker.chunk(document)
        payload = {
            "summary": {
                "document_id": document.document_id,
                "document_version": document.document_version,
                "page_count": document.page_count,
                "block_count": len(document.blocks),
                "chunk_count": len(chunks),
            },
            "document": document.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
            print(f"Wrote parse result to {args.output}")
        else:
            print(serialized)
        return 0
    except RagPipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 2

