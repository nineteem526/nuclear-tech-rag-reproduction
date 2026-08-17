from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from rag_system.chunking.tokenizer import ApproximateTokenCounter, TokenCounter
from rag_system.core.config import ChunkingConfig
from rag_system.core.exceptions import ChunkingError, ErrorCode
from rag_system.domain.documents import Chunk, DocumentBlock, ParsedDocument


_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9])"
)


@dataclass(frozen=True)
class _Piece:
    content: str
    page_number: int
    block_ids: tuple[str, ...]
    section_path: tuple[str, ...]
    overlap_tokens: int = 0


class SectionAwareChunker:
    """Merge small blocks and split oversized blocks without crossing sections."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.token_counter = token_counter or ApproximateTokenCounter()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not document.blocks:
            raise ChunkingError(
                ErrorCode.CHUNK_INPUT_EMPTY,
                "Parsed document contains no blocks.",
                details={"document_id": document.document_id},
            )

        assembled: list[list[_Piece]] = []
        current: list[_Piece] = []
        current_section: tuple[str, ...] | None = None

        for block in document.blocks:
            block_section = tuple(block.section_path)
            if current and block_section != current_section:
                assembled.append(current)
                current = []
            current_section = block_section

            for piece in self._pieces_for_block(block):
                piece_tokens = self.token_counter.count(piece.content)
                current_tokens = self._piece_token_count(current)
                if current and current_tokens + piece_tokens > self.config.target_tokens:
                    previous = current
                    assembled.append(previous)
                    overlap_budget = min(
                        self.config.overlap_tokens,
                        max(0, self.config.target_tokens - piece_tokens),
                    )
                    current = self._tail_pieces(previous, overlap_budget)
                current.append(piece)

        if current:
            assembled.append(current)

        chunks = [
            self._to_chunk(document, chunk_index, pieces)
            for chunk_index, pieces in enumerate(assembled)
            if self._piece_token_count(pieces) > 0
        ]
        if not chunks:
            raise ChunkingError(
                ErrorCode.CHUNK_RESULT_EMPTY,
                "Chunking produced no non-empty chunks.",
                details={"document_id": document.document_id},
            )
        return chunks

    def _pieces_for_block(self, block: DocumentBlock) -> list[_Piece]:
        token_count = self.token_counter.count(block.content)
        base = _Piece(
            content=block.content,
            page_number=block.page_number,
            block_ids=(block.block_id,),
            section_path=tuple(block.section_path),
        )
        if token_count <= self.config.target_tokens:
            return [base]

        pieces: list[_Piece] = []
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(block.content)
            if sentence.strip()
        ]
        sentence_buffer: list[str] = []
        sentence_tokens = 0

        def flush_sentence_buffer() -> None:
            nonlocal sentence_buffer, sentence_tokens
            if sentence_buffer:
                pieces.append(
                    replace(base, content=" ".join(sentence_buffer).strip())
                )
                sentence_buffer = []
                sentence_tokens = 0

        for sentence in sentences or [block.content]:
            count = self.token_counter.count(sentence)
            if count > self.config.target_tokens:
                flush_sentence_buffer()
                windows = self.token_counter.split_windows(
                    sentence,
                    max_tokens=self.config.target_tokens,
                    overlap_tokens=self.config.overlap_tokens,
                )
                for window_index, window in enumerate(windows):
                    pieces.append(
                        replace(
                            base,
                            content=window,
                            overlap_tokens=(
                                0
                                if window_index == 0
                                else min(
                                    self.config.overlap_tokens,
                                    self.token_counter.count(window),
                                )
                            ),
                        )
                    )
            elif sentence_buffer and (
                sentence_tokens + count > self.config.target_tokens
            ):
                flush_sentence_buffer()
                sentence_buffer.append(sentence)
                sentence_tokens = count
            else:
                sentence_buffer.append(sentence)
                sentence_tokens += count
        flush_sentence_buffer()
        return pieces

    def _tail_pieces(self, pieces: list[_Piece], token_budget: int) -> list[_Piece]:
        if token_budget <= 0:
            return []
        selected: list[_Piece] = []
        remaining = token_budget
        for piece in reversed(pieces):
            count = self.token_counter.count(piece.content)
            if count <= remaining:
                selected.append(replace(piece, overlap_tokens=count))
                remaining -= count
            else:
                tail = self.token_counter.tail(piece.content, remaining)
                if tail:
                    selected.append(
                        replace(
                            piece,
                            content=tail,
                            overlap_tokens=self.token_counter.count(tail),
                        )
                    )
                break
            if remaining == 0:
                break
        return list(reversed(selected))

    def _to_chunk(
        self,
        document: ParsedDocument,
        chunk_index: int,
        pieces: list[_Piece],
    ) -> Chunk:
        content = "\n\n".join(piece.content for piece in pieces).strip()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunk_seed = (
            f"{document.document_id}:{document.document_version}:"
            f"{chunk_index}:{content_hash}"
        )
        chunk_id = "chk_" + hashlib.sha256(
            chunk_seed.encode("utf-8")
        ).hexdigest()[:24]
        source_block_ids = list(
            dict.fromkeys(
                block_id for piece in pieces for block_id in piece.block_ids
            )
        )
        overlap_token_count = sum(piece.overlap_tokens for piece in pieces)
        token_count = self.token_counter.count(content)

        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            document_version=document.document_version,
            chunk_index=chunk_index,
            content=content,
            file_name=document.file_name,
            file_type=document.file_type,
            page_start=min(piece.page_number for piece in pieces),
            page_end=max(piece.page_number for piece in pieces),
            section_path=list(pieces[0].section_path),
            token_count=token_count,
            content_hash=content_hash,
            metadata={
                "source_block_ids": source_block_ids,
                "tokenizer": self.token_counter.name,
                "target_tokens": self.config.target_tokens,
                "configured_overlap_tokens": self.config.overlap_tokens,
                "overlap_token_count": overlap_token_count,
                "below_min_chunk_tokens": token_count
                < self.config.min_chunk_tokens,
            },
        )

    def _piece_token_count(self, pieces: list[_Piece]) -> int:
        return sum(self.token_counter.count(piece.content) for piece in pieces)

