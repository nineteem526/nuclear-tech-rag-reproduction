from __future__ import annotations

import re
from typing import Protocol


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*|[\u3400-\u9fff]|[^\s]",
    re.UNICODE,
)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def split_windows(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> list[str]: ...

    def tail(self, text: str, max_tokens: int) -> str: ...


class ApproximateTokenCounter:
    """Deterministic baseline counter; not a BGE-M3 tokenizer substitute."""

    name = "mixed_language_approx_v1"

    def count(self, text: str) -> int:
        return sum(1 for _ in _TOKEN_PATTERN.finditer(text))

    def split_windows(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> list[str]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")

        spans = list(_TOKEN_PATTERN.finditer(text))
        if not spans:
            return []
        step = max_tokens - overlap_tokens
        windows: list[str] = []
        start = 0
        while start < len(spans):
            end = min(start + max_tokens, len(spans))
            char_start = spans[start].start()
            char_end = spans[end - 1].end()
            window = text[char_start:char_end].strip()
            if window:
                windows.append(window)
            if end == len(spans):
                break
            start += step
        return windows

    def tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        spans = list(_TOKEN_PATTERN.finditer(text))
        if not spans:
            return ""
        start = max(0, len(spans) - max_tokens)
        return text[spans[start].start() : spans[-1].end()].strip()

