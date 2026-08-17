import pytest

from rag_system.chunking.tokenizer import ApproximateTokenCounter
from rag_system.core.config import ChunkingConfig


def test_split_windows_has_explicit_overlap() -> None:
    counter = ApproximateTokenCounter()
    text = "A B C D E F G"

    windows = counter.split_windows(text, max_tokens=4, overlap_tokens=2)

    assert windows == ["A B C D", "C D E F", "E F G"]
    assert all(counter.count(window) <= 4 for window in windows)


def test_tail_returns_requested_number_of_tokens() -> None:
    counter = ApproximateTokenCounter()

    assert counter.tail("A B C D E", 3) == "C D E"
    assert counter.tail("A B", 0) == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens": 16, "overlap_tokens": 16},
        {"target_tokens": 16, "overlap_tokens": 1, "min_chunk_tokens": 17},
    ],
)
def test_invalid_chunking_limits_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)
