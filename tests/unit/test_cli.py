import json
from pathlib import Path

from rag_system.cli import main


def test_cli_writes_parse_and_chunk_result(
    structured_pdf: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"

    exit_code = main(
        [
            "parse-chunk",
            str(structured_pdf),
            "--target-tokens",
            "40",
            "--overlap-tokens",
            "8",
            "--min-chunk-tokens",
            "5",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["summary"]["block_count"] == 4
    assert payload["summary"]["chunk_count"] == 2


def test_cli_returns_stable_error_for_missing_pdf(tmp_path: Path) -> None:
    exit_code = main(["parse-chunk", str(tmp_path / "missing.pdf")])

    assert exit_code == 2
