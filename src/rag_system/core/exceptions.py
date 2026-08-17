from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    PDF_NOT_FOUND = "PDF_NOT_FOUND"
    PDF_INVALID_TYPE = "PDF_INVALID_TYPE"
    PDF_PASSWORD_REQUIRED = "PDF_PASSWORD_REQUIRED"
    PDF_NO_EXTRACTABLE_TEXT = "PDF_NO_EXTRACTABLE_TEXT"
    PDF_PARSE_FAILED = "PDF_PARSE_FAILED"
    CHUNK_INPUT_EMPTY = "CHUNK_INPUT_EMPTY"
    CHUNK_RESULT_EMPTY = "CHUNK_RESULT_EMPTY"


class RagPipelineError(Exception):
    """Base exception with an API- and trace-friendly stable error contract."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class PdfNotFoundError(RagPipelineError):
    def __init__(self, path: str) -> None:
        super().__init__(
            ErrorCode.PDF_NOT_FOUND,
            "PDF file does not exist.",
            details={"path": path},
        )


class PdfInvalidTypeError(RagPipelineError):
    def __init__(self, path: str) -> None:
        super().__init__(
            ErrorCode.PDF_INVALID_TYPE,
            "Only PDF files are supported in this phase.",
            details={"path": path},
        )


class PdfPasswordRequiredError(RagPipelineError):
    def __init__(self, path: str) -> None:
        super().__init__(
            ErrorCode.PDF_PASSWORD_REQUIRED,
            "PDF is encrypted and requires a password.",
            details={"path": path},
        )


class PdfNoExtractableTextError(RagPipelineError):
    def __init__(self, path: str) -> None:
        super().__init__(
            ErrorCode.PDF_NO_EXTRACTABLE_TEXT,
            "PDF contains no extractable text; OCR is outside the current phase.",
            details={"path": path},
        )


class PdfParseError(RagPipelineError):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            ErrorCode.PDF_PARSE_FAILED,
            "PDF parsing failed.",
            details={"path": path, "reason": reason},
        )


class ChunkingError(RagPipelineError):
    pass

