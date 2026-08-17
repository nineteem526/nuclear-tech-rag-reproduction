from pydantic import BaseModel, Field, model_validator


class PdfParserConfig(BaseModel):
    """Configurable, deterministic PDF heading heuristics."""

    heading_max_chars: int = Field(default=120, ge=10)
    heading_size_ratio: float = Field(default=1.20, gt=1.0)
    bold_heading_size_ratio: float = Field(default=1.05, gt=0.0)


class ChunkingConfig(BaseModel):
    """Baseline reproduction parameters, not original-project facts."""

    target_tokens: int = Field(default=512, ge=16)
    overlap_tokens: int = Field(default=80, ge=0)
    min_chunk_tokens: int = Field(default=80, ge=1)

    @model_validator(mode="after")
    def validate_token_limits(self) -> "ChunkingConfig":
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        if self.min_chunk_tokens > self.target_tokens:
            raise ValueError("min_chunk_tokens must not exceed target_tokens")
        return self

