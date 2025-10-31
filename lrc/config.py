from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ClientConfig:
    """Configuration for connecting to the OpenAI-compatible endpoint."""

    model: str
    temperature: float
    max_tokens: Optional[int]
    base_url: Optional[str]
    api_key: Optional[str]
    organization: Optional[str]
    system_prompt_path: Path

    @staticmethod
    def default_system_prompt_path() -> Path:
        return Path.cwd() / "system_prompt.txt"
