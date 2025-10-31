from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_PROMPT: Final[str] = (
    "You are a helpful AI assistant. Keep responses concise unless the user "
    "asks for detail."
)


@dataclass(frozen=True)
class SystemPromptStore:
    """Manages loading and persisting the system prompt."""

    path: Path

    def ensure_exists(self) -> None:
        if not self.path.exists():
            self.path.write_text(DEFAULT_PROMPT + "\n", encoding="utf-8")

    def read(self) -> str:
        self.ensure_exists()
        return self.path.read_text(encoding="utf-8")

    def write(self, content: str) -> None:
        self.path.write_text(content, encoding="utf-8")
