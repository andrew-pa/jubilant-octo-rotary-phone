from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class PromptBlock:
    """A single identified text block within the system prompt."""

    identifier: str
    text: str


DEFAULT_PROMPT_BLOCKS: Sequence[PromptBlock] = (
    PromptBlock(
        identifier="core-guidance",
        text=(
            "You are a helpful AI assistant. Keep responses concise unless the user "
            "asks for detail."
        ),
    ),
)


class PromptStoreError(RuntimeError):
    """Raised when prompt block operations fail."""


@dataclass
class SystemPromptStore:
    """Manages loading, persisting, and formatting the system prompt blocks."""

    path: Path

    def ensure_exists(self) -> None:
        if not self.path.exists():
            self.save_blocks(DEFAULT_PROMPT_BLOCKS)

    def get_blocks(self) -> List[PromptBlock]:
        self.ensure_exists()
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            blocks = list(DEFAULT_PROMPT_BLOCKS)
            self.save_blocks(blocks)
            return blocks
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Legacy plain-text prompt: migrate into a single block.
            legacy_text = raw.rstrip("\n")
            if not legacy_text:
                blocks = list(DEFAULT_PROMPT_BLOCKS)
            else:
                blocks = [PromptBlock(identifier="legacy-1", text=legacy_text)]
            self.save_blocks(blocks)
            return blocks
        blocks = self._blocks_from_payload(data)
        return blocks

    def save_blocks(self, blocks: Sequence[PromptBlock]) -> None:
        serialized = {
            "blocks": [
                {"id": block.identifier, "text": block.text} for block in blocks
            ]
        }
        self.path.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def read(self) -> str:
        blocks = self.get_blocks()
        return self.render_blocks(blocks)

    def render_blocks(self, blocks: Sequence[PromptBlock]) -> str:
        if not blocks:
            return "System Prompt Blocks: (empty)\n"
        lines: List[str] = ["System Prompt Blocks (in order):"]
        for block in blocks:
            header = f"{block.identifier}. "
            lines.append(header)
            content_lines = block.text.splitlines() or [""]
            for line in content_lines:
                if line:
                    lines.append(f"  {line}")
                else:
                    lines.append("  ")
            lines.append("")  # blank line between blocks
        return "\n".join(lines).rstrip() + "\n"

    def replace_block(self, *, identifier: str, text: str) -> List[PromptBlock]:
        if not identifier:
            raise PromptStoreError("Block identifier must be a non-empty string.")
        blocks = self.get_blocks()
        for idx, block in enumerate(blocks):
            if block.identifier == identifier:
                blocks[idx] = PromptBlock(identifier=identifier, text=text)
                self.save_blocks(blocks)
                return blocks
        raise PromptStoreError(f"Block '{identifier}' was not found.")

    def append_block(self, *, identifier: str, text: str) -> List[PromptBlock]:
        if not identifier:
            raise PromptStoreError("Block identifier must be a non-empty string.")
        blocks = self.get_blocks()
        if any(block.identifier == identifier for block in blocks):
            raise PromptStoreError(f"Block '{identifier}' already exists.")
        blocks.append(PromptBlock(identifier=identifier, text=text))
        self.save_blocks(blocks)
        return blocks

    def delete_block(self, *, identifier: str) -> List[PromptBlock]:
        if not identifier:
            raise PromptStoreError("Block identifier must be a non-empty string.")
        blocks = self.get_blocks()
        new_blocks = [block for block in blocks if block.identifier != identifier]
        if len(new_blocks) == len(blocks):
            raise PromptStoreError(f"Block '{identifier}' was not found.")
        self.save_blocks(new_blocks)
        return new_blocks

    def _blocks_from_payload(self, payload: object) -> List[PromptBlock]:
        if not isinstance(payload, dict):
            raise PromptStoreError("System prompt file must contain a JSON object.")
        blocks_raw = payload.get("blocks")
        if not isinstance(blocks_raw, list):
            raise PromptStoreError("System prompt file must contain a 'blocks' list.")
        blocks: List[PromptBlock] = []
        seen_ids: set[str] = set()
        for entry in blocks_raw:
            if not isinstance(entry, dict):
                raise PromptStoreError("Each block entry must be an object.")
            identifier = entry.get("id")
            text = entry.get("text", "")
            if not isinstance(identifier, str) or not identifier.strip():
                raise PromptStoreError("Block 'id' must be a non-empty string.")
            if identifier in seen_ids:
                raise PromptStoreError(f"Duplicate block id '{identifier}' detected.")
            if not isinstance(text, str):
                raise PromptStoreError(f"Block '{identifier}' has non-string text.")
            blocks.append(PromptBlock(identifier=identifier, text=text))
            seen_ids.add(identifier)
        return blocks
