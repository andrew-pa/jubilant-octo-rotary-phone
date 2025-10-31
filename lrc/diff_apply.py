from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class UnifiedDiffHunk:
    src_start: int
    src_length: int
    dst_start: int
    dst_length: int
    lines: Sequence[Tuple[str, str]]


class DiffApplyError(RuntimeError):
    """Raised when applying a diff fails."""


def parse_unified_diff(diff_text: str) -> Sequence[UnifiedDiffHunk]:
    lines = diff_text.splitlines()
    hunks: List[UnifiedDiffHunk] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("@@"):
            match = _HUNK_RE.match(line)
            if match is None:
                raise DiffApplyError(f"Malformed hunk header: {line}")
            src_start = int(match.group(1))
            src_len = int(match.group(2) or "1")
            dst_start = int(match.group(3))
            dst_len = int(match.group(4) or "1")
            idx += 1
            payload: List[Tuple[str, str]] = []
            while idx < len(lines):
                current = lines[idx]
                if current.startswith("@@"):
                    break
                if current.startswith("---") or current.startswith("+++"):
                    break
                if current.startswith("\\"):
                    idx += 1
                    continue
                if not current:
                    raise DiffApplyError("Unexpected empty line in diff hunk")
                tag = current[0]
                if tag not in {" ", "+", "-"}:
                    raise DiffApplyError(f"Unexpected tag '{tag}' in diff hunk")
                payload.append((tag, current[1:]))
                idx += 1
            hunks.append(
                UnifiedDiffHunk(
                    src_start=src_start,
                    src_length=src_len,
                    dst_start=dst_start,
                    dst_length=dst_len,
                    lines=tuple(payload),
                )
            )
        else:
            idx += 1
    if not hunks:
        raise DiffApplyError("No hunks found in diff text")
    return tuple(hunks)


def apply_unified_diff(source_text: str, diff_text: str) -> str:
    source_lines = source_text.splitlines()
    source_has_trailing_newline = source_text.endswith("\n")
    hunks = parse_unified_diff(diff_text)
    result_lines: List[str] = []
    cursor = 0
    for hunk in hunks:
        start_index = hunk.src_start - 1
        if start_index < cursor:
            raise DiffApplyError("Overlapping diff hunks detected")
        result_lines.extend(source_lines[cursor:start_index])
        cursor = start_index
        for tag, text in hunk.lines:
            if tag == " ":
                if cursor >= len(source_lines):
                    raise DiffApplyError("Context line exceeds source length")
                if source_lines[cursor] != text:
                    raise DiffApplyError(
                        "Context line mismatch while applying diff"
                    )
                result_lines.append(source_lines[cursor])
                cursor += 1
            elif tag == "-":
                if cursor >= len(source_lines):
                    raise DiffApplyError("Removal exceeds source length")
                if source_lines[cursor] != text:
                    raise DiffApplyError("Removal line mismatch while applying diff")
                cursor += 1
            elif tag == "+":
                result_lines.append(text)
            else:  # pragma: no cover - defensive branch
                raise DiffApplyError(f"Unhandled diff tag: {tag}")
        expected_cursor = hunk.src_start - 1 + hunk.src_length
        if cursor != expected_cursor:
            # Rely on diff payload to advance cursor correctly
            cursor = expected_cursor
    result_lines.extend(source_lines[cursor:])
    result = "\n".join(result_lines)
    if source_has_trailing_newline and not result.endswith("\n"):
        result += "\n"
    return result
