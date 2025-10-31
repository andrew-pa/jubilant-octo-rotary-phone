from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from patchpy import DiffFile, PatchPyError

_DEFAULT_FILENAME: Final[str] = "patched_file.txt"


class DiffApplyError(RuntimeError):
    """Raised when applying a diff fails."""


def _normalize_diff(diff_text: str, filename: str) -> str:
    if not diff_text.strip():
        raise DiffApplyError("Diff text was empty")
    lines = diff_text.splitlines(keepends=True)
    has_minus_header = any(line.startswith("--- ") for line in lines)
    has_plus_header = any(line.startswith("+++ ") for line in lines)
    if has_minus_header != has_plus_header:
        raise DiffApplyError("Diff text is missing file header lines")
    if has_minus_header and has_plus_header:
        return "".join(lines)
    header = f"--- a/{filename}\n+++ b/{filename}\n"
    return header + "".join(lines)


def apply_unified_diff(
    source_text: str, diff_text: str, *, filename: str | None = None
) -> str:
    placeholder = (filename or _DEFAULT_FILENAME).strip() or _DEFAULT_FILENAME
    normalized_diff = _normalize_diff(diff_text, placeholder)
    try:
        diff_file = DiffFile.from_string(normalized_diff)
    except PatchPyError as exc:  # pragma: no cover
        raise DiffApplyError(f"Unable to parse diff: {exc}") from exc
    if len(diff_file.modifications) != 1:
        raise DiffApplyError("Diff must modify exactly one file")
    modification = diff_file.modifications[0]
    modification.source = placeholder if modification.source is not None else None
    modification.target = placeholder if modification.target is not None else None
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir, placeholder)
        if modification.source is not None:
            temp_path.write_text(source_text, encoding="utf-8")
        try:
            diff_file.apply(root=temp_dir)
        except PatchPyError as exc:  # pragma: no cover
            raise DiffApplyError(f"Failed to apply diff: {exc}") from exc
        if not temp_path.exists():
            raise DiffApplyError("Patched file was not created")
        return temp_path.read_text(encoding="utf-8")
