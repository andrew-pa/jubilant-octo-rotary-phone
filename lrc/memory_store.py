from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float32]


@dataclass(frozen=True)
class MemoryRecord:
    """Represents a stored text snippet and its embedding vector."""

    identifier: str
    text: str
    embedding: Vector


class MemoryStoreError(RuntimeError):
    """Raised when the on-disk memory store cannot be parsed."""


def _to_vector(embedding: Iterable[float]) -> Vector:
    if isinstance(embedding, np.ndarray):
        vector = np.asarray(embedding, dtype=np.float32)
    else:
        vector = np.fromiter((float(component) for component in embedding), dtype=np.float32)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1:
        raise ValueError("Embedding vectors must be one-dimensional.")
    return vector


def _cosine_similarities(matrix: Vector, query: Vector) -> NDArray[np.float32]:
    if matrix.size == 0:
        return np.asarray([], dtype=np.float32)
    matrix_2d = matrix if matrix.ndim == 2 else matrix.reshape(1, -1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        return np.zeros(matrix_2d.shape[0], dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix_2d, axis=1)
    dot_products = matrix_2d @ query
    denom = matrix_norms * query_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        similarities = np.divide(
            dot_products,
            denom,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=denom != 0.0,
        )
    return similarities.astype(np.float32, copy=False)


class MemoryStore:
    """A minimal persistent vector store backed by a binary NumPy archive."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: List[MemoryRecord] = []
        self._load()

    def _ensure_path(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        self._ensure_path()
        if not self._path.exists():
            self._records = []
            self._persist()
            return
        try:
            with np.load(self._path, allow_pickle=True) as data:
                ids = data.get("ids")
                texts = data.get("texts")
                embeddings = data.get("embeddings")
                if ids is None or texts is None or embeddings is None:
                    self._records = []
                    return
                ids_list = np.asarray(ids, dtype=object).tolist()
                texts_list = np.asarray(texts, dtype=object).tolist()
                embeddings_array = np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            raise MemoryStoreError(f"Failed to load memory store: {exc}") from exc

        if embeddings_array.ndim == 1 and embeddings_array.size:
            embeddings_array = embeddings_array.reshape(1, -1)
        if len(ids_list) != len(texts_list) or (
            embeddings_array.shape[0] not in {0, len(ids_list)}
        ):
            raise MemoryStoreError("Memory store file is corrupt or inconsistent.")
        self._records = [
            MemoryRecord(
                identifier=str(identifier),
                text=str(text),
                embedding=embeddings_array[idx].astype(np.float32, copy=False),
            )
            for idx, (identifier, text) in enumerate(zip(ids_list, texts_list, strict=False))
        ]

    def _persist(self) -> None:
        ids = np.asarray([record.identifier for record in self._records], dtype=object)
        texts = np.asarray([record.text for record in self._records], dtype=object)
        if self._records:
            embeddings = np.stack(
                [record.embedding.astype(np.float32, copy=False) for record in self._records],
                axis=0,
            )
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        with self._path.open("wb") as buffer:
            np.savez(buffer, ids=ids, texts=texts, embeddings=embeddings)

    def add(self, text: str, embedding: Sequence[float]) -> MemoryRecord:
        identifier = str(uuid4())
        vector = _to_vector(embedding)
        record = MemoryRecord(identifier=identifier, text=text, embedding=vector)
        self._records.append(record)
        self._persist()
        return record

    def is_empty(self) -> bool:
        return not self._records

    def find_similar(
        self, embedding: Sequence[float], limit: int
    ) -> List[Tuple[MemoryRecord, float]]:
        if limit <= 0 or not self._records:
            return []
        query_vector = _to_vector(embedding)
        matrix = np.stack(
            [record.embedding for record in self._records],
            axis=0,
        )
        similarities = _cosine_similarities(matrix, query_vector)
        order = np.argsort(similarities)[::-1]
        top_indices = order[: min(limit, len(order))]
        results: List[Tuple[MemoryRecord, float]] = []
        for index in top_indices:
            record = self._records[int(index)]
            results.append((record, float(similarities[int(index)])))
        return results
