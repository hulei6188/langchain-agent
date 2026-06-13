from __future__ import annotations

import math
import socket
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from core.config import get_settings


@dataclass
class VectorHit:
    vector_id: str
    text: str
    score: float
    metadata: dict


class PrecomputedVectorStore(VectorStore):
    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ):
        store = cls()
        store.add_texts(texts, metadatas=metadatas, ids=ids, vectors=_precomputed_vectors(kwargs))
        return store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        text_list = list(texts)
        vectors = _precomputed_vectors(kwargs)
        if vectors is None:
            raise ValueError(f"{self.__class__.__name__}.add_texts requires precomputed vectors")
        metadatas = _metadata_list(metadatas, len(text_list))
        ids = [str(item) for item in (ids or [metadata.get("vector_id") or index for index, metadata in enumerate(metadatas)])]
        return self._add_precomputed(text_list, metadatas, ids, [list(vector) for vector in vectors])

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        raise NotImplementedError(f"{self.__class__.__name__} requires vector queries; use similarity_search_by_vector")

    def upsert(self, vector_id: str, vector: list[float], text: str, metadata: dict) -> None:
        self.add_texts([text], metadatas=[{**metadata, "vector_id": vector_id}], ids=[vector_id], vectors=[vector])

    def search(self, vector: list[float], *, limit: int = 5, filters: dict | None = None) -> list[VectorHit]:
        return [_document_to_vector_hit(document) for document in self.similarity_search_by_vector(vector, k=limit, filter=filters or {})]

    def _add_precomputed(self, texts: list[str], metadatas: list[dict], ids: list[str], vectors: list[list[float]]) -> list[str]:
        raise NotImplementedError


class MemoryVectorStore(PrecomputedVectorStore):
    def __init__(self) -> None:
        self._items: list[tuple[str, list[float], str, dict]] = []

    def _add_precomputed(self, texts: list[str], metadatas: list[dict], ids: list[str], vectors: list[list[float]]) -> list[str]:
        for vector_id, vector, text, metadata in zip(ids, vectors, texts, metadatas, strict=True):
            self._items = [item for item in self._items if item[0] != vector_id]
            self._items.append((vector_id, vector, text, metadata))
        return ids

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[Document]:
        filters = kwargs.get("filter") or kwargs.get("filters") or {}
        hits = []
        for vector_id, item_vector, text, metadata in self._items:
            if any(metadata.get(key) != value for key, value in filters.items()):
                continue
            score = _cosine(embedding, item_vector)
            hits.append(Document(page_content=text, metadata={**metadata, "vector_id": vector_id, "_score": score}))
        return sorted(hits, key=lambda item: float(item.metadata.get("_score") or 0), reverse=True)[:k]

    def delete(self, ids: list[str] | None = None, *, filters: dict | None = None, **kwargs: Any) -> bool:
        ids_set = {str(item) for item in ids or []}
        filters = filters or kwargs.get("filter") or {}
        self._items = [
            item
            for item in self._items
            if not ((ids_set and item[0] in ids_set) or (filters and all(item[3].get(key) == value for key, value in filters.items())))
        ]
        return True


class MilvusVectorStore(PrecomputedVectorStore):
    def __init__(self) -> None:
        self.settings = get_settings()
        self._fallback = MemoryVectorStore()
        self._client = None
        self._error = ""
        if self.settings.vector_backend == "milvus":
            try:
                from langchain_milvus import Milvus

                self._probe_milvus_endpoint()
                connection_args = {"uri": self.settings.milvus_uri}
                if self.settings.milvus_token:
                    connection_args["token"] = self.settings.milvus_token
                self._client = Milvus(
                    embedding_function=None,
                    collection_name=self.settings.milvus_collection,
                    connection_args=connection_args,
                    auto_id=False,
                    primary_field="pk",
                    text_field="text",
                    vector_field="vector",
                    enable_dynamic_field=True,
                    timeout=2,
                )
            except Exception as exc:
                self._client = None
                self._error = str(exc)[:240]

    def _add_precomputed(self, texts: list[str], metadatas: list[dict], ids: list[str], vectors: list[list[float]]) -> list[str]:
        if not self._client:
            return self._fallback.add_texts(texts, metadatas=metadatas, ids=ids, vectors=vectors)
        for vector_id in ids:
            try:
                self._client.delete(ids=[vector_id])
            except Exception:
                pass
        return [
            str(item)
            for item in self._client.add_embeddings(
                texts,
                vectors,
                metadatas=metadatas,
                ids=ids,
            )
        ]

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[Document]:
        filters = kwargs.get("filter") or kwargs.get("filters") or {}
        if not self._client:
            return self._fallback.similarity_search_by_vector(embedding, k=k, filter=filters)
        expr = build_milvus_filter(filters)
        results = self._client.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            expr=expr or None,
        )
        documents = []
        for document, score in results:
            metadata = dict(document.metadata or {})
            vector_id = str(metadata.get("vector_id") or metadata.get("pk") or "")
            metadata["vector_id"] = vector_id
            metadata["_score"] = float(score)
            documents.append(Document(page_content=document.page_content, metadata=metadata))
        return documents

    @property
    def available(self) -> bool:
        return self.settings.vector_backend != "milvus" or self._client is not None

    @property
    def using_fallback(self) -> bool:
        return self.settings.vector_backend == "milvus" and self._client is None

    def status(self) -> dict:
        return {
            "backend": self.settings.vector_backend,
            "configured_backend": self.settings.vector_backend,
            "active_backend": "milvus" if self._client else "memory",
            "driver": "langchain-milvus" if self._client else "memory",
            "collection": self.settings.milvus_collection if self.settings.vector_backend == "milvus" else None,
            "uri": self.settings.milvus_uri if self.settings.vector_backend == "milvus" else None,
            "available": self.available,
            "fallback": self.using_fallback,
            "error": self._error or None,
        }

    def delete(self, ids: list[str] | None = None, *, filters: dict | None = None, **kwargs: Any) -> bool:
        if not self._client:
            return self._fallback.delete(ids=ids, filters=filters, **kwargs)
        if ids:
            self._client.delete(ids=[str(item) for item in ids])
        filters = filters or kwargs.get("filter") or {}
        expr = build_milvus_filter(filters)
        if expr:
            self._client.delete(expr=expr)
        return True

    def _probe_milvus_endpoint(self) -> None:
        parsed = urlparse(self.settings.milvus_uri)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 19530)
        if not host:
            raise ValueError("MILVUS_URI is invalid")
        with socket.create_connection((host, port), timeout=1):
            return


def build_milvus_filter(filters: dict) -> str:
    parts = []
    for key, value in filters.items():
        if isinstance(value, str):
            safe = value.replace('"', '\\"')
            parts.append(f'{key} == "{safe}"')
        else:
            parts.append(f"{key} == {value}")
    return " and ".join(parts)


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right_values)) or 1.0
    return numerator / (left_norm * right_norm)


def _metadata_list(metadatas: list[dict] | None, count: int) -> list[dict]:
    if metadatas is None:
        return [{} for _ in range(count)]
    return [dict(metadata or {}) for metadata in metadatas]


def _precomputed_vectors(kwargs: dict[str, Any]):
    vectors = kwargs.get("vectors")
    return vectors if vectors is not None else kwargs.get("embeddings")


def _document_to_vector_hit(document: Document) -> VectorHit:
    metadata = dict(document.metadata or {})
    score = float(metadata.pop("_score", 0.0) or 0.0)
    vector_id = str(metadata.get("vector_id") or metadata.get("pk") or "")
    return VectorHit(vector_id=vector_id, text=document.page_content, score=score, metadata=metadata)


vector_store = MilvusVectorStore()
