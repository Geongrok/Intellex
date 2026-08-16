"""Vector store wrapper around ChromaDB (persistent).

Stores document chunks as dense embeddings with file/page metadata so the
chatbot can do fast semantic retrieval alongside the BM25 keyword index.
"""

import os
from typing import Dict, List

from .embeddings import EmbeddingEngine

COLLECTION_NAME = "intellex_docs"


class VectorStore:
    def __init__(self, persist_dir: str = "cache/chroma", embedding=None):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.embedding = embedding or EmbeddingEngine()
        self._client = None
        self._collection = None
        self._available = False
        try:
            self._ensure_client()
            self._available = True
        except Exception as exc:  # chromadb not installed -> BM25-only mode
            print(f"[VectorStore] Disabled (chromadb unavailable): {exc}")
            self._available = False

    # ------------------------------------------------------------------ #
    # ChromaDB lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_client(self):
        import chromadb

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self._collection = self._client.get_collection(COLLECTION_NAME)
        except Exception:
            self._collection = self._client.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    def _check_collection_metadata(self):
        """Ensure the collection embedding dimension matches the engine."""
        meta = None
        try:
            meta = self._collection.metadata or {}
        except Exception:
            return
        dim = int(meta.get("dim", 0))
        if dim and dim != self.embedding.dimension():
            # Embedding backend changed -> rebuild collection.
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "dim": self.embedding.dimension()},
            )

# ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
    ) -> None:
        """Add chunks (with embeddings) to the store."""
        if not ids or not self._available:
            return
        try:
            self._check_collection_metadata()
            vectors = self.embedding.embed(texts)
            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=vectors,
                metadatas=metadatas,
            )
        except Exception as exc:
            print(f"[VectorStore] add_documents skipped: {exc}")

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        """Semantic search: return top-k matches with distance scores."""
        if not self._available:
            return []
        try:
            if self._collection.count() == 0:
                return []
        except Exception:
            return []
        vec = self.embedding.embed(text)[0]
        res = self._collection.query(
            query_embeddings=[vec],
            n_results=min(top_k, self._collection.count()),
        )
        results = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            meta = metas[i] or {}
            results.append(
                {
                    "id": doc_id,
                    "text": docs[i],
                    "file": meta.get("file", "unknown"),
                    "page": meta.get("page"),
                    "distance": float(dists[i]) if dists else 1.0,
                    # cosine similarity from distance (cosine space)
                    "score": max(
                        0.0, min(1.0, 1.0 - float(dists[i])) if dists else 0.0
                    ),
                }
            )
        return results

    def count(self) -> int:
        if not self._available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        if not self._available:
            return
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        try:
            self._collection = self._client.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "dim": self.embedding.dimension()},
            )
        except Exception:
            self._ensure_client()

    def info(self) -> dict:
        return {
            "count": self.count(),
            "embedding": self.embedding.info(),
        }

