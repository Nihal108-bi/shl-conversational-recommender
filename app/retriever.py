git commit -m "feat(retrieval): hybrid BM25 + dense (MiniLM) retriever with RRF fusion""""Hybrid retrieval: BM25 + dense (MiniLM) with reciprocal rank fusion.

BM25 catches exact-name and exact-skill queries (Docker, OPQ32r, AWS).
Dense catches intent-style queries ("call centre agents", "senior leadership").
RRF fuses the two without needing to tune a relative weight per query.

The index is built once at startup. Catalog is ~377 items — embeddings fit in
RAM trivially (~3 MB at 384 dims) and BM25 is in-memory by default.
"""

from __future__ import annotations
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .catalog import CatalogItem, load_catalog


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(self, items: List[CatalogItem]):
        self.items = items
        self.docs = [it.to_search_doc() for it in items]
        self.tokenized = [_tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized)
        self.embeddings: Optional[np.ndarray] = None  # lazy

    # ---- Dense side -------------------------------------------------------

    def _ensure_embeddings(self) -> Optional[np.ndarray]:
        """Load the sentence-transformer model and embed the catalog.

        Returns None if loading fails (no network, no model cache) — the
        retriever then degrades to BM25-only. Logged loudly so deployment
        misconfigurations don't go unnoticed.
        """
        if self.embeddings is None and not getattr(self, "_dense_disabled", False):
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                embs = model.encode(
                    self.docs,
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                self.embeddings = embs.astype(np.float32)
                self._encoder = model
            except Exception as e:
                # No model available -> BM25-only mode. Still useful, just less
                # robust to paraphrased queries.
                import logging
                logging.getLogger("retriever").warning(
                    "Dense retrieval disabled (couldn't load embedding model): %s", e
                )
                self._dense_disabled = True
                self.embeddings = None
        return self.embeddings

    def _encode_query(self, q: str) -> Optional[np.ndarray]:
        self._ensure_embeddings()
        if self.embeddings is None:
            return None
        return self._encoder.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)

    # ---- Public API -------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 25,
        *,
        keys_filter: Optional[List[str]] = None,
        job_level_filter: Optional[List[str]] = None,
    ) -> List[Tuple[CatalogItem, float]]:
        """Return up to `k` items ranked by hybrid score.

        Filters are applied as soft boosts AFTER ranking so a too-strict
        filter can't blank the result set.
        """
        tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm = np.argsort(-bm25_scores)[:50]

        rrf_k = 60
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(top_bm):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        # Fuse with dense if available.
        dense_emb = self._encode_query(query)
        if dense_emb is not None:
            embs = self.embeddings  # already populated
            dense_scores = embs @ dense_emb
            top_dn = np.argsort(-dense_scores)[:50]
            for rank, idx in enumerate(top_dn):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(rrf.items(), key=lambda x: -x[1])

        # Soft filter boosts.
        results: list[Tuple[CatalogItem, float]] = []
        for idx, score in ranked:
            item = self.items[idx]
            boost = 0.0
            if keys_filter and any(k in item.keys for k in keys_filter):
                boost += 0.02
            if job_level_filter and any(j in item.job_levels for j in job_level_filter):
                boost += 0.02
            results.append((item, score + boost))

        results.sort(key=lambda x: -x[1])
        return results[:k]

    # ---- Persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        # Embeddings only — BM25 is cheap to rebuild from the catalog.
        self._ensure_embeddings()
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings}, f)

    def load_embeddings(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )


_SINGLETON: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Process-wide singleton; FastAPI may call into it from any worker."""
    global _SINGLETON
    if _SINGLETON is None:
        items = load_catalog()
        _SINGLETON = HybridRetriever(items)
        _SINGLETON._ensure_embeddings()  # warm; tolerates failure
    return _SINGLETON
