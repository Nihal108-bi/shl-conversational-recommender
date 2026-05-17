"""One-shot indexer. Run at build time to download the embedding model and
warm the catalog index, so the first /health and /chat calls are fast.

Usage:
    python -m app.indexer
"""

from __future__ import annotations
import sys
import time

from .retriever import get_retriever


def main() -> int:
    t0 = time.time()
    retriever = get_retriever()
    elapsed = time.time() - t0
    if retriever.embeddings is not None:
        print(
            f"Indexed {len(retriever.items)} items, "
            f"embedding dim={retriever.embeddings.shape[1]} "
            f"in {elapsed:.1f}s."
        )
    else:
        print(
            f"Indexed {len(retriever.items)} items (BM25-only — embedding model "
            f"could not be loaded). Took {elapsed:.1f}s."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
