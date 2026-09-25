"""
Embedding backend for the semantic cache (Stage 3) and any downstream
semantic matching (e.g. Member C's deeplink retrieval).

Design note
-----------
The obvious choice for "use an existing pretrained model, don't fine-tune"
is a small sentence-transformer (e.g. all-MiniLM-L6-v2) run locally. That
needs its weights pulled from huggingface.co the first time, and on this
network huggingface.co is blocked by the org's egress allowlist (pypi.org
is allowed, huggingface.co returns 403 at the proxy). That's an
environment constraint, not a design choice, so instead of blocking on it
we ship a network-free embedding backend:

    a character n-gram TF-IDF vectorizer, fit once on the domain corpus
    (siis_responses.json + deeplinks.json text), used with cosine
    similarity.

This is still "an existing, off-the-shelf technique, not a fine-tuned
model" (TF-IDF requires no gradient training, just a corpus fit), it
needs zero network access at runtime, it is very fast (well under our
300ms cache-hit budget), and char n-grams (not word n-grams) make it
naturally robust to typos and to short "keyword" style queries, which is
exactly the register spread Stage 0 has to handle.

If a teammate later gets a real embedding model working (e.g. on a
machine/network that can reach huggingface.co), swap it in by implementing
the `Embedder` interface below and passing it to `SemanticCache` /
`Enricher` — nothing else in the pipeline needs to change.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, List, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.pkl"


class Embedder(Protocol):
    """Minimal interface any embedding backend must satisfy."""

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float array of embeddings."""
        ...

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two single embeddings."""
        ...


class TfidfEmbedder:
    """Char n-gram TF-IDF embedder, fit on the Samsung troubleshooting
    domain corpus. No network access, no fine-tuning — a fixed sklearn
    vectorizer fit once and persisted to disk.
    """

    def __init__(self, vectorizer: TfidfVectorizer):
        self._vec = vectorizer

    @classmethod
    def load(cls, path: Path = VECTORIZER_PATH) -> "TfidfEmbedder":
        with open(path, "rb") as f:
            vec = pickle.load(f)
        return cls(vec)

    def save(self, path: Path = VECTORIZER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._vec, f)

    def encode(self, texts: List[str]) -> np.ndarray:
        matrix = self._vec.transform(texts)
        return matrix.toarray().astype(np.float32)

    def encode_sparse(self, texts: List[str]):
        """Sparse, L2-row-normalized encoding — the fast path for the
        cache's internal index. TF-IDF vectors over a ~20k-dim vocabulary
        are almost entirely zero for any one short query, so keeping them
        sparse (and pre-normalized, so a dot product IS the cosine
        similarity) avoids the dense-vector norm/matmul cost that
        dominates lookup latency at more than a few hundred cached
        entries.
        """
        matrix = self._vec.transform(texts)
        from sklearn.preprocessing import normalize

        return normalize(matrix, norm="l2", axis=1)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        return float(cosine_similarity(a, b)[0][0])


def _iter_corpus_texts(data_dir: Path) -> Iterable[str]:
    """Pull representative domain text out of the provided kit files so
    the vectorizer's vocabulary matches real Samsung troubleshooting
    language (symptoms, settings names, deeplink descriptions) rather than
    generic English.
    """
    siis_path = data_dir / "siis_responses.json"
    deeplinks_path = data_dir / "deeplinks.json"
    input_path = data_dir / "input.txt"

    if siis_path.exists():
        siis = json.loads(siis_path.read_text())
        for row in siis.get("responses", []):
            yield row.get("original_query", "")
            resp = row.get("siis_response", {})
            yield resp.get("title", "")
            yield resp.get("content", "")

    if deeplinks_path.exists():
        deeplinks = json.loads(deeplinks_path.read_text())
        for dl in deeplinks.get("deeplinks", []):
            yield dl.get("description", "")
            yield dl.get("message", "") or ""
            yield dl.get("qna_description", "") or ""

    if input_path.exists():
        for line in input_path.read_text().splitlines():
            if line.strip():
                yield line.strip()


def build_and_save_vectorizer(
    data_dir: Path,
    out_path: Path = VECTORIZER_PATH,
    max_features: int = 20000,
) -> TfidfEmbedder:
    """Fit the TF-IDF vectorizer on the domain corpus and persist it.
    Run this once (scripts/build_corpus_vectorizer.py) or whenever the
    kit data changes; the pipeline itself only ever loads the artifact.
    """
    texts = [t for t in _iter_corpus_texts(data_dir) if t]
    if not texts:
        raise ValueError(f"No corpus text found under {data_dir}")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        lowercase=True,
        sublinear_tf=True,
    )
    vectorizer.fit(texts)
    embedder = TfidfEmbedder(vectorizer)
    embedder.save(out_path)
    return embedder
