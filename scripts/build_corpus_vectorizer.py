#!/usr/bin/env python3
"""One-off build step: fit the TF-IDF embedder on the domain corpus and
persist it to artifacts/tfidf_vectorizer.pkl.

Run once after cloning, or whenever data/siis_responses.json /
data/deeplinks.json change:

    python scripts/build_corpus_vectorizer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embeddings import build_and_save_vectorizer, VECTORIZER_PATH  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    embedder = build_and_save_vectorizer(DATA_DIR)
    vocab_size = len(embedder._vec.vocabulary_)
    print(f"Fit TF-IDF vectorizer: vocab_size={vocab_size}")
    print(f"Saved to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
