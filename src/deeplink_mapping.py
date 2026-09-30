from __future__ import annotations

import json
import re
from pathlib import Path

from embeddings import TfidfEmbedder
from enrichment import EnrichmentResult
from schema import ContextDeeplinkResponse, Deeplink, ValidationDeepLink, actionCategory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIMILARITY_THRESHOLD = 0.35  # placeholder — tune against sample_output.json

_catalog = None
_catalog_texts = None
_catalog_matrix = None
_embedder = None
_placeholder = None

_STOP = {"and", "or", "the", "a", "of", "to", "in", "for", "with"}


def _load_catalog():
    global _catalog, _catalog_texts, _catalog_matrix, _embedder, _placeholder
    if _catalog is not None:
        return
    raw = json.loads((DATA_DIR / "deeplinks.json").read_text())
    entries = raw["deeplinks"]
    _placeholder = next(
        (d["deeplink"] for d in entries if d.get("originalType") == "placeholder"),
        entries[0]["deeplink"].split("://")[0] + "://dummy_positive",
    )
    _catalog = [d for d in entries if d.get("originalType") != "placeholder"]
    _catalog_texts = [
        f"{d.get('description','')} {d.get('message','')} {d.get('qna_description','')}"
        for d in _catalog
    ]
    _embedder = TfidfEmbedder.load()
    _catalog_matrix = _embedder.encode_sparse(_catalog_texts)


def _best_match(step_text: str):
    _load_catalog()
    q = _embedder.encode_sparse([step_text])
    sims = (_catalog_matrix @ q.T).toarray().ravel()
    best_idx = sims.argmax()
    return _catalog[best_idx], float(sims[best_idx])


def _core(name: str) -> list[str]:
    ws = [w for w in re.findall(r"[A-Za-z0-9+]+", name)
          if w.lower() not in {"settings", "setting"}][:4]
    while ws and ws[-1].lower() in _STOP:
        ws.pop()
    return [w.lower() for w in ws]


def _fit(words: list[str], lo: int = 5, hi: int = 7) -> str:
    words = words[:hi]
    while len(words) < lo:
        words.insert(1, "device")
    return " ".join(words)


def _dummy_positive(action_name: str) -> Deeplink:
    _load_catalog()
    c = _core(action_name)
    return Deeplink(
        deeplink=_placeholder,
        description=_fit(["Open", *c, "settings", "screen"]),
        message=_fit(["Open", *c, "in", "device", "Settings"]),
    )


def _to_validation_deeplink(entry: dict) -> ValidationDeepLink | None:
    v = entry.get("validation")
    if not v:
        return None
    return ValidationDeepLink(deeplink=v["deeplink"], key=v["key"])


def deeplink_mapping(
    structured: ContextDeeplinkResponse, enrichment: EnrichmentResult
) -> ContextDeeplinkResponse:
    for goal in structured.contexts:
        for action in goal.actions:
            for step_group in action.stepGroups:
                if action.category != actionCategory.auto:
                    # Only auto actions may carry deeplinks (manual is a schema rule;
                    # critical is our design choice).
                    step_group.actionableDeeplink = None
                    step_group.validationDeeplink = None
                    continue
                query_text = " ".join(step_group.steps)
                match, score = _best_match(query_text)
                if score >= SIMILARITY_THRESHOLD:
                    step_group.actionableDeeplink = Deeplink(
                        deeplink=match["deeplink"],
                        description=match["description"],
                        message=match.get("message", ""),
                        originalType=match.get("originalType"),
                    )
                    step_group.validationDeeplink = _to_validation_deeplink(match)
                else:
                    step_group.actionableDeeplink = _dummy_positive(action.actionName)
    return structured