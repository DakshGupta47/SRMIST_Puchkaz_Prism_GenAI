"""
Semantic-paraphrase cache hit rate, measured the way the brief's §6 KPI reads:
the cache is pre-warmed from results.jsonl (exactly what api.py does at
startup), then UNSEEN paraphrases of the 20 reference complaints are sent
WITHOUT a siis_response, and we count how many are served from the cache.

Only paraphrases whose source complaint actually has a non-empty answer in
results.jsonl are scored (a paraphrase of a no_match row has nothing to hit).
No LLM is called: Stage 1 is stubbed to return nothing, so any non-empty
answer below can only have come from the cache.

Run:  python scripts/eval_paraphrase_hits.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["LLM_PROVIDER"] = "mock"

from cache import SemanticCache  # noqa: E402
from deeplink_mapping import deeplink_mapping  # noqa: E402
from embeddings import TfidfEmbedder  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from schema import ContextDeeplinkResponse  # noqa: E402

# (paraphrase, 0-based row in data/input.txt it paraphrases)
PARAPHRASES = [
    ("galaxy s22 screen goes white/blank when i open apps", 1),
    ("My S22's display is completely black, nothing shows", 1),
    ("Z Flip 7 screen went black, can't transfer my data", 2),
    ("galaxy z flip7 display dead black cant use smart switch", 2),
    ("Samsung Galaxy A16 screen went black by itself, won't display anything", 3),
    ("My Galaxy Flip 7 inner screen stopped working", 7),
    ("the folding screen on my flip 7 is dead", 7),
    ("Galaxy Z Flip 6 screen flickers then goes blank when I open it", 8),
    ("my flip 6 half the display is black, other half works", 9),
    ("Galaxy S22 no activation message, screen stays blank after carrier switch", 11),
    ("my phone's screen is cracked and bleeding", 12),
    ("galaxy s26 ultra shows a blue or black screen and won't boot", 13),
    ("S Ultra screen flashes super fast when charging", 14),
    ("My Galaxy S24 screen stays dark and blank", 15),
    ("Galaxy Z Flip 7 screen cracked at the fold", 16),
    ("Galaxy S22 touch is laggy and delayed", 18),
    ("touchscreen lag on galaxy s22 when typing", 18),
    ("galaxy s24 ultra black screen of death but it still rings", 19),
    ("My Galaxy S24 Ultra display is black though the phone is on", 19),
    ("phone screen black wont turn on", 1),
]


def main() -> dict:
    rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = {i for i, r in enumerate(rows[:20]) if r["response"].get("contexts")}
    pipe = Pipeline(
        cache=SemanticCache(TfidfEmbedder.load()),
        stage1_fn=lambda s, e: ContextDeeplinkResponse(contexts=[]),
        stage2_fn=deeplink_mapping,
    )
    warmed = pipe.warm_from_results(ROOT / "results.jsonl")
    scored = hits = 0
    lat = []
    detail = []
    for q, src in PARAPHRASES:
        t0 = time.perf_counter()
        r = pipe.run(q, {})
        ms = (time.perf_counter() - t0) * 1000
        if src not in answerable:
            detail.append({"query": q, "source_row": src, "scored": False})
            continue
        scored += 1
        hit = r["meta"]["cache_hit"]
        hits += hit
        if hit:
            lat.append(ms)
        detail.append({"query": q, "source_row": src, "scored": True, "hit": hit,
                       "similarity": r["meta"]["similarity"], "latency_ms": round(ms, 2)})
    lat.sort()
    out = {
        "warmed_entries": warmed,
        "scored": scored,
        "hits": hits,
        "hit_rate": round(hits / scored, 4) if scored else None,
        "hit_latency_p50_ms": round(statistics.median(lat), 2) if lat else None,
        "hit_latency_p95_ms": round(lat[max(0, int(0.95 * len(lat)) - 1)], 2) if lat else None,
        "detail": detail,
    }
    (ROOT / "artifacts" / "paraphrase_hit_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=2))
    for d in detail:
        if d.get("scored") and not d["hit"]:
            print("MISS:", d["query"])
    return out


if __name__ == "__main__":
    main()
