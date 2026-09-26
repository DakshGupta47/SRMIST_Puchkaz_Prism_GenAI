"""
Benchmark: measures the brief's headline Stage 0 + Stage 3 KPI directly —
"≥80% cache hit rate on unseen paraphrases" — instead of leaving it as an
assumption spot-checked on a couple of live examples.

Ground truth used: the kit's OWN two provided complaint sets, not
synthetic template text, so this isn't grading the system against text it
was built from.

  * data/input.txt          20 real sample complaints (all screen/display
                             symptoms) — the corpus normalize_query's
                             taxonomy was originally built by inspecting
                             (see enrichment.py's own module docstring).
                             Treated as the "already answered, in cache"
                             set.
  * data/unseen_scenarios.txt   38 real complaints spanning every symptom
                             family (battery, connectivity, audio, camera,
                             storage, apps, biometrics, notifications,
                             updates, plus more screen cases) — the kit's
                             own file name for this is literally "unseen
                             scenarios". Treated as the queries a live
                             system would receive after those answers.

Method — mirrors pipeline.Pipeline.run() exactly, line by line, in file
order (input.txt fully first, matching "already deployed and answered",
then unseen_scenarios.txt):
  1. enrich() the line.
  2. If is_low_confidence: skip cache read/write entirely (this is
     pipeline.py's own gate — a low-confidence query is never a fair test
     of cache generalization, it's a taxonomy-coverage question instead,
     reported separately here).
  3. Otherwise: cache.get(canonical, variations, device). Record hit/miss
     for lines that came from unseen_scenarios.txt (input.txt lines are
     priming the cache, not being scored — nothing has been cached yet
     when the first one runs).
  4. On a miss, cache.put(...) — exactly pipeline.py's write-on-miss
     behavior, so the cache also grows across unseen_scenarios.txt lines
     the same way it would in production (a later unseen line can hit an
     earlier unseen line's cache entry, not only an input.txt one).

Also reports latency percentiles for cache.get() and enrich() and writes
everything to artifacts/benchmark_results.json (metrics.md is filled in
from this run's actual output, not hand-typed).

Run:
    cd theme2_submission && python scripts/benchmark_cache.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
DATA = Path(__file__).resolve().parent.parent / "data"

from cache import DEFAULT_SIMILARITY_THRESHOLD, SemanticCache  # noqa: E402
from embeddings import TfidfEmbedder  # noqa: E402
from enrichment import enrich  # noqa: E402


def _load_lines(name: str) -> list[str]:
    return [line.strip() for line in (DATA / name).read_text().splitlines() if line.strip()]


def run_benchmark() -> dict:
    embedder = TfidfEmbedder.load()
    cache = SemanticCache(embedder, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD)

    seed_lines = _load_lines("input.txt")
    unseen_lines = _load_lines("unseen_scenarios.txt")

    enrich_latencies_ms: list[float] = []
    get_latencies_ms: list[float] = []
    skipped_low_confidence_seed = 0
    seed_cached = 0

    def process(raw: str, *, score: bool, per_line: list | None):
        nonlocal skipped_low_confidence_seed, seed_cached
        t0 = time.perf_counter()
        result = enrich(raw)
        enrich_latencies_ms.append((time.perf_counter() - t0) * 1000)

        if result.is_low_confidence:
            if not score:
                skipped_low_confidence_seed += 1
            elif per_line is not None:
                per_line.append({"query": raw, "category": result.symptom_category,
                                  "scored": False, "reason": "is_low_confidence (bypasses cache, per pipeline.py)"})
            return

        t0 = time.perf_counter()
        hit = cache.get(result.canonical_query, result.query_variations, device=result.device)
        elapsed = (time.perf_counter() - t0) * 1000
        if score:
            get_latencies_ms.append(elapsed)
            if per_line is not None:
                per_line.append({
                    "query": raw, "category": result.symptom_category, "scored": True,
                    "hit": hit is not None,
                    "similarity": round(hit.similarity, 3) if hit else None,
                })

        if hit is None:  # miss -> pipeline.py writes on miss, so this does too
            cache.put(result.canonical_query, result.query_variations,
                       response={"category": result.symptom_category}, device=result.device)
            if not score:
                seed_cached += 1

    # Phase 1: prime the cache from the kit's real sample complaints (never scored).
    for raw in seed_lines:
        process(raw, score=False, per_line=None)

    # Phase 2: score the kit's own "unseen scenarios" file against that cache,
    # growing the cache the same way pipeline.py would in production.
    per_line: list = []
    for raw in unseen_lines:
        process(raw, score=True, per_line=per_line)

    scored = [r for r in per_line if r["scored"]]
    hits = [r for r in scored if r["hit"]]

    def pctl(data: list[float], p: float) -> float:
        if not data:
            return float("nan")
        s = sorted(data)
        k = (len(s) - 1) * p
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "overall_hit_rate": round(len(hits) / len(scored), 4) if scored else None,
        "unseen_scenarios_total": len(unseen_lines),
        "unseen_scenarios_scored": len(scored),
        "unseen_scenarios_skipped_low_confidence": len(unseen_lines) - len(scored),
        "unseen_scenarios_hits": len(hits),
        "seed_lines_from_input_txt": len(seed_lines),
        "seed_lines_skipped_low_confidence": skipped_low_confidence_seed,
        "seed_lines_cached": seed_cached,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "cache_get_latency_ms": {
            "p50": round(pctl(get_latencies_ms, 0.50), 3),
            "p95": round(pctl(get_latencies_ms, 0.95), 3),
            "p99": round(pctl(get_latencies_ms, 0.99), 3),
            "max": round(max(get_latencies_ms), 3) if get_latencies_ms else None,
            "n": len(get_latencies_ms),
        },
        "stage0_enrich_latency_ms": {
            "p50": round(pctl(enrich_latencies_ms, 0.50), 3),
            "p95": round(pctl(enrich_latencies_ms, 0.95), 3),
            "max": round(max(enrich_latencies_ms), 3) if enrich_latencies_ms else None,
            "n": len(enrich_latencies_ms),
        },
        "per_line": per_line,
        "misses": [r["query"] for r in scored if not r["hit"]],
        "low_confidence_unseen_queries": [r["query"] for r in per_line if not r["scored"]],
    }


def run_controlled_register_experiment() -> dict:
    """Second, controlled experiment isolating one variable the realistic
    run above can't cleanly separate: does the similarity/threshold
    mechanism itself generalize across register (formal/casual/keyword/
    frustrated/typo) when device and category ARE already shared with a
    cached entry? The realistic run mixes in two confounds that are real
    but not about the matching mechanism per se — cold-start (a category
    with no prior cache entry at all) and cross-device gating (correct by
    design, see cache.py's module docstring) — both of which mechanically
    force a miss regardless of how good the similarity matching is.

    For every taxonomy category that keyword-classifies its own formal AND
    casual template text back to itself (see the self-classification
    check), this: enrich()s the FORMAL phrasing, caches its canonical +
    variations, then enrich()s the CASUAL phrasing of the same
    category/device and queries the cache with it. Both phrasings name the
    same fixed device, so this isolates register generalization specifically.
    """
    from cache import DEFAULT_SIMILARITY_THRESHOLD as THRESH, SemanticCache as Cache
    from embeddings import TfidfEmbedder as Embedder
    from enrichment import SYMPTOM_TAXONOMY, enrich as _enrich

    embedder = Embedder.load()
    device = "Galaxy S22"
    per_category = []
    hits = 0
    scored = 0
    for symptom in SYMPTOM_TAXONOMY:
        cache = Cache(embedder, similarity_threshold=THRESH)
        a = _enrich(f"My {device} {symptom.subject} {symptom.formal}.")
        b = _enrich(f"my {device} {symptom.subject} {symptom.casual}")
        if a.symptom_category != symptom.category or b.symptom_category != symptom.category:
            per_category.append({"category": symptom.category, "scored": False,
                                  "reason": "formal/casual template text doesn't self-classify"})
            continue
        cache.put(a.canonical_query, a.query_variations, response={}, device=a.device)
        hit = cache.get(b.canonical_query, b.query_variations, device=b.device)
        scored += 1
        hits += 1 if hit else 0
        per_category.append({"category": symptom.category, "scored": True, "hit": hit is not None,
                              "similarity": round(hit.similarity, 3) if hit else None})
    return {
        "hit_rate": round(hits / scored, 4) if scored else None,
        "categories_scored": scored,
        "categories_hit": hits,
        "categories_total": len(SYMPTOM_TAXONOMY),
        "categories_excluded_self_classification_failure": [
            c["category"] for c in per_category if not c["scored"]
        ],
        "per_category": per_category,
    }


if __name__ == "__main__":
    results = run_benchmark()
    results["controlled_register_experiment"] = run_controlled_register_experiment()
    out_path = Path(__file__).resolve().parent.parent / "artifacts" / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Overall cache hit rate on scored unseen scenarios: {results['overall_hit_rate']:.1%} "
          f"({results['unseen_scenarios_hits']}/{results['unseen_scenarios_scored']}, "
          f"{results['unseen_scenarios_skipped_low_confidence']} skipped as low-confidence)")
    print(f"Misses: {results['misses']}")
    print(f"Low-confidence unseen queries (never enter/exit cache): {results['low_confidence_unseen_queries']}")
    print(f"cache.get() p50/p95/p99 (ms): {results['cache_get_latency_ms']['p50']}/"
          f"{results['cache_get_latency_ms']['p95']}/{results['cache_get_latency_ms']['p99']}")
    print(f"enrich() p50/p95 (ms): {results['stage0_enrich_latency_ms']['p50']}/{results['stage0_enrich_latency_ms']['p95']}")
    ctrl = results["controlled_register_experiment"]
    print(f"[controlled same-device register experiment] hit rate: {ctrl['hit_rate']:.1%} "
          f"({ctrl['categories_hit']}/{ctrl['categories_scored']} categories; "
          f"{len(ctrl['categories_excluded_self_classification_failure'])} excluded — "
          f"self-classification failure, see README known gaps)")
    print(f"Wrote {out_path}")
