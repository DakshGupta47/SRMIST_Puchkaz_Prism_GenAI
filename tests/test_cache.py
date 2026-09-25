import statistics
import time

import pytest

from cache import DEFAULT_SIMILARITY_THRESHOLD, SemanticCache
from embeddings import TfidfEmbedder
from enrichment import enrich

LATENCY_BUDGET_MS = 300


@pytest.fixture(scope="module")
def embedder():
    return TfidfEmbedder.load()


def test_exact_repeat_is_a_hit(embedder):
    cache = SemanticCache(embedder)
    r = enrich("My Galaxy S22 screen is completely black and won't turn on.")
    cache.put(r.canonical_query, r.query_variations, {"contexts": [{"goal": "g"}]}, device=r.device)

    hit = cache.get(r.canonical_query, r.query_variations, device=r.device)
    assert hit is not None
    assert hit.similarity >= DEFAULT_SIMILARITY_THRESHOLD


def test_paraphrase_across_registers_is_a_hit(embedder):
    cache = SemanticCache(embedder)
    r = enrich("My Galaxy S22 screen is completely black and won't turn on.")
    cache.put(r.canonical_query, r.query_variations, {"contexts": [{"goal": "g"}]}, device=r.device)

    paraphrase = enrich("galaxy s22 display just wont come on, totally black")
    hit = cache.get(paraphrase.canonical_query, paraphrase.query_variations, device=paraphrase.device)
    assert hit is not None


def test_unrelated_query_is_a_miss_not_a_guess(embedder):
    cache = SemanticCache(embedder)
    r = enrich("My Galaxy S22 screen is completely black and won't turn on.")
    cache.put(r.canonical_query, r.query_variations, {"contexts": [{"goal": "g"}]}, device=r.device)

    unrelated = enrich("My Galaxy S25 has a floating circle overlay I want to remove.")
    miss = cache.get(unrelated.canonical_query, unrelated.query_variations, device=unrelated.device)
    assert miss is None  # must be null, never a low-confidence guess


def test_same_symptom_different_device_never_cross_served(embedder):
    """Regression guard: char n-gram similarity between two devices with
    the identical symptom text is very high (~0.9), well above the
    similarity threshold. Without an explicit device gate this would
    serve one device's (possibly wrong) deeplinks/steps to another.
    """
    cache = SemanticCache(embedder)
    a = enrich("My Galaxy S22 screen is completely black and won't turn on.")
    cache.put(a.canonical_query, a.query_variations, {"contexts": [{"goal": "s22-fix"}]}, device=a.device)

    b = enrich("My Galaxy S24 Ultra screen is completely black and won't turn on.")
    result = cache.get(b.canonical_query, b.query_variations, device=b.device)
    assert result is None


def test_same_device_different_symptom_is_a_miss(embedder):
    cache = SemanticCache(embedder)
    a = enrich("My Galaxy S22 screen is completely black and won't turn on.")
    cache.put(a.canonical_query, a.query_variations, {"contexts": [{"goal": "s22-fix"}]}, device=a.device)

    b = enrich("My Galaxy S22 screen has laggy, delayed touch response.")
    result = cache.get(b.canonical_query, b.query_variations, device=b.device)
    assert result is None


def test_empty_cache_returns_none_fast(embedder):
    cache = SemanticCache(embedder)
    r = enrich("My Galaxy S22 screen is completely black.")
    t0 = time.perf_counter()
    result = cache.get(r.canonical_query, r.query_variations, device=r.device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert result is None
    assert elapsed_ms < LATENCY_BUDGET_MS


def test_cache_hit_latency_budget_at_scale(embedder):
    """The headline SLA: a cache hit must return in <=300ms. Populates a
    cache with a few hundred entries (a realistic size for this
    hackathon's scope) and checks p99 lookup latency.
    """
    cache = SemanticCache(embedder, max_entries=2000)
    devices = ["Galaxy S22", "Galaxy S24 Ultra", "Galaxy Z Flip 7", "Galaxy A15", "Galaxy Tab S9"]
    from enrichment import SYMPTOM_TAXONOMY

    for i in range(500):
        device = devices[i % len(devices)]
        symptom = SYMPTOM_TAXONOMY[i % len(SYMPTOM_TAXONOMY)]
        r = enrich(f"My {device} {symptom.casual}, case {i}")
        cache.put(r.canonical_query, r.query_variations, {"contexts": [{"goal": f"g{i}"}]}, device=r.device)

    latencies = []
    for i in range(100):
        device = devices[i % len(devices)]
        symptom = SYMPTOM_TAXONOMY[i % len(SYMPTOM_TAXONOMY)]
        r = enrich(f"My {device} {symptom.formal}, please help {i % 20}")
        t0 = time.perf_counter()
        cache.get(r.canonical_query, r.query_variations, device=r.device)
        latencies.append((time.perf_counter() - t0) * 1000)

    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    assert p99 < LATENCY_BUDGET_MS, f"p99 lookup latency {p99:.1f}ms exceeded {LATENCY_BUDGET_MS}ms budget"
    assert max(latencies) < LATENCY_BUDGET_MS, f"max lookup latency {max(latencies):.1f}ms exceeded budget"
