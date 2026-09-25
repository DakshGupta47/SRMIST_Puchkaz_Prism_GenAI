"""
Glue for the full POST /v1/troubleshoot flow. This module owns Stage 0
(enrichment.py) and Stage 3 (cache.py) — Member A's scope — and defines
the two extension points Member B (Stage 1: LLM structuring) and
Member C (Stage 2: deeplink mapping) plug into.

Request/response contract
--------------------------
Inferred from data/siis_responses.json's readme ("siis_response is the
payload your API must accept in POST /v1/troubleshoot") and
data/sample_output.json's shape:

    POST /v1/troubleshoot
    {
      "query": "<raw customer complaint>",
      "siis_response": {"title": "...", "content": "..."}
    }

    -> {
      "query": "<raw customer complaint>",
      "response": {"contexts": [ ... Goal objects, per schema.py ... ]},
      "cache_hit": bool,
      "latency_ms": float,
      "enrichment": {"device", "symptom_category", "device_confidence",
                     "symptom_confidence", "overall_confidence", "is_low_confidence"}
    }

Flow
----
1. Stage 0 (enrichment.enrich): normalize the raw query, extract device,
   generate 8-10 query variations, and score how confident that
   normalization actually is.
2. Stage 3 (cache.SemanticCache.get): if a semantically-equivalent query
   for the same device was already answered, return that validated
   response immediately (this is the ≤300ms path) — but ONLY when Stage 0
   was confident about what it normalized. See "Why low confidence bypasses
   the cache entirely" below.
3. On a miss: Stage 1 (structure extraction from siis_response) then
   Stage 2 (deeplink mapping) run to build the response — these are
   Member B / Member C's stages, injected as callables so this module
   has zero hard dependency on their implementation landing first.
4. The freshly computed response is stored in the cache (keyed by the
   Stage 0 canonical query + variations + device) before being returned,
   so the next semantically-equivalent query is a cache hit — again, only
   when Stage 0 was confident.

Why low confidence bypasses the cache entirely
------------------------------------------------
When Stage 0 can't classify a symptom, every such query normalizes to
nearly the same canonical text regardless of what the customer actually
said (device + "is experiencing an issue that could not be automatically
classified..."). If that got cached, a completely unrelated future
low-confidence query — same device, genuinely different problem — could
register a false-positive cache HIT and be served someone else's answer.
That's a hallucination introduced by the cache layer, not by Stage 1/2, so
it's closed here: `enrichment.is_low_confidence` skips both the cache read
and the cache write, and Stage 1/2 run fresh every time until Stage 0 (or
a human) actually knows what's being asked. This does mean a low-confidence
query never gets the ≤300ms fast path — that's the correct trade: the
300ms budget is a promise about confidently-recognized repeat queries, not
a license to guess quickly.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from cache import SemanticCache
from embeddings import TfidfEmbedder
from enrichment import EnrichmentResult, enrich
from schema import ContextDeeplinkResponse

# -- Member B / Member C extension points --------------------------------
# Signature each stage must implement. `enrichment` is passed through so
# later stages can use the canonical query / device / variations without
# re-deriving them.
Stage1Fn = Callable[[dict, EnrichmentResult], ContextDeeplinkResponse]
Stage2Fn = Callable[[ContextDeeplinkResponse, EnrichmentResult], ContextDeeplinkResponse]


def _stage1_not_wired(siis_response: dict, enrichment: EnrichmentResult) -> ContextDeeplinkResponse:
    """Placeholder Stage 1. Returns an explicitly-empty result rather than
    inventing troubleshooting steps, so a pipeline run before Member B's
    code lands is obviously incomplete instead of silently wrong.
    Replace via Pipeline(stage1_fn=<member B's function>).
    """
    return ContextDeeplinkResponse(contexts=[])


def _stage2_not_wired(structured: ContextDeeplinkResponse, enrichment: EnrichmentResult) -> ContextDeeplinkResponse:
    """Placeholder Stage 2 (deeplink mapping) — passthrough, no deeplinks
    attached. Replace via Pipeline(stage2_fn=<member C's function>).
    """
    return structured


class Pipeline:
    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        stage1_fn: Stage1Fn = _stage1_not_wired,
        stage2_fn: Stage2Fn = _stage2_not_wired,
    ):
        self.cache = cache or SemanticCache(TfidfEmbedder.load())
        self.stage1_fn = stage1_fn
        self.stage2_fn = stage2_fn

    def run(self, query: str, siis_response: dict) -> dict:
        t0 = time.perf_counter()

        # Stage 0
        enrichment = enrich(query)
        enrichment_info = {
            "device": enrichment.device,
            "symptom_category": enrichment.symptom_category,
            "device_confidence": enrichment.device_confidence,
            "symptom_confidence": enrichment.symptom_confidence,
            "overall_confidence": enrichment.overall_confidence,
            "is_low_confidence": enrichment.is_low_confidence,
        }

        # Stage 3 (read path) — skipped entirely when Stage 0 isn't confident;
        # see module docstring for why (false-positive hits across unrelated
        # low-confidence queries).
        cache_hit = None
        if not enrichment.is_low_confidence:
            cache_hit = self.cache.get(
                enrichment.canonical_query, enrichment.query_variations, device=enrichment.device
            )
        if cache_hit is not None:
            return {
                "query": query,
                "response": cache_hit.response,
                "cache_hit": True,
                "similarity": cache_hit.similarity,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "enrichment": enrichment_info,
            }

        # Stage 1 + Stage 2 (Member B / Member C)
        structured = self.stage1_fn(siis_response, enrichment)
        final = self.stage2_fn(structured, enrichment)
        response_dict = final.model_dump()

        # Stage 3 (write path) — same confidence gate as the read path,
        # otherwise this is exactly what would poison the cache with a
        # near-duplicate key for unrelated future ambiguous queries.
        if not enrichment.is_low_confidence:
            self.cache.put(
                enrichment.canonical_query, enrichment.query_variations, response_dict,
                device=enrichment.device,
            )

        return {
            "query": query,
            "response": response_dict,
            "cache_hit": False,
            "similarity": None,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "enrichment": enrichment_info,
        }
