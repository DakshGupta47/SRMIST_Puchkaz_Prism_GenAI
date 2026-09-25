"""
Glue for the full POST /v1/troubleshoot flow. This module owns Stage 0
(enrichment.py) and Stage 3 (cache.py) — Member A's scope — and defines
the two extension points Member B (Stage 1: LLM structuring) and
Member C (Stage 2: deeplink mapping) plug into.

Request/response contract
--------------------------
The request shape is confirmed directly against data/siis_responses.json's
own readme ("siis_response is the payload your API must accept in
POST /v1/troubleshoot") and its actual per-record shape
({"title": ..., "content": ...}) — the theme brief's own request example
shows a bare "<optional raw text context>" string, but the real provided
data is an object, and that's what this accepts.

The response shape matches the official theme brief's §5 API contract and
Appendix B worked example exactly (not just data/sample_output.json's
smaller shape, which predates the full brief): "query_variations" is a
TOP-LEVEL key (Stage 0's paraphrases), not buried inside "enrichment", and
"cache_hit"/"latency_ms"/"model"/"cost_usd" are nested under a "meta"
object, not flat top-level fields. "enrichment" (device/symptom/confidence
signals) isn't part of the official contract at all — it's kept as an
additive debug field since nothing in the spec forbids extra top-level
keys, and it's genuinely useful for demoing Stage 0 in isolation.

    POST /v1/troubleshoot
    {
      "query": "<raw customer complaint>",
      "siis_response": {"title": "...", "content": "..."}
    }

    -> {
      "query": "<raw customer complaint>",
      "query_variations": [ ... 8-10 paraphrases from Stage 0 ... ],
      "response": {"contexts": [ ... Goal objects, per schema.py ... ]},
      "meta": {"cache_hit": bool, "similarity": float | None,
                "latency_ms": float, "model": str, "cost_usd": float},
      "enrichment": {"device", "symptom_category", "device_confidence",
                     "symptom_confidence", "overall_confidence", "is_low_confidence",
                     "classification_source"}
    }

§4.2.3 of the brief is explicit and non-negotiable: "If the reference data
contains no viable solution, the engine must return an empty list
(contexts: []) with fallback metadata (\"fallback\": \"no_match\")" — and
the roadmap (§8, Phase 4) names a second reason, "no_siis_context", for
when there was no siis_response to work from at all. Both are added to
`response` (alongside "contexts") whenever Stage 1/2 come back empty,
distinguishing "nothing to look at" from "looked, found nothing".

"meta.cost_usd" is honestly reported as 0.0 always: none of the LLMClient
implementations in llm_client.py currently capture token usage from the
provider response, so computing a real per-provider $ figure would mean
fabricating one — reporting 0.0 is the accurate statement of what's
actually tracked today, not a claim that inference is free. Documented as
a known gap in README.md rather than silently faked.

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
from llm_client import get_llm_client
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
        llm_client=None,
    ):
        self.cache = cache or SemanticCache(TfidfEmbedder.load())
        self.stage1_fn = stage1_fn
        self.stage2_fn = stage2_fn
        # Optional explicit injection (tests, or an app wiring a specific client).
        # If not given, enrich() falls back to get_llm_client(), which reads
        # LLM_PROVIDER from the environment — so the common case (just set the
        # env var, no code changes) keeps working unchanged.
        self.llm_client = llm_client

    def run(self, query: str, siis_response: dict) -> dict:
        t0 = time.perf_counter()

        # Resolved once (not inside enrich()) so meta.model can report the
        # same client actually used for this request without a second,
        # redundant resolution — get_llm_client() is cached by (provider,
        # model) so this costs nothing extra either way, but resolving once
        # here keeps "which client answered" and "what enrich() used" the
        # same object by construction, not by coincidence.
        client = self.llm_client or get_llm_client()

        # Stage 0
        enrichment = enrich(query, llm_client=client)
        enrichment_info = {
            "device": enrichment.device,
            "symptom_category": enrichment.symptom_category,
            "device_confidence": enrichment.device_confidence,
            "symptom_confidence": enrichment.symptom_confidence,
            "overall_confidence": enrichment.overall_confidence,
            "is_low_confidence": enrichment.is_low_confidence,
            "classification_source": enrichment.classification_source,
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
                "query_variations": enrichment.query_variations,
                "response": cache_hit.response,
                "meta": {
                    "cache_hit": True,
                    "similarity": cache_hit.similarity,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "model": client.model_name,
                    "cost_usd": 0.0,
                },
                "enrichment": enrichment_info,
            }

        # Stage 1 + Stage 2 (Member B / Member C)
        structured = self.stage1_fn(siis_response, enrichment)
        final = self.stage2_fn(structured, enrichment)
        response_dict = final.model_dump()

        # §4.2.3 (non-negotiable): an empty result must carry fallback
        # metadata, not just a bare empty list — "no_siis_context" when
        # there was nothing to extract from at all, "no_match" when Stage
        # 1/2 ran against real reference text but found no viable solution
        # (§8 Phase 4 names both reasons explicitly). Baked into
        # response_dict before caching so a later cache HIT on this same
        # (rare — see the confidence gate below) entry still carries it.
        if not response_dict.get("contexts"):
            response_dict["fallback"] = "no_match" if siis_response else "no_siis_context"

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
            "query_variations": enrichment.query_variations,
            "response": response_dict,
            "meta": {
                "cache_hit": False,
                "similarity": None,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "model": client.model_name,
                "cost_usd": 0.0,
            },
            "enrichment": enrichment_info,
        }
