# Repository Audit: Smart Guided Troubleshooting Engine (FastAPI)

**Date**: 2026-09-29  
**Repository**: `SRMIST_Puchkaz_Prism_GenAI`  
**Scope**: Codebase audit against the Smart Guided Troubleshooting Engine theme brief, pipeline architecture, and implementation status.

---

## 1. Pipeline Stages in `src/` & Wiring Status

### 1.1 Existing Pipeline Stages

| Pipeline Stage | Module(s) | Status | Details |
|---|---|---|---|
| **Stage 0: Query Enrichment** | `src/enrichment.py` | **Implemented** | Extracts device model (`extract_device`), categorizes symptoms into 35 taxonomy classes with keyword scoring (`normalize_query`), generates 8–10 paraphrases across 5 registers (formal, casual, keyword, frustrated, typo via `generate_variations`), computes confidence scores (`is_low_confidence`), and includes a guarded LLM fallback classifier (`_llm_classify_symptom`). |
| **Stage 1: Structure Extraction** | `src/structure_extraction.py`, `src/relevance_gate.py` | **Implemented** | Evaluates reference relevance via `assess_reference_relevance`, formats prompt for LLM structuring, parses JSON, and validates complete structure: Goal naming pattern, 2–3 word titles, score in [0, 1], action descriptions starting with "It will" (5–7 words), step grounding in source text, critical actions ordered last, and verifies actionable/validation deeplinks are `None` (deferred to Stage 2). |
| **Stage 2: Deeplink Mapping** | *Missing* (Placeholder in `src/pipeline.py`) | **Missing** | No dedicated module (`src/stage2_deeplinks.py`) exists. `src/pipeline.py` defines `Stage2Fn = Callable[[ContextDeeplinkResponse, EnrichmentResult], ContextDeeplinkResponse]` and a passthrough stub `_stage2_not_wired(structured, enrichment)` which returns the structured response with all deeplinks left as `None`. |
| **Stage 3: Semantic Cache** | `src/cache.py`, `src/embeddings.py` | **Implemented** | Embedding backend using character n-gram TF-IDF (`TfidfEmbedder`, saved in `artifacts/tfidf_vectorizer.pkl`) with cosine similarity via normalized sparse matmul. Hardware device gate (`_normalize_device`) prevents cross-device cache hits. Thread-safe with `threading.Lock`. Low-confidence queries bypass cache read/write to avoid cache poisoning. |

### 1.2 Pipeline Wiring in `src/pipeline.py` & `src/api.py`

- **In `src/pipeline.py#L129-L145`**:
  `Pipeline.__init__` declares:
  ```python
  def __init__(
      self,
      cache: Optional[SemanticCache] = None,
      stage1_fn: Stage1Fn = structure_extraction,
      stage2_fn: Stage2Fn = _stage2_not_wired,
      llm_client=None,
  ):
  ```
  `stage1_fn` defaults to the real `structure_extraction` function from `src/structure_extraction.py`. However, `stage2_fn` defaults to `_stage2_not_wired`, which is a stub passthrough that attaches no deeplinks.

- **In `src/api.py#L31-L33`**:
  ```python
  app = FastAPI(title="Smart Guided Troubleshooting Engine — Member A slice")
  pipeline = Pipeline()
  ```
  **Wiring status in `api.py`**:
  `Pipeline()` is instantiated with default arguments. It is **NOT** explicitly wired with `stage1_fn` or `stage2_fn`. While `Pipeline` defaults `stage1_fn` to `structure_extraction`, `stage2_fn` remains the un-wired placeholder (`_stage2_not_wired`). No real or standalone Stage 2 deeplink mapping function is wired into the FastAPI service.

---

## 2. Requirement Checklist (Theme Brief)

| Category | Requirement | Status | Verification & Code References |
|---|---|---|---|
| **Endpoints** | `POST /v1/troubleshoot` | **DONE** | Implemented in `src/api.py#L69-L72`. Accepts `query` and `siis_response` (`title`, `content`), executes `pipeline.run()`. |
| | `POST /v1/enrich` | **DONE** | Implemented in `src/api.py#L75-L80`. Exposes Stage 0 normalization and variation generation standalone. |
| | `GET /v1/cache/stats` | **DONE** | Implemented in `src/api.py#L83-L85`. Returns entry count and similarity threshold. |
| | `GET /health` | **DONE** | Implemented in `src/api.py#L88-L95`. Exact spec requirement (§5). Returns `{"status": "ok"}`. |
| | `GET /healthz` | **DONE** | Implemented in `src/api.py#L98-L103`. Infrastructure alias. |
| **Response Schema** | Top-level `query` & `query_variations` | **DONE** | Implemented in `src/pipeline.py#L181-L193`, `src/pipeline.py#L222-L234`. 8–10 paraphrases generated across 5 registers. |
| | Response envelope `response` (`contexts`) | **DONE** | Conforms to Pydantic models in `src/schema.py` (`ContextDeeplinkResponse`). |
| | Metadata envelope `meta` | **DONE** | Contains `cache_hit`, `similarity`, `latency_ms`, `model`, `cost_usd`. |
| | Goal structure & naming format | **DONE** | Validated in `src/structure_extraction.py#L344-L355`: Goal follows `Follow these steps to perform this <Topic> (Troubleshooting\|Configuration)`. |
| | Goal title word count | **DONE** | Validated in `src/structure_extraction.py#L268-L273`: Exactly 2 to 3 words. |
| | Goal score range | **DONE** | Validated in `src/structure_extraction.py#L481-L487`: Must be between 0.0 and 1.0. |
| | Action description format & word count | **DONE** | Validated and normalized in `src/structure_extraction.py#L274-L343`: Starts with "It will" and has 5 to 7 words. |
| | Action categorization & order | **DONE** | Enum in `src/schema.py#L32-L36` (`auto`, `manual`, `critical`). Critical actions validated to always appear after auto/manual actions (`src/structure_extraction.py#L493-L513`). |
| | Step grounding | **DONE** | Validated in `src/structure_extraction.py#L391-L449`: Steps must be grounded in source text. |
| | Deeplink mapping (`actionableDeeplink`, `validationDeeplink`) | **MISSING** | Stage 1 sets deeplinks to `None`. Stage 2 is not yet implemented, leaving deeplinks unpopulated. |
| **Fallbacks** | §4.2.3: No viable solution fallback (`fallback: "no_match"`) | **DONE** | Implemented in `src/pipeline.py#L210-L211`. When SIIS response is present but contexts are empty, `response.fallback` is set to `"no_match"`. |
| | §8 Phase 4: Missing SIIS context fallback (`fallback: "no_siis_context"`) | **DONE** | Implemented in `src/pipeline.py#L210-L211`. When no `siis_response` is provided and contexts are empty, `response.fallback` is set to `"no_siis_context"`. |
| | Low-confidence query isolation | **DONE** | Implemented in `src/pipeline.py#L176-L179`, `src/pipeline.py#L216-L220`. Unclassified queries bypass cache read and write to prevent false-positive cross-serving. |
| | Exception boundary / No raw traceback | **DONE** | Implemented in `src/api.py#L35-L52`. Global exception handler traps all exceptions and returns HTTP 500 JSON without leaking stack traces or internal paths. |
| **Zero-URL-Leak Rule** | Zero web URLs in steps, descriptions, and deeplinks | **DONE** | Implemented in `src/scrubber.py` (`scrub_response`), wired into `src/pipeline.py#L203` (applied after Stage 2 before caching and returning). Recursively sanitizes user-visible text fields, disallows non-deeplink schemes, rejects web URLs in deeplink fields, and preserves legitimate deeplinks without mutating input. Tested in `tests/test_scrubber.py`. |
| **results.jsonl** | `results.jsonl` export format / generation script | **MISSING** | No `results.jsonl` file or evaluation execution script (e.g. iterating over `data/input.txt` and `data/siis_responses.json` to emit JSONL evaluation results) exists in the repository. |
| **Submission Rules** | Self-contained & offline capable | **PARTIAL** | Stage 0 and Stage 3 run 100% offline via TF-IDF vectorizer and deterministic rule templates. However, `MockLLMClient.complete()` in `src/llm_client.py` merely echoes `user_prompt`, which causes `_extract_json()` in Stage 1 (`src/structure_extraction.py`) to fail when no real LLM key is configured. |
| | Latency budget (cache hit ≤ 300ms) | **DONE** | Verified in `metrics.md` and `tests/test_cache.py` (~2–3ms per cache hit). |
| | Clean git state (no committed secrets) | **DONE** | `.env` is gitignored; working tree changes remain uncommitted per instruction. |

---

## 3. Summary of Gaps to Address

1. **Stage 2 Module**: Create `src/stage2_deeplinks.py` implementing `map_deeplinks(result, enrichment) -> ContextDeeplinkResponse` matching `Stage2Fn`.
2. **Pipeline Wiring in `src/api.py`**: Explicitly construct `Pipeline(stage1_fn=structure_extraction, stage2_fn=map_deeplinks)`.
3. **Integration Testing**: Add integration tests asserting that end-to-end execution of a real sample from `data/input.txt` paired with `data/siis_responses.json` produces non-empty contexts, passes `src/schema.py` validation, and carries the stub deeplinks.
4. **Zero-URL-Leak Validation & results.jsonl**: Recommended for downstream roadmap completion.
