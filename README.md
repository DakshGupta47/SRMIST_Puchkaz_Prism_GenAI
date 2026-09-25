# Theme 02 — Smart Guided Troubleshooting Engine — Member A's slice

Owns **Stage 0 (query enrichment)** and **Stage 3 (semantic cache)** of the
team's pipeline: normalize a raw customer complaint, generate 8-10
paraphrases across registers, and serve a previously-validated answer
from cache in under 300ms when a semantically-equivalent query comes back.

## Layout

```
src/
  enrichment.py    Stage 0 — normalize_query / generate_variations / enrich()
  cache.py         Stage 3 — SemanticCache (embedding-based, device-gated)
  embeddings.py    embedding backend (see "Why TF-IDF, not sentence-transformers" below)
  llm_client.py    pluggable LLM client (Gemini/OpenAI/Anthropic/offline mock)
  pipeline.py      wires Stage 0 + Stage 3 together; extension points for Stage 1/2
  api.py           FastAPI app exposing POST /v1/troubleshoot, /v1/enrich, /v1/cache/stats
  schema.py        team's response schema (copied from the input kit, unmodified)
data/              copied from participant-kit/Theme02_Input_Kit/student_kit
artifacts/         tfidf_vectorizer.pkl (generated — see below)
scripts/build_corpus_vectorizer.py   fits + saves the vectorizer
tests/             pytest suite: 71 tests, enrichment + cache + latency + no-hallucination
```

## Setup

```bash
pip install -r requirements.txt
python scripts/build_corpus_vectorizer.py   # fits artifacts/tfidf_vectorizer.pkl once
pytest tests/ -v
cd src && uvicorn api:app --reload --port 8000
```

## Why TF-IDF instead of a sentence-transformer

The plan going in was a small local sentence-transformer (all-MiniLM-L6-v2
via `sentence-transformers`/`fastembed`) for the cache's embeddings. On
this network, `huggingface.co` is blocked by the org's egress allowlist
(confirmed: 403 at the proxy, both from my machine and from a cloud
sandbox I tested from) — `pypi.org` works, model downloads don't. Rather
than block on that, `embeddings.py` fits a **character n-gram TF-IDF
vectorizer** on the domain corpus (`data/siis_responses.json` +
`data/deeplinks.json` + `data/input.txt`) once, at build time
(`scripts/build_corpus_vectorizer.py`), and uses cosine similarity over
those vectors as the "embedding." No network needed at runtime, it's
fast (see latency numbers below), and char n-grams (not word n-grams)
make it naturally robust to typos and short keyword-style queries —
which matters since that's exactly the register spread Stage 0 has to
produce and Stage 3 has to match against.

`embeddings.py`'s `Embedder` protocol is the seam: if anyone gets a real
embedding model working on a network that can reach HuggingFace, drop in
a class with `encode()`/`encode_sparse()` and nothing in `cache.py` or
`pipeline.py` needs to change.

## Stage 0 — enrichment.py

- `normalize_query(raw)`: regex-extracts the device model and matches
  against a symptom taxonomy of ~26 categories spanning screen (the 20
  `data/input.txt` samples), battery, charging, overheating, random
  restarts, sluggish performance, Wi-Fi, Bluetooth, no-signal, audio
  (no-sound / distorted-crackling), camera (crashes / blurry), storage,
  app crashing, fingerprint/face unlock, missing notifications, and
  software-update failures. Deterministic — it cannot invent a device or
  symptom that isn't in the text.
- Matching is (component word present) **and** (problem word present),
  not one long exact phrase — see "Two real bugs" below for why that
  distinction mattered.
- `generate_variations(...)`: produces 8-10 paraphrases across
  formal/casual/keyword/frustrated/typo registers. Template-based by
  default (always valid, zero API keys required); if `LLM_PROVIDER` is
  set to `gemini`/`openai`/`anthropic` (see `llm_client.py`), extra
  LLM-generated paraphrases are mixed in **after** being validated
  (length, must reference the actual device/symptom) so a bad LLM call
  degrades quality, never correctness.
- Verified against all 20 sample complaints in `data/input.txt` *and* a
  38-scenario unseen set (`data/unseen_scenarios.txt`, covering every
  non-screen category plus edge cases: bare keyword complaints, ALL CAPS,
  heavy typos, compound multi-symptom complaints, unrecognized product
  lines, a false-alarm/no-issue message) — see
  `tests/test_enrichment.py` and the "Two real bugs" section.

### Two real bugs this surfaced (found by testing beyond the 20 given samples)

1. **Hallucinated symptom on non-screen complaints.** The taxonomy
   originally only covered screen symptoms, so *any* non-screen complaint
   — a battery drain report, say — fell through to a default that
   literally said `"screen is exhibiting a display issue"`. That's not a
   cosmetic bug: `data/deeplinks.json` is only ~1/3 screen-related (57
   battery, 69 sound, 67 notification, 41 software-update, 26 security,
   19 network, 16 performance/accessibility/Wi-Fi each, 12 Bluetooth, 9
   camera, 7 storage deeplinks out of 578 total), so an unseen judge-run
   scenario hitting any of those categories was a likely, not edge-case,
   failure. Fixed by adding a `subject` field per symptom (inserted by
   the template instead of a hardcoded "screen") and building out the
   taxonomy to match the catalog's actual breadth.
2. **Exact-phrase keyword matching silently dropped realistic
   rephrasings.** An earlier version required e.g. `"camera app crashes"`
   verbatim and missed `"the camera app... keeps crashing"` (different
   word order/tense) — 7 of 16 realistic non-screen test complaints
   landed in `unclassified_issue` before this was fixed. It also had a
   literal substring collision: `"crackly"` contains `"crack"`, so a
   Bluetooth earbuds audio complaint was misclassified as a *cracked
   screen*. Fixed by scoring each category on (component word present)
   **and** (problem word present) instead of one exact phrase — a
   part-specific category (camera, battery, Wi-Fi, ...) is only eligible
   at all once its component word is seen, which also structurally
   prevents the crack/crackly-style collision.

**Known remaining limitation**: heavy typo corruption (`"skreen"`,
`"blenk"`, `"trun on"` for screen/blank/turn on) can still defeat
substring/stem matching and fall back to `unclassified_issue` — see
`test_heavy_typos_are_a_known_limitation_not_a_crash`. A bare one-word
complaint like `"Battery"` also stays `unclassified_issue` on purpose:
there's no problem word to classify, and guessing a specific failure mode
from the noun alone would itself be a hallucination.

### Confidence signal — "don't answer for the sake of answering"

`EnrichmentResult` carries `device_confidence` (1.0 if a specific model was
recognized, 0.0 for the generic fallback), `symptom_confidence` (0.0 for
`unclassified_issue`, otherwise a saturating score from keyword-match
strength), and `overall_confidence`/`is_low_confidence` derived from the
**weaker** of the two — a confidently-identified device does not offset a
genuinely unclassifiable symptom, or vice versa. This is exposed all the
way out through `pipeline.py`'s response (`"enrichment": {...}`), not kept
internal, so Stage 1/2, an evaluator, or a demo UI can see when Stage 0
genuinely doesn't know what it's looking at rather than treating every
200 response as equally confident.

This also closes a concrete hallucination risk in the cache, not just a
reporting nicety: **every unclassifiable query normalizes to nearly
identical canonical text** regardless of what was actually said (device +
`"is experiencing an issue that could not be automatically classified..."`).
Caching Stage 1/2's answer under that key would let one ambiguous
customer's answer leak into a completely unrelated future ambiguous
query — a real cache-induced hallucination, not a Stage 1/2 bug.
`pipeline.Pipeline.run()` skips both the cache read and the cache write
whenever `enrichment.is_low_confidence` is true, so a low-confidence query
never gets the ≤300ms fast path (it always re-runs Stage 1/2 fresh) — see
`tests/test_pipeline.py::test_two_unrelated_low_confidence_queries_never_cross_contaminate`
for the exact scenario this prevents.

## Stage 3 — cache.py

- `SemanticCache.put(canonical_query, query_variations, response, device=...)`
  stores a validated response (whatever Stage 1+2 produced) keyed by the
  embeddings of the canonical query and every variation.
- `SemanticCache.get(query, query_variations, device=...)` returns the
  cached response if any stored vector clears the similarity threshold
  (default 0.80) **and** the device matches — otherwise `None`.
- **Device gate is load-bearing, not cosmetic**: "Galaxy S22 screen is
  black" vs "Galaxy S24 screen is black" score ~0.9 similarity from text
  alone (the symptom text dominates a couple of differing digits), but
  different devices can have different correct deeplinks/steps. Device
  is checked as a hard precondition before similarity is even considered
  — see `test_same_symptom_different_device_never_cross_served` in
  `tests/test_cache.py`.
- **Low-confidence → `None`, never a guess.** An unrelated query, or a
  same-device-different-symptom query, returns `None` rather than the
  nearest cached entry.
- **Latency**: one vectorized sparse dot product across every stored
  vector at once (not a per-entry Python loop — an earlier version of
  this that recomputed vector norms per entry per lookup blew the
  budget at ~1500 entries; fixed by pre-normalizing on write and doing
  the whole comparison as a single sparse matmul). Measured in
  `tests/test_cache.py::test_cache_hit_latency_budget_at_scale` (500
  entries) and manually up to 5000 entries:

  | cache size | mean | p95 | p99 | max |
  |---|---|---|---|---|
  | 500 | 11ms | 15ms | 17ms | 22ms |
  | 1500 | 32ms | 37ms | 45ms | 59ms |
  | 5000 | 105ms | 120ms | 138ms | 230ms |

  Comfortably inside the 300ms budget at the scale this hackathon
  actually runs at (20 primary scenarios + whatever unseen set gets run
  against it). If the cache ever needs to hold tens of thousands of
  entries, the `semantic_cache_key` SimHash bucketing that's already in
  the module (currently used just to tag/identify vectors) is the next
  lever — narrow the matmul to a bucket instead of the whole matrix.

## Stage 1 / Stage 2 — where you plug in

`pipeline.py` defines the extension points:

```python
Stage1Fn = Callable[[dict, EnrichmentResult], ContextDeeplinkResponse]   # Member B
Stage2Fn = Callable[[ContextDeeplinkResponse, EnrichmentResult], ContextDeeplinkResponse]  # Member C

pipeline = Pipeline(stage1_fn=your_structure_extraction_fn, stage2_fn=your_deeplink_mapping_fn)
```

`EnrichmentResult` (passed into both) already carries `canonical_query`,
`device`, `symptom_category`/`symptom_label`, and `query_variations`, so
neither of you needs to re-derive normalization or device extraction.
Until you wire your functions in, `/v1/troubleshoot` runs Stage 0 → cache
check → (on miss) returns an explicitly empty `{"contexts": []}` rather
than inventing steps, so the incomplete state is obvious instead of
silently wrong.

## API

```
POST /v1/troubleshoot   {"query": "...", "siis_response": {"title": "...", "content": "..."}}
                         -> {"query", "response": {"contexts":[...]}, "cache_hit", "similarity",
                             "latency_ms", "enrichment": {"device", "symptom_category",
                             "device_confidence", "symptom_confidence", "overall_confidence",
                             "is_low_confidence"}}
POST /v1/enrich          {"query": "..."}  -> Stage 0 output only, for standalone testing
GET  /v1/cache/stats     -> {"entries", "similarity_threshold"}
```

Request/response shape follows `data/siis_responses.json`'s own readme
note ("`siis_response` is the payload your API must accept in
`POST /v1/troubleshoot`") plus the shape of `data/sample_output.json`.

## Known gaps / honest limitations

- No LLM key was configured when this was built, so Stage 0's variation
  generation runs entirely on the deterministic template path in
  practice. Set `LLM_PROVIDER`/the matching API key env var any time —
  no code changes needed.
- The symptom taxonomy in `enrichment.py` covers the patterns seen in
  the 20 sample complaints; an unseen symptom category falls back to a
  generic "screen issue" label rather than guessing a specific one (by
  design — no hallucinated symptom).
- The cache's brute-force-at-scale numbers above assume a single
  process/thread; it isn't thread-lock-protected for concurrent writes.
  Fine for the hackathon's demo/eval harness; flag if the real deployment
  needs concurrent request handling.
