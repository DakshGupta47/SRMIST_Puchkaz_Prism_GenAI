# Performance Report — Stage 0 (Query Enrichment) + Stage 3 (Semantic Cache)

Generated from `scripts/benchmark_cache.py`, run against `artifacts/tfidf_vectorizer.pkl`
with `LLM_PROVIDER` unset (deterministic template path — no network calls in this run).
Raw numbers: `artifacts/benchmark_results.json`. Reproduce with:

```bash
cd theme2_submission && python scripts/benchmark_cache.py
```

## 1. Cache hit rate on unseen paraphrases

The brief's headline KPI. Two experiments, because the kit's own test data mixes
together two different questions and this report doesn't want to blur them:
*does the similarity mechanism generalize across phrasing?* vs. *does a cold cache
happen to already contain the right category/device?*

### 1a. Realistic run — `data/input.txt` (seed/"already answered") → `data/unseen_scenarios.txt` (scored)

| Metric | Value |
|---|---|
| Unseen scenarios (total) | 38 |
| Skipped — `is_low_confidence` (never enters/exits cache, by design) | 17 |
| Scored | 21 |
| Hits | 0 |
| **Hit rate** | **0.0%** — well below the ≥80% target |

**This is a real result, not spot-checked**, and it does **not** mean the similarity
matching is broken (see 1b). It means the *cache was cold* for almost every scored
query: `input.txt`'s 20 samples are all screen/display symptoms, while
`unseen_scenarios.txt` deliberately spans battery, connectivity, audio, camera,
storage, apps, biometrics, notifications and software-update categories that
`input.txt` never seeds. A query can't hit a cache entry that was never written for
its category — first-occurrence misses in a 20-entry seed set covering 36 categories
are structural, not a matching failure.

Two further, more actionable causes surfaced by this run:

- **Cross-device gating (by design).** `cache.py` never serves a hit across devices
  (see its module docstring). Several misses are same-category, different-device
  pairs — e.g. `no_signal_network` seen for a Galaxy S21 (`data/unseen_scenarios.txt`
  line 6), queried again for a Galaxy A15 (line 18): correctly gated apart, but it
  means the ≥80% target is only reachable once the cache has been primed **per
  device**, not per category.
- **Device-shorthand regression (a real, fixable gap).** `extract_device()` requires
  a `Galaxy`/`Samsung`/`SM-` prefix; bare shorthand like `"s22"` doesn't match, so
  `device_confidence` falls to 0.0 and `is_low_confidence` becomes `True` —
  bypassing the cache entirely regardless of how confidently the symptom itself
  classified. Confirmed directly:

  ```pycon
  >>> enrich("my s22 battery dying so fast wtf")
  device='Samsung device' device_confidence=0.0 symptom_category='battery_drain'
  symptom_confidence=0.75 is_low_confidence=True
  ```

  17 of `unseen_scenarios.txt`'s 38 lines are skipped for exactly this reason
  (bare device shorthand, or genuinely no device named at all, e.g. `"battery
  drain"`, `"phone wont charge"`). This is flagged here as a known gap, not
  silently patched in this pass — extending the device regex to accept bare
  model numbers (`"s22"`, `"a15"`) risks new false positives on unrelated text
  and deserves its own review.

### 1b. Controlled experiment — does register generalization itself work?

Isolates the one variable 1a couldn't cleanly separate: with device and category
held constant and a cache entry already present, does a *different register*
(formal vs. casual phrasing, per `SYMPTOM_TAXONOMY`) of the same complaint hit?

| Metric | Value |
|---|---|
| Categories scored (formal → cache, casual → query, same device) | 23 / 36 |
| Categories excluded (formal/casual template text doesn't self-classify — see §3) | 12 |
| **Hit rate** | **100.0%** (23/23) |

So the TF-IDF similarity mechanism and the 0.80 threshold comfortably clear the
≥80% bar **once a same-device entry exists for the category** — the gap in §1a is
entirely a cold-start / device-recognition problem, not a similarity-matching one.

### Net assessment

The ≥80% claim is **not validated as an end-to-end guarantee on the provided kit
data** — it depends on the cache already holding an entry for the query's specific
device, and the device-shorthand gap makes that harder to reach than it should be.
It **is validated** as a property of the matching mechanism itself, given a
same-device seed. Recommended before claiming the KPI unconditionally: fix or flag
the device-shorthand gap, and note the per-device (not per-category) priming
requirement in the submission write-up.

## 2. Latency

| Path | p50 | p95 | p99 | max | n |
|---|---|---|---|---|---|
| `cache.get()` (cache-hit fast path) | 2.08 ms | 2.49 ms | 2.71 ms | 2.71 ms | 21 |
| `enrich()` (Stage 0, deterministic template path) | 0.14 ms | 0.18 ms | — | 0.55 ms | 58 |

Both are far inside the ≤300 ms cache-hit budget named in the brief — expected,
since this run has `LLM_PROVIDER` unset, so neither path makes a network call.
**Not measured here:** end-to-end `/v1/troubleshoot` latency on a cache *miss*,
which additionally pays for Stage 1 (LLM structuring, Member B) and Stage 2
(deeplink mapping, Member C) — those aren't wired into this slice
(`pipeline.py`'s `_stage1_not_wired`/`_stage2_not_wired` placeholders), so an
end-to-end number would be misleading to publish from Member A's slice alone.
**Not measured here either:** latency with a real `LLM_PROVIDER` configured
(the `llm_fallback`/unclassified path) — `_LLM_TIMEOUT_SECONDS = 8` in
`llm_client.py` bounds its worst case, but this network has no reachable
provider to measure the typical case against.

## 3. Coverage / taxonomy gap surfaced by this benchmark

12 of 36 categories don't classify their **own** `formal`/`casual` template text
back to themselves via `extract_symptom()`'s keyword scoring (ties broken by
taxonomy order favor an earlier, more general entry):

```
screen_flicker_then_blank, keyboard_typing_problem, hardware_button_unresponsive,
half_screen_dark, inner_screen_failure, screen_partial_lit, screen_undersized,
overheating, random_restarts, sluggish_performance, mobile_data_connectivity,
no_notifications
```

This is a genuine taxonomy-overlap finding (not a benchmark artifact) worth a
follow-up pass on `SYMPTOM_TAXONOMY`'s keyword lists and tie-breaking order —
tracked here rather than silently excluded from the headline number.

## 4. Cost tracking

`meta.cost_usd` is reported as `0.0` for every response (see `pipeline.py`'s own
docstring: no `LLMClient` implementation currently captures token usage from the
provider response, so any non-zero figure would be fabricated, not measured).
Actual LLM spend in this deployment slice is bounded by `_LLM_TIMEOUT_SECONDS = 8`
per call and by `generate_variations()`'s keyword-match gate, which skips the LLM
paraphrase call entirely for confidently-classified queries (~1s/call saved, live-
tested — see `enrichment.py`). With a real provider configured, `llm_fallback`/
`unclassified` queries pay for exactly one combined classify+paraphrase call per
request (see `_llm_classify_and_paraphrase` in `enrichment.py` — merged from two
round-trips into one in this pass); `keyword_match` queries pay for zero.

## 5. Determinism

Every real `LLMClient` implementation (`Gemini`/`OpenAI`/`Anthropic`) now sets
`temperature=0` and `_TimeoutGuardedClient` additionally memoizes successful
responses per exact `(system_prompt, user_prompt)` pair, so an identical request
within a process returns an identical answer by construction — addressing §6's
"Deterministic Execution: consistent action plans for identical or semantically
identical inputs" criterion. Not exercised in this benchmark run (`LLM_PROVIDER`
unset throughout), so provider-side behavior at `temperature=0` is unverified
against a live API from this environment (network egress here doesn't reach
Gemini/OpenAI/Anthropic — see `README.md`'s "Why TF-IDF" section).
