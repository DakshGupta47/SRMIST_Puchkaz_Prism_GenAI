"""
Integration tests for pipeline.py — specifically the confidence gate on
the cache, which exists to prevent a concrete hallucination risk: two
different low-confidence (unclassifiable) queries for the same device
normalize to nearly identical canonical text, so caching either one's
answer would let it leak into the other.
"""
from cache import SemanticCache
from embeddings import TfidfEmbedder
from pipeline import Pipeline
from schema import Action, ContextDeeplinkResponse, Goal, StepGroup


def _fake_confident_stage1(siis_response: dict, enrichment) -> ContextDeeplinkResponse:
    return ContextDeeplinkResponse(contexts=[
        Goal(
            goal="Fix it",
            title=enrichment.symptom_label,
            score=0.9,
            actions=[Action(
                actionName="Do the thing",
                description="does the thing",
                stepGroups=[StepGroup(steps=["step one"])],
            )],
        )
    ])


def _identity_stage2(structured: ContextDeeplinkResponse, enrichment) -> ContextDeeplinkResponse:
    return structured


def _fresh_pipeline() -> Pipeline:
    cache = SemanticCache(TfidfEmbedder.load())
    return Pipeline(cache=cache, stage1_fn=_fake_confident_stage1, stage2_fn=_identity_stage2)


def test_confident_query_is_cached_and_hit_on_repeat():
    pipeline = _fresh_pipeline()
    q = "My Galaxy S22 battery drains extremely fast, dead by noon even with light use."

    first = pipeline.run(q, {"title": "t", "content": "c"})
    assert first["cache_hit"] is False
    assert first["enrichment"]["is_low_confidence"] is False
    assert first["enrichment"]["classification_source"] == "keyword_match"
    assert len(pipeline.cache) == 1

    second = pipeline.run("galaxy s22 battery dies so fast, gone by lunch", {"title": "t", "content": "c"})
    assert second["cache_hit"] is True


def test_low_confidence_query_is_never_cached():
    pipeline = _fresh_pipeline()
    ambiguous = "asdkjfhaskdjfh qwerty"

    result = pipeline.run(ambiguous, {"title": "t", "content": "c"})
    assert result["cache_hit"] is False
    assert result["enrichment"]["is_low_confidence"] is True
    assert len(pipeline.cache) == 0  # nothing written


def test_llm_fallback_classified_query_still_never_enters_the_cache():
    """The guarded LLM fallback in enrichment.py (see
    test_llm_fallback_classifies_a_complaint_the_keyword_matcher_could_not)
    gives Stage 1/2 a real category instead of the generic unclassified text
    — but it's deliberately scored below the confidence threshold, so it
    must still be invisible to pipeline.py's cache gate. Proving that
    composition end-to-end rather than trusting it by inspection: a fake LLM
    resolves the category, Stage 1/2 run and get a real answer, but nothing
    is cached and a repeat of the exact same query recomputes fresh rather
    than serving a (potentially wrong, LLM-guessed) cached response.
    """
    import json
    from enrichment import enrich
    from llm_client import LLMClient

    class FakeLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            if "category" in system_prompt:
                return json.dumps({"category": "keyboard_typing_problem"})
            return json.dumps([])  # variation-generation call: contribute nothing extra

    cache = SemanticCache(TfidfEmbedder.load())
    fake_llm = FakeLLM()
    pipeline = Pipeline(
        cache=cache, stage1_fn=_fake_confident_stage1, stage2_fn=_identity_stage2, llm_client=fake_llm,
    )
    # phrased to still miss the keyword taxonomy (no "keyboard" or "when I
    # type"-style phrase, which enrichment.py's taxonomy now catches directly)
    # so this still exercises the LLM fallback path being tested here.
    query = "My Galaxy S23 messages come out full of random symbols instead of the letters I actually pressed."

    # sanity check the fallback actually fires for this query before trusting the pipeline result
    enrichment = enrich(query, llm_client=fake_llm)
    assert enrichment.classification_source == "llm_fallback"
    assert enrichment.symptom_category == "keyboard_typing_problem"

    result = pipeline.run(query, {"title": "t", "content": "c"})
    assert result["enrichment"]["is_low_confidence"] is True
    assert result["response"]["contexts"][0]["title"] == enrichment.symptom_label  # Stage 1 got the real category
    assert result["cache_hit"] is False
    assert len(pipeline.cache) == 0  # the whole point: llm_fallback never gets cached


def test_two_unrelated_low_confidence_queries_never_cross_contaminate():
    """The concrete hallucination risk this gate exists for: two genuinely
    different, both-unclassifiable complaints for the same (unknown)
    device normalize to nearly identical canonical text. Without the
    confidence gate, the second call would get served the first call's
    unrelated stage1/2 answer as a cache 'hit'.
    """
    call_count = {"n": 0}

    def _stage1_tagged(siis_response: dict, enrichment) -> ContextDeeplinkResponse:
        call_count["n"] += 1
        return ContextDeeplinkResponse(contexts=[
            Goal(goal=f"answer-{call_count['n']}", title="t", score=0.5, actions=[])
        ])

    cache = SemanticCache(TfidfEmbedder.load())
    pipeline = Pipeline(cache=cache, stage1_fn=_stage1_tagged, stage2_fn=_identity_stage2)

    r1 = pipeline.run("my phone does a weird thing sometimes idk", {"title": "t", "content": "c"})
    r2 = pipeline.run("something is off with it but not sure what", {"title": "t", "content": "c"})

    assert r1["response"]["contexts"][0]["goal"] == "answer-1"
    assert r2["response"]["contexts"][0]["goal"] == "answer-2"  # NOT answer-1 — no cross-contamination
    assert r2["cache_hit"] is False
    assert len(cache) == 0
