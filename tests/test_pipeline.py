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
