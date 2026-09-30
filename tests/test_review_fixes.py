"""Regression tests for the pre-submission review fixes."""
import json

from cache import SemanticCache
from embeddings import TfidfEmbedder
from llm_client import LLMClient, _TimeoutGuardedClient
from pipeline import Pipeline
from schema import Action, ContextDeeplinkResponse, Goal, StepGroup, actionCategory


def _stage1(siis_response, enrichment):
    if not siis_response:
        return ContextDeeplinkResponse(contexts=[])
    return ContextDeeplinkResponse(contexts=[Goal(
        goal="Follow these steps to perform this Screen Troubleshooting", title="Screen issue", score=0.9,
        actions=[Action(actionName="A", description="It will do the thing.",
                        stepGroups=[StepGroup(steps=[siis_response.get("content", "")])])],
    )])


def _pipe():
    return Pipeline(cache=SemanticCache(TfidfEmbedder.load()), stage1_fn=_stage1, stage2_fn=lambda s, e: s)


def test_timeout_wrapper_reports_inner_provider_cost():
    class Paid(LLMClient):
        def complete(self, system_prompt, user_prompt):
            self._record_usage(1000, 1000, 1e-6, 1e-6)
            return "x"
    c = _TimeoutGuardedClient(Paid())
    c.complete("s", "u")
    assert abs(c.consume_cost() - 0.002) < 1e-12
    assert c.consume_cost() == 0.0


def test_empty_no_siis_answer_is_not_cached_and_does_not_poison_later_request():
    p = _pipe()
    q = "My Galaxy S22 screen went completely black and will not turn on."
    first = p.run(q, {})
    assert first["response"]["fallback"] == "no_siis_context"
    assert len(p.cache) == 0
    second = p.run(q, {"title": "Blank", "content": "Press and hold the Side key."})
    assert second["meta"]["cache_hit"] is False
    assert second["response"]["contexts"]


def test_device_less_paraphrase_hits_a_device_specific_entry_without_siis():
    p = _pipe()
    p.run("My Galaxy S22 screen went completely black and will not turn on.",
          {"title": "Blank", "content": "Press and hold the Side key."})
    hit = p.run("phone screen is black and wont turn on", {})
    assert hit["meta"]["cache_hit"] is True


def test_different_reference_text_is_not_served_from_cache():
    p = _pipe()
    q = "My Galaxy S22 screen went completely black and will not turn on."
    p.run(q, {"title": "A", "content": "Press and hold the Side key."})
    other = p.run(q, {"title": "B", "content": "Charge the phone for 30 minutes."})
    assert other["meta"]["cache_hit"] is False
    assert other["response"]["contexts"][0]["actions"][0]["stepGroups"][0]["steps"] == ["Charge the phone for 30 minutes."]


def test_cache_hit_returns_a_copy():
    p = _pipe()
    q = "My Galaxy S22 screen went completely black and will not turn on."
    r1 = p.run(q, {"title": "A", "content": "Press and hold the Side key."})
    r1["response"]["contexts"].clear()
    r2 = p.run(q, {"title": "A", "content": "Press and hold the Side key."})
    assert r2["meta"]["cache_hit"] is True and r2["response"]["contexts"]


def test_stage2_recategorises_non_auto_actions_and_puts_critical_last():
    from deeplink_mapping import deeplink_mapping
    mk = lambda n, c: Action(actionName=n, description="It will do the thing now.",
                              category=c, stepGroups=[StepGroup(steps=["Do it."])])
    s = ContextDeeplinkResponse(contexts=[Goal(goal="g", title="t", score=0.5, actions=[
        mk("Force a Restart", actionCategory.manual),
        mk("Visit Service Center", actionCategory.critical),
        mk("Safe Mode", actionCategory.manual),
    ])])
    out = deeplink_mapping(s, None)
    cats = [(a.actionName, a.category) for a in out.contexts[0].actions]
    assert cats == [("Visit Service Center", actionCategory.manual),
                    ("Force a Restart", actionCategory.critical),
                    ("Safe Mode", actionCategory.critical)]


def test_warm_from_results_serves_siis_less_queries(tmp_path):
    p = _pipe()
    resp = _stage1({"content": "Press and hold the Side key."}, None).model_dump()
    row = {"query": "My Galaxy S22 screen went completely black and will not turn on.",
           "query_variations": [], "response": resp}
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert p.warm_from_results(f) == 1
    assert p.run("galaxy s22 display is black, won't power on", {})["meta"]["cache_hit"] is True
