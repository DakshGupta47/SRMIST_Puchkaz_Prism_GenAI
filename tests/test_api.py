"""
Tests for api.py -- the actual FastAPI layer /v1/troubleshoot is served
through. Before this file, all 125+ tests in this suite exercised
enrichment.py/cache.py/pipeline.py directly and never went through a single
HTTP request, so a bug anywhere in request parsing, response serialization,
or an unhandled exception surfacing as a raw traceback would have shipped
completely untested on the team's actual deliverable endpoint.
"""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    return TestClient(api.app, raise_server_exceptions=False)


def test_health_and_healthz_ok(client):
    for path in ("/health", "/healthz"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_troubleshoot_happy_path_returns_full_contract_shape(client):
    r = client.post("/v1/troubleshoot", json={
        "query": "My Galaxy S22 battery drains extremely fast, dead by noon even with light use.",
        "siis_response": {"title": "Battery", "content": "Check battery usage in Settings."},
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {"query", "query_variations", "response", "meta", "enrichment"}
    assert isinstance(body["query_variations"], list) and len(body["query_variations"]) >= 1
    assert set(body["meta"].keys()) == {"cache_hit", "similarity", "latency_ms", "model", "cost_usd"}
    assert body["enrichment"]["symptom_category"] == "battery_drain"


def test_troubleshoot_missing_query_field_is_a_clean_422_not_a_500(client):
    r = client.post("/v1/troubleshoot", json={})
    assert r.status_code == 422


def test_troubleshoot_wrong_type_query_is_a_clean_422_not_a_500(client):
    r = client.post("/v1/troubleshoot", json={"query": 12345})
    assert r.status_code == 422


def test_troubleshoot_malformed_siis_response_is_a_clean_422_not_a_500(client):
    r = client.post("/v1/troubleshoot", json={"query": "battery drain", "siis_response": "not an object"})
    assert r.status_code == 422


def test_troubleshoot_empty_query_does_not_crash():
    """An empty/whitespace-only query carries no information Stage 0 can
    classify, but that must surface as a normal low-confidence response
    (fallback: "no_siis_context"/"no_match"), never a 500 -- a judge
    trying an empty string is a completely plausible thing to try.
    """
    client = TestClient(api.app, raise_server_exceptions=False)
    for query in ("", "   "):
        r = client.post("/v1/troubleshoot", json={"query": query})
        assert r.status_code == 200
        assert r.json()["response"].get("fallback") == "no_siis_context"


def test_enrich_endpoint_returns_stage0_output_only(client):
    r = client.post("/v1/enrich", json={"query": "My Galaxy S24 Ultra won't charge at all."})
    assert r.status_code == 200
    body = r.json()
    assert body["symptom_category"] == "charging_fails_or_slow"
    # Stage 0 only -- no "response"/"meta" pipeline wrapper at this endpoint.
    assert "response" not in body and "meta" not in body


def test_cache_stats_reports_similarity_threshold_and_starts_empty():
    # a fresh Pipeline (not the module-level api.pipeline, which accumulates
    # entries across other tests in this file) so the starting count is known.
    from pipeline import Pipeline

    fresh_app_pipeline = Pipeline()
    original = api.pipeline
    api.pipeline = fresh_app_pipeline
    try:
        client = TestClient(api.app, raise_server_exceptions=False)
        r = client.get("/v1/cache/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["entries"] == 0
        assert body["similarity_threshold"] == fresh_app_pipeline.cache.similarity_threshold
    finally:
        api.pipeline = original


def test_unhandled_exception_returns_clean_500_not_a_raw_traceback(client, monkeypatch):
    """The actual regression test for the bug this file exists to catch:
    before api.py had a global exception handler, any exception raised
    inside pipeline.run() (a bug in enrichment/cache, or -- once wired in --
    Stage 1/2 blowing up on unexpected input) propagated straight out as
    FastAPI's default response: a raw Python stack trace, with file paths
    and source lines, served to whoever sent the request. Forcing exactly
    that here with a pipeline that always raises, and asserting the caller
    gets a clean, bounded JSON error instead.
    """
    def _boom(query, siis_response):
        raise RuntimeError("simulated Stage 1/2 failure")

    monkeypatch.setattr(api.pipeline, "run", _boom)
    r = client.post("/v1/troubleshoot", json={"query": "My Galaxy S22 battery drains fast."})
    assert r.status_code == 500
    body = r.json()
    assert body == {
        "error": "internal_error",
        "detail": "An unexpected error occurred while processing the request.",
    }
    # the failure mode this guards against: internals (file paths, source
    # lines, the exception's own message) leaking into the response body.
    assert "Traceback" not in r.text
    assert "simulated Stage 1/2 failure" not in r.text
    assert "enrichment.py" not in r.text and "pipeline.py" not in r.text
