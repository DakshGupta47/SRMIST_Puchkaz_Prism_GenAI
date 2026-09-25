"""
Minimal REST API for standalone testing of Member A's stages, and the
real /v1/troubleshoot contract the whole team's pipeline exposes once
Stage 1 (Member B) and Stage 2 (Member C) are wired into pipeline.Pipeline.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /v1/troubleshoot   the team's real deliverable endpoint
    POST /v1/enrich         Stage 0 only — inspect normalization/variations
    GET  /v1/cache/stats    cache size, for demo/debugging
    GET  /health            the theme brief's §5 exact spec'd path
    GET  /healthz           kept as an alias (a common infra convention);
                            not in the brief, harmless to also expose
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from enrichment import enrich
from pipeline import Pipeline

app = FastAPI(title="Smart Guided Troubleshooting Engine — Member A slice")
pipeline = Pipeline()


class SiisResponse(BaseModel):
    title: str = ""
    content: str = ""


class TroubleshootRequest(BaseModel):
    query: str
    siis_response: Optional[SiisResponse] = None


class EnrichRequest(BaseModel):
    query: str


@app.post("/v1/troubleshoot")
def troubleshoot(req: TroubleshootRequest) -> dict:
    siis_response = req.siis_response.model_dump() if req.siis_response else {}
    return pipeline.run(req.query, siis_response)


@app.post("/v1/enrich")
def enrich_endpoint(req: EnrichRequest) -> dict:
    """Stage 0 only, exposed standalone so it can be tested/demoed without
    Stage 1/2 being implemented yet.
    """
    return enrich(req.query).to_dict()


@app.get("/v1/cache/stats")
def cache_stats() -> dict:
    return {"entries": len(pipeline.cache), "similarity_threshold": pipeline.cache.similarity_threshold}


@app.get("/health")
def health() -> dict:
    """Exact path the theme brief's §5 API contract names. `pipeline` (cache
    + embedder) is constructed at module import time above, so by the time
    FastAPI is serving requests at all, initialization is already done —
    a static "ok" here is an honest statement of that, not a shortcut.
    """
    return {"status": "ok"}


@app.get("/healthz")
def healthz() -> dict:
    """Alias for /health — a common infra convention, not in the brief,
    kept so nothing that already depends on this path breaks.
    """
    return {"status": "ok"}
