"""Integration test — calls the real Gemini API.

Run only when a valid GOOGLE_API_KEY is present:

    pytest -m integration tests/test_stage1_real_data.py -s

Excluded from the default `pytest` run (no -m flag) so the suite stays fast
and deterministic on machines without an API key configured.
"""
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

try:
    from src.enrichment import enrich
    from src.llm_client import get_llm_client
    from src.structure_extraction import structure_extraction
except ImportError:
    from enrichment import enrich  # type: ignore[no-redef]  # noqa: E402
    from llm_client import get_llm_client  # type: ignore[no-redef]  # noqa: E402
    from structure_extraction import structure_extraction  # type: ignore[no-redef]  # noqa: E402


# Find project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "siis_responses.json"


def load_row(row_id: str):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data["responses"]:
        if row["id"] == row_id:
            return row
    raise ValueError(f"Row not found: {row_id}")


@pytest.mark.integration
def test_stage1_row21(monkeypatch):
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        import pytest
        pytest.skip("GOOGLE_API_KEY not configured in environment")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set — skipping integration test")

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)


    # Clear the cached client because conftest.py deliberately resets it.
    try:
        from src import llm_client
    except ImportError:
        import llm_client  # type: ignore[no-redef]

    llm_client._client_cache.clear()

    # 1. Load real dataset record
    row = load_row("row_21")
    query = row["original_query"]
    siis_response = row["siis_response"]

    # 2. Run Stage 0 enrichment
    enrichment = enrich(query)

    print("\n========================================")
    print("QUERY:")
    print(query)
    print("\nENRICHMENT:")
    print(enrichment.to_dict())

    # 3. Get the real Gemini client
    llm_client_instance = get_llm_client()
    print("\nLLM:")
    print(llm_client_instance.model_name)

    # 4. Run Stage 1
    result = structure_extraction(
        siis_response,
        enrichment,
        llm_client=llm_client_instance,
    )

    # 5. Print result
    print("\n========== STAGE 1 RESULT ==========")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print("====================================\n")

    # 6. Basic assertions
    assert result is not None
    assert hasattr(result, "contexts")
    assert len(result.contexts) > 0

    context = result.contexts[0]
    assert context.goal
    assert context.title
    assert 0 <= context.score <= 1
    assert len(context.actions) > 0

    for action in context.actions:
        assert action.actionName
        assert action.description.startswith("It will")
        assert 5 <= len(action.description.split()) <= 7

    for group in action.stepGroups:
        assert len(group.steps) > 0
        assert group.actionableDeeplink is None
        assert group.validationDeeplink is None