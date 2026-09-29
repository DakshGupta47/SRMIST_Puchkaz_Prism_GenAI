import json
from multiprocessing import context
import os
from pathlib import Path

from dotenv import load_dotenv

from enrichment import enrich
from structure_extraction import structure_extraction
from llm_client import get_llm_client

load_dotenv()
os.environ["LLM_PROVIDER"] = "gemini"


# Find project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the real SIIS dataset
DATA_FILE = PROJECT_ROOT / "data" / "siis_responses.json"


def load_row(row_id: str):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data["responses"]:
        if row["id"] == row_id:
            return row

    raise ValueError(f"Row not found: {row_id}")


def test_stage1_row21(monkeypatch):

    load_dotenv()

    monkeypatch.setenv(
        "LLM_PROVIDER",
        "gemini",
    )

    monkeypatch.setenv(
        "GOOGLE_API_KEY",
        os.environ["GOOGLE_API_KEY"],
    )

    # Clear the cached client because conftest.py
    # deliberately resets it.
    import llm_client
    llm_client._client_cache.clear()

    # -----------------------------------------
    # 1. Load real dataset record
    # -----------------------------------------
    row = load_row("row_21")

    query = row["original_query"]
    siis_response = row["siis_response"]

    # -----------------------------------------
    # 2. Run Stage 0 enrichment
    # -----------------------------------------
    enrichment = enrich(query)

    print("\n========================================")
    print("QUERY:")
    print(query)

    print("\nENRICHMENT:")
    print(enrichment.to_dict())

    # -----------------------------------------
    # 3. Get the real Gemini client
    # -----------------------------------------
    llm_client = get_llm_client()

    print("\nLLM:")
    print(llm_client.model_name)

    # -----------------------------------------
    # 4. Run YOUR Stage 1
    # -----------------------------------------
    result = structure_extraction(
        siis_response,
        enrichment,
        llm_client=llm_client,
    )

    # -----------------------------------------
    # 5. Print result
    # -----------------------------------------
    print("\n========== STAGE 1 RESULT ==========")

    print(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("====================================\n")

    # -----------------------------------------
    # 6. Basic assertions
    # -----------------------------------------
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