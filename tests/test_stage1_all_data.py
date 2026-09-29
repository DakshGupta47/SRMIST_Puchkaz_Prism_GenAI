"""Integration test — calls the real Gemini API across all 20 SIIS records.

Run only when a valid GOOGLE_API_KEY is present:

    pytest -m integration tests/test_stage1_all_data.py -s

Excluded from the default `pytest` run so the suite stays fast and
deterministic on machines without an API key configured.
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



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "siis_responses.json"


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.integration
def test_stage1_all_records(monkeypatch):
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        import pytest
        pytest.skip("GOOGLE_API_KEY not configured in environment")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set — skipping integration test")

    monkeypatch.setenv("GOOGLE_API_KEY", api_key)


    try:
        from src import llm_client
    except ImportError:
        import llm_client  # type: ignore[no-redef]

    llm_client._client_cache.clear()

    data = load_data()

    rows = data["responses"]

    print(f"\nTotal records: {len(rows)}")

    llm_client_instance = get_llm_client()

    print(
        f"Using LLM: {llm_client_instance.model_name}"
    )

    successful = 0
    failed = []
    no_match = 0

    for row in rows:

        row_id = row["id"]

        print("\n" + "=" * 60)
        print(f"Testing {row_id}")
        print("=" * 60)

        try:

            query = row["original_query"]
            siis_response = row["siis_response"]

            enrichment = enrich(query)

            result = structure_extraction(
                siis_response,
                enrichment,
                llm_client=llm_client_instance,
            )

            if result is None:
                failed.append(
                    (row_id, "Stage 1 returned None")
                )

                print("RESULT: NONE")
                continue

            if not result.contexts:
                no_match += 1
                print("RESULT: VALID NO-MATCH")
                continue

            context = result.contexts[0]

            print(
                f"Title: {context.title}"
            )

            print(
                f"Actions: {len(context.actions)}"
            )

            for action in context.actions:
                print(
                    f"  - {action.actionName} "
                    f"[{action.category}]"
                )

            successful += 1
            print("RESULT: SUCCESS")

        except Exception as exc:

            failed.append(
                (row_id, str(exc))
            )

            print(
                f"RESULT: FAILED - {exc}"
            )

    print("\n")
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Total:      {len(rows)}")
    print(f"Success:    {successful}")
    print(f"No-match:   {no_match}")
    print(f"Failed:     {len(failed)}")

    if failed:

        print("\nFailures:")

        for row_id, reason in failed:
            print(
                f"  {row_id}: {reason}"
            )

    assert len(failed) == 0