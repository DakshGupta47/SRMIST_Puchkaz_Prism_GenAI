import json
import os
from pathlib import Path

from dotenv import load_dotenv

from enrichment import enrich
from llm_client import get_llm_client
from structure_extraction import structure_extraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "siis_responses.json"


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_stage1_all_records(monkeypatch):

    load_dotenv()

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv(
        "GOOGLE_API_KEY",
        os.environ["GOOGLE_API_KEY"],
    )

    import llm_client
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