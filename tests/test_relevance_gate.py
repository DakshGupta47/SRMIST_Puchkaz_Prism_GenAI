import json
from pathlib import Path

from enrichment import enrich
from relevance_gate import assess_reference_relevance


def test_relevant_touch_article():

    enrichment = enrich(
        "My Galaxy S22 touch input is laggy"
    )

    source = {
        "title": "Touchscreen issues on a Galaxy phone",
        "content": (
            "Touchscreen issues may occur when touch input "
            "becomes delayed or unresponsive."
        ),
    }

    relevant, score, matched = (
        assess_reference_relevance(
            source,
            enrichment,
        )
    )

    assert relevant is True
    assert score > 0
    assert matched


def test_unrelated_multiwindow_article():

    enrichment = enrich(
        "My tablet screen stays dark"
    )

    source = {
        "title": "Use Multi window and App pairs",
        "content": (
            "Customize the Edge panel and use Multi window "
            "features."
        ),
    }

    relevant, score, matched = (
        assess_reference_relevance(
            source,
            enrichment,
        )
    )

    assert relevant is False
    
def test_real_row_7_is_rejected():

    project_root = Path(__file__).resolve().parents[1]

    data_file = (
        project_root
        / "data"
        / "siis_responses.json"
    )

    with open(
        data_file,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    row_7 = next(
        row
        for row in data["responses"]
        if row["id"] == "row_7"
    )

    enrichment = enrich(
        row_7["original_query"]
    )

    relevant, score, matched = (
        assess_reference_relevance(
            row_7["siis_response"],
            enrichment,
        )
    )

    print("\nREAL ROW_7")
    print("Relevant:", relevant)
    print("Score:", score)
    print("Matched:", matched)

    assert relevant is False
    
def test_real_row_21_is_relevant():

    project_root = Path(__file__).resolve().parents[1]

    data_file = (
        project_root
        / "data"
        / "siis_responses.json"
    )

    with open(
        data_file,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    row_21 = next(
        row
        for row in data["responses"]
        if row["id"] == "row_21"
    )

    enrichment = enrich(
        row_21["original_query"]
    )

    relevant, score, matched = (
        assess_reference_relevance(
            row_21["siis_response"],
            enrichment,
        )
    )

    print("\nREAL ROW_21")
    print("Relevant:", relevant)
    print("Score:", score)
    print("Matched:", matched)

    assert relevant is True
    assert score > 0
    assert matched