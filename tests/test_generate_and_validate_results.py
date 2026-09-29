"""
Tests for scripts/generate_results.py and scripts/validate_results.py.
Verifies end-to-end generation, schema compliance, fallback generation on errors,
and scrubber validation catches all forbidden patterns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src and scripts to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest

try:
    from scripts.generate_results import _create_fallback_record, generate_results
    from scripts.validate_results import validate_file, validate_record
except ImportError:
    from generate_results import _create_fallback_record, generate_results  # type: ignore[no-redef]  # noqa: E402
    from validate_results import validate_file, validate_record  # type: ignore[no-redef]  # noqa: E402



def test_generated_results_file_passes_full_validation(tmp_path: Path):
    """Test generating a small results file and validating it."""
    test_input = tmp_path / "test_input.txt"
    test_unseen = tmp_path / "test_unseen.txt"
    test_siis = tmp_path / "test_siis.json"
    out_file = tmp_path / "test_results.jsonl"

    test_input.write_text("My Galaxy S22 screen is black\n", encoding="utf-8")
    test_unseen.write_text("Battery drain on Galaxy S22\n", encoding="utf-8")
    siis_content = {
        "responses": [
            {
                "id": "row_1",
                "original_query": "My Galaxy S22 screen is black",
                "siis_response": {
                    "title": "Blank or black display on a Samsung phone or tablet",
                    "content": "Check your device for damage. Force restart your device by holding buttons.",
                },
            }
        ]
    }
    test_siis.write_text(json.dumps(siis_content), encoding="utf-8")

    summary = generate_results(
        input_file=test_input,
        unseen_file=test_unseen,
        siis_file=test_siis,
        output_file=out_file,
    )

    assert summary["row_count"] == 2
    assert out_file.is_file()

    total, passed, errors = validate_file(out_file)
    assert total == 2
    assert passed == 2
    assert errors == []


def test_fallback_record_creation_is_valid():
    """Fallback record emitted when an exception occurs must pass validation."""
    record_with_siis = _create_fallback_record("Test query", {"title": "t", "content": "c"}, 1.5)
    errors1 = validate_record(record_with_siis)
    assert errors1 == []
    assert record_with_siis["response"]["fallback"] == "no_match"

    record_no_siis = _create_fallback_record("Test query", {}, 1.5)
    errors2 = validate_record(record_no_siis)
    assert errors2 == []
    assert record_no_siis["response"]["fallback"] == "no_siis_context"


def test_validator_detects_url_leak():
    """Validator must reject records containing URL leaks in text fields."""
    leaky_record = {
        "query": "Fix display",
        "response": {
            "contexts": [
                {
                    "goal": "Follow these steps to perform this Screen Troubleshooting",
                    "title": "Screen fix",
                    "score": 0.9,
                    "actions": [
                        {
                            "actionName": "Open Settings",
                            "description": "It will visit https://samsung.com today",
                            "stepGroups": [
                                {
                                    "steps": ["Visit http://example.com for help."],
                                    "actionableDeeplink": None,
                                    "validationDeeplink": None,
                                }
                            ],
                            "category": "manual",
                        }
                    ],
                }
            ]
        },
    }
    errors = validate_record(leaky_record)
    assert any("URL leak" in err or "Description" in err for err in errors)


def test_validator_detects_invalid_deeplink_scheme():
    """Validator must reject records with web URLs in deeplink fields."""
    leaky_dl_record = {
        "query": "Fix display",
        "response": {
            "contexts": [
                {
                    "goal": "Follow these steps to perform this Screen Troubleshooting",
                    "title": "Screen fix",
                    "score": 0.9,
                    "actions": [
                        {
                            "actionName": "Open Settings",
                            "description": "It will adjust device display settings.",
                            "stepGroups": [
                                {
                                    "steps": ["Open Settings."],
                                    "actionableDeeplink": {
                                        "deeplink": "https://malicious.com/open",
                                        "description": "It will adjust device display settings.",
                                    },
                                    "validationDeeplink": None,
                                }
                            ],
                            "category": "auto",
                        }
                    ],
                }
            ]
        },
    }
    errors = validate_record(leaky_dl_record)
    assert any("Invalid or disallowed actionableDeeplink" in err for err in errors)


def test_validator_detects_missing_fallback_on_empty_contexts():
    """Empty contexts without fallback must fail validation."""
    empty_no_fallback = {
        "query": "Fix display",
        "response": {
            "contexts": [],
        },
    }
    errors = validate_record(empty_no_fallback)
    assert any("Empty contexts must have fallback" in err for err in errors)


def test_validator_detects_invalid_word_counts():
    """Title or description with invalid word count must fail validation."""
    bad_title_record = {
        "query": "Fix display",
        "response": {
            "contexts": [
                {
                    "goal": "Follow these steps to perform this Screen Troubleshooting",
                    "title": "OneWordTitleOnly",
                    "score": 0.9,
                    "actions": [
                        {
                            "actionName": "Open Settings",
                            "description": "It will adjust device display settings.",
                            "stepGroups": [
                                {
                                    "steps": ["Open Settings."],
                                    "actionableDeeplink": None,
                                    "validationDeeplink": None,
                                }
                            ],
                            "category": "manual",
                        }
                    ],
                }
            ]
        },
    }
    errors = validate_record(bad_title_record)
    assert any("Title must be 2-3 words" in err for err in errors)
