import json

import pytest

from enrichment import enrich
from llm_client import LLMClient
from structure_extraction import structure_extraction


class FakeValidLLM(LLMClient):

    def complete(self, system_prompt, user_prompt):

        return json.dumps({
            "contexts": [
                {
                    "goal": (
                        "Follow these steps to perform "
                        "this Display Troubleshooting"
                    ),
                    "title": "Display Troubleshooting",
                    "score": 0.9,
                    "actions": [
                        {
                            "actionName": "Display Settings",
                            "description": (
                                "It will check display settings."
                            ),
                            "stepGroups": [
                                {
                                    "steps": [
                                        "Open Settings.",
                                        "Tap Display."
                                    ],
                                    "actionableDeeplink": None,
                                    "validationDeeplink": None
                                }
                            ],
                            "category": "auto"
                        }
                    ]
                }
            ]
        })


def test_valid_structure():

    source = {
        "title": "Display troubleshooting",
        "content": (
            "The display may have a flickering issue. "
            "Open Settings. "
            "Tap Display. "
            "Check the display settings."
        )
    }

    result = structure_extraction(
        source,
        enrich("my display is flickering"),
        llm_client=FakeValidLLM(),
    )
    assert len(result.contexts) == 1
    assert result.contexts[0].title == "Display Troubleshooting"


def test_empty_result_is_allowed():

    class EmptyLLM(LLMClient):

        def complete(self, system_prompt, user_prompt):
            return json.dumps({
                "contexts": []
            })

    source = {
        "title": "Unrelated article",
        "content": "Some unrelated instructions."
    }

    result = structure_extraction(
        source,
        enrich("screen is completely black"),
        llm_client=EmptyLLM(),
    )

    assert result.contexts == []


def test_bad_description_is_repaired():

    class BadDescriptionLLM(LLMClient):

        def complete(self, system_prompt, user_prompt):
            return json.dumps({
                "contexts": [
                    {
                        "goal": (
                            "Follow these steps to perform "
                            "this Camera Troubleshooting"
                        ),
                        "title": "Camera Flickering",
                        "score": 0.9,
                        "actions": [
                            {
                                "actionName": "Camera Settings",
                                "description": (
                                    "It will adjust camera settings "
                                    "to fix flickering."
                                ),
                                "stepGroups": [
                                    {
                                        "steps": [
                                            "Open camera settings."
                                        ],
                                        "actionableDeeplink": None,
                                        "validationDeeplink": None
                                    }
                                ],
                                "category": "manual"
                            }
                        ]
                    }
                ]
            })

    source = {
        "title": "Camera troubleshooting",
        "content": (
            "The camera may flicker during video recording. "
            "Open camera settings."
        )
    }

    result = structure_extraction(
        source,
        enrich("camera flickering"),
        llm_client=BadDescriptionLLM(),
    )

    description = (
        result.contexts[0]
        .actions[0]
        .description
    )

    assert description.startswith("It will")
    assert 5 <= len(description.split()) <= 7


def test_unsupported_step_is_rejected():

    class HallucinatingLLM(LLMClient):

        def complete(self, system_prompt, user_prompt):
            return json.dumps({
                "contexts": [
                    {
                        "goal": (
                            "Follow these steps to perform "
                            "this Display Troubleshooting"
                        ),
                        "title": "Display Troubleshooting",
                        "score": 0.9,
                        "actions": [
                            {
                                "actionName": "Display Settings",
                                "description": (
                                    "It will check display settings."
                                ),
                                "stepGroups": [
                                    {
                                        "steps": [
                                            "Replace the motherboard."
                                        ],
                                        "actionableDeeplink": None,
                                        "validationDeeplink": None
                                    }
                                ],
                                "category": "auto"
                            }
                        ]
                    }
                ]
            })

    source = {
        "title": "Display troubleshooting",
        "content": (
            "The display may have a flickering issue. "
            "Open Settings. "
            "Tap Display."
        )
    }

    # Hallucinated step ("Replace the motherboard") is not in the source text.
    # structure_extraction should catch the validation error and return empty
    # contexts (§4.2.3 — no crash, graceful degradation). The error is still
    # printed to stdout for debugging.
    result = structure_extraction(
        source,
        enrich("display flickering"),
        llm_client=HallucinatingLLM(),
    )
    assert result.contexts == [], (
        "Hallucinated steps should produce empty contexts, not crash"
    )

def test_irrelevant_reference_is_rejected_before_llm():

    class ExplodingLLM(LLMClient):

        def complete(self, system_prompt, user_prompt):
            raise AssertionError(
                "LLM should not be called for an irrelevant reference"
            )

    source = {
        "title": "Use Multi window and App pairs",
        "content": (
            "Customize the Edge panel and use Multi window."
        ),
    }

    result = structure_extraction(
        source,
        enrich(
            "My tablet screen is flickering"
        ),
        llm_client=ExplodingLLM(),
    )

    assert result.contexts == []