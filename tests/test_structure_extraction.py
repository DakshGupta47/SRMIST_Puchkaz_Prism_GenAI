import json

from enrichment import enrich
from llm_client import LLMClient
from structure_extraction import structure_extraction


class FakeStage1LLM(LLMClient):

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = {
            "contexts": [
                {
                    "goal": (
                        "Follow these steps to perform "
                        "this Display Troubleshooting"
                    ),
                    "title": "Screen flickering",
                    "score": 0.95,
                    "actions": [
                        {
                            "actionName": "Display Settings",
                            "description": (
                                "It will help check display settings"
                            ),
                            "stepGroups": [
                                {
                                    "steps": [
                                        "Open Settings.",
                                        "Tap Display.",
                                    ],
                                    "validationDeeplink": None,
                                    "actionableDeeplink": None,
                                }
                            ],
                            "category": "auto",
                        }
                    ],
                }
            ]
        }

        return json.dumps(response)
def test_structure_extraction():
    siis_response = {
        "title": "Display troubleshooting",
        "content": """
        If your screen is flickering or the display blinks, try these steps.

        Open Settings.
        Tap Display.
        Check the display settings.
        """
    }


    enrichment = enrich(
        "my phone screen is flickering"
    )

    result = structure_extraction(
        siis_response,
        enrichment,
        llm_client=FakeStage1LLM(),
    )

    assert len(result.contexts) == 1

    context = result.contexts[0]

    assert context.title == "Screen flickering"

    assert context.score == 0.95

    assert len(context.actions) == 1

    action = context.actions[0]

    assert action.actionName == "Display Settings"

    assert len(action.stepGroups) == 1

    group = action.stepGroups[0]

    assert len(group.steps) == 2

    assert group.actionableDeeplink is None

    assert group.validationDeeplink is None
    
def test_hallucinated_step_is_rejected():

    class HallucinatingLLM(LLMClient):

        def complete(
            self,
            system_prompt: str,
            user_prompt: str,
        ) -> str:

            response = {
                "contexts": [
                    {
                        "goal": (
                            "Follow these steps to perform "
                            "this Display Troubleshooting"
                        ),
                        "title": "Screen flickering",
                        "score": 0.95,
                        "actions": [
                            {
                                "actionName": "Display Settings",
                                "description": (
                                    "It will help check display settings"
                                ),
                                "stepGroups": [
                                    {
                                        "steps": [
                                            "Replace the motherboard immediately."
                                        ],
                                        "validationDeeplink": None,
                                        "actionableDeeplink": None,
                                    }
                                ],
                                "category": "auto",
                            }
                        ],
                    }
                ]
            }

            return json.dumps(response)

    siis_response = {
        "title": "Display troubleshooting",
        "content": """
        Open Settings.
        Tap Display.
        Check the display settings.
        """
    }

    enrichment = enrich(
        "my phone screen is flickering"
    )

    result = structure_extraction(
        siis_response,
        enrichment,
        llm_client=HallucinatingLLM(),
    )

    assert result.contexts == []