import json
from pathlib import Path

import pytest

from deeplink_mapping import deeplink_mapping
from enrichment import EnrichmentResult
from schema import Action, ContextDeeplinkResponse, Goal, StepGroup, actionCategory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG = json.loads((DATA_DIR / "deeplinks.json").read_text())["deeplinks"]
CATALOG_DEEPLINKS = {d["deeplink"] for d in CATALOG}


def _fake_enrichment() -> EnrichmentResult:
    """Minimal, fully offline EnrichmentResult -- no enrich() call, no
    network, no .env dependency.
    """
    return EnrichmentResult(
        raw_query="touch sensitivity issue",
        canonical_query="Galaxy S22 touch sensitivity setting needs adjusting.",
        device="Galaxy S22",
        symptom_category="screen_ghost_touch",
        symptom_label="Touch input not registering correctly",
        device_confidence=1.0,
        symptom_confidence=1.0,
    )


def _wrap(action: Action) -> ContextDeeplinkResponse:
    return ContextDeeplinkResponse(
        contexts=[
            Goal(
                goal="Follow these steps to perform this Display Troubleshooting",
                title="Display Troubleshooting",
                score=0.9,
                actions=[action],
            )
        ]
    )


def test_auto_action_gets_real_catalog_match_when_one_exists():
    action = Action(
        actionName="Touch Sensitivity Settings",
        description="It will adjust touch sensitivity options.",
        category=actionCategory.auto,
        stepGroups=[
            StepGroup(
                steps=[
                    "Go to Settings.",
                    "Tap Display.",
                    "Tap the switch next to Touch sensitivity.",
                ]
            )
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    step_group = result.contexts[0].actions[0].stepGroups[0]

    assert step_group.actionableDeeplink is not None
    assert step_group.actionableDeeplink.deeplink.endswith("://masked/act/1b0d34e9b4")


def test_matched_deeplink_is_always_copied_verbatim_from_catalog():
    action = Action(
        actionName="Touch Sensitivity Settings",
        description="It will adjust touch sensitivity options.",
        category=actionCategory.auto,
        stepGroups=[
            StepGroup(steps=["Go to Settings.", "Tap Display.", "Tap Touch sensitivity."])
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    matched = result.contexts[0].actions[0].stepGroups[0].actionableDeeplink

    assert matched.deeplink in CATALOG_DEEPLINKS or matched.deeplink.endswith("://dummy_positive")


def test_auto_action_with_no_real_match_falls_back_to_dummy_positive():
    action = Action(
        actionName="Lighting and Camera Modes",
        description="It will adjust camera lighting settings.",
        category=actionCategory.auto,
        stepGroups=[
            StepGroup(
                steps=[
                    "Increase the lighting in your scene.",
                    "Disable Super steady mode.",
                    "Adjust shutter speed in Pro Video mode until flickering disappears.",
                ]
            )
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    step_group = result.contexts[0].actions[0].stepGroups[0]

    assert step_group.actionableDeeplink is not None
    assert step_group.actionableDeeplink.deeplink.endswith("://dummy_positive")


def test_critical_action_never_gets_a_deeplink():
    action = Action(
        actionName="Factory Data Reset",
        description="It will erase all device data.",
        category=actionCategory.critical,
        stepGroups=[
            StepGroup(
                steps=[
                    "Navigate to and open Settings.",
                    "Tap General management.",
                    "Tap Reset.",
                    "Tap Factory data reset.",
                ]
            )
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    step_group = result.contexts[0].actions[0].stepGroups[0]

    assert step_group.actionableDeeplink is None

    assert step_group.actionableDeeplink is None
    assert step_group.validationDeeplink is None   # add this line


def test_manual_action_without_settings_steps_stays_unlinked():
    action = Action(
        actionName="Force a Restart",
        description="It will restart the device.",
        category=actionCategory.manual,
        stepGroups=[
            StepGroup(
                steps=[
                    "Press and hold both the Power button and the Volume down button for 20 seconds."
                ]
            )
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    step_group = result.contexts[0].actions[0].stepGroups[0]

    assert step_group.actionableDeeplink is None

def test_auto_action_with_restart_language_is_recategorized_and_unlinked():
    """Stage 1 sometimes mislabels a restart/reset step as "auto" -- observed
    on the real results.jsonl sample (e.g. "Force a Restart", "Force Restart
    Device"). Without correction, these still go through _best_match() and,
    depending on catalog vocabulary overlap, could score above
    SIMILARITY_THRESHOLD and get a real (wrong) deeplink attached.
    """
    action = Action(
        actionName="Force a Restart",
        description="It will restart the frozen device.",
        category=actionCategory.auto,
        stepGroups=[
            StepGroup(
                steps=[
                    "Press and hold both the Power button and the Volume down button simultaneously for at least 20 seconds."
                ]
            )
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    fixed_action = result.contexts[0].actions[0]
    step_group = fixed_action.stepGroups[0]

    assert fixed_action.category == actionCategory.critical
    assert step_group.actionableDeeplink is None
    assert step_group.validationDeeplink is None


def test_auto_action_with_physical_hardware_language_is_recategorized_and_unlinked():
    """Same bug as above, but for physical-intervention steps (e.g. "Check
    for Physical Damage and Liquid Exposure", "Charge the Device" in the real
    sample), which schema.py reserves for "manual", not "auto".
    """
    action = Action(
        actionName="Check for Physical Damage",
        description="It will rule out hardware causes first.",
        category=actionCategory.auto,
        stepGroups=[
            StepGroup(
                steps=[
                    "Carefully inspect your phone, charger, and USB cable for any physical damage or signs of liquid exposure."
                ]
            )
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    fixed_action = result.contexts[0].actions[0]
    step_group = fixed_action.stepGroups[0]

    assert fixed_action.category == actionCategory.manual
    assert step_group.actionableDeeplink is None
    assert step_group.validationDeeplink is None


def test_auto_action_with_genuine_settings_language_is_left_alone():
    """Regression guard: the restart/hardware recategorization above must not
    false-positive on a real settings action and strip its legitimate
    deeplink.
    """
    action = Action(
        actionName="Touch Sensitivity Settings",
        description="It will adjust touch sensitivity options.",
        category=actionCategory.auto,
        stepGroups=[
            StepGroup(steps=["Go to Settings.", "Tap Display.", "Tap Touch sensitivity."])
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    fixed_action = result.contexts[0].actions[0]
    step_group = fixed_action.stepGroups[0]

    assert fixed_action.category == actionCategory.auto
    assert step_group.actionableDeeplink is not None


def test_manual_action_with_settings_steps_gets_no_deeplink():
    action = Action(
        actionName="Touch Sensitivity Settings",
        description="It will adjust touch sensitivity options.",
        category=actionCategory.manual,
        stepGroups=[
            StepGroup(
                steps=[
                    "Go to Settings.",
                    "Tap Display.",
                    "Tap the switch next to Touch sensitivity.",
                ]
            )
        ],
    )
    result = deeplink_mapping(_wrap(action), _fake_enrichment())
    step_group = result.contexts[0].actions[0].stepGroups[0]

    assert step_group.actionableDeeplink is None
    assert step_group.validationDeeplink is None