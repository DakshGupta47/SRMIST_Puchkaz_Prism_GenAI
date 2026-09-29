# ============================================================================
# TEMPORARY STUB - REPLACE
# ============================================================================
"""
Stage 2 — Deeplink Mapping (Temporary Stub)

Maps extracted troubleshooting actions and step groups to Settings / Bixby
deeplinks from data/deeplinks.json.

This is a temporary stub implementation matching the Stage2Fn signature in
src/pipeline.py. It sets every step group's actionable deeplink to
bixby://dummy_positive and sets the action category to manual (since auto
would require a validated, real deeplink). Real deeplinks must not be invented.
"""
from __future__ import annotations

from enrichment import EnrichmentResult
from schema import ContextDeeplinkResponse, Deeplink, actionCategory


def map_deeplinks(
    result: ContextDeeplinkResponse,
    enrichment: EnrichmentResult,
) -> ContextDeeplinkResponse:
    """Temporary Stage 2 stub for deeplink mapping.

    Attaches dummy deeplinks (`bixby://dummy_positive`) to every step group
    and ensures the action category passes schema validation by falling back
    to `manual` whenever an action was marked `auto`.
    """
    if result is None or not result.contexts:
        return result

    for context in result.contexts:
        for action in context.actions:
            # If the LLM left category as None, default to manual.
            if action.category is None:
                action.category = actionCategory.manual

            # Only 'auto' actions are allowed to carry an actionableDeeplink.
            if action.category == actionCategory.auto:
                for group in action.stepGroups:
                    group.actionableDeeplink = Deeplink(
                        deeplink="bixby://dummy_positive",
                        description="Stub deeplink for automated execution",
                    )
            else:
                for group in action.stepGroups:
                    group.actionableDeeplink = None

    return result
