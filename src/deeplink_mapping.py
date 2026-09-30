from __future__ import annotations

import json
import re
from pathlib import Path

from embeddings import TfidfEmbedder
from enrichment import EnrichmentResult
from schema import ContextDeeplinkResponse, Deeplink, ValidationDeepLink, actionCategory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIMILARITY_THRESHOLD = 0.35  # placeholder — tune against sample_output.json

# Stage 1 (the LLM) sometimes labels a physical-intervention or restart/reset
# step as "auto" even though schema.py reserves "auto" for steps reachable via
# a settings deeplink. Left uncaught, these still go through _best_match():
# deeplinks.json is entirely software settings, so a restart/hardware step
# scores low but non-zero against it. On the current 20-query sample,
# SIMILARITY_THRESHOLD (0.35) happens to sit above all 8 such miscategorized
# actions (scores 0.19-0.31) and below both genuinely correct auto actions
# (0.38, 0.48) -- but that gap is a property of this small sample, not a rule
# the retrieval step actually enforces. A held-out restart/hardware step that
# happens to share more vocabulary with the catalog could score above 0.35
# and silently get a real (wrong) catalog deeplink attached, which breaks
# both "manual/critical actions cannot carry deeplinks" and the catalog
# integrity requirement. These patterns catch the obvious cases directly from
# the text instead of relying on similarity-score luck to keep them out.
# "power button" / "volume button" alone are too generic -- real catalog
# entries ("Side key double press action", "Lock instantly with Side key")
# legitimately name the button while describing a tap/press *binding*, not
# an instruction to physically hold it down now. What actually distinguishes
# "press and hold the Power button for 20 seconds" (force a restart) from
# "configure what a double press of the Power button does" (a settings
# toggle) is the sustained "press and hold", so that phrase is required
# rather than the bare button name.
_CRITICAL_RE = re.compile(
    r"\b(restart(?:ing)?|reboot(?:ing)?|factory\s+(?:data\s+)?reset|firmware\s+update|safe\s+mode)\b"
    r"|press(?:ing)?(?:\s+and)?\s+hold(?:ing)?\b[^.]{0,60}\b(?:power|volume)\b",
    re.IGNORECASE,
)
# Bare nouns like "charger" / "USB cable" / "inspect" are too generic on
# their own -- real catalog entries ("Fast Charging Settings", "USB
# Tethering Settings") legitimately mention them while describing a
# software toggle. Each pattern below instead requires the fuller phrase
# that actually distinguishes "physically do something to the hardware"
# from "configure a setting related to hardware".
_MANUAL_RE = re.compile(
    r"(physical\s+damage|liquid\s+exposure|service\s+cent(?:er|re)s?|repair\s+services?|"
    r"sim\S*\s*tray|removable\s+battery|replace\s+the\s+(?:battery|screen)|"
    r"clean(?:ing)?\s+(?:the\s+)?port|"
    r"(?:inspect|examine|check)\b[^.]{0,50}\bdamage\b|"
    r"connect\s+(?:your|the)\s+(?:phone|tablet|device)[^.]{0,20}to\s+its\s+(?:appropriate\s+)?charger|"
    r"disconnect\s+(?:the|your)\s+(?:phone|tablet|device)[^.]{0,20}from\s+the\s+charger|"
    r"let\s+it\s+charge\s+for\s+at\s+least)",
    re.IGNORECASE,
)
# The catalog genuinely contains settings-screen deeplinks about *configuring*
# restart behavior -- "Restart on schedule", "Inactivity restart", "Enable/
# Disable Auto Restart" -- which are legitimate "auto" actions navigable via
# Settings, not a physical button-press restart. _CRITICAL_RE's bare
# "restart" keyword can't tell these apart from "press and hold the Power
# button to force a restart", so this exception is checked first and, when it
# matches, _infer_non_auto_category defers entirely to the normal semantic
# match against the catalog instead of forcing a category.
_RESTART_SETTINGS_RE = re.compile(
    r"\b(restart\s+on\s+schedule|inactivity\s+restart|auto(?:matic)?\s+restart|"
    r"(?:enable|disable)\s+(?:auto(?:matic)?\s+)?restart)\b",
    re.IGNORECASE,
)

def _infer_non_auto_category(text: str):
    """Return actionCategory.critical / .manual if 	ext clearly describes a
    restart/reset or a physical-hardware step, else None (defer to the
    existing similarity match). Checked in this order because "restart" is
    the more specific, brief-named critical example; physical/hardware
    language is the broader manual catch-all; the settings-restart exception
    is checked before either, since it overrides both.
    """
    if _RESTART_SETTINGS_RE.search(text):
        return None
    if _CRITICAL_RE.search(text):
        return actionCategory.critical
    if _MANUAL_RE.search(text):
        return actionCategory.manual
    return None

_catalog = None
_catalog_texts = None
_catalog_matrix = None
_embedder = None
_placeholder = None

_STOP = {"and", "or", "the", "a", "of", "to", "in", "for", "with"}


def _load_catalog():
    global _catalog, _catalog_texts, _catalog_matrix, _embedder, _placeholder
    if _catalog is not None:
        return
    raw = json.loads((DATA_DIR / "deeplinks.json").read_text(encoding="utf-8"))
    entries = raw["deeplinks"]
    _placeholder = next(
        (d["deeplink"] for d in entries if d.get("originalType") == "placeholder"),
        entries[0]["deeplink"].split("://")[0] + "://dummy_positive",
    )
    _catalog = [d for d in entries if d.get("originalType") != "placeholder"]
    _catalog_texts = [
        f"{d.get('description','')} {d.get('message','')} {d.get('qna_description','')}"
        for d in _catalog
    ]
    _embedder = TfidfEmbedder.load()
    _catalog_matrix = _embedder.encode_sparse(_catalog_texts)


def _best_match(step_text: str, action_name: str = ""):
    _load_catalog()
    q = _embedder.encode_sparse([step_text])
    sims = (_catalog_matrix @ q.T).toarray().ravel()
    
    if action_name:
        an = action_name.lower()
        for i, catalog_item in enumerate(_catalog):
            msg = catalog_item.get('message', '').lower()
            
            # Penalize false positives where catalog has a strong keyword but action doesn't
            if 'sync' in msg and 'sync' not in an: sims[i] *= 0.5
            if 'back up' in msg and 'back' not in an: sims[i] *= 0.5
            if 'reset' in msg and 'reset' not in an: sims[i] *= 0.5
            if 'update' in msg and 'update' not in an: sims[i] *= 0.5
            
            # Boost true positives where both action and catalog share the specific keyword
            if 'sync' in msg and 'sync' in an: sims[i] *= 1.5
            if 'back up' in msg and 'back' in an: sims[i] *= 1.5
            if 'reset' in msg and 'reset' in an: sims[i] *= 1.5
            if 'update' in msg and 'update' in an: sims[i] *= 1.5
            
    best_idx = sims.argmax()
    return _catalog[best_idx], float(sims[best_idx])


def _core(name: str) -> list[str]:
    ws = [w for w in re.findall(r"[A-Za-z0-9+]+", name)
          if w.lower() not in {"settings", "setting"}][:4]
    while ws and ws[-1].lower() in _STOP:
        ws.pop()
    return [w.lower() for w in ws]


def _fit(words: list[str], lo: int = 5, hi: int = 7) -> str:
    words = words[:hi]
    fillers = iter(["device", "the", "Nexa"])
    while len(words) < lo:
        words.insert(1, next(fillers, "device"))
    return " ".join(words)


def _to_validation_deeplink(entry: dict) -> ValidationDeepLink | None:
    v = entry.get("validation")
    if not v:
        return None
    return ValidationDeepLink(deeplink=v["deeplink"], key=v["key"])


def deeplink_mapping(
    structured: ContextDeeplinkResponse, enrichment: EnrichmentResult
) -> ContextDeeplinkResponse:
    for goal in structured.contexts:
        for action in goal.actions:
            if action.category == actionCategory.auto:
                combined_text = " ".join(
                    [action.actionName]
                    + [s for grp in action.stepGroups for s in grp.steps]
                )
                inferred = _infer_non_auto_category(combined_text)
                if inferred is not None:
                    action.category = inferred
            else:
                # Stage 1 also mislabels in the other direction (seen in
                # results.jsonl: "Force a Restart" as manual, "Visit Service
                # Center" as critical, "Safe Mode" as manual). The brief is
                # explicit: restart/reset/safe mode/firmware = critical;
                # service centre / hardware = manual. Only the action NAME is
                # checked here -- steps of a manual action often mention
                # "restart" in passing, which must not escalate it.
                inferred = _infer_non_auto_category(action.actionName)
                if inferred is not None:
                    action.category = inferred
            for step_group in action.stepGroups:
                if action.category != actionCategory.auto:
                    # Only auto actions may carry deeplinks (manual is a schema rule;
                    # critical is our design choice).
                    step_group.actionableDeeplink = None
                    step_group.validationDeeplink = None
                    continue
                
                query_text = action.actionName + " " + " ".join(step_group.steps)
                match, score = _best_match(query_text, action.actionName)
                
                # If the full steps text is low confidence, recheck against just the action name
                if score < SIMILARITY_THRESHOLD:
                    alt_match, alt_score = _best_match(action.actionName, action.actionName)
                    if alt_score > score:
                        match, score = alt_match, alt_score

                if score >= SIMILARITY_THRESHOLD:
                    step_group.actionableDeeplink = Deeplink(
                        deeplink=match["deeplink"],
                        description=match["description"],
                        message=match.get("message", ""),
                        originalType=match.get("originalType"),
                    )
                    step_group.validationDeeplink = _to_validation_deeplink(match)
                else:
                    # Find another way to handle it: downgrade to manual instead of fabricating a dummy link
                    action.category = actionCategory.manual
                    step_group.actionableDeeplink = None
                    step_group.validationDeeplink = None
        # Plan hierarchy: critical/destructive actions last. Recategorisation
        # above can turn an early action critical, so re-sort (stable -- the
        # relative order Stage 1 chose is kept within each group).
        goal.actions.sort(key=lambda a: 1 if a.category == actionCategory.critical else 0)
    return structured
