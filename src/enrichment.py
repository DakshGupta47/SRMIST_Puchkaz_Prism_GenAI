"""
Stage 0 — Query Enrichment (Member A)

    normalize_query(raw_complaint)  -> canonical technical query + extracted
                                        device / symptom metadata
    generate_variations(canonical)  -> 8-10 paraphrases spanning
                                        formal / casual / keyword /
                                        frustrated / typo registers

Design
------
Customer complaints arrive in wildly different registers (a careful
formal sentence, a two-word keyword search, a typo-ridden rant). Stage 1
(LLM structuring) and Stage 2 (deeplink retrieval) both work far better
against a single clean technical query than against raw free text, and
the semantic cache (Stage 3) gets much better hit-rate recall if it has
several paraphrases of the same underlying request to match against
(that's what the roadmap calls "A5 — query variations": without them, a
cache primed on one phrasing of "screen is black" won't hit on "display
won't turn on").

`normalize_query` is regex/keyword based, not an LLM call: it extracts
the device model and a symptom category from a small taxonomy built by
inspecting the 20 sample complaints in data/input.txt (screen blank/black,
flicker, crack, touch unresponsive, touch lag, half-screen dark, inner
screen failure, distorted display, undersized display, floating
assistant icon). This is fast, deterministic, and — critically — cannot
hallucinate a symptom that isn't there, which matters for the "no
hallucinations" requirement.

`generate_variations` is template-based per symptom category by default
(deterministic, always produces valid output even with zero API keys
configured), and will additionally ask the configured LLM for extra
paraphrases when one is available (see llm_client.py) — those are
programmatically validated (non-empty, reasonable length, deduped)
before being mixed in, so a misbehaving LLM call degrades gracefully
instead of corrupting the output.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from llm_client import LLMClient, MockLLMClient, get_llm_client

REGISTERS = ["formal", "casual", "keyword", "frustrated", "typo"]
MIN_VARIATIONS = 8
MAX_VARIATIONS = 10

# ---------------------------------------------------------------------------
# Device extraction
# ---------------------------------------------------------------------------

_DEVICE_PATTERN = re.compile(
    r"(Galaxy\s+Z\s+Flip\s*\d+|Galaxy\s+Flip\s*\d+|Z\s+Flip\s*\d+"
    r"|Galaxy\s+Z\s+Fold\s*\d+|Galaxy\s+Fold\s*\d+"
    r"|Galaxy\s+S\d+\s*(?:Ultra|Plus|\+|FE)?"
    r"|Galaxy\s+A\d+(?:\s*/\s*A\d+)?"
    r"|Galaxy\s+Note\s*\d+\s*(?:Ultra|\+)?"
    r"|Galaxy\s+Tab\s*[A-Za-z0-9]*\s*(?:FE|Ultra|\+)?|Galaxy\s+tablet"
    r"|Galaxy\s+Watch\s*\d*\s*(?:Classic|Pro|Ultra)?"
    r"|Galaxy\s+Buds\s*[A-Za-z0-9]*\s*(?:Pro)?"
    r"|SM-[A-Z]\d{3,4}[A-Z0-9]*"
    r"|Samsung\s+[A-Z0-9]{3,}G?\s+tablet"
    r"|Samsung\s+Galaxy\s+[A-Za-z0-9]+(?:\s*/\s*[A-Za-z0-9]+)?(?:\s+Ultra)?"
    r"|Samsung\s+S\*+\s*Ultra)",
    re.IGNORECASE,
)


UNKNOWN_DEVICE_LABEL = "Samsung device"  # fallback when no specific model is recognized


def extract_device(raw: str) -> str:
    match = _DEVICE_PATTERN.search(raw)
    if not match:
        return UNKNOWN_DEVICE_LABEL
    device = re.sub(r"\s+", " ", match.group(0)).strip()
    # Normalize casing: "galaxy s22" -> "Galaxy S22"
    return " ".join(w[0].upper() + w[1:] if w and w[0].isalpha() else w for w in device.split())


# ---------------------------------------------------------------------------
# Symptom taxonomy — (category, matcher keywords, formal/casual/keyword text)
# ---------------------------------------------------------------------------

@dataclass
class Symptom:
    category: str
    label: str
    # Matching is (component word present) AND (problem word present), not one long
    # exact phrase — an earlier version required e.g. "camera app crashes" verbatim
    # and silently missed "the camera app on my S24 keeps crashing" (different word
    # order/tense). component_terms=[] means "no specific part" (whole-device
    # symptoms like overheating/slow performance), so only problem_terms need match.
    component_terms: List[str]
    problem_terms: List[str]
    subject: str  # the device component this is about — "screen", "battery", "Wi-Fi
                  # connection", etc. — inserted by the template, never assumed to be
                  # "screen" (see the canonical-query bug this replaced: a battery
                  # complaint was coming out as "screen is exhibiting a display issue").
    formal: str
    casual: str
    keyword: str


SYMPTOM_TAXONOMY: List[Symptom] = [
    # -- display/screen (data/input.txt's 20 samples are all in this group) --
    # More specific screen categories are listed first: extract_symptom() keeps the
    # first symptom seen on a tied score, so e.g. flicker-then-blank (2 hits: "screen"
    # + "flicker") outranks the generic blank/black category on a complaint that
    # mentions both, matching how these 20 samples actually read.
    Symptom(
        "screen_flicker_then_blank",
        "Screen flickers then goes blank",
        ["screen", "display"],
        ["flash", "flicker"],
        "screen",
        "flickers briefly and then goes completely blank",
        "keeps flickering and then just goes blank",
        "flicker then blank",
    ),
    Symptom(
        "screen_ghost_touch",
        "Screen registers touches on its own (ghost touches)",
        ["screen", "touch", "display"],
        ["ghost touch", "on their own", "on its own", "by itself", "random touches", "registering on"],
        "screen",
        "registers touch input on its own without anyone touching it",
        "keeps tapping stuff by itself, ghost touches everywhere",
        "ghost touch screen registers on its own",
    ),
    Symptom(
        "touch_unresponsive",
        "Touchscreen unresponsive / can't interact",
        ["screen", "touch", "touchscreen", "display"],
        ["unresponsive", "doesn't respond", "does not respond", "won't respond", "wont respond",
         "not responding", "can't interact", "cant interact", "unable to interact"],
        "touchscreen",
        "does not respond to touch input at all",
        "won't respond no matter how I tap it",
        "unresponsive no input",
    ),
    Symptom(
        "touch_lag",
        "Touch input delayed / laggy",
        ["touch", "tap", "input"],
        ["delayed", "laggy", "lag", "noticeable delay", "slow to respond"],
        "touch input",
        "exhibits a noticeable delay before responding",
        "feels laggy, there's a delay when I tap stuff",
        "delay laggy",
    ),
    Symptom(
        "screen_blank_black",
        "Screen completely blank/black, no display output",
        ["screen", "display"],
        ["completely blank", "goes blank", "blank", "black screen", "completely black",
         "totally black", "went black", "stays dark", "won't turn on", "wont turn on",
         "no image", "doesn't display anything", "dark screen", "black", "white and no text",
         "is dead", "screen dead", "dead screen"],
        "screen",
        "displays no image at all and will not turn on",
        "is just totally black, nothing shows up",
        "black no display",
    ),
    Symptom(
        "screen_cracked",
        "Screen physically cracked/damaged",
        ["screen", "display"],
        ["crack", "cracked", "shattered", "shatter"],
        "screen",
        "has a physical crack across it",
        "is cracked and busted up",
        "cracked physical damage",
    ),
    Symptom(
        "half_screen_dark",
        "Half of the display is dark/unresponsive",
        ["screen", "display", "half", "side"],
        ["half black", "one side", "half dark", "half is completely dark"],
        "screen",
        "shows one half completely dark while the other half functions normally",
        "half is dead, other half's fine",
        "half dark half working",
    ),
    Symptom(
        "inner_screen_failure",
        "Inner/foldable screen failed, outer screen fine",
        ["inner screen", "cover screen", "fold"],
        ["stopped working", "no image", "dead", "doesn't respond", "not responding", "stopped"],
        "inner foldable display",
        "has stopped producing an image or responding to touch, while the cover screen remains functional",
        "just died but the cover screen still works",
        "dead, cover screen works, foldable",
    ),
    Symptom(
        "distorted_display",
        "Display output visually distorted / discolored",
        ["screen", "display"],
        ["distorted", "warped", "glitch", "green tint", "color tint", "discolor", "weird tint",
         "weird color", "tint across"],
        "screen",
        "renders visual output that appears distorted or shows an abnormal color tint",
        "looks all warped and messed up, weird colors and everything",
        "distorted visual glitch tint",
    ),
    Symptom(
        "screen_partial_lit",
        "Only part of the display lights up",
        ["screen", "display", "icons"],
        ["only three app icons", "partially lit", "some icons light up", "stays dark and only"],
        "screen",
        "only illuminates a small portion while the remainder stays dark",
        "only a few icons light up, rest is dark",
        "partially lit rest dark",
    ),
    Symptom(
        "screen_undersized",
        "Display doesn't fill full screen size",
        ["screen", "display"],
        ["doesn't fill", "stays small", "expand to full size", "not full size", "shrunk down"],
        "screen",
        "does not scale to fill the full display area",
        "looks all shrunk down, not using the whole screen",
        "small not full size scaling",
    ),
    Symptom(
        "floating_assistant_icon",
        "Unwanted floating assistant/shortcut icon on screen",
        ["screen", "icon", "overlay", "circle", "bubble"],
        ["floating", "hovers", "shortcuts"],
        "screen",
        "displays a persistent floating icon overlay that I would like removed",
        "has this annoying floating bubble thing I want gone",
        "remove floating assistant icon overlay",
    ),

    # -- battery / power --
    Symptom(
        "battery_drain",
        "Battery drains unusually fast",
        ["battery"],
        ["drain", "dies so fast", "dead by", "doesn't last", "runs out", "dying fast",
         "life is bad", "fast", "drains"],
        "battery",
        "drains unusually quickly even under light everyday use",
        "is dying crazy fast, barely lasts half a day",
        "drains fast battery life",
    ),
    Symptom(
        "charging_fails_or_slow",
        "Device won't charge or charges very slowly",
        ["charg"],  # stem: matches charge/charging/charger/charges
        ["won't", "wont", "not charging", "slowly", "slow", "barely", "stopped", "doesn't", "no matter what cable"],
        "charging",
        "is extremely slow or does not charge at all when plugged in",
        "is barely charging, if at all, no matter how long it's plugged in",
        "wont charge charging slow",
    ),
    Symptom(
        "overheating",
        "Device overheats during normal use",
        [],
        ["overheat", "gets very hot", "getting hot", "too hot", "extremely hot", "gets hot",
         "gets extremely hot", "heats up"],
        "device",
        "becomes excessively hot during normal use",
        "gets super hot for no reason",
        "overheating hot device",
    ),
    Symptom(
        "random_restarts",
        "Device randomly restarts/reboots",
        [],
        ["randomly restart", "reboots on its own", "restarts by itself", "random reboot",
         "keeps restarting", "restarts on its own"],
        "device",
        "randomly restarts on its own with no warning or error message",
        "just reboots itself out of nowhere, no warning",
        "random restart reboot itself",
    ),
    Symptom(
        "sluggish_performance",
        "Device runs slow/laggy overall",
        [],
        ["running slow", "gotten slow", "is slow", "everything is laggy", "takes forever",
         "very sluggish", "freezes a lot", "hangs a lot"],
        "device",
        "runs noticeably slower than usual, with everything taking longer to respond",
        "is laggy and slow all of a sudden, everything takes forever",
        "running slow laggy performance",
    ),

    # -- connectivity --
    Symptom(
        "wifi_connectivity",
        "Wi-Fi keeps disconnecting or won't connect",
        ["wifi", "wi-fi", "wireless network"],
        ["disconnect", "won't connect", "wont connect", "can't connect", "cant connect",
         "drop", "keeps dropping"],
        "Wi-Fi connection",
        "repeatedly disconnects from Wi-Fi networks or fails to connect at all",
        "keeps dropping out on me",
        "wifi keeps disconnecting",
    ),
    Symptom(
        "bluetooth_pairing",
        "Bluetooth won't pair or keeps disconnecting",
        ["bluetooth"],
        ["won't pair", "wont pair", "keeps disconnecting", "not connecting", "won't connect",
         "wont connect", "pairing"],
        "Bluetooth connection",
        "fails to pair with, or keeps disconnecting from, other accessories",
        "won't connect to my earbuds no matter what",
        "bluetooth wont pair disconnects",
    ),
    Symptom(
        "no_signal_network",
        "No mobile signal / can't call or text",
        ["signal", "network", "service", "bars", "carrier"],
        ["no signal", "no service", "no network", "can't call", "cant call", "cannot receive",
         "zero bars", "no bars", "fail to go through"],
        "mobile network signal",
        "shows no signal, so calls and texts fail to go through",
        "has zero bars, can't call or text anyone",
        "no signal no network",
    ),

    # -- audio --
    Symptom(
        "distorted_sound",
        "Audio is distorted or crackling",
        ["sound", "audio", "speaker", "earbuds", "buds", "call quality"],
        ["distort", "crackl", "static", "garbled", "choppy"],  # "crackl" stems crackly/crackling
        "audio output",
        "is distorted or crackling during calls and playback",
        "sounds all crackly and messed up",
        "distorted crackling audio",
    ),
    Symptom(
        "no_sound",
        "No sound from speaker",
        ["sound", "speaker", "audio", "volume"],
        ["no sound", "silent", "can't hear", "cant hear", "dead silent", "not producing sound"],
        "speaker",
        "produces no sound at all during calls or media playback",
        "is dead silent, no sound comes out",
        "no sound speaker silent",
    ),

    # -- camera --
    Symptom(
        "camera_crashes",
        "Camera app won't open or crashes",
        ["camera"],
        ["crash", "force clos", "force-clos", "won't open", "wont open", "freeze", "closes"],
        "camera app",
        "fails to open or force-closes immediately when launched",
        "just crashes the second I try to open it",
        "camera crashes wont open",
    ),
    Symptom(
        "camera_blurry",
        "Photos come out blurry",
        ["camera", "photo", "picture", "pic"],
        ["blurry", "out of focus", "fuzzy", "not clear", "grainy"],
        "camera",
        "produces photos that are consistently blurry or out of focus",
        "takes photos that come out all blurry",
        "camera blurry photos out of focus",
    ),

    # -- storage / apps --
    Symptom(
        "storage_full",
        "Storage full, can't install apps/updates",
        ["storage", "memory", "space"],
        ["full", "out of", "not enough", "no space", "almost full"],
        "internal storage",
        "is full, preventing new apps or updates from installing",
        "is totally full, can't download or update anything",
        "storage full cant install",
    ),
    Symptom(
        "app_crashing",
        "A specific app keeps crashing",
        ["app"],
        ["crash", "force clos", "force-clos", "keeps closing", "stops working", "closing"],
        "an app",
        "repeatedly crashes or force-closes during use",
        "just keeps crashing on me over and over",
        "app keeps crashing force close",
    ),

    # -- security / biometrics / notifications / updates --
    Symptom(
        "fingerprint_face_unlock_fail",
        "Fingerprint/face unlock not working",
        ["fingerprint", "face recognition", "face unlock", "biometric"],
        ["won't", "wont", "not working", "stopped working", "fail", "doesn't recognize",
         "won't recognize", "wont recognize", "not recognizing", "recognize me"],
        "fingerprint/face unlock",
        "repeatedly fails to recognize and unlock the device",
        "scanner just won't recognize me anymore",
        "fingerprint face unlock not working",
    ),
    Symptom(
        "no_notifications",
        "Not receiving notifications",
        ["notification"],
        ["not getting", "not receiving", "not showing", "missing", "stopped getting",
         "no notifications", "aren't showing"],
        "notifications",
        "are not being delivered for messages or app alerts",
        "just aren't showing up at all anymore",
        "notifications not showing up missing",
    ),
    Symptom(
        "software_update_fails",
        "Software update fails to install",
        ["update"],
        ["fail", "won't install", "wont install", "error", "keeps failing", "stuck",
         "doesn't install"],
        "software update",
        "repeatedly fails to download or install",
        "keeps failing every single time I try",
        "software update fails wont install",
    ),
]

_DEFAULT_SYMPTOM = Symptom(
    "unclassified_issue",
    "Unclassified device issue (needs manual review)",
    [],
    [],
    "device",
    "is experiencing an issue that could not be automatically classified from the description given",
    "is just not working right and I'm not sure why",
    "device issue unclassified",
)


def _symptom_confidence_from_hits(hits: int) -> float:
    """Map a keyword-match score to a confidence in [0.0, 1.0]. Saturating,
    not linear-unbounded: a single weak hit (possible only for whole-device
    categories with one problem-term match and no component gate) is real
    evidence but not strong evidence, so it lands at 0.5, not 1.0.
    """
    if hits <= 0:
        return 0.0
    return min(1.0, 0.5 + 0.25 * (hits - 1))


def extract_symptom(raw: str) -> Tuple[Symptom, float]:
    """Score each taxonomy entry by (component word present) + (problem word
    present) rather than requiring one long exact phrase. A whole-device
    symptom (component_terms=[]) is scored on problem terms alone; a
    part-specific symptom (e.g. camera, battery, Wi-Fi) needs at least one hit
    from EACH list to be eligible at all — this is what stops a stray
    substring collision (e.g. "crackly" containing "crack") from misfiring a
    screen_cracked match on an audio complaint that never mentions a screen.

    Returns (symptom, confidence). confidence is 0.0 for the unclassified
    fallback — callers (pipeline.py, the API response, Stage 1/2) can use
    this to know Stage 0 has no real signal here, rather than silently
    treating "unclassified_issue" as just another category with no asterisk
    on it.
    """
    lowered = raw.lower()
    best: Optional[Symptom] = None
    best_hits = 0
    for symptom in SYMPTOM_TAXONOMY:
        component_hits = sum(1 for kw in symptom.component_terms if kw in lowered)
        problem_hits = sum(1 for kw in symptom.problem_terms if kw in lowered)
        if symptom.component_terms and component_hits == 0:
            continue  # part-specific symptom, but that part was never mentioned
        if problem_hits == 0:
            continue  # no evidence of the actual problem, regardless of component
        hits = component_hits + problem_hits
        if hits > best_hits:
            best, best_hits = symptom, hits
    if best is None:
        return _DEFAULT_SYMPTOM, 0.0
    return best, _symptom_confidence_from_hits(best_hits)


# ---------------------------------------------------------------------------
# Canonical query + variations
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    raw_query: str
    canonical_query: str
    device: str
    symptom_category: str
    symptom_label: str
    device_confidence: float = 0.0    # 1.0 if a specific model was recognized, 0.0 if the
                                       # generic "Samsung device" fallback was used
    symptom_confidence: float = 0.0   # 0.0 for "unclassified_issue"; see _symptom_confidence_from_hits
    query_variations: List[str] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        """The weaker of the two signals — a confidently-identified device
        with an unclassified symptom (or vice versa) is still a low-confidence
        enrichment overall; don't let one strong signal mask the other being
        absent.
        """
        return min(self.device_confidence, self.symptom_confidence)

    @property
    def is_low_confidence(self) -> bool:
        """True when Stage 0 genuinely doesn't know what it's looking at.
        Downstream stages / the API response should treat this as a signal
        to be conservative — e.g. lower a Goal's score, or prefer a null
        deeplink over a guessed one — not as license to answer anyway."""
        return self.overall_confidence < 0.5

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "canonical_query": self.canonical_query,
            "device": self.device,
            "symptom_category": self.symptom_category,
            "symptom_label": self.symptom_label,
            "device_confidence": self.device_confidence,
            "symptom_confidence": self.symptom_confidence,
            "overall_confidence": self.overall_confidence,
            "is_low_confidence": self.is_low_confidence,
            "query_variations": self.query_variations,
        }


def _clean_raw(raw: str) -> str:
    text = raw.strip()
    # strip leading numbering like "1." or list markers, and wrapping quotes
    text = re.sub(r'^\s*\d+[\.\)]\s*', "", text)
    text = text.strip('"“”\' ')
    return text


def normalize_query(raw_complaint: str) -> EnrichmentResult:
    """Turn a raw customer complaint into a canonical technical query.

    Deterministic (regex + keyword taxonomy) — no LLM call, so it cannot
    invent a device or symptom that isn't actually present in the text.
    """
    cleaned = _clean_raw(raw_complaint)
    device = extract_device(cleaned)
    symptom, symptom_confidence = extract_symptom(cleaned)
    device_confidence = 0.0 if device == UNKNOWN_DEVICE_LABEL else 1.0
    canonical = f"{device} {symptom.subject} {symptom.formal}."
    return EnrichmentResult(
        raw_query=raw_complaint,
        canonical_query=canonical,
        device=device,
        symptom_category=symptom.category,
        symptom_label=symptom.label,
        device_confidence=device_confidence,
        symptom_confidence=symptom_confidence,
    )


def _inject_typos(text: str, seed: int) -> str:
    """Deterministic, mild typo injection: adjacent-letter swaps and a
    dropped vowel here and there — enough to exercise fuzzy matching
    without mangling the query beyond recognition.
    """
    rng = random.Random(seed)
    chars = list(text)
    n_edits = max(1, len(chars) // 25)
    for _ in range(n_edits):
        if len(chars) < 4:
            break
        i = rng.randrange(1, len(chars) - 1)
        op = rng.choice(["swap", "drop", "double"])
        if op == "swap" and chars[i].isalpha() and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop" and chars[i].lower() in "aeiou":
            chars.pop(i)
        elif op == "double" and chars[i].isalpha():
            chars.insert(i, chars[i])
    return "".join(chars)


def _template_variations(result: EnrichmentResult) -> List[str]:
    device, symptom = result.device, next(
        (s for s in SYMPTOM_TAXONOMY if s.category == result.symptom_category), _DEFAULT_SYMPTOM
    )
    variations: List[str] = [
        f"I am experiencing an issue where the {device} {symptom.subject} {symptom.formal}. "
        f"Could you please advise on the appropriate troubleshooting steps?",
        f"The {symptom.subject} on my {device} {symptom.formal}; I would appreciate guidance on how to resolve this.",
        f"hey so my {device} {symptom.subject} {symptom.casual}, kinda annoying, help?",
        f"my {device}'s {symptom.subject} {symptom.casual} idk whats going on",
        f"{device} {symptom.keyword}",
        f"{device} {symptom.keyword} fix",
        f"This is so frustrating!! My {device} {symptom.subject} {symptom.casual} and nothing I do works!",
        f"I'm really annoyed, my {device} {symptom.subject} {symptom.formal} and I've tried everything already!",
    ]
    # two typo-register variants, seeded off the canonical text for determinism
    variations.append(_inject_typos(f"{device} {symptom.subject} {symptom.casual}", seed=1))
    variations.append(_inject_typos(f"{device} {symptom.subject} {symptom.formal}", seed=2))
    return variations


def _llm_variations(result: EnrichmentResult, client: LLMClient, n: int) -> List[str]:
    system_prompt = (
        "You paraphrase a customer's device-support complaint into short alternate "
        "phrasings for search/cache matching. Only reword the given complaint — "
        "never add new symptoms, devices, or facts that are not already present. "
        "Return a JSON array of strings and nothing else."
    )
    user_prompt = (
        f"Canonical query: {result.canonical_query}\n"
        f"Original complaint: {result.raw_query}\n"
        f"Give {n} short alternate phrasings of this exact same complaint."
    )
    try:
        raw = client.complete(system_prompt, user_prompt)
        candidates = json.loads(raw)
        if not isinstance(candidates, list):
            return []
        return [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    except Exception:
        return []


def generate_variations(
    result: EnrichmentResult,
    llm_client: Optional[LLMClient] = None,
) -> List[str]:
    """Produce 8-10 query variations spanning formal / casual / keyword /
    frustrated / typo registers.

    Always returns a valid, deduped, length-bounded list even if no LLM
    is configured — the template generator alone satisfies the 8-10
    requirement across all five registers. If a real LLM client is
    available it contributes extra paraphrases, which are validated the
    same way before being mixed in (never trusted to self-constrain).
    """
    base = _template_variations(result)

    client = llm_client or get_llm_client()
    if not isinstance(client, MockLLMClient):
        extra = _llm_variations(result, client, n=4)
        # validate: non-empty, plausible length, must still mention the device
        # or symptom keyword so a hallucinated unrelated paraphrase is dropped
        _matched_symptom = next(
            (s for s in SYMPTOM_TAXONOMY if s.category == result.symptom_category), _DEFAULT_SYMPTOM
        )
        _relevance_terms = _matched_symptom.component_terms + _matched_symptom.problem_terms
        valid_extra = [
            e for e in extra
            if 3 <= len(e.split()) <= 40
            and (result.device.split()[0].lower() in e.lower()
                 or any(kw in e.lower() for kw in _relevance_terms))
        ]
        base = base[:6] + valid_extra  # keep template's typo slots, swap in LLM diversity

    seen = set()
    deduped: List[str] = []
    for v in base:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(v.strip())

    # Guarantee the [8, 10] contract regardless of how many survived.
    while len(deduped) < MIN_VARIATIONS:
        deduped.append(_inject_typos(result.canonical_query, seed=len(deduped) + 10))
    return deduped[:MAX_VARIATIONS]


def enrich(raw_complaint: str, llm_client: Optional[LLMClient] = None) -> EnrichmentResult:
    """Full Stage 0 entry point: normalize + generate variations."""
    result = normalize_query(raw_complaint)
    result.query_variations = generate_variations(result, llm_client=llm_client)
    return result
