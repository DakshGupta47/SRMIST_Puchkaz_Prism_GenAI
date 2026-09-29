from __future__ import annotations

import re
from typing import Any

from enrichment import EnrichmentResult


# Words that are too generic to be useful for determining
# whether a troubleshooting article is actually relevant.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "your",
    "you",
    "is",
    "are",
    "this",
    "that",
    "it",
    "be",
    "as",
    "at",
    "by",
    "my",
    "me",
    "device",
    "phone",
    "tablet",
    "samsung",
    "galaxy",
    "issue",
    "issues",
    "problem",
    "problems",
    "troubleshooting",
    "troubleshoot",
}


# Terms that describe the general troubleshooting domain but
# are too broad to be strong relevance signals by themselves.
_GENERIC_TERMS = {
    "screen",
    "display",
    "settings",
    "mobile",
    "software",
    "hardware",
    "app",
    "apps",
}
_STRONG_SYMPTOM_SIGNALS = {
    "touch",
    "lag",
    "flicker",
    "blank",
    "mirror",
    "rotation",
    "charge",
    "email",
    "secure",
    "camera",
    "network",
    "wifi",
    "bluetooth",
    "battery",
    "audio",
    "speaker",
    "microphone",
    "call",
    "keyboard",
    "gesture",
    "pen",
    "update",
    "restart",
    "overheat",
    "freeze",
    "crash",
    "crack"
}

# A small normalization vocabulary for common symptom wording.
# This is intentionally conservative.
_SYNONYM_GROUPS = {
    "touch": {
        "touch",
        "touchscreen",
        "touches",
        "touching",
    },
    "lag": {
        "lag",
        "laggy",
        "delay",
        "delayed",
        "delays",
        "slow",
        "responsiveness",
        "responsive",
    },
    "flicker": {
        "flicker",
        "flickering",
        "flickers",
        "flickered",
    },
    "blank": {
        "blank",
        "black",
        "dark",
    },
    "mirror": {
        "mirror",
        "mirroring",
        "mirrored",
    },
    "rotation": {
        "rotate",
        "rotates",
        "rotating",
        "rotation",
        "orientation",
    },
    "charge": {
        "charge",
        "charged",
        "charging",
        "charger",
    },
    "email": {
        "email",
        "emails",
        "mail",
    },
    "secure": {
        "secure",
        "folder",
        "securefolder",
    },
    "crack": {
    "crack",
    "cracked",
    "cracks",
    "shatter",
    "shattered",
    "broken",
    },
}


def _normalize_word(word: str) -> str:
    """
    Normalize one word into a small canonical form.
    """

    word = word.lower().strip()

    for canonical, variants in _SYNONYM_GROUPS.items():
        if word in variants:
            return canonical

    # Small amount of suffix normalization.
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]

    elif len(word) > 5 and word.endswith("ed"):
        word = word[:-2]

    elif len(word) > 5 and word.endswith("ly"):
        word = word[:-2]

    return word


def _tokens(text: str) -> set[str]:
    """
    Convert text into normalized meaningful terms.
    """

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

    tokens = set()

    for word in words:

        if word in _STOPWORDS:
            continue

        normalized = _normalize_word(word)

        if not normalized:
            continue

        if normalized in _STOPWORDS:
            continue

        tokens.add(normalized)

    return tokens


def assess_reference_relevance(
    siis_response: dict[str, Any],
    enrichment: EnrichmentResult,
) -> tuple[bool, float, set[str]]:
    """
    Determine whether the SIIS reference plausibly addresses
    the customer's symptom.

    Returns:
        (is_relevant, score, matched_terms)
    """

    title = str(
        siis_response.get("title", "")
    )

    content = str(
        siis_response.get("content", "")
    )

    # ---------------------------------------------------------
    # 1. Decide what customer text to use
    # ---------------------------------------------------------
    category = str(
        enrichment.symptom_category or ""
    ).lower()

    if category.startswith("unclassified"):
        # Stage 0 could not identify a structured symptom.
        # Use the raw customer query rather than noisy generated
        # canonical text.
        customer_text = str(
            enrichment.raw_query or ""
        )
    else:
        customer_text = " ".join(
            [
                str(enrichment.symptom_label or ""),
                str(enrichment.symptom_category or ""),
                str(enrichment.canonical_query or ""),
            ]
        )

    # ---------------------------------------------------------
    # 2. Convert customer text into known symptom signals
    # ---------------------------------------------------------
    customer_tokens = _tokens(customer_text)

    query_signals = {
        token
        for token in customer_tokens
        if token in _STRONG_SYMPTOM_SIGNALS
    }

    # ---------------------------------------------------------
    # 3. Convert reference into known symptom signals
    # ---------------------------------------------------------
    title_tokens = _tokens(title)
    content_tokens = _tokens(content)

    title_signals = {
        token
        for token in title_tokens
        if token in _STRONG_SYMPTOM_SIGNALS
    }

    content_signals = {
        token
        for token in content_tokens
        if token in _STRONG_SYMPTOM_SIGNALS
    }

    # ---------------------------------------------------------
    # 4. Compare customer symptom ↔ reference title
    # ---------------------------------------------------------
    title_matches = query_signals.intersection(
        title_signals
    )

    if title_matches:
        return (
            True,
            0.9,
            title_matches,
        )

    # ---------------------------------------------------------
    # 5. Compare customer symptom ↔ reference content
    # ---------------------------------------------------------
    content_matches = query_signals.intersection(
        content_signals
    )

    # Two independent symptom signals in the article
    # provide strong evidence.
    if len(content_matches) >= 2:
        return (
            True,
            0.75,
            content_matches,
        )

    # One specific symptom signal can be enough when
    # Stage 0 has already classified the issue.
    if (
        len(content_matches) == 1
        and not category.startswith("unclassified")
    ):
        return (
            True,
            0.6,
            content_matches,
        )

    # One signal can also be sufficient for an unclassified
    # issue when that signal is very specific.
    if (
        len(content_matches) == 1
        and len(query_signals) == 1
    ):
        return (
            True,
            0.55,
            content_matches,
        )

    # ---------------------------------------------------------
    # 6. No meaningful symptom relationship found
    # ---------------------------------------------------------
    return (
        False,
        0.0,
        content_matches,
    )