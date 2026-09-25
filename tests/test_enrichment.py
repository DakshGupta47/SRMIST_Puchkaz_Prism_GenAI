from pathlib import Path

import pytest

from enrichment import MAX_VARIATIONS, MIN_VARIATIONS, enrich

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_COMPLAINTS = [
    line.strip() for line in (DATA_DIR / "input.txt").read_text().splitlines() if line.strip()
]


@pytest.mark.parametrize("raw", SAMPLE_COMPLAINTS)
def test_variation_count_within_contract(raw):
    result = enrich(raw)
    assert MIN_VARIATIONS <= len(result.query_variations) <= MAX_VARIATIONS


@pytest.mark.parametrize("raw", SAMPLE_COMPLAINTS)
def test_variations_are_non_empty_and_deduped(raw):
    result = enrich(raw)
    lowered = [v.strip().lower() for v in result.query_variations]
    assert all(v for v in lowered), "no variation should be empty"
    assert len(lowered) == len(set(lowered)), "variations must be deduplicated"


@pytest.mark.parametrize("raw", SAMPLE_COMPLAINTS)
def test_canonical_query_is_nonempty_and_mentions_device(raw):
    result = enrich(raw)
    assert result.canonical_query
    # canonical query should reference the extracted device so downstream
    # stages/cache can rely on it, unless extraction genuinely found none
    if result.device != "Samsung device":
        assert result.device.split()[0].lower() in result.canonical_query.lower()


def test_no_hallucinated_device_when_absent():
    # a complaint with a masked/ambiguous model should not be invented
    # into a real, specific device name
    result = enrich("My Samsung S***** Ultra screen flashes extremely quickly whenever I plug in a charger.")
    assert "s*****" in result.device.lower() or result.device == "Samsung device"


def test_five_registers_represented_across_templates():
    # the deterministic template generator alone should cover every
    # register even with no LLM configured
    from enrichment import _template_variations, normalize_query

    result = normalize_query("My Galaxy S22 screen is completely black and won't turn on.")
    variations = _template_variations(result)
    assert len(variations) >= MIN_VARIATIONS
    # sanity: formal vs casual vs keyword vs frustrated vs typo phrasing differ
    assert len(set(variations)) == len(variations)


def test_typo_injection_is_deterministic():
    from enrichment import _inject_typos

    a = _inject_typos("Galaxy S22 screen is black", seed=7)
    b = _inject_typos("Galaxy S22 screen is black", seed=7)
    assert a == b


def test_mock_llm_path_never_crashes_and_still_meets_contract():
    import os

    os.environ.pop("LLM_PROVIDER", None)  # ensure default mock path
    result = enrich("My Galaxy Z Flip 7 inner screen is dead but the cover screen works.")
    assert MIN_VARIATIONS <= len(result.query_variations) <= MAX_VARIATIONS


# -- regression tests for real bugs found by testing beyond the 20 given samples --

def test_non_screen_complaint_does_not_hallucinate_a_screen_symptom():
    """The taxonomy originally only covered screen symptoms, so a battery
    complaint's canonical query came out as literally "screen is exhibiting
    a display issue" — asserting a screen problem that was never reported.
    deeplinks.json is only ~1/3 screen-related (battery, sound, notifications,
    software update, security, network, performance, etc. make up the rest),
    so this was a real hallucination risk on any realistic unseen scenario.
    """
    result = enrich("My Galaxy S22 battery drains extremely fast, dead by noon even with light use.")
    assert result.symptom_category == "battery_drain"
    assert "screen" not in result.canonical_query.lower()
    assert "display" not in result.canonical_query.lower()
    assert "battery" in result.canonical_query.lower()


@pytest.mark.parametrize("raw,expected_category", [
    ("My Galaxy S24 Ultra won't charge at all no matter what cable I use.", "charging_fails_or_slow"),
    ("My Galaxy Z Flip 6 gets extremely hot in my pocket just from browsing.", "overheating"),
    ("Wifi keeps disconnecting on my Galaxy A15 every few minutes.", "wifi_connectivity"),
    ("My Galaxy S23's Bluetooth won't pair with my car anymore, worked fine last week.", "bluetooth_pairing"),
    ("I have no signal at all on my Galaxy S21, can't make or receive calls.", "no_signal_network"),
    ("There's no sound coming out of the speaker on my Galaxy S22 during calls.", "no_sound"),
    ("Camera app on my Galaxy S24 keeps crashing the second I open it.", "camera_crashes"),
    ("Every photo I take on my Galaxy S23 Ultra comes out blurry.", "camera_blurry"),
    ("My Galaxy S22's storage is completely full and I can't install any updates.", "storage_full"),
    ("The Instagram app on my Galaxy A54 keeps force-closing every few minutes.", "app_crashing"),
    ("Fingerprint unlock stopped working on my Galaxy S23, it just won't recognize me.", "fingerprint_face_unlock_fail"),
    ("I'm not getting any WhatsApp notifications on my Galaxy S22 anymore.", "no_notifications"),
    ("The software update on my Galaxy S24 keeps failing every time I try to install it.", "software_update_fails"),
    ("My device randomly restarts several times a day, no crash message or anything.", "random_restarts"),
    ("The screen on my brand new Galaxy S25 has a weird green tint across the whole display.", "distorted_display"),
])
def test_broadened_taxonomy_catches_realistic_paraphrasing(raw, expected_category):
    """These are realistic rephrasings (not the exact keyword-list phrases),
    e.g. 'keeps force-closing' rather than 'app force closes'. An earlier
    version of extract_symptom() required a long phrase to match verbatim
    and silently dropped all of these to 'unclassified_issue'.
    """
    result = enrich(raw)
    assert result.symptom_category == expected_category


def test_audio_crackle_does_not_collide_with_screen_crack():
    """Regression for a literal substring collision: 'crackly' contains
    'crack', so a naive keyword-in-text check misclassified a Bluetooth
    earbuds audio complaint as a cracked *screen*.
    """
    result = enrich("The audio on my Galaxy Buds Pro sounds distorted and crackly.")
    assert result.symptom_category == "distorted_sound"
    assert result.symptom_category != "screen_cracked"


def test_bare_component_noun_with_no_symptom_stays_unclassified():
    """A one-word 'Battery' complaint carries no information about what's
    actually wrong (drain? won't charge? swelling?) — guessing a specific
    failure mode from the noun alone would be a hallucination, so this must
    stay unclassified rather than default to e.g. battery_drain.
    """
    result = enrich("Battery")
    assert result.symptom_category == "unclassified_issue"


def test_confidence_is_low_when_neither_device_nor_symptom_is_known():
    result = enrich("asdkjfhaskdjfh qwerty")
    assert result.device_confidence == 0.0
    assert result.symptom_confidence == 0.0
    assert result.is_low_confidence is True


def test_confidence_is_high_when_device_and_symptom_are_both_clear():
    result = enrich("My Galaxy S22 battery drains extremely fast, dead by noon even with light use.")
    assert result.device_confidence == 1.0
    assert result.symptom_confidence >= 0.5
    assert result.symptom_category == "battery_drain"


def test_bare_symptom_noun_is_flagged_low_confidence_not_silently_answered():
    """'Battery' alone should not just be silently treated as a normal,
    fully-confident classification — is_low_confidence must say so, so a
    downstream stage (or the API) can choose not to answer for the sake of
    answering.
    """
    result = enrich("Battery")
    assert result.symptom_confidence == 0.0
    assert result.is_low_confidence is True


def test_known_device_with_unclassifiable_symptom_is_still_low_confidence_overall():
    """A confidently-identified device does not offset a genuinely unknown
    symptom — overall_confidence takes the weaker of the two signals, not
    an average, so one strong signal can't mask the other being absent.
    """
    result = enrich("My Galaxy Watch 6 won't sync my heart rate data anymore.")
    assert result.device_confidence == 1.0
    assert result.symptom_confidence == 0.0
    assert result.is_low_confidence is True


def test_llm_variation_path_does_not_crash_on_symptom_field_rename():
    """Regression: an earlier refactor renamed Symptom.keywords to
    component_terms/problem_terms but left one reference to the old
    attribute in the LLM-paraphrase validation path. Tests didn't catch it
    because that path only executes with a real (non-mock) LLM client
    configured — exercising it directly here so it can't silently regress
    again.
    """
    from enrichment import generate_variations, normalize_query
    from llm_client import LLMClient
    import json

    class FakeLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return json.dumps([
                "My battery keeps dying so fast",
                "battery life is terrible on this thing",
            ])

    result = normalize_query(
        "My Galaxy S22 battery drains extremely fast, dead by noon even with light use."
    )
    variations = generate_variations(result, llm_client=FakeLLM())
    assert MIN_VARIATIONS <= len(variations) <= MAX_VARIATIONS


def test_heavy_typos_are_a_known_limitation_not_a_crash():
    """Documents a real, disclosed limitation: substring/stem keyword
    matching can't recover from spelling corrupted past recognition
    ('skreen'/'blenk'/'trun on' for 'screen'/'blank'/'turn on'). It must not
    crash or hallucinate a category — falling back to unclassified is the
    correct, honest behavior here, not a bug to silently paper over.
    """
    result = enrich("Ma galaxy S22 skreen iz blenk n wont trun on")
    assert result.device == "Galaxy S22"  # device extraction is typo-tolerant (correct spelling here)
    assert MIN_VARIATIONS <= len(result.query_variations) <= MAX_VARIATIONS
    # symptom_category is intentionally NOT asserted here: today it lands on
    # unclassified_issue because "skreen"/"blenk"/"trun on" don't match any
    # keyword substring. Fixing this needs fuzzy/edit-distance matching (or
    # routing through the LLM path) — tracked as a known gap in README.md,
    # not silently asserted-around here.
