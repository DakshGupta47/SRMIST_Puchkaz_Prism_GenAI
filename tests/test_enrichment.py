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


@pytest.mark.parametrize("raw,expected_category", [
    ("My Galaxy S23's keyboard keeps messing up when I type, letters come out wrong.", "keyboard_typing_problem"),
    ("The power button on my Galaxy S22 feels stuck and is hard to press.", "hardware_button_unresponsive"),
    ("My mobile data keeps cutting out on my Galaxy A54, no internet unless I'm on wifi.", "mobile_data_connectivity"),
    ("GPS on my Galaxy S24 keeps showing me in the wrong location, way off from where I am.", "gps_location_inaccurate"),
    ("My Galaxy S22 stopped vibrating for calls and texts, used to work fine.", "vibration_not_working"),
    ("TalkBack just stopped reading anything on my screen out loud on my Galaxy S21.", "screen_reader_accessibility_fail"),
])
def test_categories_added_from_deeplinks_json_gap_analysis(raw, expected_category):
    """These 6 categories didn't exist in the original 26 — they were found by
    scanning data/deeplinks.json's actual message/qna_description text for
    clusters with no matching taxonomy entry (keyboard, power/volume button,
    mobile data, GPS, vibration, TalkBack all show real frequency there). Not
    guessed: derived from the same catalog the deliverable has to map answers
    against, the same way the original 26 were.
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


def test_llm_fallback_classifies_a_complaint_the_keyword_matcher_could_not():
    """The keyword matcher structurally cannot classify a symptom outside the
    taxonomy or too heavily typo'd to match — it correctly stays
    unclassified_issue. The guarded LLM fallback exists to do better *when a
    real LLM is configured*: pick one of the known category codes so Stage
    1/2 get a specific canonical query instead of the generic fallback text.
    Using a fake LLM client so this doesn't depend on network access.
    """
    from enrichment import LLM_FALLBACK_CONFIDENCE, normalize_query
    from llm_client import LLMClient
    import json

    class FakeLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return json.dumps({"category": "keyboard_typing_problem"})

    # deliberately phrased so the keyword matcher finds nothing: no "keyboard"
    # component word, and no "when I type"/"while typing"-style phrase either
    # (those two are now in the taxonomy directly — see
    # test_common_typing_phrasing_is_now_caught_deterministically below).
    result = normalize_query(
        "My Galaxy S23 messages come out full of random symbols instead of the letters I actually pressed.",
        llm_client=FakeLLM(),
    )
    assert result.symptom_category == "keyboard_typing_problem"
    assert result.classification_source == "llm_fallback"
    assert result.symptom_confidence == LLM_FALLBACK_CONFIDENCE
    assert result.is_low_confidence is True  # still conservative — never cached


def test_llm_fallback_never_overrides_a_successful_keyword_match():
    """The fallback must only ever run when the deterministic matcher found
    NOTHING. Proving it with a fake LLM that would return an obviously wrong
    category if it were ever consulted for an already-classified complaint.
    """
    from enrichment import normalize_query
    from llm_client import LLMClient
    import json

    class WrongAnswerLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return json.dumps({"category": "screen_cracked"})  # would be wrong here

    result = normalize_query(
        "My Galaxy S22 battery drains extremely fast, dead by noon even with light use.",
        llm_client=WrongAnswerLLM(),
    )
    assert result.symptom_category == "battery_drain"
    assert result.classification_source == "keyword_match"


def test_llm_fallback_rejects_a_category_outside_the_known_taxonomy():
    """The real guardrail: even if the LLM ignores its instructions and
    invents a category that was never in the list, the whitelist check
    discards it and the safe unclassified default is kept — no hallucinated
    new category can ever reach symptom_category.
    """
    from enrichment import normalize_query
    from llm_client import LLMClient
    import json

    class HallucinatingLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return json.dumps({"category": "haunted_device_possession"})

    result = normalize_query(
        "My phone does something weird sometimes, hard to describe.",
        llm_client=HallucinatingLLM(),
    )
    assert result.symptom_category == "unclassified_issue"
    assert result.classification_source == "unclassified"


def test_llm_fallback_malformed_response_does_not_crash():
    from enrichment import normalize_query
    from llm_client import LLMClient

    class GarbageLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return "not even json"

    result = normalize_query("My phone does something weird sometimes.", llm_client=GarbageLLM())
    assert result.symptom_category == "unclassified_issue"


def test_llm_variation_path_does_not_crash_on_symptom_field_rename():
    """Regression: an earlier refactor renamed Symptom.keywords to
    component_terms/problem_terms but left one reference to the old
    attribute in the LLM-paraphrase validation path. Tests didn't catch it
    because that path only executes with a real (non-mock) LLM client AND a
    result that isn't already a confident keyword_match (see
    test_confident_keyword_match_skips_the_llm_paraphrase_call for why that
    gate exists) — using a complaint the keyword matcher can't classify, so
    the LLM-paraphrase validation branch under test actually runs.
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
        "My phone messages come out full of random symbols instead of the letters I actually pressed."
    )
    assert result.classification_source != "keyword_match"  # sanity: LLM branch will actually run
    variations = generate_variations(result, llm_client=FakeLLM())
    assert MIN_VARIATIONS <= len(variations) <= MAX_VARIATIONS


@pytest.mark.parametrize("raw,expected_category", [
    ("My Galaxy S23 fingerprint doesnt recognize me anymore.", "fingerprint_face_unlock_fail"),
    ("The software update on my Galaxy S24 doesnt install no matter what I try.", "software_update_fails"),
    ("Im not getting any WhatsApp notifications, they arent showing up at all on my Galaxy S22.", "no_notifications"),
    ("My Galaxy S22 screen doesnt display anything anymore, just stays dark.", "screen_blank_black"),
    ("My Galaxy S22 battery doesnt last more than a couple hours now.", "battery_drain"),
])
def test_apostrophe_dropped_negation_matches_same_as_apostrophe_version(raw, expected_category):
    """Real users often type 'doesnt'/'arent' without the apostrophe, especially
    in the casual/typo registers Stage 0 itself generates. The taxonomy already
    handled this for won't/wont and can't/cant everywhere; doesn't/aren't had
    gaps in a few categories where only the apostrophe'd form was listed, so a
    perfectly common casual phrasing fell through to unclassified_issue for no
    good reason. This isn't the "heavy typo" limitation below — it's a single,
    universally-dropped punctuation mark on an otherwise clean sentence.
    """
    result = enrich(raw)
    assert result.symptom_category == expected_category


def test_sm_code_device_casing_is_fully_uppercase_even_from_lowercase_input():
    """SM-xxxx model codes are conventionally all-caps. A keyword-style
    complaint typed in lowercase ('my sm-a536e battery...') used to come out
    as 'Sm-a536e' because the generic title-casing only ever touches the
    first letter of each word.
    """
    result = enrich("my sm-a536e battery drains fast, dead by noon even with light use")
    assert result.device == "SM-A536E"


def test_llm_path_preserves_typo_and_keyword_registers_even_when_llm_contributes_nothing():
    """Regression: generate_variations() used to slice the template list as
    base[:6] when a real (non-mock) LLM client was configured, which drops
    BOTH the frustrated-register pair AND the two typo-register variants —
    not just frustrated, as the code's own comment claimed. If the LLM call
    fails, times out, or every candidate gets filtered out by the relevance
    check (extra == []), a real deployment would silently lose the typo
    register the module docstring promises is always present. Exercising the
    zero-usable-output case directly here, with a fake LLM client, so this
    can't silently regress the moment someone actually sets LLM_PROVIDER.
    """
    from enrichment import SYMPTOM_TAXONOMY, _DEFAULT_SYMPTOM, _inject_typos, generate_variations, normalize_query
    from llm_client import LLMClient
    import json

    class FailingLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return json.dumps([])  # a well-formed but empty response

    result = normalize_query("My Galaxy S22 battery drains extremely fast, dead by noon even with light use.")
    variations = generate_variations(result, llm_client=FailingLLM())

    # Recompute the exact two typo variants the template generator would have
    # produced, and confirm at least one survived the LLM-path slicing.
    symptom = next((s for s in SYMPTOM_TAXONOMY if s.category == result.symptom_category), _DEFAULT_SYMPTOM)
    expected_typo_casual = _inject_typos(f"{result.device} {symptom.subject} {symptom.casual}", seed=1)
    expected_typo_formal = _inject_typos(f"{result.device} {symptom.subject} {symptom.formal}", seed=2)

    assert expected_typo_casual in variations or expected_typo_formal in variations
    assert MIN_VARIATIONS <= len(variations) <= MAX_VARIATIONS


def test_llm_fallback_classifies_correctly_when_response_is_wrapped_in_markdown_code_fence():
    """Real bug found live against the actual Gemini API (not a hypothetical):
    despite the system prompt saying "Respond with strict JSON and nothing
    else", Gemini's raw response came back as the literal string
    '```json\\n{"category": "keyboard_typing_problem"}\\n```' — correct
    answer, wrong wrapping. json.loads() rejects that outright (it starts
    with a backtick, not '{'), which _llm_classify_symptom's bare
    `except Exception: pass` silently swallowed, so a working LLM call that
    answered correctly still produced classification_source="unclassified".
    Reproducing the exact wrapping here (not just "malformed json") so this
    specific, real failure mode can't silently regress.
    """
    from enrichment import LLM_FALLBACK_CONFIDENCE, normalize_query
    from llm_client import LLMClient

    class MarkdownFencedLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return '```json\n{"category": "keyboard_typing_problem"}\n```'

    # phrased to still miss the keyword taxonomy (no "keyboard" or "when I
    # type"-style phrase) so this exercises the LLM path, not the now-direct
    # keyword match added in test_common_typing_phrasing_is_now_caught_deterministically.
    result = normalize_query(
        "My phone messages come out full of random symbols instead of the letters I actually pressed.",
        llm_client=MarkdownFencedLLM(),
    )
    assert result.symptom_category == "keyboard_typing_problem"
    assert result.classification_source == "llm_fallback"
    assert result.symptom_confidence == LLM_FALLBACK_CONFIDENCE


def test_common_typing_phrasing_is_now_caught_deterministically():
    """A real complaint tested live against Gemini ('my phone does something
    weird when I type, letters come out wrong') is perfectly understandable to
    a person but never says "keyboard" -- it used to fall all the way through
    to the guarded LLM fallback for want of that one component word, correct
    but slower and dependent on a configured provider. "when I type"/"while
    typing"-style phrases are common enough (and specific enough not to
    collide with anything else) to classify directly and instantly, with zero
    network dependency, no LLM_PROVIDER required at all.
    """
    result = enrich("My phone does something weird when I type, letters come out wrong.")
    assert result.symptom_category == "keyboard_typing_problem"
    assert result.classification_source == "keyword_match"


def test_llm_variations_parse_correctly_when_response_is_wrapped_in_markdown_code_fence():
    """Same real bug as the classifier test above, but in _llm_variations:
    json.loads(raw) on a markdown-fenced array would raise and the function
    would silently return [] (indistinguishable from "LLM contributed
    nothing"), even though the LLM actually gave usable paraphrases.
    """
    from enrichment import generate_variations, normalize_query
    from llm_client import LLMClient

    class MarkdownFencedLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            return '```json\n["my techcorp device fails to register keystrokes right", "techcorp keyboard is terrible now"]\n```'

    # a keyword_match result would skip the LLM call entirely (see
    # test_confident_keyword_match_skips_the_llm_paraphrase_call) so this
    # needs a complaint that stays unclassified at the keyword layer for the
    # LLM-paraphrase branch under test to actually run.
    result = normalize_query(
        "My phone messages come out full of random symbols instead of the letters I actually pressed."
    )
    variations = generate_variations(result, llm_client=MarkdownFencedLLM())
    assert any("terrible" in v.lower() for v in variations)
    assert MIN_VARIATIONS <= len(variations) <= MAX_VARIATIONS


def test_confident_keyword_match_skips_the_llm_paraphrase_call():
    """Real latency finding from live testing: with a real (non-mock)
    LLM_PROVIDER configured, generate_variations() used to call out to the
    LLM for extra paraphrases on every request, even one the keyword matcher
    already classified with full confidence — adding a real network
    round-trip (~1s, observed live against Gemini) before the cache is even
    checked, for a query that already had 8-10 solid template variations.
    Since a keyword-matched result is already confident, the LLM call buys
    little and costs real latency; skip it in that case only — llm_fallback
    and unclassified results (which are already paying an LLM cost for
    classification, or could genuinely use more paraphrase diversity) still
    get it.
    """
    from enrichment import generate_variations, normalize_query
    from llm_client import LLMClient

    class ExplodingLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            raise AssertionError("LLM should not be called for an already-confident keyword_match result")

    result = normalize_query("My Galaxy S22 battery drains extremely fast, dead by noon even with light use.")
    assert result.classification_source == "keyword_match"  # sanity
    variations = generate_variations(result, llm_client=ExplodingLLM())  # must not raise
    assert MIN_VARIATIONS <= len(variations) <= MAX_VARIATIONS


def test_strip_code_fence_is_a_no_op_on_already_plain_json():
    """The fence-stripping helper must never corrupt a normal, unwrapped JSON
    response — most providers (and Gemini itself, sometimes) return plain
    JSON with no fence at all.
    """
    from enrichment import _strip_code_fence

    plain = '{"category": "battery_drain"}'
    assert _strip_code_fence(plain) == plain


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


# -- regression tests for the taxonomy scoring/tie-breaking bug scripts/benchmark_cache.py's
# self-classification check surfaced (metrics.md §3): 12 of 35 categories didn't
# keyword-classify their own formal/casual template text back to themselves --


def test_every_taxonomy_category_self_classifies_its_own_template_text():
    """The exact check scripts/benchmark_cache.py's controlled register experiment
    runs, promoted to a real regression test instead of a benchmark side-note.
    Two independent bugs were behind the original 12 failures: (1) extract_symptom()
    summed every matching keyword-list entry instead of capping each side to
    boolean presence, so a category with a long, redundant list of near-duplicate
    phrasings (e.g. screen_blank_black's 19 "blank"/"black" variants) could
    outscore a more specific category with a short, precise list purely by having
    more synonyms written down for the same evidence -- not stronger evidence; and
    (2) several categories' own formal/casual template wording simply didn't share
    any vocabulary with that category's own problem_terms (e.g. overheating's
    formal text says "excessively hot", but the keyword list only had "too hot"/
    "extremely hot"/etc.). Every category must at least recognize its own
    description now.
    """
    from enrichment import SYMPTOM_TAXONOMY, extract_symptom

    device = "Galaxy S23"
    failures = []
    for symptom in SYMPTOM_TAXONOMY:
        formal_text = f"My {device} {symptom.subject} {symptom.formal}."
        casual_text = f"my {device} {symptom.subject} {symptom.casual}"
        for label, text in [("formal", formal_text), ("casual", casual_text)]:
            got, _ = extract_symptom(text)
            if got.category != symptom.category:
                failures.append((symptom.category, label, got.category))
    assert not failures, f"self-classification failures: {failures}"


def test_screen_ghost_touch_does_not_collide_with_spontaneous_screen_failure():
    """Regression: extract_symptom()'s old summed-hit scoring let screen_ghost_touch's
    generic "on its own"/"by itself" problem terms fire on ANY spontaneous-failure
    complaint, not just touch behavior -- "screen suddenly went completely black on
    its own" (a real unseen_scenarios.txt line) misclassified as ghost-touch instead
    of a black screen, purely because "screen" + "on its own" happened to also be a
    ghost_touch match. Fixed by making ghost_touch's problem terms touch-specific
    ("touch input on its own", "touches on its own", ...) instead of bare "on its
    own"/"by itself", so it can only fire alongside real touch behavior.
    """
    result = enrich(
        "My Samsung Galaxy A15/A16 screen suddenly went completely black on its own "
        "after about a month of use. It doesn't display anything, even when I try to turn it on."
    )
    assert result.symptom_category == "screen_blank_black"
    assert result.symptom_category != "screen_ghost_touch"


def test_foldable_inner_screen_failure_not_misclassified_as_generic_touch_unresponsive():
    """Regression: a foldable's inner-screen failure ("stopped working by itself...
    doesn't respond to touch... cover screen still works") shares "screen" +
    "doesn't respond" with the generic touch_unresponsive category, which is listed
    earlier in the taxonomy. Under the old summed-hit scoring, inner_screen_failure
    won anyway by matching more of its own (redundant) problem terms; under the
    boolean-capped scoring both tie at 2, so inner_screen_failure -- correctly the
    more specific category for a complaint that names "inner screen"/"cover
    screen" -- had to be reordered ahead of touch_unresponsive to keep winning
    ties, consistent with the taxonomy's existing "more specific categories listed
    first" design principle.
    """
    result = enrich(
        "My Galaxy Flip 7 inner screen stopped working by itself; it shows no image "
        "and doesn't respond to touch, while the outer cover screen still works."
    )
    assert result.symptom_category == "inner_screen_failure"
    assert result.symptom_category != "touch_unresponsive"


def test_half_screen_dark_not_misclassified_as_generic_black_screen():
    """Regression: "half is dead, other half's fine" shares "screen" + "is dead"
    with the generic screen_blank_black category (listed earlier in the original
    taxonomy order). half_screen_dark needed its own "half is dead" problem term
    plus reordering ahead of screen_blank_black to correctly win this as the more
    specific half-display symptom.
    """
    result = enrich("My Galaxy S22 screen half is dead, other half's fine.")
    assert result.symptom_category == "half_screen_dark"
    assert result.symptom_category != "screen_blank_black"

def test_handles_extremely_vague_user_prompts():
    """
    Real users often submit incredibly vague or poorly worded prompts with zero
    diagnostic value (e.g., 'help me', 'it is broken').
    The enrichment pipeline must safely catch these, mark them unclassified,
    and flag them as low confidence to prevent cache poisoning or hallucinations.
    """
    from src.enrichment import normalize_query

    vague_prompts = [
        "help me",
        "it doesn't work",
        "my phone is broken",
        "TechCorp thing is bad",
        "fix it please",
        "hello?",
        "why is it doing this",
        "nothing works",
        "i hate this phone",
        "aaaaaaarrrgghh",
        "please respond",
        "do something"
    ]

    for prompt in vague_prompts:
        result = normalize_query(prompt)
        assert result.symptom_category == "unclassified_issue", f"Failed on: {prompt}"
        assert result.is_low_confidence is True, f"Failed on: {prompt}"
        assert "could not be automatically classified" in result.canonical_query.lower()
