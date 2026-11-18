"""answer/quote_snap.py (safety-critical): what is snapped, and what never is."""

from __future__ import annotations

import pytest

from nyaya.answer.quote_snap import Snap, best_window, snap_quote, word_key
from nyaya.text.normalize import normalize_text

# Shapes taken from the answers run of 2026-11-17 (contexts of dev-0002, dev-0006,
# dev-0019, dev-0033): a footnote inside a sentence, a split ligature, a
# hyphen followed by a line break, a middle dot left by the PDF.
TEXT = (
    "The arbitral tribunal is the\npreferred first authority to determine and decide "
    "all questions of non-arbitrability.\nAs an exception to the rule, and rarely as a "
    "demurrer,\nthe referral court may reject claims which are manifestly and ex-\n"
    "19 (2021) 5 SCC 671, paras 29, 30.\n"
    "The order makes clear that respondent No.2- society could approach the Competent "
    "Authority afresh only after getting the complications sorted out before this\u00b7 "
    "Court. The petition was \ufb01 led within time and the High Court rightly held that "
    "the dismissal was not lawful under Section 4(1) of the Act."
)


def snap(quote: str, text: str = TEXT, **kw) -> Snap | None:
    kw.setdefault("min_similarity", 0.9)
    kw.setdefault("max_edits", 6)
    return snap_quote(quote, text, **kw)


def assert_exact(s: Snap | None, text: str = TEXT) -> Snap:
    assert s is not None
    assert s.span in text  # the raw span, not just its normal form
    assert normalize_text(s.span) in normalize_text(text)
    return s


def test_word_key():
    assert word_key("\u201cNon\u2011Arbitrability,\u201d") == "nonarbitrability"
    assert word_key("\u00b7") == ""


def test_spacing_tier_ligature_and_middle_dot():
    s = assert_exact(
        snap(
            "sorted out before this Court. The petition was filed within time and the "
            "High Court rightly held"
        )
    )
    assert s.method == "spacing" and s.similarity == 1.0
    assert "\ufb01 led" in s.span and "this\u00b7 Court" in s.span


def test_spacing_tier_hyphen_space_and_case():
    s = assert_exact(snap("respondent No.2-Society could approach the Competent Authority afresh"))
    assert s.method == "spacing"
    assert s.span.startswith("respondent No.2- society")


def test_spacing_tier_needs_whole_tokens():
    # "rbitral ..." starts inside "arbitral": no spacing-tier match. The fuzzy
    # tier may drop the stray word, but a span never starts inside a word.
    s = snap("rbitral tribunal is the preferred first authority")
    assert s is not None and s.method == "fuzzy"
    assert s.span.startswith("tribunal")


def test_fuzzy_tier_drops_llm_framing_words():
    s = assert_exact(
        snap(
            "In short, the order makes clear that respondent No.2-Society could "
            "approach the Competent Authority afresh only after getting the complications "
            "sorted out before this Court."
        )
    )
    assert s.method == "fuzzy" and 0.9 <= s.similarity < 1.0
    assert s.span.startswith("The order makes clear")


def test_fuzzy_tier_stops_at_a_footnote_inside_the_sentence():
    s = assert_exact(
        snap(
            "As an exception to the rule, and rarely as a demurrer, the referral court may "
            "reject claims which are manifestly and ex-facie non-arbitrable."
        )
    )
    assert s.span.endswith("manifestly and ex")
    assert "SCC" not in s.span


def test_never_snaps_an_ellipsis():
    assert snap("The arbitral tribunal is the preferred ... authority to determine") is None
    assert snap("The arbitral tribunal is the preferred \u2026 authority to determine") is None


def test_never_snaps_a_changed_number():
    assert snap("held that the dismissal was not lawful under Section 4(2) of the Act.") is None
    # spacing tier: "41" is not "4( 1 )"
    assert snap("the dismissal was not lawful under Section 41 of the Act.") is None


def test_never_snaps_a_dropped_or_added_negation():
    assert (
        snap("the High Court rightly held that the dismissal was lawful under Section 4(1)") is None
    )
    text = "The court held that the accused was entitled to bail on the facts of the case."
    assert (
        snap("The court held that the accused was not entitled to bail on the facts", text) is None
    )


def test_never_snaps_an_affix_change():
    text = "the High Court rightly held that the dismissal was lawful and proper in every respect"
    assert (
        snap("the High Court rightly held that the dismissal was unlawful and proper", text) is None
    )


def test_edit_budget():
    words = [f"w{chr(97 + n // 26)}{chr(97 + n % 26)}" for n in range(60)]
    text = " ".join(words)
    quote_words = words[10:50]
    for i in (3, 9, 15, 21, 27, 33, 39):  # 7 replaced words out of 40
        quote_words[i] = "zz" + "q" * i
    quote = " ".join(quote_words)
    assert snap(quote, text, min_similarity=0.8, max_edits=6) is None
    s = assert_exact(snap(quote, text, min_similarity=0.8, max_edits=7), text)
    assert s.method == "fuzzy"


def test_paraphrase_is_not_snapped():
    para = (
        "A referral court may only rarely reject claims that are plainly not arbitrable, "
        "leaving the tribunal to decide first."
    )
    assert snap(para) is None
    ratio, _ = best_window(para, TEXT)
    assert ratio < 0.9  # it is the similarity bar that refuses it


def test_empty_inputs():
    assert snap("") is None
    assert snap("... \u2014 ...") is None
    assert snap("some words of a quote here", "") is None
    assert snap("- \u00b7 -", TEXT) is None


def test_best_window():
    ratio, span = best_window("the referral court may reject claims which are manifestly", TEXT)
    assert ratio == 1.0 and span in TEXT
    assert best_window("", TEXT) == (0.0, "")
    assert best_window("anything at all", "") == (0.0, "")
    ratio, _ = best_window("completely unrelated words about elephants", TEXT)
    assert 0.0 <= ratio < 0.5


def test_to_dict_rounds_similarity():
    s = Snap(span="x", similarity=0.912345, method="fuzzy")
    assert s.to_dict() == {"span": "x", "similarity": 0.9123, "method": "fuzzy"}


@pytest.mark.parametrize("threshold", [0.9, 0.95])
def test_unique_best_window_is_chosen(threshold):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
    s = assert_exact(
        snap(
            "beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi",
            text,
            min_similarity=threshold,
        ),
        text,
    )
    assert s.span == "beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
