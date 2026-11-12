import pytest

from judgment_fixtures import ERA_C, ERA_C_TWO_OPINIONS, ERA_D, load, parsed_from_text
from nyaya.config import load_settings
from nyaya.contracts import Era, OpinionType
from nyaya.zone.opinions import bench_names, names_match, split_opinions
from nyaya.zone.zoner import zone


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def run(doc, settings, judges=None):
    z = zone(doc, settings)
    return z, split_opinions(doc, z, judges or [], settings)


def test_era_c_single_opinion(settings):
    doc = load(ERA_C)
    z, ops = run(doc, settings, ["UDAY UMESH LALIT"])
    assert len(ops) == 1
    op = ops[0]
    assert op.author == "DEEPAK GUPTA, J."
    assert op.type is OpinionType.MAJORITY
    assert doc.clean_text[op.char_start :].startswith("DEEPAK GUPTA, J. 1. Applications")
    assert op.char_end == z.judgment_span[1]


def test_era_c_two_opinions_no_phrase_is_unknown(settings):
    doc = load(ERA_C_TWO_OPINIONS)
    _, ops = run(doc, settings, ["RANJAN GOGOI"])
    assert [o.author for o in ops] == ["UDAY UMESH LALIT, J.", "PRAFULLA C. PANT, J."]
    # "I beg to differ" is not in opinion.phrases, and majority is never positional
    assert [o.type for o in ops] == [OpinionType.UNKNOWN, OpinionType.UNKNOWN]
    assert ops[0].char_end == ops[1].char_start
    assert doc.clean_text[ops[1].char_start :].startswith("PRAFULLA C. PANT, J. 1. I have")


def test_era_d_single_opinion(settings):
    doc = load(ERA_D)
    _, ops = run(doc, settings, ["ABHAY S. OKA"])
    assert [(o.author, o.type) for o in ops] == [("Abhay S. Oka, J.", OpinionType.MAJORITY)]
    assert doc.clean_text[ops[0].char_start :].startswith("Abhay S. Oka, J.\nFACTUAL DETAILS")


ERA_C_DISSENT_FIRST = """A v. B
(Civil Appeal No. 1 of 2015)
[P. SMITH, Q. JONES AND R. KHAN, JJ.]
Penal Code, 1860 – s.302.
HELD: 1. Point. [Para 3]
CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1 of 2015.
Counsel for the parties.
P. SMITH, J. 1. I regret I am unable to agree with the opinion of my learned brother.
2. The appeal should be dismissed.
Q. JONES, J.
1. The appeal is allowed for the reasons below.
2. More reasons.
R. KHAN, J. 1. I agree with my learned brother Jones, J.
"""


def test_dissent_printed_first_is_not_majority(settings):
    doc = parsed_from_text("2015_1_1_9", Era.C, ERA_C_DISSENT_FIRST)
    _, ops = run(doc, settings)
    assert [o.author for o in ops] == ["P. SMITH, J.", "Q. JONES, J.", "R. KHAN, J."]
    assert [o.type for o in ops] == [
        OpinionType.DISSENTING,
        OpinionType.UNKNOWN,
        OpinionType.CONCURRING,
    ]


def test_era_d_star_names_main_opinion(settings):
    text = (
        "A v. B\n[P. Smith and Q. Jones,* JJ.]\nIssue for Consideration\nIssue.\n"
        "Judgment / Order of the Supreme Court\nJudgment\n"
        "P. Smith, J.\n1. I dissent from the view taken.\n"
        "Q. Jones, J.\n1. The appeal is allowed.\n"
    )
    _, ops = run(parsed_from_text("2025_1_1_9", Era.D, text), settings)
    assert [(o.author, o.type) for o in ops] == [
        ("P. Smith, J.", OpinionType.DISSENTING),
        ("Q. Jones, J.", OpinionType.MAJORITY),
    ]


def test_name_not_on_bench_does_not_start_opinion(settings):
    text = (
        "A v. B\n[P. SMITH AND Q. JONES, JJ.]\nHELD: 1. x. [Para 1]\n"
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1.\n"
        "P. SMITH, J. 1. Text of the judgment.\n"
        "Z. NOBODY, J.\n2. Still the same opinion.\n"
    )
    _, ops = run(parsed_from_text("2015_1_1_9", Era.C, text), settings)
    assert [o.author for o in ops] == ["P. SMITH, J."]


def test_judges_metadata_used_when_bench_missing(settings):
    text = "No bench line\nCIVIL APPELLATE JURISDICTION: x\nJ. CHELAMESWAR, J. 1. Text.\n"
    _, ops = run(parsed_from_text("2012_1_1_9", Era.C, text), settings, ["JASTI CHELAMESWAR"])
    assert [o.author for o in ops] == ["J. CHELAMESWAR, J."]


def test_any_name_accepted_when_no_names_known(settings):
    text = "No bench line\nCIVIL APPELLATE JURISDICTION: x\nS. A. BOBDE, CJI\n1. Text.\n"
    _, ops = run(parsed_from_text("2020_1_1_9", Era.C, text), settings)
    assert [o.author for o in ops] == ["S. A. BOBDE, CJI"]


def test_no_opinion_start_gives_one_opinion(settings):
    text = (
        "A v. B\n[P. SMITH AND Q. JONES, JJ.]\nHELD: 1. x. [Para 1]\n"
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1.\n"
        "ORDER\n1. Leave granted.\n2. Appeal dismissed.\n"
    )
    doc = parsed_from_text("2015_1_1_9", Era.C, text)
    z, ops = run(doc, settings)
    assert len(ops) == 1
    assert ops[0].author is None
    assert ops[0].type is OpinionType.MAJORITY
    assert (ops[0].char_start, ops[0].char_end) == z.judgment_span


def test_no_opinion_start_era_d_uses_starred_judge(settings):
    text = (
        "A v. B\n[P. Smith* and Q. Jones, JJ.]\nIssue for Consideration\nIssue.\n"
        "Judgment / Order of the Supreme Court\n1. Text.\n"
    )
    _, ops = run(parsed_from_text("2025_1_1_9", Era.D, text), settings)
    assert [(o.author, o.type) for o in ops] == [("P. Smith", OpinionType.MAJORITY)]


def test_preamble_with_numbered_paragraphs_is_its_own_opinion(settings):
    text = (
        "A v. B\n[P. SMITH AND Q. JONES, JJ.]\nHELD: 1. x. [Para 1]\n"
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1.\n"
        "1. Order of the court in brief.\n"
        "Q. JONES, J.\n1. Reasons.\n"
    )
    _, ops = run(parsed_from_text("2015_1_1_9", Era.C, text), settings)
    assert [(o.author, o.type) for o in ops] == [
        (None, OpinionType.UNKNOWN),
        ("Q. JONES, J.", OpinionType.UNKNOWN),
    ]


def test_single_opinion_with_dissent_word_is_still_majority(settings):
    text = (
        "A v. B\n[P. SMITH AND Q. JONES, JJ.]\nHELD: 1. x. [Para 1]\n"
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1.\n"
        "P. SMITH, J. 1. The dissent in the High Court was right.\n"
    )
    _, ops = run(parsed_from_text("2015_1_1_9", Era.C, text), settings)
    assert [o.type for o in ops] == [OpinionType.MAJORITY]


@pytest.mark.parametrize(
    ("bench", "expected", "star"),
    [
        ("[UDAY UMESH LALIT AND DEEPAK GUPTA, JJ.]", ["UDAY UMESH LALIT", "DEEPAK GUPTA"], None),
        (
            "[S. A. BOBDE, CJI, A. S. BOPANNA AND\nV. RAMASUBRAMANIAN, JJ.]",
            ["S A BOBDE", "A S BOPANNA", "V RAMASUBRAMANIAN"],
            None,
        ),
        ("[Abhay S Oka* and Ujjal Bhuyan, JJ.]", ["ABHAY S OKA", "UJJAL BHUYAN"], "ABHAY S OKA"),
        (
            "[J.B. Pardiwala and R. Mahadevan,* JJ.]",
            ["J B PARDIWALA", "R MAHADEVAN"],
            "R MAHADEVAN",
        ),
        ("[P. SATHASIVAM, CJI AND RANJAN GOGOi, J.]", ["P SATHASIVAM", "RANJAN GOGOI"], None),
    ],
)
def test_bench_names(bench, expected, star):
    names, starred = bench_names(f"Title\n{bench}\n")
    assert names == expected
    assert starred == star


def test_bench_names_missing():
    assert bench_names("no bench here") == ([], None)


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("D.Y. CHANDRACHUD, J.", "DR. D.Y. CHANDRACHUD", True),
        ("J. CHELAMESWAR", "JASTI CHELAMESWAR", True),
        ("CHELAMESWAR", "JASTI CHELAMESWAR", True),
        ("MARKANDEY KAT JU", "MARKANDEY KATJU", True),
        ("Abhay S. Oka", "ABHAY S OKA", True),
        ("HEMANT GUPTA", "DEEPAK GUPTA", False),
        ("A. K. SIKRI", "ASHOK BHUSHAN", False),
        ("R. F. NARIMAN", "R. BANUMATHI", False),
        ("", "R. BANUMATHI", False),
    ],
)
def test_names_match(a, b, same):
    assert names_match(a, b) is same
