"""Era C catchword Acts (author decision E1) and Act name normalising."""

import pytest

from nyaya.config import load_settings
from nyaya.zone.zoner import (
    act_refs,
    canonical_act,
    catchword_groups,
    extract_era_c_acts,
    load_act_aliases,
    unmapped_act_names,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


@pytest.fixture(scope="module")
def aliases(settings):
    return load_act_aliases(settings)


def acts(text, settings):
    return extract_era_c_acts(text, settings)


def test_en_dash_separated(settings):
    text = (
        "Motor Vehicles Act, 1988 – s.166 – Compensation – Just compensation – "
        "Assessment of – Held: ..."
    )
    assert acts(text, settings) == ["Motor Vehicles Act, 1988"]


def test_wrapped_act_name(settings):
    text = (
        "The Uttar Pradesh Krishi Utpadan Mandi\nAdhiniyam, 1964: s.17 – Market fee – "
        "Levy of – Validity."
    )
    assert acts(text, settings) == ["Uttar Pradesh Krishi Utpadan Mandi Adhiniyam, 1964"]


def test_two_acts_in_one_group(settings):
    text = "Trademarks Act, 1999 – s.89A – Finance Act, 2017 – s.184 – Tribunal – Validity"
    assert acts(text, settings) == ["Trademarks Act, 1999", "Finance Act, 2017"]


def test_penal_code_semicolon_ocr(settings):
    assert acts("Penal Code; 1860 – s.302 – Murder.", settings) == ["Penal Code, 1860"]


def test_upper_case_act(settings):
    assert acts("BOMBAY POLICE ACT, 1951: s.56 – Externment.", settings) == [
        "BOMBAY POLICE ACT, 1951"
    ]


def test_subject_heading_is_not_an_act(settings):
    assert acts("Service Law: Promotion – Seniority – Held: ...", settings) == []
    assert acts("Contempt of Court: Wilful disobedience.", settings) == []
    assert acts("Child and Family Welfare: Custody.", settings) == []
    assert acts("PARTITION: Family settlement – Held: Not a transfer.", settings) == []


def test_constitution_allowlist(settings):
    assert acts("Constitution of India – Art. 14 – Equality.", settings) == [
        "Constitution of India"
    ]
    assert acts("Constitution of India, 1950: Art. 226 – Writ jurisdiction.", settings) == [
        "Constitution of India"
    ]
    assert acts("Constitution – Arts. 14 and 21 – Held: ...", settings) == ["Constitution of India"]


def test_constitution_bare_word_without_art_is_not_matched(settings):
    assert acts("Constitution bench – reference – Held: ...", settings) == []


def test_arbitration_with_colon(settings):
    text = "Arbitration and Conciliation Act, 1996: s.34 – Setting aside award."
    assert acts(text, settings) == ["Arbitration and Conciliation Act, 1996"]


def test_rules_with_brackets(settings):
    text = "Maharashtra Civil Services (Discipline and Appeal) Rules, 1979 – r.8 – Inquiry."
    assert acts(text, settings) == [
        "Maharashtra Civil Services (Discipline and Appeal) Rules, 1979"
    ]


def test_code_of_procedure_names(settings):
    text = (
        "Code of Criminal Procedure, 1973 – ss.340 and 195 – Penal Code, 1860 – "
        "s.193 r/w. s.34 – False Evidence."
    )
    assert acts(text, settings) == ["Code of Criminal Procedure, 1973", "Penal Code, 1860"]
    assert acts("Code of Civil Procedure, 1908: s.11 – Res judicata.", settings) == [
        "Code of Civil Procedure, 1908"
    ]


def test_every_act_across_groups(settings):
    # From 2016_8_36_72 and 2021_9_866_881 (real catchwords, lines joined)
    text = (
        "Penal Code, 1860: s.302 – Conviction – Held: acquittal – Evidence Act, 1872 s.6. "
        "Evidence Act, 1872: s. 6 – Rationale behind – Penal Code, 1860 s.302. "
        "Penal Code, 1860 – ss.392, 397 – convicted u/ss.392, 397 r/w s.11/13, "
        "MPDVPK Act, 1981 – On appeal."
    )
    assert acts(text, settings) == [
        "Penal Code, 1860",
        "Evidence Act, 1872",
        "MPDVPK Act, 1981",
    ]


def test_margin_letters_and_hyphens(settings):
    # scanned years: lone margin letters inside the text, ASCII hyphens as dashes
    text = (
        "Customs Valuation Rules, 1988: rr.4, 5, 6, 10A c Import of goods - Rejected\n"
        "D minimum of the price - set aside a/together- Custom Act, 1962 - s. 112."
    )
    assert acts(text, settings) == ["Customs Valuation Rules, 1988", "Custom Act, 1962"]


def test_leading_lowercase_words_trimmed(settings):
    text = "Held: the validity of the Industrial Disputes Act, 1947 – s.25F – retrenchment."
    assert acts(text, settings) == ["Industrial Disputes Act, 1947"]


def test_word_inside_another_word_is_not_a_keyword(settings):
    assert acts("Breach of the contract, 1990 terms – damages.", settings) == []
    assert acts("Their impact, 2001 onwards – assessed.", settings) == []


def test_groups_split_at_full_stop_before_capital_unless_abbreviation(settings):
    text = (
        "Gift Tax Act, 1958: s. Exemption under No. 5 – held. Service Law: Seniority. "
        "Bombay Rent Act, 1947 – r. Tenancy."
    )
    groups = catchword_groups(text, settings)
    assert groups == [
        "Gift Tax Act, 1958: s. Exemption under No. 5 – held.",
        "Service Law: Seniority.",
        "Bombay Rent Act, 1947 – r. Tenancy.",
    ]
    assert acts(text, settings) == ["Gift Tax Act, 1958", "Bombay Rent Act, 1947"]


def test_unmatched_groups_reported(settings):
    unmatched = []
    text = "Service Law: Seniority. Penal Code, 1860 – s.302."
    assert extract_era_c_acts(text, settings, unmatched=unmatched) == ["Penal Code, 1860"]
    assert unmatched == ["Service Law: Seniority."]


def test_duplicate_acts_reported_once(settings):
    text = "Penal Code, 1860 – s.302. Penal Code, 1860 – s.34. Indian Penal Code, 1860 – s.120B."
    # "Indian Penal Code, 1860" is a different raw name, same canonical name
    assert acts(text, settings) == ["Penal Code, 1860"]


def test_empty(settings):
    assert acts("", settings) == []


# --- normalising ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Penal Code, 1860", "penal code, 1860"),
        ("Indian Penal Code, 1860", "penal code, 1860"),
        ("IPC", "penal code, 1860"),
        ("The Indian Penal Code; 1860", "penal code, 1860"),
        ("Code of Criminal Procedure, 1973", "code of criminal procedure, 1973"),
        ("CrPC", "code of criminal procedure, 1973"),
        ("Cr.P.C.", "code of criminal procedure, 1973"),
        ("CPC", "code of civil procedure, 1908"),
        ("Indian Evidence Act, 1872", "evidence act, 1872"),
        ("Constitution of India", "constitution of india"),
        ("NDPS Act", "narcotic drugs and psychotropic substances act, 1985"),
        ("Arbitration & Conciliation Act, 1996", "arbitration and conciliation act, 1996"),
        ("Income-tax Act, 1961", "income tax act, 1961"),
        ("BNS", "bharatiya nyaya sanhita, 2023"),
        ("BNSS", "bharatiya nagarik suraksha sanhita, 2023"),
        ("Bharatiya Sakshya Adhiniyam, 2023", "bharatiya sakshya adhiniyam, 2023"),
        ("BOMBAY POLICE ACT, 1951", "bombay police act, 1951"),  # unmapped: normalised form
        (
            "Maharashtra Ownership Flats (Regulation of the Promotion of Construction, Sale, "
            "Management and Transfer Act) 1963",
            "maharashtra ownership flats (regulation of the promotion of construction, sale, "
            "management and transfer act), 1963",
        ),
        ("  Motor   Vehicles Act ,1988 ", "motor vehicles act, 1988"),
    ],
)
def test_canonical_act(name, expected, aliases):
    assert canonical_act(name, aliases) == expected


def test_canonical_act_default_aliases():
    assert canonical_act("IPC") == "penal code, 1860"


def test_alias_file_covers_required_acts(aliases):
    canon = set(aliases.values())
    for name in [
        "penal code, 1860",
        "code of criminal procedure, 1973",
        "code of civil procedure, 1908",
        "evidence act, 1872",
        "constitution of india",
        "narcotic drugs and psychotropic substances act, 1985",
        "arbitration and conciliation act, 1996",
        "income tax act, 1961",
        "bharatiya nyaya sanhita, 2023",
        "bharatiya nagarik suraksha sanhita, 2023",
        "bharatiya sakshya adhiniyam, 2023",
    ]:
        assert name in canon


def test_alias_file_variants_unique(settings):
    import tomllib

    from nyaya.config import REPO_ROOT

    raw = tomllib.loads((REPO_ROOT / settings.acts.alias_file).read_text(encoding="utf-8"))
    seen = {}
    for canon, variants in raw["aliases"].items():
        for v in variants:
            key = canonical_act(v, {})
            assert key not in seen or seen[key] == canon, f"{v!r} maps to two names"
            seen[key] = canon


def test_act_refs_and_unmapped(aliases):
    names = ["Penal Code, 1860", "BOMBAY POLICE ACT, 1951"]
    assert act_refs(names, aliases) == [
        ("Penal Code, 1860", "penal code, 1860"),
        ("BOMBAY POLICE ACT, 1951", "bombay police act, 1951"),
    ]
    assert unmapped_act_names([*names, "Bombay Police Act, 1951"], aliases) == {
        "bombay police act, 1951": 2
    }


def test_hyphenated_word_is_not_split(settings):
    text = "Co-operative Societies: Maharashtra Co-operative Societies Act, 1960 - ss. 91 and 163."
    assert acts(text, settings) == ["Maharashtra Co-operative Societies Act, 1960"]


def test_bracket_with_comma_inside_name(settings):
    text = (
        "Environment – Hazardous Wastes (Management, Handling and Transboundary Movement) "
        "Rules, 2008 – r.3 – validity."
    )
    assert acts(text, settings) == [
        "Hazardous Wastes (Management, Handling and Transboundary Movement) Rules, 2008"
    ]


def test_ocr_full_stop_before_year(settings):
    text = "Recovery of Debts Due to Banks and Financial Institutions\nAct. 1993- Enforcement."
    assert acts(text, settings) == [
        "Recovery of Debts Due to Banks and Financial Institutions Act, 1993"
    ]
