"""Matching rules, on a small made-up recipient list.
No API calls: the label fields are written by hand, as Muse Spark would return them.
"""
from src.match import decide, match, split_name_line

RECIPIENTS = [
    {"id": "R1", "code": "KRTMS", "last": "Velthoven"},
    {"id": "R2", "code": "BDSAK", "last": "Doornbos"},
    {"id": "R3", "code": "PLWEX", "last": "Doornbos"},  # same last name, different person
    {"id": "R4", "code": "ZMBVF", "last": "Haverkamp"},
]


def read(line):
    return {"recipient_name": line}


def test_split_name_line():
    assert split_name_line("HVL KRTMS Velthoven") == ("KRTMS", "Velthoven")
    assert split_name_line("HVL Velthoven") == ("", "Velthoven")


def test_clean_label_is_green():
    m = match(read("HVL BDSAK Doornbos"), RECIPIENTS)
    assert (m["state"], m["recipient"]) == ("green", "R2")


def test_misspelled_name_still_finds_the_right_person():
    m = match(read("HVL KRTMS Veltohven"), RECIPIENTS)
    assert (m["state"], m["recipient"]) == ("green", "R1")


def test_one_wrong_code_letter_still_finds_the_right_person():
    m = match(read("HVL BDSAX Doornbos"), RECIPIENTS)
    assert (m["state"], m["recipient"]) == ("green", "R2")


def test_missing_code_with_a_shared_last_name_is_amber():
    m = match(read("HVL Doornbos"), RECIPIENTS)
    assert m["state"] == "amber"
    assert set(m["candidates"]) == {"R2", "R3"}


def test_someone_not_on_the_list_is_red_not_a_namesake():
    # Same last name as R2 and R3, a code that is neither: must not go green to either.
    m = match(read("HVL QQQQQ Doornbos"), RECIPIENTS)
    assert m["state"] == "red"


def test_nothing_read_is_amber():
    assert match(read(""), RECIPIENTS)["state"] == "amber"


def test_size_doubt_makes_a_clear_match_amber():
    m = match(read("HVL BDSAK Doornbos"), RECIPIENTS)
    compartment = {"id": "A1-3"}
    assert decide(m, (compartment, True, "free compartment")) == ("green", "clear match, size fits with margin")
    assert decide(m, (compartment, False, "free compartment"))[0] == "amber"
    assert decide(m, (None, False, "too big")) == ("red", "too big")
    assert decide(m, (None, False, "full")) == ("red", "full")


def test_soft_only_changes_the_hint():
    m = match(read("HVL BDSAK Doornbos"), RECIPIENTS)
    compartment = {"id": "A1-3"}
    assert decide(m, (compartment, False, "free compartment"), soft=True) == ("amber", "size near a boundary, soft: try the smaller one")
    assert decide(m, (compartment, True, "free compartment"), soft=True)[0] == "green"


def test_soft_parcel_just_over_the_largest_size_is_amber_not_red():
    matched = {"state": "green", "reason": "clear match"}
    assert decide(matched, (None, False, "too big"), soft=True)[0] == "amber"
    assert decide(matched, (None, False, "too big"), soft=False) == ("red", "too big")
