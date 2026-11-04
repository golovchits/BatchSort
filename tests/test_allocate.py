"""Booking rules, on a small made-up building.
Compartments as in configs/default.yaml: XS 30 x 10, S 35 x 16, M 40 x 22, L 50 x 40, all 60 deep.
"""
from src.allocate import book, stack

INNER = {"XS": [60, 30, 10], "S": [60, 35, 16], "M": [60, 40, 22], "L": [60, 50, 40]}


def building(*sizes, taken=None):
    """One compartment per size given. taken: {index: (recipient, contents)} for ones in use."""
    compartments = [{"id": f"A1-{i + 1}", "number": i + 1, "size": size, "inner_cm": INNER[size],
                     "occupied_by": None, "contents_cm": []} for i, size in enumerate(sizes)]
    for i, (recipient, contents) in (taken or {}).items():
        compartments[i]["occupied_by"], compartments[i]["contents_cm"] = recipient, contents
    return {"id": "A1", "compartments": compartments}


def test_stack_lays_each_parcel_flat():
    assert stack([[5, 20, 15], [30, 8, 20]]) == [30, 20, 13]


def test_smallest_free_compartment():
    c, certain, _ = book([[20, 15, 5]], "R1", building("L", "S", "XS"))
    assert (c["size"], certain) == ("XS", True)


def test_near_the_limit_goes_up_a_class_rather_than_amber():
    # 8 cm high fits XS only without the margin, and S for certain. Bigger is safe, so S.
    c, certain, _ = book([[20, 15, 8]], "R1", building("XS", "S"))
    assert (c["size"], certain) == ("S", True)


def test_near_the_limit_with_no_bigger_class_free_is_amber():
    c, certain, _ = book([[20, 15, 8]], "R1", building("XS"))
    assert (c["size"], certain) == ("XS", False)


def test_reuses_the_recipients_compartment():
    b = building("XS", "S", taken={1: ("R1", [[30, 20, 5]])})
    c, certain, reason = book([[20, 15, 5]], "R1", b)
    assert (c["id"], certain, reason) == ("A1-2", True, "added to their compartment")
    assert len(c["contents_cm"]) == 2


def test_own_compartment_too_full_gets_a_new_one():
    b = building("XS", "S", taken={1: ("R1", [[30, 20, 12]])})
    c, _, reason = book([[20, 15, 5]], "R1", b)
    assert (c["id"], reason) == ("A1-1", "free compartment")


def test_two_parcels_for_one_person_share_a_compartment():
    # Stacked they are 20 x 15 x 10: too high for XS with the margin, so S.
    c, certain, _ = book([[20, 15, 5], [20, 15, 5]], "R1", building("XS", "S"))
    assert (c["size"], certain, len(c["contents_cm"])) == ("S", True, 2)


def test_a_booked_compartment_is_not_given_out_twice():
    b = building("XS", "S")
    first, _, _ = book([[20, 15, 5]], "R1", b)
    second, _, _ = book([[20, 15, 5]], "R2", b)
    assert (first["id"], second["id"]) == ("A1-1", "A1-2")


def test_too_big_for_l_is_red():
    # D07 by ruler: 63.5 long, and every compartment is 60 deep.
    assert book([[63.5, 29, 35.5]], "R1", building("XS", "S", "M", "L"))[::2] == (None, "too big")


def test_full_building_is_red():
    b = building("XS", "S", taken={0: ("R2", [[20, 15, 5]]), 1: ("R3", [[20, 15, 5]])})
    assert book([[20, 15, 5]], "R1", b)[::2] == (None, "full")
