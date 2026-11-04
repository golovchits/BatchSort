"""The fit rule, with the fictional compartments in configs/default.yaml:
XS 30 x 10, S 35 x 16, M 40 x 22, L 50 x 40 (width x height), all 60 deep.
1 cm clearance and a 2 cm margin.
"""
import yaml

from src.parcels import size_class


def test_small_box_is_certain_xs():
    assert size_class([20, 15, 5]) == ("XS", True)


def test_height_near_xs_limit_is_amber():
    # XS takes up to 9 cm (10 minus clearance). 8 cm fits, but not with the 2 cm margin.
    assert size_class([20, 15, 8]) == ("XS", False)


def test_just_over_xs_goes_to_s():
    assert size_class([20, 15, 9.5]) == ("S", True)


def test_footprint_can_decide_too():
    # Flat, but 32 cm wide. XS has 29 cm of room across, so it goes to S.
    assert size_class([50, 32, 5]) == ("S", True)


def test_parcel_is_turned_to_fit():
    # 7 by 25 by 50 standing up: lying down it is 50 x 25 x 7, which fits XS.
    assert size_class([7, 25, 50]) == ("XS", True)


def test_tall_box_lies_down_in_l():
    # 45 cm tall is not too big: on its side it is 45 x 30 x 30, which fits L (60 x 50 x 40).
    assert size_class([30, 30, 45]) == ("L", True)


def test_too_long_is_red():
    # Every compartment has 59 cm of room in depth. 62 is more than the margin over.
    assert size_class([62, 30, 20]) == (None, False)


def test_just_too_long_for_l_is_amber():
    # 61 cm long: 59 cm of room. Within the 2 cm margin, so it might still fit.
    assert size_class([61, 40, 30]) == ("L", False)


def test_dev_parcels_by_ruler():
    """Our measured dev parcels, by their ruler sizes. Expected classes worked out by hand."""
    truth = yaml.safe_load(open("data/mock/truth.yaml"))["dev"]
    expected = {"D01": ("S", True), "D02": ("M", True), "D03": ("S", True), "D04": ("S", False),
                "D05": ("L", True), "D06": ("XS", True), "D07": (None, False)}
    for parcel_id, want in expected.items():
        p = truth[parcel_id]
        assert size_class([p["length"], p["width"], p["height"]]) == want, parcel_id
