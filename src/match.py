"""Who a label is for, and how sure we are.

A locker network's labels all carry the hub's address. The name line is the network's tag, the
recipient's 5-letter code and their last name: "HVL KRTMS Velthoven". So the code and the last
name are what we match on. Each scores 0 to 100 against every recipient, by rapidfuzz's plain
string similarity, and the match score is their average. With no code on the label, the last
name alone. Many people share a last name, so a label without a code nearly always ties: amber.
"""
import yaml
from rapidfuzz import fuzz

CONFIG = yaml.safe_load(open("configs/default.yaml"))["matching"]


def split_name_line(line):
    """("KRTMS", "Velthoven") from "HVL KRTMS Velthoven". The code is "" when the label has none."""
    words = [w for w in line.split() if w.upper() != CONFIG["network_tag"]]
    codes = [w for w in words if len(w) == 5 and w.isalpha() and w.isupper()]
    code = codes[0] if codes else ""
    last = " ".join(w for w in words if w != code)
    return code, last


def score(code, last, recipient):
    name = fuzz.ratio(last.lower(), recipient["last"].lower())
    if not code:
        return name
    return (fuzz.ratio(code, recipient["code"]) + name) / 2


def match(fields, recipients):
    """Best recipient for these label fields: state "green", "amber" or "red", with the reason
    and the numbers behind it. Size is decided separately (decide below)."""
    code, last = split_name_line(fields.get("recipient_name", ""))
    if not code and not last:
        return {"state": "amber", "reason": "label not readable, check it by hand", "code": code, "last": last, "candidates": []}
    ranked = sorted(((score(code, last, r), r["id"]) for r in recipients), reverse=True)
    best, runner_up = ranked[0], ranked[1]
    result = {"code": code, "last": last, "recipient": best[1], "score": best[0],
              "gap": best[0] - runner_up[0],
              # Everyone too close to tell apart: all 13 people called Doornbos for a label with no code.
              "candidates": [rid for s, rid in ranked if best[0] - s < CONFIG["match_gap"]]}
    if best[0] < CONFIG["match_min"]:
        return {**result, "state": "red", "reason": "nobody on the list matches"}
    if result["gap"] < CONFIG["match_gap"]:
        return {**result, "state": "amber", "reason": "several people match, pick one"}
    return {**result, "state": "green", "reason": "clear match"}


def decide(matched, booked, soft=False):
    """Combine the match with the booking from allocate.book. Any doubt is amber.
    Soft never makes a parcel green. It only tells the worker the smaller class is worth a try."""
    compartment, certain, reason = booked
    if matched["state"] == "red":
        return "red", matched["reason"]
    if matched["state"] == "amber":
        return "amber", matched["reason"]
    if compartment is None:
        if reason == "too big" and soft:  # a soft parcel's size can be 15 cm off (D05: 62 cm measured, 45 by ruler)
            return "amber", "soft, may still fit the largest compartment"
        return "red", reason  # "too big" or "full"
    if not certain:
        return "amber", "size near a boundary, soft: try the smaller one" if soft else "size near a boundary"
    return "green", f"{matched['reason']}, size fits with margin"  # "picked by worker" after a tap
