"""Every parcel on the mat and its decision.

A parcel is known by its barcode, the carrier's tracking code, which the Mac reads from the live
video several times a second. Its spot is where its label was last seen, as 0 to 1 shares of the
frame. A new code puts a "reading" card up at once. One photo then goes to SAM, which measures, and
to Spark, which reads the label. Both run side by side. A parcel whose code can't be read shows up
as a bump in depth with no code on it, and is given a card the same way, with its spot on the bump.
Every decision is appended to results/events.jsonl with the numbers behind it.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.allocate import book
from src.match import decide, match
from src.parcels import measure
from src.read_label import read_parcel
from src.watch import BUMP_CM, at, label_there, patch

EVENTS = Path("results/events.jsonl")
PAST_EDGE = "reaches past the mat, measure it by hand"
NOT_MEASURED = "not measured, check the size by hand"
OUT_OF_VIEW = "runs out of the camera's view, turn it so the whole label shows"
CODE_DIFFERS = "label and barcode don't agree, check it by hand"
GONE_S = 1.0  # a parcel standing up from the table is gone this long after its code was last seen, once its spot is flat
GONE_FLAT_S = 3.0  # a flat parcel makes no bump, so its label tells: gone when neither its code nor the picture of its label was seen for this long
FLAT_FRAMES = 5  # 0.5 s of flat table at a parcel's spot
SETTLE_S = 1.5  # a parcel's first 1.5 s on a card decide whether it stands up from the table
SAME_SPOT = 0.06  # labels this close (a share of the frame, about 6 cm) lie in the same place
GONE_LONG_S = 8  # a code not seen this long with no arm over the mat: gone, whatever depth says
SEEN_TWICE_S = 5  # a new code gets a card once seen twice within this time
MISREAD_CHARS = 2  # a new code this close to one on the mat is a misread of it: on dev video, 2 digits of 12 once
SHORTEST_CODE = 8  # tracking codes are 10 characters or more. On dev video, 4 digits of a 12 digit barcode read as a code


def log(event):
    EVENTS.parent.mkdir(exist_ok=True)
    with EVENTS.open("a") as f:
        f.write(json.dumps({"time": time.time(), **event}) + "\n")


def misread(code, other):
    """Two codes of the same length that differ in only a few characters."""
    return code != other and len(code) == len(other) and sum(a != b for a, b in zip(code, other)) <= MISREAD_CHARS


def box_of(labels, label, size):
    """The photo box around one depth bump. size: the photo's width and height."""
    rows, cols = np.nonzero(labels == label)
    sx, sy = size[0] / labels.shape[1], size[1] / labels.shape[0]
    return {"x1": int(cols.min() * sx), "y1": int(rows.min() * sy), "x2": int((cols.max() + 1) * sx), "y2": int((rows.max() + 1) * sy)}


class Station:
    def __init__(self, world):
        self.recipients = {r["id"]: r for r in world["recipients"]}
        self.buildings = {b["id"]: b for b in world["buildings"]}
        self.sizes = {c["id"]: c["size"] for b in world["buildings"] for c in b["compartments"]}
        self.on_mat, self.done, self.next_id = [], [], 1
        self.message = ""
        self.labels = None  # the latest depth bumps, a label image
        self.first_seen = {}  # when each code not yet on a card was first seen

    def add(self, code, spot, now):
        parcel = {"id": self.next_id, "code": code, "spot": spot, "seen": now, "born": now, "state": "reading", "reason": "reading…",
                  "raised": False, "flat": 0, "asked": False, "compartment": None}
        self.next_id += 1
        self.on_mat.append(parcel)
        return parcel

    def sight(self, codes, now, colour=None):
        """Codes seen on the mat, {text: spot}, in the video frame colour. Returns True when a new
        parcel needs a photo."""
        new = False
        for code, spot in codes.items():
            parcel = next((p for p in self.on_mat if p["code"] == code), None)
            if len(code) < SHORTEST_CODE or parcel is None and any(misread(code, p["code"]) for p in self.on_mat):
                continue
            if parcel is None and now - self.first_seen.setdefault(code, now) > SEEN_TWICE_S:
                self.first_seen[code] = now  # last seen too long ago: start again
            if parcel is None and self.first_seen[code] == now:
                continue  # a code seen once may be a misread. Wait for a second sighting
            if parcel is None and (parcel := next((p for p in self.done if p["code"] == code), None)):
                self.done.remove(parcel)  # taken off and put back
                parcel.pop("status")
                parcel["flat"] = 0
                self.on_mat.append(parcel)
                log({"parcel": parcel["id"], "decision": "back on the mat", "code": code})
            if parcel is None and self.labels is not None and at(self.labels, spot):
                # A parcel given a card from depth before its code was read: the same bump.
                parcel = next((p for p in self.on_mat if not p["code"] and at(self.labels, p["spot"]) == at(self.labels, spot)), None)
                if parcel:
                    parcel["code"] = code
            if parcel is None:
                parcel, new = self.add(code, spot, now), True
            parcel["spot"], parcel["seen"], parcel["by_depth"] = spot, now, False
            if colour is not None:
                parcel["picture"] = patch(colour, spot)
        return new

    def unknown(self, spot, now):
        """A bump no code explains: a parcel whose code can't be read."""
        self.add("", spot, now)

    def check_gone(self, heights, arm, now, colour=None):
        """Move parcels that left the mat to done. Nothing leaves while an arm hides the mat."""
        for parcel in list(self.on_mat):
            if colour is not None and "picture" in parcel and label_there(colour, parcel["spot"], parcel["picture"]):
                parcel["there"] = now
            flat = at(heights, parcel["spot"]) < BUMP_CM
            if now - parcel["born"] < SETTLE_S:  # whether it stands up from the table is learnt in its first moments
                parcel["raised"] |= not flat
            parcel["flat"] = parcel["flat"] + 1 if flat and not arm else 0
            unseen = now - parcel["seen"]
            if not parcel["code"] or parcel.get("by_depth"):
                gone = parcel["flat"] >= FLAT_FRAMES  # its code is not read from the video, so only depth tells
            elif parcel["raised"]:
                # Its spot went flat, or another label now lies there: swapped in the same place.
                taken = any(q is not parcel and now - q["seen"] < GONE_S and np.hypot(*np.subtract(q["spot"], parcel["spot"])) < SAME_SPOT
                            for q in self.on_mat)
                gone = unseen > GONE_S and (parcel["flat"] >= FLAT_FRAMES or taken) or unseen > GONE_LONG_S and not arm
            else:
                # A flat parcel's spot standing up later means another parcel was put on top of it.
                gone = (now - max(parcel["seen"], parcel.get("there", 0)) > GONE_FLAT_S or unseen > GONE_S and not flat) and not arm
            if gone:
                self.finish(parcel, "taken off the mat")

    def waiting(self):
        """Parcels that still need a photo."""
        return [p for p in self.on_mat if p["state"] == "reading" and not p["asked"]]

    def process(self, photo, depth_frame, homography, labels, parcel_bumps, table=None):
        """Measure and read every parcel waiting for a photo. labels and parcel_bumps: the depth bumps
        when the photo was asked for. table: the table fitted on the empty mat. SAM and Spark run side by side."""
        pending = self.waiting()
        for parcel in pending:
            parcel["asked"] = True
        size = photo["colour"].shape[1], photo["colour"].shape[0]
        to_photo = lambda spot: (spot[0] * size[0], spot[1] * size[1])
        with ThreadPoolExecutor(max_workers=8) as pool:
            sam = pool.submit(measure, photo, depth_frame, homography, {p["code"]: to_photo(p["spot"]) for p in self.on_mat}, table)
            # A parcel alone on its bump is cut out along the bump, so Spark need not wait for SAM.
            # Parcels touching each other share a bump, and wait for SAM's box.
            alone = lambda p: at(labels, p["spot"]) in parcel_bumps and [at(labels, q["spot"]) for q in self.on_mat].count(at(labels, p["spot"])) == 1
            reads = {p["id"]: pool.submit(read_parcel, photo["colour"], box_of(labels, at(labels, p["spot"]), size))
                     for p in pending if alone(p)}
            found, table = sam.result()
            if table is None:
                self.message = "Can't see the table. Is a parcel too big for the camera?"
            for parcel in pending:
                x, y = to_photo(parcel["spot"])
                box = next((b for b in found if b["box"]["x1"] <= x <= b["box"]["x2"] and b["box"]["y1"] <= y <= b["box"]["y2"]), None)
                parcel["measured"] = box
                if parcel["id"] not in reads and box:  # flat: no bump to cut along, so SAM's box
                    reads[parcel["id"]] = pool.submit(read_parcel, photo["colour"], box["box"])
            for parcel in pending:
                fields, barcodes, _ = reads[parcel["id"]].result() if parcel["id"] in reads else ({}, [], None)
                self.read(parcel, fields, barcodes)
        by_person = {}  # several green parcels for one person are booked together, into one compartment
        for parcel in pending:
            # A parcel past the edge is booked too: if even the part SAM saw is too big, it is red "too big".
            if parcel["match"]["state"] == "green" and parcel["measured"]:
                by_person.setdefault(parcel["match"]["recipient"], []).append(parcel)
            else:
                self.settle(parcel, (None, False, ""))
        for recipient_id, parcels in by_person.items():
            booked = self.book_for(recipient_id, parcels)
            for parcel in parcels:
                self.settle(parcel, booked)

    def read(self, parcel, fields, barcodes):
        box = parcel.pop("measured")
        sides = [box["length"], box["width"], box["height"]] if box else [0, 0, 0]
        parcel.update(fields=fields, length=sides[0], width=sides[1], height=sides[2], measured=bool(box),
                      past_edge=bool(box and box["past_edge"]), soft=fields.get("packaging") == "soft")
        # A parcel found by depth keeps being watched by depth: its code is read from the photo, not the video.
        parcel["by_depth"] = parcel.get("by_depth", not parcel["code"])
        parcel["code"] = parcel["code"] or (barcodes[0] if len(barcodes) == 1 else "") or fields.get("tracking_number", "")
        parcel["match"] = match(fields, list(self.recipients.values())) if fields.get("recipient_name") else {
            "state": "amber", "reason": OUT_OF_VIEW if parcel["past_edge"] else "label not readable, check it by hand", "candidates": []}
        printed = fields.get("tracking_number", "").replace(" ", "")
        # The crop may hold a neighbour's label: the printed number Spark read gives it away. A digit
        # or two apart is Spark misreading the print (D07: a 5 read as a 6), and the barcode wins.
        if printed and printed != parcel["code"] and not misread(printed, parcel["code"]):
            parcel["match"] = {**parcel["match"], "state": "amber", "reason": CODE_DIFFERS, "candidates": []}

    def finish(self, parcel, why):
        """The parcel left the mat for its compartment."""
        parcel["status"] = "done"
        self.on_mat.remove(parcel)
        self.done.insert(0, parcel)
        log({"parcel": parcel["id"], "decision": "done", "why": why, "code": parcel["code"],
             "compartment": parcel.get("compartment")})

    def book_for(self, recipient_id, parcels):
        building = self.buildings[self.recipients[recipient_id]["building"]]
        return book([[p["length"], p["width"], p["height"]] for p in parcels], recipient_id, building)

    def settle(self, parcel, booked):
        compartment, certain, reason = booked
        parcel["state"], parcel["reason"] = decide(parcel["match"], booked, parcel["soft"])
        if parcel["match"]["state"] == "green":  # the label is settled, the size may not be
            if not parcel["measured"]:
                parcel["state"], parcel["reason"] = "amber", NOT_MEASURED
            elif parcel["past_edge"] and compartment:
                parcel["state"], parcel["reason"] = "amber", PAST_EDGE
        parcel["compartment"] = compartment["id"] if compartment else None
        log({"parcel": parcel["id"], "decision": parcel["state"], "why": parcel["reason"], "code": parcel["code"],
             "recipient": parcel["match"].get("recipient"), "score": parcel["match"].get("score"),
             "gap": parcel["match"].get("gap"), "compartment": parcel["compartment"],
             "sides_cm": [round(parcel[k], 1) for k in ("length", "width", "height")]})

    def tap(self, parcel_id, recipient_id=None):
        """The worker settles an amber: picks the recipient, or confirms a size near a boundary."""
        parcel = next((p for p in self.on_mat if p["id"] == parcel_id), None)
        if parcel is None or parcel["state"] != "amber":
            return
        if recipient_id:  # the worker named the recipient, so the match is now certain
            parcel["match"] = {**parcel["match"], "state": "green", "recipient": recipient_id, "reason": "picked by worker"}
            self.settle(parcel, self.book_for(recipient_id, [parcel]))
        else:
            parcel["state"], parcel["reason"] = "green", "size confirmed by worker"
            log({"parcel": parcel_id, "decision": "green", "why": parcel["reason"], "compartment": parcel["compartment"]})

    def too_big(self, parcel_id):
        """The worker says a parcel in size doubt fits no compartment: red "too big", for the agent."""
        parcel = next((p for p in self.on_mat if p["id"] == parcel_id), None)
        if parcel is None or parcel["state"] != "amber" or parcel["match"]["state"] != "green":
            return
        parcel["state"], parcel["reason"], parcel["compartment"] = "red", "too big", None
        log({"parcel": parcel_id, "decision": "red", "why": "too big, said the worker", "code": parcel["code"]})

    def view(self):
        """What the page shows: every parcel on the mat and the last few done."""
        return {"message": self.message,
                "on_mat": [self.card(p) for p in self.on_mat],
                "done": [self.card(p) for p in self.done[:5]]}

    def card(self, p):
        card = {"id": p["id"], "state": p.get("status") or p["state"], "reason": p["reason"], "code": p["code"],
                "spot": p["spot"], "compartment": p["compartment"], "size": self.sizes.get(p["compartment"]),
                "label": "", "recipient": "", "sizes": "", "soft": False, "confirm_size": False, "candidates": [],
                "agent": p.get("agent", ""), "outcome": p.get("agent_outcome", ""), "can_be_too_big": False}
        if p["state"] == "reading":
            return card
        m, person = p["match"], self.recipients.get(p["match"].get("recipient"))
        card.update(label=p["fields"].get("recipient_name", ""), soft=p["soft"],
                    sizes=f"{p['length']:.0f} x {p['width']:.0f} x {p['height']:.0f} cm" if p["measured"] else "",
                    confirm_size=p["state"] == "amber" and m["state"] == "green" and bool(p["compartment"]),
                    can_be_too_big=p["state"] == "amber" and m["state"] == "green",
                    recipient=f"{person['first']} {person['last']} ({person['code']})" if person and m["state"] == "green" else "")
        if p["state"] == "amber" and m["state"] == "amber":
            card["candidates"] = [{"id": rid, "name": f"{self.recipients[rid]['first']} {self.recipients[rid]['last']}",
                                   "code": self.recipients[rid]["code"], "building": self.recipients[rid]["building"]}
                                  for rid in m.get("candidates", [])]
        return card
