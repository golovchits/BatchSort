"""Measure every photo in a recording and compare with the ruler values in data/mock/truth.yaml.

    python -m tools.compare_truth sessions/dev3

Parcels carry no ID the camera can read, so each measured parcel is paired with the truth
row closest in size. Fine for 8 parcels of clearly different sizes, not proof of identity.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

from src.mat import find_markers, mat_homography
from src.parcels import measure, size_class
from src.stream import parse_packet

truth = yaml.safe_load(open("data/mock/truth.yaml"))["dev"]
truth = {k: sorted([v["length"], v["width"], v["height"]], reverse=True) + [v["height"]]
         for k, v in truth.items() if v["length"] is not None}

paths = sorted(Path(sys.argv[1]).glob("*.bin"))
frames = [parse_packet(p.read_bytes()) for p in paths]
homography, used = None, set()
print(f"{'id':4} {'ruler L x W x H':>18} {'measured':>18} {'error':>20}  class ruler / measured")
for i, frame in enumerate(frames):
    if frame["kind"] == "stream" and len(find_markers(frame["colour"])) == 4:
        homography = mat_homography(find_markers(frame["colour"]))
    if frame["kind"] != "photo":
        continue
    depth_frame = next(f for f in reversed(frames[:i]) if f["kind"] == "stream")
    parcels, table = measure(frame, depth_frame, homography)
    for p in parcels:
        got = np.array([p["length"], p["width"], p["height"]])
        # Closest unused truth row by footprint and height as they lie.
        best = min((k for k in truth if k not in used),
                   key=lambda k: np.abs(np.array(truth[k][:2]) - got[:2]).sum() + abs(truth[k][3] - got[2]))
        used.add(best)
        ruler = sorted(truth[best][:3], reverse=True)
        lying = [v for v in ruler if v != truth[best][3]] or ruler[:2]
        want = np.array(lying[:2] + [truth[best][3]])
        err = got - want
        print(f"{best:4} {' x '.join(f'{v:.1f}' for v in want):>18} {' x '.join(f'{v:.1f}' for v in got):>18} "
              f"{' '.join(f'{e:+.1f}' for e in err):>20}  {size_class(want)} / {p['size_class']}")
