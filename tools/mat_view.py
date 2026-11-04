"""Live view of the mat, for checking the markers, the table plane and labels.

Live:    python -m tools.mat_view
Replay:  python -m tools.mat_view --replay sessions/test1
Record:  python -m tools.mat_view --record sessions/dev1

Click two points in the "mat" window to measure the distance between them in cm.
Press p to photograph the mat: SAM 3.1 finds the parcels, depth measures them, and the
table is fitted again around them.
Press s to save the current colour frame to sessions/snapshot.jpg. Ctrl+C to stop.
"""
import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np

from src.mat import find_markers, mat_homography, to_mat_cm
from src.parcels import measure
from src.stream import receive, replay_frames, request_photo

VIEW_SCALE = 0.5  # the window shows the colour image at half size


clicks = []  # the last two clicked points, in full-size colour pixels
state = {"homography": None, "depth_frame": None, "parcels": []}
COLOURS = {True: (0, 200, 0), False: (0, 170, 255)}  # BGR: green certain, amber near a boundary


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x / VIEW_SCALE, y / VIEW_SCALE))
        del clicks[:-2]


def describe(parcel):
    name, certain = parcel["size_class"]
    sizes = f"{parcel['length']:.1f} x {parcel['width']:.1f} x {parcel['height']:.1f} cm"
    return f"#{parcel['id']} {name or 'too big'} {'' if certain else '(amber) '}{sizes}"


def draw_parcels(view, parcels, scale):
    """Box each parcel with its ID, size class and sizes. scale: view pixels per photo pixel."""
    for parcel in parcels:
        name, certain = parcel["size_class"]
        colour = (0, 0, 255) if name is None else COLOURS[certain]  # red: nothing fits
        b = parcel["box"]
        top_left, bottom_right = (int(b["x1"] * scale), int(b["y1"] * scale)), (int(b["x2"] * scale), int(b["y2"] * scale))
        cv2.rectangle(view, top_left, bottom_right, colour, 2)
        cv2.putText(view, describe(parcel), (top_left[0], top_left[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def handle(frame, fps=0.0):
    colour = frame["colour"]
    if frame["kind"] == "photo":
        Path("sessions").mkdir(exist_ok=True)
        cv2.imwrite("sessions/photo.jpg", colour)
        print(f"Saved sessions/photo.jpg, {colour.shape[1]} by {colour.shape[0]} pixels")
        if state["homography"] is None:
            print("No marker map yet: all 4 markers must be seen once.")
            return
        state["parcels"], table = measure(frame, state["depth_frame"], state["homography"])
        state["photo_width"] = colour.shape[1]
        if table is not None:
            print(f"Table: camera {abs(table[1]) * 100:.1f} cm above it, "
                  f"95% of points within {np.percentile(np.abs(table[2]), 95):.2f} cm")
        for parcel in state["parcels"]:
            print(describe(parcel))
        return
    view = cv2.resize(colour, None, fx=VIEW_SCALE, fy=VIEW_SCALE)
    centres = find_markers(colour)
    for marker_id, (x, y) in centres.items():
        cv2.circle(view, (int(x * VIEW_SCALE), int(y * VIEW_SCALE)), 8, (0, 255, 0), 2)
        cv2.putText(view, str(marker_id), (int(x * VIEW_SCALE) + 10, int(y * VIEW_SCALE)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if len(centres) == 4:
        state["homography"] = mat_homography(centres)
    else:
        cv2.putText(view, f"markers seen: {sorted(centres)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if state["parcels"]:
        draw_parcels(view, state["parcels"], colour.shape[1] / state["photo_width"] * VIEW_SCALE)

    if len(clicks) == 2 and state["homography"] is not None:
        a, b = to_mat_cm(state["homography"], clicks)
        p, q = [(int(x * VIEW_SCALE), int(y * VIEW_SCALE)) for x, y in clicks]
        cv2.line(view, p, q, (255, 0, 255), 2)
        cv2.putText(view, f"{np.linalg.norm(a - b):.1f} cm", q, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("mat", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        Path("sessions").mkdir(exist_ok=True)
        cv2.imwrite("sessions/snapshot.jpg", colour)
        print("Saved sessions/snapshot.jpg")
    elif key == ord("p"):
        state["depth_frame"] = frame  # depth from now, before the photo capture can disturb it
        request_photo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", help="recorded session folder to play back")
    parser.add_argument("--record", help="folder to save every packet in, photos included")
    args = parser.parse_args()
    cv2.namedWindow("mat")
    cv2.setMouseCallback("mat", on_click)
    if args.replay:
        for frame in replay_frames(args.replay):
            handle(frame)
    else:
        asyncio.run(receive(args.record, handle))
