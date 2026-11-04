"""Batchsort station. Open http://localhost:8766 for the worker screen.

Live:    python app.py
Record:  python app.py --record sessions/dev5
Replay:  python app.py --replay sessions/dev4   (uses the photos in the recording, calls the API)

Nobody presses a key. Three things run side by side:
- The video goes to the page, cropped to the mat, ten times a second.
- The Mac reads barcodes from the video, about ten frames a second. A new code gets a card at once.
- Depth says whether a worker is reaching in, whether the mat is still, and whether a parcel has
  left. When a new parcel waits and the mat is still, the Mac asks the iPhone for a photo, and SAM
  and Spark work on it in a background thread, so nothing else waits for them.
"""
import argparse
import asyncio
import time
import traceback
from itertools import cycle
from pathlib import Path

import cv2
import numpy as np
import yaml

from src import agent, screen
from src.mat import CONFIG, MARKERS_CM, find_markers, mat_homography, table_plane, to_mat_cm
from src.parcels import MAT_BORDER_CM
from src.read_label import SCAN_ANGLES, scan_frame
from src.station import Station
from src.stream import parse_packet, receive, request_photo
from src.watch import Stillness, at, bumps

VIDEO_WIDTH = 960  # frames sent to the page
MARKER_CHECK = 10  # look for the markers once a second: the phone may have been moved
MARKER_MOVED_PX = 10  # a marker this far from where it was means the phone moved: measure the mat again
UNKNOWN_FRAMES = 30  # a bump with no code on it for 3 s of still mat is a parcel whose code can't be read
PHOTO_WAIT_S = 2  # don't ask for another photo while one is on its way
COVERED = "A parcel covers a marker. Move it to the middle of the mat."
COVERED_CHECKS = 3  # 3 s: a hand passing over a marker doesn't count
SETTLED_FRAMES = 10  # 1 s of still mat: whatever looks like an arm is a parcel at the edge, or a resting hand

world = yaml.safe_load(open("data/mock/world.yaml"))
station = Station(world)
state = {"centres": None, "homography": None, "plane": None, "mat_mask": None, "crop": None, "scan_box": None,
         "stillness": None, "markers": 4, "frames": 0, "arm": False, "still_for": 0, "covered_for": 0, "unknown": 0, "asked_at": 0,
         "photo_depth": None, "loop": None, "last_stream": None, "busy": False, "seconds": None}


def send_view():
    chats = [{"recipient": rid, "name": f"{station.recipients[rid]['first']} {station.recipients[rid]['last']}", "lines": lines}
             for rid, lines in phone.chats.items()]
    screen.send_view({**station.view(), "busy": state["busy"], "seconds": state["seconds"], "crop": state["crop"], "phone": chats})


def from_thread():
    """send_view, called from the agent's thread."""
    state["loop"].call_soon_threadsafe(send_view)


phone = agent.Phone(on_change=from_thread)


def start_agents():
    """Hand every red "too big" parcel to the recipient agent, once. It runs in a thread of its own."""
    for parcel in station.on_mat:
        if parcel["state"] == "red" and parcel["reason"] == "too big" and "agent" not in parcel:
            parcel["agent"] = "starting"
            person = station.recipients[parcel["match"]["recipient"]]
            state["loop"].create_task(asyncio.to_thread(agent.run, parcel, person, phone, on_change=from_thread))


def mat_box(homography, border_cm):
    """The mat in stream pixels, x1, y1, x2, y2, reaching border_cm past the marker centres."""
    xs, ys = [p[0] for p in MARKERS_CM.values()], [p[1] for p in MARKERS_CM.values()]
    corners_cm = [(x, y) for x in (min(xs) - border_cm, max(xs) + border_cm) for y in (min(ys) - border_cm, max(ys) + border_cm)]
    corners = to_mat_cm(np.linalg.inv(homography), corners_cm)
    (x1, y1), (x2, y2) = corners.min(axis=0).astype(int), corners.max(axis=0).astype(int)
    return max(x1, 0), max(y1, 0), x2, y2


def calibrate(frame, centres):
    """Everything that depends on where the phone is: the homography, the mat area, the table, the crop."""
    homography = mat_homography(centres)
    height, width = frame["colour"].shape[:2]
    shares = lambda box: [box[0] / width, box[1] / height, (min(box[2], width) - box[0]) / width, (min(box[3], height) - box[1]) / height]
    mat_px = mat_box(homography, 0)
    state["crop"] = shares(mat_box(homography, CONFIG["marker_size_cm"] / 2))  # the page shows up to the markers' outer edges
    state["scan_box"] = shares(mat_box(homography, MAT_BORDER_CM))  # codes outside it are not on the mat
    to_depth = frame["depth"].shape[1] / width
    mask = np.zeros(frame["depth"].shape, bool)
    mask[int(mat_px[1] * to_depth):int(mat_px[3] * to_depth), int(mat_px[0] * to_depth):int(mat_px[2] * to_depth)] = True
    # A big parcel can cover more of the mat than the table shows, and the fit would take its top for
    # the table. So the table is fitted again only on an empty mat. A phone that drifted 10 pixels
    # tilts it by far less than the 2 cm a bump needs.
    if state["plane"] is None or not station.on_mat:
        table = table_plane(frame, homography)
        state["plane"] = table[:2] if table else None
    state.update(centres=centres, homography=homography, mat_mask=mask, stillness=Stillness(mat_px))
    print("Mat found: all 4 markers seen")
    send_view()


def check_markers(frame):
    """Find the markers again: the phone may have moved, or a parcel may cover one."""
    centres = find_markers(frame["colour"])
    state["markers"] = len(centres)
    state["covered_for"] = state["covered_for"] + 1 if 0 < len(centres) < 4 else 0  # in checks, one a second
    if len(centres) == 4 and (state["centres"] is None or state["plane"] is None or max(
            np.hypot(*(centres[i] - state["centres"][i])) for i in centres) > MARKER_MOVED_PX):
        calibrate(frame, centres)


def warn():
    """Warnings about the mat itself. They never replace a message about a photo."""
    warning = COVERED if state["covered_for"] >= COVERED_CHECKS else ""
    if station.message in ("", COVERED) and station.message != warning:
        station.message = warning
        send_view()


async def scan_loop():
    """Read barcodes from the newest video frame, over and over, turning it a different way each time."""
    for angle in cycle(SCAN_ANGLES):
        frame = state["last_stream"]
        if frame is None or state["scan_box"] is None:
            await asyncio.sleep(0.1)
            continue
        try:
            station.sight(await asyncio.to_thread(scan_frame, frame["colour"], angle, state["scan_box"]), time.time(), frame["colour"])
            send_view()
        except Exception:  # a crash here would stop all reading, silently
            traceback.print_exc()


async def work_on(photo, depth):
    state["busy"], started = True, time.time()
    send_view()
    try:
        # A parcel covering a marker is still measured: the mat was found from the video earlier, and a
        # parcel cut off by the edge of the photo goes amber (parcels.measure, past_edge).
        await asyncio.to_thread(station.process, photo, depth["frame"], state["homography"], depth["labels"], depth["bumps"], state["plane"])
    except Exception as error:  # an API or network error must not stop the station
        station.message = f"Could not process the photo: {error}"
        print(station.message)
        for parcel in station.on_mat:
            if parcel["state"] == "reading":
                parcel["asked"] = False  # try again with the next photo
    state["busy"], state["seconds"] = False, round(time.time() - started, 1)
    start_agents()
    send_view()


def watch_depth(frame):
    """Arms, stillness, parcels leaving, bumps without a code, and when to ask for a photo."""
    heights, labels, parcel_bumps, arm = bumps(frame, state["plane"], state["mat_mask"])
    still = state["stillness"].still(frame)
    state["still_for"] = state["still_for"] + 1 if still else 0
    # An arm moves. A big parcel at the edge of the mat looks like one in depth, but it keeps still.
    state["arm"] = arm and state["still_for"] < SETTLED_FRAMES
    station.labels = labels
    before = len(station.on_mat)
    station.check_gone(heights, state["arm"], time.time(), frame["colour"])
    explained = {at(labels, p["spot"]) for p in station.on_mat}
    free = [b for b in parcel_bumps if b not in explained]
    state["unknown"] = state["unknown"] + 1 if free and still and not state["arm"] else 0
    if state["unknown"] >= UNKNOWN_FRAMES:
        rows, cols = np.nonzero(labels == max(free, key=lambda b: np.sum(labels == b)))
        station.unknown((cols.mean() / labels.shape[1], rows.mean() / labels.shape[0]), time.time())
        state["unknown"] = 0
    if len(station.on_mat) != before:
        send_view()
    ready = still and not state["arm"] and not state["busy"]
    if ready and station.waiting() and time.time() - state["asked_at"] > PHOTO_WAIT_S:
        state["asked_at"] = time.time()
        state["photo_depth"] = {"frame": frame, "labels": labels, "bumps": parcel_bumps}  # depth from before the photo
        request_photo()


def on_frame(frame, fps=0.0):
    if frame["kind"] == "photo":
        if state["homography"] is not None and not state["busy"] and station.waiting():
            depth = state["photo_depth"]
            if depth is None:  # replay: the photo arrives without our asking
                last = state["last_stream"]
                _, labels, parcel_bumps, _ = bumps(last, state["plane"], state["mat_mask"])
                depth = {"frame": last, "labels": labels, "bumps": parcel_bumps}
            state["photo_depth"] = None
            asyncio.get_running_loop().create_task(work_on(frame, depth))
        return
    state["last_stream"], state["frames"] = frame, state["frames"] + 1
    if state["homography"] is None or state["frames"] % MARKER_CHECK == 0:
        check_markers(frame)
    if state["plane"] is not None:
        watch_depth(frame)
        warn()
    colour = frame["colour"]
    if state["crop"]:  # only the table, not the floor or the worker
        height, width = colour.shape[:2]
        left, top, w, h = state["crop"]
        colour = colour[int(top * height):int((top + h) * height), int(left * width):int((left + w) * width)]
    small = cv2.resize(colour, (VIDEO_WIDTH, VIDEO_WIDTH * colour.shape[0] // colour.shape[1]))
    screen.send_frame(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())


async def replay(session_dir):
    previous = None
    for path in sorted(Path(session_dir).glob("*.bin")):
        frame = parse_packet(path.read_bytes())
        await asyncio.sleep(0 if previous is None else min(max(0.0, frame["time"] - previous), 1.0))  # a pause in the recording plays as 1 s
        previous = frame["time"]
        on_frame(frame)
    print("Replay finished")
    await asyncio.Future()


async def main(args):
    state["loop"] = asyncio.get_running_loop()

    def on_tap(message):
        if "reply" in message:  # from the simulated phone
            phone.reply(message["to"], message["reply"])
        elif message.get("too_big"):
            station.too_big(message["parcel"])
        else:
            station.tap(message["parcel"], message.get("recipient"))
        start_agents()  # a tap can turn a parcel red "too big"
        send_view()

    async with screen.start(on_tap):
        asyncio.get_running_loop().create_task(scan_loop())
        await (replay(args.replay) if args.replay else receive(args.record, on_frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", help="folder to save every packet in, photos included")
    parser.add_argument("--replay", help="recorded session folder to play back")
    asyncio.run(main(parser.parse_args()))
