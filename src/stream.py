"""Receive frames from the iPhone, show them live, record them, or replay a recording.

Live:    python -m src.stream
Record:  python -m src.stream --record sessions/test1
Replay:  python -m src.stream --replay sessions/test1
"""
import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

PORT = 8765


def parse_packet(packet):
    """One packet: 4-byte header length, JSON header, colour JPEG, depth float16, confidence uint8."""
    header_len = struct.unpack_from("<I", packet)[0]
    header = json.loads(packet[4:4 + header_len])
    pos = 4 + header_len
    jpeg = np.frombuffer(packet, np.uint8, header["jpeg_bytes"], pos)
    pos += header["jpeg_bytes"]
    w, h = header["depth_width"], header["depth_height"]
    depth = np.frombuffer(packet, np.float16, w * h, pos).reshape(h, w).astype(np.float32)
    pos += w * h * 2
    confidence = np.frombuffer(packet, np.uint8, w * h, pos).reshape(h, w)
    return {
        "kind": header.get("kind", "stream"),  # "photo" is a full-resolution still with no depth
        "time": header["time"],
        "intrinsics": header["intrinsics"],  # fx, fy, cx, cy in colour pixels, plus the colour size
        "colour": cv2.imdecode(jpeg, cv2.IMREAD_COLOR),
        "depth": depth,  # metres
        "confidence": confidence,  # 0 low, 1 medium, 2 high
    }


def replay_frames(session_dir):
    """Yield the frames of a recorded session at the speed they were recorded."""
    previous = None
    for path in sorted(Path(session_dir).glob("*.bin")):
        frame = parse_packet(path.read_bytes())
        if previous is not None:
            time.sleep(max(0.0, frame["time"] - previous))
        previous = frame["time"]
        yield frame


def show(frame, fps):
    """Depth as colours from 0.3 m (red) to 1.5 m (blue). Pixels below high confidence are dimmed."""
    scaled = np.clip((frame["depth"] - 0.3) / 1.2 * 255, 0, 255).astype(np.uint8)
    view = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    view[frame["confidence"] < 2] //= 3
    view = cv2.resize(view, (768, 576), interpolation=cv2.INTER_NEAREST)
    cv2.putText(view, f"{fps:.1f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("depth", view)
    cv2.imshow("colour", cv2.resize(frame["colour"], (640, 480)))
    cv2.waitKey(1)


phone = {"websocket": None}  # the connected iPhone, so other code can send it requests


def request_photo():
    """Ask the iPhone for one full-resolution still. It arrives later as a frame with kind "photo"."""
    if phone["websocket"] is not None:
        asyncio.get_running_loop().create_task(phone["websocket"].send("photo"))


async def receive(record_dir, on_frame=show):
    """Wait for the iPhone and call on_frame(frame, fps) for every packet it sends."""
    if record_dir:
        Path(record_dir).mkdir(parents=True, exist_ok=True)
    arrivals = []

    async def handle(websocket):
        phone["websocket"] = websocket
        async for packet in websocket:
            if not arrivals:
                print(f"First frame: {len(packet) / 1000:.0f} KB")
            if record_dir:
                (Path(record_dir) / f"{len(arrivals):06d}.bin").write_bytes(packet)
            arrivals.append(time.time())
            recent = arrivals[-11:]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
            on_frame(parse_packet(packet), fps)
        print(f"iPhone disconnected after {len(arrivals)} frames")

    # No keepalive pings: the iPhone app never reads from the socket, so it can't answer them,
    # and the server would hang up after about 40 seconds.
    async with websockets.serve(handle, "0.0.0.0", PORT, max_size=None, ping_interval=None):
        print(f"Waiting for the iPhone on port {PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", help="folder to save every packet in")
    parser.add_argument("--replay", help="recorded session folder to play back")
    args = parser.parse_args()
    if args.replay:
        for frame in replay_frames(args.replay):
            show(frame, 0.0)
    else:
        asyncio.run(receive(args.record))
