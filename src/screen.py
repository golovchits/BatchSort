"""The worker screen: a web page on http://localhost:8766. The simulated phone is on /phone.

One WebSocket per open page. The Mac pushes two kinds of message down it: each camera frame as
JPEG bytes, and the parcel decisions as JSON text. The page sends the worker's taps back as JSON,
the phone the recipient's replies.
"""
import json
from http import HTTPStatus
from pathlib import Path

from websockets.asyncio.server import broadcast, serve

PORT = 8766
PAGES = {"/": Path("web/index.html"), "/phone": Path("web/phone.html")}  # the worker screen, the simulated phone
pages = set()  # open browser tabs
latest = {"view": None}  # the last decisions sent, for a page that opens later


def send_frame(jpeg_bytes):
    broadcast(pages, jpeg_bytes)


def send_view(view):
    latest["view"] = json.dumps(view)
    broadcast(pages, latest["view"])


def serve_page(connection, request):
    """Plain HTTP for the page itself. Returning None lets a WebSocket request through."""
    if request.path in PAGES:
        response = connection.respond(HTTPStatus.OK, PAGES[request.path].read_text())
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response
    return None


def start(on_tap):
    """Start the page server. on_tap(message) gets every tap as a dict."""
    async def handle(websocket):
        pages.add(websocket)
        if latest["view"]:
            await websocket.send(latest["view"])
        try:
            async for message in websocket:
                on_tap(json.loads(message))
        finally:
            pages.discard(websocket)

    print(f"Worker screen on http://localhost:{PORT}")
    return serve(handle, "localhost", PORT, process_request=serve_page, max_size=None)
