"""Make one A4 page with the 4 ArUco markers for the mat corners.

    python -m tools.print_markers            # writes data/markers.pdf

Print at 100% (no "fit to page"), then check the 10 cm bar with a ruler.
"""
import sys

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

MARKER_CM = 8
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CORNERS = {0: "top left", 1: "top right", 2: "bottom right", 3: "bottom left"}

out = sys.argv[1] if len(sys.argv) > 1 else "data/markers.pdf"
page = canvas.Canvas(out, pagesize=A4)
page.setFont("Helvetica", 11)
page.drawString(2 * cm, 28 * cm, "Batchsort mat markers. Print at 100%. This bar must measure 10 cm:")
page.rect(2 * cm, 27 * cm, 10 * cm, 0.3 * cm, fill=1)

for marker_id, corner in CORNERS.items():
    # A 4x4 marker plus its 1-cell black border is 6 by 6 cells. Draw each black cell as a square,
    # so the printed size is exact instead of a scaled bitmap.
    cells = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, 6)
    cell = MARKER_CM * cm / 6
    left = (1.5 + (marker_id in (1, 2)) * 10) * cm  # 2 cm white gap between markers
    bottom = (15 if marker_id in (0, 1) else 3.5) * cm
    for row in range(6):
        for col in range(6):
            if cells[row, col] == 0:
                page.rect(left + col * cell, bottom + (5 - row) * cell, cell, cell, stroke=0, fill=1)
    page.drawString(left, bottom - 0.8 * cm, f"Marker {marker_id}: {corner} of the mat")

page.save()
print(f"Wrote {out}")
