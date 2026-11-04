"""Read the label on each parcel in the full-resolution photo.

Each parcel is cut out of the photo along SAM's box, with a little room around it, and only that
crop goes to Muse Spark. So Spark sees one label at a time, and nothing else on the table leaves
the Mac. The barcode is decoded on the Mac by zxing-cpp: it is the carrier's tracking number,
which says nothing about the recipient, but it tells parcels apart: the station scans the live
video for codes, and only calls SAM and Spark when a code it hasn't seen shows up.
"""
import threading

import cv2
import zxingcpp

from src.meta_api import read_label

PAD = 0.1  # room added around SAM's box, as a share of its size, so a label at the edge is kept
CROP_PATH = "sessions/label_crop_{}.jpg"  # one file per thread: labels are read side by side


def crop(colour, box):
    pad_x, pad_y = int((box["x2"] - box["x1"]) * PAD), int((box["y2"] - box["y1"]) * PAD)
    height, width = colour.shape[:2]
    return colour[max(box["y1"] - pad_y, 0):min(box["y2"] + pad_y, height),
                  max(box["x1"] - pad_x, 0):min(box["x2"] + pad_x, width)]


SCAN_ANGLES = (0, 30, 60, -30, -60)  # turns of 90 zxing tries itself
# Line barcodes only, as on carrier labels. Looking for square codes (QR and the like) as well made
# zxing hang on some video frames, and takes 5 times longer.
LINEAR = zxingcpp.BarcodeFormat.LinearCodes


def scan(image):
    """Every barcode zxing finds in the image, as {text: centre in pixels}. About 0.4 s on a full photo."""
    return {b.text: ((b.position.top_left.x + b.position.bottom_right.x) / 2,
                     (b.position.top_left.y + b.position.bottom_right.y) / 2) for b in zxingcpp.read_barcodes(image, formats=LINEAR)}


def scan_frame(colour, angle, area):
    """Barcodes on the mat in one video frame, as {text: centre as 0 to 1 shares of the frame}.

    area: the mat as left, top, width and height shares of the frame. Only that part is scanned,
    in grey and enlarged 2 times: at 1920 by 1440 the thinnest bars are under 1 pixel wide. On dev
    video this read a box in 6 of 23 frames, where the whole frame enlarged 1.5 times read it in 0.
    zxing misses a barcode turned more than about 15 degrees, so the caller cycles angle through
    SCAN_ANGLES. About 0.1 s a frame.
    """
    height, width = colour.shape[:2]
    left, top = int(area[0] * width), int(area[1] * height)
    mat = cv2.cvtColor(colour[top:top + int(area[3] * height), left:left + int(area[2] * width)], cv2.COLOR_BGR2GRAY)
    big = cv2.resize(mat, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    turn = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), angle, 1)
    back = cv2.invertAffineTransform(turn)
    found = {}
    for b in zxingcpp.read_barcodes(cv2.warpAffine(big, turn, (big.shape[1], big.shape[0])), formats=LINEAR):
        x = (b.position.top_left.x + b.position.bottom_right.x) / 2
        y = (b.position.top_left.y + b.position.bottom_right.y) / 2
        bx, by = back @ [x, y, 1]  # where it is in the unturned crop
        found[b.text] = ((left + bx / 2) / width, (top + by / 2) / height)
    return found


def read_parcel(colour, box):
    """Label fields from Spark, every barcode in the crop and Spark's token usage."""
    image = crop(colour, box)
    barcodes = [b.text for b in zxingcpp.read_barcodes(image, formats=LINEAR)]
    path = CROP_PATH.format(threading.get_ident())
    cv2.imwrite(path, image)
    fields, usage = read_label(path)
    return fields, barcodes, usage
