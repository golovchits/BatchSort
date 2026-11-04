"""Watching the live depth stream: is the mat still, is a worker reaching in, and what stands on it.

All of it works on depth, not colour: a worker's moving shadow changes the colour of the mat but not
its depth. Anything standing up from the table is a bump in depth. A bump that runs out of the
camera's view and stands over ARM_CM high well inside the mat is a worker's arm, since they stand
outside the view. Any other bump on the mat is a parcel.
"""
from collections import deque

import cv2
import numpy as np

from src.parcels import height_map

STILL_FRAMES = 6  # 0.5 s at 10 frames a second
CHANGED_CM = 1.5  # a depth pixel that moved more than this has changed
MOVING_SHARE = 0.003  # still: at most this share of the mat changed depth over those 0.5 s. A hand is about 1%
BUMP_CM = 2  # depth this far above the table is a bump: a parcel, a hand or an arm
ARM_CM = 5  # an arm stands at least this high where it crosses into the camera's view
MAT_MARGIN = 0.1  # how far inside the mat an arm must reach, as a share of the frame's width
ARM_MIN_PX = 150  # about 20 cm2: a forearm reaching in covers far more
MIN_BUMP_PX = 300  # about 40 cm2 of the mat. Smaller bumps are noise
LABEL_PX = 60  # half the side of the square kept around a barcode, in video pixels: about 4 cm
SHIFT_PX = 10  # how far the label may have moved and still count as the same
SAME_LABEL = 0.6  # correlation above this: the same label


class Stillness:
    """Whether the mat has been still for the last half second."""

    def __init__(self, mat_px):
        self.mat_px, self.recent = mat_px, deque(maxlen=STILL_FRAMES)  # the mat in stream pixels: x1, y1, x2, y2

    def mat_depth(self, frame):
        """Depth of the mat area in cm, smoothed against LiDAR noise. Pixels the phone is not sure of
        are left out (NaN): on a plain table their depth drifts by several cm with nothing there."""
        x1, y1, x2, y2 = (np.array(self.mat_px) * frame["depth"].shape[1] // frame["intrinsics"]["width"])
        depth = cv2.medianBlur(frame["depth"][y1:y2, x1:x2], 5) * 100
        return np.where(frame["confidence"][y1:y2, x1:x2] == 2, depth, np.nan)

    def still(self, frame):
        depth = self.mat_depth(frame)
        if self.recent and self.recent[0].shape != depth.shape:
            self.recent.clear()  # the mat was measured again
        self.recent.append(depth)
        return len(self.recent) == STILL_FRAMES and np.mean(np.abs(depth - self.recent[0]) > CHANGED_CM) <= MOVING_SHARE


def bumps(frame, plane, mat_mask):
    """Heights above the table, a label image of the bumps, the parcel bumps' labels, and whether an
    arm is over the mat."""
    heights = cv2.medianBlur(height_map(frame, plane).astype(np.float32), 5)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((heights > BUMP_CM).astype(np.uint8))
    edge = lambda image: set(np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])) - {0}
    border = edge(labels)
    # An arm: over ARM_CM high all the way from the edge of the view to well inside the mat. The
    # table's near edge reads 2 to 40 cm high in depth, and a soft bag's low rim can join it up
    # with the bag, so the low part of a bump doesn't count, and a few pixels inside don't either.
    margin = int(MAT_MARGIN * mat_mask.shape[1])
    inner = cv2.erode(mat_mask.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)
    high = cv2.connectedComponents((heights > ARM_CM).astype(np.uint8))[1]
    reach = np.bincount(high[inner], minlength=high.max() + 1)
    arm = any(reach[i] >= ARM_MIN_PX for i in edge(high))
    on_mat = set(np.unique(labels[mat_mask])) - {0}
    parcels = {i for i in on_mat - border if stats[i, cv2.CC_STAT_AREA] >= MIN_BUMP_PX}  # a parcel lies inside the view
    return heights, labels, parcels, arm


def patch(colour, spot, half=LABEL_PX):
    """A grey square of the video frame around a spot given as 0 to 1 shares."""
    height, width = colour.shape[:2]
    x, y = int(spot[0] * width), int(spot[1] * height)
    return cv2.cvtColor(colour[max(y - half, 0):y + half, max(x - half, 0):x + half], cv2.COLOR_BGR2GRAY)


def label_there(colour, spot, seen):
    """Whether the label last seen at spot is still there. A flat parcel makes no bump in depth, and
    some codes are read only every few seconds, so the picture of the label stands in. Compared
    with normalised correlation, so a shadow passing over it still matches. A small shift is allowed."""
    now = patch(colour, spot, LABEL_PX + SHIFT_PX)
    if seen.size == 0 or now.shape[0] < seen.shape[0] or now.shape[1] < seen.shape[1]:
        return False
    return cv2.matchTemplate(now, seen, cv2.TM_CCOEFF_NORMED).max() > SAME_LABEL


def at(image, point):
    """The value of a depth-sized image at a point given as 0 to 1 shares of the frame."""
    height, width = image.shape[:2]
    return image[min(int(point[1] * height), height - 1), min(int(point[0] * width), width - 1)]
