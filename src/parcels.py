"""Parcels on the mat. SAM 3.1 finds them in the full-resolution photo, depth measures them.

For every photo:
1. SAM 3.1 puts a box around each parcel.
2. The table is fitted again from the depth outside those boxes, so a phone that sagged a
   little since the last photo doesn't skew the heights.
3. Height: the top of the depth points above the table inside the box.
4. Length and width: from SAM's mask outline, each point placed at its true height and dropped
   onto the table. Looking down, the camera sees a parcel's top plus the side that faces the
   middle of the image. So outline points facing the middle are the parcel's base, on the
   table. The others are its top edge, at the height depth reads just inside the outline
   (low along a soft bag's border). This works for a parcel turned at any angle.
"""
import cv2
import numpy as np
import yaml

from src.mat import MARKERS_CM, table_plane, to_mat_cm
from src.meta_api import segment

CONFIG = yaml.safe_load(open("configs/default.yaml"))
COMPARTMENTS = CONFIG["compartments"]
SAM_PROMPT = "package"  # on our dev photo it found boxes and a flat paper mailer; "parcel" found nothing
ABOVE_TABLE_CM = 0.5  # depth points higher than this belong to a parcel (the table is flat within 0.3)
HEIGHT_PERCENTILE = 90  # a flat top reads its top, a soft parcel its highest part
EDGE_STEP_IN = 0.06  # how far inside the outline, towards the middle, an edge's height is read
MAT_BORDER_CM = 5  # how far past the marker centres the photo sent to SAM reaches
SENT_PHOTO = "sessions/photo_sent.jpg"
# SAM gets the photo at a quarter size: about 4 s instead of 20 to 23 s on dev photos. Same class and
# colour for all 6 dev parcels as at full size, though one read 5 cm wider (too wide is the safe side).
SAM_SCALE = 0.25
SPLIT_BUMP_CM = 2  # depth this far above the table is part of a bump
SPLIT_SAME_CM = 3  # two halves of one parcel stand about equally high


def fits(sides_cm, inner_cm, extra_cm=0):
    """Whether a parcel fits a compartment with every side made bigger by extra_cm. Both are sorted longest first, so the parcel may be turned any way."""
    room = sorted((side - COMPARTMENTS["clearance_cm"] for side in inner_cm), reverse=True)
    return all(p + extra_cm <= r for p, r in zip(sorted(sides_cm, reverse=True), room))


def size_class(sides_cm):
    """Smallest compartment the parcel fits, and whether it fits for certain.

    Returns ("S", True) for a certain fit (green), ("S", False) when it only fits without the
    margin (amber, suggest S and the next class up), or (None, False) when nothing fits (red).
    """
    margin = COMPARTMENTS["margin_cm"]
    for name, inner in COMPARTMENTS["sizes"].items():
        if fits(sides_cm, inner):
            return name, fits(sides_cm, inner, margin)
    # Measured too big for everything, but within the margin of a class: that is doubt, so amber, not red.
    for name, inner in COMPARTMENTS["sizes"].items():
        if fits(sides_cm, inner, -margin):
            return name, False
    return None, False


def height_map(frame, plane):
    """Height above the table, in cm, of every depth pixel."""
    normal, offset = plane
    depth, k = frame["depth"], frame["intrinsics"]
    scale = k["width"] / depth.shape[1]
    rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    u, v = (cols + 0.5) * scale, (rows + 0.5) * scale
    points = np.stack([(u - k["cx"]) / k["fx"] * depth, (v - k["cy"]) / k["fy"] * depth, depth], axis=-1)
    return (points @ normal - offset) * 100


def on_table_cm(pixels, heights_cm, photo_intrinsics, stream_intrinsics, plane, homography):
    """Mat cm of points seen at these photo pixels, each heights_cm above the table, dropped onto it."""
    normal, offset = plane
    k, s = photo_intrinsics, stream_intrinsics
    pixels, heights_m = np.asarray(pixels, float), np.asarray(heights_cm, float) / 100
    rays = np.stack([(pixels[:, 0] - k["cx"]) / k["fx"], (pixels[:, 1] - k["cy"]) / k["fy"], np.ones(len(pixels))], axis=1)
    points = rays * ((offset + heights_m) / (rays @ normal))[:, None]  # where each ray is that high
    dropped = points - heights_m[:, None] * normal
    stream_px = np.stack([dropped[:, 0] / dropped[:, 2] * s["fx"] + s["cx"],
                          dropped[:, 1] / dropped[:, 2] * s["fy"] + s["cy"]], axis=1)
    return to_mat_cm(homography, stream_px)


def full_mask(obj):
    """SAM's mask for one object, stretched to its box (SAM shrinks masks wider than about 1024)."""
    width, height = obj["x2"] - obj["x1"], obj["y2"] - obj["y1"]
    return cv2.resize(obj["mask"].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def outline_heights(outline, centroid, centre, smooth_heights, to_depth, top):
    """Height to place each outline point at. A point whose edge faces the image centre is the
    parcel's base, on the table. Any other point is read from depth just inside the outline."""
    towards_centre = np.sum((outline - centroid) * (centre - outline), axis=1) > 0
    inside = outline + (centroid - outline) * EDGE_STEP_IN
    cols = np.clip((inside[:, 0] * to_depth).astype(int), 0, smooth_heights.shape[1] - 1)
    rows = np.clip((inside[:, 1] * to_depth).astype(int), 0, smooth_heights.shape[0] - 1)
    read = np.clip(smooth_heights[rows, cols], 0, top)
    return np.where(towards_centre, 0.0, read)


def mat_outline(photo, stream_intrinsics, homography):
    """The mat, reaching MAT_BORDER_CM past the marker centres, as 4 corners in photo pixels."""
    xs = [p[0] for p in MARKERS_CM.values()]
    ys = [p[1] for p in MARKERS_CM.values()]
    b = MAT_BORDER_CM
    corners_cm = [(min(xs) - b, min(ys) - b), (max(xs) + b, min(ys) - b), (max(xs) + b, max(ys) + b), (min(xs) - b, max(ys) + b)]
    stream_px = to_mat_cm(np.linalg.inv(homography), corners_cm)  # the same map, run backwards
    k, s = photo["intrinsics"], stream_intrinsics
    photo_px = (stream_px - [s["cx"], s["cy"]]) / [s["fx"], s["fy"]] * [k["fx"], k["fy"]] + [k["cx"], k["cy"]]
    return photo_px.astype(np.int32)


def mat_only(photo, stream_intrinsics, homography):
    """The photo with everything outside the mat painted grey. Only this copy goes to SAM,
    so nothing on the floor or behind the table leaves the Mac, and SAM can't find boxes there."""
    inside = np.zeros(photo["colour"].shape[:2], np.uint8)
    cv2.fillPoly(inside, [mat_outline(photo, stream_intrinsics, homography)], 1)
    masked = photo["colour"].copy()
    masked[inside == 0] = 128
    return masked


def depth_box(box, to_depth, shape):
    """A photo box as a mask over the depth image."""
    x1, y1, x2, y2 = (np.array([box["x1"], box["y1"], box["x2"], box["y2"]]) * to_depth).astype(int)
    mask = np.zeros(shape, bool)
    mask[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = True
    return mask


def merge_split(boxes, smooth, to_depth, codes):
    """SAM sometimes cuts one parcel in two along tape or a colour change. Boxes standing on one
    depth bump, about as high as each other, with at most one barcode between them, are one parcel.
    Two real parcels touching usually differ in height or carry two codes, so they stay apart."""
    labels = cv2.connectedComponents((smooth > SPLIT_BUMP_CM).astype(np.uint8))[1]
    groups = {}
    for box in boxes:
        under = labels[depth_box(box, to_depth, smooth.shape)]
        bump = np.bincount(under[under > 0]).argmax() if np.any(under > 0) else -len(groups) - 1  # flat: alone
        groups.setdefault(bump, []).append(box)
    merged = []
    for group in groups.values():
        heights = [np.percentile(smooth[depth_box(b, to_depth, smooth.shape)], 90) for b in group]
        x1, y1 = min(b["x1"] for b in group), min(b["y1"] for b in group)
        x2, y2 = max(b["x2"] for b in group), max(b["y2"] for b in group)
        n_codes = sum(x1 <= x <= x2 and y1 <= y <= y2 for x, y in codes.values())
        if len(group) == 1 or max(heights) - min(heights) > SPLIT_SAME_CM or n_codes > 1:
            merged += group
            continue
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        for b in group:
            mask[b["y1"] - y1:b["y2"] - y1, b["x1"] - x1:b["x2"] - x1] |= full_mask(b)
        cv2.fillPoly(mask, [cv2.convexHull(cv2.findNonZero(mask))], 1)  # close the seam between the halves
        merged.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "mask": mask})
    return merged


def measure(photo, depth_frame, homography, codes={}, known_table=None):
    """Every parcel on the mat in this photo, with sizes in cm and a size class. Also the table fit.
    codes: the barcodes scanned in the photo, {text: (x, y)}, used to keep touching parcels apart.
    known_table: the table fitted on the empty mat, used when a parcel covers too much of it to fit again."""
    small = cv2.resize(mat_only(photo, depth_frame["intrinsics"], homography), None, fx=SAM_SCALE, fy=SAM_SCALE)
    cv2.imwrite(SENT_PHOTO, small)
    boxes = segment(SENT_PHOTO, SAM_PROMPT)
    for box in boxes:  # back to full-photo pixels
        for key in ("x1", "y1", "x2", "y2"):
            box[key] = int(box[key] / SAM_SCALE)
    shape = depth_frame["depth"].shape
    to_depth = shape[1] / photo["intrinsics"]["width"]  # same field of view, so a plain scale
    in_boxes = [depth_box(box, to_depth, shape) for box in boxes]

    table = table_plane(depth_frame, homography, exclude=np.any(in_boxes, axis=0) if in_boxes else None)
    if table is None and known_table is not None:
        table = (*known_table, None)
    if table is None:
        return [], None
    plane = table[:2]
    heights = height_map(depth_frame, plane)
    confident = (depth_frame["confidence"] == 2) & (heights > ABOVE_TABLE_CM)
    k = photo["intrinsics"]
    xs = [p[0] for p in MARKERS_CM.values()]
    ys = [p[1] for p in MARKERS_CM.values()]

    smooth = cv2.medianBlur(heights.astype(np.float32), 5)  # LiDAR edges are noisy
    boxes = merge_split(boxes, smooth, to_depth, codes)
    in_boxes = [depth_box(box, to_depth, shape) for box in boxes]
    centre = np.array([k["cx"], k["cy"]])

    parcels = []
    for box, in_box in zip(boxes, in_boxes):
        mask = full_mask(box)
        # Height from depth inside the mask, not the box: a box can take in part of a taller neighbour.
        # The mask is shrunk a little first, so its noisy edges and any side the camera sees drop out.
        core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=max(mask.shape) // 30)
        rows, cols = np.nonzero(in_box)
        photo_rows = ((rows + 0.5) / to_depth).astype(int) - box["y1"]
        photo_cols = ((cols + 0.5) / to_depth).astype(int) - box["x1"]
        ok = (photo_rows >= 0) & (photo_rows < core.shape[0]) & (photo_cols >= 0) & (photo_cols < core.shape[1])
        on_parcel = np.zeros(shape, bool)
        on_parcel[rows[ok], cols[ok]] = core[photo_rows[ok], photo_cols[ok]] > 0
        top = heights[on_parcel & confident]
        height = float(np.percentile(top, HEIGHT_PERCENTILE)) if len(top) >= 10 else ABOVE_TABLE_CM  # too flat for depth
        height *= CONFIG["parcels"]["height_scale"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outline = max(contours, key=len)[:, 0, :].astype(float) + [box["x1"], box["y1"]]
        rows, cols = np.nonzero(mask)
        centroid = np.array([cols.mean() + box["x1"], rows.mean() + box["y1"]])
        lift = outline_heights(outline, centroid, centre, smooth, to_depth, height)
        footprint = on_table_cm(outline, lift, k, depth_frame["intrinsics"], plane, homography)
        (cx, cy), (w, h), _ = cv2.minAreaRect(footprint.astype(np.float32))
        if not (min(xs) < cx < max(xs) and min(ys) < cy < max(ys)):
            continue  # a box off the mat
        sides = [max(w, h), min(w, h), height]
        # Reaching the grey paint around the mat or the photo's edge: SAM saw only part of it, so its size is unknown.
        past_edge = bool(footprint[:, 0].min() < min(xs) - MAT_BORDER_CM + 1 or footprint[:, 0].max() > max(xs) + MAT_BORDER_CM - 1
                         or footprint[:, 1].min() < min(ys) - MAT_BORDER_CM + 1 or footprint[:, 1].max() > max(ys) + MAT_BORDER_CM - 1
                         or box["x1"] <= 0 or box["y1"] <= 0 or box["x2"] >= k["width"] - 1 or box["y2"] >= k["height"] - 1)
        parcels.append({
            "id": len(parcels) + 1,
            "centre": (cx, cy),
            "length": sides[0], "width": sides[1], "height": sides[2],
            "size_class": size_class(sides),
            "past_edge": past_edge,
            "box": box,  # photo pixels, for drawing
            "outline": outline[::max(1, len(outline) // 60)],  # about 60 points, photo pixels
        })
    return parcels, table
