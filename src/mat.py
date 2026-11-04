"""The mat. ArUco markers turn colour pixels into table centimetres, depth gives the table plane.

To see it live, run tools/mat_view.py.
"""
import cv2
import numpy as np
import yaml

CONFIG = yaml.safe_load(open("configs/default.yaml"))["mat"]
MARKERS_CM = {int(k): v for k, v in CONFIG["markers"].items()}
DETECTOR = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
)


def find_markers(colour):
    """Pixel centre of each of our markers the camera can see, as {id: (x, y)}."""
    corners, ids, _ = DETECTOR.detectMarkers(colour)
    if ids is None:
        return {}
    return {int(i): c[0].mean(axis=0) for i, c in zip(ids.flatten(), corners) if int(i) in MARKERS_CM}


def mat_homography(centres_px):
    """3x3 map from colour pixels to mat cm. Needs all 4 markers. Only true at table height."""
    ids = sorted(MARKERS_CM)
    pixels = np.float32([centres_px[i] for i in ids])
    cms = np.float32([MARKERS_CM[i] for i in ids])
    return cv2.getPerspectiveTransform(pixels, cms)


def to_mat_cm(homography, points_px):
    points = np.float32(points_px).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def fit_plane(points):
    """Least-squares plane through 3D points. The normal is the direction they vary least in."""
    centre = points.mean(axis=0)
    # full_matrices=False matters: the default builds an N by N matrix, gigabytes for 30,000 points.
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][2]
    if normal[2] > 0:
        normal = -normal  # point it towards the camera, so height above the table is positive
    return normal, normal @ centre


def table_plane(frame, homography, exclude=None):
    """Fit the table to the high-confidence depth inside the mat, skipping depth pixels in exclude.

    Returns (normal, offset, residuals in cm). A point p in metres is on the table when
    normal . p == offset, and normal . p - offset is its height above the table.
    """
    depth, k = frame["depth"], frame["intrinsics"]
    scale = k["width"] / depth.shape[1]  # colour pixels per depth pixel, 7.5 on the iPhone
    rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    u = ((cols + 0.5) * scale).ravel()  # each depth pixel's position in the colour image
    v = ((rows + 0.5) * scale).ravel()

    mat_xy = to_mat_cm(homography, np.stack([u, v], axis=1))
    xs = [p[0] for p in MARKERS_CM.values()]
    ys = [p[1] for p in MARKERS_CM.values()]
    inside = (mat_xy[:, 0] >= min(xs)) & (mat_xy[:, 0] <= max(xs)) & (mat_xy[:, 1] >= min(ys)) & (mat_xy[:, 1] <= max(ys))
    keep = inside & (frame["confidence"].ravel() == 2)
    if exclude is not None:
        keep &= ~exclude.ravel()
    if keep.sum() < 500:
        print(f"Table: too few depth points to fit. {inside.sum()} inside the mat, {keep.sum()} of them high confidence.")
        return None

    # Back-project to 3D in metres with the colour camera's focal length and centre.
    z = depth.ravel()[keep]
    points = np.stack([(u[keep] - k["cx"]) / k["fx"] * z, (v[keep] - k["cy"]) / k["fy"] * z, z], axis=1)
    # Find the table among parcels and hands: try 200 planes through 3 random points and keep
    # the one most points lie within 1 cm of. There is more table than parcel, so the table wins.
    rng = np.random.default_rng(0)
    best = None
    for _ in range(200):
        normal, offset = fit_plane(points[rng.choice(len(points), 3, replace=False)])
        votes = np.sum(np.abs(points @ normal - offset) < 0.01)
        if best is None or votes > best[0]:
            best = (votes, normal, offset)
    _, normal, offset = best
    # Refine on every point within 2 cm of the winner.
    flat = np.abs(points @ normal - offset) < 0.02
    if flat.sum() < 500:  # happens for a moment while the iPhone takes a full-resolution photo
        print(f"Table: only {flat.sum()} depth points near a plane, skipping this frame.")
        return None
    normal, offset = fit_plane(points[flat])
    return normal, offset, (points[flat] @ normal - offset) * 100
