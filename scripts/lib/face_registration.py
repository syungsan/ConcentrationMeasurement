"""Image geometry shared by face registration tools (native BGR coordinates)."""
import copy
import math

import cv2
import numpy as np


def rotate_frame(image, degrees=0):
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(image, rotations[degrees]) if degrees in rotations else image


def padded_detection_image(image, pad=80, target_size=640):
    """Resize without stretching, with enough context to redetect a tight crop."""
    h, w = image.shape[:2]
    border = max(0, int(pad))
    scale = target_size / max(h + 2 * border, w + 2 * border)
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((target_size, target_size, 3), 24, dtype=np.uint8)
    x, y = (target_size - width) // 2, (target_size - height) // 2
    canvas[y:y + height, x:x + width] = resized
    return canvas


def registration_crop(image, bbox, margin=20):
    """Keep forehead/chin context; square padding never distorts facial geometry."""
    x1, y1, x2, y2 = map(float, bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    side = int(math.ceil(max(x2-x1, y2-y1) + 2 * max(margin, .25 * max(x2-x1, y2-y1))))
    h, w = image.shape[:2]
    # Prevent malformed detections from requesting unbounded allocations.
    if side > 2 * max(h, w):
        return None
    left, top = math.floor((x1+x2-side)/2), math.floor((y1+y2-side)/2)
    right, bottom = left+side, top+side
    sx, sy, ex, ey = max(0, left), max(0, top), min(w, right), min(h, bottom)
    if ex <= sx or ey <= sy:
        return None
    return cv2.copyMakeBorder(image[sy:ey, sx:ex], sy-top, bottom-ey, sx-left, right-ex,
                             cv2.BORDER_CONSTANT, value=(24, 24, 24))


def detect_capture_faces(app, image):
    faces = app.get(image)
    if faces:
        return faces
    h, w = image.shape[:2]
    short, long = min(h, w), max(h, w)
    if long < short * 1.4:
        return []
    # A portrait frame shrinks narrow faces when fitted to a square detector.
    # Retry overlapping square tiles, and restore original-frame coordinates.
    starts = list(range(0, long-short+1, max(1, short//2)))
    if starts[-1] != long-short:
        starts.append(long-short)
    found = []
    for start in starts:
        x, y = (0, start) if h > w else (start, 0)
        for face in app.get(image[y:y+short, x:x+short]):
            translated = copy.copy(face)
            translated.bbox = np.asarray(face.bbox).copy() + np.array([x, y, x, y])
            found.append(translated)
    return found
