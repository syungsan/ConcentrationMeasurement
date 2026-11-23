"""Render registration names, including Japanese, on OpenCV frames."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=8)
def _font(size):
    for path in ("meiryo.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def _label(name, size):
    font = _font(size)
    left, top, right, bottom = font.getbbox(name)
    tile = Image.new("RGB", (right - left + 8, bottom - top + 8))
    ImageDraw.Draw(tile).text((4 - left, 4 - top), name, fill="white", font=font)
    return np.asarray(tile)


def draw_student_name(frame, x, y, name, scale):
    tile = _label(name, max(14, round(24 * scale)))
    h, w = tile.shape[:2]
    height, width = frame.shape[:2]
    x = max(0, min(x, width - w))
    above = y - round(44 * scale) - h
    y = max(0, min(above if above >= 0 else y + 8, height - h))
    target = frame[y:y+h, x:x+w]
    target[:] = tile[:target.shape[0], :target.shape[1]]
