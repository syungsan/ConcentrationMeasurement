"""Resolution-scaled OpenCV overlay of the displayed class average."""
from collections import deque
import math

import cv2
import numpy as np


class ClassAverageGraph:
    def __init__(self, window_sec=60.0, sample_sec=0.5):
        self.window_sec = window_sec
        self.sample_sec = sample_sec
        self.history = deque(maxlen=int(window_sec / sample_sec) + 2)

    def update(self, t, score):
        if self.history and t < self.history[-1][0]:
            self.history.clear()
        while self.history and self.history[0][0] < t - self.window_sec:
            self.history.popleft()
        if not self.history or t - self.history[-1][0] >= self.sample_sec - 1e-6:
            value = float(score) if score is not None else math.nan
            self.history.append((t, min(7.0, max(1.0, value)) if math.isfinite(value) else math.nan))

    def draw(self, frame, t, label_scale=1.2, label_thick=3):
        h, w = frame.shape[:2]
        label_width = cv2.getTextSize(
            "Class Avg: 7.00   (n=999)", cv2.FONT_HERSHEY_SIMPLEX,
            label_scale, label_thick,
        )[0][0]
        # 420x203 at Full HD, 840x406 at 4K. Scale the entire graph,
        # including text, padding and line thickness, with the video dimensions.
        scale = min(w / 1920.0, h / 1080.0)
        scaled = lambda value: max(1, int(round(value * scale)))
        margin = scaled(20)
        pw, ph = scaled(420), scaled(203)
        if pw < 100 or ph < 48:
            return
        x, y = 20 + label_width + scaled(30), scaled(10)
        if x + pw > w - margin:
            x, y = margin, max(125, scaled(125))
        if y + ph > h - margin:
            y = h - margin - ph
        if x < 0 or y < 0 or x + pw > w or y + ph > h:
            return
        panel = frame[y:y + ph, x:x + pw]
        cv2.addWeighted(panel, 0.2, np.full_like(panel, 24), 0.8, 0, dst=panel)
        cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), (150, 150, 150), scaled(1))

        def text(s, pos, color=(230, 230, 230)):
            cv2.putText(panel, s, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, scaled(1), cv2.LINE_AA)

        text(f"Class Avg - last {self.window_sec:g}s", (scaled(10), scaled(22)))
        left, right, top, bottom = scaled(35), pw - scaled(14), scaled(38), ph - scaled(32)
        for value in (1, 4, 7):
            py = int(bottom - (value - 1) / 6 * (bottom - top))
            cv2.line(panel, (left, py), (right, py), (65, 65, 65), scaled(1))
            text(str(value), (scaled(10), py + scaled(5)))
        text(f"-{self.window_sec:g}s", (left, ph - scaled(9)))
        text("now", (right - scaled(35), ph - scaled(9)))
        previous = None
        for timestamp, value in self.history:
            if not math.isfinite(value):
                previous = None
                continue
            px = int(right - (t - timestamp) / self.window_sec * (right - left))
            py = int(bottom - (value - 1) / 6 * (bottom - top))
            point = (max(left, min(right, px)), py)
            if previous is not None:
                cv2.line(panel, previous, point, (80, 230, 120), scaled(3), cv2.LINE_AA)
            else:
                cv2.circle(panel, point, scaled(3), (80, 230, 120), -1)
            previous = point
