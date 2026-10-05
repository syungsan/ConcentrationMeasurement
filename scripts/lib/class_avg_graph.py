"""Small OpenCV overlay showing the displayed class average over video time."""
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
        pw, ph = min(300, w - 40), min(145, h - 150)
        if pw < 140 or ph < 90:
            return
        x, y = 20 + label_width + 30, 10
        if x + pw > w - 10:
            x, y = 20, 125
        panel = frame[y:y + ph, x:x + pw]
        cv2.addWeighted(panel, 0.2, np.full_like(panel, 24), 0.8, 0, dst=panel)
        cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), (150, 150, 150), 1)

        def text(s, pos, color=(230, 230, 230)):
            cv2.putText(panel, s, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        text("Class Avg - last 60s", (8, 17))
        left, right, top, bottom = 25, pw - 10, 28, ph - 24
        for value in (1, 4, 7):
            py = int(bottom - (value - 1) / 6 * (bottom - top))
            cv2.line(panel, (left, py), (right, py), (65, 65, 65), 1)
            text(str(value), (8, py + 4))
        text("-60s", (left, ph - 7))
        text("now", (right - 25, ph - 7))
        previous = None
        for timestamp, value in self.history:
            if not math.isfinite(value):
                previous = None
                continue
            px = int(right - (t - timestamp) / self.window_sec * (right - left))
            py = int(bottom - (value - 1) / 6 * (bottom - top))
            point = (max(left, min(right, px)), py)
            if previous is not None:
                cv2.line(panel, previous, point, (80, 230, 120), 2, cv2.LINE_AA)
            else:
                cv2.circle(panel, point, 2, (80, 230, 120), -1)
            previous = point
