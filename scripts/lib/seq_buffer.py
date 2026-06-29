# lib/seq_buffer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal, Tuple

import torch

Mode = Literal["image", "skeleton", "fusion"]


@dataclass
class TrackBuffer:
    mode: Mode
    T: int
    frames: List[torch.Tensor] = field(default_factory=list)   # each [3,H,W]
    poses: List[torch.Tensor] = field(default_factory=list)    # each [K,3]
    last_seen_t: float = 0.0

    def push(self, *, frame_tensor: Optional[torch.Tensor], pose_tensor: Optional[torch.Tensor], t: float):
        self.last_seen_t = float(t)
        if self.mode in ("image", "fusion"):
            assert frame_tensor is not None
            self.frames.append(frame_tensor)
            if len(self.frames) > self.T:
                self.frames = self.frames[-self.T:]
        if self.mode in ("skeleton", "fusion"):
            assert pose_tensor is not None
            self.poses.append(pose_tensor)
            if len(self.poses) > self.T:
                self.poses = self.poses[-self.T:]

    def ready(self) -> bool:
        if self.mode == "image":
            return len(self.frames) >= self.T
        if self.mode == "skeleton":
            return len(self.poses) >= self.T
        return len(self.frames) >= self.T and len(self.poses) >= self.T

    def get_batch(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        frames = None
        poses = None
        if self.mode in ("image", "fusion"):
            frames = torch.stack(self.frames[-self.T:], dim=0).unsqueeze(0)  # [1,T,3,H,W]
        if self.mode in ("skeleton", "fusion"):
            poses = torch.stack(self.poses[-self.T:], dim=0).unsqueeze(0)    # [1,T,K,3]
        return frames, poses


class MultiTrackBuffer:
    def __init__(self, mode: Mode, T: int, ttl_sec: float = 3.0):
        self.mode = mode
        self.T = T
        self.ttl_sec = float(ttl_sec)
        self.buf: Dict[int, TrackBuffer] = {}

    def get(self, track_id: int) -> TrackBuffer:
        if track_id not in self.buf:
            self.buf[track_id] = TrackBuffer(mode=self.mode, T=self.T)
        return self.buf[track_id]

    def gc(self, now_t: float):
        dead = [tid for tid, b in self.buf.items() if (now_t - b.last_seen_t) > self.ttl_sec]
        for tid in dead:
            del self.buf[tid]
