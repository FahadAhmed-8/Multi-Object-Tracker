import os
import cv2
import numpy as np
from collections import defaultdict, deque
from config import TRACK_COLORS, BOX_THICKNESS, LABEL_FONT_SCALE, LABEL_THICKNESS, TRAJECTORY_LEN


def get_color(track_id: int) -> tuple:
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


class TrajectoryStore:
    def __init__(self, max_len: int = TRAJECTORY_LEN):
        self.max_len = max_len
        self._store: dict = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, track_id: int, cx: int, cy: int):
        self._store[track_id].append((cx, cy))

    def get(self, track_id: int) -> list:
        return list(self._store[track_id])

    def all_positions(self) -> list:
        all_pts = []
        for positions in self._store.values():
            all_pts.extend(positions)
        return all_pts

    def get_all_tracks(self) -> dict:
        return {k: list(v) for k, v in self._store.items()}


def draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             track_id: int, conf: float = None) -> np.ndarray:
    color = get_color(track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    label = f"ID:{track_id}  {conf:.2f}" if conf is not None else f"ID:{track_id}"

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
    )

    label_y = max(y1 - 5, text_h + 5)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y + 2),
        color,
        thickness=cv2.FILLED
    )
    cv2.putText(
        frame, label,
        (x1 + 2, label_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        (255, 255, 255),
        LABEL_THICKNESS,
        cv2.LINE_AA
    )
    return frame


def draw_trajectory(frame: np.ndarray, track_id: int, positions: list) -> np.ndarray:
    if len(positions) < 2:
        return frame

    color = get_color(track_id)
    n = len(positions)

    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(3 * alpha))
        faded_color = tuple(int(c * alpha) for c in color)
        cv2.line(frame, positions[i - 1], positions[i], faded_color, thickness, cv2.LINE_AA)

    return frame


def draw_frame_info(frame: np.ndarray, frame_num: int,
                    total_frames: int, active_ids: int) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (230, 68), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active IDs: {active_ids}",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
    return frame


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / fps if fps > 0 else 0
    return {"width": w, "height": h, "fps": fps, "total_frames": total, "duration_s": dur}


def create_video_writer(output_path: str, props: dict, codec: str = "mp4v") -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    fps    = props["fps"] if props["fps"] > 0 else 30.0
    size   = (props["width"], props["height"])
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    if not writer.isOpened():
        print("mp4v codec failed. Trying XVID fallback...")
        output_path = output_path.replace(".mp4", "_fallback.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        if not writer.isOpened():
            raise RuntimeError("Could not open VideoWriter with mp4v or XVID codec.")
        print(f"Saving as AVI: {output_path}")

    return writer


def save_screenshot(frame: np.ndarray, frame_num: int, output_dir: str):
    path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
    cv2.imwrite(path, frame)
    return path


def get_center(x1: int, y1: int, x2: int, y2: int) -> tuple:
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def xyxy_to_int(box) -> tuple:
    return int(box[0]), int(box[1]), int(box[2]), int(box[3])


def print_progress(frame_num: int, total: int, prefix: str = "Processing"):
    pct    = frame_num / max(total, 1)
    bar_w  = 40
    filled = int(bar_w * pct)
    bar    = "\u2588" * filled + "\u2591" * (bar_w - filled)
    print(f"\r{prefix}: [{bar}] {pct*100:.1f}%  frame {frame_num}/{total}", end="", flush=True)
    if frame_num >= total:
        print()
