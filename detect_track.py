import os
import sys
import argparse
import json
import time

import cv2
import numpy as np

from config import (
    INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH,
    CONF_THRESHOLD, IOU_THRESHOLD, CLASSES, IMG_SIZE,
    TRACKER, PERSIST,
    SHOW_TRAJECTORY, SCREENSHOT_INTERVAL,
    SCREENSHOTS_DIR, ANALYTICS_DIR,
    OUTPUT_CODEC,
)
from utils import (
    TrajectoryStore,
    draw_box, draw_trajectory, draw_frame_info,
    get_video_properties, create_video_writer,
    save_screenshot, get_center, xyxy_to_int,
    print_progress,
)


def process_video(
    input_path:  str = INPUT_VIDEO,
    output_path: str = OUTPUT_VIDEO,
    model_path:  str = MODEL_PATH,
) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input video not found: {input_path}\n"
            f"Run: python download_video.py --url <YOUR_YOUTUBE_URL>"
        )

    print(f"\n{'='*60}")
    print(f"  Multi-Object Detection & Persistent ID Tracking")
    print(f"  Model  : {model_path}")
    print(f"  Tracker: {TRACKER}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"{'='*60}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("Loading model...")
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}\n")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    props = get_video_properties(cap)
    print(f"Video: {props['width']}x{props['height']}  "
          f"{props['fps']:.1f} FPS  "
          f"{props['total_frames']} frames  "
          f"({props['duration_s']:.1f}s)\n")

    writer = create_video_writer(output_path, props, codec=OUTPUT_CODEC)

    trajectories          = TrajectoryStore()
    all_track_ids         = set()
    frame_detection_counts = []
    frame_active_ids      = []
    all_confidences       = []
    id_first_seen         = {}
    id_switches           = 0
    frame_num             = 0
    start_time            = time.time()
    prev_ids              = set()

    print("Processing frames...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        print_progress(frame_num, props["total_frames"])

        results = model.track(
            source=frame,
            persist=PERSIST,
            tracker=TRACKER,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=CLASSES,
            imgsz=IMG_SIZE,
            verbose=False,
        )

        result = results[0]
        current_ids = set()
        detections_this_frame = 0

        if result.boxes is not None and result.boxes.id is not None:
            boxes       = result.boxes.xyxy.cpu().numpy()
            track_ids   = result.boxes.id.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            detections_this_frame = len(track_ids)

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                tid = int(track_id)
                x1, y1, x2, y2 = xyxy_to_int(box)
                cx, cy = get_center(x1, y1, x2, y2)

                trajectories.update(tid, cx, cy)
                current_ids.add(tid)
                all_track_ids.add(tid)
                all_confidences.append(float(conf))

                if tid not in id_first_seen:
                    id_first_seen[tid] = frame_num

                if SHOW_TRAJECTORY:
                    draw_trajectory(frame, tid, trajectories.get(tid))

                draw_box(frame, x1, y1, x2, y2, tid, conf)

        disappeared = prev_ids - current_ids
        id_switches += len(disappeared)
        prev_ids = current_ids.copy()

        draw_frame_info(frame, frame_num, props["total_frames"], len(current_ids))

        frame_detection_counts.append(detections_this_frame)
        frame_active_ids.append(len(current_ids))

        writer.write(frame)

        if frame_num % SCREENSHOT_INTERVAL == 0 or frame_num == 1:
            save_screenshot(frame, frame_num, SCREENSHOTS_DIR)

    cap.release()
    writer.release()

    elapsed      = time.time() - start_time
    fps_achieved = frame_num / elapsed if elapsed > 0 else 0

    stats = {
        "total_frames":             frame_num,
        "total_unique_ids":         len(all_track_ids),
        "total_detections":         sum(frame_detection_counts),
        "avg_detections_per_frame": round(float(np.mean(frame_detection_counts)), 2) if frame_detection_counts else 0,
        "max_simultaneous_ids":     max(frame_active_ids) if frame_active_ids else 0,
        "id_switch_events":         id_switches,
        "avg_confidence":           round(float(np.mean(all_confidences)), 4) if all_confidences else 0,
        "min_confidence":           round(float(np.min(all_confidences)), 4) if all_confidences else 0,
        "max_confidence":           round(float(np.max(all_confidences)), 4) if all_confidences else 0,
        "processing_fps":           round(fps_achieved, 2),
        "processing_time_s":        round(elapsed, 2),
        "video_fps":                round(props["fps"], 2),
        "video_duration_s":         round(props["duration_s"], 2),
        "frame_detection_counts":   frame_detection_counts,
        "frame_active_ids":         frame_active_ids,
        "all_confidences":          all_confidences,
        "id_first_seen":            {str(k): v for k, v in id_first_seen.items()},
        "trajectories":             trajectories.get_all_tracks(),
    }

    return stats


def save_stats(stats: dict, output_dir: str = ANALYTICS_DIR):
    path  = os.path.join(output_dir, "tracking_stats.json")
    clean = {k: v for k, v in stats.items()
             if k not in ("frame_detection_counts", "frame_active_ids",
                          "all_confidences", "trajectories")}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nStats saved: {path}")
    return path


def print_summary(stats: dict):
    print(f"\n{'='*60}")
    print(f"  TRACKING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total frames processed  : {stats['total_frames']}")
    print(f"  Unique track IDs seen   : {stats['total_unique_ids']}")
    print(f"  Total detections        : {stats['total_detections']}")
    print(f"  Avg detections/frame    : {stats['avg_detections_per_frame']}")
    print(f"  Max simultaneous IDs    : {stats['max_simultaneous_ids']}")
    print(f"  ID switch events        : {stats['id_switch_events']}")
    print(f"  Avg confidence          : {stats['avg_confidence']:.4f}")
    print(f"  Processing speed        : {stats['processing_fps']} FPS")
    print(f"  Total processing time   : {stats['processing_time_s']}s")
    print(f"{'='*60}\n")


def main(args):
    stats = process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
    )

    print_summary(stats)
    save_stats(stats)

    print("Generating analytics charts...")
    try:
        from analytics import generate_all_analytics
        generate_all_analytics(stats)
        print("Analytics charts saved to output/analytics/")
    except Exception as e:
        print(f"Analytics generation failed: {e}")

    print(f"\nAnnotated video saved : {args.output}")
    print(f"Screenshots saved     : {SCREENSHOTS_DIR}")
    print(f"Analytics saved       : {ANALYTICS_DIR}")
    print("\nPipeline complete. Ready for submission.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Persistent ID Tracking Pipeline"
    )
    parser.add_argument("--input",  type=str, default=INPUT_VIDEO)
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO)
    parser.add_argument("--model",  type=str, default=MODEL_PATH)
    args = parser.parse_args()
    main(args)
