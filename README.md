# Multi-Object Detection & Persistent ID Tracking
### Assessment Submission — Predusk Technology Pvt. Ltd. (ProcessVenue)

---

## What This Project Does

This pipeline takes a public sports video, detects all players/subjects in every frame using **YOLOv8**, assigns each a **unique persistent ID** using **ByteTrack**, and outputs an annotated video where every subject is tracked with a consistent numbered label throughout the entire video.

**Example input:** A 3-minute cricket/football match clip from YouTube
**Example output:** Same video with coloured bounding boxes, ID labels like `ID:3 0.89`, and motion trails showing each player's recent path

---

## Source Video

**Video URL:** [PASTE YOUR YOUTUBE LINK HERE]
**Sport/Category:** [Cricket / Football / Other]
**Duration:** ~3–5 minutes
**Why this video:** Multiple moving subjects, real-world challenges (occlusion, camera motion, similar jerseys)

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| Detection | YOLOv8 (Ultralytics) | 8.x |
| Tracking | ByteTrack (built into Ultralytics) | Built-in |
| Video I/O | OpenCV | 4.x |
| Numerics | NumPy | 1.24+ |
| Visualisation | Matplotlib + Seaborn | 3.x / 0.13+ |
| Video download | yt-dlp | 2024+ |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/multi-object-tracker.git
cd multi-object-tracker
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** If you have a GPU, install the CUDA version of torch for much faster processing:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> For CPU only (slower but works everywhere):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## How to Run

### Step 1 — Set your video URL in `config.py`
```python
VIDEO_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Step 2 — Download the video
```bash
python download_video.py
```
This saves the first 5 minutes of the video to `input/source_video.mp4`.

### Step 3 — Run the tracking pipeline
```bash
python detect_track.py
```

That's it. The script will:
- Load YOLOv8n (downloads ~6MB weights automatically on first run)
- Process every frame with detection + tracking
- Show a progress bar in the terminal
- Save the annotated video to `output/annotated_video.mp4`
- Save screenshots every 100 frames to `output/screenshots/`
- Generate analytics charts to `output/analytics/`

### Optional: Use a larger model for better accuracy
```bash
python detect_track.py --model yolov8s.pt   # Small model — better accuracy, slower
python detect_track.py --model yolov8m.pt   # Medium model — best quality, much slower
```

### Optional: Re-generate analytics charts only
```bash
python analytics.py --stats output/analytics/tracking_stats.json --width 1280 --height 720
```

---

## Project Structure

```
multi-object-tracker/
│
├── README.md                    ← You are here
├── requirements.txt             ← All Python dependencies
│
├── config.py                    ← All configuration (paths, thresholds, colours)
├── detect_track.py              ← Main pipeline — run this
├── utils.py                     ← Drawing, colour, video I/O helpers
├── analytics.py                 ← Charts: count over time, heatmap, trajectories
├── download_video.py            ← yt-dlp wrapper for video download
│
├── input/
│   └── source_video.mp4         ← Your downloaded sports video
│
├── output/
│   ├── annotated_video.mp4      ← Final tracked video (main deliverable)
│   ├── screenshots/             ← Key frame PNGs for submission
│   └── analytics/
│       ├── count_over_time.png
│       ├── confidence_distribution.png
│       ├── movement_heatmap.png
│       ├── trajectories.png
│       ├── summary_table.png
│       └── tracking_stats.json
│
├── report/
│   └── technical_report.pdf     ← 1–2 page technical writeup
│
└── demo/
    └── demo_video.mp4           ← 3–5 min screen recording
```

---

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `yolov8n.pt` | Model variant. n=nano (fastest), s=small, m=medium |
| `CONF_THRESHOLD` | `0.25` | Min detection confidence. Lower = more detections, more noise |
| `IOU_THRESHOLD` | `0.70` | NMS threshold. Higher = fewer merged boxes |
| `TRACKER` | `bytetrack.yaml` | Tracking algorithm config |
| `SHOW_TRAJECTORY` | `True` | Draw motion trails behind subjects |
| `TRAJECTORY_LEN` | `30` | How many past positions to show in trail |
| `SCREENSHOT_INTERVAL` | `100` | Save a frame screenshot every N frames |

---

## Architecture Overview

```
YouTube URL
    │
    ▼ yt-dlp
Raw Video (.mp4)
    │
    ▼ OpenCV VideoCapture (frame by frame)
Individual Frames [H × W × 3, BGR numpy array]
    │
    ▼ YOLOv8 inference (conf>0.25, class=person)
Detections [x1, y1, x2, y2, confidence, class=0]
    │
    ▼ ByteTrack (Kalman filter + IoU matching, 2-pass)
Tracked Objects [x1, y1, x2, y2, track_id]
    │
    ▼ utils.py (draw_box + draw_trajectory)
Annotated Frames
    │
    ▼ OpenCV VideoWriter
Annotated Video + Screenshots + Analytics
```

---

## Assumptions

1. **Video quality:** The pipeline is optimised for standard 720p sports footage. Very low resolution or heavily compressed video may reduce detection accuracy.
2. **Subject type:** Configured to detect `class=0` (person) from COCO. Works for all sports with human players. For vehicle tracking, change `CLASSES = [2, 5, 7]` (car, bus, truck).
3. **CPU vs GPU:** The pipeline works on CPU but is significantly slower (~3–5x). For real-time or fast processing, a CUDA-capable GPU is recommended.
4. **Video length:** The downloader limits to 5 minutes maximum. Longer videos can be processed by removing the `download_ranges` limit in `download_video.py`.
5. **Camera motion:** The tracker handles moderate camera panning/zooming. Extreme rapid camera motion may cause temporary ID loss.

---

## Limitations

1. **ID switches:** When players with similar appearance cross paths and remain overlapping for many frames, ByteTrack may swap their IDs. This is a fundamental limitation of motion-only trackers (no appearance Re-ID).
2. **Crowded scenes:** In extremely dense crowds (>20 overlapping players), NMS may suppress some detections, leading to missed tracks.
3. **Re-entry:** If a player leaves the frame and re-enters after `track_buffer` frames (default: 30), they receive a new ID rather than the old one.
4. **Small subjects:** Subjects very far from camera (small bounding boxes, <20px) may be missed or inconsistently detected.
5. **No appearance features:** ByteTrack relies purely on motion prediction. Adding Re-ID embeddings (as in StrongSORT) would significantly reduce ID switches.

---

## Possible Improvements

- **Re-ID model** (e.g., StrongSORT with OSNet): Add appearance features to matching, dramatically reducing ID switches for same-jersey players
- **Camera motion compensation:** Estimate homography between frames to normalise detections before tracking
- **Team clustering:** Use jersey colour or position clustering to group players into teams
- **Speed estimation:** Convert pixel displacement to real-world speed using camera calibration
- **Evaluation metrics:** Implement MOTA, IDF1, HOTA for quantitative tracking quality assessment
- **Real-time mode:** Stream processing with WebSocket output for live events

---

## Author

**Fahad Ahmed**
