import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR       = os.path.join(BASE_DIR, "input")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")
ANALYTICS_DIR   = os.path.join(OUTPUT_DIR, "analytics")
REPORT_DIR      = os.path.join(BASE_DIR, "report")
DEMO_DIR        = os.path.join(BASE_DIR, "demo")

for _d in [INPUT_DIR, OUTPUT_DIR, SCREENSHOTS_DIR, ANALYTICS_DIR, REPORT_DIR, DEMO_DIR]:
    os.makedirs(_d, exist_ok=True)

VIDEO_URL       = "https://youtu.be/LbyMhRQy5eE?si=Or3gjRB1SastD9XI"
INPUT_VIDEO     = os.path.join(INPUT_DIR, "source_video.mp4")
OUTPUT_VIDEO    = os.path.join(OUTPUT_DIR, "annotated_video.mp4")

MODEL_PATH      = "yolov8n.pt"

CONF_THRESHOLD  = 0.25
IOU_THRESHOLD   = 0.70
CLASSES         = [0]
IMG_SIZE        = 640

TRACKER         = "bytetrack.yaml"
PERSIST         = True

SHOW_TRAJECTORY     = True
TRAJECTORY_LEN      = 30
BOX_THICKNESS       = 2
LABEL_FONT_SCALE    = 0.6
LABEL_THICKNESS     = 2
SCREENSHOT_INTERVAL = 100

HEATMAP_ALPHA   = 0.6
CHART_DPI       = 150

OUTPUT_FPS      = None
OUTPUT_CODEC    = "mp4v"

TRACK_COLORS = [
    (255,  56,  56),
    ( 56, 255,  56),
    ( 56,  56, 255),
    (255, 157,  56),
    (255, 255,  56),
    (255,  56, 255),
    ( 56, 255, 255),
    (255, 150, 200),
    (200, 200,   0),
    (  0, 200, 200),
    (200,   0, 200),
    (100, 200, 100),
    (100, 100, 200),
    (200, 100, 100),
    (150, 255, 150),
    (255, 150, 100),
    (100, 255, 200),
    (200, 255, 100),
    (255, 200, 100),
    (100, 200, 255),
]
