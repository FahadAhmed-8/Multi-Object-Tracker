import argparse
import os
import sys

try:
    import yt_dlp
except ImportError:
    print("ERROR: yt-dlp not installed. Run: pip install yt-dlp")
    sys.exit(1)

from config import VIDEO_URL, INPUT_VIDEO, INPUT_DIR


def download(url: str, output_path: str = INPUT_VIDEO) -> str:
    if not url:
        raise ValueError(
            "No URL provided. Either pass --url argument or set VIDEO_URL in config.py"
        )

    print(f"\n{'='*60}")
    print(f"  Downloading video from YouTube")
    print(f"  URL    : {url}")
    print(f"  Saving : {output_path}")
    print(f"{'='*60}\n")

    ydl_opts = {
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title    = info.get("title", "Unknown")
        duration = info.get("duration", 0)
        print(f"\nDownloaded: '{title}'")
        print(f"Duration  : {duration}s")
        print(f"Saved to  : {output_path}")

    return output_path


def check_video(path: str) -> bool:
    import cv2
    if not os.path.exists(path):
        return False
    cap = cv2.VideoCapture(path)
    ok  = cap.isOpened()
    if ok:
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nVideo info:")
        print(f"  Resolution : {w}x{h}")
        print(f"  FPS        : {fps:.1f}")
        print(f"  Frames     : {frames}")
        print(f"  Duration   : {frames/fps:.1f}s")
    cap.release()
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a YouTube sports video for the tracking pipeline."
    )
    parser.add_argument("--url",    type=str, default=VIDEO_URL)
    parser.add_argument("--output", type=str, default=INPUT_VIDEO)
    args = parser.parse_args()

    downloaded = download(url=args.url, output_path=args.output)
    check_video(downloaded)
