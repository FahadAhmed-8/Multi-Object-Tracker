import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from config import ANALYTICS_DIR, CHART_DPI, TRACK_COLORS


def bgr_to_rgb_norm(bgr: tuple) -> tuple:
    return (bgr[2]/255, bgr[1]/255, bgr[0]/255)


def plot_count_over_time(frame_detection_counts: list,
                         frame_active_ids: list,
                         output_dir: str = ANALYTICS_DIR):
    frames = range(1, len(frame_detection_counts) + 1)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(frames, frame_detection_counts, color="#2196F3", linewidth=1.5,
            alpha=0.85, label="Detections per frame")
    ax.plot(frames, frame_active_ids, color="#4CAF50", linewidth=2,
            label="Active track IDs per frame")
    ax.fill_between(frames, frame_detection_counts, alpha=0.15, color="#2196F3")
    ax.fill_between(frames, frame_active_ids, alpha=0.2, color="#4CAF50")

    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Object Count Over Time", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(output_dir, "count_over_time.png")
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_confidence_distribution(all_confidences: list,
                                  output_dir: str = ANALYTICS_DIR):
    if not all_confidences:
        print("  No confidence data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(
        all_confidences, bins=40, color="#FF6B6B", edgecolor="white",
        linewidth=0.5, alpha=0.85
    )
    for patch, left_edge in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.RdYlGn(left_edge))

    mean_conf = np.mean(all_confidences)
    ax.axvline(mean_conf, color="#333333", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_conf:.3f}")

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Number of Detections", fontsize=12)
    ax.set_title("Detection Confidence Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(output_dir, "confidence_distribution.png")
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def generate_heatmap(trajectories: dict,
                     frame_width: int,
                     frame_height: int,
                     output_dir: str = ANALYTICS_DIR):
    if not trajectories:
        print("  No trajectory data to heatmap.")
        return None

    all_x, all_y = [], []
    for positions in trajectories.values():
        for cx, cy in positions:
            all_x.append(cx)
            all_y.append(cy)

    if not all_x:
        return None

    bins_x = min(100, frame_width  // 8)
    bins_y = min(100, frame_height // 8)

    heatmap, xedges, yedges = np.histogram2d(
        all_x, all_y,
        bins=[bins_x, bins_y],
        range=[[0, frame_width], [0, frame_height]]
    )

    from scipy.ndimage import gaussian_filter
    heatmap = gaussian_filter(heatmap.T, sigma=2)

    cmap = LinearSegmentedColormap.from_list(
        "tracking_heat",
        ["#0a1628", "#0d47a1", "#1976d2", "#43a047", "#ffeb3b", "#ff5722", "#b71c1c"]
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        heatmap,
        extent=[0, frame_width, frame_height, 0],
        aspect="equal",
        cmap=cmap,
        interpolation="bilinear",
        alpha=0.9,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Presence density")
    ax.set_xlabel("X position (pixels)", fontsize=11)
    ax.set_ylabel("Y position (pixels)", fontsize=11)
    ax.set_title("Subject Movement Heatmap", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    path = os.path.join(output_dir, "movement_heatmap.png")
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_trajectories(trajectories: dict,
                      frame_width: int,
                      frame_height: int,
                      output_dir: str = ANALYTICS_DIR,
                      max_ids: int = 20):
    if not trajectories:
        print("  No trajectory data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ids_to_plot    = list(trajectories.keys())[:max_ids]
    legend_handles = []

    for i, (tid, positions) in enumerate(trajectories.items()):
        if tid not in ids_to_plot or len(positions) < 2:
            continue

        rgb = bgr_to_rgb_norm(TRACK_COLORS[int(tid) % len(TRACK_COLORS)])
        xs  = [p[0] for p in positions]
        ys  = [p[1] for p in positions]

        ax.plot(xs, ys, color=rgb, linewidth=1.5, alpha=0.8)
        ax.plot(xs[-1], ys[-1], "o", color=rgb, markersize=5, alpha=1.0)
        ax.plot(xs[0],  ys[0],  "s", color=rgb, markersize=4, alpha=0.6)
        legend_handles.append(mpatches.Patch(color=rgb, label=f"ID {tid}"))

    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)
    ax.set_xlabel("X position (pixels)", color="white", fontsize=11)
    ax.set_ylabel("Y position (pixels)", color="white", fontsize=11)
    ax.set_title("Movement Trajectories (o = last position, s = start)",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, ncol=2,
                  facecolor="#1c2333", edgecolor="#444444",
                  labelcolor="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "trajectories.png")
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_summary_table(stats: dict, output_dir: str = ANALYTICS_DIR):
    rows = [
        ["Metric", "Value"],
        ["Total frames processed",   str(stats.get("total_frames", "-"))],
        ["Unique track IDs seen",    str(stats.get("total_unique_ids", "-"))],
        ["Total detections",         str(stats.get("total_detections", "-"))],
        ["Avg detections / frame",   str(stats.get("avg_detections_per_frame", "-"))],
        ["Max simultaneous IDs",     str(stats.get("max_simultaneous_ids", "-"))],
        ["ID switch events",         str(stats.get("id_switch_events", "-"))],
        ["Avg confidence score",     f"{stats.get('avg_confidence', 0):.4f}"],
        ["Processing speed (FPS)",   str(stats.get("processing_fps", "-"))],
        ["Total processing time",    f"{stats.get('processing_time_s', '-')}s"],
        ["Video duration",           f"{stats.get('video_duration_s', '-')}s"],
    ]

    fig, ax = plt.subplots(figsize=(8, len(rows) * 0.5 + 1))
    ax.axis("off")

    colors = [["#1B3A5C", "#1B3A5C"]] + \
             [["#EBF5FB", "#FFFFFF"] if i % 2 == 0 else ["#D6EAF8", "#EBF5FB"]
              for i in range(len(rows) - 1)]

    table = ax.table(cellText=rows, cellLoc="left", loc="center", cellColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for j in range(2):
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Tracking Summary Statistics",
                 fontsize=14, fontweight="bold", pad=15, color="#1B3A5C")

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def generate_all_analytics(stats: dict, output_dir: str = ANALYTICS_DIR):
    print(f"\n{'='*50}")
    print("  Generating Analytics Charts")
    print(f"{'='*50}")

    frame_detection_counts = stats.get("frame_detection_counts", [])
    frame_active_ids       = stats.get("frame_active_ids", [])
    all_confidences        = stats.get("all_confidences", [])
    trajectories           = stats.get("trajectories", {})

    fw = stats.get("frame_width", 1280)
    fh = stats.get("frame_height", 720)

    if frame_detection_counts:
        plot_count_over_time(frame_detection_counts, frame_active_ids, output_dir)
    if all_confidences:
        plot_confidence_distribution(all_confidences, output_dir)
    if trajectories:
        generate_heatmap(trajectories, fw, fh, output_dir)
        plot_trajectories(trajectories, fw, fh, output_dir)

    plot_summary_table(stats, output_dir)
    print(f"\nAll analytics saved to: {output_dir}")


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Re-generate analytics charts from a saved tracking_stats.json"
    )
    parser.add_argument("--stats",  type=str,
                        default=os.path.join(ANALYTICS_DIR, "tracking_stats.json"))
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not os.path.exists(args.stats):
        print(f"ERROR: Stats file not found: {args.stats}")
        sys.exit(1)

    with open(args.stats) as f:
        stats = json.load(f)

    stats["frame_width"]  = args.width
    stats["frame_height"] = args.height
    generate_all_analytics(stats)
