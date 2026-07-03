from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from nusc_lab.config import add_common_args, config_from_args, resolve_output_path
from nusc_lab.dataset import NuScenesLoadError, get_scene, iter_scene_samples, load_nuscenes
from nusc_lab.visualization.video import VIDEO_CAMERA_ORDER, render_scene_video_frame, save_preview_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a scene-level video with six camera views and LiDAR BEV annotations."
    )
    add_common_args(parser)
    parser.add_argument("--scene-name", default=None, help="Scene name, e.g. scene-0061.")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index when --scene-name is omitted.")
    parser.add_argument("--out", default=None, help="Output video path. Default: outputs/<scene>_sixcam_lidar_bev.mp4")
    parser.add_argument("--preview-out", default=None, help="Optional first-frame PNG preview path.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS. nuScenes keyframes are 2Hz.")
    parser.add_argument("--width", type=int, default=1920, help="Output video width.")
    parser.add_argument("--height", type=int, default=1080, help="Output video height.")
    parser.add_argument("--top-ratio", type=float, default=0.5, help="Canvas fraction used by the 2x3 camera grid.")
    parser.add_argument("--nsweeps", type=int, default=1, help="Number of LiDAR sweeps per frame.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip scene samples before this index.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick checks.")
    parser.add_argument("--with-boxes", action="store_true", default=True, help="Draw camera and BEV labels by default.")
    parser.add_argument("--no-boxes", dest="with_boxes", action="store_false", help="Disable annotation boxes.")
    parser.add_argument("--classes", nargs="*", default=None, help="Raw or detection classes to draw, e.g. car pedestrian.")
    parser.add_argument("--color-by", choices=["distance", "intensity"], default="distance", help="LiDAR point coloring.")
    parser.add_argument("--bev-xlim", nargs=2, type=float, default=(-35.0, 65.0), metavar=("BACK", "FRONT"))
    parser.add_argument("--bev-ylim", nargs=2, type=float, default=(-55.0, 55.0), metavar=("RIGHT", "LEFT"))
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, e.g. mp4v or XVID.")
    parser.add_argument(
        "--camera-order",
        nargs=6,
        default=VIDEO_CAMERA_ORDER,
        help="Six camera channels in display order.",
    )
    return parser.parse_args()


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for video writing. Your server environment usually already has it; "
            "otherwise install only opencv-python, not the full requirements file."
        ) from exc
    return cv2


def _sample_window(nusc, scene: dict, start_index: int, max_frames: int | None):
    for idx, sample in enumerate(iter_scene_samples(nusc, scene)):
        if idx < start_index:
            continue
        out_idx = idx - start_index
        if max_frames is not None and out_idx >= max_frames:
            break
        yield idx, sample


def main() -> int:
    args = parse_args()
    cfg = config_from_args(args)
    if args.width <= 0 or args.height <= 0:
        print("ERROR: --width and --height must be positive.", file=sys.stderr)
        return 2
    if not (0.25 <= args.top_ratio <= 0.8):
        print("ERROR: --top-ratio should be between 0.25 and 0.8.", file=sys.stderr)
        return 2

    try:
        nusc = load_nuscenes(cfg.dataroot, cfg.version, verbose=cfg.verbose)
        scene = get_scene(nusc, args.scene_name, args.scene_index)
    except (NuScenesLoadError, KeyError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cv2 = _load_cv2()
    out = resolve_output_path(
        cfg.output_dir,
        args.out,
        f"{scene['name']}_sixcam_lidar_bev.mp4",
    )
    preview_out = (
        resolve_output_path(cfg.output_dir, args.preview_out, f"{scene['name']}_sixcam_lidar_bev_preview.png")
        if args.preview_out
        else None
    )

    fourcc = cv2.VideoWriter_fourcc(*args.codec[:4])
    writer = cv2.VideoWriter(str(out), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        print(f"ERROR: Could not open video writer for {out}. Try --codec XVID --out outputs/video.avi", file=sys.stderr)
        return 2

    written = 0
    try:
        for frame_idx, sample in _sample_window(nusc, scene, args.start_index, args.max_frames):
            image = render_scene_video_frame(
                nusc,
                sample,
                frame_index=frame_idx,
                scene_name=scene["name"],
                canvas_size=(args.width, args.height),
                top_ratio=args.top_ratio,
                camera_order=args.camera_order,
                nsweeps=args.nsweeps,
                with_boxes=args.with_boxes,
                class_filter=args.classes,
                xlim=tuple(args.bev_xlim),
                ylim=tuple(args.bev_ylim),
                color_by=args.color_by,
            )
            if written == 0 and preview_out is not None:
                save_preview_frame(preview_out, image)
            frame_rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            writer.write(frame_rgb)
            written += 1
            print(f"wrote frame {written:03d} from scene sample index {frame_idx:03d}")
    finally:
        writer.release()

    if written == 0:
        print("ERROR: No frames were written. Check --start-index and --max-frames.", file=sys.stderr)
        return 2
    print(f"Wrote video: {out}")
    if preview_out:
        print(f"Wrote preview: {preview_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
