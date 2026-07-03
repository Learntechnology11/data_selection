"""Video-style scene visualization with six cameras and a LiDAR BEV panel."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nusc_lab.annotations import filter_annotation_tokens
from nusc_lab.schema import CAMERA_CHANNELS, raw_category_to_detection
from nusc_lab.sensors import load_image, load_lidar_points_from_sample, sensor_file_path
from nusc_lab.utils.colors import CATEGORY_COLORS, DEFAULT_BOX_COLOR
from nusc_lab.visualization.bev import boxes_in_sample_data_frame, box_footprint_xy


VIDEO_CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _font(size: int = 14):
    """Return a portable default font."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _rgb255(color) -> tuple[int, int, int]:
    """Convert matplotlib-style float RGB to uint8 RGB."""
    return tuple(int(max(0, min(1, c)) * 255) for c in color)


def _fit_cover(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    """Resize and center-crop an image to fill size.

    Returns the cropped image plus the transform parameters needed to map
    original image pixels into the cropped tile: display = original * scale -
    crop_offset.
    """
    target_w, target_h = size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((int(round(src_w * scale)), int(round(src_h * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h)), scale, left, top


def _draw_panel_header(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    """Draw a dark header bar for a camera or BEV panel."""
    x0, y0, x1, y1 = box
    header_h = min(26, max(18, (y1 - y0) // 10))
    draw.rectangle((x0, y0, x1, y0 + header_h), fill=(30, 34, 38))
    draw.text((x0 + 8, y0 + 5), title, fill=(220, 224, 228), font=_font(13))


def _project_camera_box(corners: np.ndarray, intrinsic) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame box corners to image pixels."""
    K = np.asarray(intrinsic, dtype=float)
    uvw = (K @ corners).T
    depth = corners[2, :]
    uv = np.zeros((corners.shape[1], 2), dtype=float)
    valid = depth > 1e-6
    uv[valid, 0] = uvw[valid, 0] / depth[valid]
    uv[valid, 1] = uvw[valid, 1] / depth[valid]
    return uv, valid


def _draw_camera_boxes(
    image: Image.Image,
    nusc,
    sample: dict,
    channel: str,
    class_filter: Iterable[str] | None,
    scale: float,
    crop_left: int,
    crop_top: int,
) -> Image.Image:
    """Draw projected 3D boxes onto a camera image."""
    if not sample.get("anns"):
        return image
    sd_token = sample["data"][channel]
    selected = filter_annotation_tokens(nusc, sample, class_filter)
    if not selected:
        return image
    _data_path, boxes, intrinsic = nusc.get_sample_data(sd_token, selected_anntokens=selected)
    draw = ImageDraw.Draw(image)

    for box in boxes:
        corners = box.corners()
        uv, valid = _project_camera_box(corners, intrinsic)
        det_name = raw_category_to_detection(box.name)
        color = _rgb255(CATEGORY_COLORS.get(det_name or box.name, DEFAULT_BOX_COLOR))
        scaled = np.column_stack([uv[:, 0] * scale - crop_left, uv[:, 1] * scale - crop_top])
        for i0, i1 in BOX_EDGES:
            if not (valid[i0] and valid[i1]):
                continue
            p0 = tuple(scaled[i0])
            p1 = tuple(scaled[i1])
            draw.line((p0, p1), fill=color, width=2)
        center = np.nanmean(scaled[valid], axis=0) if np.any(valid) else None
        if center is not None and np.all(np.isfinite(center)):
            label = det_name or box.name.split(".")[-1]
            draw.text((float(center[0]), float(center[1])), label, fill=color, font=_font(12))
    return image


def render_camera_tile(
    nusc,
    sample: dict,
    channel: str,
    size: tuple[int, int],
    with_boxes: bool = True,
    class_filter: Iterable[str] | None = None,
) -> Image.Image:
    """Render one cropped camera tile with an optional annotation overlay."""
    if channel not in CAMERA_CHANNELS:
        raise ValueError(f"Unknown camera channel {channel!r}.")
    sd = nusc.get("sample_data", sample["data"][channel])
    image = load_image(sensor_file_path(nusc, sd))
    image, scale, crop_left, crop_top = _fit_cover(image, size)
    if with_boxes:
        image = _draw_camera_boxes(image, nusc, sample, channel, class_filter, scale, crop_left, crop_top)
    draw = ImageDraw.Draw(image)
    _draw_panel_header(draw, (0, 0, image.width, image.height), f"/{channel}/image_rect_compressed")
    return image


def _bev_pixel_mapper(
    width: int,
    height: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    margin: int = 36,
):
    """Return an equal-scale mapper from reference-frame xy to BEV pixels.

    x points upward in the rendered tile and y-left points left, matching the
    standalone LiDAR BEV view. A single meters-to-pixels scale is used for both
    axes so the bottom video panel does not stretch the BEV horizontally.
    """
    usable_w = max(1, width - 2 * margin)
    usable_h = max(1, height - 2 * margin)
    xmin, xmax = xlim
    ymin, ymax = ylim
    xspan = xmax - xmin
    yspan = ymax - ymin
    if xspan <= 0 or yspan <= 0:
        raise ValueError(f"Invalid BEV limits: xlim={xlim}, ylim={ylim}")
    scale = min(usable_w / yspan, usable_h / xspan)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    u_center = width / 2
    v_center = height / 2

    def map_xy(xy: np.ndarray) -> np.ndarray:
        arr = np.asarray(xy, dtype=float)
        u = u_center - (arr[:, 1] - y_center) * scale
        v = v_center - (arr[:, 0] - x_center) * scale
        return np.column_stack([u, v])

    return map_xy


def _lidar_colors(points: np.ndarray, mode: str = "distance") -> np.ndarray:
    """Return uint8 colors for LiDAR points."""
    if points.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if mode == "intensity" and points.shape[1] > 3:
        values = points[:, 3].astype(float)
    else:
        values = np.linalg.norm(points[:, :2], axis=1)
    lo, hi = np.nanpercentile(values, [2, 98]) if len(values) > 3 else (values.min(), values.max())
    denom = max(1e-6, hi - lo)
    t = np.clip((values - lo) / denom, 0, 1)
    colors = np.empty((len(points), 3), dtype=np.uint8)
    colors[:, 0] = (30 + 70 * t).astype(np.uint8)
    colors[:, 1] = (80 + 175 * t).astype(np.uint8)
    colors[:, 2] = (230 - 130 * t).astype(np.uint8)
    return colors


def render_lidar_bev_tile(
    nusc,
    sample: dict,
    size: tuple[int, int],
    nsweeps: int = 1,
    with_boxes: bool = True,
    class_filter: Iterable[str] | None = None,
    xlim: tuple[float, float] = (-35.0, 65.0),
    ylim: tuple[float, float] = (-55.0, 55.0),
    color_by: str = "distance",
    max_points: int = 120000,
) -> Image.Image:
    """Render a black-background LiDAR BEV tile with annotation boxes."""
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (8, 10, 14)
    points = load_lidar_points_from_sample(nusc, sample, nsweeps=nsweeps)
    if points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max_points))
        points = points[::step]

    map_xy = _bev_pixel_mapper(width, height, xlim, ylim)
    xmin, xmax = xlim
    ymin, ymax = ylim
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    kept = points[mask]
    if kept.size:
        pix = np.rint(map_xy(kept[:, :2])).astype(int)
        inside = (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
        pix = pix[inside]
        colors = _lidar_colors(kept[inside], mode=color_by)
        image[pix[:, 1], pix[:, 0]] = colors

    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)

    # Draw range rings and reference-frame origin after points so orientation stays legible.
    for radius in [10, 20, 30, 40, 50]:
        ring = []
        theta = np.linspace(0, 2 * np.pi, 180)
        xy = np.column_stack([np.cos(theta) * radius, np.sin(theta) * radius])
        valid = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
        if np.any(valid):
            ring = [tuple(p) for p in map_xy(xy[valid])]
        if len(ring) > 2:
            draw.line(ring, fill=(28, 58, 92), width=1)

    ego = map_xy(np.asarray([[0.0, 0.0]]))[0]
    draw.ellipse((ego[0] - 5, ego[1] - 5, ego[0] + 5, ego[1] + 5), fill=(35, 120, 255))
    draw.line((ego[0], ego[1], ego[0], ego[1] - 24), fill=(60, 160, 255), width=2)

    if with_boxes and sample.get("anns"):
        for box in boxes_in_sample_data_frame(nusc, sample, ref_channel="LIDAR_TOP", class_filter=class_filter):
            corners = box_footprint_xy(box)
            pix = map_xy(corners)
            det_name = raw_category_to_detection(box.name)
            color = _rgb255(CATEGORY_COLORS.get(det_name or box.name, DEFAULT_BOX_COLOR))
            pts = [tuple(p) for p in pix]
            draw.line(pts, fill=color, width=3)
            draw.text(pts[0], det_name or box.name.split(".")[-1], fill=color, font=_font(12))

    _draw_panel_header(draw, (0, 0, width, height), "3D / LIDAR_TOP BEV + sample_annotation")
    return canvas


def render_scene_video_frame(
    nusc,
    sample: dict,
    frame_index: int,
    scene_name: str,
    canvas_size: tuple[int, int] = (1920, 1080),
    top_ratio: float = 0.5,
    camera_order: Sequence[str] = VIDEO_CAMERA_ORDER,
    nsweeps: int = 1,
    with_boxes: bool = True,
    class_filter: Iterable[str] | None = None,
    xlim: tuple[float, float] = (-35.0, 65.0),
    ylim: tuple[float, float] = (-55.0, 55.0),
    color_by: str = "distance",
) -> Image.Image:
    """Compose one video frame: six cameras on top and LiDAR BEV at bottom."""
    width, height = canvas_size
    top_h = int(round(height * top_ratio))
    top_h -= top_h % 2
    bev_h = height - top_h
    cell_w = width // 3
    cell_h = top_h // 2
    canvas = Image.new("RGB", (width, height), (5, 6, 8))

    for idx, channel in enumerate(camera_order):
        row, col = divmod(idx, 3)
        tile = render_camera_tile(
            nusc,
            sample,
            channel,
            size=(cell_w, cell_h),
            with_boxes=with_boxes,
            class_filter=class_filter,
        )
        canvas.paste(tile, (col * cell_w, row * cell_h))

    bev = render_lidar_bev_tile(
        nusc,
        sample,
        size=(width, bev_h),
        nsweeps=nsweeps,
        with_boxes=with_boxes,
        class_filter=class_filter,
        xlim=xlim,
        ylim=ylim,
        color_by=color_by,
    )
    canvas.paste(bev, (0, top_h))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width - 360, top_h + 8),
        f"{scene_name}  frame {frame_index:03d}  token {sample['token'][:8]}",
        fill=(210, 214, 220),
        font=_font(14),
    )
    return canvas


def save_preview_frame(path: str | Path, image: Image.Image) -> Path:
    """Save a still preview frame."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out
