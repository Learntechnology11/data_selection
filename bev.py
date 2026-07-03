"""Bird's-eye-view plotting primitives."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from nusc_lab.schema import raw_category_to_detection
from nusc_lab.utils.colors import CATEGORY_COLORS, DEFAULT_BOX_COLOR


def setup_bev_axis(ax=None, xlim=(-50, 50), ylim=(-50, 50), title: str = "BEV"):
    """Create or configure a BEV axis with the reference frame at the origin."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.scatter([0], [0], marker="x", c="red", s=40, label="origin")
    return ax


def _selected_annotation_tokens(
    nusc,
    sample: dict,
    class_filter: Optional[Iterable[str]] = None,
) -> list[str]:
    """Return annotation tokens matching raw or detection class filters."""
    filters = {item.strip() for item in class_filter or [] if item.strip()}
    selected = []
    for ann_token in sample.get("anns", []):
        ann = nusc.get("sample_annotation", ann_token)
        det_name = raw_category_to_detection(ann["category_name"])
        if filters and ann["category_name"] not in filters and det_name not in filters:
            continue
        selected.append(ann_token)
    return selected


def boxes_in_sample_data_frame(
    nusc,
    sample: dict,
    ref_channel: str = "LIDAR_TOP",
    class_filter: Optional[Iterable[str]] = None,
):
    """Return annotation boxes transformed into the reference sample_data frame.

    nuScenes sample annotations are stored in the global frame. The official
    devkit's get_sample_data() applies the same global -> ego -> sensor chain
    used by camera projection, so BEV boxes align with raw LIDAR_TOP points.
    """
    selected = _selected_annotation_tokens(nusc, sample, class_filter)
    if not selected:
        return []
    ref_token = sample["data"][ref_channel]
    _data_path, boxes, _camera_intrinsic = nusc.get_sample_data(ref_token, selected_anntokens=selected)
    return boxes


def box_footprint_xy(box) -> np.ndarray:
    """Return a closed N x 2 BEV footprint from a nuScenes devkit Box."""
    corners = box.bottom_corners()[:2, :].T
    return np.vstack([corners, corners[0]])


def plot_boxes_bev(
    nusc,
    sample: dict,
    ax,
    ref_channel: str = "LIDAR_TOP",
    class_filter: Optional[Iterable[str]] = None,
) -> None:
    """Draw sample annotations in the same sensor frame as the BEV points."""
    for box in boxes_in_sample_data_frame(nusc, sample, ref_channel=ref_channel, class_filter=class_filter):
        corners = box_footprint_xy(box)
        det_name = raw_category_to_detection(box.name)
        color = CATEGORY_COLORS.get(det_name or box.name, DEFAULT_BOX_COLOR)
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.2)
        ax.text(corners[0, 0], corners[0, 1], det_name or box.name.split(".")[-1], fontsize=6)


def save_figure(fig, save_path) -> None:
    """Save a figure with tight layout."""
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
