"""Scene trajectory and lightweight map-related plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nusc_lab.dataset import iter_scene_samples


def ego_trajectory_xy(nusc, scene: dict, channel: str = "LIDAR_TOP") -> np.ndarray:
    """Return scene ego positions in the log's global map xy frame.

    Each point is read from the ego_pose record attached to the requested
    keyframe sample_data channel. LIDAR_TOP is the default reference because
    nuScenes samples are annotated around the keyframe lidar sweep.
    """
    xy = []
    for sample in iter_scene_samples(nusc, scene):
        sd = nusc.get("sample_data", sample["data"][channel])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        xy.append(ego["translation"][:2])
    return np.asarray(xy, dtype=float)


def plot_ego_trajectory(nusc, scene: dict, save_path: str | Path | None = None):
    """Plot a scene ego trajectory in global map xy coordinates."""
    xy = ego_trajectory_xy(nusc, scene)
    fig, ax = plt.subplots(figsize=(8, 8))
    if xy.size:
        ax.plot(xy[:, 0], xy[:, 1], "-o", markersize=2)
        ax.scatter(xy[0, 0], xy[0, 1], c="green", label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], c="red", label="end")
    ax.set_title(f"{scene['name']} ego trajectory (global map frame)")
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, ax
