from __future__ import annotations

import numpy as np

from nusc_lab.visualization.maps import ego_trajectory_xy


class _FakeNuScenes:
    def __init__(self):
        self.samples = {
            "sample-0": {
                "token": "sample-0",
                "data": {"LIDAR_TOP": "lidar-0", "CAM_FRONT": "cam-0"},
                "next": "sample-1",
            },
            "sample-1": {
                "token": "sample-1",
                "data": {"LIDAR_TOP": "lidar-1", "CAM_FRONT": "cam-1"},
                "next": "",
            },
        }
        self.sample_data = {
            "lidar-0": {"ego_pose_token": "ego-lidar-0"},
            "lidar-1": {"ego_pose_token": "ego-lidar-1"},
            "cam-0": {"ego_pose_token": "ego-cam-0"},
            "cam-1": {"ego_pose_token": "ego-cam-1"},
        }
        self.ego_pose = {
            "ego-lidar-0": {"translation": [100.0, 200.0, 0.0]},
            "ego-lidar-1": {"translation": [101.5, 202.0, 0.0]},
            "ego-cam-0": {"translation": [99.0, 199.0, 0.0]},
            "ego-cam-1": {"translation": [100.0, 200.0, 0.0]},
        }

    def get(self, table_name: str, token: str):
        if table_name == "sample":
            return self.samples[token]
        if table_name == "sample_data":
            return self.sample_data[token]
        if table_name == "ego_pose":
            return self.ego_pose[token]
        raise KeyError(table_name)


def test_ego_trajectory_uses_sample_data_ego_pose_in_global_xy():
    nusc = _FakeNuScenes()
    scene = {
        "first_sample_token": "sample-0",
        "last_sample_token": "sample-1",
    }

    xy = ego_trajectory_xy(nusc, scene, channel="LIDAR_TOP")

    np.testing.assert_allclose(xy, [[100.0, 200.0], [101.5, 202.0]])


def test_ego_trajectory_channel_selects_timestamp_reference():
    nusc = _FakeNuScenes()
    scene = {
        "first_sample_token": "sample-0",
        "last_sample_token": "sample-1",
    }

    xy = ego_trajectory_xy(nusc, scene, channel="CAM_FRONT")

    np.testing.assert_allclose(xy, [[99.0, 199.0], [100.0, 200.0]])
