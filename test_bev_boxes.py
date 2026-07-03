from __future__ import annotations

import numpy as np

from nusc_lab.visualization.bev import box_footprint_xy, boxes_in_sample_data_frame


class _FakeBox:
    name = "vehicle.car"

    def bottom_corners(self):
        return np.array(
            [
                [2.0, 2.0, -2.0, -2.0],
                [1.0, -1.0, -1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )


class _FakeNuScenes:
    def __init__(self):
        self.sample_data_calls = []

    def get(self, table_name: str, token: str):
        if table_name != "sample_annotation":
            raise KeyError(table_name)
        return {"token": token, "category_name": "vehicle.car"}

    def get_sample_data(self, sample_data_token: str, selected_anntokens=None):
        self.sample_data_calls.append((sample_data_token, list(selected_anntokens or [])))
        return "unused.bin", [_FakeBox()], None


def test_boxes_in_sample_data_frame_uses_ref_sample_data_token():
    nusc = _FakeNuScenes()
    sample = {"data": {"LIDAR_TOP": "lidar-token"}, "anns": ["ann-token"]}

    boxes = boxes_in_sample_data_frame(nusc, sample, ref_channel="LIDAR_TOP")

    assert len(boxes) == 1
    assert nusc.sample_data_calls == [("lidar-token", ["ann-token"])]


def test_box_footprint_xy_closes_polygon():
    footprint = box_footprint_xy(_FakeBox())

    assert footprint.shape == (5, 2)
    np.testing.assert_allclose(footprint[0], footprint[-1])
