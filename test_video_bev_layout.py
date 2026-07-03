from __future__ import annotations

import numpy as np
import pytest

from nusc_lab.visualization.video import _bev_pixel_mapper


def test_video_bev_mapper_keeps_equal_scale_and_forward_up():
    map_xy = _bev_pixel_mapper(1920, 540, xlim=(-35.0, 65.0), ylim=(-55.0, 55.0))

    origin = map_xy(np.asarray([[0.0, 0.0]]))[0]
    forward = map_xy(np.asarray([[10.0, 0.0]]))[0]
    left = map_xy(np.asarray([[0.0, 10.0]]))[0]

    assert forward[1] < origin[1]
    assert left[0] < origin[0]
    np.testing.assert_allclose(abs(forward[1] - origin[1]), abs(left[0] - origin[0]))


def test_video_bev_mapper_centers_non_stretched_panel_in_wide_tile():
    map_xy = _bev_pixel_mapper(1920, 540, xlim=(-35.0, 65.0), ylim=(-55.0, 55.0))

    corners = map_xy(
        np.asarray(
            [
                [-35.0, -55.0],
                [-35.0, 55.0],
                [65.0, -55.0],
                [65.0, 55.0],
            ]
        )
    )

    content_width = corners[:, 0].max() - corners[:, 0].min()
    content_height = corners[:, 1].max() - corners[:, 1].min()
    assert content_width < 1920 * 0.5
    np.testing.assert_allclose(content_height / content_width, 100.0 / 110.0)


def test_video_bev_mapper_rejects_invalid_limits():
    with pytest.raises(ValueError):
        _bev_pixel_mapper(1920, 540, xlim=(1.0, 1.0), ylim=(-1.0, 1.0))
