# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest

from analyze_camera_pose_checkpoints import summarize


def state():
    source = np.eye(4)
    current = source.copy()
    current[0, 3] = .02
    return dict(iteration=10000, settings=dict(optimizer=dict(scene_scale=2, max_center_fraction=.03)),
                cameras=[dict(role=0, source=source.ravel().tolist(), current=current.ravel().tolist(),
                              accepted_steps=3, rejected_steps=4, candidate_renders=5)],
                points=[dict(source=[0, 0, 4], current=[.06, 0, 4])])


def test_movement_units_and_counters():
    result = summarize(state())
    assert result['camera_shift_scene_fraction']['median'] == pytest.approx(.01)
    assert result['camera_rotation_degrees']['max'] == 0
    assert result['camera_centers_near_bound'] == 0
    assert result['points_near_bound'] == 1
    assert result['accepted_steps'] == 3
    assert result['updated_cameras'] == 1


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_scale(value):
    saved = state()
    saved['settings']['optimizer']['scene_scale'] = value
    with pytest.raises(ValueError):
        summarize(saved)
