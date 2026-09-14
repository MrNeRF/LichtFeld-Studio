# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only movement audit of existing camera-pose matrix checkpoints.

Requires numpy and zstandard. Does not render, train, or modify projects.
Movement and accepted-step counts are not reconstruction quality measures.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_camera_pose_recovery import checkpoint_params


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite movement measurements')
    return dict(zip(('min', 'median', 'p90', 'max'), np.quantile(values, [0, .5, .9, 1]).tolist()))


def summarize(state):
    scale = float(state['settings']['optimizer']['scene_scale'])
    bound = float(state['settings']['optimizer']['max_center_fraction'])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(bound) or bound <= 0:
        raise ValueError('Invalid scene scale or movement bound')
    cameras = [c for c in state['cameras'] if c['role'] == 0]
    if not cameras:
        raise ValueError('No movable training cameras')
    source = np.array([c['source'] for c in cameras]).reshape(-1, 4, 4)
    current = np.array([c['current'] for c in cameras]).reshape(-1, 4, 4)
    def centers(poses):
        return -np.einsum('nji,nj->ni', poses[:, :3, :3], poses[:, :3, 3])
    shifts = np.linalg.norm(centers(current) - centers(source), axis=1) / scale
    relative = current[:, :3, :3] @ source[:, :3, :3].transpose(0, 2, 1)
    skew = np.stack((relative[:, 2, 1] - relative[:, 1, 2],
                     relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]), axis=1)
    rotations = np.degrees(np.arctan2(np.linalg.norm(skew, axis=1),
                                     np.trace(relative, axis1=1, axis2=2) - 1))
    accepted = sum(c['accepted_steps'] for c in cameras)
    result = dict(iteration=state['iteration'], scene_scale=scale,
                  cameras=len(cameras), updated_cameras=sum(c['accepted_steps'] > 0 for c in cameras),
                  accepted_steps=accepted, rejected_steps=sum(c['rejected_steps'] for c in cameras),
                  candidate_renders=sum(c['candidate_renders'] for c in cameras),
                  camera_shift_scene_fraction=quantiles(shifts), camera_rotation_degrees=quantiles(rotations),
                  camera_centers_near_bound=int((shifts >= bound * .999).sum()))
    points = state.get('points', [])
    if points:
        shifts = np.linalg.norm(np.array([p['current'] for p in points]) -
                                np.array([p['source'] for p in points]), axis=1) / scale
        result.update(points=len(points), point_shift_scene_fraction=quantiles(shifts),
                      points_near_bound=int((shifts >= bound * .999).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Existing matrix output directory')
    parser.add_argument('--output', type=Path, help='Optional new JSON report; never overwritten')
    args = parser.parse_args()
    projects = sorted(args.root.glob('*/*/on/project.licht'))
    if not projects:
        parser.error('No ON projects found')
    results = []
    for project in projects:
        params = checkpoint_params(project)
        results.append(dict(dataset=project.parents[2].name, strategy=project.parents[1].name,
                            project=str(project), **summarize(params['camera_pose_state'])))
    report = dict(scope='Checkpoint movement only; no geometric residual, timing profile or visual quality certification.',
                  runs=results)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as file:
            file.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
