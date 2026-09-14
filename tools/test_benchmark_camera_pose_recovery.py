# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

import benchmark_camera_pose_recovery as bench


def test_child_progress_is_visible_and_failure_is_not_hidden(tmp_path, capsys):
    log = tmp_path / 'console.log'
    command = [sys.executable, '-u', '-c', "print('detail'); print('Training [==] 10%'); raise SystemExit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        bench.run_training(command, log, 'on')
    assert error.value.returncode == 3
    assert 'ON: Training [==] 10%' in capsys.readouterr().out
    assert 'detail' in log.read_text(encoding='utf-8')


def records():
    result = []
    for i, (center, role) in enumerate([([0, 0, 0], 1), ([10, 0, 0], 1),
                                       ([3, 1, 0], 0), ([5, 2, 1], 0), ([1, 0, 2], 2)]):
        q = np.array([np.cos(i * .1), 0, np.sin(i * .1), 0])
        r = bench.rotation(q)
        result.append(dict(uid=i, name=f'{i}.png', q=q, R=r, center=np.array(center, dtype=float),
                           t=-r @ center, role=role))
    return result


def test_reproducible_noise_keeps_fixed_cameras_and_anchor_policy():
    a, b = records(), records()
    radius = bench.perturb(a, 123, .25, .0025)
    bench.perturb(b, 123, .25, .0025)
    for before, after, repeat in zip(records(), a, b):
        np.testing.assert_array_equal(after['perturbed_q'], repeat['perturbed_q'])
        np.testing.assert_array_equal(after['perturbed_t'], repeat['perturbed_t'])
        if after['role'] != 0:
            np.testing.assert_array_equal(before['q'], after['perturbed_q'])
            np.testing.assert_array_equal(before['t'], after['perturbed_t'])
        else:
            r = bench.rotation(after['perturbed_q'])
            assert bench.rotation_error(r, before['R']) == pytest.approx(.25)
            c = -r.T @ after['perturbed_t']
            assert np.linalg.norm(c - before['center']) == pytest.approx(radius * .0025)
            assert np.linalg.norm(c - a[0]['center']) < 10


@pytest.mark.parametrize('degrees,fraction', [(0,.0025), (2,.0025), (.25,0), (.25,.02), (float('nan'),.0025)])
def test_invalid_noise_rejected(degrees, fraction):
    with pytest.raises(ValueError):
        bench.perturb(records(), 1, degrees, fraction)


def test_membership_matches_source_not_refined_pose():
    inputs = records()
    state = []
    offset = np.array([12, 7, -3])
    for r in inputs:
        pose = np.eye(4)
        pose[:3, :3] = r['R']
        pose[:3, 3] = r['t'] - r['R'] @ offset
        state.append(dict(uid=r['uid'], role=r['role'], source=pose.ravel().tolist(), current=[0]*16))
    bench.match_membership(inputs, list(reversed(state)))
    assert [r['uid'] for r in inputs] == list(range(5))
    state[0]['source'][3] += 1
    with pytest.raises(ValueError, match='differ'):
        bench.match_membership(records(), state)


def test_binary_pose_edits_preserve_observations(tmp_path):
    path = tmp_path / 'images.bin'
    blob = struct.pack('<Qidddddddi', 1, 42, 1., 0., 0., 0., 0., 0., 0., 1)
    blob += b'image.png\0' + struct.pack('<Qddq', 1, 100., 200., -1)
    path.write_bytes(blob)
    data, rows = bench.read_images(path)
    assert rows[0]['offset'] == 12
    struct.pack_into('<ddddddd', data, rows[0]['offset'], 1., 0., 0., 0., 1., 2., 3.)
    assert data[:12] == blob[:12]
    assert data[68:] == blob[68:]
    assert path.read_bytes() == blob


def test_retry_accepts_only_the_original_prepared_dataset(tmp_path):
    source, root = tmp_path / 'source', tmp_path / 'trial'
    original = source / 'sparse/0'
    copied = root / 'dataset/sparse/0'
    original.mkdir(parents=True)
    copied.mkdir(parents=True)
    (source / 'images_4').mkdir()
    (root / 'dataset/images_4').mkdir()
    for folder in (original, copied):
        (folder / 'cameras.bin').write_bytes(b'cameras')
        (folder / 'points3D.bin').write_bytes(b'points')
    for folder in (source / 'images_4', root / 'dataset/images_4'):
        (folder / 'image.png').write_bytes(b'image')
    blob = (struct.pack('<Qidddddddi', 1, 42, 1., 0., 0., 0., 0., 0., 0., 1)
            + b'image.png\0' + struct.pack('<Qddq', 1, 100., 200., -1))
    (original / 'images.bin').write_bytes(blob)
    data, records = bench.read_images(original / 'images.bin')
    record = records[0]
    record.update(role=0, perturbed_q=[1., 0., 0., 0.], perturbed_t=[1., 2., 3.])
    struct.pack_into('<ddddddd', data, record['offset'], *record['perturbed_q'], *record['perturbed_t'])
    (copied / 'images.bin').write_bytes(data)
    manifest = dict(version=1, dataset=str(source), seed=12, degrees=.25,
                    center_fraction=.0025,
                    source_hashes={name: bench.digest(original / name)
                                   for name in ('images.bin', 'cameras.bin', 'points3D.bin')},
                    records=[{k: v.tolist() if isinstance(v, np.ndarray) else v
                              for k, v in record.items()}])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    args = argparse.Namespace(seed=12, degrees=.25, center_fraction=.0025)
    assert bench.validate_prepared(root, source, args)['seed'] == 12
    assert sorted(p.name for p in root.iterdir()) == ['dataset', 'manifest.json']
    bad_args = argparse.Namespace(seed=13, degrees=.25, center_fraction=.0025)
    with pytest.raises(ValueError, match='differs'):
        bench.validate_prepared(root, source, bad_args)
    (copied / 'images.bin').write_bytes(data + b'changed')
    with pytest.raises(ValueError, match='COLMAP|Invalid'):
        bench.validate_prepared(root, source, args)


def test_existing_or_nested_outputs_rejected_before_reading(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for output in (source, source / 'new', tmp_path):
        with pytest.raises(ValueError, match='new directory'):
            bench.prepare(argparse.Namespace(dataset=source, output=output, reference=tmp_path/'absent'))


def test_commands_only_differ_in_pose_switch_and_output(tmp_path):
    calls = bench.commands(tmp_path/'app.exe', tmp_path)
    assert '--refine-camera-poses' not in calls['off']
    assert '--refine-camera-poses' in calls['on']
    assert calls['off'][:calls['off'].index('-o')] == calls['on'][:calls['on'].index('-o')]
    assert '--resume' not in calls['on']


def test_report_recovers_original_coordinate_frame(tmp_path, monkeypatch):
    items = records()
    radius = bench.perturb(items, 123, .25, .0025)
    sparse = tmp_path/'original/sparse/0'
    sparse.mkdir(parents=True)
    source_file = sparse/'images.bin'
    source_file.write_bytes(b'original')
    manifest = dict(dataset=str(tmp_path/'original'), radius=radius,
                    source_hashes={'images.bin': bench.digest(source_file)},
                    records=[{k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()} for r in items])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    state = []
    offset = np.array([12., -3., 6.])
    for r in items:
        source, current = np.eye(4), np.eye(4)
        source[:3,:3] = bench.rotation(r['perturbed_q'])
        source[:3,3] = r['perturbed_t'] - source[:3,:3] @ offset
        current[:3,:3] = r['R']
        current[:3,3] = r['t'] - r['R'] @ offset
        state.append(dict(uid=r['uid'],role=r['role'],source=source.ravel().tolist(),current=current.ravel().tolist()))
    monkeypatch.setattr(bench,'checkpoint_params',lambda _: dict(dataset={},optimization={},camera_pose_state={'cameras':state}))
    for mode in ('off','on'):
        (tmp_path/mode).mkdir()
        (tmp_path/mode/'metrics.csv').write_text('iteration,psnr,ssim,lpips\n10000,23,0.6,0.3\n')
        (tmp_path/mode/'eval_step_10000').mkdir()
        (tmp_path/mode/'eval_step_10000/per_image_metrics.csv').write_text(
            'image_name,psnr,ssim,lpips\n4.png,23,0.6,0.3\n')
    result = bench.report(tmp_path)
    assert result['pose_medians']['center_after_fraction'] < 1e-12
    assert result['pose_medians']['rotation_after_deg'] < 1e-5
    state[0]['current'][3] += 1
    with pytest.raises(ValueError,match='moved'):
        bench.report(tmp_path)
