#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Controlled COLMAP pose recovery benchmark. Never builds or edits its input.

Requires numpy and zstandard. A clean-run checkpoint supplies camera membership,
not an optimizer target. Sparse geometry stays unchanged: this isolates pose
errors and does not model simultaneous errors in SfM points or calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np


def checkpoint_params(path):
    import inspect_licht as il

    with Path(path).open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        heads = [il.parse_head(data[p:p + 4096]) for p in il.HEAD_SLOTS]
        head = max((h for h in heads if h), key=lambda h: h['head_sequence'])
        p = head['commit_offset']
        _, rows = il.decode_index(data, il.parse_commit(data[p:p + 256], p))
        row = il.live_row(rows, 'CKPT')
        payload = bytes(data[row['payload_offset']:row['payload_offset'] + row['stored_bytes']])
        decoded = payload if row['compression'] == 'Stored' else b''.join(
            il._iter_uncompressed_chunks(payload, row['uncompressed_bytes']))
        if row['compression'] == 'ByteShuffleZstdFramed':
            decoded = np.frombuffer(decoded, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()
        if il.u32(decoded, 0) != 0x4C464B50:
            raise ValueError('Unsupported checkpoint magic')
        start, size = il.u64(decoded, 24), il.u64(decoded, 32)
        return json.loads(decoded[start:start + size])


def rotation(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def multiply_quaternions(a, b):
    w, v = a[0], np.asarray(a[1:])
    s, u = b[0], np.asarray(b[1:])
    return np.r_[w*s - v@u, w*u + s*v + np.cross(v, u)]


def read_images(path):
    data = bytearray(Path(path).read_bytes())
    count, = struct.unpack_from('<Q', data)
    cursor, records = 8, []
    for _ in range(count):
        start = cursor
        values = struct.unpack_from('<idddddddi', data, cursor)
        cursor += 64
        end = data.index(0, cursor)
        name = data[cursor:end].decode('utf-8')
        cursor = end + 1
        n, = struct.unpack_from('<Q', data, cursor)
        cursor += 8 + n * 24
        if cursor > len(data):
            raise ValueError('Truncated COLMAP observations')
        q, t = np.array(values[1:5]), np.array(values[5:8])
        records.append(dict(name=name, offset=start + 4, q=q, t=t,
                            R=rotation(q), center=-rotation(q).T @ t))
    if cursor != len(data) or len({r['name'] for r in records}) != count:
        raise ValueError('Invalid COLMAP image table')
    return data, records


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def match_membership(records, state):
    """Fail on ambiguous matching rather than silently perturbing eval views."""
    if len(records) != len(state):
        raise ValueError('Reference checkpoint does not contain every input camera')
    rotations = np.stack([r['R'] for r in records])
    used, offsets = set(), []
    for camera in state:
        pose = np.asarray(camera['source']).reshape(4, 4)
        matches = np.flatnonzero(np.max(abs(rotations - pose[:3, :3]), axis=(1, 2)) < 1e-5)
        if len(matches) != 1 or int(matches[0]) in used:
            raise ValueError('Ambiguous camera mapping; use a dataset with distinct source orientations')
        index = int(matches[0])
        used.add(index)
        record = records[index]
        record.update(uid=camera['uid'], role=camera['role'])
        offsets.append(-np.linalg.solve(pose[:3, :3], pose[:3, 3] - record['t']))
    if np.max(np.ptp(offsets, axis=0)) > 1e-4:
        raise ValueError('Reference source cameras differ from the input dataset')


def perturb(records, seed, degrees, center_fraction):
    if not (0 < degrees <= 1 and 0 < center_fraction <= 0.01):
        raise ValueError('Expected rotation in (0,1] degrees and center fraction in (0,0.01]')
    rng = np.random.default_rng(seed)
    centers = np.stack([r['center'] for r in records])
    radius = float(np.median(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError('Degenerate scene radius')
    training = sorted((r for r in records if r['role'] != 2), key=lambda r: r['uid'])
    anchors = [r for r in training if r['role'] == 1]
    if len(anchors) != 2 or training[0]['uid'] != anchors[0]['uid']:
        raise ValueError('Expected default two-anchor policy')
    origin = anchors[0]['center']
    baseline = np.linalg.norm(anchors[1]['center'] - origin)
    if baseline <= 0:
        raise ValueError('Degenerate anchors')
    for record in sorted(records, key=lambda r: r['uid']):
        q, center = record['q'].copy(), record['center'].copy()
        if record['role'] == 0:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = np.deg2rad(degrees)
            q = multiply_quaternions(np.r_[np.cos(angle/2), axis*np.sin(angle/2)], q)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            candidate = center + direction * radius * center_fraction
            # Do not let perturbations change the deterministic farthest anchor.
            if np.linalg.norm(candidate - origin) < baseline * (1 - 1e-6):
                center = candidate
        record['perturbed_q'] = q
        record['perturbed_t'] = -rotation(q) @ center if record['role'] == 0 else record['t'].copy()
    return radius


def prepare(args):
    source, root = args.dataset.resolve(), args.output.resolve()
    if root.exists() or root == source or source in root.parents or root in source.parents:
        raise ValueError('Output must be a new directory separate from the original dataset')
    reference = checkpoint_params(args.reference)
    if Path(reference['dataset']['data_path']).resolve() != source:
        raise ValueError('Reference checkpoint belongs to another dataset')
    if reference['dataset']['images'] != 'images_4':
        raise ValueError('This protocol requires images_4')
    sparse = source / 'sparse/0'
    data, records = read_images(sparse / 'images.bin')
    match_membership(records, reference['camera_pose_state']['cameras'])
    radius = perturb(records, args.seed, args.degrees, args.center_fraction)
    hashes = {name: digest(sparse / name) for name in ('images.bin', 'cameras.bin', 'points3D.bin')}
    root.mkdir(parents=True)
    destination = root / 'dataset'
    (destination / 'sparse/0').mkdir(parents=True)
    # Real copies: no symlinks, junctions or writable hardlinks into the source.
    shutil.copytree(source / 'images_4', destination / 'images_4')
    for name in ('cameras.bin', 'points3D.bin'):
        shutil.copy2(sparse / name, destination / 'sparse/0' / name)
    for record in records:
        if record['role'] == 0:
            struct.pack_into('<ddddddd', data, record['offset'],
                             *record['perturbed_q'], *record['perturbed_t'])
    (destination / 'sparse/0/images.bin').write_bytes(data)
    manifest = dict(version=1, dataset=str(source), seed=args.seed, radius=radius,
                    degrees=args.degrees, center_fraction=args.center_fraction,
                    source_hashes=hashes, records=[{k: v.tolist() if isinstance(v, np.ndarray) else v
                                                 for k, v in r.items()} for r in records])
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def validate_prepared(root, source, args):
    """Reuse a prepared dataset only when it is exactly the requested experiment."""
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Existing output has no benchmark manifest; choose a new output')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('version') != 1 or Path(manifest.get('dataset', '')).resolve() != source or
            manifest.get('seed') != args.seed or manifest.get('degrees') != args.degrees or
            manifest.get('center_fraction') != args.center_fraction):
        raise ValueError('Existing preparation differs from the requested experiment')
    original_sparse = source / 'sparse/0'
    copied_sparse = root / 'dataset/sparse/0'
    for name, expected in manifest['source_hashes'].items():
        if digest(original_sparse / name) != expected:
            raise ValueError(f'Original {name} changed since preparation')
        if name != 'images.bin' and digest(copied_sparse / name) != expected:
            raise ValueError(f'Prepared {name} changed since preparation')
    records = manifest['records']
    data, imported = read_images(copied_sparse / 'images.bin')
    if len(records) != len(imported):
        raise ValueError('Prepared camera count changed')
    for saved, observed in zip(records, imported):
        if (saved['name'] != observed['name'] or
                not np.allclose(observed['q'], saved['perturbed_q'], atol=1e-10) or
                not np.allclose(observed['t'], saved['perturbed_t'], atol=1e-10)):
            raise ValueError('Prepared camera poses changed')
    original_data = bytearray((original_sparse / 'images.bin').read_bytes())
    for saved in records:
        if saved['role'] == 0:
            struct.pack_into('<ddddddd', original_data, saved['offset'],
                             *saved['perturbed_q'], *saved['perturbed_t'])
    if hashlib.sha256(original_data).hexdigest() != hashlib.sha256(data).hexdigest():
        raise ValueError('Prepared COLMAP observations or metadata changed')
    if not (root / 'dataset/images_4').is_dir():
        raise ValueError('Prepared images are missing')
    original_images = source / 'images_4'
    copied_images = root / 'dataset/images_4'
    original_files = {p.relative_to(original_images): p.stat().st_size
                      for p in original_images.rglob('*') if p.is_file()}
    copied_files = {p.relative_to(copied_images): p.stat().st_size
                    for p in copied_images.rglob('*') if p.is_file()}
    if not original_files or original_files != copied_files:
        raise ValueError('Prepared image set differs from the original')
    return manifest


def commands(executable, root, dataset=None):
    base = [str(executable.resolve()), '-d', str(dataset or root / 'dataset'), '--images', 'images_4',
            '-r', '1', '--max-width', '0', '--strategy', 'mrnf', '--raster-backend', '3dgs',
            '--sh-degree', '0', '--iter', '10000', '--max-cap', '1000000', '--eval',
            '--test-every', '8', '--eval-steps', '5000', '--eval-steps', '10000', '--headless']
    return {mode: base + ['-o', str(root / mode), '--log-file', str(root / (mode + '.log'))] +
            (['--refine-camera-poses', '--camera-pose-start-step', '500', '--camera-pose-end-percent', '50']
             if mode == 'on' else []) for mode in ('off', 'on')}


def rotation_error(a, b):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))))


def run_training(command, log_path, mode):
    """Keep the full console log and show progress as the child emits it."""
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    if 'Training [' in line or '[error]' in line or '[warn]' in line:
                        print(f'{mode.upper()}: {line.rstrip()}', flush=True)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def report(root, prepared_root=None):
    prepared_root = prepared_root or root
    manifest = json.loads((prepared_root / 'manifest.json').read_text(encoding='utf-8'))
    original = Path(manifest['dataset']) / 'sparse/0'
    if any(digest(original / name) != value for name, value in manifest['source_hashes'].items()):
        raise ValueError('Original sparse dataset changed since preparation')
    params = {mode: checkpoint_params(root / mode / 'project.licht') for mode in ('off', 'on')}
    for section in ('optimization', 'dataset'):
        a, b = params['off'][section], params['on'][section]
        allowed = {'refine_camera_poses', 'camera_pose_end_percent'} if section == 'optimization' else {'output_folder'}
        if any(a.get(k) != b.get(k) for k in a.keys() | b.keys() if k not in allowed):
            raise ValueError('OFF/ON configurations differ outside the pose switch')
    state = {r['uid']: r for r in params['on']['camera_pose_state']['cameras']}
    records = manifest['records']
    if set(state) != {r['uid'] for r in records}:
        raise ValueError('Saved camera membership changed')
    offsets = []
    for record in records:
        saved = state[record['uid']]
        source = np.asarray(saved['source']).reshape(4, 4)
        expected = rotation(record['perturbed_q'])
        if saved['role'] != record['role'] or np.max(abs(source[:3, :3] - expected)) > 1e-5:
            raise ValueError('Actual imported membership/poses differ from the prepared experiment')
        offsets.append(-np.linalg.solve(source[:3, :3], source[:3, 3] - record['perturbed_t']))
    if np.max(np.ptp(offsets, axis=0)) > 1e-4:
        raise ValueError('Inconsistent checkpoint coordinate frame')
    offset = np.median(offsets, axis=0)
    pose_errors = []
    for record in records:
        saved = state[record['uid']]
        current = np.asarray(saved['current']).reshape(4, 4)
        if record['role'] != 0:
            if saved['source'] != saved['current']:
                raise ValueError('An anchor or evaluation camera moved')
            continue
        truth_r, truth_c = np.asarray(record['R']), np.asarray(record['center'])
        source_r = rotation(record['perturbed_q'])
        source_c = -source_r.T @ record['perturbed_t']
        final_c = -np.linalg.solve(current[:3, :3], current[:3, 3]) - offset
        pose_errors.append(dict(name=record['name'],
            rotation_before_deg=rotation_error(source_r, truth_r),
            rotation_after_deg=rotation_error(current[:3, :3], truth_r),
            center_before_fraction=float(np.linalg.norm(source_c - truth_c) / manifest['radius']),
            center_after_fraction=float(np.linalg.norm(final_c - truth_c) / manifest['radius'])))
    metrics = {}
    per_view = {}
    expected_views = {r['name'] for r in records if r['role'] == 2}
    for mode in ('off', 'on'):
        rows = list(csv.DictReader((root / mode / 'metrics.csv').open(encoding='utf-8')))
        final = [r for r in rows if int(r['iteration']) == 10000]
        if len(final) != 1:
            raise ValueError('Missing or duplicate final evaluation')
        metrics[mode] = {k: float(final[0][k]) for k in ('psnr', 'ssim', 'lpips')}
        with (root / mode / 'eval_step_10000/per_image_metrics.csv').open(encoding='utf-8') as file:
            views = list(csv.DictReader(file))
        if len(views) != len(expected_views) or {r['image_name'] for r in views} != expected_views:
            raise ValueError('Final evaluation does not cover the exact protected view set')
        per_view[mode] = {r['image_name']: {k: float(r[k]) for k in ('psnr', 'ssim', 'lpips')} for r in views}
        if not all(np.isfinite(v) for row in per_view[mode].values() for v in row.values()):
            raise ValueError('Nonfinite per-view metrics')
    if not all(np.isfinite(v) for values in metrics.values() for v in values.values()):
        raise ValueError('Nonfinite quality metrics')
    medians = {k: float(np.median([r[k] for r in pose_errors])) for k in pose_errors[0] if k != 'name'}
    result = dict(metrics=metrics, delta_on_minus_off={k: metrics['on'][k]-metrics['off'][k] for k in metrics['on']},
                  views_improved={k: sum((per_view['on'][name][k] - per_view['off'][name][k]) *
                                        (-1 if k == 'lpips' else 1) > 0 for name in expected_views)
                                  for k in ('psnr', 'ssim', 'lpips')},
                  pose_medians=medians, cameras=pose_errors,
                  limitation='Controlled camera errors with unchanged sparse geometry; not a real-world quality guarantee.')
    (root / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cameras'}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'report'))
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempt', help='New subfolder for a retry using an existing prepared dataset')
    parser.add_argument('--executable', type=Path, default=Path('build/LichtFeld-Studio.exe'))
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--degrees', type=float, default=0.25)
    parser.add_argument('--center-fraction', type=float, default=0.0025)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.attempt and (args.attempt in ('.', '..') or
                         Path(args.attempt).name != args.attempt or
                         '/' in args.attempt or '\\' in args.attempt):
        parser.error('--attempt must be a single new folder name')
    run_root = args.output / args.attempt if args.attempt else args.output
    if args.action == 'report':
        report(run_root, args.output)
        return
    if not args.dataset or not args.reference:
        parser.error('--dataset and --reference are required for preparation')
    if args.action == 'run' and not args.executable.is_file():
        parser.error('Existing executable missing; this tool never builds it')
    if args.attempt:
        validate_prepared(args.output, args.dataset.resolve(), args)
        if run_root.exists():
            raise ValueError('Attempt already exists; choose a new --attempt name')
        run_root.mkdir()
    else:
        prepare(args)
    executable_hash = digest(args.executable) if args.executable.is_file() else None
    calls = commands(args.executable, run_root, args.output / 'dataset')
    (run_root / 'commands.json').write_text(json.dumps(calls, indent=2), encoding='utf-8')
    if args.action == 'prepare':
        print(json.dumps(calls, indent=2))
        return
    timings = {}
    for mode, command in calls.items():
        if digest(args.executable) != executable_hash:
            raise ValueError('Executable changed between preparation and training')
        print(f'Starting {mode.upper()} (10000 steps)', flush=True)
        start = time.monotonic()
        run_training(command, run_root / (mode + '-console.log'), mode)
        timings[mode] = time.monotonic() - start
    (run_root / 'timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    (run_root / 'executable-sha256.txt').write_text(executable_hash, encoding='utf-8')
    report(run_root, args.output)


if __name__ == '__main__':
    main()
