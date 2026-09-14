#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Serial SH0 camera-pose OFF/ON comparison across three strategies and two datasets.

Uses an existing executable, never builds, never edits datasets, and refuses to
reuse an output directory. The statue preset always enables undistortion.
Python standard library only. Quality differences from one run are not proof of
statistical significance; compare each ON only with its matching strategy OFF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

STRATEGIES = ('mrnf', 'mcmc', 'igs+')
METRICS = ('psnr', 'ssim', 'lpips')


def plan(executable, bicycle, statue, output):
    runs = []
    for name, dataset, images, resize, width, undistort in (
            ('bicycle', bicycle, 'images_4', 1, 0, False),
            ('statue', statue, 'images', 8, 1600, True)):
        for strategy in STRATEGIES:
            for mode in ('off', 'on'):
                folder = output / name / strategy / mode
                command = [str(executable), '-d', str(dataset), '--images', images,
                           '-r', str(resize), '--max-width', str(width), '--strategy', strategy,
                           '--raster-backend', '3dgs', '--sh-degree', '0', '--iter', '10000',
                           '--max-cap', '1000000', '--eval', '--test-every', '8',
                           '--eval-steps', '5000', '--eval-steps', '10000', '--headless',
                           '-o', str(folder), '--log-file', str(folder / 'training.log')]
                if undistort:
                    command.append('--undistort')
                if mode == 'on':
                    command += ['--refine-camera-poses', '--camera-pose-start-step', '500',
                                '--camera-pose-end-percent', '50']
                runs.append(dict(dataset=name, strategy=strategy, mode=mode,
                                 folder=str(folder), command=command))
    return runs


def fingerprint(executable):
    # Local shared libraries can change even when the executable does not.
    files = [executable, *sorted(executable.parent.glob('*.dll'))]
    result = {}
    for path in files:
        with path.open('rb') as file:
            result[path.name] = hashlib.file_digest(file, 'sha256').hexdigest()
    return result


def run_child(command, console, label):
    with console.open('x', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as child:
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    if any(token in line for token in ('Training [', '[error]', '[warn', 'Camera pose')):
                        print(f'{label}: {line.rstrip()}', flush=True)
                code = child.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise


def read_result(folder, mode):
    log = (folder / 'training.log').read_text(encoding='utf-8', errors='replace')
    if 'Headless training completed' not in log or '[error]' in log:
        raise ValueError(f'Incomplete or failed training: {folder}')
    support = None
    if mode == 'on':
        match = re.search(r'Camera pose SfM guard: (\d+)/(\d+) movable cameras', log)
        if not match or int(match[1]) == 0:
            raise ValueError(f'No verified geometric support in ON run: {folder}')
        support = dict(protected=int(match[1]), movable=int(match[2]))
        if 'Camera pose refinement frozen at iteration 5000 (scheduled stop 5000)' not in log:
            raise ValueError(f'Missing expected pose freeze: {folder}')
    with (folder / 'metrics.csv').open(encoding='utf-8', newline='') as file:
        final = [r for r in csv.DictReader(file) if int(r['iteration']) == 10000]
    if len(final) != 1:
        raise ValueError(f'Missing or duplicate final metrics: {folder}')
    metrics = {k: float(final[0][k]) for k in METRICS}
    with (folder / 'eval_step_10000/per_image_metrics.csv').open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    views = {r['image_name']: {k: float(r[k]) for k in METRICS} for r in rows}
    if not views or len(views) != len(rows):
        raise ValueError(f'Missing or duplicate evaluation views: {folder}')
    if not all(math.isfinite(v) for row in [metrics, *views.values()] for v in row.values()):
        raise ValueError(f'Nonfinite metrics: {folder}')
    return dict(metrics=metrics, views=views, support=support,
                num_gaussians=int(final[0]['num_gaussians']),
                warnings=sorted(set(re.findall(r'\[(?:warn|warning)\] (.*)', log))))


def compare(off, on):
    if set(off['views']) != set(on['views']):
        raise ValueError('OFF/ON evaluation views differ')
    result = dict(views=len(off['views']))
    for metric in METRICS:
        result[f'{metric}_off'] = off['metrics'][metric]
        result[f'{metric}_on'] = on['metrics'][metric]
        result[f'{metric}_delta'] = on['metrics'][metric] - off['metrics'][metric]
        deltas = [on['views'][name][metric] - row[metric] for name, row in off['views'].items()]
        result[f'{metric}_views_improved'] = sum(d * (-1 if metric == 'lpips' else 1) > 0 for d in deltas)
        result[f'{metric}_delta_min'] = min(deltas)
        result[f'{metric}_delta_max'] = max(deltas)
    result['seconds_off'] = off['seconds']
    result['seconds_on'] = on['seconds']
    result['overhead_percent'] = 100 * (on['seconds'] / off['seconds'] - 1)
    result['protected_cameras'] = on['support']['protected']
    result['movable_cameras'] = on['support']['movable']
    result['gaussians_off'] = off['num_gaussians']
    result['gaussians_on'] = on['num_gaussians']
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bicycle', type=Path, required=True)
    parser.add_argument('--statue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=Path('build/LichtFeld-Studio.exe'))
    parser.add_argument('--dry-run', action='store_true', help='Print the 12 commands without writing or running')
    args = parser.parse_args()
    args.bicycle, args.statue = args.bicycle.resolve(), args.statue.resolve()
    output, executable = args.output.resolve(), args.executable.resolve()
    if not executable.is_file():
        parser.error('Existing executable missing; compile separately')
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    for dataset, images in ((args.bicycle, 'images_4'), (args.statue, 'images')):
        if output == dataset or dataset in output.parents or output in dataset.parents:
            parser.error('Output must be separate from each dataset')
        if not (dataset / images).is_dir() or not (dataset / 'sparse').is_dir():
            parser.error(f'Missing {images} or sparse directory in {dataset}')
    runs = plan(executable, args.bicycle, args.statue, output)
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        return
    hashes = fingerprint(executable)
    output.mkdir(parents=True)
    save(output / 'manifest.json', dict(version=1, binaries=hashes, runs=runs))
    results, summary = {}, []
    for index, run in enumerate(runs, 1):
        folder = Path(run['folder'])
        label = f"{run['dataset']}/{run['strategy']}/{run['mode']}"
        status = dict(run=run, status='running')
        folder.mkdir(parents=True)
        save(folder / 'status.json', status)
        try:
            if fingerprint(executable) != hashes:
                raise ValueError('Executable or local DLLs changed during benchmark')
            print(f'[{index}/{len(runs)}] {label} — 10000 steps', flush=True)
            start = time.perf_counter()
            run_child(run['command'], folder / 'console.log', label)
            elapsed = time.perf_counter() - start
            result = read_result(folder, run['mode'])
            result['seconds'] = elapsed
            save(folder / 'result.json', result)
            results[label] = result
            if run['mode'] == 'on':
                row = dict(dataset=run['dataset'], strategy=run['strategy'],
                           **compare(results[label[:-2] + 'off'], result))
                summary.append(row)
                save(output / 'summary.json', summary)
                with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(summary)
                print(json.dumps(row, indent=2), flush=True)
            status['status'] = 'completed'
            save(folder / 'status.json', status)
        except BaseException as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}')
            save(folder / 'status.json', status)
            raise
    print(f'All 12 runs completed. Results: {output / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
