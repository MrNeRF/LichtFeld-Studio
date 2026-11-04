# SPDX-License-Identifier: GPL-3.0-or-later
"""Two independent protocol publishers for a disposable GUI, without optimizers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/python'))
from lfs_plugins.training_runtime.transport import SessionPublisher, atomic_json

parser = argparse.ArgumentParser()
parser.add_argument('area', type=Path)
parser.add_argument('--publisher-label', choices=('first', 'second'))
args = parser.parse_args()
area = args.area.resolve()
area.mkdir(parents=True, exist_ok=True)
os.environ['LFS_TRAINING_SESSIONS_DIR'] = str(area / 'sessions')
stop, finish = area / 'stop', area / 'finish'

if args.publisher_label:
    label = args.publisher_label
    publisher = SessionPublisher({'label': 'First study' if label == 'first' else 'Second study',
                                  'iterations': 10000})
    rng = np.random.default_rng(4)
    arrays = {'means': rng.normal(0, .2, (500, 3)).astype(np.float32),
              'scale': np.full((500, 3), -4, np.float32),
              'rotation': np.tile(np.array([1, 0, 0, 0], np.float32), (500, 1)),
              'opacity': np.zeros((500, 1), np.float32), 'sh': np.zeros((500, 4, 3), np.float32)}
    atomic_json(area / f'publisher-{label}.json', {'id': publisher.id, 'pid': os.getpid()})
    try:
        started, step, finished = time.monotonic(), 1, False
        while not stop.exists() and time.monotonic() - started < 300:
            if finish.exists() and not finished:
                publisher.finish('completed', iteration=step)
                finished = True
            if not finished:
                publisher.publish(arrays, step)
                step += 1
            time.sleep(.25)
    finally:
        if not publisher.stop.is_set():
            publisher.finish('stopped')
        publisher.thread.join(3)
        publisher.pool.close()
else:
    # Discovery intentionally chooses one newest registration per real process.
    # Two sessions in one process would no longer exercise switching trainers.
    for path in (stop, finish, area / 'ready.json', area / 'publisher-first.json', area / 'publisher-second.json'):
        path.unlink(missing_ok=True)
    processes, logs = [], []
    try:
        for label in ('first', 'second'):
            log = (area / f'publisher-{label}.log').open('wb')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(area),
                                               '--publisher-label', label], stdout=log, stderr=subprocess.STDOUT,
                                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            assert all(process.poll() is None for process in processes), 'Fixture publisher exited'
            try:
                records = [json.loads((area / f'publisher-{label}.json').read_text()) for label in ('first', 'second')]
                break
            except (OSError, ValueError):
                time.sleep(.05)
        else:
            raise AssertionError('Independent publisher startup timed out')
        assert records[0]['pid'] != records[1]['pid']
        atomic_json(area / 'ready.json', {'first': records[0]['id'], 'second': records[1]['id']})
        print('PUBLISHERS_READY', flush=True)
        deadline = time.monotonic() + 290
        while not stop.exists() and time.monotonic() < deadline:
            assert all(process.poll() is None for process in processes), 'Fixture publisher exited'
            time.sleep(.2)
    finally:
        stop.touch()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()  # Only fixture-owned children.
                process.wait(timeout=5)
        for log in logs:
            log.close()
