# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded IPC pressure/churn qualification; no GUI, GPU or network access."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from lfs_plugins.training_runtime import transport as t
from lfs_plugins.training_runtime.reader import Observer
import numpy as np


def handles():
    if os.name != "nt":
        return len(list(Path("/proc/self/fd").iterdir()))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    value = ctypes.c_ulong()
    assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(value))
    return value.value


def run(seconds, count, consumers, warmup=True):
    if warmup:
        run(.5, min(count, 1000), 1, warmup=False)
    base_handles, base_threads = handles(), threading.active_count()
    frames = [0] * consumers
    generations = 0
    recoverable_errors = 0
    max_handles = base_handles
    with tempfile.TemporaryDirectory() as directory:
        old = os.environ.get("LFS_TRAINING_SESSIONS_DIR")
        os.environ["LFS_TRAINING_SESSIONS_DIR"] = directory
        publisher = t.SessionPublisher({"label": "pressure fixture"})
        observers = [Observer(publisher.root) for _ in range(consumers)]
        payload = {"means": np.zeros((count, 3), dtype=np.float32), "scale": np.zeros((count, 3), dtype=np.float32),
            "rotation": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 1)),
            "opacity": np.zeros((count, 1), dtype=np.float32), "sh": np.zeros((count, 4, 3), dtype=np.float32)}
        start, next_churn = time.monotonic(), 5
        try:
            while time.monotonic() - start < seconds:
                generations += 1
                publisher.publish(payload, generations)
                for index, observer in enumerate(observers):
                    frame = observer.take()
                    if frame:
                        assert frame["count"] == count
                        observer.acknowledge(frame["record"]["id"])
                        frames[index] += 1
                    recoverable_errors += bool(observer.error)
                max_handles = max(max_handles, handles())
                if time.monotonic() - start >= next_churn:
                    index = int(next_churn / 5) % consumers
                    observers[index].close()
                    observers[index].thread.join(5)
                    assert not observers[index].thread.is_alive()
                    observers[index] = Observer(publisher.root)
                    next_churn += 5
                time.sleep(.02)
            assert all(frames), frames
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                for observer in observers:
                    frame = observer.take()
                    if frame:
                        observer.acknowledge(frame["record"]["id"])
                if all(not observer.error for observer in observers):
                    break
                time.sleep(.05)
            assert all(not observer.error for observer in observers), [o.error for o in observers]
        finally:
            for observer in observers:
                observer.close()
                observer.thread.join(5)
                assert not observer.thread.is_alive()
            publisher.finish()
            publisher.thread.join(5)
            publisher.pool.close()
            if old is None:
                os.environ.pop("LFS_TRAINING_SESSIONS_DIR", None)
            else:
                os.environ["LFS_TRAINING_SESSIONS_DIR"] = old
    gc.collect()
    final_handles, final_threads = handles(), threading.active_count()
    assert final_handles <= base_handles + 4, (base_handles, final_handles)
    assert final_threads == base_threads, (base_threads, final_threads)
    return {"seconds": seconds, "points": count, "consumers": consumers, "generations": generations,
        "received": frames, "recoverable_error_samples": recoverable_errors, "handles": [base_handles, max_handles, final_handles],
        "threads": [base_threads, final_threads]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--consumers", type=int, default=4)
    args = parser.parse_args()
    assert args.seconds > 0 and 0 < args.points <= 1000000 and 0 < args.consumers <= 16
    print("SESSION_PRESSURE_PASS", json.dumps(run(args.seconds, args.points, args.consumers)), flush=True)
