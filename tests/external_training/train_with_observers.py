# SPDX-License-Identifier: GPL-3.0-or-later
"""Real native headless training with multiple independent preview readers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from lfs_plugins.training_runtime.reader import Observer
from lfs_plugins.training_runtime import transport as t

parser = argparse.ArgumentParser(description=__doc__)
for name in ("backend", "dataset", "config", "area", "runtime"):
    parser.add_argument(name, type=Path)
parser.add_argument('--pause-probe', action='store_true')
args = parser.parse_args()
area = args.area.resolve()
area.mkdir(parents=True, exist_ok=True)
config = json.loads(args.config.read_text(encoding="utf-8"))
config["optimization"].update(iterations=5000, start_refine=100, stop_refine=4500,
    grow_until_iter=2500, save_steps=[5000], eval_steps=[2500, 5000], max_cap=10000)
config_path = area / "training.json"
config_path.write_text(json.dumps(config), encoding="utf-8")
sessions = area / "sessions"
project = area / "final.licht"
env = {**os.environ, "LFS_HOME": str(args.runtime.resolve()),
       "LFS_TRAINING_SESSIONS_DIR": str(sessions), "LFS_TRAINING_PUBLISHER": "automatic"}
env.pop("LFS_BLENDER_JOB", None)
env['LFS_TEST_AREA'] = str(area)
observers = []
received = [0, 0, 0]
record = None
with (area / "trainer.log").open("wb") as log:
    command = [str(args.backend.resolve()), "--headless", "-d", str(args.dataset.resolve()),
        "-o", str(area / "output"), "--config", str(config_path), "--test-every", "4", "--centralize", "off",
        "--no-download", "--save-project-at-iter", "5000", "--save-project-path", str(project)]
    if args.pause_probe:
        command += ['--python-script', str(Path(__file__).with_name('training_pause_probe.py').resolve())]
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            paths = list(sessions.glob("*/session.json"))
            if paths and not observers:
                observers = [Observer(paths[0].parent) for _ in range(3)]
            for index, observer in enumerate(observers):
                frame = observer.take()
                if frame:
                    assert frame["count"] <= 100000
                    assert frame["data"]["sh"].shape[1] <= 4
                    observer.acknowledge(frame["record"]["id"])
                    received[index] += 1
            if process.poll() is not None:
                break
            time.sleep(.02)
        else:
            raise AssertionError("Native training timeout")
        assert process.returncode == 0, process.returncode
        assert project.is_file()
        paths = list(sessions.glob("*/session.json"))
        assert len(paths) == 1
        record = t.read_json(paths[0])
        assert record["state"] == "completed", record
        assert record.get("saved_project") == str(project), record
        assert record.get("preview_capture_seconds", -1) >= 0, record
        assert record.get("preview_publish_seconds", -1) >= 0, record
        assert all(received), received
        assert record["iteration"] == 5000, record
        if args.pause_probe:
            paused = json.loads((area/'pause-status-proof.json').read_text(encoding='utf-8'))
            assert paused['ok'], paused
    finally:
        for observer in observers:
            observer.close()
            observer.thread.join(5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
proof = {"received": received, "iteration": record["iteration"], "snapshot": record.get("snapshot"),
    "model_count": record.get("model_count"), "saved_project": record.get("saved_project"),
    "capture_seconds": record["preview_capture_seconds"], "publish_seconds": record["preview_publish_seconds"]}
(area / "proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
print("NATIVE_TRAINING_THREE_OBSERVERS_PASS", json.dumps(proof), flush=True)
