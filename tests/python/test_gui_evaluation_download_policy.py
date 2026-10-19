# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch-time download policy survives GUI dataset parameter replacement."""

import hashlib
import json
import math
import struct

import pytest

from runtime_test_app import isolated_app
from test_project_resume_dataset_contract import _make_dataset, _png
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _make_lpips_weights(path):
    """Write a sparse FP16 LFW fixture with the shapes required by VGG16 LPIPS.

    Zero convolutions and unit biases give a finite, deterministic zero distance.
    Sparse zero payloads keep disk usage small without downloading model weights.
    This fixture exercises loading and evaluation, not perceptual accuracy.
    """
    tensors = {}
    ones = []
    offset = 0

    def add(name, shape, *, one=False):
        nonlocal offset
        offset = (offset + 63) & ~63
        count = math.prod(shape)
        tensors[name] = {"dtype": "float16", "shape": shape,
                         "offset": offset, "length": count * 2}
        if one:
            ones.append((offset, count))
        offset += count * 2

    add("scaling.shift", [1, 3, 1, 1])
    add("scaling.scale", [1, 3, 1, 1], one=True)
    for index, cin, cout in (
        (0, 3, 64), (2, 64, 64), (5, 64, 128), (7, 128, 128),
        (10, 128, 256), (12, 256, 256), (14, 256, 256),
        (17, 256, 512), (19, 512, 512), (21, 512, 512),
        (24, 512, 512), (26, 512, 512), (28, 512, 512),
    ):
        add(f"vgg.features.{index}.weight", [cout, cin, 3, 3])
        add(f"vgg.features.{index}.bias", [cout], one=True)
    for index, channels in enumerate((64, 128, 256, 512, 512)):
        add(f"lin{index}.weight", [1, channels, 1, 1], one=True)
    header = json.dumps({"format": "lfw", "version": 1, "tensors": tensors},
                        separators=(",", ":")).encode()
    payload = (8 + len(header) + 63) & ~63
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(b"LFW1" + struct.pack("<I", len(header)) + header)
        output.truncate(payload + offset)
        for position, count in ones:
            output.seek(payload + position)
            output.write(struct.pack("<e", 1.0) * count)


@pytest.mark.parametrize("weights", ["missing", "cached", "override", "eval_disabled"])
def test_gui_evaluation_respects_no_download(tmp_path, monkeypatch, weights):
    monkeypatch.delenv("LFS_LPIPS_WEIGHTS", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    if weights == "override":
        local_weights = tmp_path / "weights.lfw"
        _make_lpips_weights(local_weights)
        monkeypatch.setenv("LFS_LPIPS_WEIGHTS", str(local_weights))

    dataset = tmp_path / "dataset"
    _make_dataset(dataset, "camera")
    transforms = json.loads((dataset / "transforms.json").read_text())
    transforms.update(w=32, h=32, fl_x=32.0, fl_y=32.0, cx=16.0, cy=16.0)
    (dataset / "transforms.json").write_text(json.dumps(transforms))
    for path in (dataset / "images").glob("*.png"):
        path.write_bytes(_png(32, 32, (60, 90, 150, 255)))

    cache = tmp_path / "home" / ".lichtfeld" / "onnx" / "lpips-vgg16-v0.1.lfw"
    log_path = tmp_path / "app.log"
    with isolated_app(tmp_path) as endpoint:
        if weights == "cached":
            # The automatic cache trusts only the shipped model's fixed digest.
            # A valid synthetic model must remain untouched under --no-download.
            _make_lpips_weights(cache)
            cached_digest = hashlib.sha256(cache.read_bytes()).digest()
        tools = {tool["name"] for tool in _call(endpoint, "tools/list")["tools"]}
        assert {"scene_load_dataset", "training_params_set", "training_start",
                "runtime_job_wait", "training_get_state"} <= tools
        _tool(endpoint, "scene_load_dataset", {
            "path": str(dataset), "output_path": str(tmp_path / "output"),
            "max_iterations": 1,
        })
        loaded = _tool(endpoint, "runtime_job_wait", {
            "job_id": "import.dataset", "until": "inactive", "timeout_ms": 60000,
        })
        assert "failed" not in str(loaded).lower(), loaded
        _tool(endpoint, "training_params_set", {
            "values": {"enable_eval": weights != "eval_disabled", "iterations": 1,
                       "max_cap": 1000},
        })
        _tool(endpoint, "training_start")
        _tool(endpoint, "runtime_job_wait", {
            "job_id": "training.main", "until": "inactive", "timeout_ms": 60000,
        })
        state = _tool(endpoint, "training_get_state")
        assert state["iteration"] == 1, (state, log_path.read_text())
        text = log_path.read_text()
        assert "Downloading LPIPS" not in text, text
        assert "Download failed" not in text, text
        if weights == "missing":
            assert "LPIPS unavailable: LPIPS is not cached:" in text, text
        elif weights == "cached":
            assert "LPIPS unavailable: Cached LPIPS SHA-256 mismatch" in text, text
            assert cache.is_file(), text
            assert hashlib.sha256(cache.read_bytes()).digest() == cached_digest
        else:
            assert "LPIPS unavailable" not in text, text
        if weights != "eval_disabled":
            metrics_files = list(tmp_path.rglob("per_image_metrics.json"))
            assert metrics_files, text
            metrics = json.loads(metrics_files[0].read_text())
            evaluations = [evaluation for view in metrics.values()
                           for evaluation in view["evaluations"]]
            assert evaluations
            for evaluation in evaluations:
                assert math.isfinite(evaluation["psnr"])
                assert math.isfinite(evaluation["ssim"])
                if weights in ("missing", "cached"):
                    assert evaluation["lpips"] is None
                else:
                    assert math.isfinite(evaluation["lpips"])
                    assert evaluation["lpips"] == 0.0
        if weights != "cached":
            assert not cache.exists()
