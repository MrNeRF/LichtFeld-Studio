# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""People masking: model resolution, mask writing from a matting function, and the real model when available."""
import hashlib
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import masking  # noqa: E402


def _png_rgb(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    import zlib

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_model_is_pinned_and_cached_under_the_cache_dir(tmp_path):
    assert masking.MODEL_URL.startswith("https://github.com/PeterL1n/RobustVideoMatting/releases/download/")
    assert len(masking.MODEL_SHA256) == 64
    assert masking.cached_model_path(tmp_path) == tmp_path / "models" / "rvm_mobilenetv3_fp32.onnx"


def test_ensure_model_downloads_once_and_verifies_the_hash(tmp_path):
    payload = b"fake onnx"
    calls = []

    def download(url, dest, on_progress):
        calls.append(url)
        dest.write_bytes(payload)

    good = masking.ensure_model(tmp_path, download=download, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert good == masking.cached_model_path(tmp_path) and good.read_bytes() == payload
    masking.ensure_model(tmp_path, download=download, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert calls == [masking.MODEL_URL]  # cached copy reused
    with pytest.raises(masking.MaskingError):
        masking.ensure_model(tmp_path / "other", download=download, expected_sha256="0" * 64)
    assert not masking.cached_model_path(tmp_path / "other").exists()


def test_write_masks_marks_people_black_and_reports_progress(tmp_path):
    pytest.importorskip("numpy")
    import numpy as np

    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        _png_rgb(images / f"frame_{i:06d}.png", 8, 4, (200, 100, 50))
    masks = tmp_path / "masks"
    seen = []

    def matte(rgb, state):
        seen.append(state)
        alpha = np.zeros(rgb.shape[:2], dtype=np.float32)
        alpha[:, :4] = 0.9  # a person in the left half
        return alpha, "carried"

    progress = []
    written = masking.write_masks(sorted(images.iterdir()), masks, matte, on_progress=lambda c, t: progress.append((c, t)),
                                  should_cancel=lambda: False)
    assert written == 3
    assert seen == [None, "carried", "carried"]  # the recurrent state flows through a video
    assert progress[-1] == (3, 3)
    from PIL import Image

    mask = np.asarray(Image.open(masks / "frame_000001.png"))
    assert mask.shape == (4, 8) and mask.dtype == np.uint8
    assert (mask[:, :4] == 0).all() and (mask[:, 4:] == 255).all()


def test_write_masks_resets_state_for_photo_sets_and_honours_cancel(tmp_path):
    pytest.importorskip("numpy")
    import numpy as np

    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        _png_rgb(images / f"IMG_{i}.png", 4, 4, (0, 0, 0))
    seen = []

    def matte(rgb, state):
        seen.append(state)
        return np.zeros(rgb.shape[:2], dtype=np.float32), "x"

    masking.write_masks(sorted(images.iterdir()), tmp_path / "m", matte, on_progress=lambda c, t: None,
                        should_cancel=lambda: False, carry_state=False)
    assert seen == [None, None]
    with pytest.raises(masking.MaskingCancelled):
        masking.write_masks(sorted(images.iterdir()), tmp_path / "m2", matte, on_progress=lambda c, t: None,
                            should_cancel=lambda: True)


def test_missing_packages_are_named(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("onnxruntime", "numpy", "PIL"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert masking.missing_packages() == ["numpy", "pillow", "onnxruntime"]


@pytest.mark.slow
def test_real_model_masks_the_camera_operator():
    pytest.importorskip("onnxruntime")
    import numpy as np

    model = Path("D:/AIProjects/tools/models/rvm_mobilenetv3_fp32.onnx")
    frame = Path("D:/AIProjects/v2s-task11/imgs/frame_000031.jpg")
    if not model.is_file() or not frame.is_file():
        pytest.skip("local model or frame not available")
    from PIL import Image

    rgb = np.asarray(Image.open(frame).convert("RGB"))
    matte = masking.RvmMatter(model)
    alpha, state = matte(rgb, None)
    assert alpha.shape == rgb.shape[:2]
    covered = float((alpha > 0.5).mean())
    assert 0.02 < covered < 0.6  # the operator holding the camera is visible but not the whole frame
    assert state is not None
