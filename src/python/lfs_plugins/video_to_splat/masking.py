# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic people masking with Robust Video Matting (RVM) through ONNX Runtime.

The model is the official MobileNetV3 export from the RVM release (GPL-3.0, like this
project), downloaded once with consent and pinned by SHA-256 like COLMAP. Inference runs
on the CPU: 120 frames at 1920x960 take about a minute, and the GPU is left to COLMAP.

Masks follow the trainer's convention (see MaskMode in core/parameters.hpp): 0 = ignore
this pixel, 255 = keep. A person therefore becomes a black region in ``masks/<stem>.png``
and training with mask mode "ignore" simply never sees it, so each person leaves a hole
instead of a smear. numpy, Pillow and onnxruntime are imported lazily so that the planning
code and the tests run without them.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from .text import _tr

MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
MODEL_SHA256 = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"
MODEL_SIZE_MB = 15
MODEL_FILE = "rvm_mobilenetv3_fp32.onnx"
# (import name, pip name): what lf.packages must install before the first run.
REQUIRED_PACKAGES = (("numpy", "numpy"), ("PIL", "pillow"), ("onnxruntime", "onnxruntime"))
PERSON_THRESHOLD = 0.5
# RVM's downsample ratio for its segmentation branch; 0.25 suits 1080p-class frames.
DOWNSAMPLE_RATIO = 0.25

DownloadFn = Callable[[str, Path, Callable[[str], None]], None]
MatteFn = Callable[[Any, Any], tuple[Any, Any]]  # (rgb uint8 HxWx3, state) -> (alpha float HxW, state)


class MaskingError(Exception):
    """People masking cannot run. str(error) is a user-facing message."""


class MaskingCancelled(Exception):
    pass


def missing_packages() -> list[str]:
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def cached_model_path(cache_dir: Path) -> Path:
    return cache_dir / "models" / MODEL_FILE


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(cache_dir: Path, *, download: DownloadFn, on_progress: Callable[[str], None] = lambda _m: None,
                 expected_sha256: Optional[str] = None) -> Path:
    """Return the cached model, downloading and hash-checking it the first time."""
    expected_sha256 = expected_sha256 or MODEL_SHA256  # resolved at call time so tests can pin another hash
    target = cached_model_path(cache_dir)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="rvm-dl-", dir=str(target.parent)))
    try:
        archive = staging / MODEL_FILE
        download(MODEL_URL, archive, on_progress)
        actual = _sha256_of(archive)
        if actual.lower() != expected_sha256.lower():
            raise MaskingError(_tr("video_to_splat.error.model_hash_mismatch",
                                   "The downloaded people-masking model failed its SHA-256 check and was discarded."))
        shutil.move(str(archive), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


class RvmMatter:
    """Callable matting backend: alpha matte of the people in a frame, with RVM's recurrent state."""

    def __init__(self, model_path: Path, downsample_ratio: float = DOWNSAMPLE_RATIO):
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self._ratio = np.asarray([downsample_ratio], dtype=np.float32)
        self._np = np

    def __call__(self, rgb: Any, state: Optional[tuple]) -> tuple[Any, tuple]:
        np = self._np
        src = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        if state is None:
            state = tuple(np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4))
        feeds = {"src": src, "r1i": state[0], "r2i": state[1], "r3i": state[2], "r4i": state[3],
                 "downsample_ratio": self._ratio}
        _fgr, pha, r1, r2, r3, r4 = self._session.run(None, feeds)
        return pha[0, 0], (r1, r2, r3, r4)


def write_masks(images: list[Path], masks_dir: Path, matte: MatteFn, *, on_progress: Callable[[int, int], None],
                should_cancel: Callable[[], bool], carry_state: bool = True) -> int:
    """Write one PNG mask per image (0 where a person is, 255 elsewhere). Returns the number written.

    ``carry_state`` keeps RVM's temporal memory across consecutive video frames; photo sets reset it."""
    import numpy as np
    from PIL import Image

    masks_dir.mkdir(parents=True, exist_ok=True)
    state = None
    total = len(images)
    for index, image in enumerate(images, start=1):
        if should_cancel():
            raise MaskingCancelled()
        rgb = np.asarray(Image.open(image).convert("RGB"))
        alpha, state = matte(rgb, state if carry_state else None)
        mask = np.where(alpha > PERSON_THRESHOLD, 0, 255).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(masks_dir / (image.stem + ".png"))
        on_progress(index, total)
    return total


def count_masked_people(masks_dir: Path) -> int:
    """How many masks contain at least one masked pixel; a cheap sanity figure for the log."""
    import numpy as np
    from PIL import Image

    count = 0
    for mask in sorted(masks_dir.glob("*.png")):
        if (np.asarray(Image.open(mask)) == 0).any():
            count += 1
    return count
