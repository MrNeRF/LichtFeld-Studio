# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime checks for lf.video. Needs the built lichtfeld module (LFS_BUILD_DIR on sys.path)."""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

lf = pytest.importorskip("lichtfeld")
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "data" / "video" / "vp9_timestamps.webm"  # 128x128, 185 frames, 30 fps
SMALL = ROOT / "tests" / "data" / "video" / "h264_420.mp4"  # 64x48, 5 frames


def test_probe_reports_dimensions():
    info = lf.video.probe(str(FIXTURE))
    assert info["width"] == 128
    assert info["height"] == 128
    assert info["spherical"] is False
    assert info["fps"] > 0
    assert info["duration"] > 0
    assert info["rotation"] == 0


def test_probe_small_fixture():
    info = lf.video.probe(str(SMALL))
    assert (info["width"], info["height"]) == (64, 48)


def test_probe_missing_file_raises():
    with pytest.raises(RuntimeError):
        lf.video.probe(str(ROOT / "tests" / "data" / "video" / "does_not_exist.mp4"))


def test_extract_frames_writes_jpegs_and_reports_progress(tmp_path):
    calls = []
    result = lf.video.extract_frames(
        video_path=str(FIXTURE),
        output_dir=str(tmp_path),
        fps=5.0,
        jpg_quality=90,
        sharpness_enabled=False,
        on_progress=lambda cur, total, discarded: calls.append((cur, total, discarded)),
    )
    assert result["outcome"] == "completed", result["error"]
    frames = sorted(tmp_path.glob("frame_*.jpg"))
    assert len(frames) >= 25
    assert calls and calls[-1][0] == calls[-1][1]


def test_extract_frames_scales_down(tmp_path):
    result = lf.video.extract_frames(
        video_path=str(FIXTURE), output_dir=str(tmp_path), fps=2.0, scale=0.5, sharpness_enabled=False)
    assert result["outcome"] == "completed", result["error"]
    frame = next(iter(sorted(tmp_path.glob("frame_*.jpg"))))
    data = frame.read_bytes()
    assert data[:2] == b"\xff\xd8"


def test_extract_frames_cancels(tmp_path):
    result = lf.video.extract_frames(
        video_path=str(FIXTURE),
        output_dir=str(tmp_path),
        fps=30.0,
        should_cancel=lambda: True,
    )
    assert result["outcome"] == "cancelled"


def test_extract_frames_missing_video_fails(tmp_path):
    result = lf.video.extract_frames(
        video_path=str(ROOT / "tests" / "data" / "video" / "does_not_exist.mp4"), output_dir=str(tmp_path))
    assert result["outcome"] == "failed"
    assert result["error"]


def test_extract_frames_window_mode_keeps_one_frame_per_interval(tmp_path):
    # Window mode never empties an interval: the sharpest candidate is kept even on flat footage,
    # where a fixed edge-ratio threshold would discard everything.
    result = lf.video.extract_frames(
        video_path=str(FIXTURE), output_dir=str(tmp_path), fps=5.0, sharpness_enabled=True,
        sharpness_window_mode=True, sharpness_window_candidates=-1)
    assert result["outcome"] == "completed", result["error"]
    frames = sorted(tmp_path.glob("frame_*.jpg"))
    assert 25 <= len(frames) <= 32
