# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure planning tests for the video-to-splat pipeline: 360 detection, presets, COLMAP argv, model readers."""

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import probe, sfm  # noqa: E402


def test_decide_mode_from_aspect_and_metadata():
    assert probe.decide_mode({"width": 5760, "height": 2880, "spherical": False}, "auto") == "equirectangular"
    assert probe.decide_mode({"width": 5700, "height": 2880, "spherical": False}, "auto") == "equirectangular"
    assert probe.decide_mode({"width": 3840, "height": 2160, "spherical": False}, "auto") == "perspective"
    assert probe.decide_mode({"width": 3840, "height": 2160, "spherical": True}, "auto") == "equirectangular"
    assert probe.decide_mode({"width": 5760, "height": 2880, "spherical": True}, "off") == "perspective"
    assert probe.decide_mode({"width": 1920, "height": 1080, "spherical": False}, "on") == "equirectangular"
    assert probe.decide_mode({}, "auto") == "perspective"


def test_presets_and_scale():
    assert probe.PRESETS["balanced"] == probe.Preset(fps=2.0, max_width=4096, overlap=15)
    assert probe.PRESETS["fast"] == probe.Preset(fps=1.0, max_width=2880, overlap=10)
    assert probe.PRESETS["high"] == probe.Preset(fps=3.0, max_width=0, overlap=25)
    assert probe.DEFAULT_PRESET == "balanced"
    assert probe.plan_frame_scale(5760, 2880) == 0.5
    assert probe.plan_frame_scale(1920, 4096) == 1.0
    assert probe.plan_frame_scale(5760, 0) == 1.0
    assert probe.plan_frame_scale(0, 4096) == 1.0


def test_plan_commands_equirectangular(tmp_path):
    cmds = sfm.plan_commands(
        colmap=Path("C:/tools/colmap.exe"), dataset=tmp_path, mode="equirectangular", overlap=15, use_gpu=True
    )
    feat, match, mapper = cmds
    assert feat[0].endswith("colmap.exe")
    assert feat[1] == "feature_extractor"
    assert feat[feat.index("--ImageReader.camera_model") + 1] == "EQUIRECTANGULAR"
    assert feat[feat.index("--ImageReader.single_camera") + 1] == "1"
    assert feat[feat.index("--FeatureExtraction.use_gpu") + 1] == "1"
    assert feat[feat.index("--image_path") + 1] == str(tmp_path / "images")
    assert match[1] == "sequential_matcher"
    assert match[match.index("--SequentialMatching.overlap") + 1] == "15"
    assert match[match.index("--SequentialMatching.quadratic_overlap") + 1] == "1"
    assert match[match.index("--SequentialMatching.loop_detection") + 1] == "0"
    assert match[match.index("--FeatureMatching.use_gpu") + 1] == "1"
    assert mapper[1] == "global_mapper"
    assert mapper[mapper.index("--output_path") + 1] == str(tmp_path / "sparse")
    assert mapper[mapper.index("--image_path") + 1] == str(tmp_path / "images")
    for cmd in cmds:
        assert cmd[cmd.index("--database_path") + 1] == str(tmp_path / "database.db")
        assert cmd[cmd.index("--log_target") + 1] == "stderr"


def test_plan_commands_perspective_cpu(tmp_path):
    feat, match, _ = sfm.plan_commands(
        colmap=Path("colmap"), dataset=tmp_path, mode="perspective", overlap=10, use_gpu=False
    )
    assert feat[feat.index("--ImageReader.camera_model") + 1] == "OPENCV"
    assert feat[feat.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert match[match.index("--FeatureMatching.use_gpu") + 1] == "0"


def test_read_binary_model_counts(tmp_path):
    images = tmp_path / "images.bin"
    images.write_bytes(struct.pack("<Q", 42) + b"\x00" * 16)
    cameras = tmp_path / "cameras.bin"
    cameras.write_bytes(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 17, 5760, 2880))
    assert sfm.read_registered_image_count(images) == 42
    assert sfm.read_camera_count(cameras) == 1
    assert sfm.read_registered_image_count(tmp_path / "missing.bin") == 0
    (tmp_path / "short.bin").write_bytes(b"\x01\x02")
    assert sfm.read_registered_image_count(tmp_path / "short.bin") == 0


def test_find_largest_model(tmp_path):
    for idx, n in ((0, 10), (1, 80), (2, 5)):
        d = tmp_path / "sparse" / str(idx)
        d.mkdir(parents=True)
        (d / "images.bin").write_bytes(struct.pack("<Q", n))
    (tmp_path / "sparse" / "notes.txt").write_text("ignored")
    best = sfm.find_largest_model(tmp_path / "sparse")
    assert best == (tmp_path / "sparse" / "1", 80)
    assert sfm.find_largest_model(tmp_path / "nope") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sfm.find_largest_model(empty) is None


def _colmap_database(path: Path, descriptor_rows: list[int], geometry_rows: list[int]) -> None:
    """Minimal COLMAP schema: enough for the stage checks to count real rows."""
    import sqlite3

    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, type INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
        db.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        db.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        for idx, rows in enumerate(descriptor_rows, start=1):
            db.execute("INSERT INTO descriptors VALUES (?, 0, ?, 128, NULL)", (idx, rows))
        for idx, rows in enumerate(geometry_rows, start=1):
            db.execute("INSERT INTO matches VALUES (?, ?, 2, NULL)", (idx, rows))
            db.execute("INSERT INTO two_view_geometries VALUES (?, ?, 2, NULL)", (idx, rows))
        db.commit()
    finally:
        db.close()


def test_stage_checks_count_images_with_features_and_verified_pairs(tmp_path):
    database = tmp_path / "database.db"
    _colmap_database(database, descriptor_rows=[812, 0, 455], geometry_rows=[0, 31, 7, 0])
    assert sfm.count_images_with_features(database) == 2
    assert sfm.count_verified_pairs(database) == 2


def test_clear_matches_keeps_features(tmp_path):
    database = tmp_path / "database.db"
    _colmap_database(database, descriptor_rows=[812, 455], geometry_rows=[31, 7])
    sfm.clear_matches(database)
    assert sfm.count_images_with_features(database) == 2
    assert sfm.count_verified_pairs(database) == 0
    sfm.clear_matches(tmp_path / "missing.db")  # no-op


def test_stage_checks_tolerate_missing_or_empty_database(tmp_path):
    assert sfm.count_images_with_features(tmp_path / "missing.db") == 0
    assert sfm.count_verified_pairs(tmp_path / "missing.db") == 0
    (tmp_path / "garbage.db").write_bytes(b"not a database")
    assert sfm.count_images_with_features(tmp_path / "garbage.db") == 0
    assert sfm.count_verified_pairs(tmp_path / "garbage.db") == 0
