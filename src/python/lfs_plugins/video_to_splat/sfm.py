# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP command planning and binary model reading. No subprocess calls here.

Option names were checked against `colmap <command> -h` of the 4.2.1 Windows build.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Optional

from .probe import Mode

CAMERA_MODELS: dict[str, str] = {
    "equirectangular": "EQUIRECTANGULAR",  # COLMAP >= 4.1
    "perspective": "OPENCV",
}

IMAGES_DIR = "images"
DATABASE_NAME = "database.db"
SPARSE_DIR = "sparse"


def dataset_paths(dataset: Path) -> tuple[Path, Path, Path]:
    """(images folder, database file, sparse folder) inside a dataset folder."""
    return dataset / IMAGES_DIR, dataset / DATABASE_NAME, dataset / SPARSE_DIR


def plan_commands(*, colmap: Path, dataset: Path, mode: Mode, overlap: int, use_gpu: bool,
                  max_image_size: int = 3200) -> list[list[str]]:
    """Return argv lists for feature_extractor, sequential_matcher, global_mapper in order."""
    images, database, sparse = dataset_paths(dataset)
    gpu = "1" if use_gpu else "0"
    exe = str(colmap)
    common = ["--log_target", "stderr"]
    feature = [
        exe, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images),
        "--ImageReader.camera_model", CAMERA_MODELS[mode],
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", gpu,
        "--FeatureExtraction.max_image_size", str(max_image_size),
        *common,
    ]
    match = [
        exe, "sequential_matcher",
        "--database_path", str(database),
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", "1",
        "--SequentialMatching.loop_detection", "0",
        "--FeatureMatching.use_gpu", gpu,
        *common,
    ]
    mapper = [
        exe, "global_mapper",
        "--database_path", str(database),
        "--image_path", str(images),
        "--output_path", str(sparse),
        *common,
    ]
    return [feature, match, mapper]


def _count_rows_with_data(database: Path, table: str) -> int:
    """Rows of ``table`` whose ``rows`` column is positive; 0 for a missing or unreadable database."""
    if not database.is_file():
        return 0
    try:
        db = sqlite3.connect(database)
    except sqlite3.Error:
        return 0
    try:
        return int(db.execute(f"SELECT COUNT(*) FROM {table} WHERE rows > 0").fetchone()[0])
    except sqlite3.Error:
        return 0
    finally:
        db.close()


def count_images_with_features(database: Path) -> int:
    """Images that got SIFT descriptors. The official CUDA build fails on every image when the GPU has no
    kernels for it, yet feature_extractor still exits 0, so the exit code alone proves nothing."""
    return _count_rows_with_data(database, "descriptors")


def count_verified_pairs(database: Path) -> int:
    """Image pairs with geometrically verified matches, which is what the mapper consumes."""
    return _count_rows_with_data(database, "two_view_geometries")


def clear_matches(database: Path) -> None:
    """Drop match results so a rerun of the matcher starts clean instead of skipping stale pairs."""
    if not database.is_file():
        return
    try:
        db = sqlite3.connect(database)
    except sqlite3.Error:
        return
    try:
        for table in ("matches", "two_view_geometries"):
            try:
                db.execute(f"DELETE FROM {table}")
            except sqlite3.Error:
                pass
        db.commit()
    finally:
        db.close()


def _read_leading_uint64(path: Path) -> int:
    try:
        with open(path, "rb") as handle:
            header = handle.read(8)
    except OSError:
        return 0
    if len(header) < 8:
        return 0
    return struct.unpack("<Q", header)[0]


def read_registered_image_count(images_bin: Path) -> int:
    """First uint64 of images.bin is the number of registered images."""
    return _read_leading_uint64(images_bin)


def read_camera_count(cameras_bin: Path) -> int:
    """First uint64 of cameras.bin is the number of cameras."""
    return _read_leading_uint64(cameras_bin)


def find_largest_model(sparse: Path) -> Optional[tuple[Path, int]]:
    """global_mapper writes one numbered folder per connected component; pick the biggest."""
    if not sparse.is_dir():
        return None
    best: Optional[tuple[Path, int]] = None
    for child in sorted(sparse.iterdir()):
        if not child.is_dir():
            continue
        count = read_registered_image_count(child / "images.bin")
        if count <= 0:
            continue
        if best is None or count > best[1]:
            best = (child, count)
    return best
