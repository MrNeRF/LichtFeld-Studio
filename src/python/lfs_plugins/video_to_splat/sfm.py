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
PACKAGE_DIR = "package"  # beside the project: sparse.ply, cameras.json, dense.ply, mesh.ply, textured/
DENSE_DIR = "dense"      # COLMAP dense workspace (undistorted images, depth maps)

# COLMAP camera model ids -> (name, parameter count); mirrors src/io/formats/colmap.cpp.
CAMERA_MODELS_BY_ID: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12), 11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
    17: ("EQUIRECTANGULAR", 2),
}


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


LEVELED_DIR = "leveled"


def plan_level_command(*, colmap: Path, dataset: Path, mode: Mode) -> list[str]:
    """argv for model_orientation_aligner: gravity-align sparse/0 into sparse/leveled.

    MANHATTAN-WORLD finds vanishing points from line segments, which an equirectangular projection bends
    into curves, so 360 footage uses the mean camera orientation instead (the camera is held upright)."""
    images, _database, sparse = dataset_paths(dataset)
    method = "IMAGE-ORIENTATION" if mode == "equirectangular" else "MANHATTAN-WORLD"
    return [
        str(colmap), "model_orientation_aligner",
        "--image_path", str(images),
        "--input_path", str(sparse / "0"),
        "--output_path", str(sparse / LEVELED_DIR),
        "--method", method,
        "--max_image_size", "1024",
        "--log_target", "stderr",
    ]


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


def plan_package_command(*, colmap: Path, dataset: Path, package: Path) -> list[str]:
    """argv for model_converter: the sparse points as a colored PLY for other tools."""
    _images, _database, sparse = dataset_paths(dataset)
    return [str(colmap), "model_converter", "--input_path", str(sparse / "0"),
            "--output_path", str(package / "sparse.ply"), "--output_type", "PLY", "--log_target", "stderr"]


def plan_mesh_commands(*, colmap: Path, dataset: Path, dense: Path, package: Path) -> list[list[str]]:
    """argv lists for the dense chain: undistort, patch-match stereo (CUDA), fusion, Poisson mesh, texture."""
    images, _database, sparse = dataset_paths(dataset)
    exe = str(colmap)
    common = ["--log_target", "stderr"]
    return [
        [exe, "image_undistorter", "--image_path", str(images), "--input_path", str(sparse / "0"),
         "--output_path", str(dense), "--output_type", "COLMAP", *common],
        [exe, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", *common],
        [exe, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(package / "dense.ply"), *common],
        [exe, "poisson_mesher", "--input_path", str(package / "dense.ply"), "--output_path", str(package / "mesh.ply"),
         *common],
        [exe, "mesh_texturer", "--workspace_path", str(dense), "--input_path", str(package / "mesh.ply"),
         "--output_path", str(package / "textured"), *common],
    ]


def read_cameras_bin(path: Path) -> dict[int, dict]:
    """cameras.bin -> {camera_id: {model, width, height, params}}; {} when missing or unknown."""
    cameras: dict[int, dict] = {}
    try:
        with open(path, "rb") as handle:
            (count,) = struct.unpack("<Q", handle.read(8))
            for _ in range(count):
                camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
                name, num_params = CAMERA_MODELS_BY_ID.get(model_id, ("UNKNOWN", -1))
                if num_params < 0:
                    break
                params = list(struct.unpack("<%dd" % num_params, handle.read(8 * num_params)))
                cameras[camera_id] = {"model": name, "width": int(width), "height": int(height), "params": params}
    except (OSError, struct.error):
        return cameras
    return cameras


def read_images_bin(path: Path) -> list[dict]:
    """images.bin -> [{image_id, name, camera_id, qvec, tvec}] in file order; [] when missing."""
    images: list[dict] = []
    try:
        with open(path, "rb") as handle:
            (count,) = struct.unpack("<Q", handle.read(8))
            for _ in range(count):
                (image_id,) = struct.unpack("<i", handle.read(4))
                qvec = list(struct.unpack("<4d", handle.read(32)))
                tvec = list(struct.unpack("<3d", handle.read(24)))
                (camera_id,) = struct.unpack("<i", handle.read(4))
                name = bytearray()
                while True:
                    byte = handle.read(1)
                    if not byte or byte == b"\x00":
                        break
                    name += byte
                (num_points,) = struct.unpack("<Q", handle.read(8))
                handle.seek(24 * num_points, 1)
                images.append({"image_id": image_id, "name": name.decode("utf-8", "replace"), "camera_id": camera_id,
                               "qvec": qvec, "tvec": tvec})
    except (OSError, struct.error):
        return images
    return images


def write_cameras_json(model_dir: Path, path: Path) -> None:
    """Camera intrinsics and world-to-camera poses of the registered images, for tools without a COLMAP reader."""
    import json

    payload = {"cameras": {str(k): v for k, v in read_cameras_bin(model_dir / "cameras.bin").items()},
               "images": read_images_bin(model_dir / "images.bin"),
               "convention": "COLMAP: qvec (w, x, y, z) and tvec map world points into the camera frame"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


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
