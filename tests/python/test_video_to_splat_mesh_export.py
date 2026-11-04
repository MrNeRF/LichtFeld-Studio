# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point cloud from Gaussians, PLY mesh reading, and the GLB / USDZ writers."""
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import mesh_export  # noqa: E402

np = pytest.importorskip("numpy")


def test_point_cloud_normals_follow_the_thinnest_axis_and_opacity_filters():
    # Gaussian 0: identity rotation, thinnest along z -> normal (0, 0, 1).
    # Gaussian 1: 90 degrees about x (w=cos45, x=sin45), thinnest along local y -> world (0, 0, 1).
    # Gaussian 2: transparent, dropped.
    s = np.sqrt(0.5)
    cloud = mesh_export.gaussians_to_point_cloud(
        means=[[0, 0, 0], [1, 1, 1], [5, 5, 5]], scales=[[1, 1, 0.01], [1, 0.01, 1], [1, 1, 1]],
        rotations=[[1, 0, 0, 0], [s, s, 0, 0], [1, 0, 0, 0]], colors=[[1, 0, 0], [0, 0.5, 1.2], [0, 0, 0]],
        opacity=[0.9, 0.9, 0.1])
    assert cloud["positions"].shape == (2, 3)
    assert np.allclose(cloud["normals"][0], [0, 0, 1], atol=1e-6)
    assert np.allclose(np.abs(cloud["normals"][1]), [0, 0, 1], atol=1e-6)
    assert cloud["colors"].tolist() == [[255, 0, 0], [0, 128, 255]]
    empty = mesh_export.gaussians_to_point_cloud([[0, 0, 0]], [[1, 1, 1]], [[1, 0, 0, 0]], [[0, 0, 0]], [0.0])
    assert len(empty["positions"]) == 0


def test_point_cloud_ply_round_trips_through_the_mesh_reader(tmp_path):
    cloud = {"positions": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
             "normals": np.array([[0, 0, 1]] * 3, np.float32), "colors": np.array([[1, 2, 3]] * 3, np.uint8)}
    assert mesh_export.write_point_cloud_ply(tmp_path / "dense.ply", cloud) == 3
    head = (tmp_path / "dense.ply").read_bytes()[:200].decode("ascii", "replace")
    assert "element vertex 3" in head and "property float nx" in head and "property uchar red" in head
    back = mesh_export.read_ply_mesh(tmp_path / "dense.ply")
    assert np.allclose(back["positions"], cloud["positions"]) and back["colors"].tolist() == [[1, 2, 3]] * 3
    assert back["faces"].shape == (0, 3)


def _poisson_style_ply(path: Path) -> None:
    """What COLMAP's poisson_mesher writes: x y z value red green blue, then list int int faces (one quad)."""
    vertices = [(0, 0, 0, 0.5, 255, 0, 0), (1, 0, 0, 0.5, 0, 255, 0), (1, 1, 0, 0.5, 0, 0, 255), (0, 1, 0, 0.5, 9, 9, 9)]
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
              "property float z\nproperty float value\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
              "element face 1\nproperty list int int vertex_indices\nend_header\n")
    body = b"".join(struct.pack("<ffffBBB", *v) for v in vertices) + struct.pack("<iiiii", 4, 0, 1, 2, 3)
    path.write_bytes(header.encode() + body)


def test_mesh_reader_triangulates_polygons(tmp_path):
    _poisson_style_ply(tmp_path / "mesh.ply")
    mesh = mesh_export.read_ply_mesh(tmp_path / "mesh.ply")
    assert mesh["positions"].shape == (4, 3)
    assert mesh["faces"].tolist() == [[0, 1, 2], [0, 2, 3]]
    assert mesh["colors"][0].tolist() == [255, 0, 0]


def test_glb_is_a_valid_gltf2_binary_with_colors(tmp_path):
    _poisson_style_ply(tmp_path / "mesh.ply")
    mesh = mesh_export.read_ply_mesh(tmp_path / "mesh.ply")
    glb = tmp_path / "mesh.glb"
    mesh_export.write_glb(glb, mesh["positions"], mesh["faces"], mesh["colors"], name="room")
    raw = glb.read_bytes()
    magic, version, total = struct.unpack("<4sII", raw[:12])
    assert magic == b"glTF" and version == 2 and total == len(raw) and total % 4 == 0
    doc = mesh_export.read_glb(glb)
    gltf, blob = doc["json"], doc["bin"]
    assert gltf["asset"]["version"] == "2.0" and gltf["meshes"][0]["name"] == "room"
    primitive = gltf["meshes"][0]["primitives"][0]
    assert primitive["mode"] == 4 and set(primitive["attributes"]) == {"POSITION", "COLOR_0"}
    position = gltf["accessors"][primitive["attributes"]["POSITION"]]
    assert position["count"] == 4 and position["type"] == "VEC3" and position["min"] == [0.0, 0.0, 0.0]
    color = gltf["accessors"][primitive["attributes"]["COLOR_0"]]
    assert color["type"] == "VEC4" and color["componentType"] == 5121 and color["normalized"] is True
    indices = gltf["accessors"][primitive["indices"]]
    assert indices["count"] == 6 and indices["componentType"] == 5125
    assert gltf["buffers"][0]["byteLength"] == len(blob)
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0 and view["byteOffset"] + view["byteLength"] <= len(blob)
    view = gltf["bufferViews"][indices["bufferView"]]
    assert struct.unpack("<6I", blob[view["byteOffset"]:view["byteOffset"] + 24]) == (0, 1, 2, 0, 2, 3)


def test_usdz_is_a_stored_zip_with_an_aligned_usda_mesh(tmp_path):
    _poisson_style_ply(tmp_path / "mesh.ply")
    mesh = mesh_export.read_ply_mesh(tmp_path / "mesh.ply")
    usdz = tmp_path / "mesh.usdz"
    mesh_export.write_usdz(usdz, mesh["positions"], mesh["faces"], mesh["colors"], name="room")
    with zipfile.ZipFile(usdz) as archive:
        (info,) = archive.infolist()
        assert info.compress_type == zipfile.ZIP_STORED and info.filename == "room.usda"
        data_offset = info.header_offset + 30 + len(info.filename) + len(info.extra)
        assert data_offset % 64 == 0
    usda = mesh_export.read_usdz(usdz)
    assert usda.startswith("#usda 1.0")
    assert 'def Mesh "geometry"' in usda and "faceVertexIndices = [0, 1, 2, 0, 2, 3]" in usda
    assert "faceVertexCounts = [3, 3]" in usda and "primvars:displayColor" in usda
    assert 'defaultPrim = "room"' in usda
