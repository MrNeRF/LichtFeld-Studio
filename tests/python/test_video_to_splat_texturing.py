# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Projection conventions, view selection and the baked texture atlas with its GLB / USDZ."""
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import mesh_export, texturing  # noqa: E402

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

IDENTITY = [1.0, 0.0, 0.0, 0.0]
ORIGIN = [0.0, 0.0, 0.0]


def test_equirectangular_projection_matches_the_rasterizer():
    camera = {"model": "EQUIRECTANGULAR", "width": 1920, "height": 960, "params": [1920, 960]}
    uv, ok = texturing.project([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]], camera)
    assert ok.tolist() == [True, True, True, False]
    assert np.allclose(uv[0], [960, 480])      # forward: image center
    assert np.allclose(uv[1], [1440, 480])     # +x: azimuth 90 degrees, three quarters across
    assert np.allclose(uv[2], [960, 960])      # +y (down in COLMAP): bottom edge


def test_pinhole_and_opencv_projection():
    pinhole = {"model": "PINHOLE", "width": 64, "height": 64, "params": [32, 32, 32, 32]}
    uv, ok = texturing.project([[0, 0, 2], [1, 0, 2], [0, 0, -1], [5, 0, 1]], pinhole)
    assert ok.tolist() == [True, True, False, False]
    assert np.allclose(uv[0], [32, 32]) and np.allclose(uv[1], [48, 32])
    opencv = {"model": "OPENCV", "width": 64, "height": 64, "params": [32, 32, 32, 32, 0, 0, 0, 0]}
    uv2, _ = texturing.project([[1, 0, 2]], opencv)
    assert np.allclose(uv2[0], [48, 32])
    simple = {"model": "SIMPLE_PINHOLE", "width": 64, "height": 64, "params": [32, 32, 32]}
    uv3, _ = texturing.project([[1, 0, 2]], simple)
    assert np.allclose(uv3[0], [48, 32])


def test_world_to_camera_follows_colmap():
    # 90 degrees about y: world +x maps to camera -z, and a translation shifts the result
    s = np.sqrt(0.5)
    cam = texturing.world_to_camera([[1, 0, 0]], [s, 0, s, 0], [0, 0, 1])
    assert np.allclose(cam[0], [0, 0, 0], atol=1e-9)
    assert np.allclose(texturing.camera_center([s, 0, s, 0], [0, 0, 1]), [1, 0, 0], atol=1e-9)


def _quad_scene():
    # A unit quad at z = 2 facing a pinhole camera at the origin; the frame is red on the left, blue on the right.
    positions = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    cameras = {1: {"model": "PINHOLE", "width": 64, "height": 64, "params": [32, 32, 32, 32]}}
    images = [{"image_id": 1, "name": "a.png", "camera_id": 1, "qvec": IDENTITY, "tvec": ORIGIN},
              {"image_id": 2, "name": "far.png", "camera_id": 1, "qvec": IDENTITY, "tvec": [0, 0, 10]}]
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :32] = (255, 0, 0)
    frame[:, 32:] = (0, 0, 255)
    return positions, faces, images, cameras, frame


def test_choose_views_prefers_the_facing_nearby_frame():
    positions, faces, images, cameras, _ = _quad_scene()
    views = texturing.choose_views(positions, faces, images, cameras)
    assert views.tolist() == [0, 0]  # the far frame (center at z = -10) still sees the quad but scores lower
    assert texturing.choose_views(positions, faces, [images[1]], cameras).tolist() == [0, 0]
    # x_cam = X + t: a camera whose center sits past the quad (t = (0, 0, -3), center z = 3) cannot see it.
    behind = [{"image_id": 3, "name": "x", "camera_id": 1, "qvec": IDENTITY, "tvec": [0, 0, -3]}]
    assert texturing.choose_views(positions, faces, behind, cameras).tolist() == [-1, -1]


def test_bake_atlas_samples_the_right_frame_colors(tmp_path):
    positions, faces, images, cameras, frame = _quad_scene()
    loads = []

    def load_image(name):
        loads.append(name)
        return frame

    baked = texturing.bake_atlas(positions, faces, images, cameras, load_image, patch=16)
    assert loads == ["a.png"]  # one load per used frame
    atlas, uvs = baked["atlas"], baked["uvs"]
    assert uvs.shape == (2, 3, 2) and atlas.shape[0] == atlas.shape[1] == baked["size"]
    # Corner 0 of face 0 is world (-1, -1, 2): left half of the frame -> red at its atlas pixel.
    u, v = uvs[0, 0]
    px, py = int(u * baked["size"]), int(v * baked["size"])
    assert atlas[py + 1, px + 1].tolist() == [255, 0, 0]
    # Corner 1 of face 0 is world (1, -1, 2): right half -> blue just inside that corner.
    u, v = uvs[0, 1]
    px, py = int(u * baked["size"]), int(v * baked["size"])
    assert atlas[py + 1, px - 2].tolist() == [0, 0, 255]
    # Pixels inside the triangle's patch are written, the opposite corner of the square stays untouched grey.
    for x, y in ((1, 1), (13, 1), (1, 13), (5, 5)):
        assert atlas[y, x].tolist() != [128, 128, 128], (x, y)
    assert atlas[14, 14].tolist() == [128, 128, 128]


def test_textured_glb_and_usdz_carry_the_atlas(tmp_path):
    positions, faces, images, cameras, frame = _quad_scene()
    result = texturing.texture_mesh(tmp_path, positions, faces, images, cameras, lambda name: frame, name="quad")
    assert result == {"faces": 2, "textured_faces": 2, "atlas_size": result["atlas_size"]}
    assert (tmp_path / "atlas.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    doc = mesh_export.read_glb(tmp_path / "mesh_textured.glb")
    gltf = doc["json"]
    primitive = gltf["meshes"][0]["primitives"][0]
    assert set(primitive["attributes"]) == {"POSITION", "TEXCOORD_0"} and primitive["material"] == 0
    assert gltf["accessors"][primitive["attributes"]["POSITION"]]["count"] == 6  # unwelded corners
    assert gltf["accessors"][primitive["attributes"]["TEXCOORD_0"]]["type"] == "VEC2"
    assert gltf["images"][0]["mimeType"] == "image/png" and gltf["textures"][0]["source"] == 0
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    assert doc["bin"][view["byteOffset"]:view["byteOffset"] + 8] == b"\x89PNG\r\n\x1a\n"
    raw = (tmp_path / "mesh_textured.glb").read_bytes()
    assert struct.unpack("<I", raw[8:12])[0] == len(raw)
    with zipfile.ZipFile(tmp_path / "mesh_textured.usdz") as archive:
        names = [i.filename for i in archive.infolist()]
        assert names == ["quad.usda", "atlas.png"]
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert (info.header_offset + 30 + len(info.filename) + len(info.extra)) % 64 == 0
        usda = archive.read("quad.usda").decode("utf-8")
    assert 'uniform token info:id = "UsdUVTexture"' in usda and "asset inputs:file = @atlas.png@" in usda
    assert "primvars:st" in usda and 'rel material:binding = </quad/material>' in usda
