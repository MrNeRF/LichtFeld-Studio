# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Texture atlas for the package mesh, baked from the source frames.

COLMAP's mesh_texturer cannot texture equirectangular cameras, so this does the job for any
camera the pipeline produces: every triangle picks the frame that sees it best (facing, near),
gets a small square patch in an atlas, and each patch pixel samples the frame where its 3D point
projects. Projection follows COLMAP: ``x_cam = R(q) X + t`` with (w, x, y, z) quaternions, the
PINHOLE / SIMPLE_PINHOLE / OPENCV models, and the equirectangular mapping the rasterizer uses
(azimuth = atan2(x, z), elevation = asin(y / |r|)). Triangles are unwelded so each corner owns a
UV, which keeps the glTF and USD writers trivial. numpy and Pillow are imported lazily.
"""
from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

PATCH = 10          # pixels per atlas patch, 1 px border included
MAX_ATLAS = 8192    # atlas side length limit


def quaternion_to_matrix(qvec: Any) -> Any:
    import numpy as np

    w, x, y, z = (float(v) for v in qvec)
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def world_to_camera(points: Any, qvec: Any, tvec: Any) -> Any:
    import numpy as np

    return np.asarray(points, dtype=np.float64) @ quaternion_to_matrix(qvec).T + np.asarray(tvec, dtype=np.float64)


def camera_center(qvec: Any, tvec: Any) -> Any:
    import numpy as np

    return -quaternion_to_matrix(qvec).T @ np.asarray(tvec, dtype=np.float64)


def project(cam_points: Any, camera: dict) -> tuple[Any, Any]:
    """Pixel coordinates (N, 2) and a visibility mask (in front and inside the image)."""
    import numpy as np

    p = np.asarray(cam_points, dtype=np.float64)
    model, width, height, params = camera["model"], camera["width"], camera["height"], list(camera["params"])
    if model == "EQUIRECTANGULAR":
        norm = np.linalg.norm(p, axis=1)
        ok = norm > 1e-9
        azimuth = np.arctan2(p[:, 0], p[:, 2])
        elevation = np.arcsin(np.clip(p[:, 1] / np.where(ok, norm, 1.0), -1.0, 1.0))
        u = (azimuth / (2.0 * np.pi) + 0.5) * width
        v = (elevation / np.pi + 0.5) * height
        return np.stack([u, v], 1), ok
    z = p[:, 2]
    ok = z > 1e-6
    safe_z = np.where(ok, z, 1.0)
    x, y = p[:, 0] / safe_z, p[:, 1] / safe_z
    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:  # PINHOLE, OPENCV and anything else that starts with fx fy cx cy
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    if model == "OPENCV" and len(params) >= 8:
        k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    u, v = fx * x + cx, fy * y + cy
    ok &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], 1), ok


def choose_views(positions: Any, faces: Any, images: list[dict], cameras: dict) -> Any:
    """Index into ``images`` of the best frame per face (-1 when no frame sees it)."""
    import numpy as np

    positions = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    corners = positions[faces]                      # (F, 3, 3)
    centroids = corners.mean(1)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    best_score = np.zeros(len(faces))
    best = np.full(len(faces), -1, dtype=np.int64)
    for index, image in enumerate(images):
        camera = cameras.get(image["camera_id"]) or cameras.get(str(image["camera_id"]))
        if camera is None:
            continue
        center = camera_center(image["qvec"], image["tvec"])
        to_camera = center - centroids
        distance = np.linalg.norm(to_camera, axis=1)
        facing = np.abs(np.einsum("ij,ij->i", normals, to_camera / np.maximum(distance[:, None], 1e-12)))
        _uv, visible = project(world_to_camera(centroids, image["qvec"], image["tvec"]), camera)
        score = np.where(visible, facing / (1.0 + distance), 0.0)
        better = score > best_score
        best_score[better] = score[better]
        best[better] = index
    return best


def atlas_layout(face_count: int, patch: int = PATCH) -> tuple[int, int]:
    """(patches per row, atlas side in pixels), square and power-of-two friendly."""
    import math

    per_row = max(1, math.ceil(math.sqrt(max(face_count, 1))))
    side = per_row * patch
    size = 1
    while size < side:
        size *= 2
    return per_row, min(size, MAX_ATLAS)


def bake_atlas(positions: Any, faces: Any, images: list[dict], cameras: dict,
               load_image: Callable[[str], Any], patch: int = PATCH,
               on_progress: Callable[[int, int], None] = lambda c, t: None) -> dict:
    """Atlas (H, W, 3) uint8, per-corner UVs (F, 3, 2) in glTF convention (v down), and the view per face."""
    import numpy as np

    positions = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_count = len(faces)
    per_row, size = atlas_layout(face_count, patch)
    if per_row * patch > size:
        raise ValueError("mesh has too many faces for one atlas (%d)" % face_count)
    atlas = np.full((size, size, 3), 128, dtype=np.uint8)
    views = choose_views(positions, faces, images, cameras)
    inner = patch - 2  # 1 px border on each side
    # The triangle fills the lower-left half of the inner square: corners (0,0), (1,0), (0,1).
    local = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pixel_offsets = np.stack(np.meshgrid(np.arange(patch), np.arange(patch), indexing="xy"), -1).reshape(-1, 2)  # (P², (x, y))
    # Barycentric coordinates of each patch pixel center in the triangle's local square, dilated by one pixel
    pix = (pixel_offsets + 0.5 - 1.0) / inner
    b1, b2 = pix[:, 0], pix[:, 1]
    inside = (b1 >= -1.0 / inner) & (b2 >= -1.0 / inner) & (b1 + b2 <= 1.0 + 1.0 / inner)
    b1c, b2c = np.clip(b1, 0, 1), np.clip(b2, 0, 1)
    over = np.maximum(b1c + b2c - 1.0, 0.0)
    b1c, b2c = b1c - over / 2, b2c - over / 2
    bary = np.stack([1 - b1c - b2c, b1c, b2c], 1)  # (P², 3)
    uvs = np.zeros((face_count, 3, 2), dtype=np.float32)
    cols = np.arange(face_count) % per_row
    rows = np.arange(face_count) // per_row
    origin = np.stack([cols * patch + 1, rows * patch + 1], 1)  # (F, 2) pixel origin of the inner square
    uvs[:] = ((origin[:, None, :] + local[None] * inner) / size).astype(np.float32)
    order = np.argsort(views)
    done = 0
    for view in np.unique(views):
        members = order[views[order] == view]
        if view < 0:
            done += len(members)
            continue
        image = images[view]
        camera = cameras.get(image["camera_id"]) or cameras.get(str(image["camera_id"]))
        frame = np.asarray(load_image(image["name"]))
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, 2)
        frame = frame[:, :, :3]
        fh, fw = frame.shape[:2]
        sx, sy = fw / camera["width"], fh / camera["height"]  # frames may be scaled copies of the calibrated size
        corners = positions[faces[members]]                                  # (M, 3, 3)
        points = np.einsum("pk,mkd->mpd", bary, corners).reshape(-1, 3)     # (M·P², 3)
        uv, ok = project(world_to_camera(points, image["qvec"], image["tvec"]), camera)
        px = np.clip((uv[:, 0] * sx).astype(np.int64), 0, fw - 1)
        py = np.clip((uv[:, 1] * sy).astype(np.int64), 0, fh - 1)
        colors = frame[py, px]
        mask = (ok & np.tile(inside, len(members))).reshape(len(members), -1)
        colors = colors.reshape(len(members), -1, 3)
        base = origin[members] - 1                                             # patch origin incl. border
        ax = (base[:, None, 0] + pixel_offsets[None, :, 0]).reshape(-1)
        ay = (base[:, None, 1] + pixel_offsets[None, :, 1]).reshape(-1)
        flat = mask.reshape(-1)
        atlas[ay[flat], ax[flat]] = colors.reshape(-1, 3)[flat]
        done += len(members)
        on_progress(done, face_count)
    return {"atlas": atlas, "uvs": uvs, "views": views, "size": size}


def encode_png(image: Any) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * (-len(data) % 4)


def write_textured_glb(path: Path, positions: Any, faces: Any, uvs: Any, png: bytes, name: str = "mesh") -> None:
    """glTF binary with unwelded triangles, TEXCOORD_0 and one embedded PNG base-color texture."""
    import numpy as np

    positions = np.asarray(positions, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)
    corner_positions = np.ascontiguousarray(positions[faces].reshape(-1, 3).astype("<f4"))
    corner_uvs = np.ascontiguousarray(uvs.reshape(-1, 2).astype("<f4"))
    indices = np.arange(len(corner_positions), dtype="<u4")
    blob = b""
    views, accessors = [], []

    def add(data: bytes, target: Optional[int]) -> int:
        nonlocal blob
        offset = len(blob)
        blob += _pad4(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    accessors.append({"bufferView": add(corner_positions.tobytes(), 34962), "componentType": 5126,
                      "count": int(len(corner_positions)), "type": "VEC3",
                      "min": [float(v) for v in corner_positions.min(0)], "max": [float(v) for v in corner_positions.max(0)]})
    accessors.append({"bufferView": add(corner_uvs.tobytes(), 34962), "componentType": 5126,
                      "count": int(len(corner_uvs)), "type": "VEC2"})
    accessors.append({"bufferView": add(indices.tobytes(), 34963), "componentType": 5125,
                      "count": int(len(indices)), "type": "SCALAR"})
    image_view = add(png, None)
    gltf = {
        "asset": {"version": "2.0", "generator": "LichtFeld Studio video-to-splat"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2,
                                                   "material": 0, "mode": 4}]}],
        "materials": [{"name": name, "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                                                "roughnessFactor": 1.0}, "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": image_view, "mimeType": "image/png", "name": "atlas"}],
        "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
    }
    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad4(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk)
        handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk)


def _usdz_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    """A stored member whose data starts on a 64-byte boundary, as USDZ requires."""
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    offset = archive.fp.tell() if hasattr(archive, "fp") and archive.fp else 0
    header = offset + 30 + len(name.encode("utf-8"))
    pad = (-header) % 64
    if 0 < pad < 4:
        pad += 64
    if pad:
        info.extra = struct.pack("<HH", 0x1986, pad - 4) + b"\x00" * (pad - 4)
    archive.writestr(info, data)


def write_textured_usdz(path: Path, positions: Any, faces: Any, uvs: Any, png: bytes, name: str = "mesh") -> None:
    """USDZ with a UsdPreviewSurface reading atlas.png; st uses USD's bottom-left origin."""
    import numpy as np

    positions = np.asarray(positions, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    corners = positions[faces].reshape(-1, 3)
    count = len(faces)
    lines = ['#usda 1.0', '(', '    defaultPrim = "%s"' % name, '    metersPerUnit = 1', '    upAxis = "Y"', ')', '',
             'def Xform "%s"' % name, '{',
             '    def Mesh "geometry" (', '        prepend apiSchemas = ["MaterialBindingAPI"]', '    )', '    {',
             '        int[] faceVertexCounts = [%s]' % ", ".join(["3"] * count),
             '        int[] faceVertexIndices = [%s]' % ", ".join(str(i) for i in range(3 * count)),
             '        point3f[] points = [%s]' % ", ".join("(%g, %g, %g)" % tuple(p) for p in corners),
             '        texCoord2f[] primvars:st = [%s] (' % ", ".join("(%.5f, %.5f)" % (u, 1.0 - v) for u, v in uvs),
             '            interpolation = "vertex"', '        )',
             '        uniform token subdivisionScheme = "none"',
             '        rel material:binding = </%s/material>' % name,
             '    }', '',
             '    def Material "material"', '    {',
             '        token outputs:surface.connect = </%s/material/surface.outputs:surface>' % name,
             '        def Shader "surface"', '        {',
             '            uniform token info:id = "UsdPreviewSurface"',
             '            color3f inputs:diffuseColor.connect = </%s/material/texture.outputs:rgb>' % name,
             '            float inputs:metallic = 0', '            float inputs:roughness = 1',
             '            token outputs:surface', '        }',
             '        def Shader "reader"', '        {',
             '            uniform token info:id = "UsdPrimvarReader_float2"',
             '            token inputs:varname = "st"', '            float2 outputs:result', '        }',
             '        def Shader "texture"', '        {',
             '            uniform token info:id = "UsdUVTexture"',
             '            asset inputs:file = @atlas.png@',
             '            float2 inputs:st.connect = </%s/material/reader.outputs:result>' % name,
             '            token inputs:wrapS = "clamp"', '            token inputs:wrapT = "clamp"',
             '            float3 outputs:rgb', '        }',
             '    }', '}', '']
    usda = "\n".join(lines).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        _usdz_member(archive, name + ".usda", usda)
        _usdz_member(archive, "atlas.png", png)


def texture_mesh(package: Path, positions: Any, faces: Any, images: list[dict], cameras: dict,
                 load_image: Callable[[str], Any], name: str = "mesh",
                 on_progress: Callable[[int, int], None] = lambda c, t: None) -> dict:
    """Bake the atlas and write atlas.png, mesh_textured.glb and mesh_textured.usdz into ``package``."""
    baked = bake_atlas(positions, faces, images, cameras, load_image, on_progress=on_progress)
    png = encode_png(baked["atlas"])
    (package / "atlas.png").write_bytes(png)
    write_textured_glb(package / "mesh_textured.glb", positions, faces, baked["uvs"], png, name=name)
    write_textured_usdz(package / "mesh_textured.usdz", positions, faces, baked["uvs"], png, name=name)
    textured = int((baked["views"] >= 0).sum())
    return {"faces": int(len(faces)), "textured_faces": textured, "atlas_size": baked["size"]}
