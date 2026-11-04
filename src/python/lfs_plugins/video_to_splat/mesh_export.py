# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh and point-cloud files for the download package, written without any mesh library.

- ``gaussians_to_point_cloud`` turns trained Gaussians into a colored, oriented point cloud
  (the normal of a Gaussian is its thinnest axis), the input COLMAP's Poisson mesher wants.
- ``read_ply_mesh`` reads the binary PLY the mesher writes (vertex colors, triangle faces).
- ``write_glb`` and ``write_usdz`` write a vertex-colored triangle mesh as glTF binary and as a
  USDZ package (an uncompressed zip holding a .usda), the two formats game engines and iPhone AR
  read. numpy is imported lazily so planning code and tests import this module without it.
"""
from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from typing import Any

PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
             "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
             "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def gaussians_to_point_cloud(means: Any, scales: Any, rotations: Any, colors: Any, opacity: Any,
                             min_opacity: float = 0.3) -> dict:
    """Positions, unit normals and uint8 colors of the Gaussians above ``min_opacity``.

    ``rotations`` are (w, x, y, z) quaternions and ``scales`` linear per-axis extents; the normal is
    the rotated axis with the smallest scale, which is the face direction of a flattened Gaussian."""
    import numpy as np

    means, scales, rotations, colors = (np.asarray(a, dtype=np.float32) for a in (means, scales, rotations, colors))
    opacity = np.asarray(opacity, dtype=np.float32).reshape(-1)
    keep = opacity >= min_opacity
    means, scales, rotations, colors = means[keep], scales[keep], rotations[keep], colors[keep]
    if len(means) == 0:
        return {"positions": means, "normals": means.copy(), "colors": np.zeros((0, 3), dtype=np.uint8)}
    q = rotations / np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    columns = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1),
        np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1),
        np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)], 1)  # (N, 3 axes, xyz)
    axis = np.argmin(scales, axis=1)
    normals = columns[np.arange(len(columns)), axis]
    rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    return {"positions": means, "normals": normals.astype(np.float32), "colors": rgb}


def write_point_cloud_ply(path: Path, cloud: dict) -> int:
    """Binary PLY with x y z nx ny nz red green blue; returns the point count."""
    import numpy as np

    positions, normals, colors = cloud["positions"], cloud["normals"], cloud["colors"]
    count = len(positions)
    record = np.zeros(count, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                                    ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        record[name] = positions[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        record[name] = normals[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        record[name] = colors[:, index]
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % count
              + "".join("property float %s\n" % n for n in ("x", "y", "z", "nx", "ny", "nz"))
              + "".join("property uchar %s\n" % n for n in ("red", "green", "blue")) + "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(record.tobytes())
    return count


def read_ply_mesh(path: Path) -> dict:
    """Vertices (positions, optional colors) and triangle faces of a binary little-endian PLY."""
    import numpy as np

    with open(path, "rb") as handle:
        elements: list[tuple[str, int, list]] = []
        if handle.readline().strip() != b"ply":
            raise ValueError("not a PLY file: %s" % path)
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("truncated PLY header: %s" % path)
            parts = line.decode("ascii", "replace").split()
            if not parts:
                continue
            if parts[0] == "format" and parts[1] != "binary_little_endian":
                raise ValueError("only binary little-endian PLY is supported: %s" % path)
            if parts[0] == "element":
                elements.append((parts[1], int(parts[2]), []))
            elif parts[0] == "property":
                if parts[1] == "list":
                    elements[-1][2].append((parts[4], "list", PLY_TYPES[parts[2]], PLY_TYPES[parts[3]]))
                else:
                    elements[-1][2].append((parts[2], PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                break
        result: dict = {"positions": None, "colors": None, "faces": None}
        for name, count, props in elements:
            if any(p[1] == "list" for p in props):
                if len(props) != 1:
                    raise ValueError("unsupported mixed list element in %s" % path)
                _pname, _kind, count_type, item_type = props[0]
                count_size = np.dtype("<" + count_type).itemsize
                item_size = np.dtype("<" + item_type).itemsize
                faces = []
                for _ in range(count):
                    (n,) = np.frombuffer(handle.read(count_size), dtype="<" + count_type)
                    items = np.frombuffer(handle.read(item_size * int(n)), dtype="<" + item_type)
                    for k in range(1, int(n) - 1):  # fan-triangulate polygons
                        faces.append((int(items[0]), int(items[k]), int(items[k + 1])))
                if name == "face":
                    result["faces"] = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
                continue
            dtype = np.dtype([(p[0], "<" + p[1]) for p in props])
            data = np.frombuffer(handle.read(dtype.itemsize * count), dtype=dtype)
            if name == "vertex":
                result["positions"] = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float32)
                if all(c in data.dtype.names for c in ("red", "green", "blue")):
                    result["colors"] = np.stack([data["red"], data["green"], data["blue"]], -1).astype(np.uint8)
    if result["positions"] is None:
        raise ValueError("PLY has no vertex element: %s" % path)
    if result["faces"] is None:
        result["faces"] = np.zeros((0, 3), dtype=np.uint32)
    return result


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * (-len(data) % 4)


def write_glb(path: Path, positions: Any, faces: Any, colors: Any = None, name: str = "mesh") -> None:
    """glTF 2.0 binary: one triangle mesh with POSITION and optional COLOR_0 (normalized uint8)."""
    import numpy as np

    positions = np.ascontiguousarray(np.asarray(positions, dtype="<f4"))
    faces = np.ascontiguousarray(np.asarray(faces, dtype="<u4")).reshape(-1)
    blob = b""
    buffer_views = []
    accessors = []

    def add(data: bytes, target: int) -> int:
        nonlocal blob
        offset = len(blob)
        blob += _pad4(data)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        return len(buffer_views) - 1

    view = add(positions.tobytes(), 34962)
    accessors.append({"bufferView": view, "componentType": 5126, "count": int(len(positions)), "type": "VEC3",
                      "min": [float(v) for v in positions.min(0)] if len(positions) else [0, 0, 0],
                      "max": [float(v) for v in positions.max(0)] if len(positions) else [0, 0, 0]})
    attributes = {"POSITION": 0}
    if colors is not None and len(colors):
        rgba = np.concatenate([np.asarray(colors, dtype=np.uint8), np.full((len(colors), 1), 255, np.uint8)], 1)
        view = add(np.ascontiguousarray(rgba).tobytes(), 34962)
        accessors.append({"bufferView": view, "componentType": 5121, "normalized": True,
                          "count": int(len(rgba)), "type": "VEC4"})
        attributes["COLOR_0"] = len(accessors) - 1
    view = add(faces.tobytes(), 34963)
    accessors.append({"bufferView": view, "componentType": 5125, "count": int(len(faces)), "type": "SCALAR"})
    gltf = {
        "asset": {"version": "2.0", "generator": "LichtFeld Studio video-to-splat"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attributes, "indices": len(accessors) - 1,
                                                   "mode": 4}]}],
        "buffers": [{"byteLength": len(blob)}], "bufferViews": buffer_views, "accessors": accessors,
    }
    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad4(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk)
        handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk)


def read_glb(path: Path) -> dict:
    """The JSON chunk and the binary chunk of a GLB, for checks and round trips."""
    with open(path, "rb") as handle:
        magic, version, _total = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError("not a glTF 2.0 binary: %s" % path)
        length, kind = struct.unpack("<I4s", handle.read(8))
        gltf = json.loads(handle.read(length).decode("utf-8"))
        length, kind = struct.unpack("<I4s", handle.read(8))
        blob = handle.read(length) if kind == b"BIN\x00" else b""
    return {"json": gltf, "bin": blob}


def write_usdz(path: Path, positions: Any, faces: Any, colors: Any = None, name: str = "mesh") -> None:
    """USDZ: a zip (stored, 64-byte aligned members) holding one .usda with a vertex-colored mesh."""
    import numpy as np

    positions = np.asarray(positions, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    lines = ['#usda 1.0', '(', '    defaultPrim = "%s"' % name, '    metersPerUnit = 1', '    upAxis = "Y"', ')', '',
             'def Xform "%s"' % name, '{', '    def Mesh "geometry"', '    {',
             '        int[] faceVertexCounts = [%s]' % ", ".join(["3"] * len(faces)),
             '        int[] faceVertexIndices = [%s]' % ", ".join(str(int(i)) for i in faces.reshape(-1)),
             '        point3f[] points = [%s]' % ", ".join("(%g, %g, %g)" % tuple(p) for p in positions),
             '        uniform token subdivisionScheme = "none"']
    if colors is not None and len(colors):
        rgb = np.asarray(colors, dtype=np.float32) / 255.0
        lines.append('        color3f[] primvars:displayColor = [%s] (' % ", ".join("(%.4f, %.4f, %.4f)" % tuple(c) for c in rgb))
        lines.append('            interpolation = "vertex"')
        lines.append('        )')
    lines += ['    }', '}', '']
    usda = "\n".join(lines).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # USDZ requires uncompressed members whose data starts at a 64-byte boundary.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo(name + ".usda")
        header = 30 + len(info.filename)
        pad = (-header) % 64
        if pad:
            info.extra = struct.pack("<HH", 0x1986, pad - 4) + b"\x00" * (pad - 4) if pad >= 4 else b""
        archive.writestr(info, usda)


def read_usdz(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith((".usda", ".usdc", ".usd"))]
        return archive.read(names[0]).decode("utf-8")
