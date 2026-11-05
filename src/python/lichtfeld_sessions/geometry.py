# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera calibration and explicit dataset/model coordinate metadata."""
import json
import math
import uuid

from _lfs_path_io import Path

SCHEMA = 1
MAX_CAMERAS = 10000
MAX_BYTES = 32 * 1024 * 1024
FLIP = ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1))
IDENTITY = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))


def matrix(value):
    import numpy as np

    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or np.max(np.abs(result)) > 1e10
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7)
            or abs(np.linalg.det(result[:3, :3])) < 1e-12):
        raise ValueError("Invalid session coordinate transform")
    return result


def validate(value):
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != SCHEMA:
        raise ValueError("Unsupported viewer geometry")
    for key in ("source_frame", "data_frame"):
        if key in value and (not isinstance(value[key], str) or not 0 < len(value[key]) <= 128):
            raise ValueError("Invalid session coordinate frame")
    matrix(value.get("data_to_source", FLIP))
    matrix(value.get("model_to_data", IDENTITY))
    cameras = value.get("cameras", [])
    if not isinstance(cameras, list) or len(cameras) > MAX_CAMERAS:
        raise ValueError("Invalid session camera count")
    for item in cameras:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or len(item["name"]) > 1024:
            raise ValueError("Invalid session camera name")
        matrix(item.get("camera_to_data"))
        for key in ("width", "height"):
            if type(item.get(key)) is not int or not 0 < item[key] <= 65536:
                raise ValueError("Invalid session camera size")
        for key in ("fx", "fy", "cx", "cy"):
            if (type(item.get(key)) not in (int, float)
                    or not math.isfinite(item[key]) or abs(item[key]) > 1e9):
                raise ValueError("Invalid session camera intrinsics")
        if item["fx"] <= 0 or item["fy"] <= 0:
            raise ValueError("Invalid session camera focal length")
    return value


def from_native_scene(scene, source=None):
    """Capture actual loaded calibration and poses, including resized images."""
    import numpy as np

    model = scene.get_node(scene.training_model_node_name)
    value = {"schema": SCHEMA, "source_frame": "lichtfeld-visualizer",
             "data_to_source": FLIP,
             "model_to_data": np.asarray(model.world_transform).reshape(4, 4).tolist() if model else IDENTITY,
             "cameras": []}
    if source:
        for key in ("source_frame", "data_to_source"):
            if key in source:
                value[key] = source[key]
    for node in scene.get_active_cameras():
        if not node.has_camera:
            continue
        rotation = node.camera_R.cpu().numpy().reshape(3, 3)
        translation = node.camera_T.cpu().numpy().reshape(3)
        pose = np.eye(4)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ translation
        pose = np.asarray(node.world_transform).reshape(4, 4) @ pose
        value["cameras"].append({"name": node.name, "camera_to_data": pose.tolist(),
                                 "fx": node.camera_focal_x, "fy": node.camera_focal_y,
                                 "cx": node.camera_center_x, "cy": node.camera_center_y,
                                 "width": node.camera_width, "height": node.camera_height})
    return validate(value)


def publish(publisher, value):
    from .transport import atomic_json

    value = validate(value)
    size = len(json.dumps(value, allow_nan=False).encode("utf-8"))
    if size > MAX_BYTES:
        raise ValueError("Session camera metadata exceeds the size limit")
    path = publisher.root / "view_geometry.json"
    atomic_json(path, value)
    publisher.update(geometry={"path": path.name, "bytes": size, "revision": uuid.uuid4().hex})


def load(root, descriptor):
    from .transport import read_text_shared

    if not descriptor:
        return None
    if not isinstance(descriptor, dict) or descriptor.get("path") != "view_geometry.json":
        raise ValueError("Invalid session geometry reference")
    size = descriptor.get("bytes")
    if type(size) is not int or not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid session geometry size")
    root = Path(root).resolve()
    path = root / descriptor["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Invalid session geometry path")
    data = read_text_shared(path, max_chars=MAX_BYTES)
    if len(data.encode("utf-8")) != size:
        raise ValueError("Session geometry size changed")
    return validate(json.loads(data))
