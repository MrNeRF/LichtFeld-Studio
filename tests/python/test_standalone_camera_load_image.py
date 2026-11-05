# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone Camera image loading for datasets returned by io.load."""

import gc
import json

import pytest


_ONE_PIXEL_PNG = bytes(
    [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0B,
        0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0x0F, 0x04, 0x00, 0x09, 0xFB,
        0x03, 0xFD, 0xFB, 0x5E, 0x6B, 0x2B, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ]
)


@pytest.mark.gpu
def test_io_loaded_camera_loads_image_without_application_context(lf, tmp_path):
    dataset_path = tmp_path / "standalone_image"
    dataset_path.mkdir()
    (dataset_path / "present.png").write_bytes(_ONE_PIXEL_PNG)
    (dataset_path / "transforms.json").write_text(
        """{
  "w": 1,
  "h": 1,
  "fl_x": 1.0,
  "fl_y": 1.0,
  "cx": 0.5,
  "cy": 0.5,
  "frames": [
    {
      "file_path": "present.png",
      "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ]
}
"""
    )

    dataset = lf.io.load(str(dataset_path))
    camera = dataset.cameras[0]
    assert camera.has_image
    assert tuple(camera.K.shape) == (3, 3)

    image = camera.load_image(output_uint8=True)
    assert image.is_cuda
    assert image.ndim == 3
    assert image.shape[0] in (3, 4)
    assert tuple(image.shape[1:]) == (1, 1)


def _write_dataset(root, names, with_masks=False):
    root.mkdir()
    frames = []
    for name in names:
        (root / name).write_bytes(_ONE_PIXEL_PNG)
        frame = {"file_path": name, "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        if with_masks:
            mask = f"mask_{name}"
            (root / mask).write_bytes(_ONE_PIXEL_PNG)
            frame["mask_path"] = mask
        frames.append(frame)
    (root / "transforms.json").write_text(
        json.dumps({"w": 1, "h": 1, "fl_x": 1.0, "fl_y": 1.0, "cx": 0.5, "cy": 0.5, "frames": frames})
    )
    return root


@pytest.mark.gpu
def test_io_loaded_camera_loads_mask_without_application_context(lf, tmp_path):
    dataset_path = _write_dataset(tmp_path / "standalone_mask", ["present.png"], with_masks=True)

    result = lf.io.load(str(dataset_path))
    camera = result.cameras[0]
    assert camera.has_mask

    mask = camera.load_mask()
    assert mask.is_cuda
    assert mask.numel == 1

    unmasked = _write_dataset(tmp_path / "standalone_unmasked", ["present.png"])
    plain = lf.io.load(str(unmasked))
    assert plain.cameras[0].load_mask() is None
    assert plain.cameras[0].load_depth() is None


@pytest.mark.gpu
def test_cameras_stay_valid_after_their_dataset_is_released(lf, tmp_path):
    names = [f"view_{index}.png" for index in range(3)]
    dataset_path = _write_dataset(tmp_path / "camera_lifetime", names)

    result = lf.io.load(str(dataset_path))
    cameras = result.cameras
    by_filename = cameras.get_camera_by_filename("view_1.png")
    by_index = cameras[2]
    from_list = cameras.cameras()[0]
    del result, cameras
    gc.collect()

    # Reuse the released memory before touching the cameras again.
    for _ in range(8):
        lf.io.load(str(dataset_path))
    gc.collect()

    assert by_filename.image_name == "view_1.png"
    assert by_index.image_name == "view_2.png"
    assert from_list.image_name == "view_0.png"
    for camera in (by_filename, by_index, from_list):
        assert camera.focal_x == pytest.approx(1.0)
        assert tuple(camera.load_image().shape[1:]) == (1, 1)
