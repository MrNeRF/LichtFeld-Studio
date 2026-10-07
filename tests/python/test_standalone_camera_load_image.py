# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone Camera image loading for datasets returned by io.load."""

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
    assert image.device == lf.Tensor.empty((0,), device="gpu").device
    assert image.ndim == 3
    assert image.shape[0] in (3, 4)
    assert tuple(image.shape[1:]) == (1, 1)
