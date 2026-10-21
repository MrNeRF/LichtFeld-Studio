# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image sequence binding preflight must not create output or require a viewer."""

import pytest


@pytest.mark.parametrize("changes", [
    {"format": "unknown"}, {"path": ""}, {"path": "bad\0path"},
    {"width": 0}, {"height": -1}, {"framerate": 0}, {"framerate": 1001},
    {"first_frame_number": -1}, {"filename_pattern": "../frame_%d"},
    {"filename_pattern": "folder\\frame_%d"}, {"transparent": True},
])
def test_invalid_image_sequence_rejected_before_viewer(lf, tmp_path, changes):
    request = {"width": 3, "height": 5, "framerate": 24, "path": str(tmp_path / "frames")}
    request.update(changes)
    with pytest.raises(ValueError):
        lf.ui.export_image_sequence(**request)
    assert not (tmp_path / "frames").exists()


@pytest.mark.parametrize("format", ["png", "jpeg", "exr", "tiff16", "jxl", "avif"])
def test_image_sequence_headless_boundary(lf, tmp_path, format):
    try:
        lf.ui.get_video_reconstruction_selection()
    except RuntimeError as error:
        assert str(error) == "Viewer is unavailable"
    else:
        pytest.skip("Requires the headless binding boundary")
    path = tmp_path / "frames"
    with pytest.raises(RuntimeError, match="Viewer is unavailable"):
        lf.ui.export_image_sequence(3, 5, 24, str(path), format=format)
    assert not path.exists()
