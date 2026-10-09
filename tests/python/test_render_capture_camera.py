# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""render_capture with camera_index renders that dataset camera's view."""

import base64
import io
import os
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def dataset_project():
    endpoint = os.environ.get("LFS_CAMERA_CAPTURE_ENDPOINT")
    project = os.environ.get("LFS_CAMERA_CAPTURE_PROJECT")
    if not endpoint or not project:
        pytest.skip("requires an isolated app and a trained dataset project")
    _initialize(endpoint)
    _tool(endpoint, "project_open", {"path": project, "discard_changes": True})
    deadline = time.monotonic() + 120
    while _tool(endpoint, "project_get_info").get("hydration_state") != "complete":
        assert time.monotonic() < deadline, "project did not finish hydrating"
        time.sleep(0.5)
    return endpoint


def _capture(endpoint, arguments):
    from PIL import Image

    result = _call(endpoint, "tools/call", {"name": "render_capture", "arguments": arguments})
    images = [item for item in result["content"] if item["type"] == "image"]
    if not images:
        return result, None
    return result, Image.open(io.BytesIO(base64.b64decode(images[0]["data"]))).convert("RGB")


def test_camera_index_renders_each_dataset_camera(dataset_project):
    from PIL import ImageChops

    endpoint = dataset_project
    cameras = _tool(endpoint, "camera_list")["cameras"]
    first, last = cameras[0], cameras[-1]

    _, first_image = _capture(endpoint, {"camera_index": first["uid"]})
    _, last_image = _capture(endpoint, {"camera_index": last["uid"]})
    assert first_image is not None and last_image is not None
    assert first_image.size == (first["image_width"], first["image_height"])
    assert ImageChops.difference(first_image, last_image).getbbox() is not None

    _, scaled = _capture(endpoint, {"camera_index": first["uid"], "width": first["image_width"] // 2})
    assert scaled.size[0] == first["image_width"] // 2


def test_unknown_camera_index_is_an_error(dataset_project):
    result, image = _capture(dataset_project, {"camera_index": 10**9})
    assert image is None
    assert "Camera UID not found" in str(result)
