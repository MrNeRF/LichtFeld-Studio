# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""MeshData device conversion accepts known device names only."""

import pytest


def _make_mesh_data(lf):
    vertices = lf.Tensor.ones((3, 3), dtype="float32", device="cpu")
    indices = lf.Tensor.zeros((1, 3), dtype="int32", device="cpu")
    return lf.mesh.MeshData(vertices, indices)


def test_to_rejects_unknown_device(lf):
    mesh = _make_mesh_data(lf)

    with pytest.raises(ValueError, match="Unknown device: not-a-device"):
        mesh.to("not-a-device")


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu), pytest.param("gpu", marks=pytest.mark.gpu)],
)
def test_to_preserves_supported_devices(lf, gpu_available, device):
    if device != "cpu" and not gpu_available:
        pytest.skip("GPU not available")

    result = _make_mesh_data(lf).to(device)
    expected = "cuda" if device != "cpu" else "cpu"
    assert result.vertices.device == expected
    assert result.indices.device == expected
