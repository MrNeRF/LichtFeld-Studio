# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""MeshData validates tensor contracts at construction and setter calls."""

import numpy as np
import pytest


def _mesh_data(lf):
    vertices = lf.Tensor.zeros((3, 3), dtype="float32", device="cpu")
    indices = lf.Tensor.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    return lf.mesh.MeshData(vertices, indices)


def test_constructor_rejects_invalid_vertex_shape(lf):
    vertices = lf.Tensor.zeros((3, 2), dtype="float32", device="cpu")
    indices = lf.Tensor.zeros((1, 3), dtype="int32", device="cpu")

    with pytest.raises(ValueError, match=r"vertices must be a float32 tensor with shape \[V, 3\]"):
        lf.mesh.MeshData(vertices, indices)


@pytest.mark.parametrize(
    "setter,shape,dtype,message",
    [
        ("set_vertices", (3, 2), "float32", "vertices"),
        ("set_vertices", (3, 3), "float16", "vertices"),
        ("set_indices", (1, 2), "int32", "indices"),
        ("set_indices", (1, 3), "float32", "indices"),
        ("set_normals", (3, 2), "float32", "normals"),
        ("set_normals", (2, 3), "float32", "normals"),
        ("set_texcoords", (3, 3), "float32", "texcoords"),
        ("set_colors", (3, 3), "float32", "colors"),
        ("set_colors", (3, 4), "float16", "colors"),
    ],
)
def test_setters_reject_invalid_tensor_contracts(lf, setter, shape, dtype, message):
    mesh = _mesh_data(lf)
    values = lf.Tensor.zeros(shape, dtype=dtype, device="cpu")

    with pytest.raises(ValueError, match=f"{message} must be"):
        getattr(mesh, setter)(values)


def test_setters_accept_valid_tensors_and_optional_attributes_can_be_cleared(lf):
    mesh = _mesh_data(lf)
    mesh.set_vertices(lf.Tensor.ones((3, 3), dtype="float32", device="cpu"))
    mesh.set_indices(lf.Tensor.zeros((1, 3), dtype="int32", device="cpu"))
    mesh.set_normals(lf.Tensor.ones((3, 3), dtype="float32", device="cpu"))
    mesh.set_texcoords(lf.Tensor.ones((3, 2), dtype="float32", device="cpu"))
    mesh.set_colors(lf.Tensor.ones((3, 4), dtype="float32", device="cpu"))

    assert mesh.has_normals
    assert mesh.has_texcoords
    assert mesh.has_colors

    mesh.set_normals(lf.Tensor())
    mesh.set_texcoords(lf.Tensor())
    mesh.set_colors(lf.Tensor())
    assert not mesh.has_normals
    assert not mesh.has_texcoords
    assert not mesh.has_colors
