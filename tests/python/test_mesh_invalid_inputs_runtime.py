# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Process-isolated safety tests for invalid Python mesh inputs."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUILD_DIR = os.path.abspath(os.environ.get("LFS_TEST_BUILD_DIR", os.path.join(ROOT, "build")))
numpy = pytest.importorskip("numpy")
# The child runs with an isolated HOME, so hand it the NumPy this interpreter uses.
NUMPY_SITE = os.path.dirname(os.path.dirname(numpy.__file__))


def _run_mesh_script(tmp_path, code):
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    os.makedirs(env["HOME"])
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(BUILD_DIR, "src", "python"),
            os.path.join(ROOT, "src", "python"),
            NUMPY_SITE,
        ]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [BUILD_DIR, env.get("LD_LIBRARY_PATH", "")])
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, (
        f"standalone interpreter exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "REJECTED" in result.stdout, result.stdout


@pytest.mark.parametrize(
    "mesh_type, handle_type, operation",
    [
        (mesh_type, handle_type, "mesh.set_deleted(handle, True)")
        for mesh_type in ("TriMesh", "PolyMesh")
        for handle_type in ("VertexHandle", "HalfedgeHandle", "EdgeHandle", "FaceHandle")
    ]
    + [
        (mesh_type, "VertexHandle", operation)
        for mesh_type in ("TriMesh", "PolyMesh")
        for operation in (
            "mesh.set_locked(handle, True)",
            "mesh.point(handle)",
            "mesh.set_point(handle, np.zeros(3, dtype=np.float64))",
            "mesh.delete_vertex(handle)",
            "mesh.valence(handle)",
            "mesh.is_boundary(handle)",
        )
    ]
    + [
        (mesh_type, "EdgeHandle", operation)
        for mesh_type in ("TriMesh", "PolyMesh")
        for operation in ("mesh.feature(handle)", "mesh.set_feature(handle, True)")
    ]
    + [
        ("TriMesh", "VertexHandle", operation)
        for operation in (
            "mesh.normal(handle)",
            "mesh.set_normal(handle, np.zeros(3, dtype=np.float64))",
            "mesh.update_normal(handle)",
        )
    ],
)
def test_invalid_openmesh_handles_raise_instead_of_crashing(tmp_path, mesh_type, handle_type, operation):
    code = f"""
import numpy as np
import lichtfeld as lf
mesh = lf.mesh.{mesh_type}()
handle = lf.mesh.{handle_type}(2147483647)
try:
    {operation}
except (ValueError, IndexError):
    print("REJECTED")
else:
    raise AssertionError("invalid handle was accepted")
"""
    _run_mesh_script(tmp_path, code)


@pytest.mark.parametrize("mesh_type", ["TriMesh", "PolyMesh"])
@pytest.mark.parametrize("face_indices", ["[[0, 0, 1]]", "[[0, 1, 3]]", "[[-1, 1, 2]]"])
def test_add_faces_rejects_repeated_or_out_of_range_indices(tmp_path, mesh_type, face_indices):
    code = f"""
import numpy as np
import lichtfeld as lf
points = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float64)
mesh = lf.mesh.{mesh_type}(points)
faces = np.array({face_indices}, dtype=np.int32)
try:
    mesh.add_faces(faces)
except (ValueError, IndexError):
    assert mesh.n_faces() == 0
    print("REJECTED")
else:
    raise AssertionError("invalid face row was accepted")
"""
    _run_mesh_script(tmp_path, code)


@pytest.mark.parametrize("consumer", ["mesh.compute_normals()", "lf.mesh.from_mesh_data(mesh)"])
def test_mesh_data_consumers_reject_mutated_invalid_indices(tmp_path, consumer):
    code = f"""
import numpy as np
import lichtfeld as lf
vertices = lf.Tensor.from_numpy(np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32))
indices = lf.Tensor.from_numpy(np.array([[0,1,2]], dtype=np.int32))
mesh = lf.mesh.MeshData(vertices, indices)
mesh.indices[0,2] = 3
try:
    {consumer}
except (ValueError, IndexError):
    print("REJECTED")
else:
    raise AssertionError("invalid MeshData face index was accepted")
"""
    _run_mesh_script(tmp_path, code)
