# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for OpenMesh bindings and MeshData bridge."""

import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


def _make_triangle(lf):
    """Helper: create a single-triangle TriMesh."""
    mesh = lf.mesh.TriMesh()
    v0 = mesh.add_vertex(np.array([0, 0, 0], dtype=np.float64))
    v1 = mesh.add_vertex(np.array([1, 0, 0], dtype=np.float64))
    v2 = mesh.add_vertex(np.array([0, 1, 0], dtype=np.float64))
    mesh.add_face(v0, v1, v2)
    return mesh


def _make_quad(lf):
    """Helper: create a two-triangle quad TriMesh."""
    mesh = lf.mesh.TriMesh()
    v0 = mesh.add_vertex(np.array([0, 0, 0], dtype=np.float64))
    v1 = mesh.add_vertex(np.array([1, 0, 0], dtype=np.float64))
    v2 = mesh.add_vertex(np.array([1, 1, 0], dtype=np.float64))
    v3 = mesh.add_vertex(np.array([0, 1, 0], dtype=np.float64))
    mesh.add_face(v0, v1, v2)
    mesh.add_face(v0, v2, v3)
    return mesh


class TestTriMeshCreation:

    def test_create_empty_mesh(self, lf):
        mesh = lf.mesh.TriMesh()
        assert mesh.n_vertices() == 0
        assert mesh.n_faces() == 0

    def test_add_vertex(self, lf):
        mesh = lf.mesh.TriMesh()
        vh = mesh.add_vertex(np.array([0, 0, 0], dtype=np.float64))
        assert mesh.n_vertices() == 1
        assert vh.idx() == 0

    def test_add_face(self, lf):
        mesh = _make_triangle(lf)
        assert mesh.n_faces() == 1
        assert mesh.n_vertices() == 3

    def test_points_numpy_view(self, lf):
        mesh = lf.mesh.TriMesh()
        mesh.add_vertex(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        mesh.add_vertex(np.array([4.0, 5.0, 6.0], dtype=np.float64))

        pts = mesh.points()
        assert pts.shape == (2, 3)
        assert abs(pts[0, 0] - 1.0) < 1e-6
        assert abs(pts[1, 2] - 6.0) < 1e-6

    def test_points_writable(self, lf):
        mesh = lf.mesh.TriMesh()
        mesh.add_vertex(np.array([0, 0, 0], dtype=np.float64))
        mesh.add_vertex(np.array([1, 0, 0], dtype=np.float64))
        pts = mesh.points()
        pts[0, 0] = 42.0
        pts2 = mesh.points()
        assert abs(pts2[0, 0] - 42.0) < 1e-6


class TestIterators:

    def test_vertex_iterator(self, lf):
        mesh = _make_triangle(lf)
        verts = list(mesh.vertices())
        assert len(verts) == 3

    def test_face_iterator(self, lf):
        mesh = _make_triangle(lf)
        faces = list(mesh.faces())
        assert len(faces) == 1

    def test_edge_iterator(self, lf):
        mesh = _make_triangle(lf)
        edges = list(mesh.edges())
        assert len(edges) == 3

    def test_halfedge_iterator(self, lf):
        mesh = _make_triangle(lf)
        halfedges = list(mesh.halfedges())
        assert len(halfedges) == 6


class TestCirculators:

    def test_vv_circulator(self, lf):
        mesh = _make_quad(lf)
        vh0 = next(iter(mesh.vertices()))
        neighbors = list(mesh.vv(vh0))
        assert len(neighbors) >= 2

    def test_fv_circulator(self, lf):
        mesh = _make_triangle(lf)
        fh = next(iter(mesh.faces()))
        face_verts = list(mesh.fv(fh))
        assert len(face_verts) == 3

    def test_vf_circulator(self, lf):
        mesh = _make_quad(lf)
        vh0 = next(iter(mesh.vertices()))
        adj_faces = list(mesh.vf(vh0))
        assert len(adj_faces) == 2


class TestMeshIO:

    def test_write_and_read_obj(self, lf):
        mesh = _make_triangle(lf)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.obj")
            lf.mesh.write_mesh(path, mesh)

            mesh2 = lf.mesh.read_trimesh(path)
            assert mesh2.n_vertices() == 3
            assert mesh2.n_faces() == 1

    def test_read_test_cube(self, lf, test_data_dir):
        cube_path = test_data_dir / "test_cube.obj"
        if not cube_path.exists():
            pytest.skip("test_cube.obj not found")

        mesh = lf.mesh.read_trimesh(str(cube_path))
        assert mesh.n_vertices() == 8
        assert mesh.n_faces() == 12


class TestMeshDataBridge:

    def test_to_mesh_data(self, lf):
        mesh = _make_triangle(lf)
        md = lf.mesh.to_mesh_data(mesh)
        assert md.vertex_count == 3
        assert md.face_count == 1

    def test_from_mesh_data_round_trip(self, lf):
        mesh1 = lf.mesh.TriMesh()
        mesh1.add_vertex(np.array([1.5, 2.5, 3.5], dtype=np.float64))
        mesh1.add_vertex(np.array([4.0, 0.0, 0.0], dtype=np.float64))
        mesh1.add_vertex(np.array([0.0, 5.0, 0.0], dtype=np.float64))
        verts = list(mesh1.vertices())
        mesh1.add_face(verts[0], verts[1], verts[2])

        md = lf.mesh.to_mesh_data(mesh1)
        mesh2 = lf.mesh.from_mesh_data(md)

        assert mesh2.n_vertices() == 3
        assert mesh2.n_faces() == 1

        pts1 = mesh1.points()
        pts2 = mesh2.points()
        assert np.allclose(pts1, pts2, atol=1e-5)


class TestPolyMesh:

    def test_create_polymesh(self, lf):
        mesh = lf.mesh.PolyMesh()
        assert mesh.n_vertices() == 0

    def test_add_quad_face(self, lf):
        mesh = lf.mesh.PolyMesh()
        v0 = mesh.add_vertex(np.array([0, 0, 0], dtype=np.float64))
        v1 = mesh.add_vertex(np.array([1, 0, 0], dtype=np.float64))
        v2 = mesh.add_vertex(np.array([1, 1, 0], dtype=np.float64))
        v3 = mesh.add_vertex(np.array([0, 1, 0], dtype=np.float64))
        mesh.add_face(v0, v1, v2, v3)
        assert mesh.n_faces() == 1
        assert mesh.n_vertices() == 4


class TestNormals:

    def test_update_normals(self, lf):
        mesh = lf.mesh.TriMesh()
        mesh.request_vertex_normals()
        mesh.request_face_normals()

        v0 = mesh.add_vertex(np.array([0, 0, 0], dtype=np.float64))
        v1 = mesh.add_vertex(np.array([1, 0, 0], dtype=np.float64))
        v2 = mesh.add_vertex(np.array([0, 1, 0], dtype=np.float64))
        mesh.add_face(v0, v1, v2)

        mesh.update_normals()

        normals = mesh.vertex_normals()
        assert normals.shape == (3, 3)
        for i in range(3):
            length = np.sqrt(np.sum(normals[i] ** 2))
            assert abs(length - 1.0) < 1e-5


class TestMeshCopy:

    def test_copy_trimesh(self, lf):
        mesh = _make_triangle(lf)

        import copy
        mesh2 = copy.deepcopy(mesh)
        assert mesh2.n_vertices() == 3
        assert mesh2.n_faces() == 1

        pts2 = mesh2.points()
        pts2[0, 0] = 99.0
        assert abs(mesh.points()[0, 0] - 0.0) < 1e-6
