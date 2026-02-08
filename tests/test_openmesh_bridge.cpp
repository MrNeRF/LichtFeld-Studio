/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/mesh/openmesh_bridge.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;
using lfs::io::mesh::from_openmesh;
using lfs::io::mesh::to_openmesh;
using lfs::io::mesh::TriMesh;

class OpenMeshBridgeTest : public ::testing::Test {
protected:
    TriMesh make_triangle_mesh() {
        TriMesh mesh;
        auto v0 = mesh.add_vertex(TriMesh::Point(0.0f, 0.0f, 0.0f));
        auto v1 = mesh.add_vertex(TriMesh::Point(1.0f, 0.0f, 0.0f));
        auto v2 = mesh.add_vertex(TriMesh::Point(0.0f, 1.0f, 0.0f));
        mesh.add_face(v0, v1, v2);
        return mesh;
    }

    TriMesh make_tetrahedron() {
        TriMesh mesh;
        auto v0 = mesh.add_vertex(TriMesh::Point(0.0f, 0.0f, 0.0f));
        auto v1 = mesh.add_vertex(TriMesh::Point(1.0f, 0.0f, 0.0f));
        auto v2 = mesh.add_vertex(TriMesh::Point(0.5f, 1.0f, 0.0f));
        auto v3 = mesh.add_vertex(TriMesh::Point(0.5f, 0.5f, 1.0f));
        mesh.add_face(v0, v1, v2);
        mesh.add_face(v0, v3, v1);
        mesh.add_face(v1, v3, v2);
        mesh.add_face(v0, v2, v3);
        return mesh;
    }

    MeshData make_mesh_data_quad() {
        auto verts = Tensor::empty({4, 3}, Device::CPU, DataType::Float32);
        auto vacc = verts.accessor<float, 2>();
        vacc(0, 0) = 0.0f;
        vacc(0, 1) = 0.0f;
        vacc(0, 2) = 0.0f;
        vacc(1, 0) = 1.0f;
        vacc(1, 1) = 0.0f;
        vacc(1, 2) = 0.0f;
        vacc(2, 0) = 1.0f;
        vacc(2, 1) = 1.0f;
        vacc(2, 2) = 0.0f;
        vacc(3, 0) = 0.0f;
        vacc(3, 1) = 1.0f;
        vacc(3, 2) = 0.0f;

        auto idx = Tensor::empty({2, 3}, Device::CPU, DataType::Int32);
        auto iacc = idx.accessor<int32_t, 2>();
        iacc(0, 0) = 0;
        iacc(0, 1) = 1;
        iacc(0, 2) = 2;
        iacc(1, 0) = 0;
        iacc(1, 1) = 2;
        iacc(1, 2) = 3;

        return MeshData(std::move(verts), std::move(idx));
    }
};

TEST_F(OpenMeshBridgeTest, FromOpenMeshVertexCount) {
    auto mesh = make_triangle_mesh();
    auto data = from_openmesh(mesh);

    ASSERT_EQ(data.vertex_count(), 3);
    ASSERT_EQ(data.face_count(), 1);
    EXPECT_EQ(data.vertices.dtype(), DataType::Float32);
    EXPECT_EQ(data.indices.dtype(), DataType::Int32);
}

TEST_F(OpenMeshBridgeTest, FromOpenMeshVertexPositions) {
    auto mesh = make_triangle_mesh();
    auto data = from_openmesh(mesh);

    auto vacc = data.vertices.accessor<float, 2>();
    EXPECT_FLOAT_EQ(vacc(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(vacc(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(vacc(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(vacc(2, 1), 1.0f);
}

TEST_F(OpenMeshBridgeTest, FromOpenMeshFaceIndices) {
    auto mesh = make_triangle_mesh();
    auto data = from_openmesh(mesh);

    auto iacc = data.indices.accessor<int32_t, 2>();
    std::set<int32_t> face_verts = {iacc(0, 0), iacc(0, 1), iacc(0, 2)};
    EXPECT_EQ(face_verts, (std::set<int32_t>{0, 1, 2}));
}

TEST_F(OpenMeshBridgeTest, FromOpenMeshWithNormals) {
    auto mesh = make_triangle_mesh();
    mesh.request_vertex_normals();
    mesh.request_face_normals();
    mesh.update_normals();

    auto data = from_openmesh(mesh);
    ASSERT_TRUE(data.has_normals());
    EXPECT_EQ(data.normals.shape()[0], 3u);
    EXPECT_EQ(data.normals.shape()[1], 3u);

    auto nacc = data.normals.accessor<float, 2>();
    for (int64_t i = 0; i < 3; ++i) {
        float len = std::sqrt(nacc(i, 0) * nacc(i, 0) +
                              nacc(i, 1) * nacc(i, 1) +
                              nacc(i, 2) * nacc(i, 2));
        EXPECT_NEAR(len, 1.0f, 1e-5f);
    }
}

TEST_F(OpenMeshBridgeTest, ToOpenMeshVertexCount) {
    auto data = make_mesh_data_quad();
    auto mesh = to_openmesh(data);

    EXPECT_EQ(mesh.n_vertices(), 4u);
    EXPECT_EQ(mesh.n_faces(), 2u);
}

TEST_F(OpenMeshBridgeTest, ToOpenMeshVertexPositions) {
    auto data = make_mesh_data_quad();
    auto mesh = to_openmesh(data);

    auto p0 = mesh.point(TriMesh::VertexHandle(0));
    EXPECT_FLOAT_EQ(p0[0], 0.0f);
    EXPECT_FLOAT_EQ(p0[1], 0.0f);

    auto p1 = mesh.point(TriMesh::VertexHandle(1));
    EXPECT_FLOAT_EQ(p1[0], 1.0f);
}

TEST_F(OpenMeshBridgeTest, RoundTripPreservesGeometry) {
    auto mesh_in = make_tetrahedron();
    auto data = from_openmesh(mesh_in);
    auto mesh_out = to_openmesh(data);

    ASSERT_EQ(mesh_out.n_vertices(), mesh_in.n_vertices());
    ASSERT_EQ(mesh_out.n_faces(), mesh_in.n_faces());

    for (size_t i = 0; i < mesh_in.n_vertices(); ++i) {
        auto vh = TriMesh::VertexHandle(static_cast<int>(i));
        auto p_in = mesh_in.point(vh);
        auto p_out = mesh_out.point(vh);
        EXPECT_NEAR(p_in[0], p_out[0], 1e-6f);
        EXPECT_NEAR(p_in[1], p_out[1], 1e-6f);
        EXPECT_NEAR(p_in[2], p_out[2], 1e-6f);
    }
}

TEST_F(OpenMeshBridgeTest, RoundTripPreservesNormals) {
    auto mesh_in = make_tetrahedron();
    mesh_in.request_vertex_normals();
    mesh_in.request_face_normals();
    mesh_in.update_normals();

    auto data = from_openmesh(mesh_in);
    ASSERT_TRUE(data.has_normals());

    auto mesh_out = to_openmesh(data);
    ASSERT_TRUE(mesh_out.has_vertex_normals());

    for (size_t i = 0; i < mesh_in.n_vertices(); ++i) {
        auto vh = TriMesh::VertexHandle(static_cast<int>(i));
        auto n_in = mesh_in.normal(vh);
        auto n_out = mesh_out.normal(vh);
        EXPECT_NEAR(n_in[0], n_out[0], 1e-5f);
        EXPECT_NEAR(n_in[1], n_out[1], 1e-5f);
        EXPECT_NEAR(n_in[2], n_out[2], 1e-5f);
    }
}

TEST_F(OpenMeshBridgeTest, RoundTripMeshDataVertices) {
    auto data_in = make_mesh_data_quad();
    auto mesh = to_openmesh(data_in);
    auto data_out = from_openmesh(mesh);

    ASSERT_EQ(data_out.vertex_count(), data_in.vertex_count());
    ASSERT_EQ(data_out.face_count(), data_in.face_count());

    auto vacc_in = data_in.vertices.accessor<float, 2>();
    auto vacc_out = data_out.vertices.accessor<float, 2>();
    for (int64_t i = 0; i < data_in.vertex_count(); ++i) {
        EXPECT_NEAR(vacc_in(i, 0), vacc_out(i, 0), 1e-6f);
        EXPECT_NEAR(vacc_in(i, 1), vacc_out(i, 1), 1e-6f);
        EXPECT_NEAR(vacc_in(i, 2), vacc_out(i, 2), 1e-6f);
    }
}

TEST_F(OpenMeshBridgeTest, ToOpenMeshWithTexCoords) {
    auto data = make_mesh_data_quad();
    data.texcoords = Tensor::empty({4, 2}, Device::CPU, DataType::Float32);
    auto tcacc = data.texcoords.accessor<float, 2>();
    tcacc(0, 0) = 0.0f;
    tcacc(0, 1) = 0.0f;
    tcacc(1, 0) = 1.0f;
    tcacc(1, 1) = 0.0f;
    tcacc(2, 0) = 1.0f;
    tcacc(2, 1) = 1.0f;
    tcacc(3, 0) = 0.0f;
    tcacc(3, 1) = 1.0f;

    auto mesh = to_openmesh(data);
    ASSERT_TRUE(mesh.has_vertex_texcoords2D());

    auto tc0 = mesh.texcoord2D(TriMesh::VertexHandle(0));
    EXPECT_FLOAT_EQ(tc0[0], 0.0f);
    EXPECT_FLOAT_EQ(tc0[1], 0.0f);

    auto tc2 = mesh.texcoord2D(TriMesh::VertexHandle(2));
    EXPECT_FLOAT_EQ(tc2[0], 1.0f);
    EXPECT_FLOAT_EQ(tc2[1], 1.0f);
}
