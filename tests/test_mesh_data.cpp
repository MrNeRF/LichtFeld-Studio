/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

class MeshDataTest : public ::testing::Test {
protected:
    MeshData make_triangle() {
        auto verts = Tensor::empty({3, 3}, Device::CPU, DataType::Float32);
        auto vacc = verts.accessor<float, 2>();
        vacc(0, 0) = 0.0f;
        vacc(0, 1) = 0.0f;
        vacc(0, 2) = 0.0f;
        vacc(1, 0) = 1.0f;
        vacc(1, 1) = 0.0f;
        vacc(1, 2) = 0.0f;
        vacc(2, 0) = 0.0f;
        vacc(2, 1) = 1.0f;
        vacc(2, 2) = 0.0f;

        auto idx = Tensor::empty({1, 3}, Device::CPU, DataType::Int32);
        auto iacc = idx.accessor<int32_t, 2>();
        iacc(0, 0) = 0;
        iacc(0, 1) = 1;
        iacc(0, 2) = 2;

        return MeshData(std::move(verts), std::move(idx));
    }

    MeshData make_quad() {
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

TEST_F(MeshDataTest, ConstructorValidatesDimensions) {
    auto mesh = make_triangle();
    ASSERT_EQ(mesh.vertex_count(), 3);
    ASSERT_EQ(mesh.face_count(), 1);
    EXPECT_EQ(mesh.vertices.shape()[0], 3u);
    EXPECT_EQ(mesh.vertices.shape()[1], 3u);
    EXPECT_EQ(mesh.indices.shape()[0], 1u);
    EXPECT_EQ(mesh.indices.shape()[1], 3u);
}

TEST_F(MeshDataTest, DefaultConstructorEmpty) {
    MeshData mesh;
    EXPECT_EQ(mesh.vertex_count(), 0);
    EXPECT_EQ(mesh.face_count(), 0);
    EXPECT_FALSE(mesh.has_normals());
    EXPECT_FALSE(mesh.has_tangents());
    EXPECT_FALSE(mesh.has_texcoords());
    EXPECT_FALSE(mesh.has_colors());
}

TEST_F(MeshDataTest, OptionalAttributes) {
    auto mesh = make_triangle();
    EXPECT_FALSE(mesh.has_normals());
    EXPECT_FALSE(mesh.has_tangents());
    EXPECT_FALSE(mesh.has_texcoords());
    EXPECT_FALSE(mesh.has_colors());

    mesh.normals = Tensor::empty({3, 3}, Device::CPU, DataType::Float32);
    EXPECT_TRUE(mesh.has_normals());

    mesh.texcoords = Tensor::empty({3, 2}, Device::CPU, DataType::Float32);
    EXPECT_TRUE(mesh.has_texcoords());
}

TEST_F(MeshDataTest, VertexDataPreserved) {
    auto mesh = make_triangle();
    auto vacc = mesh.vertices.accessor<float, 2>();
    EXPECT_FLOAT_EQ(vacc(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(vacc(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(vacc(2, 1), 1.0f);
}

TEST_F(MeshDataTest, IndexDataPreserved) {
    auto mesh = make_quad();
    auto iacc = mesh.indices.accessor<int32_t, 2>();
    EXPECT_EQ(iacc(0, 0), 0);
    EXPECT_EQ(iacc(0, 1), 1);
    EXPECT_EQ(iacc(0, 2), 2);
    EXPECT_EQ(iacc(1, 0), 0);
    EXPECT_EQ(iacc(1, 1), 2);
    EXPECT_EQ(iacc(1, 2), 3);
}

TEST_F(MeshDataTest, ComputeNormals) {
    auto mesh = make_triangle();
    ASSERT_FALSE(mesh.has_normals());

    mesh.compute_normals();

    ASSERT_TRUE(mesh.has_normals());
    EXPECT_EQ(mesh.normals.shape()[0], 3u);
    EXPECT_EQ(mesh.normals.shape()[1], 3u);

    auto nacc = mesh.normals.accessor<float, 2>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(nacc(i, 0), 0.0f, 1e-5f);
        EXPECT_NEAR(nacc(i, 1), 0.0f, 1e-5f);
        EXPECT_NEAR(nacc(i, 2), 1.0f, 1e-5f);
    }
}

TEST_F(MeshDataTest, ToDeviceCPU) {
    auto mesh = make_triangle();
    mesh.normals = Tensor::ones({3, 3}, Device::CPU, DataType::Float32);
    mesh.texcoords = Tensor::zeros({3, 2}, Device::CPU, DataType::Float32);

    auto cpu_mesh = mesh.to(Device::CPU);
    EXPECT_EQ(cpu_mesh.vertex_count(), 3);
    EXPECT_EQ(cpu_mesh.face_count(), 1);
    EXPECT_TRUE(cpu_mesh.has_normals());
    EXPECT_TRUE(cpu_mesh.has_texcoords());
}

TEST_F(MeshDataTest, ToDeviceCUDA) {
    auto mesh = make_triangle();
    auto gpu_mesh = mesh.to(Device::CUDA);

    EXPECT_EQ(gpu_mesh.vertex_count(), 3);
    EXPECT_EQ(gpu_mesh.face_count(), 1);
    EXPECT_EQ(gpu_mesh.vertices.device(), Device::CUDA);
    EXPECT_EQ(gpu_mesh.indices.device(), Device::CUDA);

    auto roundtrip = gpu_mesh.to(Device::CPU);
    auto vacc = roundtrip.vertices.accessor<float, 2>();
    EXPECT_FLOAT_EQ(vacc(1, 0), 1.0f);
}

TEST_F(MeshDataTest, ComputeNormalsOnGPUMesh) {
    auto mesh = make_quad();
    mesh = mesh.to(Device::CUDA);
    EXPECT_EQ(mesh.vertices.device(), Device::CUDA);

    mesh.compute_normals();

    EXPECT_TRUE(mesh.has_normals());
    EXPECT_EQ(mesh.normals.device(), Device::CUDA);

    auto cpu_normals = mesh.normals.to(Device::CPU);
    auto nacc = cpu_normals.accessor<float, 2>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(nacc(i, 2), 1.0f, 1e-5f);
    }
}

TEST_F(MeshDataTest, MaterialsAndSubmeshes) {
    auto mesh = make_quad();
    Material mat;
    mat.metallic = 0.5f;
    mat.roughness = 0.8f;
    mat.name = "test_material";
    mesh.materials.push_back(mat);
    mesh.submeshes.push_back({0, 6, 0});

    EXPECT_EQ(mesh.materials.size(), 1u);
    EXPECT_FLOAT_EQ(mesh.materials[0].metallic, 0.5f);
    EXPECT_EQ(mesh.materials[0].name, "test_material");
    EXPECT_EQ(mesh.submeshes[0].start_index, 0u);
    EXPECT_EQ(mesh.submeshes[0].index_count, 6u);
    EXPECT_EQ(mesh.submeshes[0].material_index, 0u);
}
