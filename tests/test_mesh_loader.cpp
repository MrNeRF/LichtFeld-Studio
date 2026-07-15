/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "io/loaders/mesh_loader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <variant>

using namespace lfs::core;

namespace {

    class MeshLoaderTest : public ::testing::Test {
    protected:
        void SetUp() override {
            temp_dir_ = std::filesystem::temp_directory_path() / "lfs_mesh_loader_test";
            std::filesystem::remove_all(temp_dir_);
            std::filesystem::create_directories(temp_dir_);
            mesh_path_ = temp_dir_ / "triangle.obj";
            std::ofstream mesh(mesh_path_);
            ASSERT_TRUE(mesh.is_open());
            mesh << "v -0.5 -0.5 0\n"
                    "v 0.5 -0.5 0\n"
                    "v 0 0.5 0\n"
                    "vn 0 0 1\n"
                    "f 1//1 2//1 3//1\n";
            mesh.close();
            ASSERT_TRUE(mesh.good());
        }

        void TearDown() override { std::filesystem::remove_all(temp_dir_); }

        lfs::io::MeshLoader loader_;
        std::filesystem::path temp_dir_;
        std::filesystem::path mesh_path_;
    };

} // namespace

TEST_F(MeshLoaderTest, ReportsStableLoaderContract) {
    EXPECT_TRUE(loader_.canLoad(mesh_path_));
    EXPECT_FALSE(loader_.canLoad(temp_dir_ / "nonexistent.xyz"));
    EXPECT_EQ(loader_.name(), "Mesh");
    EXPECT_EQ(loader_.priority(), 5);

    const auto extensions = loader_.supportedExtensions();
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), ".obj"),
              extensions.end());
}

TEST_F(MeshLoaderTest, LoadsGeometryNormalsAndBoundedIndices) {
    const auto result = loader_.load(mesh_path_);
    ASSERT_TRUE(result.has_value()) << "Failed to load generated OBJ";
    EXPECT_EQ(result->loader_used, "Mesh");

    const auto* mesh_ptr = std::get_if<std::shared_ptr<MeshData>>(&result->data);
    ASSERT_NE(mesh_ptr, nullptr);
    auto& mesh = **mesh_ptr;
    EXPECT_EQ(mesh.vertex_count(), 3);
    EXPECT_EQ(mesh.face_count(), 1);
    ASSERT_TRUE(mesh.has_normals());
    EXPECT_EQ(mesh.vertices.shape(), TensorShape({3, 3}));
    EXPECT_EQ(mesh.normals.shape(), TensorShape({3, 3}));
    EXPECT_EQ(mesh.indices.shape(), TensorShape({1, 3}));

    auto vertices = mesh.vertices.accessor<float, 2>();
    auto normals = mesh.normals.accessor<float, 2>();
    for (int64_t i = 0; i < mesh.vertex_count(); ++i) {
        const float length = std::sqrt(normals(i, 0) * normals(i, 0) +
                                       normals(i, 1) * normals(i, 1) +
                                       normals(i, 2) * normals(i, 2));
        EXPECT_NEAR(length, 1.0f, 1e-5f);
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_GE(vertices(i, axis), -0.5f);
            EXPECT_LE(vertices(i, axis), 0.5f);
        }
    }

    auto indices = mesh.indices.accessor<int32_t, 2>();
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_GE(indices(0, axis), 0);
        EXPECT_LT(indices(0, axis), mesh.vertex_count());
    }
}

TEST_F(MeshLoaderTest, MissingFileReturnsError) {
    EXPECT_FALSE(loader_.load(temp_dir_ / "missing.obj").has_value());
}
