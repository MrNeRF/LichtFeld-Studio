/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "io/loaders/mesh_loader.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace lfs::core;

namespace {
    const std::filesystem::path data_dir{TEST_DATA_DIR};
}

class MeshLoaderTest : public ::testing::Test {
protected:
    lfs::io::MeshLoader loader;

    void SetUp() override {
        ASSERT_TRUE(std::filesystem::exists(data_dir / "test_cube.obj"))
            << "test_cube.obj not found in " << data_dir;
    }
};

TEST_F(MeshLoaderTest, CanLoadOBJ) {
    EXPECT_TRUE(loader.canLoad(data_dir / "test_cube.obj"));
}

TEST_F(MeshLoaderTest, CannotLoadNonMesh) {
    EXPECT_FALSE(loader.canLoad(data_dir / "nonexistent.xyz"));
}

TEST_F(MeshLoaderTest, SupportedExtensions) {
    auto exts = loader.supportedExtensions();
    EXPECT_FALSE(exts.empty());

    bool has_obj = false;
    for (const auto& e : exts) {
        if (e == ".obj" || e == ".OBJ")
            has_obj = true;
    }
    EXPECT_TRUE(has_obj);
}

TEST_F(MeshLoaderTest, LoadCubeOBJ) {
    auto result = loader.load(data_dir / "test_cube.obj");
    ASSERT_TRUE(result.has_value()) << "Failed to load test_cube.obj";

    auto& load_result = result.value();
    EXPECT_EQ(load_result.loader_used, "Mesh");

    auto* mesh_ptr = std::get_if<std::shared_ptr<MeshData>>(&load_result.data);
    ASSERT_NE(mesh_ptr, nullptr);
    auto& mesh = **mesh_ptr;

    EXPECT_EQ(mesh.vertex_count(), 24); // Assimp splits at normal discontinuities
    EXPECT_EQ(mesh.face_count(), 12);
}

TEST_F(MeshLoaderTest, LoadCubeHasNormals) {
    auto result = loader.load(data_dir / "test_cube.obj");
    ASSERT_TRUE(result.has_value());

    auto& mesh = *std::get<std::shared_ptr<MeshData>>(result->data);
    EXPECT_TRUE(mesh.has_normals());
    EXPECT_EQ(mesh.normals.shape()[0], static_cast<size_t>(mesh.vertex_count()));
    EXPECT_EQ(mesh.normals.shape()[1], 3u);

    auto nacc = mesh.normals.accessor<float, 2>();
    for (int64_t i = 0; i < mesh.vertex_count(); ++i) {
        float len = std::sqrt(nacc(i, 0) * nacc(i, 0) +
                              nacc(i, 1) * nacc(i, 1) +
                              nacc(i, 2) * nacc(i, 2));
        EXPECT_NEAR(len, 1.0f, 1e-4f) << "Normal not unit length at vertex " << i;
    }
}

TEST_F(MeshLoaderTest, LoadCubeVertexBounds) {
    auto result = loader.load(data_dir / "test_cube.obj");
    ASSERT_TRUE(result.has_value());

    auto& mesh = *std::get<std::shared_ptr<MeshData>>(result->data);
    auto vacc = mesh.vertices.accessor<float, 2>();

    for (int64_t i = 0; i < mesh.vertex_count(); ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_GE(vacc(i, j), -0.5f - 1e-6f);
            EXPECT_LE(vacc(i, j), 0.5f + 1e-6f);
        }
    }
}

TEST_F(MeshLoaderTest, LoadCubeIndexBounds) {
    auto result = loader.load(data_dir / "test_cube.obj");
    ASSERT_TRUE(result.has_value());

    auto& mesh = *std::get<std::shared_ptr<MeshData>>(result->data);
    auto iacc = mesh.indices.accessor<int32_t, 2>();

    for (int64_t i = 0; i < mesh.face_count(); ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_GE(iacc(i, j), 0);
            EXPECT_LT(iacc(i, j), static_cast<int32_t>(mesh.vertex_count()));
        }
    }
}

TEST_F(MeshLoaderTest, LoadNonexistentFile) {
    auto result = loader.load(data_dir / "does_not_exist.obj");
    EXPECT_FALSE(result.has_value());
}

TEST_F(MeshLoaderTest, LoaderName) {
    EXPECT_EQ(loader.name(), "Mesh");
}

TEST_F(MeshLoaderTest, LoaderPriority) {
    EXPECT_EQ(loader.priority(), 5);
}
