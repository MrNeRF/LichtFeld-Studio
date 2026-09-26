/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "io/formats/ply.hpp"
#include "io/loaders/mesh_loader.hpp"
#include "rendering/mesh2splat.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <glm/gtc/quaternion.hpp>
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

TEST_F(MeshLoaderTest, MeshToSplatPreservesCoordinatesThroughPlyExport) {
    // An offset, tilted, asymmetric triangle makes axis swaps and origin
    // rotations observable. Exercise both PLY encodings from issue #2135.
    const std::array points{glm::vec3(1, 2, 3), glm::vec3(5, 2, 4), glm::vec3(1, 4, 5)};
    const glm::vec3 normal = glm::normalize(glm::cross(points[1] - points[0], points[2] - points[0]));
    for (const std::string format : {"obj", "ascii", "binary_little_endian"}) {
        SCOPED_TRACE(format);
        const auto path = temp_dir_ / (format + (format == "obj" ? ".obj" : ".ply"));
        std::ofstream file(path, std::ios::binary);
        if (format == "obj") {
            for (const auto& p : points)
                file << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
            file << "f 1 2 3\n";
        } else {
            file << "ply\nformat " << format << " 1.0\nelement vertex 3\n"
                                                "property float x\nproperty float y\nproperty float z\n"
                                                "element face 1\nproperty list uchar int vertex_indices\nend_header\n";
            if (format == "ascii") {
                for (const auto& p : points)
                    file << p.x << ' ' << p.y << ' ' << p.z << '\n';
                file << "3 0 1 2\n";
            } else {
                for (const auto& p : points) {
                    const std::array coords{p.x, p.y, p.z};
                    file.write(reinterpret_cast<const char*>(coords.data()), sizeof(coords));
                }
                file.put(3);
                const std::array<int32_t, 3> indices{0, 1, 2};
                file.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
            }
        }
        file.close();
        ASSERT_TRUE(file.good());
        const auto loaded = lfs::io::Loader::create()->load(path);
        ASSERT_TRUE(loaded.has_value());
        const auto* mesh = std::get_if<std::shared_ptr<MeshData>>(&loaded->data);
        ASSERT_NE(mesh, nullptr);
        Mesh2SplatOptions options;
        options.resolution_target = 32;
        auto converted = lfs::rendering::mesh_to_splat(**mesh, options);
        ASSERT_TRUE(converted.has_value()) << converted.error();
        const auto exported = temp_dir_ / (format + "_splat.ply");
        ASSERT_TRUE(lfs::io::save_ply(**converted, {.output_path = exported, .binary = true}).has_value());
        auto reloaded = lfs::io::Loader::create()->load(exported);
        ASSERT_TRUE(reloaded.has_value());
        const auto* splat = std::get_if<std::shared_ptr<SplatData>>(&reloaded->data);
        ASSERT_NE(splat, nullptr);
        const auto means = (*splat)->means_raw().cpu();
        const auto rotations = (*splat)->rotation_raw().cpu();
        ASSERT_GT(means.size(0), 0u);
        for (size_t i = 0; i < means.size(0); ++i) {
            const auto* p = means.ptr<float>() + 3 * i;
            const glm::vec3 point(p[0], p[1], p[2]);
            ASSERT_NEAR(glm::dot(point - points[0], normal), 0.0f, 2e-5f);
            const float u = (point.x - 1.0f) / 4.0f;
            const float v = (point.y - 2.0f) / 2.0f;
            ASSERT_GE(u, -1e-5f);
            ASSERT_GE(v, -1e-5f);
            ASSERT_LE(u + v, 1.0f + 1e-5f);
            const auto* q = rotations.ptr<float>() + 4 * i;
            const glm::vec3 thin_axis = glm::quat(q[0], q[1], q[2], q[3]) * glm::vec3(0, 0, 1);
            ASSERT_NEAR(std::abs(glm::dot(thin_axis, normal)), 1.0f, 2e-5f);
        }
    }
}
