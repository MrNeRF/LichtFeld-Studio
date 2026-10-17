/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"
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
#include <nlohmann/json.hpp>
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

        std::filesystem::path write_scene(const nlohmann::json& nodes, bool two_materials = false) {
            const std::array<float, 36> attributes{
                0, 0, 0, 1, 0, 1, 0, 1, 1,
                -0.577350269f, -0.577350269f, 0.577350269f,
                -0.577350269f, -0.577350269f, 0.577350269f,
                -0.577350269f, -0.577350269f, 0.577350269f,
                0, 0, 1, 0, 0, 1,
                1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1};
            const std::array<uint16_t, 3> indices{0, 1, 2};
            std::ofstream binary(temp_dir_ / "scene.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(attributes.data()), sizeof(attributes));
            binary.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
            binary.close();
            auto scene = nlohmann::json::parse(R"({
                "asset":{"version":"2.0"},
                "buffers":[{"uri":"scene.bin","byteLength":150}],
                "bufferViews":[
                    {"buffer":0,"byteOffset":0,"byteLength":36},
                    {"buffer":0,"byteOffset":36,"byteLength":36},
                    {"buffer":0,"byteOffset":72,"byteLength":24},
                    {"buffer":0,"byteOffset":96,"byteLength":48},
                    {"buffer":0,"byteOffset":144,"byteLength":6}],
                "accessors":[
                    {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,1]},
                    {"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},
                    {"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"},
                    {"bufferView":3,"componentType":5126,"count":3,"type":"VEC4"},
                    {"bufferView":4,"componentType":5123,"count":3,"type":"SCALAR"}],
                "materials":[{"name":"red","pbrMetallicRoughness":{"baseColorFactor":[1,0,0,1]}},
                             {"name":"blue","pbrMetallicRoughness":{"baseColorFactor":[0,0,1,1]}}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2,"COLOR_0":3},"indices":4,"material":0}]}],
                "scenes":[{"nodes":[0]}],"scene":0})");
            scene["nodes"] = nodes;
            if (two_materials) {
                auto primitive = scene["meshes"][0]["primitives"][0];
                primitive["material"] = 1;
                scene["meshes"][0]["primitives"].push_back(primitive);
            }
            const auto path = temp_dir_ / "scene.gltf";
            std::ofstream(path) << scene.dump();
            return path;
        }

        std::filesystem::path write_planar_scene(const nlohmann::json& nodes,
                                                 const glm::vec3 normal = {0, 0, 1}) {
            const auto path = write_scene(nodes);
            const std::array<float, 18> attributes{
                0, 0, 0, 1, 0, 0, 0, 1, 0,
                normal.x, normal.y, normal.z,
                normal.x, normal.y, normal.z,
                normal.x, normal.y, normal.z};
            std::fstream binary(temp_dir_ / "scene.bin", std::ios::in | std::ios::out | std::ios::binary);
            binary.write(reinterpret_cast<const char*>(attributes.data()), sizeof(attributes));
            auto scene = nlohmann::json::parse(std::ifstream(path));
            scene["accessors"][0]["max"] = {1, 1, 0};
            std::ofstream(path) << scene.dump();
            return path;
        }

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
    ASSERT_EQ(mesh.materials.size(), 1u);
    EXPECT_EQ(mesh.materials[0].alpha_mode, AlphaMode::Opaque);
    EXPECT_FLOAT_EQ(mesh.materials[0].alpha_cutoff, 0.5f);
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

TEST_F(MeshLoaderTest, ImportsGltfAlphaModesAndDefaults) {
    const auto path = write_scene(nlohmann::json::parse(R"([{"mesh":0}])"));
    auto scene = nlohmann::json::parse(std::ifstream(path));
    scene["materials"][0]["alphaMode"] = "MASK";
    scene["materials"][0]["alphaCutoff"] = 0.25f;
    std::ofstream(path) << scene.dump();

    auto loaded = loader_.load(path);
    ASSERT_TRUE(loaded.has_value());
    const auto& masked = std::get<std::shared_ptr<MeshData>>(loaded->data)->materials[0];
    EXPECT_EQ(masked.alpha_mode, AlphaMode::Mask);
    EXPECT_FLOAT_EQ(masked.alpha_cutoff, 0.25f);

    scene["materials"][0].erase("alphaCutoff");
    std::ofstream(path) << scene.dump();
    loaded = loader_.load(path);
    ASSERT_TRUE(loaded.has_value());
    const auto& default_cutoff =
        std::get<std::shared_ptr<MeshData>>(loaded->data)->materials[0];
    EXPECT_EQ(default_cutoff.alpha_mode, AlphaMode::Mask);
    EXPECT_FLOAT_EQ(default_cutoff.alpha_cutoff, 0.5f);

    scene["materials"][0]["alphaMode"] = "BLEND";
    scene["materials"][0]["alphaCutoff"] = 0.25f;
    std::ofstream(path) << scene.dump();
    loaded = loader_.load(path);
    ASSERT_TRUE(loaded.has_value());
    const auto& blended = std::get<std::shared_ptr<MeshData>>(loaded->data)->materials[0];
    EXPECT_EQ(blended.alpha_mode, AlphaMode::Blend);
    EXPECT_FLOAT_EQ(blended.alpha_cutoff, 0.25f);

    scene["materials"][0].erase("alphaMode");
    scene["materials"][0].erase("alphaCutoff");
    std::ofstream(path) << scene.dump();
    loaded = loader_.load(path);
    ASSERT_TRUE(loaded.has_value());
    const auto& opaque = std::get<std::shared_ptr<MeshData>>(loaded->data)->materials[0];
    EXPECT_EQ(opaque.alpha_mode, AlphaMode::Opaque);
    EXPECT_FLOAT_EQ(opaque.alpha_cutoff, 0.5f);
}

TEST_F(MeshLoaderTest, MissingFileReturnsError) {
    EXPECT_FALSE(loader_.load(temp_dir_ / "missing.obj").has_value());
}

TEST_F(MeshLoaderTest, RejectsMeshesWithoutTriangleFaces) {
    const std::array cases{
        std::pair{"vertices_only.obj", std::string("v 0 0 0\nv 1 0 0\nv 0 1 0\n")},
        std::pair{"unparsed.obj", std::string("this is not an obj\n\0\1\2 random bytes \n", 37)}};
    for (const auto& [filename, contents] : cases) {
        const auto path = temp_dir_ / filename;
        std::ofstream out(path, std::ios::binary);
        out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        out.close();
        ASSERT_TRUE(out.good());
        for (const bool geometry_only : {false, true}) {
            SCOPED_TRACE(std::string(filename) + (geometry_only ? " geometry" : " render"));
            lfs::io::LoadOptions options;
            options.mesh_geometry_only = geometry_only;
            const auto result = loader_.load(path, options);
            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
            EXPECT_FALSE(result.error().message.empty());
        }
    }
}

TEST_F(MeshLoaderTest, RejectsNonFiniteVertexCoordinates) {
    const auto obj = temp_dir_ / "nonfinite.obj";
    std::ofstream(obj) << "v 0 0 0\n"
                          "v 1 0 0\n"
                          "v nan 1 0\n"
                          "v 0 inf 0\n"
                          "f 1 2 3\n"
                          "f 1 2 4\n";
    const auto ply = temp_dir_ / "nonfinite.ply";
    std::ofstream(ply) << "ply\nformat ascii 1.0\n"
                          "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
                          "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
                          "0 0 0\n1 0 0\n0 1 0\nnan inf 0\n"
                          "3 0 1 3\n3 1 2 3\n";

    for (const auto& path : {obj, ply}) {
        for (const bool geometry_only : {false, true}) {
            SCOPED_TRACE(path.extension().string());
            SCOPED_TRACE(geometry_only);
            lfs::io::LoadOptions options;
            options.mesh_geometry_only = geometry_only;
            const auto result = loader_.load(path, options);
            EXPECT_FALSE(result.has_value());
            if (!result)
                EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
        }
    }
}

TEST_F(MeshLoaderTest, KeepsFinitePlyGeometry) {
    const auto path = temp_dir_ / "finite.ply";
    std::ofstream(path) << "ply\nformat ascii 1.0\n"
                           "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                           "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                           "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
    for (const bool geometry_only : {false, true}) {
        lfs::io::LoadOptions options;
        options.mesh_geometry_only = geometry_only;
        const auto result = loader_.load(path, options);
        ASSERT_TRUE(result.has_value());
        const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
        ASSERT_EQ(mesh->vertex_count(), 3);
        ASSERT_EQ(mesh->face_count(), 1);
        auto vertices = mesh->vertices.accessor<float, 2>();
        bool found_top = false;
        for (int i = 0; i < 3; ++i) {
            for (int axis = 0; axis < 3; ++axis)
                EXPECT_TRUE(std::isfinite(vertices(i, axis)));
            found_top |= vertices(i, 0) == 0.0f && vertices(i, 1) == 1.0f && vertices(i, 2) == 0.0f;
        }
        EXPECT_TRUE(found_top);
    }
}

// Both import modes bake placement; geometry-only loads omit render attributes.
TEST_F(MeshLoaderTest, GeometryOnlyBakesNodeTransformAndSkipsRenderAttributes) {
    const auto gltf_path = temp_dir_ / "transformed.gltf";
    const auto bin_path = temp_dir_ / "transformed.bin";
    const std::array<float, 24> attributes{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<uint16_t, 3> triangle{0, 1, 2};
    {
        std::ofstream binary(bin_path, std::ios::binary);
        binary.write(reinterpret_cast<const char*>(attributes.data()), sizeof(attributes));
        binary.write(reinterpret_cast<const char*>(triangle.data()), sizeof(triangle));
        ASSERT_TRUE(binary.good());
    }
    {
        std::ofstream gltf(gltf_path);
        gltf << R"({
  "asset":{"version":"2.0"},
  "buffers":[{"uri":"transformed.bin","byteLength":102}],
  "bufferViews":[
    {"buffer":0,"byteOffset":0,"byteLength":36},
    {"buffer":0,"byteOffset":36,"byteLength":36},
    {"buffer":0,"byteOffset":72,"byteLength":24},
    {"buffer":0,"byteOffset":96,"byteLength":6}
  ],
  "accessors":[
    {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
    {"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},
    {"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"},
    {"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}
  ],
  "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3}]}],
  "nodes":[{"mesh":0,"translation":[3,4,5]}],
  "scenes":[{"nodes":[0]}],
  "scene":0
})";
        ASSERT_TRUE(gltf.good());
    }

    const auto normal_result = loader_.load(gltf_path);
    ASSERT_TRUE(normal_result.has_value());
    const auto* normal_mesh = std::get_if<std::shared_ptr<MeshData>>(&normal_result->data);
    ASSERT_NE(normal_mesh, nullptr);
    EXPECT_TRUE((*normal_mesh)->has_normals());
    EXPECT_TRUE((*normal_mesh)->has_texcoords());
    auto normal_vertices = (*normal_mesh)->vertices.accessor<float, 2>();
    EXPECT_NEAR(normal_vertices(0, 0), 3.0f, 1e-6f);

    lfs::io::LoadOptions options;
    options.mesh_geometry_only = true;
    const auto geometry_result = loader_.load(gltf_path, options);
    ASSERT_TRUE(geometry_result.has_value());
    const auto* geometry_mesh = std::get_if<std::shared_ptr<MeshData>>(&geometry_result->data);
    ASSERT_NE(geometry_mesh, nullptr);
    ASSERT_EQ((*geometry_mesh)->vertex_count(), 3);
    EXPECT_EQ((*geometry_mesh)->face_count(), 1);
    EXPECT_FALSE((*geometry_mesh)->has_normals());
    EXPECT_FALSE((*geometry_mesh)->has_tangents());
    EXPECT_FALSE((*geometry_mesh)->has_texcoords());
    EXPECT_FALSE((*geometry_mesh)->has_colors());
    EXPECT_TRUE((*geometry_mesh)->texture_images.empty());
    auto vertices = (*geometry_mesh)->vertices.accessor<float, 2>();
    EXPECT_NEAR(vertices(0, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(vertices(0, 1), 4.0f, 1e-6f);
    EXPECT_NEAR(vertices(0, 2), 5.0f, 1e-6f);
    EXPECT_NEAR(vertices(1, 0), 4.0f, 1e-6f);
    EXPECT_NEAR(vertices(2, 1), 5.0f, 1e-6f);
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

// Catches vertex colors taken as linear for formats that store display values (a PLY red of 200 left
// Mesh2Splat as 228), and glTF's already linear COLOR_0 being converted a second time.
TEST_F(MeshLoaderTest, VertexColorsLoadLinearAndLeaveMesh2SplatAsStored) {
    const auto ply = temp_dir_ / "colored.ply";
    std::ofstream(ply) << "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                          "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                          "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                          "0 0 0 200 100 50\n1 0 0 200 100 50\n0 1 0 200 100 50\n3 0 1 2\n";
    const auto loaded = loader_.load(ply);
    ASSERT_TRUE(loaded.has_value());
    const auto mesh = std::get<std::shared_ptr<MeshData>>(loaded->data);
    ASSERT_TRUE(mesh->has_colors());
    const std::array<float, 3> stored{200 / 255.0f, 100 / 255.0f, 50 / 255.0f};
    auto colors = mesh->colors.accessor<float, 2>();
    for (int c = 0; c < 3; ++c)
        EXPECT_NEAR(colors(0, c), std::pow(stored[c], 2.2f), 1e-5f);

    Mesh2SplatOptions options;
    options.resolution_target = 32;
    const auto converted = lfs::rendering::mesh_to_splat(*mesh, options);
    ASSERT_TRUE(converted.has_value()) << converted.error();
    const auto sh0 = (*converted)->sh0().cpu().contiguous();
    for (int c = 0; c < 3; ++c)
        EXPECT_NEAR(sh0.ptr<float>()[c] * 0.28209479177387814f + 0.5f, stored[c], 1.0f / 255.0f);

    const auto gltf = write_scene(nlohmann::json::parse(R"([{"mesh":0}])"));
    const std::array<float, 12> half_grey{0.5f, 0.5f, 0.5f, 1, 0.5f, 0.5f, 0.5f, 1, 0.5f, 0.5f, 0.5f, 1};
    std::fstream binary(temp_dir_ / "scene.bin", std::ios::in | std::ios::out | std::ios::binary);
    binary.seekp(96);
    binary.write(reinterpret_cast<const char*>(half_grey.data()), sizeof(half_grey));
    binary.close();
    const auto gltf_loaded = loader_.load(gltf);
    ASSERT_TRUE(gltf_loaded.has_value());
    const auto gltf_mesh = std::get<std::shared_ptr<MeshData>>(gltf_loaded->data);
    auto gltf_colors = gltf_mesh->colors.accessor<float, 2>();
    EXPECT_FLOAT_EQ(gltf_colors(0, 0), 0.5f);
}

TEST_F(MeshLoaderTest, ImportsNestedInstancesWithParentTransform) {
    const auto path = write_scene(nlohmann::json::parse(R"([
        {"translation":[0,2,0],"children":[1,2]},
        {"mesh":0,"translation":[3,0,0]},
        {"mesh":0,"translation":[-3,0,0],"scale":[0.5,0.5,0.5]}])"));
    for (const bool geometry_only : {false, true}) {
        SCOPED_TRACE(geometry_only);
        lfs::io::LoadOptions options;
        options.mesh_geometry_only = geometry_only;
        const auto result = loader_.load(path, options);
        ASSERT_TRUE(result.has_value());
        const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
        ASSERT_EQ(mesh->vertex_count(), 6);
        ASSERT_EQ(mesh->face_count(), 2);
        auto vertices = mesh->vertices.accessor<float, 2>();
        const std::array<std::array<float, 3>, 6> expected{{{3, 2, 0}, {4, 2, 1}, {3, 3, 1}, {-3, 2, 0}, {-2.5, 2, 0.5}, {-3, 2.5, 0.5}}};
        for (int i = 0; i < 6; ++i)
            for (int axis = 0; axis < 3; ++axis)
                EXPECT_FLOAT_EQ(vertices(i, axis), expected[i][axis]);
        auto indices = mesh->indices.accessor<int32_t, 2>();
        for (int face = 0; face < 2; ++face)
            for (int corner = 0; corner < 3; ++corner)
                EXPECT_EQ(indices(face, corner), face * 3 + corner);
    }
}

TEST_F(MeshLoaderTest, ImportsEveryCoincidentInstance) {
    const auto path = write_scene(nlohmann::json::parse(R"([
        {"children":[1,2]},{"mesh":0},{"mesh":0}])"));
    const auto result = loader_.load(path);
    ASSERT_TRUE(result.has_value());
    const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
    EXPECT_EQ(mesh->vertex_count(), 6);
    EXPECT_EQ(mesh->face_count(), 2);
    ASSERT_EQ(mesh->submeshes.size(), 2);
    EXPECT_EQ(mesh->submeshes[0].material_index, mesh->submeshes[1].material_index);
}

TEST_F(MeshLoaderTest, ImportsMatrixRotationScaleAndReflectionWithShadingFrame) {
    for (const float scale_x : {2.0f, -2.0f}) {
        SCOPED_TRACE(scale_x);
        const auto identity_path = write_scene(nlohmann::json::parse(R"([{"mesh":0}])"));
        const auto identity_result = loader_.load(identity_path);
        ASSERT_TRUE(identity_result.has_value());
        const auto identity = std::get<std::shared_ptr<MeshData>>(identity_result->data);
        auto nodes = nlohmann::json::parse(R"([
            {"matrix":[0,1,0,0,-1,0,0,0,0,0,1,0,3,4,5,1],"children":[1]},
            {"mesh":0,"scale":[2,3,4]}])");
        nodes[1]["scale"][0] = scale_x;
        const auto result = loader_.load(write_scene(nodes));
        ASSERT_TRUE(result.has_value());
        const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
        ASSERT_EQ(mesh->vertex_count(), 3);
        ASSERT_TRUE(mesh->has_normals());
        ASSERT_TRUE(mesh->has_tangents());
        auto vertices = mesh->vertices.accessor<float, 2>();
        auto normals = mesh->normals.accessor<float, 2>();
        auto tangents = mesh->tangents.accessor<float, 2>();
        auto old_vertices = identity->vertices.accessor<float, 2>();
        auto old_normals = identity->normals.accessor<float, 2>();
        auto old_tangents = identity->tangents.accessor<float, 2>();
        for (int i = 0; i < 3; ++i) {
            const glm::vec3 expected_position(3 - 3 * old_vertices(i, 1),
                                              4 + scale_x * old_vertices(i, 0), 5 + 4 * old_vertices(i, 2));
            const auto expected_normal = glm::normalize(glm::vec3(-old_normals(i, 1) / 3,
                                                                  old_normals(i, 0) / scale_x, old_normals(i, 2) / 4));
            const auto expected_tangent = glm::normalize(glm::vec3(-3 * old_tangents(i, 1),
                                                                   scale_x * old_tangents(i, 0), 4 * old_tangents(i, 2)));
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_NEAR(vertices(i, axis), expected_position[axis], 1e-5f);
                EXPECT_NEAR(normals(i, axis), expected_normal[axis], 1e-5f);
                EXPECT_NEAR(tangents(i, axis), expected_tangent[axis], 1e-5f);
            }
            EXPECT_FLOAT_EQ(tangents(i, 3), old_tangents(i, 3) * (scale_x < 0 ? -1 : 1));
        }
        auto indices = mesh->indices.accessor<int32_t, 2>();
        const glm::vec3 a(vertices(indices(0, 0), 0), vertices(indices(0, 0), 1), vertices(indices(0, 0), 2));
        const glm::vec3 b(vertices(indices(0, 1), 0), vertices(indices(0, 1), 1), vertices(indices(0, 1), 2));
        const glm::vec3 c(vertices(indices(0, 2), 0), vertices(indices(0, 2), 1), vertices(indices(0, 2), 2));
        EXPECT_GT(glm::dot(glm::cross(b - a, c - a), glm::vec3(normals(0, 0), normals(0, 1), normals(0, 2))), 0);
    }
}

TEST_F(MeshLoaderTest, PlanarZeroScalePreservesShadingFrame) {
    // The rotated case exercises off-diagonal cofactors as well as a collapsed axis.
    for (const bool rotated : {false, true}) {
        SCOPED_TRACE(rotated);
        auto nodes = nlohmann::json::parse(R"([{"mesh":0,"scale":[1,1,0]}])");
        if (rotated)
            nodes[0] = nlohmann::json::parse(R"({"mesh":0,"matrix":[2,0,0,0,0,0,3,0,0,0,0,0,0,0,0,1]})");
        const auto result = loader_.load(write_planar_scene(nodes));
        ASSERT_TRUE(result.has_value());
        const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
        ASSERT_EQ(mesh->vertex_count(), 3);
        ASSERT_EQ(mesh->face_count(), 1);
        ASSERT_TRUE(mesh->has_normals());
        ASSERT_TRUE(mesh->has_tangents());
        auto vertices = mesh->vertices.accessor<float, 2>();
        auto normals = mesh->normals.accessor<float, 2>();
        auto tangents = mesh->tangents.accessor<float, 2>();
        for (int i = 0; i < 3; ++i) {
            const glm::vec3 expected_position(i == 1 ? (rotated ? 2 : 1) : 0, !rotated && i == 2 ? 1 : 0,
                                              rotated && i == 2 ? 3 : 0);
            const glm::vec3 expected_normal = rotated ? glm::vec3(0, -1, 0) : glm::vec3(0, 0, 1);
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_FLOAT_EQ(vertices(i, axis), expected_position[axis]);
                EXPECT_NEAR(normals(i, axis), expected_normal[axis], 1e-6f);
                EXPECT_NEAR(tangents(i, axis), axis == 0 ? 1.0f : 0.0f, 1e-6f);
            }
        }
    }
}

TEST_F(MeshLoaderTest, SingularNormalFallsBackToTransformedFace) {
    const auto nodes = nlohmann::json::parse(R"([{"mesh":0,"scale":[1,1,0]}])");
    const auto result = loader_.load(write_planar_scene(nodes, {1, 0, 0}));
    ASSERT_TRUE(result.has_value());
    const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
    ASSERT_TRUE(mesh->has_normals());
    auto normals = mesh->normals.accessor<float, 2>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(normals(i, 0), 0);
        EXPECT_FLOAT_EQ(normals(i, 1), 0);
        EXPECT_FLOAT_EQ(normals(i, 2), 1);
    }
}

TEST_F(MeshLoaderTest, DegenerateZeroScaleKeepsOriginalShadingDirections) {
    const auto cases = nlohmann::json::parse(R"([
        [{"mesh":0,"scale":[0,0,0]}],
        [{"mesh":0,"scale":[0,1,0]}],
        [{"mesh":0,"matrix":[1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]}]
    ])");
    for (const auto& nodes : cases) {
        SCOPED_TRACE(nodes.dump());
        const auto result = loader_.load(write_planar_scene(nodes));
        ASSERT_TRUE(result.has_value());
        const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
        ASSERT_TRUE(mesh->has_normals());
        ASSERT_TRUE(mesh->has_tangents());
        auto normals = mesh->normals.accessor<float, 2>();
        auto tangents = mesh->tangents.accessor<float, 2>();
        for (int i = 0; i < 3; ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_FLOAT_EQ(normals(i, axis), axis == 2 ? 1 : 0);
                EXPECT_FLOAT_EQ(tangents(i, axis), axis == 0 ? 1 : 0);
            }
            EXPECT_FLOAT_EQ(tangents(i, 3), -1);
        }
    }
}

TEST_F(MeshLoaderTest, IdentityImportPreservesAttributesAndMaterialRanges) {
    const auto result = loader_.load(write_scene(nlohmann::json::parse(R"([{"mesh":0}])"), true));
    ASSERT_TRUE(result.has_value());
    const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
    ASSERT_EQ(mesh->vertex_count(), 6);
    ASSERT_EQ(mesh->face_count(), 2);
    ASSERT_TRUE(mesh->has_normals());
    ASSERT_TRUE(mesh->has_tangents());
    ASSERT_TRUE(mesh->has_texcoords());
    ASSERT_TRUE(mesh->has_colors());
    ASSERT_EQ(mesh->submeshes.size(), 2);
    for (int part = 0; part < 2; ++part) {
        const auto& submesh = mesh->submeshes[part];
        EXPECT_EQ(submesh.start_index, part * 3);
        EXPECT_EQ(submesh.index_count, 3);
        ASSERT_LT(submesh.material_index, mesh->materials.size());
        EXPECT_EQ(mesh->materials[submesh.material_index].name, part == 0 ? "red" : "blue");
        auto vertices = mesh->vertices.accessor<float, 2>();
        auto indices = mesh->indices.accessor<int32_t, 2>();
        auto colors = mesh->colors.accessor<float, 2>();
        auto uv = mesh->texcoords.accessor<float, 2>();
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(indices(part, i), part * 3 + i);
            EXPECT_FLOAT_EQ(vertices(part * 3 + i, 0), i == 1 ? 1 : 0);
            EXPECT_FLOAT_EQ(vertices(part * 3 + i, 1), i == 2 ? 1 : 0);
            EXPECT_FLOAT_EQ(vertices(part * 3 + i, 2), i == 0 ? 0 : 1);
            EXPECT_FLOAT_EQ(uv(part * 3 + i, 0), i == 1 ? 1 : 0);
            EXPECT_FLOAT_EQ(uv(part * 3 + i, 1), i == 2 ? 1 : 0);
            for (int channel = 0; channel < 4; ++channel)
                EXPECT_FLOAT_EQ(colors(part * 3 + i, channel), channel == i || channel == 3 ? 1 : 0);
        }
    }
}

TEST_F(MeshLoaderTest, AppliesSmallNodeTranslations) {
    const auto result = loader_.load(write_scene(nlohmann::json::parse(R"([{"mesh":0,"translation":[0.0001,0,0]}])")));
    ASSERT_TRUE(result.has_value());
    const auto mesh = std::get<std::shared_ptr<MeshData>>(result->data);
    auto vertices = mesh->vertices.accessor<float, 2>();
    EXPECT_NEAR(vertices(0, 0), 0.0001f, 1e-8f);
}
