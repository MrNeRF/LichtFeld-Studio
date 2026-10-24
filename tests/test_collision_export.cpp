/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/uuid.hpp"
#include "io/collision_export.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {
    using namespace lfs;
    struct CollisionExport : testing::Test {
        std::filesystem::path dir = std::filesystem::temp_directory_path() / ("lfs-collision-export-" + core::generate_uuid_v4().to_string());
        void SetUp() override { std::filesystem::create_directories(dir); }
        void TearDown() override { std::filesystem::remove_all(dir); }
        core::CollisionVolume volume(glm::ivec3 dimensions = {13, 7, 9}) {
            core::CollisionVolume v;
            v.origin = {-1.25f, 2.5f, -.75f};
            v.dimensions = dimensions;
            v.settings.voxel_size = .125f;
            v.solid.resize(size_t(dimensions.x) * dimensions.y * dimensions.z);
            for (int z = 0; z < dimensions.z; ++z)
                for (int y = 0; y < dimensions.y; ++y)
                    for (int x = 0; x < dimensions.x; ++x)
                        v.solid[v.index({x, y, z})] = ((x + y * 3 + z * 7) % 5 == 0) || (x < 4 && y < 4 && z < 4);
            v.tree = v.octreeOccupancy();
            v.tree_size = 1;
            while (v.tree_size < std::max({dimensions.x, dimensions.y, dimensions.z}))
                v.tree_size *= 2;
            v.solid.clear();
            return v;
        }
        uint32_t readWord(std::istream& in) {
            unsigned char b[4];
            in.read(reinterpret_cast<char*>(b), 4);
            if (!in)
                throw std::runtime_error("truncated export");
            return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
        }
        void roundtrip(const core::CollisionVolume& v, const glm::mat4& transform) {
            const auto path = dir / "scene.voxel.json";
            io::exportCollisionVoxels(v, transform, path);
            nlohmann::json meta;
            std::ifstream(path) >> meta;
            EXPECT_EQ(meta["version"], "1.1");
            EXPECT_EQ(meta["leafSize"], 4);
            std::ifstream in(dir / "scene.voxel.bin", std::ios::binary);
            std::vector<uint32_t> nodes(meta["nodeCount"].get<size_t>()), leaves(meta["leafDataCount"].get<size_t>());
            for (auto& w : nodes)
                w = readWord(in);
            for (auto& w : leaves)
                w = readWord(in);
            EXPECT_EQ(std::filesystem::file_size(dir / "scene.voxel.bin"), (nodes.size() + leaves.size()) * 4);
            glm::vec3 origin;
            for (int a = 0; a < 3; ++a)
                origin[a] = meta["gridBounds"]["min"][a];
            const float cell = meta["voxelResolution"];
            const int edge = 4 << meta["treeDepth"].get<int>();
            uint64_t count = 0;
            auto pc = glm::mat4(1);
            pc[0][0] = pc[1][1] = -1;
            const auto inverse = glm::inverse(pc * transform);
            for (int z = 0; z < edge; ++z)
                for (int y = 0; y < edge; ++y)
                    for (int x = 0; x < edge; ++x) {
                        glm::ivec3 p(x, y, z);
                        uint32_t index = 0;
                        int n = edge;
                        bool occupied = false;
                        if (!nodes.empty())
                            for (;;) {
                                const auto w = nodes.at(index);
                                if (w == 0xff000000u) {
                                    occupied = true;
                                    break;
                                }
                                if ((w >> 24) == 0) {
                                    const auto bit = p.x + 4 * p.y + 16 * p.z;
                                    occupied = (leaves.at(2 * w + bit / 32) >> (bit % 32)) & 1;
                                    break;
                                }
                                n /= 2;
                                auto b = glm::ivec3(glm::greaterThanEqual(p, glm::ivec3(n)));
                                const int oct = b.x + 2 * b.y + 4 * b.z;
                                const auto mask = w >> 24;
                                if (!(mask & (1u << oct)))
                                    break;
                                index = (w & 0xffffff) + std::popcount(mask & ((1u << oct) - 1));
                                p -= b * n;
                            }
                        auto world = origin + (glm::vec3(x, y, z) + .5f) * cell;
                        const auto source = glm::vec3(inverse * glm::vec4(world, 1));
                        ASSERT_EQ(occupied, v.occupied(v.cell(source))) << x << ',' << y << ',' << z;
                        count += occupied;
                    }
            EXPECT_EQ(count, v.occupiedCount());
        }
    };
} // namespace
TEST_F(CollisionExport, SuperSplatMasksBoundsAndChildOrderPreserveTransformedCells) {
    const auto v = volume();
    // Signed axis permutations include the PLY import flip, all grid rotations,
    // negative scale, translation and uniform scale without resampling.
    std::array<int, 3> axes{0, 1, 2};
    do {
        for (int signs = 0; signs < 8; ++signs) {
            glm::mat4 transform(0);
            transform[3] = {3, -4, 5, 1};
            for (int a = 0; a < 3; ++a)
                transform[a][axes[a]] = (signs & (1 << a)) ? -2.5f : 2.5f;
            roundtrip(v, transform);
        }
    } while (std::next_permutation(axes.begin(), axes.end()));
}
TEST_F(CollisionExport, TinySolidAndEmptyVolumesAreValid) {
    for (int edge : {1, 2, 4, 8})
        roundtrip(volume(glm::ivec3(edge)), glm::mat4(1));
    auto v = volume(glm::ivec3(8));
    v.tree = {1};
    roundtrip(v, glm::mat4(1));
    v.tree = {0};
    roundtrip(v, glm::mat4(1));
}
TEST_F(CollisionExport, NonrepresentableVoxelTransformsFailWithoutWriting) {
    auto v = volume();
    EXPECT_THROW(io::exportCollisionVoxels(v, glm::rotate(glm::mat4(1), .2f, glm::vec3(0, 1, 0)), dir / "scene.voxel.json"), std::runtime_error);
    EXPECT_THROW(io::exportCollisionVoxels(v, glm::scale(glm::mat4(1), glm::vec3(1, 2, 1)), dir / "scene.voxel.json"), std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_empty(dir));
}
TEST_F(CollisionExport, CancellationKeepsPriorFilesAndCleansTemporaryPayloads) {
    auto v = volume();
    {
        std::ofstream(dir / "scene.voxel.json") << "old json";
        std::ofstream(dir / "scene.voxel.bin") << "old bin";
    }
    std::stop_source stop;
    EXPECT_THROW(io::exportCollisionVoxels(v, glm::mat4(1), dir / "scene.voxel.json", stop.get_token(), [&](float value, const auto&) {
                     if (value >= .8f)
                         stop.request_stop();
                 }),
                 std::runtime_error);
    std::string content;
    std::getline(std::ifstream(dir / "scene.voxel.json"), content);
    EXPECT_EQ(content, "old json");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(dir), std::filesystem::directory_iterator{}), 2);
}
TEST_F(CollisionExport, ProgressIsMonotonicAndGlbCancellationPreservesPriorOutput) {
    const auto v = volume();
    const auto surface = v.boundsMesh();
    float previous = 0;
    auto report = [&](float value, const auto& stage) {
        EXPECT_GE(value, previous);
        EXPECT_LE(value, 1);
        EXPECT_FALSE(stage.empty());
        previous = value;
    };
    io::exportCollisionVoxels(v, glm::mat4(1), dir / "scene.voxel.json", {}, report);
    EXPECT_FLOAT_EQ(previous, 1);
    previous = 0;
    io::exportCollisionGlb(surface.vertices, surface.indices, glm::mat4(1), dir / "scene.glb", {}, report);
    EXPECT_FLOAT_EQ(previous, 1);
    std::ofstream(dir / "scene.glb") << "prior surface";
    std::stop_source stop;
    EXPECT_THROW(io::exportCollisionGlb(surface.vertices, surface.indices, glm::mat4(1), dir / "scene.glb", stop.get_token(), [&](float value, auto) {
                     if (value >= .5f)
                         stop.request_stop();
                 }),
                 std::runtime_error);
    std::string prior;
    std::getline(std::ifstream(dir / "scene.glb"), prior);
    EXPECT_EQ(prior, "prior surface");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(dir), std::filesystem::directory_iterator{}), 3);
}
TEST_F(CollisionExport, OverwritePublishesBothFilesAndInvalidDestinationKeepsThePriorPair) {
    auto v = volume();
    roundtrip(v, glm::mat4(1));
    roundtrip(volume({2, 2, 2}), glm::mat4(1));
    const auto bytes = std::filesystem::file_size(dir / "scene.voxel.bin");
    std::filesystem::remove(dir / "scene.voxel.json");
    std::filesystem::create_directory(dir / "scene.voxel.json");
    EXPECT_THROW(io::exportCollisionVoxels(v, glm::mat4(1), dir / "scene.voxel.json"), std::runtime_error);
    EXPECT_EQ(std::filesystem::file_size(dir / "scene.voxel.bin"), bytes);
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(dir), std::filesystem::directory_iterator{}), 2);
}
TEST_F(CollisionExport, GlbCarriesBakedWorldCoordinatesAndCorrectWinding) {
    core::CollisionMesh m{{0, 0, 0, 1, 0, 0, 0, 1, 0}, {0, 1, 2}};
    auto transform = glm::translate(glm::mat4(1), glm::vec3(4, 5, 6));
    transform = glm::rotate(transform, .4f, glm::vec3(0, 1, 0));
    transform = glm::scale(transform, glm::vec3(-2, 3, 4));
    io::exportCollisionGlb(m.vertices, m.indices, transform, dir / "scene.glb");
    std::ifstream in(dir / "scene.glb", std::ios::binary);
    EXPECT_EQ(readWord(in), 0x46546c67u);
    EXPECT_EQ(readWord(in), 2u);
    EXPECT_EQ(readWord(in), std::filesystem::file_size(dir / "scene.glb"));
    const auto size = readWord(in);
    EXPECT_EQ(readWord(in), 0x4e4f534au);
    std::string json(size, ' ');
    in.read(json.data(), size);
    auto meta = nlohmann::json::parse(json);
    EXPECT_EQ(meta["accessors"][0]["count"], 3);
    EXPECT_EQ(readWord(in), 48u);
    EXPECT_EQ(readWord(in), 0x004e4942u);
    glm::mat4 pc(1);
    pc[0][0] = pc[1][1] = -1;
    for (int i = 0; i < 3; ++i) {
        auto p = pc * transform * glm::vec4(m.vertices[i * 3], m.vertices[i * 3 + 1], m.vertices[i * 3 + 2], 1);
        for (int k = 0; k < 3; ++k)
            EXPECT_FLOAT_EQ(std::bit_cast<float>(readWord(in)), p[k]);
    }
    EXPECT_EQ(readWord(in), 0u);
    EXPECT_EQ(readWord(in), 2u);
    EXPECT_EQ(readWord(in), 1u);
    EXPECT_THROW(io::exportCollisionGlb(m.vertices, std::array<int32_t, 3>{0, 1, 99}, transform, dir / "invalid.glb"), std::runtime_error);
}
