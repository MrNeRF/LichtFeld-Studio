/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../src/core/collision_sparse.hpp"
#include "core/collision.hpp"
#include "core/collision_serialization.hpp"
#include <algorithm>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <map>
#include <random>

using namespace lfs::core;
namespace {
    CollisionVolume grid(glm::ivec3 dimensions = {12, 12, 12}, float size = 0.1f) {
        CollisionVolume g;
        g.dimensions = dimensions;
        g.settings.applyPreset(CollisionPreset::Object);
        g.settings.voxel_size = size;
        g.solid.resize(size_t(dimensions.x) * dimensions.y * dimensions.z);
        return g;
    }
    void box(CollisionVolume& g, glm::ivec3 lo, glm::ivec3 hi, bool shell = false) {
        for (int z = lo.z; z <= hi.z; ++z)
            for (int y = lo.y; y <= hi.y; ++y)
                for (int x = lo.x; x <= hi.x; ++x)
                    if (!shell || x == lo.x || y == lo.y || z == lo.z || x == hi.x || y == hi.y || z == hi.z)
                        g.solid[g.index({x, y, z})] = 1;
    }
    size_t count(const CollisionVolume& g) { return g.occupiedCount(); }
    CollisionVolume sparse(CollisionVolume g) {
        g.tree = g.octreeOccupancy();
        g.tree_size = 1;
        while (g.tree_size < std::max({g.dimensions.x, g.dimensions.y, g.dimensions.z}))
            g.tree_size *= 2;
        g.solid.clear();
        g.validate();
        return g;
    }
    void sameOccupancy(const CollisionVolume& a, const CollisionVolume& b) {
        ASSERT_EQ(a.dimensions, b.dimensions);
        ASSERT_EQ(a.occupiedCount(), b.occupiedCount());
        for (int z = 0; z < a.dimensions.z; ++z)
            for (int y = 0; y < a.dimensions.y; ++y)
                for (int x = 0; x < a.dimensions.x; ++x)
                    ASSERT_EQ(a.occupied({x, y, z}), b.occupied({x, y, z})) << x << "," << y << "," << z;
    }
    void watertight(const CollisionMesh& mesh) {
        ASSERT_FALSE(mesh.indices.empty());
        std::map<std::pair<int, int>, std::pair<int, int>> edges;
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            for (int a = 0; a < 3; ++a) {
                int u = mesh.indices[i + a], v = mesh.indices[i + (a + 1) % 3];
                ASSERT_NE(u, v);
                auto& edge = edges[std::minmax(u, v)];
                ++edge.first;
                edge.second += u < v ? 1 : -1;
            }
            auto p = [&](int a) {
                auto index = mesh.indices[i + a] * 3;
                return glm::vec3(mesh.vertices[index], mesh.vertices[index + 1], mesh.vertices[index + 2]);
            };
            EXPECT_GT(glm::length(glm::cross(p(1) - p(0), p(2) - p(0))), 1e-10f);
        }
        for (const auto& [edge, counts] : edges) {
            EXPECT_EQ(counts.first, 2) << edge.first << "," << edge.second;
            EXPECT_EQ(counts.second, 0) << edge.first << "," << edge.second;
        }
    }
    CapsuleQuery query(const CollisionVolume& g) {
        return [&g](glm::vec3 feet, glm::vec3 up, float height, float radius) {
            return g.capsuleIntersects(feet, up, height, radius);
        };
    }
} // namespace
TEST(CollisionSettings, PresetsDomainsAndBounds) {
    CollisionSettings s;
    s.applyPreset(CollisionPreset::Indoor);
    EXPECT_TRUE(s.external_fill);
    EXPECT_FALSE(s.floor_fill);
    EXPECT_EQ(s.carve_height, 1.6f);
    s.applyPreset(CollisionPreset::Outdoor);
    EXPECT_FALSE(s.external_fill);
    EXPECT_FALSE(s.floor_fill);
    EXPECT_EQ(s.carve_height, 0);
    s.opacity = 0;
    s.cluster_opacity = 1;
    s.cluster_min_contribution = 0;
    EXPECT_NO_THROW(s.validate());
    s.voxel_size = 0;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    auto g = grid();
    g.dimensions = {16'000'000, 2, 1};
    EXPECT_THROW(g.validate(), std::runtime_error);
}
TEST(CollisionVolume, PreviewPackingPreservesEveryCellAndClearsPadding) {
    auto g = grid({7, 5, 3});
    for (size_t i = 0; i < g.solid.size(); ++i)
        g.solid[i] = i % 3 == 0;
    g.solid[31] = g.solid[32] = g.solid.back() = 1;
    const auto packed = g.packedOccupancy();
    ASSERT_EQ(packed.size(), 4u);
    for (size_t i = 0; i < packed.size() * 32; ++i)
        EXPECT_EQ(bool(packed[i / 32] & (uint32_t(1) << (i % 32))),
                  i < g.solid.size() && bool(g.solid[i]))
            << i;
    g.solid.pop_back();
    EXPECT_THROW((void)g.packedOccupancy(), std::invalid_argument);
}
TEST(CollisionVoxelization, RotationScaleAndCombinedOpacity) {
    CollisionSettings s;
    s.applyPreset(CollisionPreset::Object);
    s.voxel_size = 0.05f;
    s.opacity = 0.5f;
    CollisionGaussian a{{0, 0, 0}, glm::mat3(0.2f), 0.4f};
    EXPECT_THROW((void)generateCollision(std::span(&a, 1), s), std::runtime_error);
    // Density adds: two 0.4 contributions yield 1-exp(-0.8), above 0.5.
    std::array both{a, a};
    auto combined = generateCollision(both, s);
    EXPECT_GT(count(combined), 0u);
    a.opacity = 1;
    a.axes = glm::mat3(1);
    a.axes[0] *= 0.35f;
    a.axes[1] *= 0.03f;
    a.axes[2] *= 0.03f;
    s.opacity = 0.1f;
    auto horizontal = generateCollision(std::span(&a, 1), s);
    EXPECT_TRUE(horizontal.occupied(horizontal.cell({0.4f, 0, 0})));
    EXPECT_FALSE(horizontal.occupied(horizontal.cell({0, 0.4f, 0})));
    a.axes = glm::mat3(glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 0, 1))) * a.axes;
    auto vertical = generateCollision(std::span(&a, 1), s);
    EXPECT_FALSE(vertical.occupied(vertical.cell({0.4f, 0, 0})));
    EXPECT_TRUE(vertical.occupied(vertical.cell({0, 0.4f, 0})));
}
TEST(CollisionPipeline, ClusterUsesNearestOccupiedConnectedComponent) {
    CollisionSettings s;
    s.applyPreset(CollisionPreset::Object);
    s.cluster = true;
    s.cluster_resolution = 0.2f;
    s.cluster_opacity = 0.1f;
    s.voxel_size = 0.1f;
    s.seed = {0, 0, 0};
    std::array gaussians{CollisionGaussian{{1, 0, 0}, glm::mat3(0.3f), 1},
                         CollisionGaussian{{5, 0, 0}, glm::mat3(0.3f), 1}};
    auto g = generateCollision(gaussians, s);
    EXPECT_TRUE(g.occupied(g.cell({1, 0, 0})));
    EXPECT_FALSE(g.occupied(g.cell({5, 0, 0})));
}
TEST(CollisionPipeline, ExteriorLeakReturnsOriginalAndClosedRoomFillsOutside) {
    auto g = grid({20, 20, 20});
    box(g, {3, 3, 3}, {16, 16, 16}, true);
    g.settings.external_fill = true;
    g.settings.external_size = 0;
    g.settings.seed = {0.95f, 0.95f, 0.95f};
    auto filled = processCollisionVolume(g);
    EXPECT_TRUE(filled.occupied({0, 0, 0}));
    EXPECT_FALSE(filled.occupied({9, 9, 9}));
    g.solid[g.index({3, 9, 9})] = 0;
    auto leaked = processCollisionVolume(g);
    EXPECT_EQ(leaked.solid, g.solid);
    EXPECT_FALSE(leaked.warning.empty());
}
TEST(CollisionPipeline, FloorPatchesOnlySurroundedColumnsWithinReach) {
    auto g = grid({15, 10, 15});
    box(g, {3, 5, 3}, {11, 5, 11});
    g.solid[g.index({7, 5, 7})] = 0;
    g.settings.floor_fill = true;
    g.settings.floor_size = 0.1f;
    auto filled = processCollisionVolume(g);
    EXPECT_TRUE(filled.occupied({7, 0, 7}));
    EXPECT_FALSE(filled.occupied({0, 0, 0}));
    g.settings.floor_size = 0;
    auto no_patch = processCollisionVolume(g);
    EXPECT_FALSE(no_patch.occupied({7, 0, 7}));
}
TEST(CollisionPipeline, CapsuleCarveRejectsInvalidSeedAndPreservesSolidShell) {
    auto g = grid({40, 40, 40});
    box(g, {2, 2, 2}, {37, 37, 37}, true);
    g.settings.seed = {2, 2, 2};
    g.settings.carve_height = 1.6f;
    g.settings.carve_radius = 0.2f;
    auto carved = processCollisionVolume(g);
    EXPECT_FALSE(carved.occupied(carved.cell({2, 2, 2})));
    EXPECT_TRUE(carved.occupied({2, 20, 20}));
    EXPECT_TRUE(carved.occupied({0, 0, 0}));
    g.settings.seed = {0.25f, 0.25f, 0.25f};
    EXPECT_THROW((void)processCollisionVolume(g), std::runtime_error);
}
TEST(CollisionPipeline, CancellationIsExplicit) {
    auto g = grid();
    box(g, {3, 3, 3}, {8, 8, 8});
    std::stop_source source;
    source.request_stop();
    EXPECT_THROW((void)processCollisionVolume(g, source.get_token()), std::runtime_error);
    EXPECT_THROW((void)g.mesh(CollisionShape::Faces, source.get_token()), std::runtime_error);
}
TEST(CollisionSparse, DistantGaussianComponentsDoNotAllocateEmptyWorldBounds) {
    CollisionSettings s;
    s.applyPreset(CollisionPreset::Object);
    s.voxel_size = 0.05f;
    const std::array source{CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 1},
                            CollisionGaussian{{40'000, 40'000, 40'000}, glm::mat3(0.1f), 1}};
    auto g = generateCollision(source, s);
    EXPECT_TRUE(g.solid.empty());
    EXPECT_GT(uint64_t(g.dimensions.x) * g.dimensions.y * g.dimensions.z, 1'000'000'000'000ULL);
    EXPECT_LT(g.tree.size(), 5'000u);
    EXPECT_TRUE(g.occupied(g.cell(source.front().center)));
    EXPECT_TRUE(g.occupied(g.cell(source.back().center)));
    EXPECT_FALSE(g.occupied(g.cell({20'000, 20'000, 20'000})));
    EXPECT_TRUE(g.capsuleIntersects({39'999.9f, 39'999.9f, 40'000}, {0, 1, 0}, 0.2f, 0.05f));
    watertight(g.mesh(CollisionShape::Faces));
    watertight(g.mesh(CollisionShape::Smooth));
    const auto restored = nlohmann::json(g).get<CollisionVolume>();
    EXPECT_EQ(restored, g);
    EXPECT_LT(nlohmann::json(g).dump().size(), 10'000u);
}
TEST(CollisionSparse, AccumulatedOpacityMatchesDirectCellEvaluation) {
    CollisionSettings s;
    s.applyPreset(CollisionPreset::Object);
    s.voxel_size = 0.1f;
    s.opacity = 0.3f;
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> position(-1, 1), scale(0.02f, 0.5f), alpha(0.005f, 0.8f);
    std::vector<CollisionGaussian> source;
    for (int i = 0; i < 50; ++i) {
        auto axes = glm::mat3(glm::rotate(glm::mat4(1), position(rng) * 3, glm::normalize(glm::vec3(1, 2, 3))));
        for (int a = 0; a < 3; ++a)
            axes[a] *= scale(rng);
        source.push_back({{position(rng), position(rng), position(rng)}, axes, alpha(rng)});
    }
    auto g = generateCollision(source, s);
    for (int z = 0; z < g.dimensions.z; ++z)
        for (int y = 0; y < g.dimensions.y; ++y)
            for (int x = 0; x < g.dimensions.x; ++x) {
                glm::ivec3 p(x, y, z);
                float a = 0;
                for (const auto& gaussian : source) {
                    const auto center = g.origin + (glm::vec3(p) + 0.5f) * s.voxel_size;
                    glm::vec3 extent;
                    for (int axis = 0; axis < 3; ++axis)
                        extent[axis] = 3 * glm::length(glm::vec3(gaussian.axes[0][axis], gaussian.axes[1][axis], gaussian.axes[2][axis]));
                    if (glm::any(glm::greaterThan(glm::abs(center - gaussian.center), extent + s.voxel_size * 0.5f)))
                        continue;
                    const auto closest = glm::clamp(gaussian.center, center - s.voxel_size * 0.5f, center + s.voxel_size * 0.5f);
                    const auto q = glm::inverse(gaussian.axes) * (closest - gaussian.center);
                    a += gaussian.opacity * std::exp(-0.5f * glm::dot(q, q));
                }
                ASSERT_EQ(g.occupied(p), a > 0 && -std::expm1(-a) >= s.opacity) << x << "," << y << "," << z;
            }
}
TEST(CollisionSparse, WeakIsolatedAndDegenerateFarSupportsDoNotExpandBounds) {
    CollisionSettings s;
    s.applyPreset(CollisionPreset::Object);
    const std::array source{
        CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 1},
        CollisionGaussian{{1e9f, 1e9f, 1e9f}, glm::mat3(1), 0.01f},
        CollisionGaussian{{-1e9f, -1e9f, -1e9f}, glm::mat3(0), 1}};
    const auto full = generateCollision(source, s);
    const auto foreground = generateCollision(std::span(source.data(), 1), s);
    sameOccupancy(full, foreground);
    EXPECT_EQ(full.origin, foreground.origin);
    // The proof must not discard sub-threshold supports which combine.
    const std::array combined{
        CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 0.06f},
        CollisionGaussian{{0, 0, 0}, glm::mat3(0.1f), 0.06f}};
    EXPECT_GT(generateCollision(combined, s).occupiedCount(), 0u);
}
TEST(CollisionSparse, ConservativeBoundsPreserveCombinedWeakSupportsAndEmptyRegions) {
    CollisionSettings s;
    s.voxel_size = .1f;
    s.opacity = .1f;
    const std::array source{
        CollisionGaussian{{0, 0, 0}, glm::mat3(.1f), .06f},
        CollisionGaussian{{0, 0, 0}, glm::mat3(.1f), .06f},
        CollisionGaussian{{10000, 0, 0}, glm::mat3(.1f), .01f}};
    const auto volume = generateCollision(source, s);
    EXPECT_TRUE(volume.occupied(volume.cell({0, 0, 0})));
    EXPECT_FALSE(volume.occupied(volume.cell({10000, 0, 0})));
    EXPECT_FALSE(volume.occupied(volume.cell({5000, 0, 0})));
    EXPECT_LT(volume.tree.size(), 10000u);
    const auto foreground = generateCollision(std::span(source.data(), 2), s);
    EXPECT_EQ(volume.occupiedCount(), foreground.occupiedCount());
}
TEST(CollisionSparse, PostprocessingMatchesDenseFloorExteriorAndCapsuleSemantics) {
    auto floor = grid({40, 20, 40});
    box(floor, {3, 8, 3}, {36, 8, 36});
    floor.solid[floor.index({31, 8, 31})] = 0;
    floor.settings.floor_fill = true;
    floor.settings.floor_size = 0.2f;
    sameOccupancy(processCollisionVolume(floor), processCollisionVolume(sparse(floor)));
    auto room = grid({40, 40, 40});
    box(room, {2, 2, 2}, {37, 37, 37}, true);
    room.settings.seed = {2, 2, 2};
    room.settings.external_fill = true;
    room.settings.external_size = 0.2f;
    sameOccupancy(processCollisionVolume(room), processCollisionVolume(sparse(room)));
    room.settings.external_fill = false;
    room.settings.carve_height = 1.6f;
    room.settings.carve_radius = 0.2f;
    sameOccupancy(processCollisionVolume(room), processCollisionVolume(sparse(room)));
    room.settings.seed = {0.25f, 0.25f, 0.25f};
    EXPECT_THROW((void)processCollisionVolume(sparse(room)), std::runtime_error);
}
TEST(CollisionSparse, ExternalLeakAndCancelledProcessingNeverProduceSuccess) {
    auto room = grid({40, 40, 40});
    box(room, {2, 2, 2}, {37, 37, 37}, true);
    room.solid[room.index({2, 20, 20})] = 0;
    room.settings.seed = {2, 2, 2};
    room.settings.external_fill = true;
    room.settings.external_size = 0;
    auto input = sparse(room);
    auto result = processCollisionVolume(input);
    sameOccupancy(input, result);
    EXPECT_FALSE(result.warning.empty());
    std::stop_source stop;
    EXPECT_THROW((void)processCollisionVolume(input, stop.get_token(), [&](auto) { stop.request_stop(); }), std::runtime_error);
}
TEST(CollisionSparse, BatchedFloorPreservesUnevenHeightsHolesAndPadding) {
    auto input = grid({67, 43, 73});
    for (int z = 1; z < input.dimensions.z - 1; ++z)
        for (int x = 1; x < input.dimensions.x - 1; ++x) {
            if ((x / 4 + z / 5) % 3 == 0)
                continue;
            const int y = (x * 7 + z * 11) % input.dimensions.y;
            input.solid[input.index({x, y, z})] = 1;
        }
    input.settings.floor_fill = true;
    input.settings.floor_size = 0.3f;
    auto result = processCollisionVolume(sparse(input));
    result.validate();
    sameOccupancy(processCollisionVolume(input), result);
    sameOccupancy(result, nlohmann::json(result).get<CollisionVolume>());
    std::stop_source stop;
    EXPECT_THROW((void)processCollisionVolume(sparse(input), stop.get_token(), [&](auto stage) {
                     if (stage == "Filling sparse ground columns")
                         stop.request_stop();
                 }),
                 std::runtime_error);
}
TEST(CollisionSparse, FacesAndSmoothRemainWatertightAcrossChunkAndBrickSeams) {
    auto g = grid({70, 12, 70});
    box(g, {1, 1, 1}, {68, 10, 68}, true);
    g = sparse(std::move(g));
    EXPECT_TRUE(g.occupied({31, 1, 31}));
    EXPECT_FALSE(g.occupied({32, 2, 32}));
    watertight(g.mesh(CollisionShape::Faces));
    watertight(g.mesh(CollisionShape::Smooth));
    uint64_t faces = 0;
    g.forEachBoundaryFace([&](auto p, auto n) {
        ASSERT_TRUE(g.occupied(p));
        ASSERT_FALSE(g.occupied(p + n));
        ++faces;
    });
    EXPECT_EQ(faces * 6, g.mesh(CollisionShape::Faces).indices.size());
}
TEST(CollisionSparse, PersistenceRejectsMalformedTopologyAndCoordinateOverflow) {
    auto g = grid();
    box(g, {3, 3, 3}, {8, 8, 8});
    auto v = sparse(g);
    auto j = nlohmann::json(v);
    EXPECT_EQ(j.at("version"), 2);
    EXPECT_EQ(j.get<CollisionVolume>(), v);
    j["occupancy_tree"] = "3";
    EXPECT_THROW((void)j.get<CollisionVolume>(), std::invalid_argument);
    j = nlohmann::json(v);
    j["tree_nodes"] = CollisionVolume::max_tree_words + 1;
    EXPECT_THROW((void)j.get<CollisionVolume>(), std::invalid_argument);
    j = nlohmann::json(v);
    j["tree_size"] = 3;
    EXPECT_THROW((void)j.get<CollisionVolume>(), std::invalid_argument);
    j = nlohmann::json(v);
    j["dimensions"] = {1 << 22, 1, 1};
    EXPECT_THROW((void)j.get<CollisionVolume>(), std::invalid_argument);
}
TEST(CollisionMesh, WatertightFacesAndSmoothWithCoplanarReduction) {
    auto g = grid({16, 16, 16});
    box(g, {2, 2, 2}, {12, 12, 12});
    auto faces = g.mesh(CollisionShape::Faces), smooth = g.mesh(CollisionShape::Smooth);
    watertight(faces);
    watertight(smooth);
    EXPECT_LT(smooth.indices.size(), faces.indices.size());
}
TEST(CollisionMesh, EveryMarchingCubeConfigurationIsWatertight) {
    for (int mask = 1; mask < 255; ++mask) {
        SCOPED_TRACE(mask);
        auto g = grid({4, 4, 4});
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x)
                    if (mask & (1 << (x + y * 2 + z * 4)))
                        g.solid[g.index({x + 1, y + 1, z + 1})] = 1;
        watertight(g.mesh(CollisionShape::Smooth));
    }
}
TEST(CollisionMesh, ElongatedGridDoesNotAliasVertexKeys) {
    auto g = grid({1'100'000, 2, 2}, 1);
    g.solid[g.index({0, 0, 0})] = 1;
    g.solid[g.index({1'099'999, 1, 1})] = 1;
    auto mesh = g.mesh(CollisionShape::Faces);
    EXPECT_EQ(mesh.vertices.size() / 3, 16u);
    watertight(mesh);
}
TEST(CollisionMesh, RandomMultiCubeVolumesPreserveWatertightTopologyAfterSimplification) {
    std::mt19937 random(74174);
    for (int trial = 0; trial < 30; ++trial) {
        SCOPED_TRACE(trial);
        auto g = grid({12, 12, 12});
        for (int z = 2; z < 10; ++z)
            for (int y = 2; y < 10; ++y)
                for (int x = 2; x < 10; ++x)
                    g.solid[g.index({x, y, z})] = random() % 3 != 0;
        watertight(g.mesh(CollisionShape::Smooth));
    }
}
TEST(CollisionPersistence, RleRoundTripAndMalformedPayload) {
    auto g = grid();
    box(g, {2, 2, 2}, {8, 8, 8}, true);
    g.show_voxels = true;
    g.settings.applyPreset(CollisionPreset::Indoor);
    nlohmann::json json = g;
    auto restored = nlohmann::json::parse(json.dump()).get<CollisionVolume>();
    EXPECT_EQ(g, restored);
    json["occupancy_rle"] = {16'000'001};
    EXPECT_THROW(json.get<CollisionVolume>(), std::invalid_argument);
}
TEST(WalkFrame, VerticalLookIsFiniteForBothUpAxes) {
    for (auto up : {glm::vec3(0, 1, 0), glm::vec3(0, 0, -1)})
        for (float sign : {-1.0f, 1.0f}) {
            auto frame = walkTangentFrame(up * sign, {1, 0, 0}, up);
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    EXPECT_TRUE(std::isfinite(frame[a][b]));
            EXPECT_NEAR(glm::dot(frame[0], up), 0, 1e-6);
            EXPECT_NEAR(glm::length(frame[2]), 1, 1e-6);
        }
}
TEST(WalkCollision, TranslatedRotatedAndScaledNode) {
    auto g = grid();
    box(g, {3, 3, 3}, {8, 8, 8});
    auto transform = glm::translate(glm::mat4(1), glm::vec3(5, 0, 0)) *
                     glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 1, 0)) *
                     glm::scale(glm::mat4(1), glm::vec3(2));
    auto point = glm::vec3(transform * glm::vec4(0.5f, 0.5f, 0.5f, 1));
    EXPECT_TRUE(transformedCapsuleIntersects(g, transform, point, {0, 1, 0}, 0.4f, 0.1f));
    EXPECT_FALSE(transformedCapsuleIntersects(g, transform, point + glm::vec3(5, 0, 0), {0, 1, 0}, 0.4f, 0.1f));
    EXPECT_THROW((void)transformedCapsuleIntersects(g, glm::mat4(0), point, {0, 1, 0}, 0.4f, 0.1f), std::invalid_argument);
}
TEST(WalkMotion, GravityGroundingWallSlidingAndFrameClamp) {
    auto g = grid({60, 40, 60});
    box(g, {0, 0, 0}, {59, 2, 59});
    box(g, {35, 3, 0}, {36, 35, 59});
    WalkSettings s;
    s.height = 1.8f;
    s.radius = 0.2f;
    WalkState state{.eye = {2, 3, 2}};
    for (int i = 0; i < 180; ++i)
        advanceWalk(state, s, {0, 1, 0}, {0, 0, 0}, 1.0f / 60, query(g));
    EXPECT_TRUE(state.grounded);
    EXPECT_NEAR(state.eye.y, 2.1f, 0.025f);
    for (int i = 0; i < 120; ++i)
        advanceWalk(state, s, {0, 1, 0}, {2, 0, 1}, 1.0f / 60, query(g));
    EXPECT_LT(state.eye.x, 3.31f);
    EXPECT_GT(state.eye.z, 3.8f);
    EXPECT_FALSE(g.capsuleIntersects(state.eye - glm::vec3(0, s.height, 0), {0, 1, 0}, s.height, s.radius));
}
TEST(WalkCollision, FortyFiveDegreeRigidRadiusAndConservativeAnisotropy) {
    auto g = grid({20, 20, 20});
    box(g, {9, 0, 0}, {9, 19, 19});
    auto rotation = glm::rotate(glm::mat4(1), glm::radians(45.0f), glm::vec3(0, 1, 0));
    auto local_feet = glm::vec3(0.68f, 0.2f, 1);
    auto world_feet = glm::vec3(rotation * glm::vec4(local_feet, 1));
    EXPECT_FALSE(transformedCapsuleIntersects(g, rotation, world_feet, {0, 1, 0}, 1.4f, 0.2f));
    local_feet.x = 0.72f;
    world_feet = glm::vec3(rotation * glm::vec4(local_feet, 1));
    EXPECT_TRUE(transformedCapsuleIntersects(g, rotation, world_feet, {0, 1, 0}, 1.4f, 0.2f));
    auto scaled = rotation * glm::scale(glm::mat4(1), glm::vec3(2, 1, 0.5f));
    auto contact = glm::vec3(scaled * glm::vec4(0.89f, 0.2f, 1, 1));
    EXPECT_TRUE(transformedCapsuleIntersects(g, scaled, contact, {0, 1, 0}, 1.4f, 0.2f));
}
TEST(WalkMotion, StepsAndThinCeilingSweep) {
    WalkSettings s;
    s.height = 1.0f;
    s.radius = 0.15f;
    s.step_height = 0.3f;
    auto g = grid({50, 30, 30}, 0.05f);
    box(g, {0, 0, 0}, {49, 1, 29});
    box(g, {20, 2, 0}, {25, 4, 29});
    WalkState state{.eye = {0.6f, 1.1f, 0.6f}, .grounded = true};
    for (int i = 0; i < 90; ++i)
        advanceWalk(state, s, {0, 1, 0}, {0.8f, 0, 0}, 1.0f / 60, query(g));
    EXPECT_GT(state.eye.x, 1.3f);
    box(g, {0, 24, 0}, {19, 24, 29});
    state = {.eye = {0.6f, 1.1f, 0.6f}, .grounded = true};
    for (int i = 0; i < 90; ++i)
        advanceWalk(state, s, {0, 1, 0}, {0.8f, 0, 0}, 1.0f / 60, query(g));
    EXPECT_LT(state.eye.x, 1.0f);
    EXPECT_LT(state.eye.y, 1.22f);
}
TEST(CollisionPerformance, IndoorFiveCentimeterCarve) {
    auto g = grid({120, 70, 120}, 0.05f);
    box(g, {2, 2, 2}, {117, 67, 117}, true);
    g.settings.carve_height = 1.6f;
    g.settings.carve_radius = 0.2f;
    g.settings.seed = {3, 1.8f, 3};
    auto start = std::chrono::steady_clock::now();
    auto carved = processCollisionVolume(std::move(g));
    auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    RecordProperty("carve_seconds", seconds);
    std::cout << "Native 6m x 3.5m x 6m / 5cm carve: " << seconds << " s\n";
    EXPECT_FALSE(carved.occupied(carved.cell({3, 1.8f, 3})));
    EXPECT_LT(seconds, 15);
}

TEST(CollisionAuthoring, CoarseSurfaceRetainsAllSupportsWithoutChangingWalkData) {
    auto g = grid({17, 13, 19});
    box(g, {1, 2, 3}, {3, 4, 5});
    g.solid[g.index({16, 12, 18})] = 1;
    const auto before = g.solid;
    const auto coarse = g.coarsened(4);
    EXPECT_EQ(coarse.dimensions, glm::ivec3(5, 4, 5));
    EXPECT_FLOAT_EQ(coarse.settings.voxel_size, g.settings.voxel_size * 4);
    for (int z = 0; z < 19; ++z)
        for (int y = 0; y < 13; ++y)
            for (int x = 0; x < 17; ++x)
                if (g.occupied({x, y, z}))
                    EXPECT_TRUE(coarse.occupied(glm::ivec3(x, y, z) / 4));
    EXPECT_EQ(g.solid, before);
    EXPECT_EQ(g.boundsMesh().vertices.size(), 24u);
    EXPECT_EQ(g.boundsMesh().indices.size(), 36u);
    EXPECT_THROW((void)g.coarsened(3), std::invalid_argument);
    std::stop_source cancelled;
    cancelled.request_stop();
    EXPECT_THROW((void)g.coarsened(4, cancelled.get_token()), std::runtime_error);
}

TEST(CollisionAuthoring, CalibratedSceneScaleKeepsMetricPresetDimensions) {
    CollisionSettings settings;
    settings.units_per_meter = 10;
    settings.applyPreset(CollisionPreset::Indoor);
    EXPECT_FLOAT_EQ(settings.carve_height, 16);
    EXPECT_FLOAT_EQ(settings.carve_radius, 2);
    EXPECT_FLOAT_EQ(settings.floor_size, 16);
    WalkSettings walk;
    walk.units_per_meter = 10;
    walk.height = 18;
    walk.radius = 2;
    walk.speed = 25;
    walk.step_height = 2.5f;
    EXPECT_NO_THROW(walk.validate());
    settings.units_per_meter = 0;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST(CollisionPipeline, GroundGuardKeepsCanopyButDoesNotExtrudeItDownward) {
    auto original = grid({20, 40, 20});
    box(original, {2, 1, 2}, {5, 1, 5});
    box(original, {12, 30, 12}, {15, 30, 15});
    box(original, {8, 12, 8}, {9, 12, 9});
    original.settings.floor_fill = true;
    original.settings.floor_size = 0;
    original.settings.floor_below_seed = true;
    original.settings.seed = {0.4f, 1.8f, 0.4f};
    auto sparse = original;
    sparse.tree = sparse.octreeOccupancy();
    sparse.tree_size = 64;
    sparse.solid.clear();
    const auto dense_result = processCollisionVolume(original);
    const auto sparse_result = processCollisionVolume(std::move(sparse));
    EXPECT_TRUE(sparse_result.occupied({3, 0, 3}));
    EXPECT_TRUE(sparse_result.occupied({13, 30, 13}));
    EXPECT_FALSE(sparse_result.occupied({13, 0, 13}));
    EXPECT_TRUE(sparse_result.occupied({8, 12, 8}));
    EXPECT_FALSE(sparse_result.occupied({8, 0, 8}));
    for (int z = 0; z < 20; ++z)
        for (int y = 0; y < 40; ++y)
            for (int x = 0; x < 20; ++x)
                EXPECT_EQ(dense_result.occupied({x, y, z}), sparse_result.occupied({x, y, z}));
}

TEST(WalkCollision, SparseCapsuleTraversalMatchesDenseForRotatedSegments) {
    auto dense = grid({17, 17, 17});
    box(dense, {2, 2, 2}, {5, 8, 5});
    box(dense, {10, 9, 10}, {13, 12, 13});
    auto sparse = dense;
    sparse.tree = sparse.octreeOccupancy();
    sparse.tree_size = 32;
    sparse.solid.clear();
    std::mt19937 generator(81741);
    std::uniform_real_distribution<float> position(-0.3f, 1.9f), direction(-1, 1);
    for (int i = 0; i < 200; ++i) {
        const glm::vec3 feet(position(generator), position(generator), position(generator));
        const auto up = glm::normalize(glm::vec3(direction(generator), direction(generator), direction(generator)));
        EXPECT_EQ(dense.capsuleIntersects(feet, up, 0.8f, 0.12f),
                  sparse.capsuleIntersects(feet, up, 0.8f, 0.12f));
    }
}

TEST(CollisionAuthoring, DisplaySettingsAndScalePersistWithLegacyDefaults) {
    CollisionDisplay display{.preview = true, .sharp = true, .surface_ready = false, .opacity = 0.3f, .slice = 0.6f, .surface_stride = 4};
    EXPECT_EQ(nlohmann::json(display).get<CollisionDisplay>(), display);
    auto legacy_display = nlohmann::json(display);
    legacy_display.erase("sharp");
    EXPECT_TRUE(legacy_display.get<CollisionDisplay>().sharp);
    CollisionSettings settings;
    settings.units_per_meter = 0.25f;
    settings.floor_below_seed = true;
    EXPECT_EQ(nlohmann::json(settings).get<CollisionSettings>(), settings);
    auto legacy = nlohmann::json(settings);
    legacy.erase("units_per_meter");
    legacy.erase("floor_below_seed");
    const auto old = legacy.get<CollisionSettings>();
    EXPECT_EQ(old.units_per_meter, 1);
    EXPECT_FALSE(old.floor_below_seed);
    auto broken = nlohmann::json(display);
    broken["slice"] = -1;
    EXPECT_THROW(broken.get<CollisionDisplay>(), std::invalid_argument);
}

TEST(CollisionWalk, AxisAlignedCapsulesRespectFaceAndCornerTangency) {
    CollisionVolume v;
    v.settings.voxel_size = 1;
    v.dimensions = {1, 1, 1};
    v.solid = {1};
    for (int axis = 0; axis < 3; ++axis) {
        glm::vec3 up(0);
        up[axis] = 1;
        glm::vec3 feet(.5f);
        feet[axis] = -2;
        EXPECT_TRUE(v.capsuleIntersects(feet, up, 4, .25f));
        const int side = (axis + 1) % 3;
        feet[side] = 1.25f;
        EXPECT_FALSE(v.capsuleIntersects(feet, up, 4, .25f));
        feet[side] = 1.249f;
        EXPECT_TRUE(v.capsuleIntersects(feet, up, 4, .25f));
        feet[(axis + 2) % 3] = 1.249f;
        EXPECT_FALSE(v.capsuleIntersects(feet, up, 4, .25f));
    }
}

TEST(CollisionVoxelization, ParallelBricksMatchSerialIncludingPartialBoundaryCells) {
    std::vector<CollisionGaussian> source;
    for (int i = 0; i < 4096; ++i)
        source.push_back({{float(i % 16) * 0.12f, float((i / 16) % 16) * 0.12f, float(i / 256) * 0.12f}, glm::mat3(0.035f), 0.8f});
    CollisionSettings settings;
    settings.applyPreset(CollisionPreset::Object);
    settings.voxel_size = 0.07f;
    const auto serial = collision_detail::voxelizeSparse(source, settings, {}, {}, 1);
    const auto parallel = collision_detail::voxelizeSparse(source, settings, {}, {}, 4);
    ASSERT_EQ(serial.dimensions, parallel.dimensions);
    ASSERT_EQ(serial.origin, parallel.origin);
    EXPECT_EQ(serial.occupiedCount(), parallel.occupiedCount());
    for (int z = 0; z < serial.dimensions.z; ++z)
        for (int y = 0; y < serial.dimensions.y; ++y)
            for (int x = 0; x < serial.dimensions.x; ++x)
                ASSERT_EQ(serial.occupied({x, y, z}), parallel.occupied({x, y, z}));
}

TEST(CollisionVoxelization, ThinSurfaceBetweenVoxelCentersRemainsCollidable) {
    CollisionSettings s;
    s.voxel_size = 0.1f;
    const CollisionGaussian sheet{{0, 0.001f, 0}, glm::mat3(0.2f, 0, 0, 0, 0.0002f, 0, 0, 0, 0.2f), 1};
    const auto v = generateCollision(std::span(&sheet, 1), s);
    EXPECT_TRUE(v.occupied(v.cell(sheet.center)));
    EXPECT_FALSE(v.occupied(v.cell({0, 0.15f, 0})));
    EXPECT_FALSE(v.occupied(v.cell({0, -0.15f, 0})));
    EXPECT_FALSE(v.settings.floor_fill);
    s.floor_fill = true;
    s.floor_size = 0;
    const auto filled = generateCollision(std::span(&sheet, 1), s);
    EXPECT_GT(filled.occupiedCount(), v.occupiedCount());
}
TEST(CollisionVoxelization, OpacityUsesExtinctionRatherThanOverComposition) {
    CollisionSettings s;
    s.opacity = 0.5f;
    const CollisionGaussian source{{0, 0, 0}, glm::mat3(.2f), .6f};
    EXPECT_THROW((void)generateCollision(std::span(&source, 1), s), std::runtime_error);
    const std::array both{source, source};
    EXPECT_GT(generateCollision(both, s).occupiedCount(), 0u);
}

TEST(CollisionVoxelization, SurfaceDefaultDoesNotExtrudeLowCanopyIntoGround) {
    CollisionSettings s;
    s.voxel_size = 0.1f;
    s.seed = {0, 3, 0};
    s.floor_below_seed = true;
    s.floor_size = 0;
    const auto axes = glm::mat3(0.2f, 0, 0, 0, 0.005f, 0, 0, 0, 0.2f);
    const std::array source{CollisionGaussian{{-1, -2, 0}, axes, 1},
                            CollisionGaussian{{1, 1, 0}, axes, 1}};
    const auto surface = generateCollision(source, s);
    EXPECT_TRUE(surface.occupied(surface.cell(source.back().center)));
    EXPECT_FALSE(surface.occupied(surface.cell({1, 0, 0})));
    s.floor_fill = true;
    const auto filled = generateCollision(source, s);
    // Even the seed-height guard cannot recognize a low canopy as non-ground.
    EXPECT_TRUE(filled.occupied(filled.cell({1, 0, 0})));
}

TEST(CollisionCluster, CenterShortcutPreservesThinLowOpacityDetail) {
    auto coarse = grid({12, 12, 12});
    box(coarse, {4, 4, 4}, {4, 4, 4});
    const std::array source{CollisionGaussian{{0.451f, 0.451f, 0.451f}, glm::mat3(0.0001f), 0.001f},
                            CollisionGaussian{{0.9f, 0.9f, 0.9f}, glm::mat3(0.01f), 1}};
    const auto selected = collision_detail::selectCluster(source, coarse, source[0].center, 0.1f);
    EXPECT_EQ(selected.indices, (std::vector<uint32_t>{0}));
    EXPECT_EQ(selected.occupied_voxels, 1u);
}
TEST(CollisionCluster, RejectsElongatedTailAndRetainsAttachedSmallSupports) {
    auto coarse = grid({80, 20, 20});
    box(coarse, {20, 5, 5}, {22, 15, 15});
    const std::array source{CollisionGaussian{{2.05f, 0.95f, 0.95f}, glm::mat3(0.001f), 0.01f},
                            CollisionGaussian{{4, 1, 1}, glm::mat3(1, 0, 0, 0, .1f, 0, 0, 0, .1f), 1},
                            CollisionGaussian{{1.999f, 0.95f, 0.95f}, glm::mat3(.07f, 0, 0, 0, .05f, 0, 0, 0, .05f), 1}};
    const auto selected = collision_detail::selectCluster(source, coarse, source[0].center, 0.1f);
    EXPECT_EQ(selected.indices, (std::vector<uint32_t>{0, 2}));
}
TEST(CollisionCluster, SixConnectedFloodDoesNotJoinCornerTouchingComponents) {
    auto coarse = grid();
    coarse.solid[coarse.index({2, 2, 2})] = 1;
    coarse.solid[coarse.index({3, 3, 3})] = 1;
    const std::array source{CollisionGaussian{{.25f, .25f, .25f}, glm::mat3(.001f), 1},
                            CollisionGaussian{{.35f, .35f, .35f}, glm::mat3(.001f), 1}};
    EXPECT_EQ(collision_detail::selectCluster(source, coarse, source[0].center, .1f).indices,
              (std::vector<uint32_t>{0}));
    // Seed conversion is bounded before converting finite far coordinates to integers.
    EXPECT_EQ(collision_detail::selectCluster(source, coarse, glm::vec3(1e30f), .1f).indices,
              (std::vector<uint32_t>{1}));
}
TEST(CollisionCluster, HierarchicalQueriesMatchDirectOccupiedCellReference) {
    auto coarse = grid({24, 16, 24}, 0.2f);
    box(coarse, {4, 4, 4}, {18, 10, 18});
    std::mt19937 rng(6172);
    std::uniform_real_distribution<float> position(-1, 6), scale(.01f, 2), opacity(0, 1);
    std::vector<CollisionGaussian> source;
    source.push_back({{2, 1.2f, 2}, glm::mat3(.01f), .001f});
    for (int i = 0; i < 250; ++i) {
        auto axes = glm::mat3(glm::rotate(glm::mat4(1), position(rng), glm::normalize(glm::vec3(1, 2, 3))));
        for (int a = 0; a < 3; ++a)
            axes[a] *= scale(rng);
        source.push_back({{position(rng), position(rng), position(rng)}, axes, opacity(rng)});
    }
    for (const float minimum : {0.f, .1f, .8f, 2.f}) {
        std::vector<uint32_t> expected;
        for (size_t i = 0; i < source.size(); ++i) {
            const auto& g = source[i];
            if (coarse.occupied(coarse.cell(g.center))) {
                expected.push_back(uint32_t(i));
                continue;
            }
            if (g.opacity < minimum)
                continue;
            glm::dvec3 extent;
            for (int a = 0; a < 3; ++a)
                extent[a] = 3 * glm::length(glm::dvec3(g.axes[0][a], g.axes[1][a], g.axes[2][a]));
            const auto min = glm::ivec3(glm::floor((glm::dvec3(g.center) - extent) / double(coarse.settings.voxel_size)));
            const auto max = glm::ivec3(glm::floor((glm::dvec3(g.center) + extent) / double(coarse.settings.voxel_size)));
            const auto voxels = (2 * extent.x / coarse.settings.voxel_size) * (2 * extent.y / coarse.settings.voxel_size) * (2 * extent.z / coarse.settings.voxel_size);
            const auto needed = std::max({extent.x, extent.y, extent.z}) > coarse.settings.voxel_size ? std::max(1.0, std::ceil(voxels * .1)) : 1.0;
            uint64_t hits = 0;
            for (int z = 0; z < coarse.dimensions.z; ++z)
                for (int y = 0; y < coarse.dimensions.y; ++y)
                    for (int x = 0; x < coarse.dimensions.x; ++x) {
                        const glm::ivec3 p(x, y, z);
                        if (!coarse.occupied(p) || glm::any(glm::lessThan(p, min)) || glm::any(glm::greaterThan(p, max)))
                            continue;
                        const auto q = glm::inverse(g.axes) * (coarse.origin + (glm::vec3(p) + .5f) * coarse.settings.voxel_size - g.center);
                        hits += g.opacity * std::exp(-.5f * glm::dot(q, q)) >= minimum;
                    }
            if (hits >= needed)
                expected.push_back(uint32_t(i));
        }
        EXPECT_EQ(collision_detail::selectCluster(source, coarse, {2, 1.2f, 2}, minimum).indices, expected) << minimum;
    }
    std::stop_source stop;
    stop.request_stop();
    EXPECT_THROW((void)collision_detail::selectCluster(source, coarse, {2, 1.2f, 2}, .1f, stop.get_token()), std::runtime_error);
}
