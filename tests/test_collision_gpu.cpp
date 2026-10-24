/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../src/core/collision_bvh.hpp"
#include "core/collision.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <random>
using namespace lfs::core;

TEST(CollisionBvh, ExactBalancedNodeCountsAndParallelLayout) {
    auto count = [](auto&& self, size_t n) -> size_t {
        return n <= 8 ? 1 : 1 + self(self, n / 2) + self(self, n - n / 2);
    };
    for (size_t n = 1; n < 2048; ++n)
        ASSERT_EQ(collision_detail::collisionBvhNodeCount(n), count(count, n)) << n;
    std::mt19937 random(7731);
    for (const size_t n : {9, 1024, 65537, 300000}) {
        std::vector<cuda::CollisionGaussianGpu> serial(n);
        for (auto& gaussian : serial)
            for (int axis = 0; axis < 3; ++axis) {
                gaussian.min[axis] = int(random() % 10000) - 5000;
                gaussian.max[axis] = gaussian.min[axis] + 1 + random() % 100;
                gaussian.extent[axis] = float(random() % 1000) * .01f;
            }
        auto parallel = serial;
        const auto a = collision_detail::buildCollisionBvh(std::span(serial), {}, 1);
        const auto b = collision_detail::buildCollisionBvh(std::span(parallel), {}, 8);
        ASSERT_EQ(a.size, count(count, n));
        ASSERT_EQ(a.size, b.size);
        EXPECT_EQ(std::memcmp(a.nodes.get(), b.nodes.get(), a.size * sizeof(cuda::CollisionBvhNode)), 0);
        EXPECT_EQ(std::memcmp(serial.data(), parallel.data(), serial.size() * sizeof(serial[0])), 0);
        // Every leaf owns each source exactly once and bounds all its supports.
        std::vector<unsigned> visited(n);
        for (const auto& node : b.view()) {
            if (!node.count) {
                ASSERT_LT(node.left, b.size);
                ASSERT_LT(node.right, b.size);
                auto support = [&](uint32_t child) {
                    const auto& c = b.nodes[child];
                    return std::bit_cast<float>(c.count ? c.left : c.begin);
                };
                EXPECT_EQ(std::bit_cast<float>(node.begin), std::max(support(node.left), support(node.right)));
                continue;
            }
            ASSERT_LE(node.begin + node.count, n);
            float expected_support = 0;
            for (size_t i = node.begin; i < node.begin + node.count; ++i) {
                ++visited[i];
                expected_support = std::max(expected_support, std::min({parallel[i].extent[0], parallel[i].extent[1], parallel[i].extent[2]}));
                for (int axis = 0; axis < 3; ++axis) {
                    EXPECT_LE(node.min[axis], parallel[i].min[axis]);
                    EXPECT_GE(node.max[axis], parallel[i].max[axis]);
                }
            }
            EXPECT_EQ(std::bit_cast<float>(node.left), expected_support);
        }
        EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](auto visits) { return visits == 1; }));
    }
}

TEST(CollisionBvh, CancellationAndEmptyInputFailBeforeAllocation) {
    std::vector<cuda::CollisionGaussianGpu> source(100);
    std::stop_source stop;
    stop.request_stop();
    EXPECT_THROW(collision_detail::buildCollisionBvh(std::span(source), stop.get_token()), std::runtime_error);
    EXPECT_THROW(collision_detail::buildCollisionBvh(std::span<cuda::CollisionGaussianGpu>()), std::invalid_argument);
}

namespace {
    class CollisionGpu : public ::testing::Test {
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
                GTEST_SKIP() << "CUDA device unavailable";
        }
    };
    void same(const CollisionVolume& a, const CollisionVolume& b) {
        ASSERT_EQ(a.origin, b.origin);
        ASSERT_EQ(a.dimensions, b.dimensions);
        for (int z = 0; z < a.dimensions.z; ++z)
            for (int y = 0; y < a.dimensions.y; ++y)
                for (int x = 0; x < a.dimensions.x; ++x)
                    ASSERT_EQ(a.occupied({x, y, z}), b.occupied({x, y, z})) << x << "," << y << "," << z;
        EXPECT_EQ(a.occupiedCount(), b.occupiedCount());
    }
    TEST_F(CollisionGpu, RotatedAnisotropicOverlappingGaussiansMatchCpu) {
        std::mt19937 random(1979);
        std::uniform_real_distribution<float> position(-.5f, .5f), scale(.01f, .2f), alpha(.02f, 1);
        std::vector<CollisionGaussian> source;
        for (int i = 0; i < 80; ++i) {
            auto axes = glm::mat3(glm::rotate(glm::mat4(1), position(random) * 4, glm::normalize(glm::vec3(1, 2, 3))));
            for (int a = 0; a < 3; ++a)
                axes[a] *= scale(random);
            source.push_back({{position(random), position(random), position(random)}, axes, alpha(random)});
        }
        for (int i = 0; i < 12; ++i)
            source.push_back({glm::vec3(0), glm::mat3(.3f), 1});
        for (float threshold : {0.f, .1f, .5f, .999f}) {
            SCOPED_TRACE(threshold);
            CollisionSettings s;
            s.voxel_size = .1f;
            s.opacity = threshold;
            same(generateCollision(source, s), generateCollision(source, s, {}, {}, voxelizeCollisionGpu));
        }
    }
    TEST_F(CollisionGpu, UniformInteriorProofMatchesExactCellsAndPartialPadding) {
        std::vector<CollisionGaussian> source;
        for (int i = 0; i < 40; ++i)
            source.push_back({{float(i % 3) * .03f, float(i % 5) * .02f, float(i % 7) * .01f}, glm::mat3(.4f), .2f});
        CollisionSettings s;
        s.voxel_size = .05f;
        s.opacity = .5f;
        same(generateCollision(source, s), generateCollision(source, s, {}, {}, voxelizeCollisionGpu));
    }
    TEST_F(CollisionGpu, ThinSheetsAndDistantComponentsKeepCoverage) {
        CollisionSettings s;
        s.voxel_size = .1f;
        const auto axes = glm::mat3(.2f, 0, 0, 0, .0002f, 0, 0, 0, .2f);
        const std::array source{CollisionGaussian{{0, .001f, 0}, axes, 1}, CollisionGaussian{{40000, .001f, 40000}, axes, 1},
                                CollisionGaussian{{1e9f, 1e9f, 1e9f}, glm::mat3(1), .01f}};
        auto gpu = generateCollision(source, s, {}, {}, voxelizeCollisionGpu);
        auto cpu = generateCollision(source, s);
        EXPECT_EQ(gpu.origin, cpu.origin);
        EXPECT_EQ(gpu.dimensions, cpu.dimensions);
        EXPECT_EQ(gpu.occupiedCount(), cpu.occupiedCount());
        for (auto center : {source[0].center, source[1].center})
            EXPECT_TRUE(gpu.occupied(gpu.cell(center)));
        EXPECT_FALSE(gpu.occupied(gpu.cell({20000, 0, 20000})));
        EXPECT_LT(gpu.tree.size(), 10000u);
    }
    TEST_F(CollisionGpu, ClusterAndPostprocessUseGpuWithoutChangingSemantics) {
        CollisionSettings s;
        s.cluster = true;
        s.cluster_resolution = .2f;
        s.cluster_opacity = .1f;
        s.voxel_size = .1f;
        s.seed = {0, 0, 0};
        s.floor_fill = true;
        s.floor_size = 0;
        const std::array source{CollisionGaussian{{1, 0, 0}, glm::mat3(.2f), 1}, CollisionGaussian{{5, 0, 0}, glm::mat3(.2f), 1}};
        same(generateCollision(source, s), generateCollision(source, s, {}, {}, voxelizeCollisionGpu));
    }
    TEST_F(CollisionGpu, CancellationAndEmptyOpacityAreExplicit) {
        CollisionSettings s;
        const CollisionGaussian g{{0, 0, 0}, glm::mat3(.1f), 1};
        std::stop_source stop;
        stop.request_stop();
        EXPECT_THROW((void)voxelizeCollisionGpu(std::span(&g, 1), s, stop.get_token()), std::runtime_error);
        s.opacity = 1;
        EXPECT_THROW((void)voxelizeCollisionGpu(std::span(&g, 1), s), std::runtime_error);
    }
    TEST_F(CollisionGpu, CancellationAfterDeviceDispatchReleasesTheJob) {
        CollisionSettings s;
        s.voxel_size = .02f;
        const CollisionGaussian g{{0, 0, 0}, glm::mat3(.5f), 1};
        std::stop_source stop;
        EXPECT_THROW((void)voxelizeCollisionGpu(std::span(&g, 1), s, stop.get_token(), [&](const std::string& message) {
                         if (message.starts_with("Voxelizing GPU: ") && std::stoi(message.substr(16)) > 15 && std::stoi(message.substr(16)) < 97)
                             stop.request_stop();
                     }),
                     std::runtime_error);
        EXPECT_TRUE(stop.stop_requested());
        EXPECT_GT(voxelizeCollisionGpu(std::span(&g, 1), s).occupiedCount(), 0u);
    }
} // namespace
