/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/cuda/collision_voxelization.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include <cuda_runtime.h>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>
#include <random>
using namespace lfs::core;
namespace {
    struct Gaussian {
        glm::vec3 center, scale;
        glm::quat rotation{1, 0, 0, 0};
        float alpha = .6f;
    };
    CollisionSource source(const std::vector<Gaussian>& gs) {
        std::vector<float> means, scales, rotations, opacity;
        for (const auto& g : gs) {
            for (int a = 0; a < 3; ++a) {
                means.push_back(g.center[a]);
                scales.push_back(g.scale[a]);
            }
            rotations.insert(rotations.end(), {g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z});
            opacity.push_back(g.alpha);
        }
        return {Tensor::from_vector(means, {gs.size(), 3}), Tensor::from_vector(scales, {gs.size(), 3}), Tensor::from_vector(rotations, {gs.size(), 4}), Tensor::from_vector(opacity, {gs.size()}), {}, glm::mat4(1)};
    }
    // Independent scalar leaf predicate: no BVH, support selection or uniform
    // proofs. This is the clamped-centre density rule, not exact ellipsoid/box distance.
    bool reference(const std::vector<Gaussian>& gs, glm::vec3 center, float size, float opacity) {
        double density = 0;
        for (const auto& g : gs) {
            auto axes = glm::mat3_cast(g.rotation);
            for (int a = 0; a < 3; ++a)
                axes[a] *= g.scale[a];
            glm::vec3 extent;
            for (int a = 0; a < 3; ++a)
                extent[a] = 3 * glm::length(glm::vec3(axes[0][a], axes[1][a], axes[2][a]));
            if (glm::any(glm::greaterThan(glm::abs(center - g.center), extent + size * .5f)))
                continue;
            const auto delta = glm::clamp(g.center, center - size * .5f, center + size * .5f) - g.center;
            const auto q = glm::inverse(glm::dmat3(axes)) * glm::dvec3(delta);
            density += g.alpha * std::exp(-.5 * glm::dot(q, q));
        }
        return density > 0 && density >= -std::log1p(-double(opacity));
    }
    class CollisionGpu : public ::testing::Test {
        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
                GTEST_SKIP() << "CUDA device unavailable";
        }
    };
    TEST_F(CollisionGpu, RotatedThinFaintOverlapAndDuplicateMortonKeys) {
        std::mt19937 rng(791);
        std::uniform_real_distribution<float> p(-.3f, .3f);
        std::vector<Gaussian> gs;
        for (int i = 0; i < 33; ++i)
            gs.push_back({{p(rng), p(rng), p(rng)}, {.12f, .06f, .003f}, glm::angleAxis(p(rng) * 8, glm::normalize(glm::vec3(1, 2, 3))), .4f});
        for (int i = 0; i < 20; ++i)
            gs.push_back({{.8f, .1f, 0}, {.04f, .04f, .04f}, {1, 0, 0, 0}, .01f});
        auto input = source(gs);
        CollisionSettings settings{.04f, .1f, 0, false, false};
        const auto volume = generateCollision({&input, 1}, settings);
        uint64_t count = 0;
        for (int z = 0; z < volume.dimensions.z; ++z)
            for (int y = 0; y < volume.dimensions.y; ++y)
                for (int x = 0; x < volume.dimensions.x; ++x) {
                    glm::vec3 center;
                    const glm::ivec3 cell{x, y, z};
                    for (int a = 0; a < 3; ++a)
                        center[a] = std::fma(float(cell[a]) + .5f, settings.voxel_size, volume.origin[a]);
                    const auto expected = reference(gs, center, settings.voxel_size, settings.opacity);
                    ASSERT_EQ(volume.occupied(cell), expected) << x << ',' << y << ',' << z;
                    count += expected;
                }
        EXPECT_EQ(volume.occupied_count, count);
    }
    TEST_F(CollisionGpu, NearCutoffDistantSupportAndBoundedFaintProof) {
        const float alpha = .003192742820829153f;
        float sum = 0;
        for (int i = 0; i < 33; ++i)
            sum += alpha;
        ASSERT_GE(sum, -std::log1p(-double(.1f)));
        std::vector<Gaussian> gs(33, {{0, 0, 0}, {.01f, .01f, .01f}, {1, 0, 0, 0}, alpha});
        gs.insert(gs.end(), 33, {{100, 0, 0}, {.01f, .01f, .01f}, {1, 0, 0, 0}, alpha});
        auto input = source(gs);
        const auto v = generateCollision({&input, 1}, {.1f, .1f, 0, false, false});
        for (const float x : {0.f, 100.f})
            EXPECT_TRUE(v.occupied(glm::ivec3(glm::floor((glm::vec3(x, 0, 0) - v.origin) / .1f))));
        input = source(std::vector<Gaussian>(8192, {{0, 0, 0}, {.01f, .01f, .01f}, {1, 0, 0, 0}, 1e-6f}));
        EXPECT_THROW(generateCollision({&input, 1}, {.1f, .1f, 0, false, false}), std::runtime_error);
    }
    TEST_F(CollisionGpu, CaptureCompletesBeforeCrossStreamMutation) {
        cudaStream_t writer, reader;
        ASSERT_EQ(cudaStreamCreateWithFlags(&writer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking), cudaSuccess);
        {
            CUDAStreamGuard writing(writer);
            constexpr size_t n = 100000;
            std::vector<float> rotations(n * 4);
            for (size_t i = 0; i < n; ++i)
                rotations[i * 4] = 1;
            SplatData model(0, Tensor::ones({n, 3}, Device::CUDA), Tensor::zeros({n, 1, 3}, Device::CUDA), {},
                            Tensor::zeros({n, 3}, Device::CUDA), Tensor::from_vector(rotations, {n, 4}, Device::CUDA),
                            Tensor::zeros({n, 1}, Device::CUDA), 1);
            CollisionSource captured;
            {
                CUDAStreamGuard reading(reader);
                captured = captureCollisionSource(model);
            }
            EXPECT_EQ(cudaStreamQuery(reader), cudaSuccess);
            EXPECT_EQ(cudaStreamQuery(writer), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(model.means_raw().ptr<float>(), 0, n * 3 * sizeof(float), writer), cudaSuccess);
            EXPECT_EQ(captured.means.to(Device::CPU).ptr<float>()[0], 1);
            ASSERT_EQ(cudaStreamSynchronize(writer), cudaSuccess);
        }
        EXPECT_EQ(cudaStreamDestroy(reader), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(writer), cudaSuccess);
    }
    TEST_F(CollisionGpu, PlansLargeOutputSortWithoutAllocatingRecords) {
        // Exercise the real CUB storage query above the former output cap.
        // The production append path uses the same checked record-byte layout.
        const auto bytes = cuda::collisionOutputBytes(64'000'000, 1'000'000);
        const auto records = bytes / sizeof(cuda::CollisionBrick);
        const auto workspace = cuda::collisionSortWorkspaceBytes(records);
        EXPECT_GT(workspace, 0u);
        EXPECT_GE(workspace, cuda::collisionSortWorkspaceBytes(records / 2));
        EXPECT_THROW(cuda::collisionSortWorkspaceBytes(SIZE_MAX), std::length_error);
    }
    TEST_F(CollisionGpu, AutoTenfoldNodeAndBakedScalePreserveRelativeOccupancy) {
        const std::vector<Gaussian> gs{{{-.13f, .07f, .11f}, {.1f, .2f, .3f}},
                                       {{.31f, -.09f, -.21f}, {.2f, .1f, .15f}}};
        auto original = source(gs);
        const CollisionSettings settings{.05f, .1f, 0, true, true};
        float resolved = 0;
        const auto base = generateCollision({&original, 1}, settings, {}, {}, [&](float size) { resolved = size; });
        EXPECT_FLOAT_EQ(resolved, base.settings.voxel_size);
        for (bool baked : {false, true}) {
            auto input = source(gs);
            if (baked) {
                auto scaled = gs;
                for (auto& g : scaled) {
                    g.center *= 10;
                    g.scale *= 10;
                }
                input = source(scaled);
            } else {
                input.transform = glm::mat4(10);
                input.transform[3][3] = 1;
            }
            const auto volume = generateCollision({&input, 1}, settings);
            EXPECT_NEAR(volume.settings.voxel_size / base.settings.voxel_size, 10., 1e-5);
            EXPECT_NEAR(double(volume.occupied_count) / base.occupied_count, 1., .02);
            for (int z = -8; z <= 8; ++z)
                for (int y = -8; y <= 8; ++y)
                    for (int x = -8; x <= 8; ++x) {
                        const auto cell = glm::ivec3(glm::floor((glm::vec3(x, y, z) * .1f - base.origin) / base.settings.voxel_size));
                        // Probe cell centres, not world points that round across a shared grid boundary.
                        const auto p = base.origin + (glm::vec3(cell) + .5f) * base.settings.voxel_size;
                        EXPECT_EQ(base.occupied(cell),
                                  volume.occupied(glm::ivec3(glm::floor((p * 10.f - volume.origin) / volume.settings.voxel_size))));
                    }
        }
    }
    TEST_F(CollisionGpu, AutoScaleInvariantEmptyAndInvalidSources) {
        auto input = source({{{0, 0, 0}, {.1f, .2f, .3f}}});
        CollisionSettings settings{.05f, .1f, 0, true, false};
        CollisionVolume a, b;
        ASSERT_NO_THROW(a = generateCollision({&input, 1}, settings));
        input.transform = glm::mat4(.01f);
        input.transform[3][3] = 1;
        ASSERT_NO_THROW(b = generateCollision({&input, 1}, settings));
        EXPECT_NEAR(b.settings.voxel_size / a.settings.voxel_size, .01f, 1e-7f);
        EXPECT_NEAR(double(b.occupied_count) / a.occupied_count, 1., .02);
        input.opacity = Tensor::zeros({1});
        EXPECT_THROW(generateCollision({&input, 1}, settings), std::runtime_error);
        input = source({{{0, 0, 0}, {INFINITY, .1f, .1f}}});
        EXPECT_THROW(generateCollision({&input, 1}, settings), std::invalid_argument);
        input = source({{{0, 0, 0}, {.1f, .1f, .1f}}});
        EXPECT_THROW(generateCollision({&input, 1}, settings, {}, [](float, const auto&) { return false; }), std::runtime_error);
    }
} // namespace
