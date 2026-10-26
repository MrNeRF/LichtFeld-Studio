/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_fixture.hpp"
#include "core/collision_walk.hpp"
#include "core/cuda/collision_voxelization.hpp"
#include "core/cuda_error.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "tensor_test_support.hpp"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <tuple>

using namespace lfs::core;

TEST(Collision, LogicalBoundsAndPackedBricks) {
    CollisionVolume v{.dimensions = {1, 3, 3}, .edge = 4, .occupied_count = 4, .words = {CollisionVolume::brick_tag | 1, 0x00110011, 0}};
    ASSERT_NO_THROW(v.validate());
    EXPECT_TRUE(v.occupied({0, 1, 1}));
    EXPECT_FALSE(v.occupied({1, 1, 1}));
    EXPECT_FALSE(v.occupied({0, 3, 1}));
    EXPECT_FALSE(v.occupied({-1, 0, 0}));
    v.words[1] |= 2;
    EXPECT_THROW(v.validate(), std::invalid_argument);
    v.words = {1};
    v.occupied_count = 9;
    EXPECT_NO_THROW(v.validate());
}

TEST(Collision, MalformedTopologyAndSettings) {
    CollisionVolume v{.dimensions = {5, 5, 5}, .edge = 8, .words = {3, 0, 0, 0, 0, 0, 0, 0, 0}};
    EXPECT_NO_THROW(v.validate());
    for (const auto pointer : {2u, 4u, 0x80000000u, 0xffffffffu}) {
        v.words[1] = pointer;
        EXPECT_THROW(v.validate(), std::invalid_argument);
    }
    v.words = {1};
    v.occupied_count = 125;
    EXPECT_NO_THROW(v.validate());
    v.occupied_count++;
    EXPECT_THROW(v.validate(), std::invalid_argument);
    for (const float invalid : {0.f, -1.f, INFINITY, NAN}) {
        v.settings.voxel_size = invalid;
        EXPECT_THROW(v.settings.validate(), std::invalid_argument);
    }
    for (const auto settings : {CollisionSettings{}, CollisionSettings{.max_gaussian_area = 1}, CollisionSettings{.cleanup = false}})
        EXPECT_NO_THROW(settings.validate());
    EXPECT_THROW((CollisionSettings{.max_gaussian_area = 1, .cleanup = false}.validate()), std::invalid_argument);
    std::stop_source stop;
    stop.request_stop();
    EXPECT_THROW(generateCollision({}, {}, stop.get_token()), std::runtime_error);
    const CollisionSource invalid;
    EXPECT_THROW(generateCollision(invalid, {}), std::invalid_argument);
}

TEST(Collision, OutputLayoutAcrossDispatchesAndAddressOverflow) {
    constexpr size_t records = 61 * (1u << 20);
    for (size_t batch = 0; batch < 61; ++batch)
        EXPECT_EQ(cuda::collisionOutputBytes(batch * (1u << 20), 1u << 20), (batch + 1) * (1u << 20) * sizeof(cuda::CollisionBrick));
    EXPECT_THROW(cuda::collisionOutputBytes(records, 1u << 20), std::length_error);
    EXPECT_EQ(cuda::collisionOutputBytes(records, 36'864), size_t(64'000'000) * sizeof(cuda::CollisionBrick));
    EXPECT_THROW(cuda::collisionOutputBytes(records, 36'865), std::length_error);
    EXPECT_EQ(cuda::collisionOutputBytes(records, 17), cuda::collisionOutputBytes(records + 17));
    EXPECT_THROW(cuda::collisionOutputBytes(SIZE_MAX, 1), std::length_error);
    EXPECT_THROW(cuda::collisionOutputBytes(std::vector<cuda::CollisionBrick>().max_size(), 1), std::length_error);
}

namespace {
    using collision_test::cells;
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
    // Independent clamped-centre density oracle, without BVH or uniform proofs.
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
    class CollisionGpu : public lfs::test::CudaBackendTest {};
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
        uint64_t mixed = 0;
        const auto volume = generateCollision(input, settings, {}, {}, {}, [&](uint64_t value) {
            EXPECT_GE(value, mixed);
            mixed = value;
        });
        uint64_t expected_mixed = 0;
        for (const auto block : cells((volume.dimensions + 3) / 4)) {
            const auto count = std::ranges::count_if(cells(glm::ivec3(4)), [&](auto p) { return volume.occupied(block * 4 + p); });
            expected_mixed += count > 0 && count < 64;
        }
        EXPECT_EQ(mixed, expected_mixed);
        EXPECT_THROW(generateCollision(input, settings, {}, {}, {}, [](uint64_t value) {
                         if (value)
                             throw OperationCancelled();
                     }),
                     OperationCancelled);
        uint64_t count = 0;
        for (const auto cell : cells(volume.dimensions)) {
            glm::vec3 center;
            for (int a = 0; a < 3; ++a)
                center[a] = std::fma(float(cell[a]) + .5f, settings.voxel_size, volume.origin[a]);
            const auto expected = reference(gs, center, settings.voxel_size, settings.opacity);
            ASSERT_EQ(volume.occupied(cell), expected) << cell.x << ',' << cell.y << ',' << cell.z;
            count += expected;
        }
        EXPECT_EQ(volume.occupied_count, count);
    }
    TEST_F(CollisionGpu, NearCutoffDistantSupportAndBoundedFaintProof) {
        const float alpha = .003192742820829153f;
        std::vector<Gaussian> gs(33, {{0, 0, 0}, {.01f, .01f, .01f}, {1, 0, 0, 0}, alpha});
        gs.insert(gs.end(), 33, {{100, 0, 0}, {.01f, .01f, .01f}, {1, 0, 0, 0}, alpha});
        auto input = source(gs);
        const auto v = generateCollision(input, {.1f, .1f, 0, false, false});
        for (const float x : {0.f, 100.f})
            EXPECT_TRUE(v.occupied(glm::ivec3(glm::floor((glm::vec3(x, 0, 0) - v.origin) / .1f))));
        input = source(std::vector<Gaussian>(8192, {{0, 0, 0}, {.01f, .01f, .01f}, {1, 0, 0, 0}, 1e-6f}));
        EXPECT_THROW(generateCollision(input, {.1f, .1f, 0, false, false}), std::runtime_error);
    }
    TEST_F(CollisionGpu, CaptureSuccessAndExceptionFenceReadsBeforeCrossStreamMutation) {
        struct FailingRead : TensorExpr<FailingRead> {
            Tensor input;
            Tensor* copy;
            std::atomic<bool>* completed;
            const TensorShape& shape_impl() const { return input.shape(); }
            Device device_impl() const { return input.device(); }
            auto gpu_backend_impl() const { return gpu_backend_of(input); }
            DataType dtype_impl() const { return input.dtype(); }
            cudaStream_t stream_hint_impl() const { return input.stream(); }
            FailingRead snapshot_impl() const { return *this; }
            Tensor eval_impl() const {
                CUDAStreamGuard home(input.stream());
                const auto delay = [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); };
                LFS_CUDA_CHECK_MSG(cudaLaunchHostFunc(input.stream(), delay, nullptr), "delay partial snapshot read");
                *copy = input.clone();
                const auto mark = [](void* state) { static_cast<std::atomic<bool>*>(state)->store(true); };
                LFS_CUDA_CHECK_MSG(cudaLaunchHostFunc(input.stream(), mark, completed), "observe partial snapshot read");
                throw std::runtime_error("injected snapshot failure");
            }
        };
        lfs::test::ForceDeferredTensorExecution force_deferred;
        cudaStream_t writer, reader;
        ASSERT_EQ(cudaStreamCreateWithFlags(&writer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking), cudaSuccess);
        for (const auto capture_stream : {reader, cudaStream_t{nullptr}})
            for (bool fail : {false, true}) {
                SCOPED_TRACE(fail);
                CUDAStreamGuard writing(writer);
                constexpr size_t n = 100000;
                std::vector<float> rotations(n * 4);
                for (size_t i = 0; i < n; ++i)
                    rotations[i * 4] = 2;
                auto raw_rotation = Tensor::from_vector(rotations, {n, 4}, Device::GPU);
                SplatData model(0, Tensor::ones({n, 3}, Device::GPU), Tensor::zeros({n, 1, 3}, Device::GPU), {},
                                Tensor::full({n, 3}, std::log(2.f), Device::GPU), raw_rotation, Tensor::zeros({n, 1}, Device::GPU), 1);
                model.deleted() = Tensor::zeros({n}, Device::GPU, DataType::Bool);
                Tensor partial_copy;
                std::atomic<bool> read_completed{false};
                if (fail)
                    model.rotation_raw() = FailingRead{{}, raw_rotation, &partial_copy, &read_completed};
                CollisionSource captured;
                {
                    CUDAStreamGuard reading(capture_stream);
                    if (fail) {
                        ASSERT_TRUE(model.rotation_raw().is_deferred());
                        EXPECT_THROW(captureCollisionSource(model), std::runtime_error);
                        EXPECT_TRUE(read_completed.load());
                    } else {
                        captured = captureCollisionSource(model);
                    }
                }
                EXPECT_EQ(cudaStreamQuery(capture_stream), cudaSuccess);
                EXPECT_EQ(cudaStreamQuery(writer), cudaSuccess);
                for (auto* tensor : {&model.means_raw(), &model.scaling_raw(), &raw_rotation, &model.opacity_raw()})
                    ASSERT_EQ(cudaMemsetAsync(tensor->data_ptr(), 0, tensor->bytes(), writer), cudaSuccess);
                ASSERT_EQ(cudaMemsetAsync(model.deleted().data_ptr(), 1, model.deleted().bytes(), writer), cudaSuccess);
                if (fail) {
                    ASSERT_TRUE(partial_copy.is_valid());
                    EXPECT_FLOAT_EQ(partial_copy.to(Device::CPU).ptr<float>()[0], 2);
                } else {
                    for (const auto& [tensor, expected] : {std::pair{captured.means, 1.f}, {captured.scales, 2.f}, {captured.opacity, .5f}}) {
                        const auto cpu = tensor.cpu();
                        EXPECT_TRUE(std::ranges::all_of(std::span(cpu.ptr<float>(), cpu.numel()), [expected](float x) { return x == expected; }));
                    }
                    const auto rotation = captured.rotations.cpu(), deleted = captured.deleted.cpu();
                    for (size_t i = 0; i < rotation.numel(); ++i)
                        ASSERT_FLOAT_EQ(rotation.ptr<float>()[i], i % 4 == 0 ? 1 : 0);
                    EXPECT_TRUE(std::ranges::all_of(std::span(deleted.ptr<unsigned char>(), deleted.numel()), [](auto x) { return x == 0; }));
                }
                ASSERT_EQ(cudaStreamSynchronize(writer), cudaSuccess);
            }
        EXPECT_EQ(cudaStreamDestroy(reader), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(writer), cudaSuccess);
    }
    TEST_F(CollisionGpu, BoundsOutputBeforeAllocationAndPlansSortWorkspace) {
        constexpr size_t records = 64'000'000; // Append/overflow boundaries are covered by Collision.OutputLayout.
        const auto workspace = cuda::collisionSortWorkspaceBytes(records);
        EXPECT_GT(workspace, 0u);
        EXPECT_GE(workspace, cuda::collisionSortWorkspaceBytes(records / 2));
        EXPECT_THROW(cuda::collisionSortWorkspaceBytes(SIZE_MAX), std::length_error);
    }
    TEST_F(CollisionGpu, AutoResolutionIsRelativeForBakedAndNodeScaleAndRejectsInvalidSources) {
        const std::vector<Gaussian> gs{{{-.13f, .07f, .11f}, {.1f, .2f, .3f}},
                                       {{.31f, -.09f, -.21f}, {.2f, .1f, .15f}}};
        auto original = source(gs);
        const CollisionSettings settings{.05f, .1f, 0, true, true};
        float resolved = 0;
        const auto base = generateCollision(original, settings, {}, {}, [&](float size) { resolved = size; });
        EXPECT_FLOAT_EQ(resolved, base.settings.voxel_size);
        for (float scale : {.01f, 10.f})
            for (bool baked : {false, true}) {
                auto scaled = gs;
                for (auto& g : scaled) {
                    g.center *= baked ? scale : 1;
                    g.scale *= baked ? scale : 1;
                }
                auto input = source(scaled);
                input.transform = glm::mat4(baked ? 1.f : scale);
                input.transform[3][3] = 1;
                const auto volume = generateCollision(input, settings);
                EXPECT_NEAR(volume.settings.voxel_size / base.settings.voxel_size, scale, scale * 1e-5);
                EXPECT_NEAR(double(volume.occupied_count) / base.occupied_count, 1., .02);
                for (const auto probe : cells(glm::ivec3(17))) {
                    const auto cell = glm::ivec3(glm::floor((glm::vec3(probe - 8) * .1f - base.origin) / base.settings.voxel_size));
                    const auto p = base.origin + (glm::vec3(cell) + .5f) * base.settings.voxel_size;
                    EXPECT_EQ(base.occupied(cell), volume.occupied(glm::ivec3(glm::floor((p * scale - volume.origin) / volume.settings.voxel_size))));
                }
            }
        auto input = source({{{0, 0, 0}, {.1f, .2f, .3f}}});
        input.opacity = Tensor::zeros({1});
        EXPECT_THROW(generateCollision(input, settings), std::runtime_error);
        input = source({{{0, 0, 0}, {INFINITY, .1f, .1f}}});
        EXPECT_THROW(generateCollision(input, settings), std::invalid_argument);
        input = source({{{0, 0, 0}, {.1f, .1f, .1f}}});
        EXPECT_THROW(generateCollision(input, settings, {}, [](float, const auto&) { return false; }), std::runtime_error);
    }
    using collision_test::box;
    CapsuleQuery queryVolumes(std::vector<std::shared_ptr<const CollisionVolume>> volumes, glm::mat4 transform = glm::mat4(1)) {
        return [volumes = std::move(volumes), transform](auto feet, auto up, float height, float radius) {
            return std::ranges::any_of(volumes, [&](const auto& v) { return collisionCapsuleIntersects(*v, transform, feet, up, height, radius); });
        };
    }
    TEST(CollisionWalk, AnalyticalClearanceAndTransformedSegment) {
        const auto v = box({}, glm::ivec3(1));
        for (const auto [feet, up, height, radius, hit] : {
                 std::tuple{glm::vec3(-.25f, 0, .5f), glm::vec3(0, 1, 0), 2.f, .25f, false},
                 {{-.249f, 0, .5f}, {0, 1, 0}, 2.f, .25f, true},
                 {{1.2f, 1.2f, .5f}, {0, 0, 1}, .4f, .2f, false}})
            EXPECT_EQ(collisionCapsuleIntersects(*v, glm::mat4(1), feet, up, height, radius), hit);
        EXPECT_TRUE(collisionCapsuleIntersects(*v, glm::rotate(glm::mat4(1), .5f, {0, 0, 1}), {.3f, .3f, .5f}, {0, 1, 0}, 1, .1f));
    }
    TEST(CollisionWalk, WallSweepLedgeAndEditedGeometryAreSafe) {
        const auto floor = box({-2, -1, -2}, {4, 1, 4});
        auto transform = glm::mat4(1);
        const auto wall = box({.5f, 0, -2}, {1, 400, 400}, .01f);
        const auto ceiling = box({-2, 2, -2}, {4, 1, 4});
        const auto obstacles = queryVolumes({wall, ceiling});
        const CapsuleQuery query = [&](auto feet, auto up, float height, float radius) {
            return collisionCapsuleIntersects(*floor, transform, feet, up, height, radius) ||
                   obstacles(feet, up, height, radius);
        };
        WalkSettings settings;
        WalkState state{.eye = {0, 1.8f, -1.99f}};
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query), std::invalid_argument);
        settleWalk(state, settings, {0, 1, 0}, query);
        EXPECT_NEAR(state.eye.y, settings.height, .001f);
        for (int i = 0; i < 500; ++i)
            advanceWalk(state, settings, {0, 1, 0}, {10, 0, 0}, .1f, query);
        EXPECT_LE(state.eye.x, .251f);
        const auto grounded = state.feet;
        for (int i = 0; i < 500; ++i) {
            ASSERT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, {0, 0, 100}, .1f, query));
            EXPECT_EQ(state.feet, grounded);
            EXPECT_TRUE(state.grounded);
        }
        for (bool edit_geometry : {false, true}) {
            settings.height = edit_geometry ? 1.7f : 2.1f;
            transform[3].y = edit_geometry ? .1f : 0;
            EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query), std::runtime_error);
            EXPECT_EQ(state.feet, grounded);
        }
        transform[3].y = 0;
        EXPECT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query));
        WalkState unsupported{.eye = {100, 100, 100}};
        EXPECT_THROW(settleWalk(unsupported, settings, {0, 1, 0}, query), std::runtime_error);
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, [](auto, auto, float, float) { return false; }), std::runtime_error);
    }
    TEST(CollisionWalk, MinimumStepFitsButLowerCeilingBlocksLift) {
        for (float height : {1.85f, 1.75f}) {
            const auto query = queryVolumes({box({-2, -1, -2}, {4, 1, 4}), box({.5f, 0, -2}, {100, 10, 400}, .01f),
                                             box({-2, height, -2}, {400, 10, 400}, .01f)});
            WalkSettings settings;
            WalkState state{.eye = {0, 1.7f, 0}, .presentation = {0, 1.7f, 0}};
            settleWalk(state, settings, {0, 1, 0}, query);
            for (int i = 0; i < 550; ++i) {
                ASSERT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, i < 50 ? glm::vec3(.5f, 0, 0) : glm::vec3(0), i < 50 ? .05f : .016f, query));
                ASSERT_FALSE(query(state.feet, {0, 1, 0}, settings.height, settings.radius));
            }
            EXPECT_EQ(state.eye.x > 1, height == 1.85f);
            if (height == 1.85f)
                EXPECT_NEAR(state.eye.y, 1.8f, .001f);
            EXPECT_LE(state.presentation.y, height);
        }
    }
    TEST(CollisionWalk, RepeatedIdleAndMovementAcrossTransformsAndPrecisionBoundaries) {
        auto exercise = [](const auto& query, WalkSettings settings, glm::vec3 up, glm::vec3 eye, glm::vec3 velocity) {
            WalkState state{.eye = eye};
            ASSERT_NO_THROW(settleWalk(state, settings, up, query));
            const auto grounded = state.feet;
            for (int frame = 0; frame < 700; ++frame) {
                ASSERT_NO_THROW(advanceWalk(state, settings, up, frame < 500 ? glm::vec3(0) : velocity, .016f, query));
                ASSERT_FALSE(query(state.feet, up, settings.height, settings.radius));
                ASSERT_TRUE(state.grounded);
                if (velocity == glm::vec3(0))
                    ASSERT_EQ(state.feet, grounded);
            }
        };
        for (float translation : {0.f, 100.f, 1000.f, -10000.f})
            for (float scale : {.01f, 1.f, 100.f})
                for (float angle : {0.f, .37f, 1.1f}) {
                    auto t = glm::scale(glm::rotate(glm::translate(glm::mat4(1), glm::vec3(translation)), angle, glm::normalize(glm::vec3(1, 2, 3))), glm::vec3(scale));
                    exercise(queryVolumes({box({-20, -1, -20}, {40, 1, 40})}, t), {1.7f * scale, .25f * scale, .3f * scale, 2 * scale},
                             glm::normalize(glm::vec3(t * glm::vec4(0, 1, 0, 0))), glm::vec3(t * glm::vec4(0, 2, 0, 1)), glm::vec3(t * glm::vec4(.2f, 0, 0, 0)));
                }
        for (float altitude : {.1234567f, 1.99999f, 15.9999f, 127.999f, 1023.9999f, -1024.0001f, 8191.9995f})
            for (float height : {.7f, 1.7f, 3.14159f, 17.f})
                for (float radius : {.025f, .25f})
                    exercise(queryVolumes({box({-20, altitude - 1, -20}, {40, 1, 40})}), {height, radius, .3f, 2}, {0, 1, 0}, {0, altitude + height + radius, 0}, {});
    }
} // namespace
