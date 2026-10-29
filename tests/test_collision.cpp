/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/converter.hpp"
#include "core/collision.hpp"
#include "core/cuda/collision_voxelization_kernels.hpp"
#include "core/cuda_error.hpp"
#include "core/native_fs.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "io/argument_parser.hpp"
#include "io/collision_export.hpp"
#include "io/project/collision_serialization.hpp"
#include "licht_test_support.hpp"
#include "tensor_test_support.hpp"
#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <random>
#include <thread>
#include <tuple>

using namespace lfs::core;

TEST(Collision, MalformedTopologyAndSettings) {
    CollisionVolume volume{.dimensions = {5, 5, 5}, .edge = 8, .words = {3, 0, 0, 0, 0, 0, 0, 0, 0}};
    ASSERT_NO_THROW(volume.validate());
    for (const auto pointer : {2u, 4u, CollisionVolume::brick_tag, UINT32_MAX}) {
        SCOPED_TRACE(pointer);
        volume.words[1] = pointer;
        collision_test::expectError([&] { volume.validate(); }, "collision.data_invalid");
    }
    volume.words = {1};
    volume.occupied_count = 125;
    ASSERT_NO_THROW(volume.validate());
    for (const uint64_t count : {uint64_t(126), UINT64_MAX}) {
        volume.occupied_count = count;
        collision_test::expectError([&] { volume.validate(); }, "collision.data_invalid");
    }
    for (const float invalid : {-1.f, INFINITY, NAN})
        for (const auto field : {&CollisionSettings::voxel_size, &CollisionSettings::opacity, &CollisionSettings::max_gaussian_area}) {
            CollisionSettings settings;
            settings.*field = invalid;
            collision_test::expectError([&] { settings.validate(); }, "collision.settings_invalid");
        }
    for (const auto settings : {CollisionSettings{.voxel_size = 0}, CollisionSettings{.opacity = 1},
                                CollisionSettings{.max_gaussian_area = 1, .cleanup = false}})
        collision_test::expectError([&] { settings.validate(); }, "collision.settings_invalid");
    volume.occupied_count = 125;
    for (const auto settings : {CollisionSettings{}, CollisionSettings{.opacity = 0},
                                CollisionSettings{.max_gaussian_area = 1}, CollisionSettings{.cleanup = false}}) {
        volume.settings = settings;
        EXPECT_NO_THROW(volume.validate());
    }
}

TEST(Collision, InvalidVolumeAndPayloadErrorsKeepKeysAndDiagnostics) {
    const CollisionVolume valid{.dimensions = {4, 4, 4}, .edge = 4, .words = {0}};
    for (const auto words : {std::vector<uint32_t>{}, {2}, {CollisionVolume::brick_tag}, {CollisionVolume::brick_tag | 1, 0}, {0, 0}}) {
        auto volume = valid;
        volume.words = words;
        collision_test::expectError([&] { volume.validate(); }, "collision.data_invalid");
    }
    auto padded = valid;
    padded.dimensions = {1, 1, 1};
    padded.words = {CollisionVolume::brick_tag | 1, 2, 0};
    collision_test::expectError([&] { padded.validate(); }, "collision.data_invalid");
    using namespace lfs::io::project;
    const auto bytes = encodeCollision(valid);
    ASSERT_NO_THROW(decodeCollision(bytes));
    collision_test::expectError([&] { decodeCollision(std::span(bytes).first(3)); }, "collision.data_invalid");
    for (const auto [offset, value] : {std::pair{0u, 0u}, {4u, 2u}, {56u, 2u}, {60u, 2u}, {64u, 2u}, {68u, 2u}}) {
        auto damaged = bytes;
        chapter_binary::write_u32(damaged, offset, value);
        collision_test::expectError([&] { decodeCollision(damaged); }, "collision.data_invalid");
    }
}

TEST(Collision, PresentationTranslatesKeysAndHidesUnexpectedDiagnostics) {
    auto& locale = lfs::event::LocalizationManager::getInstance();
    const auto language = locale.getCurrentLanguage();
    ASSERT_TRUE(locale.initialize(PROJECT_ROOT_PATH "/src/visualizer/gui/resources/locales"));
    ASSERT_TRUE(locale.setLanguage("de"));
    const auto failure = collisionError("collision.data_invalid", "Invalid collision pointer or shared subtree", lfs::ErrorCode::DataLoss, "37 / 17");
    EXPECT_EQ(localizedCollisionError(failure), std::string(LOC("collision.data_invalid")) + "\n37 / 17");
    EXPECT_NE(std::string(failure.what()).find("Invalid collision pointer"), std::string::npos);
    EXPECT_EQ(localizedCollisionError(std::runtime_error("unexpected diagnostic")), LOC("selection.operation_failed_generic"));
    ASSERT_TRUE(locale.setLanguage(language.empty() ? "en" : language));
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
            means.insert(means.end(), {g.center.x, g.center.y, g.center.z});
            scales.insert(scales.end(), {g.scale.x, g.scale.y, g.scale.z});
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
    TEST_F(CollisionGpu, InvalidGeometryErrorsCarryLocaleKeys) {
        const auto valid = source({{{0, 0, 0}, {.1f, .1f, .1f}}});
        for (const auto means : {Tensor{}, Tensor::zeros({1, 2}), Tensor::zeros({0, 3})}) {
            auto input = valid;
            input.means = means;
            collision_test::expectError([&] { generateCollision(input, {}); }, "collision.geometry_invalid");
        }
        for (auto field : {&CollisionSource::means, &CollisionSource::scales, &CollisionSource::rotations, &CollisionSource::opacity, &CollisionSource::deleted}) {
            auto input = valid;
            input.*field = Tensor::zeros({1, 3}, Device::CPU, DataType::Bool);
            collision_test::expectError([&] { generateCollision(input, {}); }, "collision.geometry_invalid");
        }
    }
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
        EXPECT_GT(expected_mixed, 0);
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
        EXPECT_THROW(generateCollision(input, {.1f, .1f, 0, false, false}), lfs::Exception);
    }
    struct FailingRead : TensorExpr<FailingRead> {
        Tensor input;
        Tensor* copy;
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
            throw std::runtime_error("injected snapshot failure");
        }
    };
    struct CaptureCase {
        static constexpr size_t n = 100000, begin = 37, count = 17;
        Tensor raw_rotation = Tensor::full({n, 4}, 2.f, Device::GPU), partial_copy;
        SplatData model{0, Tensor::arange(0, n * 3, 1, Device::GPU).reshape({n, 3}), Tensor::zeros({n, 1, 3}, Device::GPU), {}, Tensor::full({n, 3}, std::log(2.f), Device::GPU), raw_rotation, Tensor::zeros({n, 1}, Device::GPU), 1};
        CollisionSource captured;
        explicit CaptureCase(bool fail) {
            model.deleted() = Tensor::zeros({n}, Device::GPU, DataType::Bool);
            if (fail)
                model.rotation_raw() = FailingRead{{}, raw_rotation, &partial_copy};
        }
        void mutate(cudaStream_t writer) {
            for (auto* tensor : {&model.means_raw(), &model.scaling_raw(), &raw_rotation, &model.opacity_raw()})
                ASSERT_EQ(cudaMemsetAsync(tensor->data_ptr(), 0, tensor->bytes(), writer), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(model.deleted().data_ptr(), 1, model.deleted().bytes(), writer), cudaSuccess);
        }
        void expectOwnedRange() const {
            for (const auto& [tensor, columns, expected, step] : {
                     std::tuple{captured.means, 3u, float(begin * 3), 1.f},
                     {captured.scales, 3u, 2.f, 0.f},
                     {captured.rotations, 4u, .5f, 0.f},
                     {captured.opacity, 1u, .5f, 0.f},
                     {captured.deleted, 1u, 0.f, 0.f}}) {
                EXPECT_EQ(tensor.size(0), count);
                EXPECT_EQ(tensor.numel(), count * columns);
                const auto cpu = tensor.to(DataType::Float32).cpu();
                for (size_t i = 0; i < cpu.numel(); ++i)
                    EXPECT_FLOAT_EQ(cpu.ptr<float>()[i], expected + step * i);
            }
        }
    };
    TEST_F(CollisionGpu, CaptureSuccessAndExceptionFenceReadsBeforeCrossStreamMutation) {
        lfs::test::ForceDeferredTensorExecution force_deferred;
        cudaStream_t writer, reader;
        ASSERT_EQ(cudaStreamCreateWithFlags(&writer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking), cudaSuccess);
        for (const auto capture_stream : {reader, cudaStream_t{nullptr}})
            for (bool fail : {false, true}) {
                SCOPED_TRACE(fail);
                CUDAStreamGuard writing(writer);
                CaptureCase sample(fail);
                {
                    CUDAStreamGuard reading(capture_stream);
                    if (fail) {
                        ASSERT_TRUE(sample.model.rotation_raw().is_deferred());
                        EXPECT_THROW(captureCollisionSource(sample.model), std::runtime_error);
                    } else
                        sample.captured = captureCollisionSource(sample.model, glm::mat4(1), CaptureCase::begin, CaptureCase::count);
                }
                EXPECT_EQ(cudaStreamQuery(capture_stream), cudaSuccess);
                EXPECT_EQ(cudaStreamQuery(writer), cudaSuccess);
                ASSERT_NO_FATAL_FAILURE(sample.mutate(writer));
                if (fail)
                    EXPECT_FLOAT_EQ(sample.partial_copy.to(Device::CPU).ptr<float>()[0], 2);
                else
                    sample.expectOwnedRange();
                ASSERT_EQ(cudaStreamSynchronize(writer), cudaSuccess);
            }
        EXPECT_EQ(cudaStreamDestroy(reader), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(writer), cudaSuccess);
    }
    TEST_F(CollisionGpu, AutoResolutionIsRelativeForBakedAndNodeScaleAndRejectsInvalidSources) {
        const std::vector<Gaussian> gs{{{-.13f, .07f, .11f}, {.1f, .2f, .3f}},
                                       {{.31f, -.09f, -.21f}, {.2f, .1f, .15f}}};
        const CollisionSettings settings{.05f, .1f, 0, true, true};
        float resolved = 0;
        const auto base = generateCollision(source(gs), settings, {}, {}, [&](float size) { resolved = size; });
        EXPECT_FLOAT_EQ(resolved, base.settings.voxel_size);
        auto explicit_settings = base.settings;
        explicit_settings.auto_resolution = false;
        const auto explicit_volume = generateCollision(source(gs), explicit_settings);
        EXPECT_EQ(base.words, explicit_volume.words);
        EXPECT_EQ(base.origin, explicit_volume.origin);
        EXPECT_EQ(base.dimensions, explicit_volume.dimensions);
        EXPECT_EQ(base.occupied_count, explicit_volume.occupied_count);
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
        EXPECT_THROW(generateCollision(input, settings), lfs::Exception);
        input = source({{{0, 0, 0}, {INFINITY, .1f, .1f}}});
        collision_test::expectError([&] { generateCollision(input, settings); }, "collision.geometry_nonfinite");
        input = source({{{0, 0, 0}, {.1f, .1f, .1f}}});
        EXPECT_THROW(generateCollision(input, settings, {}, [](float, const auto&) { return false; }), std::runtime_error);
    }
    using collision_test::box;
    CapsuleQuery queryVolumes(std::vector<std::shared_ptr<const CollisionVolume>> volumes, glm::mat4 transform = glm::mat4(1)) {
        return [volumes = std::move(volumes), transform](auto feet, auto up, float height, float radius) {
            return std::ranges::any_of(volumes, [&](const auto& v) { return collisionCapsuleIntersects(*v, transform, feet, up, height, radius); });
        };
    }
    void advanceFrames(WalkState& state, const WalkSettings& settings, const CapsuleQuery& query, glm::vec3 up,
                       glm::vec3 velocity, int frames, float dt, bool hold_position = false) {
        const auto grounded = state.feet;
        for (int frame = 0; frame < frames; ++frame) {
            ASSERT_NO_THROW(advanceWalk(state, settings, up, velocity, dt, query));
            ASSERT_FALSE(query(state.feet, up, settings.height, settings.radius));
            ASSERT_TRUE(state.grounded);
            if (hold_position)
                ASSERT_EQ(state.feet, grounded);
        }
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
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query), lfs::Exception);
        settleWalk(state, settings, {0, 1, 0}, query);
        EXPECT_NEAR(state.eye.y, settings.height, .001f);
        advanceFrames(state, settings, query, {0, 1, 0}, {10, 0, 0}, 500, .1f);
        EXPECT_LE(state.eye.x, .251f);
        const auto grounded = state.feet;
        advanceFrames(state, settings, query, {0, 1, 0}, {0, 0, 100}, 500, .1f, true);
        for (bool edit_geometry : {false, true}) {
            settings.height = edit_geometry ? 1.7f : 2.1f;
            transform[3].y = edit_geometry ? .1f : 0;
            EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query), lfs::Exception);
            EXPECT_EQ(state.feet, grounded);
        }
        transform[3].y = 0;
        EXPECT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query));
        WalkState unsupported{.eye = {100, 100, 100}};
        EXPECT_THROW(settleWalk(unsupported, settings, {0, 1, 0}, query), lfs::Exception);
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, [](auto, auto, float, float) { return false; }), lfs::Exception);
    }
    TEST(CollisionWalk, MinimumStepFitsButLowerCeilingBlocksLift) {
        for (float height : {1.85f, 1.75f}) {
            const auto query = queryVolumes({box({-2, -1, -2}, {4, 1, 4}), box({.5f, 0, -2}, {100, 10, 400}, .01f),
                                             box({-2, height, -2}, {400, 10, 400}, .01f)});
            WalkSettings settings;
            WalkState state{.eye = {0, 1.7f, 0}, .presentation = {0, 1.7f, 0}};
            settleWalk(state, settings, {0, 1, 0}, query);
            advanceFrames(state, settings, query, {0, 1, 0}, {.5f, 0, 0}, 50, .05f);
            advanceFrames(state, settings, query, {0, 1, 0}, {}, 500, .016f);
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
            advanceFrames(state, settings, query, up, {}, 500, .016f, velocity == glm::vec3(0));
            advanceFrames(state, settings, query, up, velocity, 200, .016f, velocity == glm::vec3(0));
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
    auto parse(std::initializer_list<const char*> args) {
        return lfs::io::args::parse_args(int(args.size()), args.begin());
    }
    TEST(VoxelizeCli, ParsesControlsAndRejectsDeferredOrInvalidOptions) {
        const auto source = PROJECT_ROOT_PATH "/tests/test_collision.cpp";
        auto result = parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--filter", "none", "--max-gaussian-area", "0"});
        ASSERT_TRUE(result) << result.error();
        const auto& mode = std::get<lfs::io::args::VoxelizeMode>(*result);
        EXPECT_TRUE(mode.settings.auto_resolution);
        EXPECT_FALSE(mode.settings.cleanup);
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--filter", "none", "--max-gaussian-area", "1"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--filter", "connected"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--voxel-size", "nan"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--voxel-size", ".01"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.bin"}));
    }
    namespace fs = lfs::core::native_fs;
    using namespace lfs::test::licht;
    struct CollisionExportTest : testing::Test {
        TemporaryDirectory temporary{"voxel_test"};
        const fs::path& directory = temporary.path;
        fs::path path = directory / "scene.voxel.json", binary = directory / "scene.voxel.bin";
        static std::string read(const fs::path& p) {
            const auto bytes = read_file_bytes(fs::native_path(p));
            return bytes.empty() ? std::string{} : std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        static void write(const fs::path& p, std::string_view text) { write_file_bytes(fs::native_path(p), std::as_bytes(std::span(text))); }
        static CollisionVolume volume(glm::ivec3 dims = {5, 3, 2}) {
            return *collision_test::box({2, 4, 6}, dims, .25f);
        }
        // Independent little-endian v1.1 decoder, with bounds checks on both pointer spaces.
        std::vector<bool> decode() {
            const auto j = nlohmann::json::parse(read(path));
            EXPECT_EQ(j.at("version"), "1.1");
            EXPECT_EQ(j.at("leafSize"), 4);
            const auto bytes = read(binary);
            const size_t nodes = j.at("nodeCount"), leaves = j.at("leafDataCount");
            EXPECT_EQ(bytes.size(), 4 * (nodes + leaves));
            EXPECT_EQ(leaves, 2 * j.at("numMixedLeaves").get<size_t>());
            auto word = [&](size_t i) {
                if (i >= bytes.size() / 4)
                    throw std::runtime_error("Decoder pointer exceeds binary");
                uint32_t value = 0;
                for (int b = 0; b < 4; ++b)
                    value |= uint32_t(uint8_t(bytes[4 * i + b])) << (8 * b);
                return value;
            };
            const int edge = 4 << j.at("treeDepth").get<int>();
            std::vector<bool> result(size_t(edge) * edge * edge);
            auto visit = [&](auto&& self, size_t index, glm::ivec3 origin, int size) -> void {
                if (index >= nodes || size < 4)
                    throw std::runtime_error("Decoder node exceeds topology bounds");
                const auto value = word(index), mask = value >> 24, pointer = value & 0xffffff;
                if (value == 0xff000000 || size == 4) {
                    if (value != 0xff000000 && size_t(pointer) * 2 + 1 >= leaves)
                        throw std::runtime_error("Decoder leaf exceeds leafDataCount");
                    for (const auto p : cells(glm::ivec3(size))) {
                        const auto cell = origin + p;
                        const int bit = p.x + 4 * p.y + 16 * p.z;
                        result[cell.x + edge * (cell.y + edge * cell.z)] = value == 0xff000000 ||
                                                                           ((word(nodes + 2 * pointer + bit / 32) >> (bit % 32)) & 1);
                    }
                } else {
                    size_t child_index = pointer;
                    for (const auto p : cells(glm::ivec3(2)))
                        if (mask & (1u << (p.x + 2 * p.y + 4 * p.z)))
                            self(self, child_index++, origin + p * (size / 2), size / 2);
                }
            };
            if (nodes)
                visit(visit, 0, {}, edge);
            return result;
        }
    };
    TEST(CollisionExportLimits, MixedLeafBoundary) {
        EXPECT_EQ(lfs::core::cuda::collisionOutputBytes(64'000'000), 64'000'000 * sizeof(lfs::core::cuda::CollisionBrick));
        for (const auto count : {size_t(64'000'000), SIZE_MAX})
            EXPECT_THROW(lfs::core::cuda::collisionOutputBytes(count, 1), lfs::Exception);
        EXPECT_NO_THROW(lfs::io::checkCollisionExportMixedBlocks(14'680'064));
        for (uint64_t count : {uint64_t(14'680'065), UINT64_MAX})
            collision_test::expectError([&] { lfs::io::checkCollisionExportMixedBlocks(count); }, "collision.output_limit");
    }
    TEST_F(CollisionExportTest, EverySignedPermutationPreservesCellsAndLogicalBlockBounds) {
        for (const auto [dims, kind] : {std::tuple{glm::ivec3(4), 0}, {glm::ivec3(4), 1}, {glm::ivec3(4), 2}, {glm::ivec3(8), 1}, {glm::ivec3(24, 16, 12), 1}, {glm::ivec3(21, 13, 9), 1}, {glm::ivec3(5, 3, 2), 2}}) {
            auto v = volume(dims);
            const auto padded = ((dims + 3) / 4) * 4;
            if (kind == 0) {
                v.words = {0};
                v.occupied_count = 0;
            } else if (kind == 2) {
                const auto q = (dims - 1) % 4;
                const uint64_t bit = uint64_t(1) << (q.x + 4 * q.y + 16 * q.z);
                // A minimum native brick, or an interior with only its +X child populated.
                v.words = dims.x == 4 ? std::vector<uint32_t>{CollisionVolume::brick_tag | 1, uint32_t(bit), uint32_t(bit >> 32)} : std::vector<uint32_t>{3, 0, CollisionVolume::brick_tag | 9, 0, 0, 0, 0, 0, 0, uint32_t(bit), uint32_t(bit >> 32)};
                v.occupied_count = 1;
            }
            auto occupied = [&](glm::ivec3 p) {
                return glm::all(glm::greaterThanEqual(p, glm::ivec3(0))) && glm::all(glm::lessThan(p, dims)) &&
                       (kind == 1 || (kind == 2 && p == dims - 1));
            };
            v.validate();
            std::array axes{0, 1, 2};
            do {
                for (int signs = 0; signs < 8; ++signs) {
                    SCOPED_TRACE(::testing::Message() << dims.x << ":" << axes[0] << axes[1] << axes[2] << ":" << signs);
                    glm::mat4 world(0);
                    world[3] = {3, 4, 5, 1};
                    for (int s = 0; s < 3; ++s)
                        world[s][axes[s]] = 2.f * ((signs & (1 << s)) ? -1 : 1) * (axes[s] < 2 ? -1 : 1);
                    lfs::io::exportCollision(v, path, world);
                    if (dims == glm::ivec3(4) && axes == std::array{0, 1, 2} && signs == 0) {
                        const std::array<std::string, 3> expected{"", std::string("\1\0\0\1\0\0\0\xff", 8),
                                                                  std::string("\1\0\0\1\0\0\0\0\0\0\0\0\0\0\0\x80", 16)};
                        EXPECT_EQ(read(binary), expected[kind]);
                    }
                    const auto decoded = decode();
                    const int edge = std::max(8, v.edge);
                    ASSERT_EQ(decoded.size(), size_t(edge) * edge * edge);
                    for (const auto p : cells(glm::ivec3(edge))) {
                        glm::ivec3 source;
                        for (int s = 0; s < 3; ++s)
                            source[s] = signs & (1 << s) ? padded[s] - 1 - p[axes[s]] : p[axes[s]];
                        ASSERT_EQ(decoded[p.x + edge * (p.y + edge * p.z)], occupied(source));
                    }
                    auto bounds = [&](glm::ivec3 size) {
                        const auto a = glm::vec3(world * glm::vec4(v.origin, 1)) * glm::vec3(-1, -1, 1);
                        const auto corner = v.origin + v.settings.voxel_size * glm::vec3(size);
                        const auto b = glm::vec3(world * glm::vec4(corner, 1)) * glm::vec3(-1, -1, 1);
                        const auto lo = glm::min(a, b), hi = glm::max(a, b);
                        return nlohmann::json{{"min", {lo.x, lo.y, lo.z}}, {"max", {hi.x, hi.y, hi.z}}};
                    };
                    const auto j = nlohmann::json::parse(read(path));
                    EXPECT_EQ(j["gridBounds"], bounds(padded));
                    EXPECT_EQ(j["sceneBounds"], bounds(dims));
                    EXPECT_EQ(j["voxelResolution"], .5);
                }
            } while (std::next_permutation(axes.begin(), axes.end()));
        }
    }
    TEST_F(CollisionExportTest, InvalidTransformAndPairInstallFailuresPreserveDestinations) {
        for (bool previous : {false, true}) {
            if (previous) {
                write(path, "old-json");
                write(binary, "old-bin");
            }
            auto preserved = [&] {
                EXPECT_EQ(fs::exists(path), previous);
                EXPECT_EQ(fs::exists(binary), previous);
                if (previous) {
                    EXPECT_EQ(read(path), "old-json");
                    EXPECT_EQ(read(binary), "old-bin");
                }
                EXPECT_EQ(std::distance(fs::directory_iterator(directory), std::filesystem::directory_iterator{}), previous ? 2 : 0);
            };
            for (const auto [column, row, value] : {std::tuple{0, 0, 2.f}, {0, 1, .2f}, {0, 3, .1f}, {3, 0, INFINITY}}) {
                glm::mat4 transform(1);
                transform[column][row] = value;
                EXPECT_THROW(lfs::io::exportCollision(volume(), path, transform), lfs::Exception);
                preserved();
            }
            for (bool cancel : {false, true}) {
                auto inject = [&](float p, const std::string&) {
                    if (p != .95f)
                        return true;
                    if (cancel)
                        return false;
                    for (const auto& entry : fs::directory_iterator(directory))
                        if (entry.path() != path && read(entry.path()).starts_with('{'))
                            fs::remove(entry.path());
                    return true;
                };
                if (cancel)
                    EXPECT_THROW(lfs::io::exportCollision(volume(), path, glm::mat4(1), {}, inject), lfs::core::OperationCancelled);
                else
                    collision_test::expectError([&] { lfs::io::exportCollision(volume(), path, glm::mat4(1), {}, inject); }, "collision.export_publish");
                preserved();
            }
        }
    }
    TEST_F(CollisionExportTest, DatasetSourceCarriesLocaleKey) {
        const auto dataset = directory / "transforms.json";
        write(dataset, "{}");
        collision_test::expectError([&] { lfs::io::loadCollisionSource(dataset); }, "collision.source_unsupported");
    }
    TEST_F(CollisionGpu, PointCloudSourceCarriesLocaleKey) {
        TemporaryDirectory temporary{"collision_source"};
        const auto points = temporary.path / "points.ply";
        CollisionExportTest::write(points, "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n");
        collision_test::expectError([&] { lfs::io::loadCollisionSource(points); }, "collision.source_unsupported");
    }
    TEST_F(CollisionExportTest, SourceProtectionLongNamesAndBlockedPaths) {
        path = directory / (std::string(240, 'a') + ".voxel.json");
        lfs::io::exportCollision(volume(), path);
        lfs::io::exportCollision(volume(), path);
        const auto blocked = directory / "file";
        write(blocked, "source-preserved");
        collision_test::expectError([&] { lfs::io::loadCollisionSource(directory / "missing.ply"); }, "collision.source_load_failed");
        EXPECT_EQ(lfs::app::run_voxelizer({.input_path = blocked, .output_path = blocked, .overwrite = true}), 1);
        try {
            lfs::io::exportCollision(volume(), blocked / "out.voxel.json");
            FAIL();
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("file"), std::string::npos);
        }
        EXPECT_EQ(read(blocked), "source-preserved");
    }
} // namespace
