/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "checkpoint_fixture.hpp"
#include "core/camera.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "io/loader.hpp"
#include "io/loaders/checkpoint_loader.hpp"
#include "io/project_document.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "training/checkpoint.hpp"
#include "training/components/sparsity_optimizer.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/rasterization_api.h"
#include "training/strategies/mcmc.hpp"
#include "training/strategies/strategy_factory.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

namespace lfs::training {
    struct TrainerRetryTestAccess {
        static bool should_retry(const lfs::Error& error, const unsigned attempts) {
            const Trainer::MutationStamp stamp{
                .iteration = 1,
                .epoch = 0,
                .phase = Trainer::StepPhase::Forward,
                .persistent_commit = false,
            };
            return Trainer::classify_forward_retry(error, stamp, attempts) ==
                   Trainer::RetryDecision::RetryForwardOnce;
        }

        static lfs::Status recover_with_sync_status(
            Trainer& trainer, const lfs::Error& cause, const cudaError_t sync_status) {
            trainer.recovery_sync_for_testing_ = [sync_status] { return sync_status; };
            auto result = trainer.recover_forward_oom(cause);
            trainer.recovery_sync_for_testing_ = {};
            return result;
        }
    };
} // namespace lfs::training

namespace {

    lfs::Error make_retry_test_error(const lfs::ErrorCode code) {
        return lfs::make_error(lfs::ErrorInit{
            .code = code,
            .domain = lfs::ErrorDomain::CUDA,
            .user_message = "retry test",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

    const lfs::SmallFields::Entry* find_field(
        const lfs::ErrorFrame& frame, const std::string_view key) {
        for (const auto& entry : frame.fields.entries()) {
            if (entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    TEST(TrainerRetrySemantics, ClassificationIsTypedAndBoundedToOneRetry) {
        const auto exhausted = make_retry_test_error(lfs::ErrorCode::ResourceExhausted);
        EXPECT_TRUE(lfs::training::TrainerRetryTestAccess::should_retry(exhausted, 1));
        EXPECT_FALSE(lfs::training::TrainerRetryTestAccess::should_retry(exhausted, 2));

        const auto invalid = make_retry_test_error(lfs::ErrorCode::InvalidArgument);
        EXPECT_FALSE(lfs::training::TrainerRetryTestAccess::should_retry(invalid, 1));
    }

    TEST(TrainerRetrySemantics, InvalidDimensionsAreNotResourceExhaustion) {
        const auto context = fast_lfs::rasterization::forward_raw(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            /*bg_color_ptr=*/nullptr, /*bg_image_ptr=*/nullptr,
            0, 1, 1, 1, 1,
            1.0f, 1.0f, 0.5f, 0.5f, 0.01f, 100.0f, false, nullptr);
        EXPECT_FALSE(context.success);
        EXPECT_FALSE(context.resource_exhausted);

        const auto typed = make_retry_test_error(lfs::ErrorCode::InvalidArgument);
        EXPECT_FALSE(lfs::training::TrainerRetryTestAccess::should_retry(typed, 1));
    }

    TEST(TrainerRetrySemantics, UninitializedTrainErrorCarriesCompleteMutationStamp) {
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        auto camera = std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0);
        scene.addCamera("camera.png", cameras, std::move(camera));
        lfs::training::Trainer trainer(scene);
        auto result = trainer.train();
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::FailedPrecondition);
        ASSERT_GE(result.error().frames().size(), 2u);

        const auto& frame = result.error().frames().back();
        EXPECT_EQ(frame.operation, "train");
        ASSERT_NE(find_field(frame, "iteration"), nullptr);
        ASSERT_NE(find_field(frame, "mutation_epoch"), nullptr);
        ASSERT_NE(find_field(frame, "step_phase"), nullptr);
        ASSERT_NE(find_field(frame, "persistent_commit"), nullptr);
        EXPECT_EQ(std::get<std::string>(find_field(frame, "step_phase")->value),
                  "AcquireData");
        EXPECT_FALSE(std::get<bool>(find_field(frame, "persistent_commit")->value));
    }

    TEST(TrainerRetrySemantics, GlobalArenaCanBeReconfiguredForCapacityInjection) {
        lfs::core::RasterizerMemoryArena::Config config;
        config.virtual_size = 128ULL << 20;
        config.max_physical = 64ULL << 20;
        config.granularity = 64ULL << 20;
        config.enable_vmm = false;

        auto& manager = lfs::core::GlobalArenaManager::instance();
        manager.reconfigure_for_testing(config);
        EXPECT_NE(manager.try_get_arena(), nullptr);
        manager.reset();
    }

    TEST(TrainerRetrySemantics, CapacityFailureIsRetryableOnlyOnFirstAttempt) {
        void* storage = nullptr;
        ASSERT_EQ(cudaMalloc(&storage, 4096), cudaSuccess);

        lfs::core::RasterizerMemoryArena::Config config;
        config.virtual_size = 128ULL << 20;
        config.max_physical = 64ULL << 20;
        config.granularity = 64ULL << 20;
        config.enable_vmm = false;
        auto& manager = lfs::core::GlobalArenaManager::instance();
        manager.reconfigure_for_testing(config);

        const auto attempt = [storage] {
            auto* ptr = static_cast<float*>(storage);
            return fast_lfs::rasterization::forward_raw(
                ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                ptr, ptr, ptr, nullptr,
                /*bg_color_ptr=*/nullptr, /*bg_image_ptr=*/nullptr,
                10'000'000, 1, 1, 1, 1,
                1.0f, 1.0f, 0.5f, 0.5f, 0.01f, 100.0f, false, nullptr);
        };

        const auto first = attempt();
        EXPECT_FALSE(first.success);
        EXPECT_TRUE(first.resource_exhausted)
            << (first.error_message ? first.error_message : "no error message");
        const auto exhausted = make_retry_test_error(lfs::ErrorCode::ResourceExhausted);
        EXPECT_TRUE(lfs::training::TrainerRetryTestAccess::should_retry(exhausted, 1));

        const auto second = attempt();
        EXPECT_FALSE(second.success);
        EXPECT_TRUE(second.resource_exhausted)
            << (second.error_message ? second.error_message : "no error message");
        EXPECT_FALSE(lfs::training::TrainerRetryTestAccess::should_retry(exhausted, 2));

        manager.reset();
        EXPECT_EQ(cudaFree(storage), cudaSuccess);
    }

    TEST(TrainerRetrySemantics, RecoveryAsyncFailureKeepsExactlyOneSuppressedOom) {
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        auto camera = std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0);
        scene.addCamera("camera.png", cameras, std::move(camera));
        lfs::training::Trainer trainer(scene);
        const auto cause = make_retry_test_error(lfs::ErrorCode::ResourceExhausted);

        auto result = lfs::training::TrainerRetryTestAccess::recover_with_sync_status(
            trainer, cause, cudaErrorIllegalAddress);
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().code(), lfs::ErrorCode::ResourceExhausted);
        ASSERT_EQ(result.error().suppressed().size(), 1u);
        EXPECT_EQ(result.error().suppressed().front().code(),
                  lfs::ErrorCode::ResourceExhausted);
    }

    constexpr const char* TEST_IMAGES = "images_4";
    std::unique_ptr<lfs::core::SplatData> make_checkpoint_test_splat(
        const size_t count,
        const lfs::core::Device device = lfs::core::Device::CPU,
        const int max_sh_degree = 0) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3] = static_cast<float>(i);
            rotations[i * 4] = 1.0f;
        }

        const size_t rest = max_sh_degree > 0
                                ? static_cast<size_t>(max_sh_degree * (max_sh_degree + 2))
                                : size_t{0};
        auto shN = rest == 0
                       ? lfs::core::Tensor::zeros({size_t{0}}, device, lfs::core::DataType::Float32)
                       : lfs::core::Tensor::zeros({count, rest, size_t{3}}, device, lfs::core::DataType::Float32);
        if (rest > 0 && shN.is_valid()) {
            auto cpu = shN.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < count * rest * 3; ++i)
                p[i] = 0.01f * static_cast<float>((i % 17) + 1);
            shN = cpu.to(device);
        }

        return std::make_unique<lfs::core::SplatData>(
            max_sh_degree,
            lfs::core::Tensor::from_vector(means, {count, size_t{3}}, device),
            lfs::core::Tensor::zeros({count, size_t{1}, size_t{3}}, device, lfs::core::DataType::Float32),
            std::move(shN),
            lfs::core::Tensor::zeros({count, size_t{3}}, device, lfs::core::DataType::Float32),
            lfs::core::Tensor::from_vector(rotations, {count, size_t{4}}, device),
            lfs::core::Tensor::zeros({count, size_t{1}}, device, lfs::core::DataType::Float32),
            1.0f);
    }

    void add_checkpoint_test_camera(lfs::core::Scene& scene) {
        const auto cameras = scene.addGroup("Cameras");
        auto camera = std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f,
            100.0f,
            32.0f,
            32.0f,
            lfs::core::Tensor{},
            lfs::core::Tensor{},
            lfs::core::CameraModelType::PINHOLE,
            "camera.png",
            std::filesystem::path{},
            std::filesystem::path{},
            64,
            64,
            0);
        scene.addCamera("camera.png", cameras, std::move(camera));
    }

    std::streamoff first_model_tensor_header_offset(const std::filesystem::path& checkpoint) {
        std::ifstream file(checkpoint, std::ios::binary);
        if (!file)
            return -1;
        file.seekg(static_cast<std::streamoff>(sizeof(lfs::core::CheckpointHeader)));
        uint32_t strategy_name_size = 0;
        file.read(reinterpret_cast<char*>(&strategy_name_size), sizeof(strategy_name_size));
        file.seekg(static_cast<std::streamoff>(strategy_name_size), std::ios::cur);
        constexpr std::streamoff splat_prefix_bytes =
            sizeof(uint32_t) * 2 + sizeof(int32_t) * 2 + sizeof(float);
        file.seekg(splat_prefix_bytes, std::ios::cur);
        return file ? static_cast<std::streamoff>(file.tellg()) : -1;
    }

    template <typename T>
    bool overwrite_checkpoint_field(const std::filesystem::path& checkpoint,
                                    const std::streamoff offset,
                                    const T& value) {
        std::fstream file(checkpoint, std::ios::binary | std::ios::in | std::ios::out);
        if (!file)
            return false;
        file.seekp(offset);
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        file.flush();
        return file.good();
    }

    bool write_probe_fixture(const std::filesystem::path& path, const void* data, const size_t size) {
        std::ofstream file(path, std::ios::binary);
        if (!file)
            return false;
        if (size > 0)
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        file.flush();
        return file.good();
    }

    TEST(CheckpointVersionTest, VersionOneWithoutSparsityLoads) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_v1_load";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 20;
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = 16;
        auto source_model = make_checkpoint_test_splat(3);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 4, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        const auto checkpoint = lfs::test::checkpoint_fixture_path(temp_dir);
        ASSERT_TRUE(overwrite_checkpoint_field(
            checkpoint,
            static_cast<std::streamoff>(offsetof(lfs::core::CheckpointHeader, version)),
            lfs::core::CHECKPOINT_MIN_SUPPORTED_VERSION));

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        const auto loaded = lfs::training::load_checkpoint(
            checkpoint, target_strategy, params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        EXPECT_EQ(*loaded, 4);
        EXPECT_EQ(target_strategy.get_model().size(), 3);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointVersionTest, VersionOneRejectsSparsityFlag) {
        lfs::core::CheckpointHeader header;
        header.version = 1;
        header.flags = lfs::core::CheckpointFlags::HAS_SPARSITY;

        const auto result = lfs::core::validate_checkpoint_header(header, sizeof(header));
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("unknown feature flags"), std::string::npos);
    }

    TEST(CheckpointVersionTest, FutureVersionIsRejected) {
        lfs::core::CheckpointHeader header;
        header.version = lfs::core::CHECKPOINT_VERSION + 1;

        const auto result = lfs::core::validate_checkpoint_header(header, sizeof(header));
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(
            result.error().find("Unsupported version: " + std::to_string(header.version)),
            std::string::npos);
    }

    TEST(CheckpointVersionTest, UnknownFlagsAreRejectedForCurrentVersion) {
        lfs::core::CheckpointHeader header;
        header.flags = static_cast<lfs::core::CheckpointFlags>(1u << 31);

        const auto result = lfs::core::validate_checkpoint_header(header, sizeof(header));
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("unknown feature flags"), std::string::npos);
    }
    TEST(TrainingSetupRegressionTest, ApplyLoadedDatasetKeepsFullInitPointCloudUntilTrainingStarts) {
        constexpr size_t initial_points = 12;
        constexpr int target_splats = 5;

        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_training_setup_full_init_regression";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir);

        const auto init_path = temp_dir / "init_points.ply";
        {
            std::ofstream ply(init_path);
            ASSERT_TRUE(ply.is_open());
            ply << "ply\n"
                   "format ascii 1.0\n"
                   "element vertex "
                << initial_points
                << "\n"
                   "property float x\n"
                   "property float y\n"
                   "property float z\n"
                   "property uchar red\n"
                   "property uchar green\n"
                   "property uchar blue\n"
                   "end_header\n";
            for (size_t i = 0; i < initial_points; ++i) {
                ply << static_cast<float>(i) << ' '
                    << static_cast<float>(i % 3) << ' '
                    << static_cast<float>(i % 5) << ' '
                    << static_cast<int>(10 + i) << ' '
                    << static_cast<int>(20 + i) << ' '
                    << static_cast<int>(30 + i) << '\n';
            }
        }

        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = temp_dir / "dataset";
        params.init_path = lfs::core::path_to_utf8(init_path);
        params.optimization.max_cap = target_splats;

        lfs::io::LoadedScene loaded_scene;
        loaded_scene.cameras.push_back(std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "test.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0));

        lfs::io::LoadResult load_result;
        load_result.data = std::move(loaded_scene);
        load_result.scene_center = lfs::core::Tensor::from_vector(
            std::vector<float>{0.0f, 0.0f, 0.0f},
            {size_t{3}},
            lfs::core::Device::CPU);
        load_result.loader_used = "test";

        lfs::core::Scene scene;
        auto apply_result = lfs::training::applyLoadResultToScene(params, scene, std::move(load_result));
        ASSERT_TRUE(apply_result.has_value()) << apply_result.error();

        const auto* model = scene.getTrainingModel();
        ASSERT_NE(model, nullptr);
        EXPECT_EQ(static_cast<size_t>(model->size()), initial_points);
        EXPECT_EQ(scene.getTrainingModelGaussianCount(), initial_points);

        auto trainer = std::make_unique<lfs::training::Trainer>(scene);
        auto init_result = trainer->initialize(params);
        ASSERT_TRUE(init_result.has_value()) << init_result.error();

        EXPECT_EQ(static_cast<size_t>(trainer->get_strategy().get_model().size()), static_cast<size_t>(target_splats));
        EXPECT_EQ(scene.getTrainingModelGaussianCount(), static_cast<size_t>(target_splats));

        trainer->shutdown();
        std::filesystem::remove_all(temp_dir, ec);
    }

    // joint Adam + optional q16 shN must survive save→load resume with both
    // codec modes. Round-trips moments (joint_bits/packed) and dequantised shN.
    TEST(CheckpointResumeRoundtripTest, JointCodecAndQ16ShN) {
        namespace sh_value = lfs::training::sh_value;

        sh_value::set_sh_value_quant_enabled_for_testing(true);

        constexpr size_t count = 32;
        constexpr size_t max_cap = 64;
        constexpr int sh_degree = 1;

        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_ckpt_joint_q16_roundtrip";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.max_cap = static_cast<int>(max_cap);
        params.freeze_lr_scale = 0.25f;

        auto source_model = make_checkpoint_test_splat(count, lfs::core::Device::CUDA, sh_degree);
        ASSERT_TRUE(sh_value::apply_shN_value_quant(*source_model));
        EXPECT_TRUE(source_model->shN_value_quantized() ||
                    source_model->shN_raw().dtype() == lfs::core::DataType::Float16);

        lfs::training::MCMC source_strategy(*source_model);
        source_strategy.initialize(params.optimization);
        source_strategy.get_optimizer().set_frozen_lr_scale(params.freeze_lr_scale);

        // Touch one joint state so step_count is non-zero and packed tensors are live.
        auto* means_state = source_strategy.get_optimizer().get_state_mutable(
            lfs::training::ParamType::Means);
        ASSERT_NE(means_state, nullptr);
        ASSERT_TRUE(means_state->is_joint()) << "expected joint codec moments";
        means_state->step_count = 11;
        const int expected_joint_bits = means_state->joint_bits;
        const auto packed_shape = means_state->exp_avg.shape();
        const auto bounds_shape = means_state->joint_bounds.shape();

        // Seed a non-trivial packed pattern so resume is not a pure zero-state check.
        {
            auto packed_cpu = means_state->exp_avg.cpu();
            auto* bytes = packed_cpu.ptr<uint8_t>();
            for (size_t i = 0; i < packed_cpu.numel(); ++i)
                bytes[i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
            means_state->exp_avg = packed_cpu.cuda();
            auto bounds_cpu = means_state->joint_bounds.cpu();
            auto* b = bounds_cpu.ptr<float>();
            for (size_t i = 0; i < bounds_cpu.numel(); ++i)
                b[i] = (i % 2 == 0) ? -0.5f : 0.5f;
            means_state->joint_bounds = bounds_cpu.cuda();
        }

        auto shN_before = source_model->shN_canonical_cpu();

        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 42, source_strategy, params,
                        nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        auto target_model = make_checkpoint_test_splat(1, lfs::core::Device::CUDA, sh_degree);
        lfs::training::MCMC target_strategy(*target_model);
        target_strategy.initialize(params.optimization);
        auto load_params = params;
        const auto load_result = lfs::training::load_checkpoint(
            lfs::test::checkpoint_fixture_path(temp_dir),
            target_strategy, load_params,
            nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(load_result.has_value()) << load_result.error();
        EXPECT_EQ(*load_result, 42);
        EXPECT_EQ(static_cast<size_t>(target_strategy.get_model().size()), count);

        const auto& restored_opt = target_strategy.get_optimizer();
        // freeze_lr_scale is adopted with optimizer state
        // (set on the loaded optimizer before adopt).
        const auto* restored_means = restored_opt.get_state(lfs::training::ParamType::Means);
        ASSERT_NE(restored_means, nullptr);
        EXPECT_TRUE(restored_means->is_joint());
        EXPECT_EQ(restored_means->joint_bits, expected_joint_bits);
        EXPECT_EQ(restored_means->step_count, 11);
        EXPECT_EQ(restored_means->exp_avg.shape(), packed_shape);
        EXPECT_EQ(restored_means->joint_bounds.shape(), bounds_shape);

        {
            auto got = restored_means->exp_avg.cpu();
            auto* bytes = got.ptr<uint8_t>();
            size_t mismatches = 0;
            for (size_t i = 0; i < got.numel(); ++i) {
                if (bytes[i] != static_cast<uint8_t>((i * 17 + 3) & 0xff))
                    ++mismatches;
            }
            EXPECT_EQ(mismatches, 0u) << "joint packed moments did not round-trip";
        }

        auto shN_after = target_strategy.get_model().shN_canonical_cpu();
        ASSERT_EQ(shN_before.shape(), shN_after.shape());
        {
            const auto* a = shN_before.ptr<float>();
            const auto* b = shN_after.ptr<float>();
            double max_abs = 0.0;
            for (size_t i = 0; i < shN_before.numel(); ++i)
                max_abs = std::max(max_abs, std::abs(static_cast<double>(a[i] - b[i])));
            // q16 encode → disk dequant → re-quant on resume is lossy; tolerate
            // a small absolute error on the synthetic 0.01-scale SH values.
            EXPECT_LT(max_abs, 0.15) << "shN q16/canonical round-trip error too high";
        }

        sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointAllocatorRegressionTest, LoadCheckpointUsesAllocatorWithMaxCapacity) {
        constexpr size_t count = 4;
        constexpr size_t max_cap = 16;

        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_allocator_regression";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.max_cap = max_cap;

        auto source_model = make_checkpoint_test_splat(count);
        lfs::training::MCMC source_strategy(*source_model);
        auto save_result = lfs::test::write_checkpoint_fixture(
            temp_dir, 7, source_strategy, params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(save_result.has_value()) << save_result.error();

        struct AllocationCall {
            std::string name;
            size_t capacity = 0;
        };
        std::vector<AllocationCall> calls;
        lfs::core::SplatTensorAllocator allocator =
            [&calls](lfs::core::TensorShape shape,
                     const size_t capacity,
                     const lfs::core::DataType dtype,
                     const std::string_view name) -> lfs::core::Tensor {
            calls.push_back({std::string{name}, capacity});
            EXPECT_EQ(dtype, lfs::core::DataType::Float32);
            auto tensor = lfs::core::Tensor::zeros_direct(std::move(shape), capacity, lfs::core::Device::CUDA);
            tensor.set_name(std::string{name});
            return tensor;
        };

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        auto load_params = params;
        const auto checkpoint_path = lfs::test::checkpoint_fixture_path(temp_dir);
        auto load_result = lfs::training::load_checkpoint(
            checkpoint_path, target_strategy, load_params, nullptr, nullptr, nullptr, nullptr, allocator);
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        EXPECT_EQ(*load_result, 7);
        EXPECT_EQ(static_cast<size_t>(target_strategy.get_model().size()), count);
        EXPECT_GE(target_strategy.get_model().means_raw().capacity(), max_cap);
        EXPECT_EQ(calls.size(), 5u);
        for (const auto& call : calls) {
            EXPECT_GE(call.capacity, max_cap) << call.name;
        }

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointInputValidationTest, RejectsInvalidTensorDtypeAndPreservesLiveModel) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_invalid_tensor_dtype";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.max_cap = 16;

        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 7, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        const auto checkpoint = lfs::test::checkpoint_fixture_path(temp_dir);
        const auto tensor_offset = first_model_tensor_header_offset(checkpoint);
        ASSERT_GE(tensor_offset, 0);
        constexpr uint8_t invalid_dtype = 0xff;
        ASSERT_TRUE(overwrite_checkpoint_field(
            checkpoint,
            tensor_offset + static_cast<std::streamoff>(offsetof(lfs::core::TensorFileHeader, dtype)),
            invalid_dtype));

        auto target_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC target_strategy(*target_model);
        auto loaded_params = params;
        const auto result = lfs::training::load_checkpoint(
            checkpoint, target_strategy, loaded_params, nullptr, nullptr, nullptr, nullptr);

        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("unsupported dtype"), std::string::npos);
        ASSERT_EQ(target_strategy.get_model().size(), 2);
        const auto means = target_strategy.get_model().means().cpu().to_vector();
        ASSERT_EQ(means.size(), 6u);
        EXPECT_FLOAT_EQ(means[3], 1.0f);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointInputValidationTest, RejectsLateStrategyCorruptionWithoutPartialCommit) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_late_corruption";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.max_cap = 16;

        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);
        source_strategy.initialize(params.optimization);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 9, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        const auto checkpoint = lfs::test::checkpoint_fixture_path(temp_dir);
        const auto header = lfs::core::load_checkpoint_header(checkpoint);
        ASSERT_TRUE(header.has_value()) << header.error();
        ASSERT_GT(header->params_json_offset, 0u);
        constexpr uint8_t invalid_scheduler_parameter = 0xff;
        ASSERT_TRUE(overwrite_checkpoint_field(
            checkpoint,
            static_cast<std::streamoff>(header->params_json_offset - 1),
            invalid_scheduler_parameter));

        auto target_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC target_strategy(*target_model);
        target_strategy.initialize(params.optimization);
        target_strategy.get_optimizer().set_lr(0.123f);
        auto loaded_params = params;

        const auto result = lfs::training::load_checkpoint(
            checkpoint, target_strategy, loaded_params, nullptr, nullptr, nullptr, nullptr);

        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("invalid parameter id"), std::string::npos);
        EXPECT_EQ(target_strategy.get_model().size(), 2);
        EXPECT_FLOAT_EQ(target_strategy.get_optimizer().get_lr(), 0.123f);
        EXPECT_EQ(loaded_params.optimization.max_cap, params.optimization.max_cap);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointInputValidationTest, RejectsJsonRangeOutsideFileBeforeStateMutation) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_invalid_json_range";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        auto source_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 3, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        const auto checkpoint = lfs::test::checkpoint_fixture_path(temp_dir);
        constexpr uint64_t oversized_json = lfs::core::MAX_CHECKPOINT_JSON_BYTES + 1;
        ASSERT_TRUE(overwrite_checkpoint_field(
            checkpoint,
            static_cast<std::streamoff>(offsetof(lfs::core::CheckpointHeader, params_json_size)),
            oversized_json));

        const auto header = lfs::core::load_checkpoint_header(checkpoint);
        ASSERT_FALSE(header.has_value());
        EXPECT_NE(header.error().find("JSON exceeds byte budget"), std::string::npos);

        std::filesystem::remove_all(temp_dir, ec);
    }

    class CheckpointLoaderShortProbeTest : public ::testing::TestWithParam<size_t> {};

    TEST_P(CheckpointLoaderShortProbeTest, RejectsTruncatedMagicDeterministically) {
        const auto short_bytes = GetParam();
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_loader_short_probe";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir);

        const auto probe_path = temp_dir / std::format("truncated_{}.resume", short_bytes);
        ASSERT_TRUE(write_probe_fixture(probe_path, &lfs::core::CHECKPOINT_MAGIC, short_bytes));

        lfs::io::CheckpointLoader loader;
        EXPECT_FALSE(loader.canLoad(probe_path));

        std::filesystem::remove_all(temp_dir, ec);
    }

    INSTANTIATE_TEST_SUITE_P(
        ByteLength,
        CheckpointLoaderShortProbeTest,
        ::testing::Values(size_t{0}, size_t{1}, size_t{2}, size_t{3}));

    TEST(CheckpointLoaderProbeTest, AcceptsValidFourByteMagicPrefix) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_loader_valid_probe";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir);

        const auto probe_path = temp_dir / "valid_magic.resume";
        ASSERT_TRUE(write_probe_fixture(
            probe_path, &lfs::core::CHECKPOINT_MAGIC, sizeof(lfs::core::CHECKPOINT_MAGIC)));

        lfs::io::CheckpointLoader loader;
        EXPECT_TRUE(loader.canLoad(probe_path));

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointLoaderProbeTest, RejectsWrongFourByteMagic) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_loader_wrong_probe";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir);

        const auto probe_path = temp_dir / "wrong_magic.resume";
        constexpr uint32_t wrong_magic = ~lfs::core::CHECKPOINT_MAGIC;
        ASSERT_TRUE(write_probe_fixture(probe_path, &wrong_magic, sizeof(wrong_magic)));

        lfs::io::CheckpointLoader loader;
        EXPECT_FALSE(loader.canLoad(probe_path));

        std::filesystem::remove_all(temp_dir, ec);
    }

    class CheckpointStrategyStateRoundTripTest : public ::testing::TestWithParam<std::string> {};

    TEST_P(CheckpointStrategyStateRoundTripTest, ModelOptimizerAndStrategyState) {
        const auto& strategy_name = GetParam();
        const auto temp_dir = std::filesystem::temp_directory_path() /
                              std::format("lfs_checkpoint_state_{}", strategy_name);
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = strategy_name;
        params.optimization.iterations = 20;
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = 16;

        const auto model_device = strategy_name == "igs+"
                                      ? lfs::core::Device::CUDA
                                      : lfs::core::Device::CPU;
        auto source_model = make_checkpoint_test_splat(4, model_device);
        auto source_result = lfs::training::StrategyFactory::instance().create(
            strategy_name, *source_model);
        ASSERT_TRUE(source_result.has_value()) << source_result.error();
        auto source = std::move(*source_result);
        source->initialize(params.optimization);
        source->get_optimizer().set_lr(0.0123f);

        auto save_result = lfs::test::write_checkpoint_fixture(
            temp_dir, 11, *source, params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(save_result.has_value()) << save_result.error();

        auto target_model = make_checkpoint_test_splat(1, model_device);
        auto target_result = lfs::training::StrategyFactory::instance().create(
            strategy_name, *target_model);
        ASSERT_TRUE(target_result.has_value()) << target_result.error();
        auto target = std::move(*target_result);
        target->initialize(params.optimization);
        auto loaded_params = params;
        const auto load_result = lfs::training::load_checkpoint(
            lfs::test::checkpoint_fixture_path(temp_dir),
            *target, loaded_params, nullptr, nullptr, nullptr, nullptr);

        ASSERT_TRUE(load_result.has_value()) << load_result.error();
        EXPECT_EQ(*load_result, 11);
        EXPECT_EQ(target->strategy_type(), strategy_name);
        EXPECT_EQ(target->get_model().size(), 4);
        EXPECT_EQ(target->get_model().means().cpu().to_vector(),
                  source->get_model().means().cpu().to_vector());
        EXPECT_FLOAT_EQ(target->get_optimizer().get_lr(), 0.0123f);
        EXPECT_EQ(loaded_params.optimization.strategy, strategy_name);

        std::filesystem::remove_all(temp_dir, ec);
    }

    INSTANTIATE_TEST_SUITE_P(
        CheckpointStrategies,
        CheckpointStrategyStateRoundTripTest,
        ::testing::Values("mcmc", "mrnf", "igs+"),
        [](const ::testing::TestParamInfo<std::string>& info) {
            auto name = info.param;
            std::replace_if(
                name.begin(), name.end(), [](const unsigned char c) { return !std::isalnum(c); }, '_');
            return name;
        });

    TEST(SplatDataFrozenRangesTest, FrozenRangesSurviveSerializeRoundTrip) {
        auto model = make_checkpoint_test_splat(6);
        model->set_frozen_ranges({{1, 2}, {4, 1}});

        std::stringstream stream;
        model->serialize(stream);

        lfs::core::SplatData loaded;
        loaded.deserialize(stream, {});

        ASSERT_EQ(loaded.size(), 6);
        const auto& ranges = loaded.frozen_ranges();
        ASSERT_EQ(ranges.size(), 2u);
        EXPECT_EQ(ranges[0].start, 1u);
        EXPECT_EQ(ranges[0].count, 2u);
        EXPECT_EQ(ranges[1].start, 4u);
        EXPECT_EQ(ranges[1].count, 1u);
    }

    TEST(SplatDataFrozenRangesTest, Version3StreamLoadsWithEmptyRanges) {
        auto model = make_checkpoint_test_splat(4);
        std::stringstream v4_stream;
        model->serialize(v4_stream);

        // A v4 stream with no frozen ranges is a v3 stream plus one trailing flag byte.
        auto bytes = v4_stream.str();
        ASSERT_GT(bytes.size(), sizeof(uint32_t) * 2);
        ASSERT_EQ(bytes.back(), '\0');
        constexpr uint32_t v3 = 3;
        std::memcpy(bytes.data() + sizeof(uint32_t), &v3, sizeof(v3));
        bytes.pop_back();

        std::stringstream v3_stream(bytes);
        lfs::core::SplatData loaded;
        loaded.set_frozen_ranges({{0, 1}});
        loaded.deserialize(v3_stream, {});

        ASSERT_EQ(loaded.size(), 4);
        EXPECT_TRUE(loaded.frozen_ranges().empty());
    }

    TEST(SplatDataFrozenRangesTest, RejectsFrozenRangeBeyondGaussianCount) {
        auto model = make_checkpoint_test_splat(4);
        model->set_frozen_ranges({{3, 1}});

        std::stringstream stream;
        model->serialize(stream);

        auto bytes = stream.str();
        ASSERT_GE(bytes.size(), sizeof(uint64_t));
        constexpr uint64_t corrupted_count = 2;
        std::memcpy(bytes.data() + bytes.size() - sizeof(corrupted_count),
                    &corrupted_count,
                    sizeof(corrupted_count));

        std::stringstream corrupted(bytes);
        lfs::core::SplatData loaded;
        EXPECT_THROW(loaded.deserialize(corrupted, {}), std::runtime_error);
    }

    TEST(SplatDataFrozenRangesTest, RejectsInvalidRangesBeforeWriting) {
        const auto expect_rejected = [](std::vector<lfs::core::SplatData::FrozenRange> ranges) {
            auto model = make_checkpoint_test_splat(4);
            model->set_frozen_ranges(std::move(ranges));

            std::stringstream stream;
            EXPECT_THROW(model->serialize(stream), std::runtime_error);
            EXPECT_TRUE(stream.str().empty());
        };

        expect_rejected({{0, 0}});
        expect_rejected({{4, 1}});
        expect_rejected({{0, 2}, {1, 2}});
    }

    lfs::core::param::TrainingParameters make_params_json_test_params(
        const std::filesystem::path& output_path) {
        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = output_path;
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 20;
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = 16;
        return params;
    }

    TEST(CheckpointFrozenRangesRoundTripTest, EmbeddedSplatRangesSurviveFullCheckpoint) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_frozen_ranges";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        auto params = make_params_json_test_params(temp_dir);
        auto source_model = make_checkpoint_test_splat(6);
        source_model->set_frozen_ranges({{1, 2}, {4, 1}});
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        const auto loaded = lfs::training::load_checkpoint(
            lfs::test::checkpoint_fixture_path(temp_dir),
            target_strategy,
            params,
            nullptr,
            nullptr,
            nullptr,
            nullptr);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();

        const auto& ranges = target_strategy.get_model().frozen_ranges();
        ASSERT_EQ(target_strategy.get_model().size(), 6);
        ASSERT_EQ(ranges.size(), 2u);
        EXPECT_EQ(ranges[0].start, 1u);
        EXPECT_EQ(ranges[0].count, 2u);
        EXPECT_EQ(ranges[1].start, 4u);
        EXPECT_EQ(ranges[1].count, 1u);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(TrainerCheckpointFrozenMaskTest, LoadedRangesReplacePreexistingOptimizerMask) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_trainer_checkpoint_frozen_mask";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        auto params = make_params_json_test_params(temp_dir);
        params.dataset.data_path = temp_dir;
        auto source_model = make_checkpoint_test_splat(4, lfs::core::Device::CUDA);
        source_model->set_frozen_ranges({{1, 2}});
        lfs::training::MCMC source_strategy(*source_model);
        source_strategy.initialize(params.optimization);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        lfs::core::Scene scene;
        add_checkpoint_test_camera(scene);
        auto target_model = make_checkpoint_test_splat(4, lfs::core::Device::CUDA);
        target_model->set_frozen_ranges({{0, 1}});
        scene.addSplat("Model", std::move(target_model));
        scene.setTrainingModelNode("Model");

        lfs::training::Trainer trainer(scene);
        const auto initialized = trainer.initialize(params);
        ASSERT_TRUE(initialized.has_value()) << initialized.error();
        EXPECT_EQ(trainer.get_strategy().get_optimizer().frozen_mask().cpu().to_vector_bool(),
                  (std::vector<bool>{true, false, false, false}));

        const auto loaded = trainer.load_checkpoint(lfs::test::checkpoint_fixture_path(temp_dir));
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        const auto& loaded_model = trainer.get_strategy().get_model();
        ASSERT_EQ(loaded_model.frozen_ranges().size(), 1u);
        EXPECT_EQ(loaded_model.frozen_ranges()[0].start, 1u);
        EXPECT_EQ(loaded_model.frozen_ranges()[0].count, 2u);
        EXPECT_EQ(trainer.get_strategy().get_optimizer().frozen_mask().cpu().to_vector_bool(),
                  (std::vector<bool>{false, true, true, false}));

        trainer.shutdown();
        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(TrainerCheckpointFrozenMaskTest, EmptyLoadedRangesClearPreexistingOptimizerMask) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_trainer_checkpoint_empty_frozen_mask";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        auto params = make_params_json_test_params(temp_dir);
        params.dataset.data_path = temp_dir;
        auto source_model = make_checkpoint_test_splat(4, lfs::core::Device::CUDA);
        lfs::training::MCMC source_strategy(*source_model);
        source_strategy.initialize(params.optimization);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        lfs::core::Scene scene;
        add_checkpoint_test_camera(scene);
        auto target_model = make_checkpoint_test_splat(4, lfs::core::Device::CUDA);
        target_model->set_frozen_ranges({{0, 1}});
        scene.addSplat("Model", std::move(target_model));
        scene.setTrainingModelNode("Model");

        lfs::training::Trainer trainer(scene);
        const auto initialized = trainer.initialize(params);
        ASSERT_TRUE(initialized.has_value()) << initialized.error();
        ASSERT_TRUE(trainer.get_strategy().get_optimizer().frozen_mask().is_valid());

        const auto loaded = trainer.load_checkpoint(lfs::test::checkpoint_fixture_path(temp_dir));
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        EXPECT_TRUE(trainer.get_strategy().get_model().frozen_ranges().empty());
        EXPECT_FALSE(trainer.get_strategy().get_optimizer().frozen_mask().is_valid());

        trainer.shutdown();
        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointParamsJsonTest, RejectsMismatchedSplatFreezeMetadataOnSave) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_params_mismatch_save";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);

        auto params = make_params_json_test_params(temp_dir);
        params.add_splat_paths = {"one.ply", "two.ply"};
        params.add_splat_freeze = {true};
        auto source_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC source_strategy(*source_model);

        const auto saved = lfs::test::write_checkpoint_fixture(
            temp_dir, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_FALSE(saved.has_value());
        EXPECT_NE(saved.error().find("add_splat_freeze count"), std::string::npos);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointParamsJsonTest, SplatCompositionParamsRoundTrip) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_params_json";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        auto params = make_params_json_test_params(temp_dir);
        params.view_paths = {"/tmp/lfs_view_a.ply", "/tmp/lfs_view_b.spz"};
        params.import_cameras_path = std::filesystem::path("/tmp/lfs_sparse/0");
        params.add_splat_paths = {"/tmp/lfs_statue.ply", "/tmp/lfs_pedestal.ply"};
        params.add_splat_freeze = {true, false};

        auto source_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        const auto checkpoint = lfs::test::checkpoint_fixture_path(temp_dir);
        auto loaded = lfs::core::load_checkpoint_params(checkpoint);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        EXPECT_EQ(loaded->view_paths, params.view_paths);
        ASSERT_TRUE(loaded->import_cameras_path.has_value());
        EXPECT_EQ(*loaded->import_cameras_path, *params.import_cameras_path);
        EXPECT_EQ(loaded->add_splat_paths, params.add_splat_paths);
        EXPECT_EQ(loaded->add_splat_freeze, params.add_splat_freeze);

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        auto resumed_params = make_params_json_test_params(temp_dir);
        const auto load_result = lfs::training::load_checkpoint(
            checkpoint, target_strategy, resumed_params, nullptr, nullptr, nullptr, nullptr);
        ASSERT_TRUE(load_result.has_value()) << load_result.error();
        EXPECT_EQ(resumed_params.view_paths, params.view_paths);
        ASSERT_TRUE(resumed_params.import_cameras_path.has_value());
        EXPECT_EQ(*resumed_params.import_cameras_path, *params.import_cameras_path);
        EXPECT_EQ(resumed_params.add_splat_paths, params.add_splat_paths);
        EXPECT_EQ(resumed_params.add_splat_freeze, params.add_splat_freeze);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointParamsJsonTest, RejectsMismatchedSplatFreezeMetadataOnLoad) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_params_mismatch_load";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        auto params = make_params_json_test_params(temp_dir);
        params.add_splat_paths = {"one.ply", "two.ply"};
        params.add_splat_freeze = {true, false};
        auto source_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        const auto checkpoint = lfs::test::checkpoint_fixture_path(temp_dir);
        std::fstream file(checkpoint, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file.is_open());
        std::string bytes(std::istreambuf_iterator<char>(file), {});
        constexpr std::string_view original = "[true,false]";
        constexpr std::string_view replacement = "[true      ]";
        static_assert(original.size() == replacement.size());
        const auto offset = bytes.find(original);
        ASSERT_NE(offset, std::string::npos);
        file.clear();
        file.seekp(static_cast<std::streamoff>(offset));
        file.write(replacement.data(), static_cast<std::streamsize>(replacement.size()));
        file.close();

        const auto loaded = lfs::core::load_checkpoint_params(checkpoint);
        ASSERT_FALSE(loaded.has_value());
        EXPECT_NE(loaded.error().find("add_splat_freeze count"), std::string::npos);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointParamsJsonTest, MissingSplatCompositionKeysDefaultCleanly) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_params_json_defaults";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        const auto params = make_params_json_test_params(temp_dir);
        auto source_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 5, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        auto loaded = lfs::core::load_checkpoint_params(
            lfs::test::checkpoint_fixture_path(temp_dir));
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        EXPECT_TRUE(loaded->view_paths.empty());
        EXPECT_FALSE(loaded->import_cameras_path.has_value());
        EXPECT_TRUE(loaded->add_splat_paths.empty());
        EXPECT_TRUE(loaded->add_splat_freeze.empty());

        std::filesystem::remove_all(temp_dir, ec);
    }

    constexpr lfs::training::ADMMSparsityOptimizer::Config kAdmmTestConfig{
        .sparsify_steps = 100,
        .init_rho = 0.001f,
        .prune_ratio = 0.25f,
        .update_every = 50,
        .start_iteration = 20};

    TEST(CheckpointSparsityRoundTripTest, AdmmStateRoundTripsThroughCheckpoint) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_sparsity_roundtrip";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        auto params = make_params_json_test_params(temp_dir);
        params.optimization.enable_sparsity = true;
        params.optimization.sparsify_steps = kAdmmTestConfig.sparsify_steps;
        params.optimization.init_rho = kAdmmTestConfig.init_rho;
        params.optimization.prune_ratio = kAdmmTestConfig.prune_ratio;
        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);

        lfs::training::ADMMSparsityOptimizer source_admm(kAdmmTestConfig);
        const auto opacities = lfs::core::Tensor::from_vector(
            std::vector<float>{-1.0f, 0.5f, 2.0f, -0.25f},
            {size_t{4}, size_t{1}}, lfs::core::Device::CUDA);
        ASSERT_TRUE(source_admm.initialize(opacities).has_value());
        ASSERT_TRUE(source_admm.update_state(opacities).has_value());

        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(temp_dir, 7, source_strategy, params,
                                                        nullptr, nullptr, nullptr, &source_admm)
                        .has_value());
        const auto header = lfs::core::load_checkpoint_header(
            lfs::test::checkpoint_fixture_path(temp_dir));
        ASSERT_TRUE(header.has_value()) << header.error();
        EXPECT_EQ(header->version, lfs::core::CHECKPOINT_VERSION);
        EXPECT_TRUE(lfs::core::has_flag(header->flags, lfs::core::CheckpointFlags::HAS_SPARSITY));

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        lfs::training::ADMMSparsityOptimizer target_admm({
            .sparsify_steps = 1,
            .init_rho = 0.5f,
            .prune_ratio = 0.9f,
            .update_every = 1,
            .start_iteration = 0,
        });
        auto loaded_params = params;
        const auto load_result = lfs::training::load_checkpoint(
            lfs::test::checkpoint_fixture_path(temp_dir), target_strategy, loaded_params,
            nullptr, nullptr, nullptr, &target_admm);
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        ASSERT_TRUE(target_admm.is_initialized());
        EXPECT_EQ(target_admm.state_size(), 4u);
        EXPECT_EQ(target_admm.config().sparsify_steps, kAdmmTestConfig.sparsify_steps);
        EXPECT_FLOAT_EQ(target_admm.config().init_rho, kAdmmTestConfig.init_rho);
        EXPECT_FLOAT_EQ(target_admm.config().prune_ratio, kAdmmTestConfig.prune_ratio);
        EXPECT_EQ(target_admm.config().update_every, kAdmmTestConfig.update_every);
        EXPECT_EQ(target_admm.config().start_iteration, kAdmmTestConfig.start_iteration);
        std::ostringstream source_bytes, target_bytes;
        source_admm.serialize(source_bytes);
        target_admm.serialize(target_bytes);
        EXPECT_EQ(source_bytes.str(), target_bytes.str());

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointSparsityRoundTripTest, CheckpointWithoutSparsityClearsInitializedOptimizer) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_sparsity_absent";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        const auto params = make_params_json_test_params(temp_dir);
        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 7, source_strategy, params, nullptr, nullptr, nullptr, nullptr)
                        .has_value());

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        lfs::training::ADMMSparsityOptimizer target_admm(kAdmmTestConfig);
        const auto target_opacity = lfs::core::Tensor::zeros(
            {size_t{1}, size_t{1}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        ASSERT_TRUE(target_admm.initialize(target_opacity).has_value());
        ASSERT_TRUE(target_admm.is_initialized());
        auto loaded_params = params;
        const auto load_result = lfs::training::load_checkpoint(
            lfs::test::checkpoint_fixture_path(temp_dir), target_strategy, loaded_params,
            nullptr, nullptr, nullptr, &target_admm);
        ASSERT_TRUE(load_result.has_value()) << load_result.error();
        EXPECT_FALSE(target_admm.is_initialized());

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointSparsityRoundTripTest, SparsityBlobIsConsumedWhenNoOptimizerIsProvided) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_sparsity_consume";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        auto params = make_params_json_test_params(temp_dir);
        params.optimization.enable_sparsity = true;
        params.optimization.sparsify_steps = kAdmmTestConfig.sparsify_steps;
        params.optimization.init_rho = kAdmmTestConfig.init_rho;
        params.optimization.prune_ratio = kAdmmTestConfig.prune_ratio;
        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);
        lfs::training::ADMMSparsityOptimizer source_admm(kAdmmTestConfig);
        const auto opacities = lfs::core::Tensor::zeros(
            {size_t{4}, size_t{1}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        ASSERT_TRUE(source_admm.initialize(opacities).has_value());
        ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                        temp_dir, 7, source_strategy, params, nullptr, nullptr, nullptr, &source_admm)
                        .has_value());

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        const auto loaded = lfs::training::load_checkpoint(
            lfs::test::checkpoint_fixture_path(temp_dir),
            target_strategy,
            params,
            nullptr,
            nullptr,
            nullptr,
            nullptr);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        EXPECT_EQ(target_strategy.get_model().size(), 4);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointSparsityRoundTripTest, RejectsSparsityStateWithWrongModelRowCount) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_sparsity_row_mismatch";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);

        const auto params = make_params_json_test_params(temp_dir);
        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);
        lfs::training::ADMMSparsityOptimizer source_admm(kAdmmTestConfig);
        const auto wrong_size_opacities = lfs::core::Tensor::zeros(
            {size_t{3}, size_t{1}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        ASSERT_TRUE(source_admm.initialize(wrong_size_opacities).has_value());

        const auto saved = lfs::test::write_checkpoint_fixture(
            temp_dir, 7, source_strategy, params, nullptr, nullptr, nullptr, &source_admm);
        ASSERT_FALSE(saved.has_value());
        EXPECT_NE(saved.error().find("does not match model count"), std::string::npos);

        std::filesystem::remove_all(temp_dir, ec);
    }

    class CheckpointResumeTest : public ::testing::TestWithParam<std::tuple<std::string, int, int, int>> {
    protected:
        void SetUp() override {
            const auto& [strategy, sh_degree, checkpoint_iteration, total_iterations] = GetParam();
            strategy_ = strategy;
            sh_degree_ = sh_degree;

            // Create unique output directory for this test
            output_path_ = std::filesystem::temp_directory_path() /
                           std::format("lfs_test_checkpoint_{}_{}_{}", strategy_, sh_degree_, total_iterations);
            std::filesystem::create_directories(output_path_);
            std::filesystem::create_directories(output_path_ / "checkpoints");
        }

        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(output_path_, ec);
        }

        lfs::core::param::TrainingParameters createParams(int iterations) {
            lfs::core::param::TrainingParameters params;
            params.dataset.data_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
            params.dataset.images = TEST_IMAGES;
            params.dataset.output_path = output_path_;
            params.optimization.iterations = iterations;
            params.optimization.strategy = strategy_;
            params.optimization.sh_degree = sh_degree_;
            params.optimization.headless = true;
            params.optimization.max_cap = 100000;
            params.optimization.refine_every = 100;
            const size_t stop_refine = static_cast<size_t>(iterations);
            params.optimization.start_refine = std::min<size_t>(500, stop_refine);
            params.optimization.stop_refine = stop_refine;
            return params;
        }

        std::string strategy_;
        int sh_degree_;
        std::filesystem::path output_path_;
    };

    TEST_P(CheckpointResumeTest, TrainSaveLoadResume) {
        const auto& [strategy, sh_degree, checkpoint_iter, total_iter] = GetParam();
        LOG_INFO("Testing checkpoint resume: strategy={}, sh_degree={}", strategy, sh_degree);
        const int phase_one_iterations = checkpoint_iter + 1;
        // Phase 1 publishes a project generation at the completed iteration.
        const int checkpoint_iteration = phase_one_iterations;

        // Phase 1: Write multiple project snapshots and verify the final
        // generation is the sole product output.
        {
            auto params = createParams(phase_one_iterations);
            params.optimization.save_steps = {
                static_cast<size_t>(std::max(1, checkpoint_iter / 2)),
                static_cast<size_t>(checkpoint_iter)};
            lfs::core::Scene scene;

            auto load_result = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(load_result.has_value()) << "Failed to load training data: " << load_result.error();

            auto model_result = lfs::training::initializeTrainingModel(params, scene);
            ASSERT_TRUE(model_result.has_value()) << "Failed to init model: " << model_result.error();

            auto trainer = std::make_unique<lfs::training::Trainer>(scene);
            auto init_result = trainer->initialize(params);
            ASSERT_TRUE(init_result.has_value()) << "Failed to init trainer: " << init_result.error();
            lfs::training::grant_headless_project_saves(*trainer, params);

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value())
                << "Training failed: " << lfs::format_for_developer(train_result.error());

            EXPECT_EQ(trainer->get_current_iteration(), phase_one_iterations);

            trainer->shutdown();
        }

        const auto project_path = output_path_ / "project.licht";
        ASSERT_TRUE(std::filesystem::is_regular_file(project_path))
            << "Project file not found: " << project_path;
        size_t resume_file_count = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(output_path_)) {
            if (entry.path().extension() == ".resume") {
                ++resume_file_count;
            }
        }
        EXPECT_EQ(resume_file_count, 0u);

        lfs::core::Uuid phase_one_project_uuid;
        {
            auto phase_one_reader =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(phase_one_reader)
                << lfs::format_for_developer(
                       phase_one_reader.error());
            phase_one_project_uuid =
                phase_one_reader->superblock()
                    .project_uuid;
        }

        // Phase 2: Hydrate the display shell, stream the embedded CKPT into
        // the trainer, and continue into a second project generation.
        {
            auto document =
                lfs::io::project::ProjectDocument::open(project_path);
            ASSERT_TRUE(document)
                << lfs::format_for_developer(document.error());
            const auto checkpoint_uuids =
                document->checkpoint_uuids();
            ASSERT_EQ(checkpoint_uuids.size(), 1u);
            const auto* checkpoint =
                document->find_checkpoint(checkpoint_uuids.front());
            ASSERT_NE(checkpoint, nullptr);

            std::optional<lfs::core::CheckpointParametersLoadResult>
                checkpoint_params_result;
            const auto params_visit =
                checkpoint->visit_stream(
                    [&](std::istream& source,
                        const std::uint64_t bytes)
                        -> lfs::Result<void> {
                        checkpoint_params_result =
                            lfs::core::load_checkpoint_params(
                                source, bytes);
                        return {};
                    });
            ASSERT_TRUE(params_visit)
                << lfs::format_for_developer(
                       params_visit.error());
            ASSERT_TRUE(checkpoint_params_result.has_value());
            ASSERT_TRUE(checkpoint_params_result->has_value())
                << checkpoint_params_result->error();

            auto params =
                std::move(**checkpoint_params_result);
            params.resume_checkpoint.reset();
            params.resume_project = project_path;
            params.dataset.data_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
            params.dataset.output_path = output_path_;
            auto resumed_params = params;
            resumed_params.optimization.iterations = total_iter;
            resumed_params.optimization.stop_refine = total_iter;
            resumed_params.cli_iterations_set = true;

            lfs::core::Scene scene;
            const auto hydration =
                document->hydrate(scene);
            ASSERT_TRUE(hydration)
                << lfs::format_for_developer(
                       hydration.error());
            ASSERT_TRUE(hydration->trainer_state_pending);
            ASSERT_EQ(
                hydration->checkpoint_uuid,
                std::optional(checkpoint_uuids.front()));
            ASSERT_TRUE(hydration->checkpoint_header.has_value());

            const auto* display_model =
                scene.getTrainingModel();
            ASSERT_NE(display_model, nullptr);
            const size_t display_gaussians =
                display_model->size();
            ASSERT_GT(display_gaussians, 0u);

            auto installed =
                lfs::training::
                    installTrainerFromProjectCheckpoint(
                        scene, *document,
                        checkpoint_uuids.front(),
                        resumed_params,
                        lfs::core::path_to_utf8(
                            project_path),
                        hydration->checkpoint_header
                            ->iteration);
            ASSERT_TRUE(installed.has_value())
                << installed.error();
            ASSERT_NE(installed->trainer, nullptr);
            EXPECT_EQ(
                installed->iteration,
                checkpoint_iteration);
            EXPECT_EQ(
                installed->trainer
                    ->get_current_iteration(),
                checkpoint_iteration);
            EXPECT_EQ(
                static_cast<size_t>(
                    installed->trainer->get_strategy()
                        .get_model()
                        .size()),
                display_gaussians);
            // Optimizer/strategy adoption: strategy and
            // Adam are live after stream restore.
            EXPECT_GT(
                installed->trainer->get_strategy()
                    .get_optimizer()
                    .get_lr(),
                0.0f);
            EXPECT_EQ(
                installed->trainer->getParams()
                    .optimization.refine_every,
                static_cast<size_t>(100));
            EXPECT_TRUE(
                installed->trainer->getParams()
                    .optimization.headless);

            auto trainer =
                std::move(installed->trainer);
            trainer->get_strategy_mutable()
                .set_optimization_params(
                    resumed_params.optimization);
            trainer->setParams(resumed_params);
            lfs::training::grant_headless_project_saves(
                *trainer, resumed_params);

            EXPECT_EQ(
                trainer->getParams()
                    .optimization.iterations,
                static_cast<size_t>(total_iter));
            EXPECT_EQ(
                trainer->getParams()
                    .optimization.stop_refine,
                static_cast<size_t>(total_iter));

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value())
                << "Resume training failed: " << lfs::format_for_developer(train_result.error());

            EXPECT_EQ(trainer->get_current_iteration(), total_iter);

            trainer->shutdown();
        }

        auto continued =
            lfs::io::project::ProjectDocument::open(project_path);
        ASSERT_TRUE(continued)
            << lfs::format_for_developer(continued.error());
        {
            auto continued_reader =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(continued_reader)
                << lfs::format_for_developer(
                       continued_reader.error());
            EXPECT_EQ(
                continued_reader->superblock()
                    .project_uuid,
                phase_one_project_uuid);
        }
        const auto continued_checkpoints =
            continued->checkpoint_uuids();
        ASSERT_EQ(continued_checkpoints.size(), 1u);
        const auto* continued_checkpoint =
            continued->find_checkpoint(
                continued_checkpoints.front());
        ASSERT_NE(continued_checkpoint, nullptr);
        std::optional<int> continued_iteration;
        const auto continued_header =
            continued_checkpoint->visit_stream(
                [&](std::istream& source,
                    const std::uint64_t bytes)
                    -> lfs::Result<void> {
                    auto header =
                        lfs::core::load_checkpoint_header(
                            source, bytes);
                    if (header) {
                        continued_iteration =
                            header->iteration;
                    }
                    return {};
                });
        ASSERT_TRUE(continued_header)
            << lfs::format_for_developer(
                   continued_header.error());
        ASSERT_TRUE(continued_iteration.has_value());
        EXPECT_EQ(*continued_iteration, total_iter);

        LOG_INFO(
            "Project resume test passed: strategy={}, sh_degree={}",
            strategy, sh_degree);
    }

    std::string TestName(const ::testing::TestParamInfo<CheckpointResumeTest::ParamType>& info) {
        const bool nightly = std::get<3>(info.param) > 10;
        auto name = std::format("{}_{}_{}", nightly ? "nightly" : "tiny",
                                std::get<0>(info.param), std::get<1>(info.param));
        std::replace_if(
            name.begin(), name.end(), [](const unsigned char c) { return !std::isalnum(c); }, '_');
        return name;
    }

    INSTANTIATE_TEST_SUITE_P(
        CheckpointStrategies,
        CheckpointResumeTest,
        ::testing::Values(
            std::make_tuple("mcmc", 0, 2, 4),
            std::make_tuple("mcmc", 0, 1200, 2100)),
        TestName);

    class ProjectCheckpointTrainerInstall
        : public ::testing::Test {
    protected:
        void TearDown() override {
            lfs::training::TrainingSnapshotService::
                reset_process_pinned_d2h_calibration_for_testing();
        }

        template <typename Predicate>
        [[nodiscard]] bool waitUntil(
            Predicate&& condition,
            const std::chrono::milliseconds timeout =
                std::chrono::seconds(10)) {
            const auto deadline =
                std::chrono::steady_clock::now() + timeout;
            while (!condition() &&
                   std::chrono::steady_clock::now() <
                       deadline) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(2));
            }
            return condition();
        }
    };

    TEST_F(ProjectCheckpointTrainerInstall,
           SharedHelperRestoresTrainerAndGaussianCount) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_project_ckpt_helper_pos";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        constexpr int iterations = 3;
        lfs::core::param::TrainingParameters params;
        params.dataset.data_path =
            std::filesystem::path(TEST_DATA_DIR) /
            "bicycle";
        params.dataset.images = TEST_IMAGES;
        params.dataset.output_path = output_path;
        params.optimization.iterations = iterations;
        params.optimization.strategy = "mcmc";
        params.optimization.sh_degree = 0;
        params.optimization.headless = true;
        params.optimization.max_cap = 100000;
        params.optimization.refine_every = 100;
        const size_t stop_refine =
            static_cast<size_t>(iterations);
        params.optimization.start_refine =
            std::min<size_t>(500, stop_refine);
        params.optimization.stop_refine = stop_refine;
        params.optimization.save_steps = {
            static_cast<size_t>(iterations)};

        {
            lfs::core::Scene scene;
            ASSERT_TRUE(
                lfs::training::loadTrainingDataIntoScene(
                    params, scene))
                << "load training data failed";
            ASSERT_TRUE(
                lfs::training::initializeTrainingModel(
                    params, scene))
                << "init model failed";
            auto trainer =
                std::make_unique<lfs::training::Trainer>(
                    scene);
            auto init = trainer->initialize(params);
            ASSERT_TRUE(init)
                << "Failed to init trainer: "
                << init.error();
            lfs::training::grant_headless_project_saves(
                *trainer, params);
            auto train = trainer->train();
            ASSERT_TRUE(train)
                << "seed train failed: "
                << lfs::format_for_developer(
                       train.error());
            EXPECT_EQ(
                trainer->get_current_iteration(),
                iterations);
            trainer->shutdown();
        }

        const auto project_path =
            output_path / "project.licht";
        ASSERT_TRUE(std::filesystem::is_regular_file(
            project_path));

        auto document =
            lfs::io::project::ProjectDocument::open(
                project_path);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(
                   document.error());
        const auto checkpoint_uuids =
            document->checkpoint_uuids();
        ASSERT_EQ(checkpoint_uuids.size(), 1u);

        std::optional<
            lfs::core::CheckpointParametersLoadResult>
            checkpoint_params_result;
        const auto* checkpoint =
            document->find_checkpoint(
                checkpoint_uuids.front());
        ASSERT_NE(checkpoint, nullptr);
        ASSERT_TRUE(checkpoint->visit_stream(
            [&](std::istream& source,
                const std::uint64_t bytes)
                -> lfs::Result<void> {
                checkpoint_params_result =
                    lfs::core::load_checkpoint_params(
                        source, bytes);
                return {};
            }));
        ASSERT_TRUE(checkpoint_params_result.has_value());
        ASSERT_TRUE(checkpoint_params_result->has_value())
            << checkpoint_params_result->error();

        auto ckpt_params =
            std::move(**checkpoint_params_result);
        ckpt_params.resume_checkpoint.reset();
        ckpt_params.resume_project = project_path;
        ckpt_params.dataset.data_path =
            std::filesystem::path(TEST_DATA_DIR) /
            "bicycle";
        ckpt_params.dataset.output_path = output_path;

        lfs::core::Scene scene;
        const auto hydration = document->hydrate(scene);
        ASSERT_TRUE(hydration)
            << lfs::format_for_developer(
                   hydration.error());
        ASSERT_TRUE(hydration->trainer_state_pending);
        ASSERT_TRUE(hydration->checkpoint_header);
        const auto* display =
            scene.getTrainingModel();
        ASSERT_NE(display, nullptr);
        const size_t display_count = display->size();
        ASSERT_GT(display_count, 0u);

        auto installed =
            lfs::training::
                installTrainerFromProjectCheckpoint(
                    scene, *document,
                    checkpoint_uuids.front(),
                    ckpt_params,
                    lfs::core::path_to_utf8(
                        project_path),
                    hydration->checkpoint_header
                        ->iteration);
        ASSERT_TRUE(installed.has_value())
            << installed.error();
        ASSERT_NE(installed->trainer, nullptr);
        EXPECT_EQ(
            installed->iteration,
            hydration->checkpoint_header->iteration);
        EXPECT_EQ(
            installed->trainer->get_current_iteration(),
            hydration->checkpoint_header->iteration);
        EXPECT_EQ(
            static_cast<size_t>(
                installed->trainer->get_strategy()
                    .get_model()
                    .size()),
            display_count);
        EXPECT_GT(
            installed->trainer->get_strategy()
                .get_optimizer()
                .get_lr(),
            0.0f);
        EXPECT_EQ(
            scene.getTrainingModel()->size(),
            display_count);

        installed->trainer->shutdown();
        std::filesystem::remove_all(output_path, ec);
    }

    TEST_F(ProjectCheckpointTrainerInstall,
           MissingDatasetPathLeavesDisplayIntact) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_project_ckpt_helper_neg";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        constexpr int iterations = 3;
        lfs::core::param::TrainingParameters params;
        params.dataset.data_path =
            std::filesystem::path(TEST_DATA_DIR) /
            "bicycle";
        params.dataset.images = TEST_IMAGES;
        params.dataset.output_path = output_path;
        params.optimization.iterations = iterations;
        params.optimization.strategy = "mcmc";
        params.optimization.sh_degree = 0;
        params.optimization.headless = true;
        params.optimization.max_cap = 100000;
        params.optimization.refine_every = 100;
        const size_t stop_refine =
            static_cast<size_t>(iterations);
        params.optimization.start_refine =
            std::min<size_t>(500, stop_refine);
        params.optimization.stop_refine = stop_refine;
        params.optimization.save_steps = {
            static_cast<size_t>(iterations)};

        {
            lfs::core::Scene scene;
            ASSERT_TRUE(
                lfs::training::loadTrainingDataIntoScene(
                    params, scene))
                << "load training data failed";
            ASSERT_TRUE(
                lfs::training::initializeTrainingModel(
                    params, scene))
                << "init model failed";
            auto trainer =
                std::make_unique<lfs::training::Trainer>(
                    scene);
            auto init = trainer->initialize(params);
            ASSERT_TRUE(init)
                << "Failed to init trainer: "
                << init.error();
            lfs::training::grant_headless_project_saves(
                *trainer, params);
            auto train = trainer->train();
            ASSERT_TRUE(train)
                << "seed train failed: "
                << lfs::format_for_developer(
                       train.error());
            trainer->shutdown();
        }

        const auto project_path =
            output_path / "project.licht";
        ASSERT_TRUE(std::filesystem::is_regular_file(
            project_path));

        auto document =
            lfs::io::project::ProjectDocument::open(
                project_path);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(
                   document.error());
        const auto checkpoint_uuids =
            document->checkpoint_uuids();
        ASSERT_EQ(checkpoint_uuids.size(), 1u);

        lfs::core::Scene scene;
        const auto hydration = document->hydrate(scene);
        ASSERT_TRUE(hydration)
            << lfs::format_for_developer(
                   hydration.error());
        ASSERT_TRUE(hydration->trainer_state_pending);
        ASSERT_TRUE(hydration->checkpoint_header);
        const auto* display =
            scene.getTrainingModel();
        ASSERT_NE(display, nullptr);
        const size_t display_count = display->size();
        const auto node_count_before =
            scene.getNodes().size();

        lfs::core::param::TrainingParameters bad_params;
        bad_params.dataset.data_path =
            output_path / "missing_dataset_root";
        bad_params.dataset.output_path = output_path;
        bad_params.optimization.strategy = "mcmc";
        bad_params.optimization.headless = true;

        auto installed =
            lfs::training::
                installTrainerFromProjectCheckpoint(
                    scene, *document,
                    checkpoint_uuids.front(),
                    bad_params,
                    lfs::core::path_to_utf8(
                        project_path),
                    hydration->checkpoint_header
                        ->iteration);
        ASSERT_FALSE(installed.has_value());
        EXPECT_NE(
            installed.error().find(
                "Dataset path does not exist"),
            std::string::npos)
            << installed.error();

        // Display shell intact; no trainer constructed.
        ASSERT_NE(scene.getTrainingModel(), nullptr);
        EXPECT_EQ(
            scene.getTrainingModel()->size(),
            display_count);
        EXPECT_EQ(
            scene.getNodes().size(),
            node_count_before);

        std::filesystem::remove_all(output_path, ec);
    }

    std::shared_ptr<lfs::core::Camera> make_grant_test_camera() {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros(
                {3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{},
            std::filesystem::path{}, 64, 64, 0);
    }

    lfs::core::param::TrainingParameters make_tiny_headless_params(
        const std::filesystem::path& output_path,
        const int iterations) {
        lfs::core::param::TrainingParameters params;
        params.dataset.data_path =
            std::filesystem::path(TEST_DATA_DIR) / "bicycle";
        params.dataset.images = TEST_IMAGES;
        params.dataset.output_path = output_path;
        params.optimization.iterations = iterations;
        params.optimization.strategy = "mcmc";
        params.optimization.sh_degree = 0;
        params.optimization.headless = true;
        params.optimization.max_cap = 100000;
        params.optimization.refine_every = 100;
        const size_t stop_refine =
            static_cast<size_t>(iterations);
        params.optimization.start_refine =
            std::min<size_t>(500, stop_refine);
        params.optimization.stop_refine = stop_refine;
        return params;
    }

    TEST_F(ProjectCheckpointTrainerInstall,
           UngrantedTrainerNeverWritesProjectFiles) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_ungranted_never_writes";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        constexpr int iterations = 3;
        auto params = make_tiny_headless_params(
            output_path, iterations);
        params.optimization.save_steps = {1, 2};
        params.save_project_at_iteration = 2;

        lfs::core::Scene scene;
        ASSERT_TRUE(
            lfs::training::loadTrainingDataIntoScene(
                params, scene))
            << "load training data failed";
        ASSERT_TRUE(
            lfs::training::initializeTrainingModel(
                params, scene))
            << "init model failed";
        auto trainer =
            std::make_unique<lfs::training::Trainer>(
                scene);
        auto init = trainer->initialize(params);
        ASSERT_TRUE(init)
            << "Failed to init trainer: " << init.error();
        auto train = trainer->train();
        ASSERT_TRUE(train)
            << "ungranted train failed: "
            << lfs::format_for_developer(train.error());
        trainer->shutdown();

        EXPECT_FALSE(std::filesystem::exists(
            output_path / "project.licht"));
        bool found_licht = false;
        for (const auto& entry :
             std::filesystem::recursive_directory_iterator(
                 output_path)) {
            if (entry.path().extension() == ".licht") {
                found_licht = true;
                break;
            }
        }
        EXPECT_FALSE(found_licht)
            << "ungranted trainer wrote a .licht file under "
            << output_path;

        std::filesystem::remove_all(output_path, ec);
    }

    TEST_F(ProjectCheckpointTrainerInstall,
           GrantedTrainerSaveAtIterWithoutPathWritesBoundFile) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_granted_save_at_iter_bound";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        constexpr int iterations = 3;
        auto params = make_tiny_headless_params(
            output_path, iterations);
        params.optimization.save_steps = {};
        params.save_project_at_iteration = 2;
        ASSERT_TRUE(params.save_project_path.empty());

        lfs::core::Scene scene;
        ASSERT_TRUE(
            lfs::training::loadTrainingDataIntoScene(
                params, scene))
            << "load training data failed";
        ASSERT_TRUE(
            lfs::training::initializeTrainingModel(
                params, scene))
            << "init model failed";
        auto trainer =
            std::make_unique<lfs::training::Trainer>(
                scene);
        auto init = trainer->initialize(params);
        ASSERT_TRUE(init)
            << "Failed to init trainer: " << init.error();
        const auto bound_path =
            output_path / "project.licht";
        trainer->set_live_project_snapshot(bound_path);
        trainer->set_trainer_project_save_policy({
            .on_completion = false,
            .on_stop_or_error = false,
            .at_step_boundaries = true,
        });
        auto train = trainer->train();
        ASSERT_TRUE(train)
            << "granted save-at-iter train failed: "
            << lfs::format_for_developer(train.error());
        ASSERT_TRUE(waitUntil([&] {
            const auto metrics =
                trainer->get_project_snapshot_metrics();
            return !metrics.writer_in_flight &&
                   std::filesystem::is_regular_file(
                       bound_path) &&
                   metrics.last_path.lexically_normal() ==
                       bound_path.lexically_normal();
        })) << "timed out waiting for trainer project publish to "
            << bound_path;
        trainer->shutdown();
        EXPECT_TRUE(std::filesystem::is_regular_file(
            bound_path));
        EXPECT_EQ(
            trainer->get_project_snapshot_metrics()
                .last_path.lexically_normal(),
            bound_path.lexically_normal());

        std::filesystem::remove_all(output_path, ec);
    }

    TEST_F(ProjectCheckpointTrainerInstall,
           UngrantedTrainerStillHonorsExplicitSave) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_ungranted_explicit_save";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        constexpr int iterations = 3;
        auto params = make_tiny_headless_params(
            output_path, iterations);
        params.optimization.save_steps = {1};

        lfs::core::Scene scene;
        ASSERT_TRUE(
            lfs::training::loadTrainingDataIntoScene(
                params, scene))
            << "load training data failed";
        ASSERT_TRUE(
            lfs::training::initializeTrainingModel(
                params, scene))
            << "init model failed";
        auto trainer =
            std::make_unique<lfs::training::Trainer>(
                scene);
        auto init = trainer->initialize(params);
        ASSERT_TRUE(init)
            << "Failed to init trainer: " << init.error();
        auto train = trainer->train();
        ASSERT_TRUE(train)
            << "ungranted train failed: "
            << lfs::format_for_developer(train.error());

        EXPECT_FALSE(std::filesystem::exists(
            output_path / "project.licht"));
        const auto explicit_path =
            output_path / "explicit.licht";
        auto saved = trainer->save_project_to(
            explicit_path,
            trainer->get_current_iteration());
        ASSERT_TRUE(saved) << saved.error();
        EXPECT_TRUE(std::filesystem::is_regular_file(
            explicit_path));
        trainer->shutdown();

        std::filesystem::remove_all(output_path, ec);
    }

    TEST_F(ProjectCheckpointTrainerInstall,
           RequestProjectSaveWithoutDestinationFails) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_request_save_no_dest";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        constexpr int iterations = 3;
        auto params = make_tiny_headless_params(
            output_path, iterations);

        lfs::core::Scene scene;
        ASSERT_TRUE(
            lfs::training::loadTrainingDataIntoScene(
                params, scene))
            << "load training data failed";
        ASSERT_TRUE(
            lfs::training::initializeTrainingModel(
                params, scene))
            << "init model failed";
        auto trainer =
            std::make_unique<lfs::training::Trainer>(
                scene);
        auto init = trainer->initialize(params);
        ASSERT_TRUE(init)
            << "Failed to init trainer: " << init.error();

        const auto request_id =
            trainer->request_project_save();
        ASSERT_NE(request_id, 0u);
        const auto metrics =
            trainer->get_project_snapshot_metrics();
        EXPECT_EQ(metrics.last_failed_request_id,
                  request_id);
        EXPECT_NE(
            metrics.last_writer_error.find(
                "No project destination is bound"),
            std::string::npos)
            << metrics.last_writer_error;
        trainer->shutdown();

        std::filesystem::remove_all(output_path, ec);
    }

    TEST_F(ProjectCheckpointTrainerInstall,
           GrantHeadlessProjectSavesBindsOverrideAndFallback) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_grant_headless_bind";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        scene.addCamera(
            "camera.png", cameras, make_grant_test_camera());
        auto trainer =
            std::make_unique<lfs::training::Trainer>(
                scene);

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = output_path;
        lfs::training::grant_headless_project_saves(
            *trainer, params);
        {
            const auto bound = trainer->bound_project_path();
            ASSERT_TRUE(bound.has_value());
            EXPECT_EQ(
                bound->lexically_normal(),
                (output_path / "project.licht")
                    .lexically_normal());
        }

        const auto override_path =
            output_path / "resumed.licht";
        lfs::training::grant_headless_project_saves(
            *trainer, params, override_path);
        {
            const auto bound = trainer->bound_project_path();
            ASSERT_TRUE(bound.has_value());
            EXPECT_EQ(
                bound->lexically_normal(),
                override_path.lexically_normal());
        }

        trainer->set_live_project_snapshot(std::nullopt);
        trainer->set_trainer_project_save_policy({});
        params.dataset.output_path.clear();
        lfs::training::grant_headless_project_saves(
            *trainer, params);
        EXPECT_FALSE(
            trainer->bound_project_path().has_value());
        const auto policy =
            trainer->trainer_project_save_policy();
        EXPECT_FALSE(policy.on_completion);
        EXPECT_FALSE(policy.on_stop_or_error);
        EXPECT_FALSE(policy.at_step_boundaries);

        trainer->shutdown();
        std::filesystem::remove_all(output_path, ec);
    }

    TEST_F(ProjectCheckpointTrainerInstall,
           SecondRunCompactsLeftoverProjectFile) {
        const auto output_path =
            std::filesystem::temp_directory_path() /
            "lfs_test_second_run_compacts_leftover";
        std::error_code ec;
        std::filesystem::remove_all(output_path, ec);
        std::filesystem::create_directories(output_path);

        constexpr int iterations = 3;
        auto params = make_tiny_headless_params(
            output_path, iterations);
        params.optimization.save_steps = {1};

        const auto project_path =
            output_path / "project.licht";

        {
            lfs::core::Scene scene;
            ASSERT_TRUE(
                lfs::training::loadTrainingDataIntoScene(
                    params, scene))
                << "load training data failed";
            ASSERT_TRUE(
                lfs::training::initializeTrainingModel(
                    params, scene))
                << "init model failed";
            auto trainer =
                std::make_unique<lfs::training::Trainer>(
                    scene);
            auto init = trainer->initialize(params);
            ASSERT_TRUE(init)
                << "Failed to init trainer: "
                << init.error();
            lfs::training::grant_headless_project_saves(
                *trainer, params);
            auto train = trainer->train();
            ASSERT_TRUE(train)
                << "run A train failed: "
                << lfs::format_for_developer(
                       train.error());
            trainer->shutdown();
        }

        ASSERT_TRUE(std::filesystem::is_regular_file(
            project_path));
        const auto s1 =
            std::filesystem::file_size(project_path);
        ASSERT_GT(s1, 0u);
        lfs::core::Uuid project_uuid;
        {
            auto run_a_reader =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(run_a_reader)
                << lfs::format_for_developer(
                       run_a_reader.error());
            project_uuid =
                run_a_reader->superblock().project_uuid;
        }

        {
            lfs::core::Scene scene;
            ASSERT_TRUE(
                lfs::training::loadTrainingDataIntoScene(
                    params, scene))
                << "load training data failed";
            ASSERT_TRUE(
                lfs::training::initializeTrainingModel(
                    params, scene))
                << "init model failed";
            auto trainer =
                std::make_unique<lfs::training::Trainer>(
                    scene);
            auto init = trainer->initialize(params);
            ASSERT_TRUE(init)
                << "Failed to init trainer: "
                << init.error();
            lfs::training::grant_headless_project_saves(
                *trainer, params);
            auto train = trainer->train();
            ASSERT_TRUE(train)
                << "run B train failed: "
                << lfs::format_for_developer(
                       train.error());
            trainer->shutdown();
        }

        ASSERT_TRUE(std::filesystem::is_regular_file(
            project_path));
        const auto s2 =
            std::filesystem::file_size(project_path);
        auto run_b_reader =
            lfs::io::project::ProjectReader::open(
                project_path);
        ASSERT_TRUE(run_b_reader)
            << lfs::format_for_developer(
                   run_b_reader.error());
        EXPECT_EQ(
            run_b_reader->superblock().project_uuid,
            project_uuid);
        EXPECT_LT(s2, s1 + s1 / 2);

        std::filesystem::remove_all(output_path, ec);
    }

} // namespace
