/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "training/checkpoint.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/strategies/mcmc.hpp"
#include "training/training_snapshot_service.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

    constexpr std::size_t MIB = 1024 * 1024;

    std::unique_ptr<lfs::core::SplatData>
    make_snapshot_test_splat(const std::size_t count) {
        std::vector<float> means(count * 3);
        std::vector<float> rotations(count * 4, 0.0f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] =
                static_cast<float>(index) * 0.001f;
            means[index * 3 + 1] = 1.0f;
            means[index * 3 + 2] = -2.0f;
            rotations[index * 4] = 1.0f;
        }

        auto result =
            std::make_unique<lfs::core::SplatData>(
                0,
                lfs::core::Tensor::from_vector(
                    means, {count, 3},
                    lfs::core::Device::CUDA),
                lfs::core::Tensor::zeros(
                    {count, 1, 3},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32),
                lfs::core::Tensor::zeros(
                    {0}, lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32),
                lfs::core::Tensor::zeros(
                    {count, 3},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32),
                lfs::core::Tensor::from_vector(
                    rotations, {count, 4},
                    lfs::core::Device::CUDA),
                lfs::core::Tensor::zeros(
                    {count, 1},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32),
                1.0f);
        EXPECT_EQ(result->means().shape(),
                  lfs::core::TensorShape({count, 3}));
        EXPECT_EQ(result->sh0().shape(),
                  lfs::core::TensorShape({count, 1, 3}));
        EXPECT_EQ(result->shN().numel(), 0u);
        EXPECT_EQ(result->scaling_raw().shape(),
                  lfs::core::TensorShape({count, 3}));
        EXPECT_EQ(result->rotation_raw().shape(),
                  lfs::core::TensorShape({count, 4}));
        EXPECT_EQ(result->opacity_raw().shape(),
                  lfs::core::TensorShape({count, 1}));
        return result;
    }

    lfs::core::param::TrainingParameters
    make_snapshot_test_params(const std::size_t count) {
        lfs::core::param::TrainingParameters params;
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 500;
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = count;
        return params;
    }

    bool cuda_device_available() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess &&
               count > 0;
    }

    TEST(TrainingSnapshotServiceConfigTest,
         RejectsPinnedRingLargerThan512MiB) {
        EXPECT_THROW(
            lfs::training::TrainingSnapshotService({
                .ring_slots = 5,
                .band_bytes = 128 * MIB,
                .calibration_bytes = MIB,
                .calibration_iterations = 1,
            }),
            std::invalid_argument);
    }

    TEST(TrainingSnapshotServiceTest,
         CapturesByteExactLfkpAndOwnsPostResumeBytes) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }

        constexpr std::size_t GAUSSIAN_COUNT = 8192;
        constexpr int SAVED_ITERATION = 137;
        auto params =
            make_snapshot_test_params(GAUSSIAN_COUNT);
        auto model =
            make_snapshot_test_splat(GAUSSIAN_COUNT);
        lfs::training::MCMC strategy(*model);
        strategy.initialize(params.optimization);

        auto* source_moments =
            strategy.get_optimizer().get_state_mutable(
                lfs::training::ParamType::Means);
        ASSERT_NE(source_moments, nullptr);
        ASSERT_TRUE(
            source_moments->exp_avg_scale.is_valid());
        ASSERT_EQ(
            source_moments->exp_avg_scale.shape(),
            lfs::core::TensorShape({GAUSSIAN_COUNT}));
        source_moments->exp_avg_scale.fill_(3.5f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto original_means =
            model->means().cpu().to_vector();
        const auto original_moment_scales =
            source_moments->exp_avg_scale.cpu().to_vector();

        std::ostringstream reference_stream(
            std::ios::binary | std::ios::out);
        const auto reference_result =
            lfs::training::serialize_checkpoint(
                reference_stream, SAVED_ITERATION,
                strategy, params, nullptr, nullptr,
                nullptr, nullptr);
        ASSERT_TRUE(reference_result.has_value())
            << lfs::format_for_developer(
                   reference_result.error());
        const auto reference_bytes =
            reference_stream.str();
        ASSERT_EQ(reference_result->bytes,
                  reference_bytes.size());

        lfs::training::TrainingSnapshotService service({
            .ring_slots = 4,
            .band_bytes = 64 * 1024,
            .calibration_bytes = 64 * 1024,
            .calibration_iterations = 4,
        });
        std::optional<lfs::core::Uuid> cpu_state_stamp;
        const auto assigned_snapshot_uuid =
            lfs::core::generate_uuid_v4();
        const lfs::training::TrainingSnapshotCaptureRequest
            request{
                .iteration = SAVED_ITERATION,
                .snapshot_uuid = assigned_snapshot_uuid,
                .strategy = strategy,
                .params = params,
                .capture_additional_cpu_state =
                    [&cpu_state_stamp](
                        const lfs::core::Uuid& uuid)
                    -> lfs::Result<void> {
                    cpu_state_stamp = uuid;
                    return {};
                },
            };

        auto prepared = service.prepare(request);
        ASSERT_TRUE(prepared.has_value())
            << lfs::format_for_developer(
                   prepared.error());
        EXPECT_EQ(prepared->checkpoint_bytes(),
                  reference_bytes.size());
        const auto prepared_uuid =
            prepared->snapshot_uuid();
        EXPECT_EQ(prepared_uuid, assigned_snapshot_uuid);

        auto capture_request = request;
        capture_request.safe_point_entered_at =
            std::chrono::steady_clock::now() -
            std::chrono::milliseconds(1);
        auto pending =
            service.capture(
                std::move(*prepared), capture_request);
        ASSERT_TRUE(pending.has_value())
            << lfs::format_for_developer(
                   pending.error());

        // capture() has released the optimizer pause. Mutating both model
        // parameters and optimizer moments must not affect the pending
        // pageable checkpoint bytes.
        model->means().fill_(42.0f);
        source_moments->exp_avg_scale.fill_(7.5f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        auto captured = pending->wait();
        ASSERT_TRUE(captured.has_value())
            << lfs::format_for_developer(
                   captured.error());
        ASSERT_TRUE(captured->checkpoint_bytes);
        ASSERT_TRUE(cpu_state_stamp.has_value());
        EXPECT_EQ(*cpu_state_stamp, prepared_uuid);
        EXPECT_EQ(captured->snapshot_uuid, prepared_uuid);
        EXPECT_EQ(captured->iteration, SAVED_ITERATION);
        ASSERT_EQ(captured->checkpoint_bytes->size(),
                  reference_bytes.size());
        EXPECT_EQ(
            std::memcmp(
                captured->checkpoint_bytes->data(),
                reference_bytes.data(),
                reference_bytes.size()),
            0);

        const auto& metrics = captured->metrics;
        EXPECT_EQ(metrics.snapshot_uuid, prepared_uuid);
        EXPECT_EQ(metrics.iteration, SAVED_ITERATION);
        EXPECT_EQ(metrics.checkpoint_bytes,
                  reference_bytes.size());
        EXPECT_GT(metrics.device_snapshot_bytes, 0u);
        EXPECT_LE(metrics.device_snapshot_bytes,
                  metrics.checkpoint_bytes);
        EXPECT_EQ(metrics.pinned_peak_bytes,
                  4u * 64u * 1024u);
        EXPECT_LE(metrics.pinned_peak_bytes,
                  512u * MIB);
        EXPECT_EQ(metrics.host_staging_bytes,
                  metrics.checkpoint_bytes);
        EXPECT_TRUE(metrics.host_memory_preflight_passed);
        EXPECT_TRUE(metrics.host_ram_within_gate);
        EXPECT_GT(metrics.preparation_ms, 0.0);
        EXPECT_TRUE(metrics.cold_first_snapshot);
        EXPECT_GT(metrics.tensor_piece_count, 0u);
        EXPECT_GT(metrics.cpu_piece_count, 0u);
        EXPECT_TRUE(metrics.consistency_proven);
        EXPECT_GE(metrics.safe_point_entry_ms, 0.5);
        ASSERT_GT(
            metrics.measured_pinned_d2h_bytes_per_second,
            0.0);
        const double expected_gate_ms =
            static_cast<double>(
                metrics.checkpoint_bytes) /
            metrics.measured_pinned_d2h_bytes_per_second *
            1.12 * 1000.0;
        EXPECT_NEAR(metrics.rig_gate_ms,
                    expected_gate_ms, 1e-9);

        const auto aggregate = service.metrics();
        EXPECT_EQ(aggregate.completed_snapshots, 1u);
        EXPECT_DOUBLE_EQ(aggregate.pause_p95_ms,
                         metrics.pause_ms);

        const std::string captured_string(
            reinterpret_cast<const char*>(
                captured->checkpoint_bytes->data()),
            captured->checkpoint_bytes->size());
        std::istringstream captured_stream(
            captured_string,
            std::ios::binary | std::ios::in);
        auto target_model = make_snapshot_test_splat(1);
        lfs::training::MCMC target_strategy(
            *target_model);
        target_strategy.initialize(params.optimization);
        auto loaded_params = params;
        const auto loaded =
            lfs::training::load_checkpoint(
                captured_stream,
                captured->checkpoint_bytes->size(),
                target_strategy, loaded_params,
                nullptr, nullptr, nullptr, nullptr,
                {}, "snapshot-service test CKPT");
        ASSERT_TRUE(loaded.has_value())
            << loaded.error();
        EXPECT_EQ(*loaded, SAVED_ITERATION);
        EXPECT_EQ(target_strategy.get_model().size(),
                  GAUSSIAN_COUNT);
        EXPECT_EQ(
            target_strategy.get_model()
                .means()
                .cpu()
                .to_vector(),
            original_means);
        const auto* loaded_moments =
            target_strategy.get_optimizer().get_state(
                lfs::training::ParamType::Means);
        ASSERT_NE(loaded_moments, nullptr);
        EXPECT_EQ(
            loaded_moments->exp_avg_scale
                .cpu()
                .to_vector(),
            original_moment_scales);
    }

} // namespace
