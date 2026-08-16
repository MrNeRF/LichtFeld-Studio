/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "training/checkpoint.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "training/strategies/mcmc.hpp"
#include "training/training_snapshot_service.hpp"
#include "training_snapshot_test_helpers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
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

    lfs::core::Tensor selection_mask(
        const std::size_t count,
        const std::uint8_t value) {
        auto result = lfs::core::Tensor::empty(
            {count},
            lfs::core::Device::CPU,
            lfs::core::DataType::UInt8);
        std::fill_n(
            result.ptr<std::uint8_t>(), count, value);
        return result;
    }

    std::shared_ptr<lfs::core::Camera>
    make_snapshot_test_camera(const int uid) {
        const auto empty_distortion =
            lfs::core::Tensor::zeros(
                {0}, lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
        auto camera =
            std::make_shared<lfs::core::Camera>(
                lfs::core::Tensor::eye(
                    3, lfs::core::Device::CPU),
                lfs::core::Tensor::zeros(
                    {3}, lfs::core::Device::CPU),
                1000.0f, 1000.0f, 960.0f, 540.0f,
                empty_distortion, empty_distortion,
                lfs::core::CameraModelType::PINHOLE,
                std::format("camera_{:04}.png", uid),
                std::filesystem::path{},
                std::filesystem::path{},
                1920, 1080, uid, uid);
        camera->set_image_dimensions(1920, 1080);
        return camera;
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
        ASSERT_TRUE(source_moments->is_joint());
        ASSERT_TRUE(source_moments->exp_avg.is_valid());
        ASSERT_TRUE(
            source_moments->joint_bounds.is_valid());
        // Joint codec (9169a2a00 / #1588) serializes exp_avg +
        // joint_bounds only. Seed a unique fp32 pattern so a
        // capture path that drops joint_bounds or aliases the
        // live tensor (post-capture fill_ to 7.5f) fails the
        // reload compare and the moment-byte helper.
        const std::size_t expected_bounds =
            lfs::training::joint_adam::n_bounds_for_prims(
                GAUSSIAN_COUNT);
        ASSERT_EQ(
            source_moments->joint_bounds.shape(),
            lfs::core::TensorShape(
                {expected_bounds, std::size_t{4}}));
        source_moments->joint_bounds.fill_(3.5f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto original_means =
            model->means().cpu().to_vector();
        const auto original_moment_scales =
            source_moments->joint_bounds.cpu().to_vector();
        const auto original_optimizer_moments =
            lfs::training::test::
                capture_optimizer_moment_bytes(
                    strategy.get_optimizer());

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
            .calibration_bytes = 64,
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
                    -> lfs::Result<
                        lfs::training::
                            TrainingSnapshotCpuStateMetrics> {
                    cpu_state_stamp = uuid;
                    return lfs::training::
                        TrainingSnapshotCpuStateMetrics{
                            .scng_ms = 0.125,
                            .selm_ms = 0.25,
                            .prms_ms = 0.5,
                        };
                },
            };

        auto initialized = service.initialize(request);
        ASSERT_TRUE(initialized.has_value())
            << lfs::format_for_developer(
                   initialized.error());
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
        source_moments->joint_bounds.fill_(7.5f);
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
        EXPECT_GT(metrics.service_initialization_ms, 0.0);
        EXPECT_LT(metrics.prepare_stall_ms, 10.0);
        EXPECT_DOUBLE_EQ(
            metrics.preparation_ms,
            metrics.prepare_stall_ms);
        EXPECT_GT(metrics.preparation_ms, 0.0);
        EXPECT_TRUE(metrics.cold_first_snapshot);
        EXPECT_DOUBLE_EQ(metrics.scng_ms, 0.125);
        EXPECT_DOUBLE_EQ(metrics.selm_ms, 0.25);
        EXPECT_DOUBLE_EQ(metrics.prms_ms, 0.5);
        EXPECT_DOUBLE_EQ(
            metrics.cold_path_ms,
            metrics.prepare_stall_ms +
                metrics.pause_ms);
        EXPECT_TRUE(metrics.cold_path_within_rig_gate);
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
        EXPECT_EQ(aggregate.p95_n, 1u);
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
        ASSERT_TRUE(loaded_moments->is_joint());
        ASSERT_TRUE(
            loaded_moments->joint_bounds.is_valid());
        EXPECT_EQ(
            loaded_moments->joint_bounds
                .cpu()
                .to_vector(),
            original_moment_scales);
        lfs::training::test::
            expect_optimizer_moment_bytes_equal(
                original_optimizer_moments,
                target_strategy.get_optimizer());
    }

    TEST(TrainingSnapshotServiceTest,
         CpuChaptersCaptureExactSaveIterationInsideSafePoint) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }

        constexpr std::size_t GAUSSIAN_COUNT = 4096;
        constexpr int SAVED_ITERATION = 6;
        lfs::core::Scene scene;
        const auto model_id = scene.addSplat(
            "iter_0000",
            make_snapshot_test_splat(
                GAUSSIAN_COUNT));
        ASSERT_NE(model_id, lfs::core::NULL_NODE);
        scene.setTrainingModelNode(model_id);
        const auto model_uuid =
            scene.getNodeUuid(model_id);
        ASSERT_FALSE(model_uuid.is_nil());
        const std::array selected_node_uuids{
            model_uuid};
        auto* model_node =
            scene.getNodeById(model_id);
        ASSERT_NE(model_node, nullptr);
        ASSERT_NE(model_node->model, nullptr);

        auto params =
            make_snapshot_test_params(
                GAUSSIAN_COUNT);
        params.dataset.images = "iter_0000";
        const auto selection_group =
            scene.addSelectionGroup(
                "iter_0000", {0.25f, 0.5f, 0.75f});
        ASSERT_NE(selection_group, 0);
        scene.setActiveSelectionGroup(
            selection_group);
        lfs::training::MCMC strategy(
            *model_node->model);
        strategy.initialize(params.optimization);

        lfs::training::TrainingSnapshotService service({
            .ring_slots = 4,
            .band_bytes = 64 * 1024,
            .calibration_bytes = 64,
            .calibration_iterations = 4,
        });
        const auto snapshot_uuid =
            lfs::core::generate_uuid_v4();
        lfs::training::ProjectSnapshotChapters
            chapters;
        lfs::training::ProjectSnapshotCpuState
            cpu_state;
        lfs::training::TrainingSnapshotCaptureRequest
            request{
                .iteration = SAVED_ITERATION,
                .snapshot_uuid = snapshot_uuid,
                .strategy = strategy,
                .params = params,
                .capture_additional_cpu_state =
                    [&](const lfs::core::Uuid& uuid) {
                        return lfs::training::
                            capture_project_snapshot_cpu_state(
                                scene, params, uuid,
                                SAVED_ITERATION,
                                cpu_state,
                                selected_node_uuids);
                    },
            };
        auto initialized = service.initialize(request);
        ASSERT_TRUE(initialized)
            << lfs::format_for_developer(
                   initialized.error());

        for (int iteration = 1;
             iteration < SAVED_ITERATION;
             ++iteration) {
            const auto name =
                std::format("iter_{:04}", iteration);
            ASSERT_TRUE(
                scene.renameNode(model_id, name));
            params.dataset.images = name;
            scene.renameSelectionGroup(
                selection_group, name);
            scene.applyPerNodeSelectionSlices(
                lfs::core::SelectionDomain::Splat,
                {{model_uuid,
                  selection_mask(
                      GAUSSIAN_COUNT,
                      selection_group)}});
        }

        auto prepared = service.prepare(request);
        ASSERT_TRUE(prepared)
            << lfs::format_for_developer(
                   prepared.error());

        const auto saved_name =
            std::format(
                "iter_{:04}", SAVED_ITERATION);
        ASSERT_TRUE(
            scene.renameNode(model_id, saved_name));
        params.dataset.images = saved_name;
        scene.renameSelectionGroup(
            selection_group, saved_name);
        scene.applyPerNodeSelectionSlices(
            lfs::core::SelectionDomain::Splat,
            {{model_uuid,
              selection_mask(
                  GAUSSIAN_COUNT,
                  selection_group)}});

        request.safe_point_entered_at =
            std::chrono::steady_clock::now();
        auto pending = service.capture(
            std::move(*prepared), request);
        ASSERT_TRUE(pending)
            << lfs::format_for_developer(
                   pending.error());
        auto materialized =
            lfs::training::
                materialize_project_snapshot_cpu_chapters(
                    std::move(cpu_state), chapters);
        ASSERT_TRUE(materialized)
            << lfs::format_for_developer(
                   materialized.error());
        auto captured = pending->wait();
        ASSERT_TRUE(captured)
            << lfs::format_for_developer(
                   captured.error());

        EXPECT_EQ(
            captured->snapshot_uuid,
            snapshot_uuid);
        EXPECT_EQ(
            captured->iteration,
            SAVED_ITERATION);
        EXPECT_EQ(chapters.snapshot_uuid, snapshot_uuid);
        EXPECT_EQ(
            chapters.iteration,
            SAVED_ITERATION);

        auto training_uuid =
            chapters.scene_graph
                .training_model_uuid();
        ASSERT_TRUE(training_uuid)
            << lfs::format_for_developer(
                   training_uuid.error());
        ASSERT_TRUE(*training_uuid);
        EXPECT_EQ(**training_uuid, model_uuid);
        auto saved_node =
            chapters.scene_graph.find(model_uuid);
        ASSERT_TRUE(saved_node)
            << lfs::format_for_developer(
                   saved_node.error());
        ASSERT_TRUE(*saved_node);
        EXPECT_EQ((*saved_node)->name, saved_name);
        ASSERT_TRUE((*saved_node)->payload);
        EXPECT_EQ(
            (*saved_node)->payload->fourcc,
            "CKPT");
        EXPECT_EQ(
            (*saved_node)->payload->instance_uuid,
            snapshot_uuid);

        ASSERT_EQ(chapters.selection.slices().size(), 1u);
        const auto& saved_groups =
            chapters.selection.groups();
        const auto saved_group =
            std::ranges::find(
                saved_groups, selection_group,
                &lfs::core::SelectionGroup::id);
        ASSERT_NE(saved_group, saved_groups.end());
        EXPECT_EQ(
            saved_group->name,
            saved_name);
        const auto& saved_mask =
            chapters.selection.slices().front().mask;
        ASSERT_EQ(
            saved_mask.size(),
            GAUSSIAN_COUNT);
        EXPECT_TRUE(std::ranges::all_of(
            saved_mask,
            [selection_group](
                const std::uint8_t value) {
                return value == selection_group;
            }));
        EXPECT_EQ(
            chapters.parameters.dataset.images,
            saved_name);
        EXPECT_EQ(
            chapters.selection
                .selected_node_uuids(),
            std::vector<lfs::core::Uuid>{
                model_uuid});

        const std::string checkpoint_string(
            reinterpret_cast<const char*>(
                captured->checkpoint_bytes->data()),
            captured->checkpoint_bytes->size());
        std::istringstream checkpoint_stream(
            checkpoint_string,
            std::ios::binary | std::ios::in);
        auto header =
            lfs::core::load_checkpoint_header(
                checkpoint_stream,
                captured->checkpoint_bytes->size());
        ASSERT_TRUE(header) << header.error();
        EXPECT_EQ(
            header->iteration,
            SAVED_ITERATION);
        EXPECT_TRUE(
            captured->metrics
                .pause_within_rig_gate);
    }

    TEST(TrainingSnapshotServiceTest,
         CapturesRepresentativeSceneValuesInSafePoint) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }

        constexpr std::size_t GAUSSIAN_COUNT =
            1'300'000;
        constexpr int CAMERA_COUNT = 200;
        constexpr int SAVED_ITERATION = 42;

        lfs::core::Scene scene;
        const auto model_id = scene.addSplat(
            "representative_model",
            make_snapshot_test_splat(
                GAUSSIAN_COUNT));
        ASSERT_NE(model_id, lfs::core::NULL_NODE);
        scene.setTrainingModelNode(model_id);
        const auto model_uuid =
            scene.getNodeUuid(model_id);
        ASSERT_FALSE(model_uuid.is_nil());
        const std::array selected_node_uuids{
            model_uuid};

        const auto camera_group =
            scene.addGroup("Cameras");
        for (int index = 0;
             index < CAMERA_COUNT; ++index) {
            const auto name =
                std::format(
                    "camera_{:04}.png", index);
            ASSERT_NE(
                scene.addCamera(
                    name, camera_group,
                    make_snapshot_test_camera(index)),
                lfs::core::NULL_NODE);
        }

        const auto selection_group =
            scene.addSelectionGroup(
                "representative",
                {0.25f, 0.5f, 0.75f});
        ASSERT_NE(selection_group, 0);
        scene.setActiveSelectionGroup(
            selection_group);
        scene.applyPerNodeSelectionSlices(
            lfs::core::SelectionDomain::Splat,
            {{model_uuid,
              selection_mask(
                  GAUSSIAN_COUNT,
                  selection_group)}});

        auto params =
            make_snapshot_test_params(
                GAUSSIAN_COUNT);
        const auto snapshot_uuid =
            lfs::core::generate_uuid_v4();
        lfs::training::ProjectSnapshotCpuState
            cpu_state;
        const auto started =
            std::chrono::steady_clock::now();
        auto metrics =
            lfs::training::
                capture_project_snapshot_cpu_state(
                    scene, params, snapshot_uuid,
                    SAVED_ITERATION, cpu_state,
                    selected_node_uuids);
        const double capture_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                started)
                .count();
        std::cout
            << "[ P7 METRIC ] representative CPU value capture: "
            << capture_ms << " ms (SCNG="
            << (metrics ? metrics->scng_ms : 0.0)
            << " ms, SELM="
            << (metrics ? metrics->selm_ms : 0.0)
            << " ms, PRMS="
            << (metrics ? metrics->prms_ms : 0.0)
            << " ms)\n";
        ASSERT_TRUE(metrics)
            << lfs::format_for_developer(
                   metrics.error());
        EXPECT_EQ(cpu_state.snapshot_uuid,
                  snapshot_uuid);
        EXPECT_EQ(cpu_state.iteration,
                  SAVED_ITERATION);
        EXPECT_EQ(
            cpu_state.scene_graph.nodes.size(),
            static_cast<std::size_t>(
                CAMERA_COUNT + 2));
        lfs::training::ProjectSnapshotChapters
            chapters;
        auto materialized =
            lfs::training::
                materialize_project_snapshot_cpu_chapters(
                    std::move(cpu_state), chapters);
        ASSERT_TRUE(materialized)
            << lfs::format_for_developer(
                   materialized.error());
        EXPECT_EQ(chapters.snapshot_uuid,
                  snapshot_uuid);
        EXPECT_EQ(chapters.iteration,
                  SAVED_ITERATION);
        EXPECT_EQ(
            chapters.selection
                .selected_node_uuids(),
            std::vector<lfs::core::Uuid>{
                model_uuid});
    }

    TEST(TrainingStepRegressionTrackerTest,
         SelectsDisclosedSteadyWindowsWithoutDensifyEvents) {
        lfs::training::TrainingStepRegressionTracker
            tracker(4);
        for (int iteration = 1;
             iteration <= 4; ++iteration) {
            tracker.observe(iteration, 20.0, false);
        }
        tracker.observe(5, 200.0, true);
        for (int iteration = 6;
             iteration <= 9; ++iteration) {
            tracker.observe(iteration, 10.0, false);
        }
        tracker.arm_after_snapshot(10);
        tracker.observe(11, 10.9, false);
        tracker.observe(12, 10.9, false);
        tracker.observe(13, 200.0, true);
        for (int iteration = 14;
             iteration <= 17; ++iteration) {
            tracker.observe(iteration, 10.9, false);
        }

        const auto metrics = tracker.metrics();
        EXPECT_EQ(
            metrics.pre_snapshot.first_iteration, 6);
        EXPECT_EQ(
            metrics.pre_snapshot.last_iteration, 9);
        EXPECT_EQ(
            metrics.pre_snapshot.sample_count, 4u);
        EXPECT_EQ(
            metrics.post_resume.first_iteration, 14);
        EXPECT_EQ(
            metrics.post_resume.last_iteration, 17);
        EXPECT_EQ(
            metrics.post_resume.sample_count, 4u);
        EXPECT_NEAR(
            metrics.regression_percent, 9.0, 1e-12);
        EXPECT_TRUE(metrics.gate_evaluated);
        EXPECT_TRUE(metrics.within_gate);
    }

} // namespace
