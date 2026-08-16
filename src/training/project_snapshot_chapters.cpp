/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "project_snapshot_chapters.hpp"

#include "core/scene.hpp"
#include "io/scene_chapter_adapter.hpp"

#include <chrono>
#include <format>
#include <string>
#include <system_error>

namespace lfs::training {

    namespace {

        using Clock = std::chrono::steady_clock;
        using Milliseconds =
            std::chrono::duration<double, std::milli>;

        [[nodiscard]] lfs::Error capture_error(
            const lfs::ErrorCode code,
            std::string detail) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Training,
                .user_message =
                    "The project snapshot CPU state could not be captured.",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        [[nodiscard]] lfs::Result<
            lfs::io::project::ParameterManagerSnapshot>
        capture_parameters(
            const lfs::core::param::TrainingParameters&
                checkpoint_params) {
            lfs::io::project::ParameterManagerSnapshot
                parameters;
            parameters.dataset =
                checkpoint_params.dataset;
            absolutize_dataset_path_for_snapshot(
                parameters.dataset.data_path);
            parameters.mcmc_session =
                lfs::core::param::
                    OptimizationParameters::mcmc_defaults();
            parameters.mrnf_session =
                lfs::core::param::
                    OptimizationParameters::mrnf_defaults();
            parameters.igs_session =
                lfs::core::param::
                    OptimizationParameters::igs_plus_defaults();
            parameters.mcmc_current =
                parameters.mcmc_session;
            parameters.mrnf_current =
                parameters.mrnf_session;
            parameters.igs_current =
                parameters.igs_session;
            parameters.active_strategy =
                std::string(
                    lfs::core::param::
                        canonical_strategy_name(
                            checkpoint_params
                                .optimization
                                .strategy));
            if (parameters.active_strategy ==
                lfs::core::param::kStrategyMCMC) {
                parameters.mcmc_session =
                    checkpoint_params.optimization;
                parameters.mcmc_current =
                    checkpoint_params.optimization;
            } else if (
                parameters.active_strategy ==
                lfs::core::param::kStrategyIGSPlus) {
                parameters.igs_session =
                    checkpoint_params.optimization;
                parameters.igs_current =
                    checkpoint_params.optimization;
            } else if (
                parameters.active_strategy ==
                lfs::core::param::kStrategyMRNF) {
                parameters.mrnf_session =
                    checkpoint_params.optimization;
                parameters.mrnf_current =
                    checkpoint_params.optimization;
            } else {
                return capture_error(
                    lfs::ErrorCode::Unsupported,
                    std::format(
                        "Snapshot strategy '{}' is not registered",
                        checkpoint_params
                            .optimization.strategy));
            }
            return parameters;
        }

    } // namespace

    lfs::Result<TrainingSnapshotCpuStateMetrics>
    capture_project_snapshot_cpu_state(
        const lfs::core::Scene& scene,
        const lfs::io::project::ParameterManagerSnapshot&
            parameters,
        const lfs::core::Uuid& snapshot_uuid,
        const int iteration,
        ProjectSnapshotCpuState& output,
        const std::span<const lfs::core::Uuid>
            selected_node_uuids) {
        if (snapshot_uuid.is_nil()) {
            return capture_error(
                lfs::ErrorCode::InvalidArgument,
                "Snapshot CPU chapters require a non-null UUID");
        }
        if (iteration < 0) {
            return capture_error(
                lfs::ErrorCode::InvalidArgument,
                "Snapshot CPU chapters require a non-negative iteration");
        }
        const auto training_uuid =
            scene.getTrainingModelNodeUuid();
        if (training_uuid.is_nil()) {
            return capture_error(
                lfs::ErrorCode::FailedPrecondition,
                "Snapshot scene has no training-model UUID");
        }

        TrainingSnapshotCpuStateMetrics metrics;
        lfs::io::project::ScenePayloadBindings bindings;
        bindings.emplace(
            training_uuid,
            lfs::io::project::PayloadBinding{
                .fourcc = "CKPT",
                .instance_uuid = snapshot_uuid,
                .source_kind = "checkpoint",
            });

        const auto scng_begin = Clock::now();
        auto scene_graph =
            lfs::io::project::capture_scene_graph_state(
                scene, bindings);
        metrics.scng_ms =
            Milliseconds(Clock::now() - scng_begin)
                .count();
        if (!scene_graph) {
            return std::move(scene_graph)
                .error()
                .with_context(
                    "capture snapshot SCNG",
                    LFS_SOURCE_SITE_CURRENT());
        }

        const auto selm_begin = Clock::now();
        auto selection =
            lfs::io::project::capture_selection_state(
                scene, selected_node_uuids);
        metrics.selm_ms =
            Milliseconds(Clock::now() - selm_begin)
                .count();
        if (!selection) {
            return std::move(selection)
                .error()
                .with_context(
                    "capture snapshot SELM",
                    LFS_SOURCE_SITE_CURRENT());
        }

        const auto prms_begin = Clock::now();
        auto captured_parameters = parameters;
        absolutize_dataset_path_for_snapshot(
            captured_parameters.dataset.data_path);
        metrics.prms_ms =
            Milliseconds(Clock::now() - prms_begin)
                .count();

        ProjectSnapshotCpuState staged;
        staged.snapshot_uuid = snapshot_uuid;
        staged.iteration = iteration;
        staged.scene_graph = std::move(*scene_graph);
        staged.selection = std::move(*selection);
        staged.parameters =
            std::move(captured_parameters);
        output = std::move(staged);
        return metrics;
    }

    lfs::Result<TrainingSnapshotCpuStateMetrics>
    capture_project_snapshot_cpu_state(
        const lfs::core::Scene& scene,
        const lfs::core::param::TrainingParameters&
            checkpoint_params,
        const lfs::core::Uuid& snapshot_uuid,
        const int iteration,
        ProjectSnapshotCpuState& output,
        const std::span<const lfs::core::Uuid>
            selected_node_uuids) {
        auto parameters =
            capture_parameters(checkpoint_params);
        if (!parameters) {
            return std::move(parameters)
                .error()
                .with_context(
                    "capture snapshot PRMS",
                    LFS_SOURCE_SITE_CURRENT());
        }
        return capture_project_snapshot_cpu_state(
            scene, *parameters, snapshot_uuid,
            iteration, output,
            selected_node_uuids);
    }

    lfs::Result<void>
    materialize_project_snapshot_cpu_chapters(
        ProjectSnapshotCpuState state,
        ProjectSnapshotChapters& output) {
        if (state.snapshot_uuid.is_nil() ||
            state.iteration < 0) {
            return lfs::Status::failure(
                capture_error(
                    lfs::ErrorCode::InvalidArgument,
                    "Detached snapshot CPU state has an invalid stamp"));
        }
        auto scene_graph =
            lfs::io::project::
                materialize_scene_graph_chapter(
                    std::move(state.scene_graph));
        if (!scene_graph) {
            return lfs::Status::failure(
                std::move(scene_graph)
                    .error()
                    .with_context(
                        "materialize snapshot SCNG",
                        LFS_SOURCE_SITE_CURRENT()));
        }
        auto selection =
            lfs::io::project::
                materialize_selection_chapter(
                    std::move(state.selection));
        if (!selection) {
            return lfs::Status::failure(
                std::move(selection)
                    .error()
                    .with_context(
                        "materialize snapshot SELM",
                        LFS_SOURCE_SITE_CURRENT()));
        }

        const auto context =
            std::move(output.document_context);
        ProjectSnapshotChapters staged;
        staged.snapshot_uuid = state.snapshot_uuid;
        staged.iteration = state.iteration;
        staged.scene_graph = std::move(*scene_graph);
        staged.selection = std::move(*selection);
        staged.parameters =
            std::move(state.parameters);
        staged.document_context = std::move(context);
        output = std::move(staged);
        return {};
    }

    void absolutize_dataset_path_for_snapshot(
        std::filesystem::path& path) {
        if (path.empty() || path.is_absolute())
            return;
        std::error_code error;
        auto absolute = std::filesystem::absolute(path, error);
        if (error)
            return;
        path = absolute.lexically_normal();
    }

} // namespace lfs::training
