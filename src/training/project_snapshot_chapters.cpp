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
    capture_project_snapshot_cpu_chapters(
        const lfs::core::Scene& scene,
        const lfs::core::param::TrainingParameters&
            checkpoint_params,
        const lfs::core::Uuid& snapshot_uuid,
        const int iteration,
        ProjectSnapshotChapters& output,
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
            lfs::io::project::capture_scene_graph(
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
            lfs::io::project::capture_selection_chapter(
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
        auto parameters =
            capture_parameters(checkpoint_params);
        metrics.prms_ms =
            Milliseconds(Clock::now() - prms_begin)
                .count();
        if (!parameters) {
            return std::move(parameters)
                .error()
                .with_context(
                    "capture snapshot PRMS",
                    LFS_SOURCE_SITE_CURRENT());
        }

        ProjectSnapshotChapters staged;
        staged.snapshot_uuid = snapshot_uuid;
        staged.iteration = iteration;
        staged.scene_graph = std::move(*scene_graph);
        staged.selection = std::move(*selection);
        staged.parameters = std::move(*parameters);
        output = std::move(staged);
        return metrics;
    }

} // namespace lfs::training
