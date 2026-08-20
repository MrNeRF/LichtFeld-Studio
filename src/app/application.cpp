/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "app/headless_recovery_document.hpp"
#include "app/headless_run_coordinator.hpp"
#include "control/command_api.hpp"
#include "core/checkpoint_format.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda_version.hpp"
#include "core/environment.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/image_loader.hpp"
#include "core/legacy_settings_migration.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/provenance.hpp"
#include "core/scene.hpp"
#include "core/session_breadcrumb.hpp"
#include "core/tensor.hpp"
#include "core/user_paths.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "io/cache_image_loader.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"
#include "tcp/include/tcp_publisher.hpp"
#include "tcp/include/tcp_responder.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/training/training_manager.hpp"
#include "visualizer/visualizer.hpp"

#include "app/mcp_gui_tools.hpp"
#include "io/loader.hpp"
#include "io/video/video_encoder.hpp"
#include "mcp/mcp_http_server.hpp"
#include "mcp/mcp_tools.hpp"
#include "python/runner.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "sequencer/timeline.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "visualizer/gui/layout_state.hpp"
#include "visualizer/gui/panels/python_scripts_panel.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer/gui/windows/video_extractor_dialog.hpp"
#include "visualizer/input/input_bindings.hpp"
#include "visualizer/preferences.hpp"
#include <cmath>
#include <condition_variable>
#include <cuda_runtime.h>
#include <future>
#include <mutex>
#include <print>
#include <rasterization_api.h>
#include <string>
#include <string_view>

#ifdef WIN32
#include <windows.h>
#endif

#ifndef LFS_MIN_SM
#error "LFS_MIN_SM must be defined by the build (CMakeLists.txt)"
#endif

namespace lfs::app {

    namespace {

        struct HeadlessPluginSignalGuard {
            HeadlessPluginSignalGuard() {
                python::set_plugin_preload_completion_hook(
                    &HeadlessRunCoordinator::install_signal_handlers);
            }

            ~HeadlessPluginSignalGuard() {
                python::set_plugin_preload_completion_hook(nullptr);
            }

            HeadlessPluginSignalGuard(const HeadlessPluginSignalGuard&) = delete;
            HeadlessPluginSignalGuard& operator=(
                const HeadlessPluginSignalGuard&) = delete;
        };

        [[nodiscard]] lfs::Error training_project_error(
            const lfs::ErrorCode code,
            std::string detail,
            const lfs::core::SourceSite source) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::App,
                .user_message =
                    "The training project could not be restored.",
                .detail = std::move(detail),
                .detection = source,
            });
        }

        std::expected<core::param::TrainingParameters, std::string> loadCheckpointParams(const core::param::TrainingParameters& params, core::Scene& scene) {
            LOG_INFO("Resuming from checkpoint: {}", core::path_to_utf8(*params.resume_checkpoint));

            auto params_result = core::load_checkpoint_params(*params.resume_checkpoint);
            if (!params_result) {
                return std::unexpected(std::format("Failed to load checkpoint params: {}", params_result.error()));
            }
            auto checkpoint_params = std::move(*params_result);

            if (!params.dataset.data_path.empty())
                checkpoint_params.dataset.data_path = params.dataset.data_path;
            if (!params.dataset.output_path.empty())
                checkpoint_params.dataset.output_path = params.dataset.output_path;
            if (!params.dataset.output_name.empty())
                checkpoint_params.dataset.output_name = params.dataset.output_name;

            // Runtime-only CLI controls are not part of the serialized
            // training state. Preserve them when a resume is requested so
            // --perf-bench (and its warmup) still applies to the resumed run.
            checkpoint_params.optimization.perf_bench = params.optimization.perf_bench;
            checkpoint_params.optimization.perf_bench_warmup = params.optimization.perf_bench_warmup;

            if (checkpoint_params.dataset.data_path.empty()) {
                return std::unexpected("Checkpoint has no dataset path and none provided via --data-path");
            }
            if (!std::filesystem::exists(checkpoint_params.dataset.data_path)) {
                return std::unexpected(std::format("Dataset path does not exist: {}", core::path_to_utf8(checkpoint_params.dataset.data_path)));
            }

            if (const auto result = training::validateDatasetPath(checkpoint_params); !result) {
                return std::unexpected(std::format("Dataset validation failed: {}", result.error()));
            }

            if (const auto result = training::loadTrainingDataIntoScene(checkpoint_params, scene); !result) {
                return std::unexpected(std::format("Failed to load training data: {}", result.error()));
            }

            for (const auto* node : scene.getNodes()) {
                if (node->type == core::NodeType::POINTCLOUD) {
                    scene.removeNode(node->name, false);
                    break;
                }
            }

            auto splat_result = core::load_checkpoint_splat_data(*params.resume_checkpoint);
            if (!splat_result) {
                return std::unexpected(std::format("Failed to load checkpoint splat data: {}", splat_result.error()));
            }

            auto splat_data = std::make_unique<core::SplatData>(std::move(*splat_result));
            const auto model_id = scene.addSplat("Model", std::move(splat_data), core::NULL_NODE);
            if (model_id == core::NULL_NODE) {
                return std::unexpected("Failed to add checkpoint training model to scene");
            }
            scene.setTrainingModelNode(model_id);

            checkpoint_params.resume_checkpoint = *params.resume_checkpoint;
            return checkpoint_params;
        }

        struct LoadedTrainingProject {
            detail::HeadlessRecoveryDocument document;
            core::param::TrainingParameters params;
            core::Uuid checkpoint_uuid;
            int iteration = 0;
        };

        lfs::Result<LoadedTrainingProject>
        loadTrainingProject(
            const core::param::TrainingParameters& cli_params,
            core::Scene& scene) {
            if (!cli_params.resume_project) {
                return training_project_error(
                    lfs::ErrorCode::InvalidArgument,
                    "No .licht resume project was provided",
                    LFS_SOURCE_SITE_CURRENT());
            }
            const auto& path =
                *cli_params.resume_project;
            std::filesystem::path open_path = path;
            std::optional<
                io::project::RecoverySession>
                recovery_session;
            LOG_INFO(
                "Opening training project: {}",
                core::path_to_utf8(path));

            auto recovery =
                io::project::inspect_autosave_recovery(
                    path);
            if (!recovery) {
                return std::move(recovery)
                    .error()
                    .with_context(
                        "inspect training project autosave",
                        LFS_SOURCE_SITE_CURRENT());
            }
            if (recovery->disposition ==
                    io::project::RecoveryDisposition::Offer &&
                recovery->selected_path) {
                auto session =
                    io::project::
                        begin_recovery_session(
                            path,
                            *recovery
                                 ->selected_path);
                if (!session) {
                    return std::move(session)
                        .error()
                        .with_context(
                            "hold headless recovery master lock",
                            LFS_SOURCE_SITE_CURRENT());
                }
                open_path = io::project::
                    recovery_session_temp_path(
                        path);
                auto materialized =
                    io::project::
                        materialize_recovered_project(
                            path,
                            *recovery
                                 ->selected_path,
                            open_path, *session);
                if (!materialized) {
                    return std::move(
                               materialized)
                        .error()
                        .with_context(
                            "materialize headless autosave recovery",
                            LFS_SOURCE_SITE_CURRENT());
                }
                recovery_session.emplace(
                    std::move(*session));
                LOG_INFO(
                    "Automatically recovering autosave sequence {} for headless project open",
                    recovery
                        ->autosave_sequence);
            }

            auto document =
                io::project::ProjectDocument::open(
                    open_path);
            if (!document) {
                if (recovery_session) {
                    static_cast<void>(
                        recovery_session
                            ->release());
                }
                return std::move(document)
                    .error()
                    .with_context(
                        "open training project",
                        LFS_SOURCE_SITE_CURRENT());
            }
            detail::HeadlessRecoveryDocument
                recovery_document(
                    std::move(*document),
                    std::move(recovery_session));
            const auto checkpoint_uuids =
                recovery_document.document()
                    .checkpoint_uuids();
            if (checkpoint_uuids.size() != 1) {
                return training_project_error(
                    lfs::ErrorCode::DataLoss,
                    std::format(
                        "Training project must contain exactly one CKPT "
                        "instance (found {})",
                        checkpoint_uuids.size()),
                    LFS_SOURCE_SITE_CURRENT());
            }
            const auto checkpoint_uuid =
                checkpoint_uuids.front();
            const auto* checkpoint =
                recovery_document.document()
                    .find_checkpoint(checkpoint_uuid);
            if (!checkpoint) {
                return training_project_error(
                    lfs::ErrorCode::ContractViolation,
                    "Training project CKPT handle disappeared",
                    LFS_SOURCE_SITE_CURRENT());
            }

            std::optional<
                core::CheckpointParametersLoadResult>
                parsed_params;
            auto visited =
                checkpoint->visit_stream(
                    [&](std::istream& source,
                        const std::uint64_t bytes)
                        -> lfs::Result<void> {
                        parsed_params =
                            core::load_checkpoint_params(
                                source, bytes);
                        return {};
                    });
            if (!visited) {
                return std::move(visited)
                    .error()
                    .with_context(
                        "stream CKPT parameters",
                        LFS_SOURCE_SITE_CURRENT());
            }
            if (!parsed_params ||
                !*parsed_params) {
                return training_project_error(
                    lfs::ErrorCode::DataLoss,
                    parsed_params
                        ? std::format(
                              "Failed to parse CKPT parameters: {}",
                              parsed_params->error())
                        : "CKPT parameter visitor did not run",
                    LFS_SOURCE_SITE_CURRENT());
            }
            auto checkpoint_params =
                std::move(**parsed_params);

            if (!cli_params.dataset.data_path.empty()) {
                checkpoint_params.dataset.data_path =
                    cli_params.dataset.data_path;
            }
            if (!cli_params.dataset.output_path.empty()) {
                checkpoint_params.dataset.output_path =
                    cli_params.dataset.output_path;
            }
            if (!cli_params.dataset.output_name.empty()) {
                checkpoint_params.dataset.output_name =
                    cli_params.dataset.output_name;
            }
            if (cli_params.cli_iterations_set) {
                checkpoint_params.optimization.iterations =
                    cli_params.optimization.iterations;
            }
            checkpoint_params.optimization.headless =
                cli_params.optimization.headless;
            checkpoint_params.optimization.auto_train =
                cli_params.optimization.auto_train;
            checkpoint_params.optimization.no_splash =
                cli_params.optimization.no_splash;
            checkpoint_params.server =
                cli_params.server;
            checkpoint_params.python_scripts =
                cli_params.python_scripts;
            checkpoint_params.resume_checkpoint.reset();
            checkpoint_params.resume_project = path;
            checkpoint_params.save_project_at_iteration =
                cli_params.save_project_at_iteration;
            checkpoint_params.save_project_path =
                cli_params.save_project_path;
            checkpoint_params.cli_iterations_set =
                cli_params.cli_iterations_set;

            auto hydration =
                recovery_document.document()
                    .hydrate(scene);
            if (!hydration) {
                return std::move(hydration)
                    .error()
                    .with_context(
                        "hydrate training project display state",
                        LFS_SOURCE_SITE_CURRENT());
            }
            if (!hydration->trainer_state_pending ||
                !hydration->checkpoint_uuid ||
                *hydration->checkpoint_uuid !=
                    checkpoint_uuid ||
                !hydration->checkpoint_header) {
                return training_project_error(
                    lfs::ErrorCode::ContractViolation,
                    "Project hydration did not preserve the lazy CKPT "
                    "trainer-state barrier",
                    LFS_SOURCE_SITE_CURRENT());
            }
            if (hydration->checkpoint_header->iteration < 0) {
                return training_project_error(
                    lfs::ErrorCode::DataLoss,
                    "Project CKPT iteration is negative",
                    LFS_SOURCE_SITE_CURRENT());
            }

            return LoadedTrainingProject{
                .document =
                    std::move(recovery_document),
                .params =
                    std::move(checkpoint_params),
                .checkpoint_uuid =
                    checkpoint_uuid,
                .iteration =
                    hydration
                        ->checkpoint_header
                        ->iteration,
            };
        }

        int runHeadlessWithTCP(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->dataset.data_path.empty() &&
                !params->resume_checkpoint &&
                !params->resume_project) {
                LOG_ERROR("Headless with TCP mode requires --data-path or --resume");
                return 1;
            }

            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());
            HeadlessRunCoordinator coordinator;
            HeadlessPluginSignalGuard plugin_signals;

            {
                core::Scene scene;
                std::optional<core::param::TrainingParameters> checkpoint_params{std::nullopt};
                std::optional<LoadedTrainingProject>
                    training_project;

                if (params->resume_project) {
                    auto loaded =
                        loadTrainingProject(
                            *params, scene);
                    if (!loaded) {
                        LOG_ERROR(
                            "Failed to load training project: {}",
                            lfs::format_for_developer(
                                loaded.error()));
                        return 1;
                    }
                    checkpoint_params =
                        loaded->params;
                    training_project.emplace(
                        std::move(*loaded));
                } else if (params->resume_checkpoint) {
                    const auto ckpt_params_result = loadCheckpointParams(*params, scene);
                    if (!ckpt_params_result) {
                        LOG_ERROR("Failed to load checkpoint: {}", ckpt_params_result.error());
                        return 1;
                    }
                    checkpoint_params = *ckpt_params_result;
                } else {
                    LOG_INFO("Starting headless with TCP training...");

                    if (const auto result = training::loadTrainingDataIntoScene(*params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::initializeTrainingModel(*params, scene); !result) {
                        LOG_ERROR("Failed to initialize model: {}", result.error());
                        return 1;
                    }
                }

                auto manager = std::make_shared<vis::TrainerManager>();
                {
                    const auto& effective_params =
                        checkpoint_params
                            ? *checkpoint_params
                            : *params;

                    if (!effective_params.python_scripts.empty()) {
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(effective_params.python_scripts);
                    }

                    if (training_project) {
                        std::optional<
                            io::project::RecoverySession>
                            recovery_session;
                        if (const auto* session =
                                training_project->document
                                    .recovery_session()) {
                            recovery_session = *session;
                        }
                        auto installed =
                            training::
                                installTrainerFromProjectCheckpoint(
                                    scene,
                                    training_project
                                        ->document
                                        .document(),
                                    training_project
                                        ->checkpoint_uuid,
                                    effective_params,
                                    core::path_to_utf8(
                                        *params
                                             ->resume_project),
                                    training_project
                                        ->iteration,
                                    recovery_session);
                        if (!installed) {
                            LOG_ERROR(
                                "Failed to complete project CKPT "
                                "hydration before TCP training: {}",
                                installed.error());
                            return 1;
                        }
                        training::grant_headless_project_saves(
                            *installed->trainer,
                            effective_params,
                            *params->resume_project);
                        manager->setTrainer(
                            std::move(installed->trainer));
                    } else {
                        // Legacy .resume auto-load remains unchanged.
                        auto trainer =
                            std::make_unique<training::Trainer>(
                                scene);
                        if (!effective_params.python_scripts
                                 .empty()) {
                            trainer->set_python_scripts(
                                effective_params
                                    .python_scripts);
                        }
                        trainer->setParams(effective_params);
                        training::grant_headless_project_saves(
                            *trainer, effective_params);
                        manager->setTrainer(std::move(trainer));
                    }
                }

                core::Tensor::trim_memory_pool();

                try {
                    tcp::ResponderServer responder(params->server.tcp_server_connection_port, manager);
                    tcp::PublisherServer publisher(params->server.tcp_broadcast_connection_port, manager);

                    responder.start();
                    publisher.start();
                    LOG_INFO("Responder server listening on {}", responder.getEndpoint());
                    LOG_INFO("Publisher server listening on {}", publisher.getEndpoint());

                    std::mutex completion_mutex;
                    std::condition_variable completion_cv;
                    std::optional<core::events::state::TrainingCompleted> completion;
                    lfs::event::ScopedHandler completion_subscription;
                    completion_subscription.subscribe<core::events::state::TrainingCompleted>(
                        [&](const core::events::state::TrainingCompleted& event) {
                            {
                                std::lock_guard lock(completion_mutex);
                                if (completion) {
                                    return;
                                }
                                completion = event;
                            }
                            completion_cv.notify_all();
                        });

                    if (!manager->startTraining()) {
                        throw std::runtime_error("Failed to start TCP headless training");
                    }

                    bool stop_requested = false;
                    std::optional<std::chrono::steady_clock::time_point> stop_deadline;
                    std::string completion_error;
                    std::unique_lock completion_lock(completion_mutex);
                    while (!completion) {
                        completion_cv.wait_for(completion_lock, std::chrono::milliseconds(100));
                        if (completion) {
                            break;
                        }

                        if (coordinator.interrupted() && !stop_requested) {
                            stop_requested = true;
                            stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                            completion_lock.unlock();
                            LOG_INFO("Interrupt signal received, requesting TCP training stop");
                            manager->stopTraining();
                            completion_lock.lock();
                        }

                        if (manager->isFinished()) {
                            completion_error = "Training reached a terminal state without a completion event";
                            break;
                        }
                        if (stop_deadline && std::chrono::steady_clock::now() >= *stop_deadline) {
                            completion_error = "Timed out waiting for interrupted TCP training to save and stop";
                            break;
                        }
                    }
                    completion_lock.unlock();

                    if (!manager->waitForCompletion()) {
                        throw std::runtime_error("Training worker did not finish after terminal event");
                    }

                    if (!completion_error.empty()) {
                        throw std::runtime_error(completion_error);
                    }
                    if (!completion) {
                        throw std::runtime_error("TCP training completion was not observed");
                    }

                    publisher.stop();
                    responder.stop();
                    responder.join();
                } catch (const std::exception& e) {
                    LOG_ERROR("Headless TCP lifecycle failed: {}", e.what());
                    return 1;
                }

                if (manager->getStateMachine().getFinishReason() == vis::FinishReason::Error) {
                    LOG_ERROR("Training error: {}", manager->getLastError());
                    if (!params->python_scripts.empty()) {
                        core::teardown_gpu_before_exit();
                        python::finalize();
                        core::flush_and_exit(1);
                    }
                    return 1;
                }

                if (training_project) {
                    if (auto rebound =
                            training_project->document
                                .rebind_after_durable_merge();
                        !rebound) {
                        LOG_ERROR(
                            "Headless TCP recovery merge could not rebind the live project document: {}",
                            lfs::format_for_developer(
                                rebound.error()));
                        return 1;
                    }
                }

                LOG_INFO("Headless with TCP training completed");
            }

            core::teardown_gpu_before_exit();

            const int exit_code = coordinator.interrupted() ? coordinator.interrupted_exit_code() : 0;
            if (!params->python_scripts.empty()) {
                python::finalize();
                core::flush_and_exit(exit_code);
            }
            return exit_code;
        }

        int runHeadless(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->dataset.data_path.empty() &&
                !params->resume_checkpoint &&
                !params->resume_project) {
                LOG_ERROR("Headless mode requires --data-path or --resume");
                return 1;
            }

            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());
            HeadlessRunCoordinator coordinator;
            HeadlessPluginSignalGuard plugin_signals;

            {
                core::Scene scene;

                if (params->resume_project) {
                    auto project =
                        loadTrainingProject(
                            *params, scene);
                    if (!project) {
                        LOG_ERROR(
                            "Failed to load training project: {}",
                            lfs::format_for_developer(
                                project.error()));
                        return 1;
                    }

                    if (!project->params
                             .python_scripts
                             .empty()) {
                        vis::gui::panels::
                            PythonScriptManagerState::
                                getInstance()
                                    .setScripts(
                                        project->params
                                            .python_scripts);
                    }
                    if (!project->params
                             .python_scripts
                             .empty() &&
                        !python::ensure_plugins_loaded(
                            true)) {
                        LOG_ERROR(
                            "Failed to load plugins before running "
                            "headless Python scripts");
                        return 1;
                    }
                    std::optional<
                        io::project::RecoverySession>
                        recovery_session;
                    if (const auto* session =
                            project->document
                                .recovery_session()) {
                        recovery_session = *session;
                    }
                    auto installed =
                        training::
                            installTrainerFromProjectCheckpoint(
                                scene,
                                project->document
                                    .document(),
                                project
                                    ->checkpoint_uuid,
                                project->params,
                                core::path_to_utf8(
                                    *params
                                         ->resume_project),
                                project->iteration,
                                recovery_session);
                    if (!installed) {
                        LOG_ERROR(
                            "Failed to restore project trainer state: {}",
                            installed.error());
                        return 1;
                    }
                    auto trainer =
                        std::move(installed->trainer);
                    training::grant_headless_project_saves(
                        *trainer, project->params,
                        *params->resume_project);
                    LOG_INFO(
                        "Project display hydration complete; full "
                        "trainer state restored at iteration {}",
                        project->iteration);

                    core::Tensor::trim_memory_pool();
                    if (const auto result =
                            trainer->train(
                                coordinator
                                    .stop_token());
                        !result) {
                        LOG_ERROR(
                            "Training error: {}",
                            lfs::format_for_developer(
                                result.error()));
                        return 1;
                    }
                    if (auto rebound =
                            project->document
                                .rebind_after_durable_merge();
                        !rebound) {
                        LOG_ERROR(
                            "Headless recovery merge could not rebind the live project document: {}",
                            lfs::format_for_developer(
                                rebound.error()));
                        return 1;
                    }
                    trainer->shutdown();
                    static_cast<void>(
                        trainer.release());
                } else if (params->resume_checkpoint) {
                    const auto ckpt_params_result = loadCheckpointParams(*params, scene);
                    if (!ckpt_params_result) {
                        LOG_ERROR("Failed to load checkpoint: {}", ckpt_params_result.error());
                        return 1;
                    }

                    auto trainer = std::make_unique<training::Trainer>(scene);

                    if (!params->python_scripts.empty()) {
                        trainer->set_python_scripts(params->python_scripts);
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    }

                    if (!params->python_scripts.empty() && !python::ensure_plugins_loaded(true)) {
                        LOG_ERROR("Failed to load plugins before running headless Python scripts");
                        return 1;
                    }

                    if (const auto result = trainer->initialize(*ckpt_params_result); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }
                    training::grant_headless_project_saves(
                        *trainer, *ckpt_params_result);

                    const auto ckpt_result = trainer->load_checkpoint(*params->resume_checkpoint);
                    if (!ckpt_result) {
                        LOG_ERROR("Failed to restore checkpoint state: {}", ckpt_result.error());
                        return 1;
                    }
                    LOG_INFO("Resumed from iteration {}", *ckpt_result);

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(coordinator.stop_token()); !result) {
                        LOG_ERROR("Training error: {}", lfs::format_for_developer(result.error()));
                        if (!params->python_scripts.empty()) {
                            core::teardown_gpu_before_exit();
                            python::finalize();
                            core::flush_and_exit(1);
                        }
                        return 1;
                    }
                    trainer->shutdown();
                    static_cast<void>(trainer.release());
                } else {
                    LOG_INFO("Starting headless training...");

                    if (const auto result = training::loadTrainingDataIntoScene(*params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::initializeTrainingModel(*params, scene); !result) {
                        LOG_ERROR("Failed to initialize model: {}", result.error());
                        return 1;
                    }

                    auto trainer = std::make_unique<training::Trainer>(scene);

                    if (!params->python_scripts.empty()) {
                        trainer->set_python_scripts(params->python_scripts);
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    }

                    if (!params->python_scripts.empty() && !python::ensure_plugins_loaded(true)) {
                        LOG_ERROR("Failed to load plugins before running headless Python scripts");
                        return 1;
                    }

                    if (const auto result = trainer->initialize(*params); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }
                    training::grant_headless_project_saves(
                        *trainer, *params);

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(coordinator.stop_token()); !result) {
                        LOG_ERROR("Training error: {}", lfs::format_for_developer(result.error()));
                        if (!params->python_scripts.empty()) {
                            core::teardown_gpu_before_exit();
                            python::finalize();
                            core::flush_and_exit(1);
                        }
                        return 1;
                    }
                    trainer->shutdown();
                    static_cast<void>(trainer.release());
                }

                LOG_INFO("Headless training {}",
                         coordinator.interrupted() ? "stopped by user" : "completed");
                core::teardown_gpu_before_exit();
                core::mark_clean_exit();
                core::flush_and_exit(0);
            }

            core::teardown_gpu_before_exit();

            const int exit_code = coordinator.interrupted() ? coordinator.interrupted_exit_code() : 0;
            if (!params->python_scripts.empty()) {
                python::finalize();
                core::flush_and_exit(exit_code);
            }
            return exit_code;
        }

        // Renders a sequencer camera path against a trained scene to a video file, headless.
        int runHeadlessRender(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            const auto& cfg = *params->render_path;

            // Load the trained scene.
            std::shared_ptr<core::SplatData> model;
            const auto load_ext = cfg.load_path.extension().string();
            if (load_ext == ".resume") {
                auto splat_result = core::load_checkpoint_splat_data(cfg.load_path);
                if (!splat_result) {
                    LOG_ERROR("Failed to load checkpoint: {}", splat_result.error());
                    return 1;
                }
                model = std::make_shared<core::SplatData>(std::move(*splat_result));
            } else {
                auto loader = lfs::io::Loader::create();
                auto load_result = loader->load(cfg.load_path);
                if (!load_result) {
                    LOG_ERROR("Failed to load scene: {}", load_result.error().message);
                    return 1;
                }
                if (auto* const splat = std::get_if<std::shared_ptr<core::SplatData>>(&load_result->data)) {
                    model = *splat;
                } else {
                    LOG_ERROR("--render-load is not a Gaussian splat scene: {}", core::path_to_utf8(cfg.load_path));
                    return 1;
                }
            }
            if (!model || model->size() == 0) {
                LOG_ERROR("Loaded scene has no Gaussians: {}", core::path_to_utf8(cfg.load_path));
                return 1;
            }

            lfs::sequencer::Timeline timeline;
            if (!timeline.loadFromJson(core::path_to_utf8(cfg.camera_path)) || timeline.empty()) {
                LOG_ERROR("Failed to load camera path (or it has no keyframes): {}", core::path_to_utf8(cfg.camera_path));
                return 1;
            }

            // Solid black background, matching Trainer's default bg_color init.
            auto background = core::Tensor::empty({3}, core::Device::CPU, core::DataType::Float32);
            {
                auto* const bg_ptr = background.ptr<float>();
                bg_ptr[0] = bg_ptr[1] = bg_ptr[2] = 0.0f;
            }
            background = background.to(core::Device::CUDA);

            lfs::io::video::VideoEncoder encoder;
            lfs::io::video::VideoExportOptions options;
            options.preset = lfs::io::video::VideoPreset::CUSTOM;
            options.width = cfg.width;
            options.height = cfg.height;
            options.framerate = cfg.fps;
            options.crf = cfg.crf;
            options.provenance = cfg.include_provenance ? core::make_provenance_stamp()
                                                        : core::make_minimal_provenance_stamp();
            if (const auto open_result = encoder.open(cfg.output_path, options); !open_result) {
                LOG_ERROR("Failed to open video encoder: {}", open_result.error());
                return 1;
            }

            const float duration = timeline.duration();
            const int total_frames = static_cast<int>(std::ceil(duration * cfg.fps)) + 1;
            LOG_INFO("Rendering {} frame(s) ({:.2f}s @ {}fps) from {} to {}",
                     total_frames, duration, cfg.fps,
                     core::path_to_utf8(cfg.camera_path), core::path_to_utf8(cfg.output_path));

            for (int frame = 0; frame < total_frames; ++frame) {
                const float t = std::min(static_cast<float>(frame) / static_cast<float>(cfg.fps), duration);
                const auto cam_state = timeline.evaluate(t);

                // CameraState is camera-to-world; Camera's R/T are world-to-camera, so invert.
                const glm::mat3 r_c2w = glm::mat3_cast(cam_state.rotation);
                const glm::mat3 r_w2c = glm::transpose(r_c2w);
                const glm::vec3 t_w2c = -(r_w2c * cam_state.position);

                std::vector<float> r_flat(9);
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        r_flat[row * 3 + col] = r_w2c[col][row]; // glm is column-major
                    }
                }
                auto R = core::Tensor::from_vector(r_flat, {3, 3}, core::Device::CPU);
                auto T = core::Tensor::from_vector(
                    std::vector<float>{t_w2c.x, t_w2c.y, t_w2c.z}, {3}, core::Device::CPU);

                const auto [focal_x, focal_y] = rendering::computePixelFocalLengths(
                    {cfg.width, cfg.height}, cam_state.focal_length_mm);

                core::Camera camera(
                    R, T,
                    focal_x, focal_y,
                    static_cast<float>(cfg.width) * 0.5f, static_cast<float>(cfg.height) * 0.5f,
                    core::Tensor::empty({0}, core::Device::CPU),
                    core::Tensor::empty({0}, core::Device::CPU),
                    core::CameraModelType::PINHOLE,
                    std::format("frame_{:06d}", frame),
                    {}, {},
                    cfg.width, cfg.height,
                    frame);

                auto render_output = training::fast_rasterize(camera, *model, background);
                auto image = render_output.image;
                if (image.dtype() != core::DataType::Float32) {
                    image = image.to(core::DataType::Float32);
                }
                if (image.device() != core::Device::CUDA) {
                    image = image.cuda();
                }
                auto image_hwc = image.permute({1, 2, 0}).contiguous();

                const auto write_result = encoder.writeFrameGpu(image_hwc.data_ptr(), cfg.width, cfg.height, nullptr);
                if (!write_result) {
                    LOG_ERROR("Failed to encode frame {}: {}", frame, write_result.error());
                    if (const auto close_result = encoder.close(); !close_result)
                        LOG_WARN("Failed to finalize partial video: {}", close_result.error());
                    return 1;
                }
                LOG_INFO("Encoded frame {}/{}", frame + 1, total_frames);
            }

            if (const auto close_result = encoder.close(); !close_result) {
                LOG_ERROR("Failed to finalize video: {}", close_result.error());
                return 1;
            }

            LOG_INFO("Wrote {} frame(s) to {}", total_frames, core::path_to_utf8(cfg.output_path));
            core::mark_clean_exit();
            return 0;
        }

        // Only an accurate paraphrase of SM 7.5 — Turing also covers the GTX 16-series and T4,
        // so this must not say "RTX only". Drop the hint if the floor ever moves.
        constexpr std::string_view kMinGpuHint =
            LFS_MIN_SM == 75 ? " Cards from the GTX 16-series, RTX 20-series and newer qualify." : "";

        // English literals on purpose: this runs before the visualizer exists, so
        // LocalizationManager has no catalog loaded yet. Do not convert to LOC(...).
        void reportFatalStartupError(const std::string& title, const std::string& message,
                                     const bool show_dialog) {
            // Only the training argument path reaches Logger::init, so for convert, mesh2splat,
            // preprocess and --warmup a LOG_ERROR would be dropped and the refusal would look
            // like a silent exit. Same fallback contract as error_reporter.cpp.
            if (core::Logger::get().is_ready()) {
                LOG_ERROR("{}", message);
            } else {
                std::println(stderr, "{}", message);
            }
#ifdef WIN32
            if (show_dialog) {
                MessageBoxW(nullptr, core::utf8_to_wstring(message).c_str(),
                            core::utf8_to_wstring(title).c_str(), MB_ICONERROR | MB_OK);
            }
#else
            (void)title;
            (void)show_dialog;
#endif
        }

    } // namespace

    // The binary only carries code for SM >= LFS_MIN_SM, so a lower card can never JIT our
    // kernels. Without this gate the first launch inside warmup_kernels dies with no
    // user-facing message (#1540). show_dialog is false for CLI-only modes: a modal in a
    // non-interactive process blocks it forever.
    bool preflightGpu(const bool show_dialog) {
        const auto info = lfs::core::check_cuda_version();
        if (info.query_failed) {
            LOG_WARN("Failed to query CUDA driver version");
        } else {
            LOG_INFO("CUDA driver version: {}.{}", info.major, info.minor);
            if (!info.supported) {
                reportFatalStartupError(
                    "LichtFeld Studio - Incompatible driver",
                    std::format("CUDA {}.{} is too old. LichtFeld Studio requires CUDA 12.8 or "
                                "newer, which needs NVIDIA driver 570 or newer.",
                                info.major, info.minor),
                    show_dialog);
                return false;
            }
        }

        cudaDeviceProp prop{};
        if (const cudaError_t err = cudaGetDeviceProperties(&prop, 0); err != cudaSuccess) {
            reportFatalStartupError(
                "LichtFeld Studio - No usable GPU",
                std::format("No usable NVIDIA GPU found ({}). LichtFeld Studio requires an "
                            "NVIDIA GPU with compute capability {}.{} or newer.{}",
                            cudaGetErrorString(err), LFS_MIN_SM / 10, LFS_MIN_SM % 10, kMinGpuHint),
                show_dialog);
            return false;
        }

        LOG_INFO("GPU: {} (SM {}.{}, {} MB)", prop.name, prop.major, prop.minor,
                 prop.totalGlobalMem / (1024 * 1024));

        if (const int device_sm = prop.major * 10 + prop.minor; device_sm < LFS_MIN_SM) {
            reportFatalStartupError(
                "LichtFeld Studio - Incompatible GPU",
                std::format("This PC's GPU ({}) has compute capability {}.{}, below the {}.{} "
                            "this build requires.{}",
                            prop.name, prop.major, prop.minor, LFS_MIN_SM / 10, LFS_MIN_SM % 10,
                            kMinGpuHint),
                show_dialog);
            return false;
        }
        return true;
    }

    namespace {

        std::future<void>& cudaWarmupFuture() {
            static std::future<void> fut;
            return fut;
        }

        void warmupCudaAsync() {
            LOG_INFO("Initializing CUDA (async)...");
            cudaWarmupFuture() = std::async(std::launch::async, [] {
                fast_lfs::rasterization::warmup_kernels();
                lfs::diagnostics::VramProfiler::instance().captureCudaWarmupDelta();
            });
        }

        int runGui(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            const bool safe_mode = params->safe_mode ||
                                   lfs::core::environment::flag("LFS_SAFE_MODE", false);
            python::set_user_plugin_loading_enabled(!safe_mode);
            vis::gui::LayoutState::setPersistenceEnabled(!safe_mode);
            vis::input::InputBindings::setPersistenceEnabled(!safe_mode);
            if (const auto paths = lfs::core::UserPaths::resolve()) {
                if (!safe_mode) {
                    if (const auto migration = lfs::core::migrateLegacySettings(*paths); !migration)
                        LOG_WARN("Unable to migrate legacy user settings: {}",
                                 lfs::format_for_developer(migration.error()));
                }
                const auto reset_file = [&paths](const bool requested, const char* const label,
                                                 const auto& reset) {
                    if (!requested)
                        return;
                    const auto result = reset();
                    if (!result) {
                        LOG_ERROR("Unable to reset {}: {}", label,
                                  lfs::format_for_developer(result.error()));
                    } else if (*result) {
                        LOG_INFO("Reset {}. Backup saved to {}", label,
                                 lfs::core::path_to_utf8(**result));
                    } else {
                        LOG_INFO("Reset {}. No existing settings file required a backup", label);
                    }
                };
                const bool reset_preferences = params->reset_preferences || params->reset_all_settings;
                const bool reset_layout = params->reset_layout || params->reset_all_settings;
                reset_file(reset_preferences, "preferences", [&paths] { return paths->resetPreferences(); });
                reset_file(reset_layout, "layout", [&paths] { return paths->resetLayout(); });
                reset_file(reset_layout, "UI preferences", [&paths] { return paths->resetUiPreferences(); });
                reset_file(params->reset_all_settings, "window", [&paths] { return paths->resetWindowState(); });
                reset_file(params->reset_all_settings, "project lifecycle", [&paths] {
                    return paths->resetProjectLifecycle();
                });

            } else {
                LOG_WARN("Unable to resolve user settings path: {}",
                         lfs::format_for_developer(paths.error()));
            }

            if (safe_mode) {
                LOG_WARN("Safe mode active: user plugin loading is disabled for this process");
            }

            const std::string window_title = safe_mode
                                                 ? "LichtFeld Studio (Safe Mode)"
                                                 : "LichtFeld Studio";

            if (!params->python_scripts.empty()) {
                vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
            }

            const bool disable_splash =
#ifdef LFS_BUILD_PORTABLE
                false;
#else
                params->optimization.no_splash;
#endif

            // Warm up on every path, not just import/resume: warmup_kernels forces the
            // lazily-loaded cubins to upload so captureCudaWarmupDelta can attribute that
            // module memory (the cuda.modules row). Without it the modules land in the
            // unattributed NVML residual. The pre-flight gate in run_mode covers
            // hardware compatibility before this warmup starts.
            warmupCudaAsync();

            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());

            lfs::gui::setVideoWidgetFactory([] {
                return std::make_unique<lfs::gui::VideoExtractorDialog>();
            });
            lfs::gui::setVideoEncoderFactory([] {
                return std::make_unique<lfs::io::video::VideoEncoder>();
            });

            constexpr auto graphics_backend = lfs::vis::GraphicsBackend::Vulkan;
            mcp::McpHttpServer mcp_http({.enable_resources = true});
            const auto mcp_preferences = vis::loadMcpPreferences();
            const auto startup_project =
                params->project_path
                    ? params->project_path
                    : params->resume_project;
            auto viewer = vis::Visualizer::create({
                .title = window_title,
                .width = 1280,
                .height = 720,
                .antialiasing = false,
                .show_startup_overlay = !disable_splash,
                .safe_mode = safe_mode,
                .mcp_status_provider = [] {
                    const auto status = mcp::activeMcpHttpStatus();
                    return vis::RuntimeServiceStatus{
                        .enabled = status.enabled,
                        .running = status.running,
                        .phase = [&] {
                            switch (status.phase) {
                            case mcp::McpHttpPhase::Disabled:
                                return vis::RuntimeServicePhase::Disabled;
                            case mcp::McpHttpPhase::Starting:
                                return vis::RuntimeServicePhase::Starting;
                            case mcp::McpHttpPhase::Running:
                                return vis::RuntimeServicePhase::Running;
                            case mcp::McpHttpPhase::Stopping:
                                return vis::RuntimeServicePhase::Stopping;
                            case mcp::McpHttpPhase::Failed:
                                return vis::RuntimeServicePhase::Failed;
                            }
                            return vis::RuntimeServicePhase::Failed; }(),
                        .network_exposed = status.expose_network,
                        .port = status.port,
                        .request_count = status.request_count,
                        .success_count = status.success_count,
                        .error_count = status.error_count,
                        .endpoints = status.endpoints,
                        .request_logging = status.request_logging,
                        .log_file = status.log_file,
                        .error = status.error,
                        .error_kind = [&] {
                            switch (status.error_kind) {
                            case mcp::McpHttpErrorKind::None:
                                return vis::RuntimeServiceErrorKind::None;
                            case mcp::McpHttpErrorKind::InvalidPort:
                                return vis::RuntimeServiceErrorKind::InvalidPort;
                            case mcp::McpHttpErrorKind::BindFailed:
                                return vis::RuntimeServiceErrorKind::BindFailed;
                            case mcp::McpHttpErrorKind::ListenerFailed:
                                return vis::RuntimeServiceErrorKind::RuntimeFailure;
                            }
                            return vis::RuntimeServiceErrorKind::RuntimeFailure; }(),
                        .error_address = status.error_address,
                        .error_port = status.error_port,
                    };
                },
                .gut = params->optimization.gut,
                .graphics_backend = graphics_backend,
                .startup_project = startup_project,
            });

            viewer->setParameters(*params);

            for (const auto& vp : params->view_paths) {
                if (!std::filesystem::exists(vp)) {
                    LOG_ERROR("File not found: {}", lfs::core::path_to_utf8(vp));
                    return 1;
                }
            }
            if (!params->dataset.data_path.empty() && !std::filesystem::exists(params->dataset.data_path)) {
                LOG_ERROR("Dataset not found: {}", lfs::core::path_to_utf8(params->dataset.data_path));
                return 1;
            }

            if (params->import_cameras_path ||
                params->resume_checkpoint ||
                startup_project) {
                if (auto& fut = cudaWarmupFuture(); fut.valid())
                    fut.wait();
            }

            if (params->import_cameras_path) {
                LOG_INFO("Importing COLMAP cameras: {}", lfs::core::path_to_utf8(*params->import_cameras_path));
                lfs::core::events::cmd::ImportColmapCameras{.sparse_path = *params->import_cameras_path}.emit();
            } else if (params->resume_checkpoint) {
                LOG_INFO("Loading checkpoint: {}", lfs::core::path_to_utf8(*params->resume_checkpoint));
                if (const auto result = viewer->loadCheckpointForTraining(*params->resume_checkpoint); !result) {
                    LOG_ERROR("Failed to load checkpoint: {}", result.error());
                    return 1;
                }
            }

            mcp::register_core_tools();
            mcp::register_core_resources();
            register_gui_scene_tools(viewer.get());
            register_gui_scene_resources(viewer.get());

            mcp::setActiveMcpHttpServer(&mcp_http);
            vis::setRuntimeServiceControls({
                .toggle_mcp_enabled = [safe_mode] {
                    if (safe_mode)
                        return false;
                    const auto status = mcp::activeMcpHttpStatus();
                    const mcp::McpHttpConfig config{
                        .enabled = !status.enabled,
                        .expose_network = status.expose_network,
                        .port = status.port,
                        .request_logging = status.request_logging,
                    };
                    if (!mcp::applyActiveMcpHttpConfig(config))
                        return false;
                    vis::saveMcpPreferences({
                        .enabled = config.enabled,
                        .expose_network = config.expose_network,
                        .port = config.port,
                        .request_logging = config.request_logging,
                    });
                    return true; },
                .toggle_mcp_binding = [safe_mode] {
                    if (safe_mode)
                        return false;
                    const auto status = mcp::activeMcpHttpStatus();
                    const mcp::McpHttpConfig config{
                        .enabled = status.enabled,
                        .expose_network = !status.expose_network,
                        .port = status.port,
                        .request_logging = status.request_logging,
                    };
                    if (!mcp::applyActiveMcpHttpConfig(config))
                        return false;
                    vis::saveMcpPreferences({
                        .enabled = config.enabled,
                        .expose_network = config.expose_network,
                        .port = config.port,
                        .request_logging = config.request_logging,
                    });
                    return true; },
            });
            viewer->setShutdownRequestedCallback([&mcp_http]() {
                vis::setRuntimeServiceControls({});
                mcp::setActiveMcpHttpServer(nullptr);
                mcp_http.stop();
            });
            if (!mcp_http.start({
                    .enabled = mcp_preferences.enabled,
                    .expose_network = mcp_preferences.expose_network,
                    .port = mcp_preferences.port,
                    .request_logging = mcp_preferences.request_logging,
                }))
                LOG_ERROR("Failed to start MCP HTTP server");

            viewer->run();

            vis::setRuntimeServiceControls({});
            mcp::setActiveMcpHttpServer(nullptr);
            mcp_http.stop();

            python::finalize();

            viewer.reset();

            core::teardown_gpu_before_exit();

            core::mark_clean_exit();
            core::flush_and_exit(0);
        }

#ifdef WIN32
        void hideConsoleWindow() {
            HWND hwnd = GetConsoleWindow();
            Sleep(1);
            HWND owner = GetWindow(hwnd, GW_OWNER);
            DWORD processId;
            GetWindowThreadProcessId(hwnd, &processId);

            if (GetCurrentProcessId() == processId) {
                ShowWindow(owner ? owner : hwnd, SW_HIDE);
            }
        }
#endif

    } // namespace

    int Application::run(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
        // Pre-initialize CacheLoader for the exe module.
        // On Windows, lfs_io (static lib) is linked into both the exe and
        // lfs_visualizer.dll, giving each its own CacheLoader singleton.
        // The callback below executes in the exe's context, so the exe's
        // copy must be initialized before it is invoked.
        lfs::io::CacheLoader::getInstance(params->dataset.loading_params.use_cpu_memory);

        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path,
                {.resize_factor = p.resize_factor,
                 .max_width = p.max_width,
                 .cuda_stream = p.stream,
                 .output_uint8 = p.output_uint8,
                 .skip_blob_cache = p.skip_blob_cache});
        });

        if (params->render_path) {
            const int result = runHeadlessRender(std::move(params));
            if (result == 0) {
                core::teardown_gpu_before_exit();
            }
            return result;
        }

        if (params->optimization.headless && params->server.tcp_connection) {
            return runHeadlessWithTCP(std::move(params));
        }

        if (params->optimization.headless) {
            return runHeadless(std::move(params));
        }

#ifdef WIN32
        hideConsoleWindow();
#endif

        return runGui(std::move(params));
    }

} // namespace lfs::app
