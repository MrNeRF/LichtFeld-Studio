/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "project_lifecycle.hpp"

#include "core/assert.hpp"
#include "core/checkpoint_format.hpp"
#include "core/data_loading_service.hpp"
#include "core/environment.hpp"
#include "core/error_bus.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/modal_request.hpp"
#include "core/parameter_manager.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/error_surface_types.hpp"
#include "gui/gui_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loader.hpp"
#include "io/project_container.hpp"
#include "io/project_path.hpp"
#include "io/project_recovery.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "io/selection_chapter.hpp"
#include "ipc/view_context.hpp"
#include "operation/undo_history.hpp"
#include "project/session_state.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "scene/scene_manager.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "training/training_setup.hpp"
#include "visualizer_impl.hpp"

#include <nlohmann/json.hpp>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <cwctype>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <istream>
#include <ranges>
#include <span>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::vis::project {

    namespace {

        using Json = nlohmann::json;
        using lfs::io::project::ChunkKey;
        using lfs::io::project::Fourcc;
        using lfs::io::project::PayloadBinding;
        using lfs::io::project::ProjectDocument;
        using lfs::io::project::ProjectDocumentSaveOptions;
        using lfs::io::project::ProjectSessionChapters;

        [[nodiscard]] lfs::Error lifecycleError(
            const lfs::ErrorCode code,
            std::string message,
            std::string detail,
            const std::string_view field = {}) {
            lfs::SmallFields fields;
            if (!field.empty()) {
                fields.add("field", field);
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::App,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        [[nodiscard]] lfs::Result<T> fail(
            const lfs::ErrorCode code,
            std::string message,
            std::string detail,
            const std::string_view field = {}) {
            if constexpr (std::same_as<T, void>) {
                return lfs::Result<void>::failure(
                    lifecycleError(
                        code, std::move(message),
                        std::move(detail), field));
            } else {
                return lifecycleError(
                    code, std::move(message),
                    std::move(detail), field);
            }
        }

        [[nodiscard]] std::optional<int>
        readBoundCheckpointHeaderIteration(
            const lfs::io::project::LazyChunkValue&
                checkpoint) {
            std::optional<int> stored_iteration;
            auto visited = checkpoint.visit_stream(
                [&](std::istream& source,
                    const std::uint64_t bytes)
                    -> lfs::Result<void> {
                    auto header =
                        lfs::core::load_checkpoint_header(
                            source, bytes);
                    if (!header) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Could not read the bound checkpoint header.",
                            header.error(),
                            "CKPT.header");
                    }
                    stored_iteration = header->iteration;
                    return {};
                });
            if (!visited || !stored_iteration) {
                return std::nullopt;
            }
            return stored_iteration;
        }

        [[nodiscard]] std::string developerError(
            const lfs::Error& error) {
            return lfs::format_for_developer(error);
        }

        [[nodiscard]] std::filesystem::path
        projectRootFor(
            const ProjectDocument& document) {
            if (const auto source =
                    document.source_path();
                source && !source->empty()) {
                return source->parent_path();
            }
            return {};
        }

        void publishProjectToast(lfs::Error error,
                                 const char* operation) {
            lfs::Error contextual = std::move(error);
            lfs::ErrorBus::instance().publish(
                lfs::ErrorNotification{
                    .error = std::move(contextual)
                                 .with_context(
                                     operation,
                                     LFS_SOURCE_SITE_CURRENT()),
                    .surface = lfs::ErrorSurface::Toast,
                    .actions = {},
                    .operation_id =
                        lfs::OperationId::generate(),
                });
        }

        void publishProjectToast(
            const lfs::ErrorCode code,
            const lfs::ErrorDomain domain,
            std::string user_message,
            const char* operation) {
            publishProjectToast(
                lfs::make_error(lfs::ErrorInit{
                    .code = code,
                    .domain = domain,
                    .severity = lfs::Severity::Error,
                    .retryability =
                        lfs::Retryability::NotRetryable,
                    .operation_id = {},
                    .user_message =
                        std::move(user_message),
                    .detail = {},
                    .detection =
                        LFS_SOURCE_SITE_CURRENT(),
                    .fields = {},
                    .native = std::nullopt,
                }),
                operation);
        }

        void notifyTrainerRestoreFailure(
            VisualizerImpl& viewer,
            const std::string& detail) {
            LOG_ERROR(
                "Project trainer restore failed (display model kept): {}",
                detail);
            if (auto* gui = viewer.getGuiManager()) {
                const auto generic = LOC(
                    "toast.trainer_not_restored.message");
                gui->enqueueToast({
                    .title = LOC(
                        "toast.trainer_not_restored.title"),
                    .message =
                        detail.empty()
                            ? generic
                            : std::format(
                                  "{}\n{}", generic,
                                  detail),
                    .level =
                        lfs::vis::gui::
                            ErrorNoticeLevel::
                                Error,
                    .fingerprint =
                        std::hash<std::string>{}(
                            "project-trainer-restore-failed"),
                });
            }
        }

        [[nodiscard]] std::optional<
            std::filesystem::path>
        resolveDatasetRootForTrainer(
            const ProjectDocument& document,
            const std::filesystem::path& dataset_hint) {
            const auto project_root =
                projectRootFor(document);
            if (const auto dataset_ref =
                    document.project()
                        .dataset_reference();
                dataset_ref && *dataset_ref) {
                if (auto resolved =
                        lfs::io::project::
                            resolve_path_reference(
                                document
                                    .references(),
                                project_root,
                                **dataset_ref,
                                dataset_hint)) {
                    return *resolved;
                }
            }
            if (!dataset_hint.empty()) {
                return dataset_hint;
            }
            return std::nullopt;
        }

        struct PersistedDatasetScene {
            bool has_training_model = false;
            bool has_dataset_node = false;
        };

        [[nodiscard]] PersistedDatasetScene
        inspectPersistedDatasetScene(
            const ProjectDocument& document) {
            PersistedDatasetScene result;
            if (const auto training =
                    document.scene_graph()
                        .training_model_uuid();
                training) {
                result.has_training_model =
                    training->has_value();
            } else {
                LOG_WARN(
                    "Persisted SCNG training_model_uuid is unreadable: {}",
                    developerError(
                        training.error()));
            }
            if (const auto nodes =
                    document.scene_graph().nodes();
                nodes) {
                result.has_dataset_node =
                    std::ranges::any_of(
                        *nodes, [](const auto& node) {
                            return node.type == "dataset";
                        });
            } else {
                LOG_WARN(
                    "Persisted SCNG nodes are unreadable: {}",
                    developerError(nodes.error()));
            }
            return result;
        }

        [[nodiscard]] std::optional<std::string>
        describeDatasetFolderProblem(
            const std::filesystem::path& root) {
            if (lfs::io::Loader::getDatasetType(root) !=
                lfs::io::DatasetType::Unknown) {
                return std::nullopt;
            }

            const auto info =
                lfs::io::detect_dataset_info(root);
            std::vector<std::string> missing;
            const bool has_images =
                lfs::io::safe_is_directory(
                    info.images_path) &&
                info.image_count > 0;
            if (!has_images) {
                missing.push_back(LOC(
                    lichtfeld::Strings::DatasetRelocate::
                        MISSING_IMAGES));
            }

            const bool has_transforms =
                lfs::io::safe_exists(
                    root / "transforms.json") ||
                lfs::io::safe_exists(
                    root / "transforms_train.json");
            bool has_colmap_cameras = false;
            std::filesystem::path sparse_dir;
            for (const auto& search :
                 lfs::io::get_colmap_search_paths(root)) {
                const bool has_camera_marker =
                    !lfs::io::find_file_ci(
                         search, "cameras.bin")
                         .empty() ||
                    !lfs::io::find_file_ci(
                         search, "cameras.txt")
                         .empty() ||
                    !lfs::io::find_file_ci(
                         search, "images.bin")
                         .empty() ||
                    !lfs::io::find_file_ci(
                         search, "images.txt")
                         .empty();
                if (has_camera_marker) {
                    has_colmap_cameras = true;
                    if (sparse_dir.empty()) {
                        sparse_dir = search;
                    }
                }
                if (search != root &&
                    lfs::io::safe_is_directory(search) &&
                    sparse_dir.empty()) {
                    sparse_dir = search;
                }
            }
            if (!has_colmap_cameras && !has_transforms) {
                missing.push_back(LOC(
                    lichtfeld::Strings::DatasetRelocate::
                        MISSING_CAMERAS));
            }
            if (!sparse_dir.empty()) {
                const bool has_points =
                    !lfs::io::find_file_ci(
                         sparse_dir, "points3D.bin")
                         .empty() ||
                    !lfs::io::find_file_ci(
                         sparse_dir, "points3D.txt")
                         .empty();
                if (!has_points) {
                    missing.push_back(LOC(
                        lichtfeld::Strings::
                            DatasetRelocate::
                                MISSING_POINT_CLOUD));
                }
            }

            if (missing.empty()) {
                return LOC(
                    lichtfeld::Strings::DatasetRelocate::
                        INVALID_MESSAGE);
            }
            std::string detail = missing.front();
            for (std::size_t i = 1; i < missing.size();
                 ++i) {
                detail += ", ";
                detail += missing[i];
            }
            return detail;
        }

        void installCheckpointTrainerWithDatasetRoot(
            VisualizerImpl& viewer,
            SceneManager& scene_manager,
            ProjectDocument& document,
            const lfs::core::Uuid& checkpoint_uuid,
            lfs::core::param::TrainingParameters
                ckpt_params,
            const int expected_iteration,
            const std::filesystem::path& dataset_root) {
            const auto old_root =
                ckpt_params.dataset.data_path;
            if (!old_root.empty() &&
                old_root.lexically_normal() !=
                    dataset_root.lexically_normal()) {
                const auto rebased =
                    scene_manager.getScene()
                        .rebaseCameraAssetPaths(
                            old_root, dataset_root);
                LOG_INFO(
                    "Rebased {} camera asset path(s) from {} to {}",
                    rebased,
                    lfs::core::path_to_utf8(old_root),
                    lfs::core::path_to_utf8(
                        dataset_root));
            }
            ckpt_params.dataset.data_path = dataset_root;

            // Reset path authority without re-applying
            // PRMS onto the live trainer (ownership
            // matrix rule 3).
            scene_manager.setDatasetPath(dataset_root);
            if (auto* data_loader =
                    viewer.getDataLoader()) {
                auto loader_params =
                    data_loader->getParameters();
                loader_params.dataset.data_path =
                    dataset_root;
                if (!ckpt_params.dataset.output_path
                         .empty()) {
                    loader_params.dataset.output_path =
                        ckpt_params.dataset.output_path;
                }
                data_loader->setParameters(loader_params);
            }
            if (auto* param_mgr =
                    viewer.getParameterManager()) {
                param_mgr->getDatasetConfig().data_path =
                    dataset_root;
            }

            auto* trainer_manager =
                viewer.getTrainerManager();
            if (!trainer_manager) {
                notifyTrainerRestoreFailure(
                    viewer,
                    "No trainer manager available");
                return;
            }
            if (trainer_manager->hasTrainer()) {
                if (!trainer_manager->clearTrainer()) {
                    notifyTrainerRestoreFailure(
                        viewer,
                        "Previous training worker is still stopping");
                    return;
                }
            }

            const auto source_name =
                document.source_path()
                    ? lfs::core::path_to_utf8(
                          *document.source_path())
                    : std::string{"project CKPT"};
            auto installed =
                lfs::training::
                    installTrainerFromProjectCheckpoint(
                        scene_manager.getScene(),
                        document,
                        checkpoint_uuid,
                        ckpt_params,
                        source_name,
                        expected_iteration,
                        std::nullopt,
                        makeViewerSplatTensorAllocator());
            if (!installed) {
                notifyTrainerRestoreFailure(
                    viewer, installed.error());
                return;
            }
            trainer_manager->setScene(
                &scene_manager.getScene());
            trainer_manager->setTrainerFromCheckpoint(
                std::move(installed->trainer),
                installed->iteration);
            if (!trainer_manager->hasTrainer()) {
                notifyTrainerRestoreFailure(
                    viewer,
                    "Checkpoint trainer install was rejected");
                return;
            }
            if (auto* trainer =
                    trainer_manager->getTrainer()) {
                // The user opened this project; trainer
                // runs continue publishing generations
                // into it.
                trainer->set_trainer_project_save_policy({
                    .on_completion = true,
                    .on_stop_or_error = false,
                    .at_step_boundaries = true,
                });
            }
            LOG_INFO(
                "Project trainer restored at iteration {} (dataset={})",
                installed->iteration,
                lfs::core::path_to_utf8(dataset_root));
        }

        // Pre-training (and relocated-dataset) projects have cameras in
        // the already-hydrated SCNG. Do not LoadFile/re-import: that
        // rebuilds cameras from the folder and drops node-level
        // training_enabled / visible. Missing-image cameras stay
        // Camera::has_image, never a user disable.
        void installTrainerFromHydratedDatasetScene(
            VisualizerImpl& viewer,
            SceneManager& scene_manager,
            const std::filesystem::path& dataset_root,
            const std::filesystem::path& output_path,
            const std::filesystem::path& rebase_from =
                {}) {
            if (!rebase_from.empty() &&
                rebase_from.lexically_normal() !=
                    dataset_root.lexically_normal()) {
                const auto rebased =
                    scene_manager.getScene()
                        .rebaseCameraAssetPaths(
                            rebase_from, dataset_root);
                LOG_INFO(
                    "Rebased {} camera asset path(s) from {} to {}",
                    rebased,
                    lfs::core::path_to_utf8(
                        rebase_from),
                    lfs::core::path_to_utf8(
                        dataset_root));
            }

            auto* const parameter_manager =
                viewer.getParameterManager();
            auto* const trainer_manager =
                viewer.getTrainerManager();
            auto* const data_loader =
                viewer.getDataLoader();
            if (!parameter_manager ||
                !trainer_manager) {
                notifyTrainerRestoreFailure(
                    viewer,
                    "Project has no trainer manager or parameter manager");
                return;
            }
            if (!data_loader) {
                notifyTrainerRestoreFailure(
                    viewer,
                    "Project dataset loader is unavailable");
                return;
            }
            if (!scene_manager.getScene()
                     .hasTrainingData()) {
                notifyTrainerRestoreFailure(
                    viewer,
                    "Hydrated scene has no cameras");
                return;
            }
            if (trainer_manager->hasTrainer()) {
                if (!trainer_manager->clearTrainer()) {
                    notifyTrainerRestoreFailure(
                        viewer,
                        "Previous training worker is still stopping");
                    return;
                }
            }

            auto params =
                parameter_manager->createForDataset(
                    dataset_root, output_path);
            data_loader->setParameters(params);
            scene_manager.setDatasetPath(dataset_root);
            parameter_manager->getDatasetConfig()
                .data_path = dataset_root;

            try {
                auto trainer = std::make_unique<
                    lfs::training::Trainer>(
                    scene_manager.getScene());
                trainer->setParams(params);
                trainer_manager->setScene(
                    &scene_manager.getScene());
                trainer_manager->setTrainer(
                    std::move(trainer));
            } catch (const std::exception& error) {
                notifyTrainerRestoreFailure(
                    viewer, error.what());
                return;
            }
            if (!trainer_manager->hasTrainer()) {
                notifyTrainerRestoreFailure(
                    viewer,
                    "Hydrated dataset trainer install was rejected");
                return;
            }
            LOG_INFO(
                "Project trainer restored from hydrated scene cameras (dataset={})",
                lfs::core::path_to_utf8(
                    dataset_root));
        }

    } // namespace

    // After display hydration, stream CKPT into a
    // Trainer and install it as Paused. Soft-fails:
    // keeps the display model, never dirties the
    // document.
    void ProjectLifecycle::tryInstallTrainerFromHydratedProject(
        SceneManager& scene_manager,
        lfs::io::project::ProjectDocument& document,
        const lfs::io::project::
            ProjectDocumentHydrationReport&
                report) {
        if (!report.trainer_state_pending ||
            !report.checkpoint_uuid ||
            !report.checkpoint_header) {
            const auto persisted_dataset =
                inspectPersistedDatasetScene(document);
            if (!persisted_dataset.has_dataset_node) {
                if (persisted_dataset.has_training_model) {
                    notifyTrainerRestoreFailure(
                        viewer_,
                        "Project has a training model but no checkpoint to resume");
                }
                return;
            }

            const auto dataset =
                document.project().dataset_reference();
            if (!dataset || !*dataset) {
                notifyTrainerRestoreFailure(
                    viewer_,
                    "Project dataset reference is missing");
                return;
            }

            auto* const parameter_manager =
                viewer_.getParameterManager();
            auto* const trainer_manager =
                viewer_.getTrainerManager();
            if (!parameter_manager || !trainer_manager) {
                notifyTrainerRestoreFailure(
                    viewer_,
                    "Project has no trainer manager or parameter manager");
                return;
            }

            auto dataset_root =
                resolveDatasetRootForTrainer(
                    document, std::filesystem::path{});
            if (!dataset_root || dataset_root->empty() ||
                !std::filesystem::exists(*dataset_root)) {
                if (dataset_root && !dataset_root->empty()) {
                    armMissingDatasetReload(
                        *dataset_root,
                        report.pending_parameters.dataset
                            .output_path);
                } else {
                    notifyTrainerRestoreFailure(
                        viewer_,
                        "Project dataset reference could not be resolved");
                }
                return;
            }

            installTrainerFromHydratedDatasetScene(
                viewer_, scene_manager, *dataset_root,
                report.pending_parameters.dataset
                    .output_path);
            return;
        }
        if (report.checkpoint_header->iteration < 0) {
            notifyTrainerRestoreFailure(
                viewer_, "Project CKPT iteration is negative");
            return;
        }
        const auto* checkpoint =
            document.find_checkpoint(*report.checkpoint_uuid);
        if (!checkpoint) {
            notifyTrainerRestoreFailure(
                viewer_, "Project CKPT handle disappeared");
            return;
        }

        std::optional<lfs::core::CheckpointParametersLoadResult>
            parsed_params;
        auto params_visit = checkpoint->visit_stream(
            [&](std::istream& source, const std::uint64_t bytes)
                -> lfs::Result<void> {
                parsed_params = lfs::core::load_checkpoint_params(
                    source, bytes);
                return {};
            });
        if (!params_visit) {
            notifyTrainerRestoreFailure(
                viewer_, developerError(params_visit.error()));
            return;
        }
        if (!parsed_params || !*parsed_params) {
            notifyTrainerRestoreFailure(
                viewer_,
                parsed_params ? parsed_params->error()
                              : "CKPT parameter visitor did not run");
            return;
        }
        auto ckpt_params = std::move(**parsed_params);
        ckpt_params.resume_checkpoint.reset();
        if (const auto source = document.source_path()) {
            ckpt_params.resume_project = *source;
        }

        const auto dataset_root = resolveDatasetRootForTrainer(
            document, ckpt_params.dataset.data_path);
        if (!dataset_root || dataset_root->empty() ||
            !std::filesystem::exists(*dataset_root)) {
            if (dataset_root && !dataset_root->empty()) {
                armMissingCheckpointDataset(
                    *dataset_root,
                    *report.checkpoint_uuid,
                    ckpt_params,
                    report.checkpoint_header->iteration);
            } else {
                notifyTrainerRestoreFailure(
                    viewer_,
                    "Project has no resolvable dataset root");
            }
            return;
        }
        installCheckpointTrainerWithDatasetRoot(
            viewer_,
            scene_manager,
            document,
            *report.checkpoint_uuid,
            std::move(ckpt_params),
            report.checkpoint_header->iteration,
            *dataset_root);
    }

    namespace {

        void bindRolePathReference(
            lfs::io::project::ReferencesChapter&
                references,
            const std::filesystem::path& project_root,
            const std::filesystem::path& live_path,
            const std::string_view key,
            const std::string_view kind,
            std::optional<lfs::core::Uuid>& binding) {
            if (live_path.empty()) {
                binding = std::nullopt;
                return;
            }
            auto minted =
                lfs::io::project::upsert_path_reference(
                    references, project_root, live_path,
                    key, kind, binding);
            if (minted) {
                binding = *minted;
            }
        }

        void mintParameterPathReferences(
            lfs::io::project::ReferencesChapter&
                references,
            const std::filesystem::path& project_root,
            lfs::io::project::ParameterManagerSnapshot&
                snapshot) {
            using Role = std::tuple<
                lfs::core::param::OptimizationParameters*,
                lfs::io::project::
                    ParameterManagerSnapshot::
                        ReferenceBindings*,
                std::string_view>;
            const std::array roles{
                Role{&snapshot.mcmc_session,
                     &snapshot.mcmc_session_references,
                     "mcmc.session"},
                Role{&snapshot.mrnf_session,
                     &snapshot.mrnf_session_references,
                     "mrnf.session"},
                Role{&snapshot.igs_session,
                     &snapshot.igs_session_references,
                     "igs+.session"},
                Role{&snapshot.mcmc_current,
                     &snapshot.mcmc_current_references,
                     "mcmc.current"},
                Role{&snapshot.mrnf_current,
                     &snapshot.mrnf_current_references,
                     "mrnf.current"},
                Role{&snapshot.igs_current,
                     &snapshot.igs_current_references,
                     "igs+.current"},
            };
            for (const auto& [params, bindings, role] :
                 roles) {
                bindRolePathReference(
                    references, project_root,
                    params->bg_image_path,
                    std::format(
                        "presets.{}.background_image",
                        role),
                    "background_image",
                    bindings->background_image_reference);
                bindRolePathReference(
                    references, project_root,
                    params->ppisp_sidecar_path,
                    std::format(
                        "presets.{}.ppisp_sidecar",
                        role),
                    "ppisp_sidecar",
                    bindings->ppisp_reference);
            }
        }

        void resolveParameterPathReferences(
            const lfs::io::project::ReferencesChapter&
                references,
            const std::filesystem::path& project_root,
            lfs::io::project::ParameterManagerSnapshot&
                snapshot) {
            using Role = std::tuple<
                lfs::core::param::OptimizationParameters*,
                const lfs::io::project::
                    ParameterManagerSnapshot::
                        ReferenceBindings*>;
            const std::array roles{
                Role{&snapshot.mcmc_session,
                     &snapshot.mcmc_session_references},
                Role{&snapshot.mrnf_session,
                     &snapshot.mrnf_session_references},
                Role{&snapshot.igs_session,
                     &snapshot.igs_session_references},
                Role{&snapshot.mcmc_current,
                     &snapshot.mcmc_current_references},
                Role{&snapshot.mrnf_current,
                     &snapshot.mrnf_current_references},
                Role{&snapshot.igs_current,
                     &snapshot.igs_current_references},
            };
            for (const auto& [params, bindings] :
                 roles) {
                if (bindings
                        ->background_image_reference) {
                    if (auto resolved =
                            lfs::io::project::
                                resolve_path_reference(
                                    references,
                                    project_root,
                                    *bindings
                                         ->background_image_reference,
                                    params
                                        ->bg_image_path)) {
                        params->bg_image_path =
                            *resolved;
                    }
                }
                if (bindings->ppisp_reference) {
                    if (auto resolved =
                            lfs::io::project::
                                resolve_path_reference(
                                    references,
                                    project_root,
                                    *bindings
                                         ->ppisp_reference,
                                    params
                                        ->ppisp_sidecar_path)) {
                        params->ppisp_sidecar_path =
                            *resolved;
                    }
                }
            }
        }

        [[nodiscard]] std::vector<lfs::core::Uuid>
        selectedNodeUuids(VisualizerImpl& viewer);

        [[nodiscard]] lfs::Result<
            lfs::training::
                ProjectSnapshotDocumentContext>
        captureTrainingDocumentContext(
            VisualizerImpl& viewer,
            const ProjectDocument& document,
            const lfs::io::project::CommitKind
                durable_commit_kind,
            std::optional<
                lfs::io::project::WriterLockLease>
                writer_lock_lease =
                    std::nullopt) {
            const auto project_root =
                projectRootFor(document);
            // Mutable REFS for minting; training context
            // carries the updated chapter.
            auto references = document.references();
            auto session = viewer.captureProjectSession(
                &references, project_root);
            if (!session) {
                return std::move(session).error();
            }
            auto* const parameter_manager =
                viewer.getParameterManager();
            if (!parameter_manager) {
                return fail<lfs::training::
                                ProjectSnapshotDocumentContext>(
                    lfs::ErrorCode::Unavailable,
                    "The parameter manager is unavailable.",
                    "Training project snapshots require live role-qualified parameters",
                    "project.parameters");
            }
            auto parameters = parameter_manager
                                  ->capturePendingProjectState();
            if (!parameters) {
                return std::move(parameters).error();
            }
            lfs::training::absolutize_dataset_path_for_snapshot(
                parameters->dataset.data_path);
            mintParameterPathReferences(
                references, project_root, *parameters);
            auto project = document.project();
            std::filesystem::path dataset_path;
            if (const auto* trainer =
                    viewer.getTrainer()) {
                dataset_path =
                    trainer->getParams()
                        .dataset.data_path;
            }
            if (dataset_path.empty()) {
                dataset_path =
                    parameters->dataset.data_path;
            }
            lfs::training::absolutize_dataset_path_for_snapshot(
                dataset_path);
            if (!dataset_path.empty()) {
                std::optional<lfs::core::Uuid>
                    existing;
                if (auto current =
                        project.dataset_reference();
                    current && *current) {
                    existing = **current;
                }
                if (auto minted =
                        lfs::io::project::
                            upsert_path_reference(
                                references,
                                project_root,
                                dataset_path,
                                "dataset",
                                "dataset",
                                existing);
                    minted) {
                    if (auto set =
                            project
                                .set_dataset_reference(
                                    *minted);
                        !set) {
                        LOG_WARN(
                            "Could not bind dataset reference: {}",
                            developerError(
                                set.error()));
                    }
                } else {
                    LOG_WARN(
                        "Could not mint dataset path reference: {}",
                        developerError(
                            minted.error()));
                }
            }
            return lfs::training::
                ProjectSnapshotDocumentContext{
                    .project_uuid =
                        document.project_uuid(),
                    .source_path =
                        document.source_path(),
                    .project = std::move(project),
                    .references =
                        std::move(references),
                    .gui_layout =
                        std::move(
                            session->gui_layout),
                    .view =
                        std::move(session->view),
                    .editor =
                        std::move(session->editor),
                    .sequencer =
                        std::move(
                            session->sequencer),
                    .metrics =
                        std::move(session->metrics),
                    .selected_node_uuids =
                        selectedNodeUuids(viewer),
                    .parameters =
                        std::move(*parameters),
                    .durable_commit_kind =
                        durable_commit_kind,
                    .writer_lock_lease =
                        std::move(
                            writer_lock_lease),
                };
        }

        [[nodiscard]] std::uint64_t unixTimeNs() {
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::system_clock::now()
                        .time_since_epoch())
                    .count());
        }

        [[nodiscard]] bool isLichtExtension(
            const std::filesystem::path& path) {
            std::string extension =
                path.extension().string();
            std::ranges::transform(
                extension, extension.begin(),
                [](const unsigned char value) {
                    return static_cast<char>(
                        std::tolower(value));
                });
            return extension == ".licht";
        }

        [[nodiscard]] bool isLichtPath(
            const std::filesystem::path& path) {
            return lfs::io::project::isPublishedLichtPath(
                path);
        }

        [[nodiscard]] lfs::Result<
            std::filesystem::path>
        resolveLichtFilePath(
            const std::filesystem::path& path) {
            if (path.empty() || !isLichtExtension(path)) {
                return fail<std::filesystem::path>(
                    lfs::ErrorCode::InvalidArgument,
                    "A project path must end in .licht.",
                    std::format(
                        "received '{}'", path.string()),
                    "project.path");
            }
            std::error_code error;
            auto absolute =
                std::filesystem::absolute(path, error);
            if (error) {
                return fail<std::filesystem::path>(
                    lfs::ErrorCode::InvalidArgument,
                    "The project path could not be resolved.",
                    error.message(), "project.path");
            }
            return absolute.lexically_normal();
        }

        [[nodiscard]] lfs::Result<
            std::filesystem::path>
        normalizedProjectPath(
            const std::filesystem::path& path) {
            auto resolved = resolveLichtFilePath(path);
            if (!resolved) {
                return resolved;
            }
            if (!isLichtPath(path)) {
                return fail<std::filesystem::path>(
                    lfs::ErrorCode::InvalidArgument,
                    lfs::io::project::
                        unpublishedLichtUserMessage(
                            path),
                    std::format(
                        "received '{}'", path.string()),
                    "project.path");
            }
            return resolved;
        }

        [[nodiscard]] bool sameBytes(
            const std::span<const std::byte> lhs,
            const std::span<const std::byte> rhs) {
            return lhs.size() == rhs.size() &&
                   std::ranges::equal(lhs, rhs);
        }

        [[nodiscard]] bool sameBytes(
            const std::vector<std::byte>& lhs,
            const std::vector<std::byte>& rhs) {
            return sameBytes(
                std::span<const std::byte>(lhs),
                std::span<const std::byte>(rhs));
        }

        [[nodiscard]] bool isSessionSoftDirtyChapter(
            const std::string_view fourcc) {
            return fourcc == "GUIL" || fourcc == "VIEW" ||
                   fourcc == "EDTR" || fourcc == "SEQR" ||
                   fourcc == "METR";
        }

        [[nodiscard]] bool hasHardDirtyChapters(
            const lfs::io::project::ProjectDocument&
                document) {
            if (!document.source_path()) {
                return document.dirty();
            }
            for (const auto& chapter :
                 document.dirty_chapters()) {
                if (!isSessionSoftDirtyChapter(chapter)) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool hasUntitledCrashDirtyChapters(
            const lfs::io::project::ProjectDocument&
                document) {
            for (const auto& chapter :
                 document.dirty_chapters()) {
                if (!isSessionSoftDirtyChapter(chapter)) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] std::string formatRecoverySavedTime(
            const std::uint64_t wallclock_unix_ns,
            const std::filesystem::path& path) {
            std::time_t unix_seconds = 0;
            if (wallclock_unix_ns != 0) {
                unix_seconds = static_cast<std::time_t>(
                    wallclock_unix_ns / 1'000'000'000ull);
            } else {
                std::error_code error;
                const auto file_time =
                    std::filesystem::last_write_time(
                        path, error);
                if (!error) {
                    const auto system_time =
                        std::chrono::clock_cast<
                            std::chrono::system_clock>(
                            file_time);
                    unix_seconds =
                        std::chrono::system_clock::to_time_t(
                            system_time);
                }
            }
            if (unix_seconds == 0) {
                return "—";
            }
            std::tm local{};
#ifdef _WIN32
            if (localtime_s(&local, &unix_seconds) != 0) {
                return "—";
            }
#else
            if (localtime_r(&unix_seconds, &local) ==
                nullptr) {
                return "—";
            }
#endif
            char buffer[32];
            if (std::strftime(
                    buffer, sizeof(buffer),
                    "%Y-%m-%d %H:%M", &local) == 0) {
                return "—";
            }
            return buffer;
        }

        [[nodiscard]] bool hasSessionSoftDirtyChapters(
            const lfs::io::project::ProjectDocument&
                document) {
            if (!document.source_path()) {
                return false;
            }
            for (const auto& chapter :
                 document.dirty_chapters()) {
                if (isSessionSoftDirtyChapter(chapter)) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] std::vector<lfs::core::Uuid>
        selectedNodeUuids(
            VisualizerImpl& viewer) {
            std::vector<lfs::core::Uuid> result;
            const auto* manager =
                viewer.getSceneManager();
            if (!manager) {
                return result;
            }
            const auto& scene = manager->getScene();
            for (const auto& name :
                 manager->getSelectedNodeNames()) {
                const auto* node = scene.getNode(name);
                if (node) {
                    result.push_back(node->uuid);
                }
            }
            std::ranges::sort(
                result, {}, [](const auto& uuid) {
                    return uuid.bytes;
                });
            result.erase(
                std::unique(result.begin(), result.end()),
                result.end());
            return result;
        }

        [[nodiscard]] lfs::io::project::
            ReferenceFingerprint
            syntheticFingerprint() {
            return {
                .kind = lfs::io::project::
                    FingerprintKind::File,
                .size = 0,
                .mtime_unix_ns = 0,
                .head_xxh3 = {},
                .tail_xxh3 = {},
                .full_xxh3 = std::nullopt,
            };
        }

        [[nodiscard]] lfs::Result<
            lfs::io::project::
                EmbeddedPayloadProvenance>
        payloadProvenance(
            const SceneManager& manager,
            const lfs::core::SceneNode& node,
            const std::string& fourcc) {
            auto fingerprint =
                syntheticFingerprint();
            std::string locator =
                std::format(
                    "generated:{}", node.uuid.to_string());
            if (const auto source =
                    manager.getPlyPath(node.uuid);
                source && !source->empty()) {
                locator =
                    lfs::core::path_to_utf8(*source);
                auto observed =
                    lfs::io::project::fingerprint_path(
                        *source);
                if (observed) {
                    fingerprint = std::move(*observed);
                }
            }
            return lfs::io::project::
                EmbeddedPayloadProvenance{
                    .uuid = node.uuid,
                    .node_uuid = node.uuid,
                    .fourcc = fourcc,
                    .import_locator =
                        {
                            .preferred =
                                std::move(locator),
                            .base =
                                lfs::io::project::
                                    LocatorBase::Absolute,
                            .absolute_fallback =
                                std::nullopt,
                        },
                    .import_fingerprint =
                        std::move(fingerprint),
                    .content_xxh3_128 = {},
                };
        }

        [[nodiscard]] SceneManager::ContentType
        inferContentType(const lfs::core::Scene& scene) {
            if (!scene.hasNodes()) {
                return SceneManager::ContentType::Empty;
            }
            if (std::ranges::any_of(
                    scene.getNodes(), [](const auto* node) {
                        return node &&
                               node->type ==
                                   lfs::core::NodeType::DATASET;
                    })) {
                return SceneManager::ContentType::Dataset;
            }
            return SceneManager::ContentType::SplatFiles;
        }

        void pngWriteCallback(
            void* context, void* bytes, const int size) {
            auto& destination =
                *static_cast<std::vector<std::byte>*>(
                    context);
            const auto* begin =
                static_cast<const std::byte*>(bytes);
            destination.insert(
                destination.end(), begin, begin + size);
        }

    } // namespace

    lfs::Result<ProjectLifecycleSettings>
    loadProjectLifecycleSettings(
        const std::filesystem::path& path) {
        ProjectLifecycleSettings settings;
        std::error_code error;
        if (!std::filesystem::exists(path, error)) {
            if (error) {
                return fail<ProjectLifecycleSettings>(
                    lfs::ErrorCode::PermissionDenied,
                    "Project lifecycle settings could not be inspected.",
                    error.message(), "settings.path");
            }
            return settings;
        }
        try {
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                return fail<ProjectLifecycleSettings>(
                    lfs::ErrorCode::PermissionDenied,
                    "Project lifecycle settings could not be opened.",
                    path.string(), "settings.path");
            }
            const Json json = Json::parse(stream);
            if (!json.is_object()) {
                return fail<ProjectLifecycleSettings>(
                    lfs::ErrorCode::DataLoss,
                    "Project lifecycle settings are invalid.",
                    "expected an object with version 1 or 2",
                    "settings.version");
            }
            const int version =
                json.value("version", 0);
            if (version != 1 && version != 2) {
                return fail<ProjectLifecycleSettings>(
                    lfs::ErrorCode::DataLoss,
                    "Project lifecycle settings are invalid.",
                    "expected an object with version 1 or 2",
                    "settings.version");
            }
            settings.reopen_last_project =
                json.value("reopen_last_project", true);
            // Version 1 defaulted auto_save_on_close to true.
            // Migrate once to the prompt-first default (false);
            // only an explicit re-enable under version 2 survives.
            if (version == 1) {
                settings.auto_save_on_close = false;
            } else {
                settings.auto_save_on_close =
                    json.value(
                        "auto_save_on_close", false);
            }
            settings.autosave_interval_seconds =
                json.value(
                    "autosave_interval_seconds",
                    std::uint64_t{5 * 60});
            settings
                .autosave_dirty_epoch_threshold =
                std::max<std::uint64_t>(
                    1,
                    json.value(
                        "autosave_dirty_epoch_threshold",
                        std::uint64_t{20}));
            settings.compaction_idle_seconds =
                json.value(
                    "compaction_idle_seconds",
                    std::uint64_t{30});
            const auto entries = json.find("mru");
            if (entries != json.end()) {
                if (!entries->is_array()) {
                    return fail<ProjectLifecycleSettings>(
                        lfs::ErrorCode::DataLoss,
                        "The recent-project list is invalid.",
                        "mru must be an array", "settings.mru");
                }
                for (const auto& entry : *entries) {
                    if (!entry.is_object()) {
                        continue;
                    }
                    const auto uuid =
                        lfs::core::Uuid::from_string(
                            entry.value(
                                "project_uuid",
                                std::string{}));
                    const auto path_text =
                        entry.value(
                            "last_known_path",
                            std::string{});
                    if (!uuid || uuid->is_nil() ||
                        path_text.empty()) {
                        continue;
                    }
                    settings.mru.push_back({
                        .project_uuid = *uuid,
                        .last_known_path =
                            lfs::core::utf8_to_path(
                                path_text),
                    });
                }
            }
            const auto dismissed =
                json.find("dismissed_recovery");
            if (dismissed != json.end()) {
                if (!dismissed->is_array()) {
                    return fail<ProjectLifecycleSettings>(
                        lfs::ErrorCode::DataLoss,
                        "The dismissed-recovery list is invalid.",
                        "dismissed_recovery must be an array",
                        "settings.dismissed_recovery");
                }
                for (const auto& entry : *dismissed) {
                    if (!entry.is_object()) {
                        continue;
                    }
                    const auto path_text =
                        entry.value(
                            "sidecar_path",
                            std::string{});
                    if (path_text.empty()) {
                        continue;
                    }
                    DismissedRecoveryEntry item;
                    item.sidecar_path =
                        lfs::core::utf8_to_path(
                            path_text)
                            .lexically_normal();
                    item.autosave_sequence =
                        entry.value(
                            "autosave_sequence",
                            std::uint64_t{0});
                    if (const auto commit =
                            lfs::core::Uuid::from_string(
                                entry.value(
                                    "commit_uuid",
                                    std::string{}));
                        commit) {
                        item.commit_uuid = *commit;
                    }
                    settings.dismissed_recovery.push_back(
                        std::move(item));
                }
            }
            return settings;
        } catch (const std::exception& exception) {
            // LFS-CENSUS-OK(empty-catch): JSON exceptions are converted to a typed settings error.
            return fail<ProjectLifecycleSettings>(
                lfs::ErrorCode::DataLoss,
                "Project lifecycle settings are invalid.",
                exception.what(), "settings");
        }
    }

    lfs::Result<void>
    saveProjectLifecycleSettings(
        const std::filesystem::path& path,
        const ProjectLifecycleSettings& settings) {
        try {
            Json entries = Json::array();
            for (const auto& entry : settings.mru) {
                entries.push_back({
                    {"project_uuid",
                     entry.project_uuid.to_string()},
                    {"last_known_path",
                     lfs::core::path_to_utf8(
                         entry.last_known_path)},
                });
            }
            Json dismissed = Json::array();
            for (const auto& entry :
                 settings.dismissed_recovery) {
                dismissed.push_back({
                    {"sidecar_path",
                     lfs::core::path_to_utf8(
                         entry.sidecar_path)},
                    {"autosave_sequence",
                     entry.autosave_sequence},
                    {"commit_uuid",
                     entry.commit_uuid.to_string()},
                });
            }
            const Json json{
                {"version", 2},
                {"reopen_last_project",
                 settings.reopen_last_project},
                {"auto_save_on_close",
                 settings.auto_save_on_close},
                {"autosave_interval_seconds",
                 settings
                     .autosave_interval_seconds},
                {"autosave_dirty_epoch_threshold",
                 settings
                     .autosave_dirty_epoch_threshold},
                {"compaction_idle_seconds",
                 settings
                     .compaction_idle_seconds},
                {"mru", std::move(entries)},
                {"dismissed_recovery",
                 std::move(dismissed)},
            };
            if (const auto written = lfs::core::writeTextFileAtomically(
                    path, json.dump(2) + '\n');
                !written) {
                return fail<void>(
                    lfs::ErrorCode::Unavailable,
                    "Project lifecycle settings could not be published.",
                    developerError(written.error()), "settings.path");
            }
            return {};
        } catch (const std::exception& exception) {
            // LFS-CENSUS-OK(empty-catch): filesystem exceptions are converted to a typed settings error.
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle settings could not be saved.",
                exception.what(), "settings");
        }
    }

    std::filesystem::path resolveProjectMruPath(
        const std::filesystem::path& path) {
        std::error_code error;
        auto absolute_path =
            std::filesystem::absolute(path, error);
        if (error) {
            absolute_path = path;
            error.clear();
        }

        auto resolved = std::filesystem::weakly_canonical(
            absolute_path, error);
        return error ? absolute_path.lexically_normal()
                     : resolved;
    }

    namespace {
        [[nodiscard]] bool projectMruPathsEqual(
            const std::filesystem::path& lhs,
            const std::filesystem::path& rhs) {
#ifdef _WIN32
            const auto lowercase = [](std::wstring value) {
                std::transform(
                    value.begin(), value.end(),
                    value.begin(), [](const wchar_t character) {
                        return static_cast<wchar_t>(
                            std::towlower(character));
                    });
                return value;
            };
            return lowercase(lhs.generic_wstring()) ==
                   lowercase(rhs.generic_wstring());
#else
            return lhs == rhs;
#endif
        }
    } // namespace

    void pruneMissingMruEntries(
        ProjectLifecycleSettings& settings) {
        settings.mru.erase(
            std::remove_if(
                settings.mru.begin(), settings.mru.end(),
                [](const ProjectMruEntry& entry) {
                    std::error_code error;
                    const bool exists =
                        std::filesystem::exists(
                            entry.last_known_path, error);
                    return error || !exists ||
                           !lfs::io::project::
                               isPublishedLichtPath(
                                   entry.last_known_path);
                }),
            settings.mru.end());
    }

    void rememberProject(
        ProjectLifecycleSettings& settings,
        const lfs::core::Uuid& project_uuid,
        const std::filesystem::path& path) {
        if (!lfs::io::project::isPublishedLichtPath(path)) {
            return;
        }
        const auto resolved = resolveProjectMruPath(path);
        settings.mru.erase(
            std::remove_if(
                settings.mru.begin(), settings.mru.end(),
                [&](const ProjectMruEntry& entry) {
                    return entry.project_uuid ==
                               project_uuid ||
                           projectMruPathsEqual(
                               resolveProjectMruPath(
                                   entry.last_known_path),
                               resolved);
                }),
            settings.mru.end());
        pruneMissingMruEntries(settings);
        settings.mru.insert(
            settings.mru.begin(),
            ProjectMruEntry{
                .project_uuid = project_uuid,
                .last_known_path = resolved,
            });
        constexpr std::size_t MAX_MRU = 12;
        if (settings.mru.size() > MAX_MRU) {
            settings.mru.resize(MAX_MRU);
        }
    }

    ProjectLifecycle::ProjectLifecycle(
        VisualizerImpl& viewer,
        std::optional<std::filesystem::path>
            settings_path)
        : viewer_(viewer),
          settings_path_([&settings_path] {
              if (settings_path)
                  return *settings_path;
              const auto paths = lfs::core::UserPaths::resolve();
              if (!paths) {
                  LOG_WARN("Unable to resolve project lifecycle settings path: {}",
                           developerError(paths.error()));
                  return std::filesystem::path{};
              }
              return paths->projectLifecycleFile();
          }()),
          recovery_directory_([&settings_path] {
              if (settings_path)
                  return settings_path->parent_path() /
                         "recovery";
              const auto paths = lfs::core::UserPaths::resolve();
              if (!paths) {
                  return std::filesystem::path{};
              }
              return paths->recoveryDir();
          }()),
          settings_persistence_enabled_(
              !lfs::core::environment::flag("LFS_SAFE_MODE", false) &&
              !settings_path_.empty()) {
        resetMaintenanceClocks();
        if (settings_persistence_enabled_) {
            if (auto loaded =
                    loadProjectLifecycleSettings(
                        settings_path_);
                loaded) {
                settings_ = std::move(*loaded);
            } else {
                LOG_WARN(
                    "Ignoring invalid project lifecycle settings: {}",
                    developerError(loaded.error()));
            }
        }
        auto created =
            ProjectDocument::create(
                lfs::core::generate_uuid_v4());
        if (created) {
            document_ =
                std::make_shared<ProjectDocument>(
                    std::move(*created));
        } else {
            LOG_ERROR(
                "Cannot create initial project document: {}",
                developerError(created.error()));
        }
    }

    ProjectLifecycle::~ProjectLifecycle() {
        if (auto* manager = viewer_.getTrainerManager()) {
            if (auto* trainer = manager->getTrainer()) {
                trainer->set_live_project_snapshot(
                    std::nullopt, {});
            }
        }
        recovery_prompt_pending_ = false;
        epoch_.fetch_add(1, std::memory_order_acq_rel);
        pending_dataset_relocation_.reset();
        project_write_thread_.request_stop();
        if (project_write_thread_.joinable()) {
            project_write_thread_.join();
        }
        stopHydrationThreads();
        std::optional<std::filesystem::path> discard_master;
        if (close_discard_requested_) {
            if (recovered_master_path_) {
                discard_master = *recovered_master_path_;
            } else if (document_ && document_->source_path()) {
                discard_master = *document_->source_path();
            }
        }
        document_.reset();
        cleanupRecoverySession();
        // The user explicitly discarded unsaved changes at exit; delete
        // the autosave overlay only after the write thread has joined and
        // the document/recovery locks are released. Emergency ForceExit
        // paths (X11 error, interrupt) never set
        // close_discard_requested_, so crash recovery survives.
        if (discard_master) {
            removeDiscardedAutosaveArtifacts(
                *discard_master);
        }
        if (close_discard_requested_) {
            removeScratchAutosave();
        }
    }

    void ProjectLifecycle::stopHydrationThreads() {
        std::vector<std::jthread> hydration_threads;
        {
            std::lock_guard lock(thread_mutex_);
            for (auto& thread : hydration_threads_) {
                thread.request_stop();
            }
            hydration_threads.swap(
                hydration_threads_);
        }
        hydration_threads.clear();
        if (project_open_job_) {
            if (auto job = viewer_.jobs().peek(*project_open_job_);
                job && job->running()) {
                viewer_.jobs().canceled(*project_open_job_);
                viewer_.jobs().free(*project_open_job_);
            }
            project_open_job_.reset();
        }
    }

    std::string ProjectLifecycle::hydrationName(
        const Hydration state) {
        switch (state) {
        case Hydration::Empty:
            return "empty";
        case Hydration::ShellReady:
            return "shell_ready";
        case Hydration::Hydrating:
            return "hydrating";
        case Hydration::Complete:
            return "complete";
        case Hydration::Failed:
            return "failed";
        }
        return "failed";
    }

    lfs::Result<void>
    ProjectLifecycle::persistSettings() {
        if (!settings_persistence_enabled_)
            return {};
        ProjectLifecycleSettings settings;
        {
            const std::lock_guard lock(
                settings_mutex_);
            pruneMissingMruEntries(settings_);
            settings = settings_;
        }
        return saveProjectLifecycleSettings(
            settings_path_, settings);
    }

    lfs::Result<void>
    ProjectLifecycle::preflightSwitch(
        const ProjectSwitchDisposition disposition,
        const bool allow_active_training) {
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The current project is still being saved.",
                "Project switching is blocked until the close save finishes",
                "project.save");
        }
        if (viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "A project write is still running.",
                "Project switching waits for save, autosave, or compaction",
                "project.job");
        }
        if (const auto* trainer =
                viewer_.getTrainerManager();
            trainer) {
            const bool publish_in_flight =
                trainer->isCompletionPending() &&
                !trainer->isTrainingActive();
            const bool training_active =
                trainer->isTrainingActive();
            if (publish_in_flight ||
                (training_active &&
                 !allow_active_training)) {
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    "Stop training before switching projects.",
                    "Project switching is blocked while training is active",
                    "project.training");
            }
        }
        if (disposition ==
                ProjectSwitchDisposition::RequireClean &&
            hasDirtyProject()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The current project has unsaved changes.",
                "Save the current project or retry with explicit discard authorization",
                "project.dirty");
        }
        return {};
    }

    std::optional<std::filesystem::path>
    ProjectLifecycle::pendingDatasetRelocationPath()
        const {
        if (!isDatasetRelocationCurrent(
                epoch_.load(std::memory_order_acquire))) {
            return std::nullopt;
        }
        return pending_dataset_relocation_->missing_path;
    }

    bool ProjectLifecycle::relocateProjectDataset(
        const std::filesystem::path& new_root,
        std::string* error_message) {
        const auto set_error = [&](std::string message) {
            if (error_message) {
                *error_message = std::move(message);
            }
        };
        const auto epoch =
            epoch_.load(std::memory_order_acquire);
        if (!pending_dataset_relocation_) {
            set_error(LOC(
                lichtfeld::Strings::DatasetRelocate::
                    UNAVAILABLE));
            return false;
        }
        if (pending_dataset_relocation_->open_epoch !=
            epoch) {
            pending_dataset_relocation_.reset();
            set_error(LOC(
                lichtfeld::Strings::DatasetRelocate::
                    UNAVAILABLE));
            return false;
        }
        if (auto problem =
                describeDatasetFolderProblem(new_root)) {
            set_error(std::move(*problem));
            return false;
        }
        auto retry =
            std::move(pending_dataset_relocation_->retry);
        pending_dataset_relocation_.reset();
        retry(new_root);
        return true;
    }

    bool ProjectLifecycle::isDatasetRelocationCurrent(
        const std::uint64_t epoch) const {
        return pending_dataset_relocation_ &&
               pending_dataset_relocation_->open_epoch ==
                   epoch &&
               epoch_.load(std::memory_order_acquire) ==
                   epoch;
    }

    void ProjectLifecycle::beginPendingDatasetRelocation(
        std::filesystem::path missing_path,
        std::function<void(const std::filesystem::path&)>
            retry) {
        const auto epoch =
            epoch_.load(std::memory_order_acquire);
        pending_dataset_relocation_ =
            PendingDatasetRelocation{
                .missing_path = std::move(missing_path),
                .open_epoch = epoch,
                .retry = std::move(retry),
            };
        LOG_WARN(
            "Project dataset root is missing; offering a relocation prompt (missing={})",
            lfs::core::path_to_utf8(
                pending_dataset_relocation_
                    ->missing_path));
        if (viewer_.getGuiManager()) {
            enqueueMissingDatasetDialog(epoch);
            return;
        }
        notifyTrainerRestoreFailure(
            viewer_,
            std::format(
                "Dataset path does not exist: {}",
                lfs::core::path_to_utf8(
                    pending_dataset_relocation_
                        ->missing_path)));
    }

    void ProjectLifecycle::armMissingDatasetReload(
        std::filesystem::path missing_path,
        std::filesystem::path output_path) {
        const auto epoch =
            epoch_.load(std::memory_order_acquire);
        const auto old_root = missing_path;
        beginPendingDatasetRelocation(
            std::move(missing_path),
            [this, epoch,
             output_path = std::move(output_path),
             old_root](
                const std::filesystem::path& new_root) {
                if (!viewer_.postWork({
                        .run =
                            [this, epoch, output_path,
                             old_root, new_root] {
                                if (epoch_.load(
                                        std::memory_order_acquire) !=
                                    epoch) {
                                    return;
                                }
                                auto* scene_manager =
                                    viewer_
                                        .getSceneManager();
                                if (!scene_manager) {
                                    notifyTrainerRestoreFailure(
                                        viewer_,
                                        "Project dataset loader is unavailable");
                                    return;
                                }
                                installTrainerFromHydratedDatasetScene(
                                    viewer_,
                                    *scene_manager,
                                    new_root,
                                    output_path,
                                    old_root);
                            },
                        .cancel = {},
                    })) {
                    notifyTrainerRestoreFailure(
                        viewer_,
                        "Project dataset loader is unavailable");
                }
            });
    }

    void ProjectLifecycle::armMissingCheckpointDataset(
        std::filesystem::path missing_path,
        lfs::core::Uuid checkpoint_uuid,
        const lfs::core::param::TrainingParameters&
            ckpt_params,
        const int expected_iteration) {
        const auto epoch =
            epoch_.load(std::memory_order_acquire);
        beginPendingDatasetRelocation(
            std::move(missing_path),
            [this, epoch, checkpoint_uuid, ckpt_params,
             expected_iteration](
                const std::filesystem::path& new_root) {
                if (!viewer_.postWork({
                        .run =
                            [this, epoch, checkpoint_uuid,
                             ckpt_params,
                             expected_iteration,
                             new_root] {
                                if (epoch_.load(
                                        std::memory_order_acquire) !=
                                    epoch) {
                                    return;
                                }
                                if (!document_ ||
                                    !document_
                                         ->find_checkpoint(
                                             checkpoint_uuid)) {
                                    notifyTrainerRestoreFailure(
                                        viewer_,
                                        "Project CKPT handle disappeared");
                                    return;
                                }
                                auto* scene_manager =
                                    viewer_
                                        .getSceneManager();
                                if (!scene_manager) {
                                    notifyTrainerRestoreFailure(
                                        viewer_,
                                        "No trainer manager available");
                                    return;
                                }
                                installCheckpointTrainerWithDatasetRoot(
                                    viewer_,
                                    *scene_manager,
                                    *document_,
                                    checkpoint_uuid,
                                    ckpt_params,
                                    expected_iteration,
                                    new_root);
                            },
                        .cancel = {},
                    })) {
                    notifyTrainerRestoreFailure(
                        viewer_,
                        "No trainer manager available");
                }
            });
    }

    void ProjectLifecycle::cancelPendingDatasetRelocation(
        const std::uint64_t epoch) {
        if (!isDatasetRelocationCurrent(epoch)) {
            return;
        }
        const auto missing_path =
            pending_dataset_relocation_->missing_path;
        pending_dataset_relocation_.reset();
        notifyTrainerRestoreFailure(
            viewer_,
            std::format(
                "Dataset path does not exist: {}",
                lfs::core::path_to_utf8(missing_path)));
    }

    void ProjectLifecycle::enqueueMissingDatasetDialog(
        const std::uint64_t epoch) {
        auto* gui = viewer_.getGuiManager();
        if (!gui || !isDatasetRelocationCurrent(epoch)) {
            return;
        }
        namespace Keys = lichtfeld::Strings::DatasetRelocate;
        const auto missing_utf8 = lfs::core::path_to_utf8(
            pending_dataset_relocation_->missing_path);
        lfs::core::ModalRequest request;
        request.title = LOC(Keys::TITLE);
        request.body_rml = std::format(
            "<div>{}</div>"
            "<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>",
            lfs::vis::gui::escapeRmlText(LOC(Keys::MESSAGE)),
            lfs::vis::gui::escapeRmlText(
                LOC(Keys::EXPECTED_LABEL)),
            lfs::vis::gui::escapeRmlText(missing_utf8));
        request.style = lfs::core::ModalStyle::Warning;
        request.width_dp = 520;
        request.buttons = {
            {LOC(lichtfeld::Strings::Common::CANCEL),
             "secondary"},
            {LOC(Keys::LOCATE), "primary"},
        };
        request.on_result =
            [this, epoch](const lfs::core::ModalResult&
                              result) {
                if (!isDatasetRelocationCurrent(epoch)) {
                    return;
                }
                if (result.button_label ==
                    LOC(lichtfeld::Strings::DatasetRelocate::
                            LOCATE)) {
                    handleLocateDatasetPicker(epoch);
                    return;
                }
                cancelPendingDatasetRelocation(epoch);
            };
        request.on_cancel = [this, epoch] {
            cancelPendingDatasetRelocation(epoch);
        };
        gui->enqueueModal(std::move(request));
    }

    void ProjectLifecycle::enqueueInvalidDatasetDialog(
        const std::uint64_t epoch,
        const std::filesystem::path& chosen_path,
        const std::string& detail) {
        auto* gui = viewer_.getGuiManager();
        if (!gui || !isDatasetRelocationCurrent(epoch)) {
            return;
        }
        namespace Keys = lichtfeld::Strings::DatasetRelocate;
        lfs::core::ModalRequest request;
        request.title = LOC(Keys::INVALID_TITLE);
        request.body_rml = std::format(
            "<div>{}</div>"
            "<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>"
            "<div class=\"content-row\"><span class=\"dim-text\">{}</span></div>",
            lfs::vis::gui::escapeRmlText(
                LOC(Keys::INVALID_MESSAGE)),
            lfs::vis::gui::escapeRmlText(
                LOC(Keys::MISSING_LABEL)),
            lfs::vis::gui::escapeRmlText(detail),
            lfs::vis::gui::escapeRmlText(
                lfs::core::path_to_utf8(chosen_path)));
        request.style = lfs::core::ModalStyle::Error;
        request.width_dp = 520;
        request.buttons = {
            {LOC(lichtfeld::Strings::Common::CANCEL),
             "secondary"},
            {LOC(Keys::CHOOSE_AGAIN), "primary"},
        };
        request.on_result =
            [this, epoch](const lfs::core::ModalResult&
                              result) {
                if (!isDatasetRelocationCurrent(epoch)) {
                    return;
                }
                if (result.button_label ==
                    LOC(lichtfeld::Strings::DatasetRelocate::
                            CHOOSE_AGAIN)) {
                    handleLocateDatasetPicker(epoch);
                    return;
                }
                cancelPendingDatasetRelocation(epoch);
            };
        request.on_cancel = [this, epoch] {
            cancelPendingDatasetRelocation(epoch);
        };
        gui->enqueueModal(std::move(request));
    }

    void ProjectLifecycle::handleLocateDatasetPicker(
        const std::uint64_t epoch) {
        if (!isDatasetRelocationCurrent(epoch)) {
            return;
        }
        const auto default_path =
            pending_dataset_relocation_->missing_path
                .parent_path();
        const auto chosen =
            lfs::vis::gui::OpenDatasetFolderDialog(
                default_path);
        if (chosen.empty()) {
            enqueueMissingDatasetDialog(epoch);
            return;
        }
        std::string error_message;
        if (relocateProjectDataset(
                chosen, &error_message)) {
            return;
        }
        if (!isDatasetRelocationCurrent(epoch)) {
            return;
        }
        enqueueInvalidDatasetDialog(
            epoch, chosen, error_message);
    }

    lfs::Result<void>
    ProjectLifecycle::setReopenLastProject(
        const bool enabled) {
        bool previous = false;
        {
            const std::lock_guard lock(
                settings_mutex_);
            if (settings_.reopen_last_project ==
                enabled) {
                return {};
            }
            previous =
                settings_.reopen_last_project;
            settings_.reopen_last_project =
                enabled;
        }
        if (auto saved = persistSettings(); !saved) {
            const std::lock_guard lock(
                settings_mutex_);
            settings_.reopen_last_project =
                previous;
            return saved;
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::setAutoSaveOnClose(
        const bool enabled) {
        bool previous = false;
        {
            const std::lock_guard lock(
                settings_mutex_);
            if (settings_.auto_save_on_close ==
                enabled) {
                return {};
            }
            previous =
                settings_.auto_save_on_close;
            settings_.auto_save_on_close =
                enabled;
        }
        if (auto saved = persistSettings(); !saved) {
            const std::lock_guard lock(
                settings_mutex_);
            settings_.auto_save_on_close =
                previous;
            return saved;
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::clearRecentProjects() {
        std::vector<ProjectMruEntry> previous;
        {
            const std::lock_guard lock(
                settings_mutex_);
            if (settings_.mru.empty()) {
                return {};
            }
            previous = settings_.mru;
            settings_.mru.clear();
        }
        if (auto saved = persistSettings(); !saved) {
            const std::lock_guard lock(
                settings_mutex_);
            settings_.mru = std::move(previous);
            return saved;
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::removeRecentProject(
        const std::filesystem::path& path) {
        std::vector<ProjectMruEntry> previous;
        {
            const std::lock_guard lock(
                settings_mutex_);
            previous = settings_.mru;
            const auto new_end = std::remove_if(
                settings_.mru.begin(),
                settings_.mru.end(),
                [&](const ProjectMruEntry& entry) {
                    return projectMruPathsEqual(
                        resolveProjectMruPath(
                            entry.last_known_path),
                        resolveProjectMruPath(path));
                });
            if (new_end == settings_.mru.end()) {
                return {};
            }
            settings_.mru.erase(
                new_end, settings_.mru.end());
        }
        if (auto saved = persistSettings(); !saved) {
            const std::lock_guard lock(
                settings_mutex_);
            settings_.mru = std::move(previous);
            return saved;
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::setAutosaveIntervalSeconds(
        const std::uint64_t seconds) {
        std::uint64_t previous = 0;
        {
            const std::lock_guard lock(
                settings_mutex_);
            previous =
                settings_.autosave_interval_seconds;
            settings_.autosave_interval_seconds =
                seconds;
        }
        if (auto saved = persistSettings(); !saved) {
            const std::lock_guard lock(
                settings_mutex_);
            settings_
                .autosave_interval_seconds =
                previous;
            return saved;
        }
        last_autosave_at_ =
            std::chrono::steady_clock::now();
        return {};
    }

    void ProjectLifecycle::resetMaintenanceClocks() {
        const auto now =
            std::chrono::steady_clock::now();
        last_autosave_at_ = now;
        last_mutation_at_ = now;
        next_storage_check_at_ = now;
        if (document_) {
            last_autosaved_dirty_epoch_ =
                document_->dirty_epoch();
        }
        last_autosaved_scene_serial_ =
            scene_mutation_serial_.load(
                std::memory_order_acquire);
        clearAutosaveFailureBackoff();
    }

    void ProjectLifecycle::clearAutosaveFailureBackoff() {
        autosave_failure_backoff_seconds_ = 0;
        autosave_retry_not_before_ = {};
    }

    void ProjectLifecycle::scheduleAutosaveFailureBackoff() {
        constexpr std::uint64_t kMinBackoffSeconds =
            60;
        constexpr std::uint64_t kMaxBackoffSeconds =
            600;
        if (autosave_failure_backoff_seconds_ ==
            0) {
            autosave_failure_backoff_seconds_ =
                kMinBackoffSeconds;
        } else {
            autosave_failure_backoff_seconds_ =
                std::min(
                    autosave_failure_backoff_seconds_ *
                        2,
                    kMaxBackoffSeconds);
        }
        autosave_retry_not_before_ =
            std::chrono::steady_clock::now() +
            std::chrono::seconds(
                autosave_failure_backoff_seconds_);
    }

    bool ProjectLifecycle::
        isBackgroundAutosaveSuppressed() const {
        if (application_close_pending_) {
            return true;
        }
        if (close_save_state_.load(
                std::memory_order_acquire) !=
            CloseSaveState::Idle) {
            return true;
        }
        if (const auto* window =
                viewer_.getWindowManager();
            window && window->shouldClose()) {
            return true;
        }
        return false;
    }

    bool ProjectLifecycle::
        isTrainingWriteWindowOpen() const {
        const auto* manager =
            viewer_.getTrainerManager();
        if (!manager) {
            return false;
        }
        return manager->isTrainingActive() ||
               manager->isCompletionPending() ||
               manager->getState() ==
                   TrainingState::Stopping;
    }

    void ProjectLifecycle::
        cancelBackgroundAutosaveIfRunning() {
        if (project_write_purpose_ !=
                ProjectWritePurpose::Autosave &&
            project_write_purpose_ !=
                ProjectWritePurpose::
                    TrainingAutosave) {
            return;
        }
        if (project_write_job_) {
            viewer_.jobs().requestCancel(
                *project_write_job_,
                "Cancelling autosave for exit");
        }
        if (project_write_thread_.joinable()) {
            project_write_thread_.request_stop();
        }
    }

    lfs::Result<void>
    ProjectLifecycle::
        waitOutBackgroundAutosaveForExplicitSave() {
        if (!viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            return {};
        }
        if (project_write_purpose_ !=
                ProjectWritePurpose::Autosave &&
            project_write_purpose_ !=
                ProjectWritePurpose::
                    TrainingAutosave) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "A project write is already in progress.",
                "Manual save, autosave, and compaction share one exclusive job slot",
                "project.job");
        }
        // Autosave occupies the exclusive ProjectWrite
        // slot. Join and settle it so a user Save / Save
        // As never loses the slot to a background write.
        joinPendingWrite();
        return {};
    }

    void ProjectLifecycle::cleanupRecoverySession() {
        if (recovery_session_) {
            const bool document_still_sources_temporary =
                recovery_session_path_ && document_ &&
                document_->source_path() &&
                document_->source_path()
                        ->lexically_normal() ==
                    recovery_session_path_
                        ->lexically_normal();
            LFS_DEBUG_ASSERT_MSG(
                !document_still_sources_temporary,
                "Recovery cleanup requires document replacement or durable rebinding first");
            if (document_still_sources_temporary) {
                LOG_ERROR(
                    "Refusing to release recovered-session staging file {} while the live project document still sources it",
                    lfs::core::path_to_utf8(
                        *recovery_session_path_));
                return;
            }
            recovery_session_->detach_document();
            if (auto released =
                    recovery_session_->release();
                !released) {
                LOG_WARN(
                    "Could not release recovered-session staging file: {}",
                    developerError(
                        released.error()));
            }
            recovery_session_.reset();
        } else if (recovery_session_path_) {
            std::error_code error;
            std::filesystem::remove(
                *recovery_session_path_, error);
            if (error) {
                LOG_WARN(
                    "Could not remove recovered-session staging file {}: {}",
                    lfs::core::path_to_utf8(
                        *recovery_session_path_),
                    error.message());
            }
        }
        recovery_session_path_.reset();
        recovered_master_path_.reset();
    }

    lfs::Result<void>
    ProjectLifecycle::startDocumentWrite(
        const ProjectWritePurpose purpose,
        std::shared_ptr<ProjectDocument> document,
        std::filesystem::path destination,
        ProjectDocumentSaveOptions options,
        std::optional<
            lfs::io::project::
                ProjectDocumentAutosaveOptions>
            autosave) {
        if (!cached_project_info_) {
            cached_project_info_ =
                ProjectInfo{
                    .path =
                        recovered_master_path_
                            ? recovered_master_path_
                            : document
                                  ->source_path(),
                    .project_uuid =
                        document
                            ->project_uuid()
                            .to_string(),
                    .generation =
                        document->generation(),
                    .dirty =
                        hasHardDirtyChapters(
                            *document),
                    .session_dirty =
                        hasSessionSoftDirtyChapters(
                            *document),
                    .dirty_chapters =
                        document
                            ->dirty_chapters(),
                    .hydration_state =
                        hydrationName(
                            hydration_.load(
                                std::memory_order_acquire)),
                    .payloads = {},
                    .contains_embedded_secrets =
                        containsEmbeddedSecrets(),
                    .reopen_last_project = [&] {
                        const std::lock_guard lock(
                            settings_mutex_);
                        return settings_.reopen_last_project; }(),
                    .auto_save_on_close = [&] {
                        const std::lock_guard lock(
                            settings_mutex_);
                        return settings_.auto_save_on_close; }(),
                    .autosave_interval_seconds =
                        settings_
                            .autosave_interval_seconds,
                    .autosave_dirty_epoch_threshold =
                        settings_
                            .autosave_dirty_epoch_threshold,
                    .project_write_running =
                        true,
                    .project_write_stage =
                        autosave
                            ? "Preparing autosave"
                            : "Preparing project save",
                    .project_write_progress =
                        0.0F,
                    .project_write_error =
                        {},
                    .project_write_error_code =
                        std::nullopt,
                    .autosave_sequence =
                        autosave_sequence_,
                    .recovery_session =
                        recovered_master_path_
                            .has_value(),
                    .compaction_suggested =
                        compaction_suggested_,
                    .physical_bytes =
                        storage_stats_
                            .physical_bytes,
                    .estimated_live_bytes =
                        storage_stats_
                            .estimated_live_bytes,
                    .dead_bytes =
                        storage_stats_.dead_bytes,
                    .dead_ratio =
                        storage_stats_.dead_ratio,
                    .hydration_error =
                        hydration_error_,
                    .recent_projects =
                        {},
                };
        }
        auto& jobs = viewer_.jobs();
        auto handle = jobs.init(
            JobType::ProjectWrite,
            autosave ? "Preparing autosave"
                     : "Preparing project save");
        if (!handle) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Another project write is already running.",
                "Project save, autosave, and compaction share one exclusive job slot",
                "project.job");
        }
        if (project_write_thread_.joinable()) {
            project_write_thread_.join();
        }
        project_write_job_ = *handle;
        project_write_purpose_ = purpose;
        project_write_destination_ =
            destination;
        project_write_scene_serial_ =
            scene_mutation_serial_.load(
                std::memory_order_acquire);
        project_write_parameter_serial_ =
            viewer_.getParameterManager()
                ? viewer_.getParameterManager()
                      ->dirtySerial()
                : 0;
        last_project_write_error_.clear();
        last_project_write_error_code_.reset();
        std::vector<std::byte> owned_preview(
            options.preview_png.begin(),
            options.preview_png.end());
        try {
            project_write_thread_ =
                std::jthread(
                    [this, handle = *handle,
                     document =
                         std::move(document),
                     destination =
                         std::move(destination),
                     options =
                         std::move(options),
                     owned_preview =
                         std::move(
                             owned_preview),
                     autosave =
                         std::move(autosave)](
                        const std::stop_token stop) mutable {
                        auto& registry =
                            viewer_.jobs();
                        options.preview_png =
                            owned_preview;
                        registry.work(handle);
                        registry.report(
                            handle, 0.05F,
                            autosave
                                ? "Writing autosave sidecar"
                                : "Writing project");
                        const std::lock_guard
                            document_lock(
                                document_access_mutex_);
                        lfs::Result<
                            lfs::io::project::
                                ProjectDocumentSaveReport>
                            saved =
                                autosave
                                    ? document
                                          ->save_autosave(
                                              destination,
                                              *autosave)
                                    : (document
                                                   ->source_path() &&
                                               document
                                                       ->source_path()
                                                       ->lexically_normal() ==
                                                   destination
                                                       .lexically_normal()
                                           ? document
                                                 ->save(
                                                     destination,
                                                     options)
                                           : document
                                                 ->save_as(
                                                     destination,
                                                     options));
                        std::string error;
                        std::optional<lfs::ErrorCode> error_code;
                        if (!saved) {
                            error_code = saved.error().code();
                            error =
                                developerError(
                                    saved.error());
                        } else {
                            registry.report(
                                handle, 0.95F,
                                "Verifying published project");
                        }
                        registry.finishWork(
                            handle,
                            stop.stop_requested(),
                            std::move(error),
                            error_code);
                        queueProjectWriteSettlement(
                            handle);
                    });
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): thread construction failure is returned as a typed job error.
            jobs.failed(
                *handle, error.what());
            jobs.free(*handle);
            project_write_job_.reset();
            project_write_purpose_ =
                ProjectWritePurpose::None;
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "The project writer could not start.",
                error.what(), "project.job");
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::startTrainingWrite(
        const ProjectWritePurpose purpose,
        const std::uint64_t request_id,
        std::filesystem::path master_path,
        const std::uint64_t dirty_epoch,
        const std::uint64_t scene_serial) {
        auto* const trainer =
            viewer_.getTrainer();
        if (!trainer || request_id == 0) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The training snapshot request could not start.",
                "Trainer and nonzero request identity are required",
                "project.training_snapshot");
        }
        auto handle = viewer_.jobs().init(
            JobType::ProjectWrite,
            purpose ==
                    ProjectWritePurpose::
                        TrainingAutosave
                ? "Waiting for training autosave safe point"
                : "Waiting for training save safe point");
        if (!handle) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Another project write is already running.",
                "Training snapshots share project-write exclusivity",
                "project.job");
        }
        if (project_write_thread_.joinable()) {
            project_write_thread_.join();
        }
        project_write_job_ = *handle;
        project_write_purpose_ = purpose;
        project_write_destination_ =
            std::move(master_path);
        project_write_dirty_epoch_ =
            dirty_epoch;
        project_write_scene_serial_ =
            scene_serial;
        project_write_parameter_serial_ =
            viewer_.getParameterManager()
                ? viewer_.getParameterManager()
                      ->dirtySerial()
                : 0;
        last_project_write_error_.clear();
        last_project_write_error_code_.reset();
        try {
            project_write_thread_ =
                std::jthread(
                    [this, trainer,
                     handle = *handle,
                     request_id](
                        const std::stop_token stop) {
                        auto& jobs =
                            viewer_.jobs();
                        jobs.work(handle);
                        std::string error;
                        bool canceled = false;
                        while (true) {
                            if (stop.stop_requested() ||
                                jobs.cancelRequested(
                                    handle)) {
                                canceled = true;
                                auto cancellation =
                                    lifecycleError(
                                        lfs::ErrorCode::
                                            Cancelled,
                                        "The training snapshot wait was canceled.",
                                        std::format(
                                            "Lifecycle cancellation terminalized trainer request {} before settlement",
                                            request_id),
                                        "project.training_snapshot");
                                trainer
                                    ->cancel_project_snapshot_request(
                                        request_id,
                                        cancellation);
                                error = developerError(
                                    cancellation);
                                break;
                            }
                            const auto metrics =
                                trainer
                                    ->get_project_snapshot_metrics();
                            if (metrics
                                    .last_failed_request_id >=
                                request_id) {
                                error =
                                    metrics
                                            .last_writer_error
                                            .empty()
                                        ? "The training snapshot request was superseded or failed"
                                        : metrics
                                              .last_writer_error;
                                break;
                            }
                            if (metrics
                                    .last_completed_request_id >=
                                request_id) {
                                break;
                            }
                            jobs.report(
                                handle,
                                metrics
                                        .writer_in_flight
                                    ? 0.7F
                                    : 0.25F,
                                metrics
                                        .writer_in_flight
                                    ? "Writing captured training snapshot"
                                    : "Waiting for training safe point");
                            std::this_thread::
                                sleep_for(
                                    std::chrono::
                                        milliseconds(
                                            5));
                        }
                        jobs.finishWork(
                            handle,
                            canceled,
                            std::move(error));
                        queueProjectWriteSettlement(
                            handle);
                    });
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): thread construction failure is returned as a typed job error.
            viewer_.jobs().failed(
                *handle, error.what());
            viewer_.jobs().free(*handle);
            project_write_job_.reset();
            project_write_purpose_ =
                ProjectWritePurpose::None;
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "The training snapshot monitor could not start.",
                error.what(),
                "project.training_snapshot");
        }
        return {};
    }

    void ProjectLifecycle::bindTrainerSnapshotTarget(
        std::optional<std::filesystem::path> destination,
        const bool allow_existing_destination_replacement) {
        auto* trainer = viewer_.getTrainer();
        if (!trainer) {
            return;
        }
        std::optional<std::filesystem::path> path =
            std::move(destination);
        if (!path && document_ && document_->source_path()) {
            path = recovered_master_path_.value_or(
                *document_->source_path());
        }
        trainer->set_live_project_snapshot(
            std::move(path),
            [this, allow_existing_destination_replacement]()
                -> std::optional<
                    lfs::training::
                        ProjectSnapshotDocumentContext> {
                if (!document_) {
                    return std::nullopt;
                }
                auto context =
                    captureTrainingDocumentContext(
                        viewer_, *document_,
                        recovered_master_path_
                            ? lfs::io::project::
                                  CommitKind::Recovered
                            : lfs::io::project::
                                  CommitKind::Explicit,
                        recovery_session_
                            ? std::optional{
                                  recovery_session_
                                      ->writer_lock()}
                            : std::nullopt);
                if (!context) {
                    LOG_WARN(
                        "Live training snapshot context failed: {}",
                        developerError(
                            context.error()));
                    return std::nullopt;
                }
                context->allow_existing_destination_replacement =
                    allow_existing_destination_replacement;
                return std::move(*context);
            });
    }

    lfs::Result<void>
    ProjectLifecycle::prepareTrainingStartProject() {
        auto* trainer = viewer_.getTrainer();
        if (!trainer) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Training cannot start without a trainer.",
                "A trainer must be installed before a project can be prepared",
                "project.training");
        }

        std::filesystem::path destination;
        if (document_ && document_->source_path()) {
            destination = recovered_master_path_.value_or(
                *document_->source_path());
        } else {
            const auto output =
                trainer->getParams().dataset.output_path;
            if (output.empty()) {
                return fail<void>(
                    lfs::ErrorCode::InvalidArgument,
                    "Training cannot create a project without an output path.",
                    "dataset.output_path is empty",
                    "dataset.output_path");
            }
            destination = output / "project.licht";
        }

        if (!hasSourcePath()) {
            if (auto saved = saveAs(
                    destination, /*regenerate_preview=*/false, true);
                !saved) {
                return saved;
            }
            // saveAs publishes on a worker. Join the
            // untitled create so the document has a
            // source path before the trainer is bound
            // and training starts. When the trainer is
            // training-active (including a paused
            // checkpoint-installed trainer), saveAs
            // routes through startTrainingWrite with
            // TrainingExplicitSave. jobs() exclusivity
            // means the saveAs we just issued is the
            // only write that can be running.
            if ((project_write_purpose_ ==
                     ProjectWritePurpose::SaveAs ||
                 project_write_purpose_ ==
                     ProjectWritePurpose::
                         TrainingExplicitSave) &&
                project_write_thread_.joinable()) {
                project_write_thread_.join();
                settleProjectWrite();
            }
            if (!hasSourcePath()) {
                return fail<void>(
                    lfs::ErrorCode::Unavailable,
                    "The training project could not be created.",
                    last_project_write_error_.empty()
                        ? "Untitled saveAs did not bind a source path"
                        : last_project_write_error_,
                    "project.training");
            }
        }

        bindTrainerSnapshotTarget();
        trainer->set_trainer_project_save_policy({
            .on_completion = true,
            .on_stop_or_error = false,
            .at_step_boundaries = true,
        });
        return {};
    }

    std::optional<int>
    ProjectLifecycle::trainingStartOverwriteConflict() {
        static_cast<void>(adoptCompletedTrainingSnapshot());
        auto* trainer = viewer_.getTrainer();
        if (!trainer) {
            return std::nullopt;
        }
        if (trainer->get_current_iteration() > 0) {
            return std::nullopt;
        }
        if (hasSourcePath() && document_) {
            const auto uuids =
                document_->checkpoint_uuids();
            if (!uuids.empty()) {
                if (cached_bound_checkpoint_iteration_) {
                    return *cached_bound_checkpoint_iteration_;
                }
                const auto* checkpoint =
                    document_->find_checkpoint(uuids.front());
                if (!checkpoint) {
                    return std::nullopt;
                }
                const auto stored_iteration =
                    readBoundCheckpointHeaderIteration(
                        *checkpoint);
                if (!stored_iteration) {
                    return std::nullopt;
                }
                cached_bound_checkpoint_iteration_ =
                    stored_iteration;
                return stored_iteration;
            }

            const auto master_path =
                recovered_master_path_.value_or(
                    *document_->source_path());
            try {
                auto opened = ProjectDocument::open(
                    master_path,
                    {
                        .reader = {},
                        .geometry = {},
                        .defer_geometry_payloads = true,
                    });
                if (!opened) {
                    return -1;
                }
                const auto disk_uuids =
                    opened->checkpoint_uuids();
                if (disk_uuids.empty()) {
                    return std::nullopt;
                }
                const auto* checkpoint =
                    opened->find_checkpoint(
                        disk_uuids.front());
                if (!checkpoint) {
                    return -1;
                }
                const auto stored_iteration =
                    readBoundCheckpointHeaderIteration(
                        *checkpoint);
                if (!stored_iteration) {
                    return -1;
                }
                return stored_iteration;
            } catch (const lfs::Exception& exception) {
                LOG_WARN(
                    "Training start overwrite probe failed to inspect {}: {}",
                    lfs::core::path_to_utf8(master_path),
                    developerError(exception.error()));
                return -1;
            } catch (const std::exception& exception) {
                LOG_WARN(
                    "Training start overwrite probe failed to inspect {}: {}",
                    lfs::core::path_to_utf8(master_path),
                    exception.what());
                return -1;
            }
        }

        const auto output =
            trainer->getParams().dataset.output_path;
        if (output.empty()) {
            return std::nullopt;
        }
        std::error_code ec;
        if (std::filesystem::exists(
                output / "project.licht", ec)) {
            return -1;
        }
        return std::nullopt;
    }

    bool ProjectLifecycle::isTrainingCheckpointStale()
        const {
        auto* trainer = viewer_.getTrainer();
        if (!trainer || !document_) {
            return false;
        }
        const auto uuids = document_->checkpoint_uuids();
        if (uuids.empty()) {
            return true;
        }
        if (cached_bound_checkpoint_iteration_) {
            return *cached_bound_checkpoint_iteration_ !=
                   trainer->get_current_iteration();
        }
        const auto* checkpoint =
            document_->find_checkpoint(uuids.front());
        if (!checkpoint) {
            return true;
        }
        const auto stored_iteration =
            readBoundCheckpointHeaderIteration(*checkpoint);
        if (!stored_iteration) {
            return true;
        }
        cached_bound_checkpoint_iteration_ =
            stored_iteration;
        return *stored_iteration !=
               trainer->get_current_iteration();
    }

    bool ProjectLifecycle::canFlushFinishedTrainerSnapshot()
        const {
        auto* trainer = viewer_.getTrainer();
        auto* manager = viewer_.getTrainerManager();
        return trainer && manager &&
               manager->isFinished() &&
               trainer->can_flush_project_snapshot() &&
               isTrainingCheckpointStale();
    }

    lfs::Result<void>
    ProjectLifecycle::startLiveTrainingSnapshotWrite(
        const ProjectWritePurpose purpose,
        const bool regenerate_preview) {
        auto* trainer = viewer_.getTrainer();
        if (!trainer || !document_ ||
            !document_->source_path()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The training snapshot request could not start.",
                "A live trainer and project path are required",
                "project.training_snapshot");
        }
        bindTrainerSnapshotTarget();
        auto context = captureTrainingDocumentContext(
            viewer_, *document_,
            recovered_master_path_
                ? lfs::io::project::
                      CommitKind::Recovered
                : lfs::io::project::
                      CommitKind::Explicit,
            recovery_session_
                ? std::optional{
                      recovery_session_->writer_lock()}
                : std::nullopt);
        if (!context) {
            return lfs::Status::failure(
                std::move(context).error());
        }
        std::vector<std::byte> preview;
        if (regenerate_preview) {
            auto captured = capturePreviewPng();
            if (!captured) {
                return lfs::Status::failure(
                    std::move(captured).error());
            }
            preview = std::move(*captured);
        }
        const auto destination =
            recovered_master_path_.value_or(
                *document_->source_path());
        const auto request_id =
            trainer->request_project_save(
                destination, std::move(preview),
                std::move(*context));
        auto* const manager = viewer_.getTrainerManager();
        if (manager && !manager->hasLiveTrainingThread() &&
            trainer->can_flush_project_snapshot()) {
            trainer->consume_requested_project_snapshot(
                trainer->project_snapshot_iteration());
        }
        return startTrainingWrite(
            purpose, request_id, destination,
            document_->dirty_epoch(),
            scene_mutation_serial_.load(
                std::memory_order_acquire));
    }

    lfs::Result<void>
    ProjectLifecycle::startAutosave() {
        if (!document_) {
            return {};
        }
        const bool untitled =
            !document_->source_path();
        if (viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            return {};
        }
        if (auto* trainer = viewer_.getTrainer();
            trainer &&
            trainer->get_project_snapshot_metrics()
                .writer_in_flight) {
            // Sidecar create requires the on-disk master
            // commit to match the held source UUID.
            return {};
        }
        if (auto adopted =
                adoptSettledTrainerPublishOntoCurrentMaster();
            !adopted) {
            return adopted;
        }
        const auto hydration = hydration_.load(
            std::memory_order_acquire);
        if (hydration != Hydration::Empty &&
            hydration != Hydration::Complete) {
            return {};
        }
        if (untitled && isBlankUntitledSession()) {
            last_autosave_at_ =
                std::chrono::steady_clock::now();
            return {};
        }

        const bool training =
            viewer_.getTrainer() &&
            viewer_.getTrainerManager() &&
            viewer_.getTrainerManager()
                ->isTrainingActive();
        if (auto synchronized =
                synchronizeDocumentFromViewer(
                    training
                        ? DocumentSyncMode::
                              LightTrainingAutosave
                        : DocumentSyncMode::
                              Default);
            !synchronized) {
            return synchronized;
        }
        const bool untitled_crash_dirty =
            untitled &&
            (hasUntitledCrashDirtyChapters(
                 *document_) ||
             scene_dirty_.load(
                 std::memory_order_acquire) ||
             payload_dirty_.load(
                 std::memory_order_acquire));
        if (untitled ? !untitled_crash_dirty
                     : !hasHardDirtyChapters(
                           *document_)) {
            last_autosave_at_ =
                std::chrono::steady_clock::now();
            return {};
        }
        if (training) {
            LOG_INFO(
                "Training-time autosave is light-only (no checkpoint capture)");
        }

        const auto sequence =
            autosave_sequence_ + 1;
        const auto dirty_epoch =
            document_->dirty_epoch();
        const auto scene_serial =
            scene_mutation_serial_.load(
                std::memory_order_acquire);

        if (untitled) {
            if (auto locked = lockScratchAutosave();
                !locked) {
                return locked;
            }
            lfs::io::project::
                ProjectDocumentSaveOptions options;
            options.commit.kind =
                lfs::io::project::CommitKind::
                    Explicit;
            options.commit.commit_uuid =
                lfs::core::generate_uuid_v4();
            options.file_uuid =
                lfs::core::generate_uuid_v4();
            options.index_compression =
                lfs::io::project::
                    IndexCompression::Zstd;
            options.disk_reserve_bytes =
                64ull * 1024 * 1024;
            options.allow_existing_destination_replacement =
                true;
            options.leave_unbound = true;
            options.writer_lock_lease = scratch_lock_;
            auto started = startDocumentWrite(
                ProjectWritePurpose::Autosave,
                document_, *scratch_autosave_path_,
                std::move(options));
            if (!started) {
                return started;
            }
            project_write_autosave_sequence_ =
                sequence;
            project_write_dirty_epoch_ =
                dirty_epoch;
            project_write_scene_serial_ =
                scene_serial;
            return {};
        }

        const auto base_commit_uuid =
            document_->source_commit_uuid();
        if (!base_commit_uuid) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The open project has no in-memory source commit.",
                "Autosave requires the already-held source reader",
                "project.source_reader");
        }
        std::error_code exists_error;
        if (!std::filesystem::exists(
                *document_->source_path(),
                exists_error) ||
            exists_error) {
            return fail<void>(
                lfs::ErrorCode::NotFound,
                "The project file is no longer available for autosave.",
                exists_error
                    ? exists_error.message()
                    : "The master path disappeared after the project was opened",
                "project.source_path");
        }
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(
                    *document_->source_path());
        auto started = startDocumentWrite(
            ProjectWritePurpose::Autosave,
            document_, sidecar, {},
            lfs::io::project::
                ProjectDocumentAutosaveOptions{
                    .file_uuid =
                        lfs::core::
                            generate_uuid_v4(),
                    .base_explicit_commit_uuid =
                        *base_commit_uuid,
                    .autosave_sequence =
                        sequence,
                    .snapshot_uuid = {},
                    .index_compression =
                        lfs::io::project::
                            IndexCompression::Zstd,
                    .disk_reserve_bytes =
                        64ull * 1024 * 1024,
                    .writer_lock_lease =
                        recovery_session_
                            ? std::optional{
                                  recovery_session_->writer_lock()}
                            : std::nullopt,
                    .boundary_observer = {},
                });
        if (!started) {
            return started;
        }
        project_write_autosave_sequence_ =
            sequence;
        project_write_dirty_epoch_ =
            dirty_epoch;
        project_write_scene_serial_ =
            scene_serial;
        return {};
    }

    void ProjectLifecycle::refreshStorageStats() {
        const auto* reader =
            document_ ? document_->source_reader()
                      : nullptr;
        if (!reader) {
            if (!document_ ||
                !document_->source_path()) {
                storage_stats_ = {};
                compaction_suggested_ = false;
            }
            return;
        }
        auto stats =
            lfs::io::project::
                project_storage_stats(*reader);
        if (!stats) {
            LOG_WARN(
                "Could not inspect .licht dead bytes: {}",
                developerError(stats.error()));
            return;
        }
        storage_stats_ = *stats;
        compaction_suggested_ =
            storage_stats_.dead_ratio >= 0.50;
        if (compaction_suggested_ &&
            !compaction_suggestion_reported_) {
            compaction_suggestion_reported_ =
                true;
            LOG_INFO(
                "Project compaction suggested: {} dead bytes ({:.1f}%)",
                storage_stats_.dead_bytes,
                storage_stats_.dead_ratio *
                    100.0);
        }
        if (!compaction_suggested_) {
            compaction_suggestion_reported_ =
                false;
        }
    }

    lfs::Result<void>
    ProjectLifecycle::startCompaction(
        const bool automatic) {
        if (!document_ ||
            !document_->source_path()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "This project has no durable master to compact.",
                "Save the project before compacting it",
                "project.path");
        }
        if (recovered_master_path_) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Save or discard the recovered state before compacting.",
                "Compaction cannot make the recovery base unreachable",
                "project.recovery");
        }
        if (hydration_.load(std::memory_order_acquire) !=
            Hydration::Complete) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Wait for project opening to finish before compacting.",
                "Compaction requires a fully hydrated project",
                "project.hydration");
        }
        if (viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Another project write is already running.",
                "Project save, autosave, and compaction are exclusive",
                "project.job");
        }
        if (hasDirtyProject()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Save the project before compacting it.",
                "Compaction only rewrites a durable clean head",
                "project.dirty");
        }
        auto recovery =
            lfs::io::project::
                inspect_autosave_recovery(
                    *document_->source_path());
        if (!recovery) {
            return lfs::Status::failure(
                std::move(recovery).error());
        }
        if (recovery->disposition ==
            lfs::io::project::
                RecoveryDisposition::Offer) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Recover or discard the autosave before compacting.",
                "The sidecar base must remain reachable until a recovery decision is made",
                "project.recovery");
        }
        auto handle = viewer_.jobs().init(
            JobType::ProjectWrite,
            automatic ? "Idle project compaction"
                      : "Compacting project");
        if (!handle) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Another project write is already running.",
                "Project-write exclusivity was lost before compaction",
                "project.job");
        }
        if (project_write_thread_.joinable()) {
            project_write_thread_.join();
        }
        const auto path =
            *document_->source_path();
        project_write_job_ = *handle;
        project_write_purpose_ =
            ProjectWritePurpose::Compaction;
        project_write_destination_ = path;
        project_write_automatic_ =
            automatic;
        last_project_write_error_.clear();
        last_project_write_error_code_.reset();
        try {
            project_write_thread_ =
                std::jthread(
                    [this, handle = *handle,
                     path](
                        const std::stop_token stop) {
                        auto& jobs =
                            viewer_.jobs();
                        jobs.work(handle);
                        auto compacted =
                            lfs::io::project::
                                ProjectWriter::compact(
                                    path,
                                    lfs::io::project::
                                        CompactionOptions{
                                            .compatibility =
                                                {},
                                            .new_file_uuid =
                                                lfs::core::
                                                    generate_uuid_v4(),
                                            .commit_uuid =
                                                lfs::core::
                                                    generate_uuid_v4(),
                                            .snapshot_uuid =
                                                {},
                                            .creation_time_unix_ns =
                                                unixTimeNs(),
                                            .wallclock_unix_ns =
                                                unixTimeNs(),
                                            .disk_reserve_bytes =
                                                64ull *
                                                1024 *
                                                1024,
                                            .boundary_observer =
                                                [this, handle](
                                                    const lfs::io::project::
                                                        CommitBoundary
                                                            boundary) {
                                                    const auto value =
                                                        static_cast<float>(
                                                            static_cast<int>(
                                                                boundary));
                                                    viewer_.jobs().report(
                                                        handle,
                                                        value /
                                                            static_cast<
                                                                float>(
                                                                static_cast<
                                                                    int>(
                                                                    lfs::io::project::
                                                                        CommitBoundary::
                                                                            Committed)),
                                                        "Building and verifying compact project");
                                                },
                                        });
                        const auto compact_error_code =
                            compacted
                                ? std::optional<lfs::ErrorCode>{}
                                : std::optional{
                                      compacted.error().code()};
                        jobs.finishWork(
                            handle,
                            stop.stop_requested(),
                            compacted
                                ? std::string{}
                                : developerError(
                                      compacted
                                          .error()),
                            compact_error_code);
                        queueProjectWriteSettlement(
                            handle);
                    });
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): thread construction failure is returned as a typed job error.
            viewer_.jobs().failed(
                *handle, error.what());
            viewer_.jobs().free(*handle);
            project_write_job_.reset();
            project_write_purpose_ =
                ProjectWritePurpose::None;
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "The compaction worker could not start.",
                error.what(), "project.job");
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::compact() {
        return startCompaction(false);
    }

    void ProjectLifecycle::joinPendingWrite() {
        // Blocks until the in-flight project write settles.
        if (project_write_thread_.joinable()) {
            project_write_thread_.join();
        }
        settleProjectWrite();
    }

    void ProjectLifecycle::queueProjectWriteSettlement(
        const JobHandle handle) {
        const auto settle =
            [this, handle] {
                if (project_write_job_ &&
                    *project_write_job_ == handle) {
                    settleProjectWrite();
                }
            };
        if (!viewer_.postWork({
                .run = settle,
                .cancel = settle,
            })) {
            viewer_.wakeMainLoop();
        }
    }

    void ProjectLifecycle::settleProjectWrite() {
        if (!project_write_job_) {
            return;
        }
        auto& jobs = viewer_.jobs();
        const auto snapshot =
            jobs.update(*project_write_job_);
        if (!snapshot ||
            snapshot->status !=
                JobStatus::CompletionPending) {
            return;
        }
        if (project_write_thread_.joinable()) {
            project_write_thread_.join();
        }

        std::string error = snapshot->error;
        last_project_write_error_code_ = snapshot->error_code;
        if (snapshot->worker_canceled &&
            error.empty()) {
            last_project_write_error_code_ =
                lfs::ErrorCode::Cancelled;
            error =
                "The project write was canceled.";
        }
        if (error.empty() &&
            (project_write_purpose_ ==
                 ProjectWritePurpose::
                     ExplicitSave ||
             project_write_purpose_ ==
                 ProjectWritePurpose::SaveAs ||
             project_write_purpose_ ==
                 ProjectWritePurpose::CloseSave ||
             project_write_purpose_ ==
                 ProjectWritePurpose::
                     TrainingExplicitSave ||
             project_write_purpose_ ==
                 ProjectWritePurpose::
                     TrainingCloseSave)) {
            if (project_write_purpose_ ==
                    ProjectWritePurpose::
                        TrainingExplicitSave ||
                project_write_purpose_ ==
                    ProjectWritePurpose::
                        TrainingCloseSave) {
                if (auto adopted =
                        adoptCompletedTrainingSnapshot(
                            true);
                    !adopted) {
                    last_project_write_error_code_ =
                        adopted.error().code();
                    error =
                        developerError(
                            adopted.error());
                }
            }
        }
        if (error.empty() &&
            project_write_purpose_ ==
                ProjectWritePurpose::
                    TrainingAutosave) {
            if (auto adopted =
                    adoptSettledTrainerPublishOntoCurrentMaster();
                !adopted) {
                last_project_write_error_code_ =
                    adopted.error().code();
                error =
                    developerError(
                        adopted.error());
            }
        }
        if (error.empty() &&
            (project_write_purpose_ ==
                 ProjectWritePurpose::
                     ExplicitSave ||
             project_write_purpose_ ==
                 ProjectWritePurpose::SaveAs ||
             project_write_purpose_ ==
                 ProjectWritePurpose::CloseSave ||
             project_write_purpose_ ==
                 ProjectWritePurpose::
                     TrainingExplicitSave ||
             project_write_purpose_ ==
                 ProjectWritePurpose::
                     TrainingCloseSave)) {
            std::string cleanup_warning;
            if (recovery_session_) {
                // ProjectDocument::save_as (or the training-snapshot
                // adoption path) has rebound every lazy source row to the
                // durable destination before the worker reports success.
                recovery_session_->detach_document();
                if (auto released =
                        recovery_session_
                            ->release();
                    !released) {
                    cleanup_warning =
                        developerError(
                            released.error());
                }
            }
            if (auto removed =
                    lfs::io::project::
                        remove_autosave_artifacts(
                            project_write_destination_);
                !removed) {
                if (!cleanup_warning.empty()) {
                    cleanup_warning += "; ";
                }
                cleanup_warning +=
                    developerError(
                        removed.error());
            }
            removeScratchAutosave();
            declined_recovery_.reset();
            if (!cleanup_warning.empty()) {
                LOG_WARN(
                    "Project save is durable, but autosave cleanup failed: {}",
                    cleanup_warning);
                if (auto* gui =
                        viewer_.getGuiManager()) {
                    gui->enqueueToast({
                        .title =
                            "Project saved with a cleanup warning",
                        .message =
                            "The project is safely saved. A stale autosave file could not be removed and will be retried on the next open.",
                        .level =
                            lfs::vis::gui::
                                ErrorNoticeLevel::
                                    Warning,
                        .fingerprint =
                            std::hash<std::string>{}(
                                "project-autosave-cleanup-warning"),
                    });
                }
            }
        }
        if (error.empty() &&
            project_write_purpose_ ==
                ProjectWritePurpose::Compaction) {
            auto reopened =
                ProjectDocument::open(
                    project_write_destination_,
                    {
                        .reader = {},
                        .geometry = {},
                        .defer_geometry_payloads =
                            true,
                    });
            if (!reopened) {
                last_project_write_error_code_ =
                    reopened.error().code();
                error = developerError(
                    reopened.error());
            } else {
                document_ =
                    std::make_shared<
                        ProjectDocument>(
                        std::move(*reopened));
                cached_bound_checkpoint_iteration_
                    .reset();
            }
        }

        const bool was_autosave =
            project_write_purpose_ ==
                ProjectWritePurpose::Autosave ||
            project_write_purpose_ ==
                ProjectWritePurpose::
                    TrainingAutosave;
        const bool rebases_project_dirty_state =
            project_write_purpose_ ==
                ProjectWritePurpose::ExplicitSave ||
            project_write_purpose_ ==
                ProjectWritePurpose::SaveAs ||
            project_write_purpose_ ==
                ProjectWritePurpose::CloseSave ||
            project_write_purpose_ ==
                ProjectWritePurpose::TrainingExplicitSave ||
            project_write_purpose_ ==
                ProjectWritePurpose::TrainingCloseSave;
        if (error.empty()) {
            if (rebases_project_dirty_state) {
                const bool write_inputs_unchanged =
                    scene_mutation_serial_.load(
                        std::memory_order_acquire) ==
                    project_write_scene_serial_;
                if (write_inputs_unchanged) {
                    scene_dirty_.store(
                        false, std::memory_order_release);
                    payload_dirty_.store(
                        false, std::memory_order_release);
                }
                if (auto* parameter_manager =
                        viewer_.getParameterManager()) {
                    parameter_manager->clearDirtyIfUnchanged(
                        project_write_parameter_serial_);
                }
            }
            jobs.completed(*project_write_job_);
            clearAutosaveFailureBackoff();
            if (was_autosave) {
                autosave_sequence_ = std::max(
                    autosave_sequence_,
                    project_write_autosave_sequence_);
                last_autosaved_dirty_epoch_ =
                    project_write_dirty_epoch_;
                last_autosaved_scene_serial_ =
                    project_write_scene_serial_;
                last_autosave_at_ =
                    std::chrono::steady_clock::
                        now();
                LOG_INFO(
                    "Autosave sidecar sequence {} published",
                    autosave_sequence_);
            } else if (
                project_write_purpose_ ==
                    ProjectWritePurpose::
                        ExplicitSave ||
                project_write_purpose_ ==
                    ProjectWritePurpose::
                        SaveAs ||
                project_write_purpose_ ==
                    ProjectWritePurpose::
                        CloseSave ||
                project_write_purpose_ ==
                    ProjectWritePurpose::
                        TrainingExplicitSave ||
                project_write_purpose_ ==
                    ProjectWritePurpose::
                        TrainingCloseSave) {
                {
                    const std::lock_guard lock(
                        settings_mutex_);
                    rememberProject(
                        settings_,
                        document_->project_uuid(),
                        project_write_destination_);
                }
                if (auto persisted =
                        persistSettings();
                    !persisted) {
                    LOG_WARN(
                        "Project saved, but MRU settings failed: {}",
                        developerError(
                            persisted.error()));
                }
                cleanupRecoverySession();
                resetMaintenanceClocks();
                bindTrainerSnapshotTarget();
            } else if (
                project_write_purpose_ ==
                ProjectWritePurpose::
                    Compaction) {
                LOG_INFO(
                    "{} project compaction completed: {}",
                    project_write_automatic_
                        ? "Idle"
                        : "Explicit",
                    lfs::core::path_to_utf8(
                        project_write_destination_));
                resetMaintenanceClocks();
            }
        } else {
            if (snapshot->worker_canceled) {
                jobs.canceled(
                    *project_write_job_);
            } else {
                jobs.failed(
                    *project_write_job_, error);
            }
            last_project_write_error_ =
                error;
            if (snapshot->worker_canceled) {
                LOG_INFO(
                    "Project background write canceled: {}",
                    error);
            } else {
                LOG_ERROR(
                    "Project background write failed: {}",
                    error);
                publishProjectToast(
                    last_project_write_error_code_.value_or(
                        lfs::ErrorCode::Unavailable),
                    lfs::ErrorDomain::IO,
                    error,
                    gui::error_op::kSave);
            }
            if (was_autosave) {
                scheduleAutosaveFailureBackoff();
            }
        }
        const bool close_save =
            project_write_purpose_ ==
                ProjectWritePurpose::CloseSave ||
            project_write_purpose_ ==
                ProjectWritePurpose::
                    TrainingCloseSave;
        if (close_save) {
            {
                std::lock_guard lock(
                    close_save_mutex_);
                close_save_error_ = error;
            }
            // CancelExit during Saving clears the close
            // latch but does not abort the writer. A
            // sticky Succeeded would skip later dirty
            // checks, so settle to Idle when close is
            // no longer wanted.
            if (!application_close_pending_) {
                close_save_state_.store(
                    CloseSaveState::Idle,
                    std::memory_order_release);
            } else {
                close_save_state_.store(
                    error.empty()
                        ? CloseSaveState::Succeeded
                        : CloseSaveState::Failed,
                    std::memory_order_release);
            }
        }
        jobs.free(*project_write_job_);
        project_write_job_.reset();
        project_write_purpose_ =
            ProjectWritePurpose::None;
        project_write_destination_.clear();
        project_write_autosave_sequence_ = 0;
        project_write_automatic_ = false;
        cached_project_info_.reset();
        if (!(was_autosave && !error.empty())) {
            refreshStorageStats();
        }
        if (application_close_pending_) {
            viewer_.requestApplicationClose();
        }
    }

    void ProjectLifecycle::updateMaintenance() {
        settleProjectWrite();
        if (!viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            if (auto adopted =
                    adoptSettledTrainerPublishOntoCurrentMaster();
                !adopted) {
                const auto warning =
                    developerError(adopted.error());
                if (warning !=
                    last_unadoptable_training_snapshot_warning_) {
                    last_unadoptable_training_snapshot_warning_ =
                        warning;
                    LOG_WARN(
                        "Could not adopt the settled training project generation: {}",
                        warning);
                }
            }
        }
        if (!document_ ||
            viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            return;
        }
        if (isBlankUntitledSession()) {
            return;
        }
        const auto now =
            std::chrono::steady_clock::now();
        const bool training =
            viewer_.getTrainerManager() &&
            viewer_.getTrainerManager()
                ->isTrainingActive();
        const bool training_write_window =
            isTrainingWriteWindowOpen();
        if (!training_write_window &&
            compaction_suggested_ &&
            settings_.compaction_idle_seconds !=
                0 &&
            now - last_mutation_at_ >=
                std::chrono::seconds(
                    settings_
                        .compaction_idle_seconds) &&
            !scene_dirty_.load(
                std::memory_order_acquire) &&
            !payload_dirty_.load(
                std::memory_order_acquire) &&
            !hasHardDirtyChapters(*document_)) {
            if (auto started =
                    startCompaction(true);
                !started) {
                LOG_DEBUG(
                    "Idle compaction deferred: {}",
                    developerError(
                        started.error()));
            }
            return;
        }
        if (recovered_master_path_) {
            return;
        }
        const auto scene_serial =
            scene_mutation_serial_.load(
                std::memory_order_acquire);
        const bool plausibly_dirty =
            scene_dirty_.load(
                std::memory_order_acquire) ||
            payload_dirty_.load(
                std::memory_order_acquire) ||
            hasHardDirtyChapters(*document_);
        if (!plausibly_dirty) {
            return;
        }
        const bool timer_due =
            settings_.autosave_interval_seconds !=
                0 &&
            now - last_autosave_at_ >=
                std::chrono::seconds(
                    settings_
                        .autosave_interval_seconds);
        const auto threshold_reached =
            [](const std::uint64_t current,
               const std::uint64_t previous,
               const std::uint64_t threshold) {
                return current >= previous &&
                       current - previous >= threshold;
            };
        const bool epoch_due =
            threshold_reached(
                document_->dirty_epoch(),
                last_autosaved_dirty_epoch_,
                settings_
                    .autosave_dirty_epoch_threshold) ||
            threshold_reached(
                scene_serial,
                last_autosaved_scene_serial_,
                settings_
                    .autosave_dirty_epoch_threshold);
        if (!(timer_due || epoch_due)) {
            return;
        }
        if (isBackgroundAutosaveSuppressed()) {
            return;
        }
        if (training_write_window && !training) {
            return;
        }
        if (autosave_failure_backoff_seconds_ !=
                0 &&
            now < autosave_retry_not_before_) {
            return;
        }
        if (auto started = startAutosave();
            !started) {
            last_project_write_error_ =
                developerError(
                    started.error());
            LOG_WARN(
                "Autosave deferred: {}",
                last_project_write_error_);
            scheduleAutosaveFailureBackoff();
        }
    }

    void ProjectLifecycle::noteProjectFrameRendered(
        const double render_ms) {
        if (!project_first_render_pending_) {
            return;
        }
        project_first_render_pending_ = false;
        const auto rendered_at =
            std::chrono::steady_clock::now();
        const double commit_to_present_ms =
            std::chrono::duration<double, std::milli>(
                rendered_at -
                hydration_committed_at_)
                .count();
        const double open_to_present_ms =
            std::chrono::duration<double, std::milli>(
                rendered_at -
                project_open_started_at_)
                .count();
        LOG_DEBUG(
            "Project first render stages: wait_after_hydration={:.3f} ms render_present={:.3f} ms open_to_present={:.3f} ms",
            std::max(
                0.0,
                commit_to_present_ms - render_ms),
            render_ms, open_to_present_ms);
    }

    void ProjectLifecycle::markSceneMutation(
        const std::uint32_t mutation_flags) {
        cached_project_info_.reset();
        last_mutation_at_ =
            std::chrono::steady_clock::now();
        scene_mutation_serial_.fetch_add(
            1, std::memory_order_acq_rel);
        const auto selection_flag =
            static_cast<std::uint32_t>(
                lfs::core::Scene::MutationType::
                    SELECTION_CHANGED);
        if ((mutation_flags & selection_flag) != 0) {
            selection_mutation_serial_.fetch_add(
                1, std::memory_order_acq_rel);
        }
        if (mutation_flags != 0) {
            scene_dirty_.store(
                true, std::memory_order_release);
        }
        const auto model_flag =
            static_cast<std::uint32_t>(
                lfs::core::Scene::MutationType::
                    MODEL_CHANGED);
        if ((mutation_flags & model_flag) != 0) {
            payload_dirty_.store(
                true, std::memory_order_release);
        }
    }

    lfs::Result<void>
    ProjectLifecycle::adoptCompletedTrainingSnapshot(
        const bool allow_during_application_close) {
        // ForceExit still suppresses adoption. Close-pending
        // only blocks *silent* snapshot adoption (info /
        // hasDirtyProject) so a background trainer write
        // cannot clear the exit gate. Settlement of a
        // GUI-requested training write must still rebase
        // the in-memory document onto the published head.
        if (suppress_training_adoption_ ||
            (application_close_pending_ &&
             !allow_during_application_close)) {
            return {};
        }
        auto* trainer = viewer_.getTrainer();
        if (!trainer) {
            return {};
        }
        const auto metrics =
            trainer->get_project_snapshot_metrics();
        if (metrics.writer_in_flight ||
            metrics.last_path.empty()) {
            return {};
        }
        const bool counter_advanced =
            metrics.capture.completed_snapshots >
            adopted_training_snapshot_count_;
        if (!counter_advanced) {
            return {};
        }
        if (counter_advanced &&
            !metrics.last_writer_error.empty()) {
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "The latest training project generation failed.",
                metrics.last_writer_error,
                "project.training_snapshot");
        }
        auto opened = ProjectDocument::open(
            metrics.last_path,
            {
                .reader = {},
                .geometry = {},
                .defer_geometry_payloads = true,
            });
        if (!opened) {
            if (!metrics.last_writer_error.empty()) {
                return fail<void>(
                    lfs::ErrorCode::Unavailable,
                    "The latest training project generation failed.",
                    metrics.last_writer_error,
                    "project.training_snapshot");
            }
            return lfs::Status::failure(
                std::move(opened).error());
        }
        if (document_ &&
            document_->source_path() &&
            document_->source_path()
                    ->lexically_normal() ==
                metrics.last_path
                    .lexically_normal() &&
            document_->project_uuid() ==
                opened->project_uuid() &&
            document_->generation() >=
                opened->generation()) {
            adopted_training_snapshot_count_ =
                metrics.capture
                    .completed_snapshots;
            return {};
        }
        document_ =
            std::make_shared<ProjectDocument>(
                std::move(*opened));
        cached_project_info_.reset();
        cached_bound_checkpoint_iteration_.reset();
        adopted_training_snapshot_count_ =
            metrics.capture.completed_snapshots;
        hydration_.store(
            Hydration::Complete,
            std::memory_order_release);
        hydration_error_.clear();
        {
            const std::lock_guard lock(
                settings_mutex_);
            rememberProject(
                settings_,
                document_->project_uuid(),
                metrics.last_path);
        }
        if (auto persisted = persistSettings();
            !persisted) {
            LOG_WARN(
                "Training project generation was adopted, but MRU settings failed: {}",
                developerError(persisted.error()));
        }
        LOG_INFO(
            "Adopted training .licht generation {} from {}",
            document_->generation(),
            lfs::core::path_to_utf8(
                metrics.last_path));
        bindTrainerSnapshotTarget();
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::
        adoptSettledTrainerPublishOntoCurrentMaster() {
        // Step-boundary / sparsity publishes complete on the
        // trainer writer, not through TrainingExplicitSave.
        // Rebase only a successful append onto the bound master;
        // Save As already rebinds via adopt(true) on settlement.
        auto* trainer = viewer_.getTrainer();
        if (!trainer || !document_ ||
            !document_->source_path()) {
            return {};
        }
        const auto metrics =
            trainer->get_project_snapshot_metrics();
        if (metrics.writer_in_flight ||
            metrics.last_path.empty() ||
            !metrics.last_writer_error.empty()) {
            return {};
        }
        if (metrics.capture.completed_snapshots <=
            adopted_training_snapshot_count_) {
            return {};
        }
        if (document_->source_path()
                ->lexically_normal() !=
            metrics.last_path.lexically_normal()) {
            return {};
        }
        return adoptCompletedTrainingSnapshot();
    }

    lfs::Result<void>
    ProjectLifecycle::prepareForEditModeTransition() {
        auto* const trainer = viewer_.getTrainer();
        if (!trainer) {
            return {};
        }
        const auto before =
            trainer->get_project_snapshot_metrics();
        if (before.writer_in_flight) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Wait for the final training snapshot before entering Edit Mode.",
                "Edit Mode cannot release the trainer while its project writer still owns the next durable generation",
                "project.training_snapshot");
        }
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            return adopted;
        }
        const auto after =
            trainer->get_project_snapshot_metrics();
        if (after.capture.completed_snapshots >
            adopted_training_snapshot_count_) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The final training snapshot is not ready for Edit Mode.",
                "A completed trainer generation remains unadopted; retaining the trainer prevents stale clean proofs",
                "project.training_snapshot");
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::synchronizeDocumentFromViewer() {
        return synchronizeDocumentFromViewer(
            DocumentSyncMode::Default);
    }

    lfs::Result<void>
    ProjectLifecycle::synchronizeDocumentFromViewer(
        const DocumentSyncMode mode) {
        if (!document_) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "There is no active project document.",
                "Project lifecycle initialization failed",
                "project.document");
        }
        auto* manager = viewer_.getSceneManager();
        if (!manager) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The scene is not available.",
                "Project capture requires SceneManager",
                "project.scene");
        }
        auto& scene = manager->getScene();

        auto existing_nodes =
            document_->scene_graph().nodes();
        if (!existing_nodes) {
            return lfs::Status::failure(
                std::move(existing_nodes).error());
        }
        lfs::io::project::ScenePayloadBindings
            bindings;
        for (const auto& record : *existing_nodes) {
            if (record.payload) {
                bindings.emplace(
                    record.uuid, *record.payload);
            }
        }

        const auto training_uuid =
            scene.getTrainingModelNodeUuid();
        std::vector<lfs::core::Uuid>
            omit_unbound_training;
        for (const auto* node : scene.getNodes()) {
            if (!node) {
                continue;
            }
            const bool geometry =
                node->type ==
                    lfs::core::NodeType::SPLAT ||
                node->type ==
                    lfs::core::NodeType::POINTCLOUD ||
                node->type ==
                    lfs::core::NodeType::MESH;
            if (!geometry) {
                continue;
            }
            const auto existing =
                bindings.find(node->uuid);
            if (existing != bindings.end()) {
                continue;
            }
            if (node->uuid == training_uuid) {
                if (mode ==
                    DocumentSyncMode::
                        LightTrainingAutosave) {
                    omit_unbound_training.push_back(
                        node->uuid);
                    continue;
                }
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    "The training model needs a safe-point project snapshot.",
                    "A new training node has no CKPT binding; save while the "
                    "trainer is active or after its terminal snapshot",
                    "SCNG.training_model_uuid");
            }
            std::string fourcc;
            std::string source_kind;
            if (node->type ==
                lfs::core::NodeType::SPLAT) {
                fourcc = "SPLT";
                source_kind = "generated";
            } else if (
                node->type ==
                lfs::core::NodeType::POINTCLOUD) {
                fourcc = "PCLD";
                source_kind = "pointcloud";
            } else {
                fourcc = "MESH";
                source_kind = "mesh";
            }
            bindings.emplace(
                node->uuid,
                PayloadBinding{
                    .fourcc = std::move(fourcc),
                    .instance_uuid = node->uuid,
                    .reference_uuid = std::nullopt,
                    .source_kind =
                        std::move(source_kind),
                });
        }

        auto captured_scene =
            lfs::io::project::capture_scene_graph(
                scene, bindings,
                omit_unbound_training);
        if (!captured_scene) {
            return lfs::Status::failure(
                std::move(captured_scene).error());
        }
        std::unordered_set<lfs::core::Uuid>
            captured_scene_uuids;
        if (const auto captured_nodes =
                captured_scene->nodes();
            captured_nodes) {
            captured_scene_uuids.reserve(
                captured_nodes->size());
            for (const auto& node : *captured_nodes) {
                captured_scene_uuids.insert(node.uuid);
            }
        }
        const auto old_scene_bytes =
            document_->scene_graph().to_bytes();
        const auto new_scene_bytes =
            captured_scene->to_bytes();
        if (!sameBytes(
                old_scene_bytes, new_scene_bytes)) {
            document_->edit_scene_graph() =
                std::move(*captured_scene);
        }
        // Entering Edit Mode turns the live training model into an ordinary
        // splat and clears its SCNG training binding.  A full sync must also
        // retire the formerly resumable CKPT; otherwise validation correctly
        // rejects the now-orphaned checkpoint.  Keep it during lightweight
        // autosaves while a training session is still bound.
        if (mode == DocumentSyncMode::Default &&
            training_uuid.is_nil()) {
            const auto checkpoint_uuids =
                document_->checkpoint_uuids();
            for (const auto& uuid : checkpoint_uuids) {
                static_cast<void>(
                    document_->remove_checkpoint(uuid));
            }
            if (!checkpoint_uuids.empty()) {
                cached_bound_checkpoint_iteration_.reset();
            }
        }
        // Light autosave omits the unbound live training
        // node from SCNG. Drop CKPT chapters that node no
        // longer binds; otherwise V21 rejects the sidecar.
        if (mode ==
                DocumentSyncMode::
                    LightTrainingAutosave &&
            !omit_unbound_training.empty()) {
            std::unordered_set<lfs::core::Uuid>
                bound_checkpoints;
            if (const auto nodes =
                    document_->scene_graph().nodes();
                nodes) {
                for (const auto& node : *nodes) {
                    if (node.payload &&
                        node.payload->fourcc == "CKPT") {
                        bound_checkpoints.insert(
                            node.payload->instance_uuid);
                    }
                }
            }
            const auto checkpoint_uuids =
                document_->checkpoint_uuids();
            bool removed_any = false;
            for (const auto& uuid : checkpoint_uuids) {
                if (!bound_checkpoints.contains(uuid)) {
                    static_cast<void>(
                        document_->remove_checkpoint(
                            uuid));
                    removed_any = true;
                }
            }
            if (removed_any) {
                cached_bound_checkpoint_iteration_.reset();
            }
        }
        // Same bookkeeping as the trainer writer after a
        // wholesale SCNG install: drop SPLT/PCLD/MESH
        // payloads the captured scene no longer binds.
        document_
            ->remove_geometry_payloads_not_bound_by_scene();

        std::unordered_set<lfs::core::Uuid>
            live_splats;
        std::unordered_set<lfs::core::Uuid>
            live_points;
        std::unordered_set<lfs::core::Uuid>
            live_meshes;
        const bool capture_payloads =
            payload_dirty_.load(
                std::memory_order_acquire) ||
            !document_->source_path();
        for (const auto* node : scene.getNodes()) {
            if (!node) {
                continue;
            }
            const auto binding =
                bindings.find(node->uuid);
            if (binding == bindings.end()) {
                continue;
            }
            const auto& fourcc =
                binding->second.fourcc;
            if (fourcc == "SPLT") {
                live_splats.insert(node->uuid);
            } else if (fourcc == "PCLD") {
                live_points.insert(node->uuid);
            } else if (fourcc == "MESH") {
                live_meshes.insert(node->uuid);
            } else {
                continue;
            }
            const bool already_present =
                (fourcc == "SPLT" &&
                 document_->find_splat(node->uuid)) ||
                (fourcc == "PCLD" &&
                 document_->find_point_cloud(
                     node->uuid)) ||
                (fourcc == "MESH" &&
                 document_->find_mesh(node->uuid)) ||
                std::ranges::any_of(
                    document_->payload_states(),
                    [&](const auto& state) {
                        return state.instance_uuid ==
                                   node->uuid &&
                               state.fourcc.to_string() ==
                                   fourcc;
                    });
            if (already_present && !capture_payloads) {
                continue;
            }
            if (node->payload_hydration !=
                lfs::core::PayloadHydrationState::Loaded) {
                if (!already_present) {
                    return fail<void>(
                        lfs::ErrorCode::FailedPrecondition,
                        "An unloaded node has no clean project payload.",
                        std::format(
                            "{} node {} cannot be written as empty",
                            fourcc, node->uuid.to_string()),
                        "project.partial_save");
                }
                continue;
            }

            if (fourcc == "SPLT") {
                if (!node->model) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A loaded splat node has no model.",
                        node->uuid.to_string(),
                        "SPLT");
                }
                auto payload =
                    lfs::io::project::
                        SplatChapterPayload::capture(
                            *node->model,
                            lfs::io::project::
                                SplatSourceKind::Generated,
                            false);
                if (!payload) {
                    return lfs::Status::failure(
                        std::move(payload).error());
                }
                if (auto set =
                        document_->set_splat(
                            node->uuid,
                            std::move(*payload));
                    !set) {
                    return set;
                }
            } else if (fourcc == "PCLD") {
                if (!node->point_cloud) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A loaded point-cloud node has no payload.",
                        node->uuid.to_string(),
                        "PCLD");
                }
                if (auto set =
                        document_->set_point_cloud(
                            node->uuid,
                            lfs::io::project::
                                PointCloudPayload(
                                    node->point_cloud));
                    !set) {
                    return set;
                }
            } else if (fourcc == "MESH") {
                if (!node->mesh) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A loaded mesh node has no payload.",
                        node->uuid.to_string(),
                        "MESH");
                }
                if (auto set =
                        document_->set_mesh(
                            node->uuid,
                            lfs::io::project::
                                MeshPayload(node->mesh));
                    !set) {
                    return set;
                }
            }
            auto provenance =
                payloadProvenance(
                    *manager, *node, fourcc);
            if (!provenance) {
                return lfs::Status::failure(
                    std::move(provenance).error());
            }
            auto& project =
                document_->edit_project();
            if (auto decision =
                    project.upsert_embed_decision(
                        {
                            .uuid = node->uuid,
                            .node_uuid = node->uuid,
                            .payload_fourcc = fourcc,
                            .decision = "embedded",
                            .reference_uuid =
                                std::nullopt,
                            .reason =
                                "project-owned scene payload",
                        });
                !decision) {
                return decision;
            }
            if (auto recorded =
                    project
                        .upsert_embedded_payload_provenance(
                            *provenance);
                !recorded) {
                return recorded;
            }
        }
        for (const auto& uuid :
             document_->splat_uuids()) {
            if (!live_splats.contains(uuid)) {
                static_cast<void>(
                    document_->remove_splat(uuid));
            }
        }
        for (const auto& uuid :
             document_->point_cloud_uuids()) {
            if (!live_points.contains(uuid)) {
                static_cast<void>(
                    document_->remove_point_cloud(
                        uuid));
            }
        }
        for (const auto& uuid :
             document_->mesh_uuids()) {
            if (!live_meshes.contains(uuid)) {
                static_cast<void>(
                    document_->remove_mesh(uuid));
            }
        }

        const bool all_geometry_loaded =
            std::ranges::all_of(
                scene.getNodes(), [](const auto* node) {
                    if (!node) {
                        return true;
                    }
                    const bool geometry =
                        node->type ==
                            lfs::core::NodeType::SPLAT ||
                        node->type ==
                            lfs::core::NodeType::
                                POINTCLOUD ||
                        node->type ==
                            lfs::core::NodeType::MESH;
                    return !geometry ||
                           node->payload_hydration ==
                               lfs::core::
                                   PayloadHydrationState::
                                       Loaded;
                });
        const auto selected =
            selectedNodeUuids(viewer_);
        auto captured_selection =
            lfs::io::project::
                capture_selection_chapter(
                    scene, selected,
                    omit_unbound_training);
        if (!captured_selection) {
            return lfs::Status::failure(
                std::move(captured_selection).error());
        }
        lfs::io::project::SelectionChapter
            selection =
                all_geometry_loaded
                    ? std::move(*captured_selection)
                    : document_->selection();
        if (!all_geometry_loaded) {
            if (auto groups = selection.set_groups(
                    captured_selection->groups(),
                    captured_selection
                        ->active_group_id(),
                    captured_selection
                        ->next_group_id());
                !groups) {
                return groups;
            }
            if (auto selected_result =
                    selection
                        .set_selected_node_uuids(
                            captured_selection
                                ->selected_node_uuids());
                !selected_result) {
                return selected_result;
            }

            for (const auto* node :
                 scene.getNodes()) {
                if (!node) {
                    continue;
                }
                const bool geometry =
                    node->type ==
                        lfs::core::NodeType::SPLAT ||
                    node->type ==
                        lfs::core::NodeType::
                            POINTCLOUD ||
                    node->type ==
                        lfs::core::NodeType::MESH;
                if (!geometry) {
                    continue;
                }
                if (node->payload_hydration !=
                    lfs::core::
                        PayloadHydrationState::
                            Loaded) {
                    continue;
                }
                static_cast<void>(
                    selection.remove_slice(
                        node->uuid,
                        lfs::core::
                            SelectionDomain::Splat));
                static_cast<void>(
                    selection.remove_slice(
                        node->uuid,
                        lfs::core::
                            SelectionDomain::
                                PointCloud));
                for (const auto& slice :
                     captured_selection->slices()) {
                    if (slice.node_uuid ==
                        node->uuid) {
                        if (auto upsert =
                                selection.upsert_slice(
                                    slice);
                            !upsert) {
                            return upsert;
                        }
                    }
                }
            }
            const auto old_slices =
                selection.slices();
            for (const auto& slice :
                 old_slices) {
                if (!captured_scene_uuids.contains(
                        slice.node_uuid)) {
                    static_cast<void>(
                        selection.remove_slice(
                            slice.node_uuid,
                            slice.domain));
                }
            }
        }
        auto old_selection =
            lfs::io::project::
                encode_selection_chapter(
                    document_->selection());
        auto new_selection =
            lfs::io::project::
                encode_selection_chapter(
                    selection);
        if (!old_selection) {
            return lfs::Status::failure(
                std::move(old_selection).error());
        }
        if (!new_selection) {
            return lfs::Status::failure(
                std::move(new_selection).error());
        }
        if (!sameBytes(
                *old_selection, *new_selection)) {
            document_->edit_selection() =
                std::move(selection);
        }

        const auto project_root =
            projectRootFor(*document_);
        auto staged_references =
            document_->references();
        if (auto* parameters =
                viewer_.getParameterManager()) {
            auto snapshot =
                parameters
                    ->capturePendingProjectState();
            if (!snapshot) {
                return lfs::Status::failure(
                    std::move(snapshot).error());
            }
            std::filesystem::path dataset_path;
            if (manager->hasDataset()) {
                dataset_path =
                    manager->getDatasetPath();
                if (dataset_path.empty()) {
                    if (const auto* trainer =
                            viewer_.getTrainer()) {
                        dataset_path = trainer->getParams()
                                           .dataset.data_path;
                    }
                }
            }
            lfs::training::absolutize_dataset_path_for_snapshot(
                dataset_path);
            if (!dataset_path.empty()) {
                snapshot->dataset.data_path =
                    dataset_path;

                auto staged_project =
                    document_->project();
                std::optional<lfs::core::Uuid>
                    existing;
                if (const auto current =
                        staged_project
                            .dataset_reference();
                    current && *current) {
                    existing = **current;
                }
                auto minted =
                    lfs::io::project::
                        upsert_path_reference(
                            staged_references,
                            project_root,
                            dataset_path,
                            "dataset", "dataset",
                            existing);
                if (!minted) {
                    return lfs::Status::failure(
                        std::move(minted).error());
                }
                if (auto set =
                        staged_project
                            .set_dataset_reference(
                                *minted);
                    !set) {
                    return set;
                }
                if (!sameBytes(
                        document_->project()
                            .to_bytes(),
                        staged_project.to_bytes())) {
                    document_->edit_project() =
                        std::move(staged_project);
                }
            }
            mintParameterPathReferences(
                staged_references, project_root,
                *snapshot);
            auto staged_parameters =
                document_->parameters();
            if (auto set =
                    staged_parameters.set_snapshot(
                        *snapshot);
                !set) {
                return set;
            }
            if (!sameBytes(
                    document_->parameters().to_bytes(),
                    staged_parameters.to_bytes())) {
                document_->edit_parameters() =
                    std::move(staged_parameters);
            }
        }

        // The prepared session remains authoritative until both GUI restore
        // gates have installed it. Capturing the still-default live owners
        // here would falsely dirty a read-only project and an early save or
        // close would permanently replace clean GUIL/VIEW/EDTR/SEQR/METR
        // chapters with those defaults.
        if (!viewer_.isProjectSessionRestorePending()) {
            auto session =
                viewer_.captureProjectSession(
                    &staged_references, project_root,
                    omit_unbound_training);
            if (!session) {
                return lfs::Status::failure(
                    std::move(session).error());
            }
            if (!sameBytes(
                    document_->gui_layout().to_bytes(),
                    session->gui_layout.to_bytes())) {
                document_->edit_gui_layout() =
                    std::move(session->gui_layout);
            }
            if (!sameBytes(
                    document_->view().to_bytes(),
                    session->view.to_bytes())) {
                document_->edit_view() =
                    std::move(session->view);
            }
            if (!sameBytes(
                    document_->editor().to_bytes(),
                    session->editor.to_bytes())) {
                document_->edit_editor() =
                    std::move(session->editor);
            }
            if (!sameBytes(
                    document_->sequencer().to_bytes(),
                    session->sequencer.to_bytes())) {
                document_->edit_sequencer() =
                    std::move(session->sequencer);
            }
            auto current_metrics =
                document_->metrics().to_bytes();
            auto captured_metrics =
                session->metrics.to_bytes();
            if (!current_metrics) {
                return lfs::Status::failure(
                    std::move(current_metrics).error());
            }
            if (!captured_metrics) {
                return lfs::Status::failure(
                    std::move(captured_metrics).error());
            }
            if (!sameBytes(
                    *current_metrics,
                    *captured_metrics)) {
                document_->edit_metrics() =
                    std::move(session->metrics);
            }
        }
        if (!sameBytes(
                document_->references().to_bytes(),
                staged_references.to_bytes())) {
            document_->edit_references() =
                std::move(staged_references);
        }

        scene_dirty_.store(
            false, std::memory_order_release);
        payload_dirty_.store(
            false, std::memory_order_release);
        return {};
    }

    lfs::Result<std::vector<std::byte>>
    ProjectLifecycle::capturePreviewPng() const {
        if (!viewer_.isOnViewerThread()) {
            LOG_ERROR(
                "capturePreviewPng called off the viewer thread; skipping viewport capture");
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::FailedPrecondition,
                "The viewport preview is unavailable off the viewer thread.",
                "Preview capture must be requested on the viewer thread",
                "THMB.thread");
        }
        auto captured = lfs::vis::capture_viewport_render();
        if (!captured || !captured->image) {
            // THMB is optional.  A new or otherwise frame-less project must
            // still be saveable; for an existing project, an empty span also
            // carries the previous preview forward.
            LOG_WARN("Viewport preview is unavailable; saving without regenerating THMB");
            return std::vector<std::byte>{};
        }
        auto image =
            captured->image->clone()
                .to(lfs::core::Device::CPU)
                .to(lfs::core::DataType::Float32);
        if (image.ndim() == 4) {
            image = image.squeeze(0);
        }
        if (image.ndim() != 3) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::DataLoss,
                "The viewport preview has an unsupported tensor shape.",
                "THMB capture requires a 3D image tensor",
                "THMB.tensor");
        }
        const auto layout =
            lfs::rendering::detectImageLayout(image);
        if (layout ==
            lfs::rendering::ImageLayout::CHW) {
            image = image.permute({1, 2, 0});
        } else if (
            layout ==
            lfs::rendering::ImageLayout::Unknown) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::DataLoss,
                "The viewport preview has an unsupported layout.",
                "THMB capture requires HWC or CHW",
                "THMB.tensor");
        }
        image =
            (image.clamp(0, 1) * 255.0f)
                .to(lfs::core::DataType::UInt8)
                .contiguous();
        const int source_height =
            static_cast<int>(image.shape()[0]);
        const int source_width =
            static_cast<int>(image.shape()[1]);
        const int channels =
            static_cast<int>(image.shape()[2]);
        if (source_width <= 0 || source_height <= 0 ||
            channels < 1 || channels > 4) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::DataLoss,
                "The viewport preview dimensions are invalid.",
                std::format(
                    "{}x{}x{}", source_width,
                    source_height, channels),
                "THMB.tensor");
        }
        constexpr int LONG_EDGE = 256;
        const double scale =
            std::min(
                1.0,
                static_cast<double>(LONG_EDGE) /
                    std::max(source_width,
                             source_height));
        const int width =
            std::max(
                1, static_cast<int>(
                       std::lround(
                           source_width * scale)));
        const int height =
            std::max(
                1, static_cast<int>(
                       std::lround(
                           source_height * scale)));
        const auto* source =
            image.ptr<std::uint8_t>();
        std::vector<std::uint8_t> resized(
            static_cast<std::size_t>(width) *
            height * channels);
        for (int y = 0; y < height; ++y) {
            const int source_y =
                std::min(
                    source_height - 1,
                    y * source_height / height);
            for (int x = 0; x < width; ++x) {
                const int source_x =
                    std::min(
                        source_width - 1,
                        x * source_width / width);
                std::memcpy(
                    resized.data() +
                        (static_cast<std::size_t>(y) *
                             width +
                         x) *
                            channels,
                    source +
                        (static_cast<std::size_t>(
                             source_y) *
                             source_width +
                         source_x) *
                            channels,
                    static_cast<std::size_t>(
                        channels));
            }
        }
        std::vector<std::byte> png;
        if (!stbi_write_png_to_func(
                pngWriteCallback, &png, width, height,
                channels, resized.data(),
                width * channels)) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::Unavailable,
                "The project preview could not be encoded.",
                "stbi_write_png_to_func failed", "THMB");
        }
        return png;
    }

    lfs::Result<void>
    ProjectLifecycle::save(
        const bool regenerate_preview) {
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "A project save is already in progress.",
                "Only one project save may run at a time",
                "project.save");
        }
        if (auto waited =
                waitOutBackgroundAutosaveForExplicitSave();
            !waited) {
            return waited;
        }
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            if (canFlushFinishedTrainerSnapshot()) {
                LOG_WARN(
                    "Discarding unadoptable training snapshot; a finished trainer can still flush: {}",
                    developerError(adopted.error()));
            } else {
                return lfs::Status::failure(
                    std::move(adopted).error());
            }
        }
        if (!document_ ||
            !document_->source_path()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "This project has no path; use Save As.",
                "An untitled project cannot be appended in place",
                "project.path");
        }
        if (auto* trainer = viewer_.getTrainer();
            trainer &&
            viewer_.getTrainerManager() &&
            (viewer_.getTrainerManager()->hasLiveTrainingThread() ||
             trainer->can_flush_project_snapshot()) &&
            (viewer_.getTrainerManager()->isTrainingActive() ||
             viewer_.getTrainerManager()->isCompletionPending())) {
            if (viewer_.getTrainerManager()->isPublishingFinalSnapshot()) {
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    "Wait for training completion before saving.",
                    "The trainer is still publishing its final snapshot",
                    "project.training");
            }
            return startLiveTrainingSnapshotWrite(
                ProjectWritePurpose::
                    TrainingExplicitSave,
                regenerate_preview);
        }
        if (canFlushFinishedTrainerSnapshot()) {
            return startLiveTrainingSnapshotWrite(
                ProjectWritePurpose::
                    TrainingExplicitSave,
                regenerate_preview);
        }
        if (auto synchronized =
                synchronizeDocumentFromViewer();
            !synchronized) {
            return lfs::Status::failure(
                std::move(synchronized).error());
        }
        std::vector<std::byte> preview;
        if (regenerate_preview) {
            auto captured = capturePreviewPng();
            if (!captured) {
                return lfs::Status::failure(
                    std::move(captured).error());
            }
            preview = std::move(*captured);
        }
        const auto destination =
            recovered_master_path_
                .value_or(
                    *document_->source_path());
        const auto purpose =
            recovered_master_path_
                ? ProjectWritePurpose::SaveAs
                : ProjectWritePurpose::
                      ExplicitSave;
        auto started = startDocumentWrite(
            purpose, document_, destination,
            ProjectDocumentSaveOptions{
                .commit =
                    {
                        .kind =
                            recovered_master_path_
                                ? lfs::io::project::
                                      CommitKind::Recovered
                                : lfs::io::project::
                                      CommitKind::Explicit,
                        .commit_uuid =
                            lfs::core::
                                generate_uuid_v4(),
                        .snapshot_uuid = {},
                        .wallclock_unix_ns =
                            unixTimeNs(),
                        .extra_reader_capabilities =
                            {},
                        .extra_writer_capabilities =
                            {},
                    },
                .file_uuid =
                    lfs::core::generate_uuid_v4(),
                .index_compression =
                    lfs::io::project::
                        IndexCompression::Zstd,
                .disk_reserve_bytes =
                    64ull * 1024 * 1024,
                .preview_png = preview,
                .writer_lock_lease =
                    recovery_session_
                        ? std::optional{
                              recovery_session_
                                  ->writer_lock()}
                        : std::nullopt,
            });
        if (!started) {
            return started;
        }
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::saveAs(
        const std::filesystem::path& path,
        const bool regenerate_preview,
        const bool allow_existing_destination_replacement) {
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "A project save is already in progress.",
                "Only one project save may run at a time",
                "project.save");
        }
        if (auto waited =
                waitOutBackgroundAutosaveForExplicitSave();
            !waited) {
            return waited;
        }
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            if (canFlushFinishedTrainerSnapshot()) {
                LOG_WARN(
                    "Discarding unadoptable training snapshot; a finished trainer can still flush: {}",
                    developerError(adopted.error()));
            } else {
                return lfs::Status::failure(
                    std::move(adopted).error());
            }
        }
        auto normalized =
            normalizedProjectPath(path);
        if (!normalized) {
            return lfs::Status::failure(
                std::move(normalized).error());
        }
        if (auto* trainer = viewer_.getTrainer();
            trainer &&
            viewer_.getTrainerManager() &&
            (viewer_.getTrainerManager()->hasLiveTrainingThread() ||
             trainer->can_flush_project_snapshot()) &&
            (viewer_.getTrainerManager()->isTrainingActive() ||
             viewer_.getTrainerManager()->isCompletionPending() ||
             canFlushFinishedTrainerSnapshot())) {
            if (viewer_.getTrainerManager()->isPublishingFinalSnapshot()) {
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    "Wait for training completion before saving.",
                    "The trainer is still publishing its final snapshot",
                    "project.training");
            }
            auto context =
                captureTrainingDocumentContext(
                    viewer_, *document_,
                    recovered_master_path_
                        ? lfs::io::project::
                              CommitKind::Recovered
                        : lfs::io::project::
                              CommitKind::Explicit,
                    recovery_session_ &&
                            recovered_master_path_ &&
                            normalized->lexically_normal() ==
                                recovered_master_path_
                                    ->lexically_normal()
                        ? std::optional{
                              recovery_session_
                                  ->writer_lock()}
                        : std::nullopt);
            if (!context) {
                return lfs::Status::failure(
                    std::move(context).error());
            }
            context->allow_existing_destination_replacement =
                allow_existing_destination_replacement;
            std::vector<std::byte> preview;
            if (regenerate_preview) {
                auto captured =
                    capturePreviewPng();
                if (!captured) {
                    return lfs::Status::failure(
                        std::move(captured).error());
                }
                preview = std::move(*captured);
            }
            const auto request_id =
                trainer
                    ->request_project_save(
                        *normalized,
                        std::move(preview),
                        std::move(*context));
            if (!viewer_.getTrainerManager()
                     ->hasLiveTrainingThread() &&
                trainer->can_flush_project_snapshot()) {
                trainer
                    ->consume_requested_project_snapshot(
                        trainer
                            ->project_snapshot_iteration());
            }
            return startTrainingWrite(
                ProjectWritePurpose::
                    TrainingExplicitSave,
                request_id, *normalized,
                document_->dirty_epoch(),
                scene_mutation_serial_.load(
                    std::memory_order_acquire));
        }
        if (auto synchronized =
                synchronizeDocumentFromViewer();
            !synchronized) {
            return lfs::Status::failure(
                std::move(synchronized).error());
        }
        std::vector<std::byte> preview;
        if (regenerate_preview) {
            auto captured = capturePreviewPng();
            if (!captured) {
                return lfs::Status::failure(
                    std::move(captured).error());
            }
            preview = std::move(*captured);
        }
        auto started = startDocumentWrite(
            ProjectWritePurpose::SaveAs,
            document_, *normalized,
            ProjectDocumentSaveOptions{
                .commit =
                    {
                        .kind =
                            recovered_master_path_
                                ? lfs::io::project::
                                      CommitKind::Recovered
                                : lfs::io::project::
                                      CommitKind::Explicit,
                        .commit_uuid =
                            lfs::core::
                                generate_uuid_v4(),
                        .snapshot_uuid = {},
                        .wallclock_unix_ns =
                            unixTimeNs(),
                        .extra_reader_capabilities =
                            {},
                        .extra_writer_capabilities =
                            {},
                    },
                .file_uuid =
                    lfs::core::generate_uuid_v4(),
                .index_compression =
                    lfs::io::project::
                        IndexCompression::Zstd,
                .disk_reserve_bytes =
                    64ull * 1024 * 1024,
                .allow_existing_destination_replacement =
                    allow_existing_destination_replacement,
                .preview_png = preview,
                .writer_lock_lease =
                    recovery_session_ &&
                            recovered_master_path_ &&
                            normalized->lexically_normal() ==
                                recovered_master_path_
                                    ->lexically_normal()
                        ? std::optional{
                              recovery_session_
                                  ->writer_lock()}
                        : std::nullopt,
            });
        if (!started) {
            return started;
        }
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        return {};
    }

    lfs::Result<ProjectOpenOutcome>
    ProjectLifecycle::open(
        const std::filesystem::path& path,
        const ProjectSwitchDisposition disposition) {
        const auto open_started =
            std::chrono::steady_clock::now();
        project_open_started_at_ = open_started;
        project_first_render_pending_ = false;
        recovery_prompt_pending_ = false;
        if (auto preflight =
                preflightSwitch(disposition);
            !preflight) {
            return std::move(preflight).error();
        }
        if (viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            return fail<ProjectOpenOutcome>(
                lfs::ErrorCode::FailedPrecondition,
                "A project write is still running.",
                "Wait for save, autosave, or compaction before opening another project",
                "project.job");
        }
        const auto preflight_finished =
            std::chrono::steady_clock::now();
        auto normalized =
            normalizedProjectPath(path);
        if (!normalized) {
            return std::move(normalized).error();
        }
        const auto normalized_at =
            std::chrono::steady_clock::now();
        const bool opening_scratch =
            lfs::io::project::is_scratch_autosave_path(
                *normalized, recovery_directory_);
        auto inspection =
            opening_scratch
                ? lfs::io::project::
                      inspect_scratch_autosave(
                          *normalized)
                : lfs::io::project::
                      inspect_autosave_recovery(
                          *normalized);
        if (!inspection) {
            return std::move(inspection).error();
        }
        const auto recovery_inspected_at =
            std::chrono::steady_clock::now();
        const auto previous_autosave_sequence =
            autosave_sequence_;
        autosave_sequence_ =
            std::max(
                autosave_sequence_,
                inspection
                    ->autosave_sequence);
        if (inspection->disposition ==
                lfs::io::project::
                    RecoveryDisposition::Offer &&
            inspection->selected_path) {
            RecoveryCandidate candidate{
                .master_path =
                    opening_scratch
                        ? std::filesystem::path{}
                        : *normalized,
                .selected_path =
                    inspection->selected_path
                        ->lexically_normal(),
                .autosave_sequence =
                    inspection->autosave_sequence,
                .commit_uuid =
                    inspection->commit_uuid,
                .snapshot_uuid =
                    inspection->snapshot_uuid,
                .wallclock_unix_ns =
                    inspection->wallclock_unix_ns,
                .untitled_scratch = opening_scratch ||
                                    inspection
                                        ->untitled_scratch,
            };
            const DeclinedRecoveryIdentity
                offered_identity{
                    .sidecar_path =
                        candidate.selected_path,
                    .autosave_sequence =
                        candidate.autosave_sequence,
                    .commit_uuid =
                        candidate.commit_uuid,
                };
            if (isRecoveryDismissed(
                    offered_identity)) {
                if (candidate.untitled_scratch) {
                    return ProjectOpenOutcome::Opened;
                }
                auto opened = openMaster(
                    *normalized, disposition);
                if (!opened) {
                    return std::move(opened).error();
                }
                return ProjectOpenOutcome::Opened;
            }
            // The offered sidecar is the unsaved state
            // the user just authorized discarding for
            // this project. Prompting Recover? right
            // after Discard? is the #1641 confusion.
            // Do not set declined_recovery_ here:
            // openMaster's discard cleanup deletes the
            // sidecar instead.
            const auto current_master =
                recovered_master_path_
                    ? recovered_master_path_
                    : (document_
                           ? document_->source_path()
                           : std::nullopt);
            if (!candidate.untitled_scratch &&
                disposition ==
                    ProjectSwitchDisposition::
                        DiscardChanges &&
                current_master &&
                current_master->lexically_normal() ==
                    normalized->lexically_normal()) {
                auto opened = openMaster(
                    *normalized, disposition);
                if (!opened) {
                    return std::move(opened).error();
                }
                return ProjectOpenOutcome::Opened;
            }
            auto* gui =
                viewer_.getGuiManager();
            if (!gui) {
                return fail<ProjectOpenOutcome>(
                    lfs::ErrorCode::
                        FailedPrecondition,
                    "A recoverable autosave is available.",
                    "Headless project resume performs recovery automatically",
                    "project.recovery");
            }
            enqueueRecoveryPrompt(
                std::move(candidate),
                disposition,
                previous_autosave_sequence);
            return ProjectOpenOutcome::
                RecoveryPromptPending;
        }
        auto opened = openMaster(
            *normalized, disposition);
        if (!opened) {
            return std::move(opened).error();
        }
        const auto open_finished =
            std::chrono::steady_clock::now();
        const auto milliseconds =
            [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(
                           end - begin)
                    .count();
            };
        LOG_DEBUG(
            "Project open dispatch stages: path={} preflight={:.3f} ms normalize={:.3f} ms recovery_scan={:.3f} ms master_open={:.3f} ms total={:.3f} ms",
            lfs::core::path_to_utf8(*normalized),
            milliseconds(
                open_started,
                preflight_finished),
            milliseconds(
                preflight_finished,
                normalized_at),
            milliseconds(
                normalized_at,
                recovery_inspected_at),
            milliseconds(
                recovery_inspected_at,
                open_finished),
            milliseconds(open_started, open_finished));
        return ProjectOpenOutcome::Opened;
    }

    lfs::Result<void>
    ProjectLifecycle::openRecovered(
        const std::filesystem::path& master_path,
        const std::filesystem::path& sidecar_path,
        const ProjectSwitchDisposition disposition) {
        auto session =
            lfs::io::project::
                begin_recovery_session(
                    master_path,
                    sidecar_path);
        if (!session) {
            return lfs::Status::failure(
                std::move(session).error());
        }
        const auto temporary =
            lfs::io::project::
                recovery_session_temp_path(
                    master_path);
        if (auto materialized =
                lfs::io::project::
                    materialize_recovered_project(
                        master_path,
                        sidecar_path,
                        temporary, *session);
            !materialized) {
            return materialized;
        }
        auto opened =
            openMaster(
                temporary, disposition);
        if (!opened) {
            static_cast<void>(
                session->release());
            return opened;
        }
        session->attach_document();
        recovered_master_path_ =
            master_path;
        recovery_session_path_ =
            temporary;
        recovery_session_ =
            std::move(*session);
        {
            const std::lock_guard lock(
                settings_mutex_);
            rememberProject(
                settings_,
                document_->project_uuid(),
                master_path);
        }
        if (auto persisted = persistSettings();
            !persisted) {
            LOG_WARN(
                "Recovered project opened, but MRU settings failed: {}",
                developerError(
                    persisted.error()));
        }
        resetMaintenanceClocks();
        LOG_INFO(
            "Recovered autosave {} over master {}",
            lfs::core::path_to_utf8(
                sidecar_path),
            lfs::core::path_to_utf8(
                master_path));
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::openMaster(
        const std::filesystem::path& path,
        const ProjectSwitchDisposition disposition) {
        recovery_prompt_pending_ = false;
        if (auto preflight =
                preflightSwitch(disposition);
            !preflight) {
            return preflight;
        }
        std::optional<std::filesystem::path> discard_master;
        if (disposition ==
            ProjectSwitchDisposition::DiscardChanges) {
            if (recovered_master_path_) {
                discard_master = *recovered_master_path_;
            } else if (document_ &&
                       document_->source_path()) {
                discard_master =
                    *document_->source_path();
            }
        }
        const auto started =
            std::chrono::steady_clock::now();
        auto normalized =
            resolveLichtFilePath(path);
        if (!normalized) {
            return lfs::Status::failure(
                std::move(normalized).error());
        }
        const auto normalized_at =
            std::chrono::steady_clock::now();
        auto opened = ProjectDocument::open(
            *normalized,
            {
                .reader = {},
                .geometry = {},
                .defer_geometry_payloads = true,
            });
        if (!opened) {
            return lfs::Status::failure(
                std::move(opened).error());
        }
        auto candidate =
            std::make_shared<ProjectDocument>(
                std::move(*opened));
        const auto document_opened_at =
            std::chrono::steady_clock::now();
        const auto project_root =
            projectRootFor(*candidate);
        auto session =
            prepareGuiSessionRestore(
                {
                    .gui_layout =
                        candidate->gui_layout(),
                    .editor = candidate->editor(),
                    .view = candidate->view(),
                    .sequencer =
                        candidate->sequencer(),
                    .metrics = candidate->metrics(),
                },
                &candidate->references(),
                project_root);
        if (!session) {
            return lfs::Status::failure(
                std::move(session).error());
        }
        const auto session_prepared_at =
            std::chrono::steady_clock::now();
        auto parameters =
            candidate->parameters().snapshot();
        if (!parameters) {
            return lfs::Status::failure(
                std::move(parameters).error());
        }
        resolveParameterPathReferences(
            candidate->references(), project_root,
            *parameters);
        auto* parameter_manager =
            viewer_.getParameterManager();
        if (!parameter_manager) {
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "The parameter manager is unavailable.",
                "The visualizer has not initialized its parameter manager",
                "project.parameters");
        }
        if (auto valid =
                ParameterManager::
                    validatePendingProjectState(
                        *parameters);
            !valid) {
            return valid;
        }
        const auto parameters_ready_at =
            std::chrono::steady_clock::now();
        auto* manager = viewer_.getSceneManager();
        if (!manager) {
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "The scene manager is unavailable.",
                "The visualizer has not initialized its scene manager",
                "project.open");
        }
        auto shell =
            candidate->stage_shell(
                manager->getScene());
        if (!shell) {
            return lfs::Status::failure(
                std::move(shell).error());
        }
        const auto shell_staged_at =
            std::chrono::steady_clock::now();

        // A stale import completion must not outlive
        // a project switch.
        if (auto* const gui = viewer_.getGuiManager()) {
            gui->asyncTasks().cancelImport();
        }

        stopHydrationThreads();
        if (auto* trainer_manager = viewer_.getTrainerManager();
            trainer_manager && trainer_manager->hasTrainer() &&
            !trainer_manager->clearTrainer()) {
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "The previous training session could not be cleared.",
                "Project switching requires the trainer to reach its terminal state",
                "project.training");
        }
        viewer_.deactivateProjectTools();
        viewer_.resetProjectState();
        manager->setDatasetPath({});
        manager->getScene().commitRestoreStage(
            std::move(*shell));
        manager->changeContentType(
            inferContentType(
                manager->getScene()));
        parameter_manager
            ->installValidatedPendingProjectState(
                *parameters);
        active_restore_ticket_ =
            viewer_.stagePreparedProjectSessionRestore(
                std::move(*session));
        const auto shell_committed_at =
            std::chrono::steady_clock::now();

        cached_project_info_.reset();
        cached_bound_checkpoint_iteration_.reset();
        document_ = candidate;
        bindTrainerSnapshotTarget();
        cleanupRecoverySession();
        // Removal runs only after the replacement
        // document is installed and
        // cleanupRecoverySession() has released the
        // old recovery session's writer lock. Every
        // failure path of openMaster returns before
        // this point, so a failed switch never
        // deletes the old project's recovery
        // artifacts
        // (FailedNewProjectKeepsRecoveredSessionTemp
        // relies on that). Capture happens before
        // cleanupRecoverySession() because it resets
        // recovered_master_path_.
        if (discard_master) {
            removeDiscardedAutosaveArtifacts(
                *discard_master);
        }
        adopted_training_snapshot_count_ = 0;
        application_close_pending_ = false;
        suppress_training_adoption_ = false;
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        op::undoHistory().clear();
        const auto epoch =
            epoch_.fetch_add(
                1, std::memory_order_acq_rel) +
            1;
        pending_dataset_relocation_.reset();
        hydration_.store(
            Hydration::ShellReady,
            std::memory_order_release);
        hydration_error_.clear();
        lfs::core::events::state::SceneChanged{
            .mutation_flags =
                static_cast<std::uint32_t>(
                    lfs::core::Scene::MutationType::
                        CLEARED) |
                static_cast<std::uint32_t>(
                    lfs::core::Scene::MutationType::
                        NODE_ADDED)}
            .emit();
        scene_dirty_.store(
            false, std::memory_order_release);
        payload_dirty_.store(
            false, std::memory_order_release);
        const auto shell_selection_serial =
            selection_mutation_serial_.load(
                std::memory_order_acquire);
        const auto shell_selected_nodes =
            selectedNodeUuids(viewer_);
        const bool opened_scratch =
            lfs::io::project::is_scratch_autosave_path(
                *normalized, recovery_directory_);
        if (!opened_scratch) {
            removeScratchAutosave();
            const std::lock_guard lock(
                settings_mutex_);
            rememberProject(
                settings_,
                candidate->project_uuid(),
                *normalized);
        }
        if (!opened_scratch) {
            if (auto persisted = persistSettings();
                !persisted) {
                LOG_WARN(
                    "Project opened, but MRU settings failed: {}",
                    developerError(persisted.error()));
            }
        }
        const auto lifecycle_installed_at =
            std::chrono::steady_clock::now();
        project_open_job_ = viewer_.jobs().init(
            JobType::ProjectOpen, "Reading project");
        if (!project_open_job_) {
            markHydrationFailed(
                epoch,
                "The project-open progress job could not be started.");
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "Project opening could not be started.",
                "The project-open job registry has no available slot",
                "project.open.job");
        }
        if (auto launched =
                launchHydration(
                    candidate, epoch,
                    shell_selection_serial,
                    shell_selected_nodes,
                    active_restore_ticket_);
            !launched) {
            markHydrationFailed(
                epoch,
                developerError(launched.error()));
        }
        const auto hydration_launched_at =
            std::chrono::steady_clock::now();
        const auto shell_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                started)
                .count();
        const auto milliseconds =
            [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(
                           end - begin)
                    .count();
            };
        LOG_DEBUG(
            "Project shell stages: path={} normalize={:.3f} ms document_open={:.3f} ms session_restore={:.3f} ms parameters={:.3f} ms shell_stage={:.3f} ms shell_commit={:.3f} ms lifecycle_install={:.3f} ms hydration_launch={:.3f} ms total={:.3f} ms",
            lfs::core::path_to_utf8(*normalized),
            milliseconds(started, normalized_at),
            milliseconds(
                normalized_at,
                document_opened_at),
            milliseconds(
                document_opened_at,
                session_prepared_at),
            milliseconds(
                session_prepared_at,
                parameters_ready_at),
            milliseconds(
                parameters_ready_at,
                shell_staged_at),
            milliseconds(
                shell_staged_at,
                shell_committed_at),
            milliseconds(
                shell_committed_at,
                lifecycle_installed_at),
            milliseconds(
                lifecycle_installed_at,
                hydration_launched_at),
            milliseconds(started, hydration_launched_at));
        LOG_INFO(
            "Project shell ready in {:.3f} ms: {} (generation {}, {} payload units)",
            shell_ms,
            lfs::core::path_to_utf8(*normalized),
            candidate->generation(),
            candidate->payload_states().size());
        resetMaintenanceClocks();
        refreshStorageStats();
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::launchHydration(
        std::shared_ptr<ProjectDocument> document,
        const std::uint64_t epoch,
        const std::uint64_t selection_mutation_serial,
        std::vector<lfs::core::Uuid>
            selected_node_uuids,
        const std::uint64_t restore_ticket) {
        auto* manager = viewer_.getSceneManager();
        if (!manager) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Project hydration cannot start.",
                "SceneManager is unavailable",
                "hydrate");
        }
        if (!document->source_path()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Project hydration cannot start.",
                "The active project has no immutable source generation",
                "hydrate.source");
        }
        const auto source_path =
            *document->source_path();
        const auto project_uuid =
            document->project_uuid();
        const auto minimum_generation =
            document->generation();
        hydration_.store(
            Hydration::Hydrating,
            std::memory_order_release);
        auto allocator =
            manager->makeExternalSplatAllocator();
        if (allocator) {
            manager->getScene().setCombinedModelAllocator(
                allocator);
        }
        std::lock_guard lock(thread_mutex_);
        try {
            hydration_threads_.emplace_back(
                [this, document = std::move(document),
                 source_path, project_uuid,
                 minimum_generation,
                 epoch, selection_mutation_serial,
                 restore_ticket,
                 selected_node_uuids =
                     std::move(selected_node_uuids),
                 allocator = std::move(allocator)](
                    const std::stop_token stop) mutable {
                    const auto project_open_job = project_open_job_;
                    if (project_open_job) {
                        viewer_.jobs().work(*project_open_job);
                        viewer_.jobs().report(
                            *project_open_job, 0.05F, "Reading project");
                    }
                    const auto hydration_started =
                        std::chrono::steady_clock::now();
                    if (stop.stop_requested()) {
                        return;
                    }
                    auto opened_source =
                        ProjectDocument::open(
                            source_path,
                            {
                                .reader = {},
                                .geometry = {},
                                .defer_geometry_payloads =
                                    true,
                            });
                    if (!opened_source) {
                        if (!stop.stop_requested()) {
                            markHydrationFailed(
                                epoch,
                                developerError(
                                    opened_source
                                        .error()));
                        }
                        return;
                    }
                    if (opened_source->project_uuid() !=
                            project_uuid ||
                        opened_source->generation() <
                            minimum_generation) {
                        if (!stop.stop_requested()) {
                            markHydrationFailed(
                                epoch,
                                std::format(
                                    "The immutable hydration source changed from project {} generation {} to project {} generation {}",
                                    project_uuid.to_string(),
                                    minimum_generation,
                                    opened_source
                                        ->project_uuid()
                                        .to_string(),
                                    opened_source
                                        ->generation()));
                        }
                        return;
                    }
                    const auto source_opened_at =
                        std::chrono::steady_clock::now();
                    if (project_open_job) {
                        viewer_.jobs().report(
                            *project_open_job, 0.20F, "Decompressing project data");
                    }
                    auto* scene_manager =
                        viewer_.getSceneManager();
                    if (!scene_manager) {
                        if (!stop.stop_requested()) {
                            markHydrationFailed(
                                epoch,
                                "The scene manager became unavailable during project hydration.");
                        }
                        return;
                    }
                    auto staged =
                        opened_source->stage_hydration(
                            scene_manager->getScene(), {},
                            std::move(allocator),
                            [this, project_open_job](const std::size_t completed,
                                                     const std::size_t total) {
                                if (project_open_job && total != 0) {
                                    viewer_.jobs().report(
                                        *project_open_job,
                                        0.20F + 0.55F * static_cast<float>(completed) /
                                                    static_cast<float>(total),
                                        "Decompressing project data");
                                }
                            });
                    if (!staged) {
                        const auto detail =
                            developerError(staged.error());
                        if (!stop.stop_requested()) {
                            markHydrationFailed(
                                epoch, detail);
                        }
                        return;
                    }
                    const auto payload_staged_at =
                        std::chrono::steady_clock::now();
                    if (project_open_job) {
                        viewer_.jobs().report(
                            *project_open_job, 0.75F, "Uploading project data");
                    }
                    auto plan =
                        std::make_shared<
                            std::optional<
                                lfs::io::project::
                                    ProjectHydrationPlan>>(
                            std::move(*staged));
                    if (project_open_job) {
                        viewer_.jobs().finishWork(*project_open_job, false);
                    }
                    const auto queued_at =
                        std::chrono::steady_clock::now();
                    const bool posted =
                        viewer_.postWork({
                            .run =
                                [this, document, epoch,
                                 selection_mutation_serial,
                                 selected_node_uuids,
                                 restore_ticket,
                                 plan, hydration_started,
                                 source_opened_at,
                                 payload_staged_at,
                                 queued_at, project_open_job] {
                                    if (epoch_.load(
                                            std::memory_order_acquire) !=
                                            epoch ||
                                        document_ != document ||
                                        !plan->has_value()) {
                                        return;
                                    }
                                    auto* manager =
                                        viewer_.getSceneManager();
                                    if (!manager) {
                                        markHydrationFailed(
                                            epoch,
                                            "The scene manager became unavailable while committing project hydration.");
                                        return;
                                    }
                                    const auto commit_started =
                                        std::chrono::steady_clock::
                                            now();
                                    const bool install_selection =
                                        selection_mutation_serial_.load(
                                            std::memory_order_acquire) ==
                                            selection_mutation_serial &&
                                        selectedNodeUuids(viewer_) ==
                                            selected_node_uuids;
                                    const bool scene_was_dirty =
                                        scene_dirty_.load(
                                            std::memory_order_acquire);
                                    const bool payload_was_dirty =
                                        payload_dirty_.load(
                                            std::memory_order_acquire);
                                    const auto report =
                                        ProjectDocument::
                                            commit_partial_hydration(
                                                manager->getScene(),
                                                std::move(
                                                    plan->value()),
                                                install_selection);
                                    plan->reset();
                                    manager->changeContentType(
                                        inferContentType(
                                            manager->getScene()));
                                    const auto scene_committed_at =
                                        std::chrono::steady_clock::
                                            now();
                                    if (report.selection_installed) {
                                        std::vector<
                                            lfs::core::NodeId>
                                            selected_ids;
                                        selected_ids.reserve(
                                            report.selection
                                                .selected_node_uuids
                                                .size());
                                        for (const auto& uuid :
                                             report.selection
                                                 .selected_node_uuids) {
                                            const auto id =
                                                manager->getScene()
                                                    .getNodeIdByUuid(
                                                        uuid);
                                            if (id !=
                                                lfs::core::
                                                    NULL_NODE) {
                                                selected_ids.push_back(
                                                    id);
                                            }
                                        }
                                        manager->clearSelection();
                                        if (!selected_ids.empty()) {
                                            manager->selectNodesById(
                                                selected_ids);
                                        }
                                    }
                                    viewer_.noteHydrationTerminalForRestoreTicket(
                                        restore_ticket);
                                    const auto selection_restored_at =
                                        std::chrono::steady_clock::
                                            now();
                                    // Trainer restore is soft: display
                                    // hydration already succeeded.
                                    if (epoch_.load(
                                            std::memory_order_acquire) ==
                                            epoch &&
                                        document_ == document) {
                                        tryInstallTrainerFromHydratedProject(
                                            *manager, *document,
                                            report);
                                    }
                                    const auto trainer_restored_at =
                                        std::chrono::steady_clock::
                                            now();
                                    if (report
                                            .checkpoint_header) {
                                        cached_bound_checkpoint_iteration_ =
                                            report
                                                .checkpoint_header
                                                ->iteration;
                                    }
                                    hydration_.store(
                                        Hydration::Complete,
                                        std::memory_order_release);
                                    hydration_error_.clear();
                                    if (report
                                            .hydrated_payload_units >
                                        0) {
                                        lfs::core::events::
                                            state::SceneChanged{
                                                .mutation_flags =
                                                    static_cast<
                                                        std::uint32_t>(
                                                        lfs::core::
                                                            Scene::
                                                                MutationType::
                                                                    MODEL_CHANGED)}
                                                .emit();
                                    }
                                    scene_dirty_.store(
                                        scene_was_dirty,
                                        std::memory_order_release);
                                    payload_dirty_.store(
                                        payload_was_dirty,
                                        std::memory_order_release);
                                    const auto hydration_finished =
                                        std::chrono::steady_clock::
                                            now();
                                    hydration_committed_at_ =
                                        hydration_finished;
                                    project_first_render_pending_ =
                                        true;
                                    if (project_open_job) {
                                        viewer_.jobs().completed(*project_open_job);
                                        viewer_.jobs().free(*project_open_job);
                                        project_open_job_.reset();
                                    }
                                    const auto milliseconds =
                                        [](const auto begin,
                                           const auto end) {
                                            return std::chrono::
                                                duration<double,
                                                         std::milli>(
                                                       end - begin)
                                                    .count();
                                        };
                                    LOG_DEBUG(
                                        "Project hydration stages: path={} source_reopen={:.3f} ms payload_stage={:.3f} ms queue_wait={:.3f} ms scene_commit={:.3f} ms selection_restore={:.3f} ms trainer_restore={:.3f} ms finalize={:.3f} ms total={:.3f} ms",
                                        document->source_path()
                                            ? lfs::core::
                                                  path_to_utf8(
                                                      *document
                                                           ->source_path())
                                            : std::string{
                                                  "<untitled>"},
                                        milliseconds(hydration_started, source_opened_at), milliseconds(source_opened_at, payload_staged_at), milliseconds(queued_at, commit_started), milliseconds(commit_started, scene_committed_at), milliseconds(scene_committed_at, selection_restored_at), milliseconds(selection_restored_at, trainer_restored_at), milliseconds(trainer_restored_at, hydration_finished), milliseconds(hydration_started, hydration_finished));
                                    LOG_INFO(
                                        "Project background hydration complete: {} (hydrated={}, invalidated={}, selection={})",
                                        document->source_path()
                                            ? lfs::core::
                                                  path_to_utf8(
                                                      *document
                                                           ->source_path())
                                            : std::string{
                                                  "<untitled>"},
                                        report.hydrated_payload_units, report.invalidated_payload_units, report.selection_installed ? "restored" : "preserved-live");
                                },
                            .cancel =
                                [plan] {
                                    plan->reset();
                                },
                        });
                    if (!posted) {
                        plan->reset();
                        if (project_open_job) {
                            viewer_.jobs().finishWork(
                                *project_open_job, false,
                                "The project hydration work was canceled.");
                        }
                        if (!stop.stop_requested()) {
                            markHydrationFailed(
                                epoch,
                                "The project hydration work could not be queued.");
                        }
                    }
                });
        } catch (const std::bad_alloc& error) {
            hydration_.store(
                Hydration::Failed,
                std::memory_order_release);
            return fail<void>(
                lfs::ErrorCode::ResourceExhausted,
                "Project hydration could not start.",
                error.what(), "hydrate.thread");
        } catch (const std::exception& error) {
            hydration_.store(
                Hydration::Failed,
                std::memory_order_release);
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "Project hydration could not start.",
                error.what(), "hydrate.thread");
        } catch (...) {
            hydration_.store(
                Hydration::Failed,
                std::memory_order_release);
            return fail<void>(
                lfs::ErrorCode::Internal,
                "Project hydration could not start.",
                "unknown exception while creating hydration worker",
                "hydrate.thread");
        }
        return {};
    }

    void ProjectLifecycle::markHydrationFailed(
        const std::uint64_t epoch,
        const std::string& detail) {
        viewer_.postWork({
            .run =
                [this, epoch, detail] {
                    if (epoch_.load(
                            std::memory_order_acquire) !=
                        epoch) {
                        return;
                    }
                    hydration_error_ = detail;
                    if (project_open_job_) {
                        if (auto job = viewer_.jobs().peek(*project_open_job_);
                            job && job->running()) {
                            viewer_.jobs().failed(
                                *project_open_job_, detail, "Project open failed");
                            viewer_.jobs().free(*project_open_job_);
                        }
                        project_open_job_.reset();
                    }
                    hydration_.store(
                        Hydration::Failed,
                        std::memory_order_release);
                    viewer_.noteHydrationTerminalForRestoreTicket(
                        active_restore_ticket_);
                    if (auto* manager =
                            viewer_.getSceneManager()) {
                        auto& scene =
                            manager->getScene();
                        std::unordered_set<
                            lfs::core::Uuid>
                            project_payloads;
                        if (document_) {
                            for (const auto& state :
                                 document_
                                     ->payload_states()) {
                                project_payloads.insert(
                                    state.instance_uuid);
                            }
                        }
                        for (const auto* node :
                             scene.getNodes()) {
                            if (node &&
                                project_payloads
                                    .contains(
                                        node->uuid) &&
                                node->payload_hydration ==
                                    lfs::core::
                                        PayloadHydrationState::
                                            Unloaded) {
                                static_cast<void>(
                                    scene.setPayloadHydrationState(
                                        node->uuid,
                                        lfs::core::
                                            PayloadHydrationState::
                                                Failed));
                            }
                        }
                    }
                    LOG_ERROR(
                        "Project hydration failed; the coherent shell remains active: {}",
                        detail);
                    publishProjectToast(
                        lfs::ErrorCode::Unavailable,
                        lfs::ErrorDomain::IO,
                        detail,
                        gui::error_op::kOpenProject);
                },
            .cancel = [] {},
        });
    }

    lfs::Result<void>
    ProjectLifecycle::newProject(
        const ProjectSwitchDisposition disposition) {
        if (auto preflight =
                preflightSwitch(disposition);
            !preflight) {
            return preflight;
        }
        std::optional<std::filesystem::path> discard_master;
        if (disposition ==
            ProjectSwitchDisposition::DiscardChanges) {
            if (recovered_master_path_) {
                discard_master = *recovered_master_path_;
            } else if (document_ &&
                       document_->source_path()) {
                discard_master =
                    *document_->source_path();
            }
        }
        auto created =
            ProjectDocument::create(
                lfs::core::generate_uuid_v4());
        if (!created) {
            return lfs::Status::failure(
                std::move(created).error());
        }
        if (!viewer_.getDataLoader() ||
            !viewer_.getDataLoader()->clearScene()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The current scene could not be cleared.",
                "The scene clear request was rejected before switching projects",
                "project.new");
        }
        stopHydrationThreads();
        cached_project_info_.reset();
        cached_bound_checkpoint_iteration_.reset();
        document_ =
            std::make_shared<ProjectDocument>(
                std::move(*created));
        cleanupRecoverySession();
        if (discard_master) {
            removeDiscardedAutosaveArtifacts(
                *discard_master);
        }
        removeScratchAutosave();
        adopted_training_snapshot_count_ = 0;
        application_close_pending_ = false;
        suppress_training_adoption_ = false;
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        viewer_.deactivateProjectTools();
        // A new document clears project-owned data, but it is not a workspace
        // preset. Preserve the live panel arrangement and desktop geometry.
        viewer_.resetProjectState(/*reset_panel_registry=*/false);
        active_restore_ticket_ = 0;
        recovery_prompt_pending_ = false;
        epoch_.fetch_add(
            1, std::memory_order_acq_rel);
        pending_dataset_relocation_.reset();
        hydration_.store(
            Hydration::Empty,
            std::memory_order_release);
        hydration_error_.clear();
        op::undoHistory().clear();
        scene_dirty_.store(
            false, std::memory_order_release);
        payload_dirty_.store(
            false, std::memory_order_release);
        storage_stats_ = {};
        compaction_suggested_ = false;
        autosave_sequence_ = 0;
        resetMaintenanceClocks();
        return {};
    }

    bool ProjectLifecycle::isDirty() {
        return hasDirtyProject();
    }

    bool ProjectLifecycle::hasSourcePath() const {
        return recovered_master_path_.has_value() ||
               (document_ &&
                document_->source_path().has_value());
    }

    bool ProjectLifecycle::hasDirtyProject() {
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return true;
        }
        if (viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            return true;
        }
        if (!document_) {
            return false;
        }
        // Never let a silent training snapshot adoption
        // satisfy the exit gate as NotDirty.
        if (!application_close_pending_ &&
            !suppress_training_adoption_) {
            if (auto adopted =
                    adoptCompletedTrainingSnapshot();
                !adopted) {
                const auto warning =
                    developerError(adopted.error());
                if (warning !=
                    last_unadoptable_training_snapshot_warning_) {
                    last_unadoptable_training_snapshot_warning_ =
                        warning;
                    LOG_ERROR(
                        "Could not adopt the completed training project generation: {}",
                        warning);
                }
                return true;
            }
            last_unadoptable_training_snapshot_warning_
                .clear();
        }
        if (viewer_.getTrainer() &&
            viewer_.getTrainerManager() &&
            viewer_.getTrainerManager()
                ->isTrainingActive() &&
            !viewer_.getTrainerManager()
                 ->isPausedAtCheckpointBaseline()) {
            return true;
        }
        if (isBlankUntitledSession()) {
            return false;
        }
        if (scene_dirty_.load(
                std::memory_order_acquire) ||
            payload_dirty_.load(
                std::memory_order_acquire)) {
            return true;
        }
        if (const auto* parameter_manager =
                viewer_.getParameterManager();
            parameter_manager && parameter_manager->isDirty()) {
            return true;
        }
        return hasHardDirtyChapters(*document_);
    }

    bool ProjectLifecycle::containsEmbeddedSecrets()
        const {
        if (!document_) {
            return false;
        }
        const auto value =
            document_->editor().dom().get_json(
                "contains_embedded_secrets");
        return value && value->is_boolean() &&
               value->get<bool>();
    }

    ProjectLifecycle::CloseSaveStatus
    ProjectLifecycle::beginOrPollCloseSave() {
        settleProjectWrite();
        application_close_pending_ = true;
        switch (close_save_state_.load(
            std::memory_order_acquire)) {
        case CloseSaveState::Saving:
            return CloseSaveStatus::Saving;
        case CloseSaveState::Succeeded:
            return CloseSaveStatus::Succeeded;
        case CloseSaveState::Failed:
            // Report this attempt once, then arm the state machine for a
            // subsequent titlebar/File close attempt. close_save_error_ stays
            // available for the fallback prompt until that attempt begins or
            // the user cancels it.
            close_save_state_.store(
                CloseSaveState::Idle,
                std::memory_order_release);
            return CloseSaveStatus::Failed;
        case CloseSaveState::Idle:
            break;
        }

        if (!hasDirtyProject()) {
            // Window-close X / File-Exit of a clean session: Skip any
            // unanswered recovery offer and drop the untitled scratch.
            // Emergency ForceExit never reaches this function, so crash
            // files survive an X11/interrupt teardown.
            if (recovery_prompt_pending_ &&
                pending_recovery_candidate_) {
                ++recovery_prompt_generation_;
                const auto candidate =
                    *pending_recovery_candidate_;
                handleRecoverySkip(candidate);
                recovery_prompt_pending_ = false;
            }
            removeScratchAutosave();
            return CloseSaveStatus::NotDirty;
        }
        if (viewer_.jobs().anyRunning(
                JobType::ProjectWrite)) {
            cancelBackgroundAutosaveIfRunning();
            return CloseSaveStatus::Saving;
        }
        if (!settings_.auto_save_on_close ||
            !document_ ||
            !document_->source_path()) {
            return CloseSaveStatus::NeedsPrompt;
        }
        if (viewer_.getTrainer()) {
            auto* const manager =
                viewer_.getTrainerManager();
            const bool training_snapshot =
                (manager &&
                 (manager->isTrainingActive() ||
                  manager
                      ->isCompletionPending())) ||
                canFlushFinishedTrainerSnapshot();
            if (training_snapshot) {
                auto started =
                    startLiveTrainingSnapshotWrite(
                        ProjectWritePurpose::
                            TrainingCloseSave,
                        false);
                if (!started) {
                    {
                        std::lock_guard lock(
                            close_save_mutex_);
                        close_save_error_ =
                            developerError(
                                started.error());
                    }
                    close_save_state_.store(
                        CloseSaveState::Failed,
                        std::memory_order_release);
                    return CloseSaveStatus::Failed;
                }
                close_save_state_.store(
                    CloseSaveState::Saving,
                    std::memory_order_release);
                return CloseSaveStatus::Saving;
            }
        }
        if (auto synchronized =
                synchronizeDocumentFromViewer();
            !synchronized) {
            {
                std::lock_guard lock(
                    close_save_mutex_);
                close_save_error_ =
                    developerError(
                        synchronized.error());
            }
            close_save_state_.store(
                CloseSaveState::Failed,
                std::memory_order_release);
            return CloseSaveStatus::Failed;
        }

        const auto document = document_;
        const auto path =
            recovered_master_path_.value_or(
                *document_->source_path());
        const ProjectDocumentSaveOptions options{
            .commit =
                {
                    .kind =
                        recovered_master_path_
                            ? lfs::io::project::
                                  CommitKind::Recovered
                            : lfs::io::project::
                                  CommitKind::Explicit,
                    .commit_uuid =
                        lfs::core::
                            generate_uuid_v4(),
                    .snapshot_uuid = {},
                    .wallclock_unix_ns =
                        unixTimeNs(),
                    .extra_reader_capabilities =
                        {},
                    .extra_writer_capabilities =
                        {},
                },
            .file_uuid =
                lfs::core::generate_uuid_v4(),
            .index_compression =
                lfs::io::project::
                    IndexCompression::Zstd,
            .disk_reserve_bytes =
                64ull * 1024 * 1024,
            // Save-on-close is automatic: carry THMB
            // forward rather than issuing a render.
            .preview_png = {},
            .writer_lock_lease =
                recovery_session_
                    ? std::optional{
                          recovery_session_->writer_lock()}
                    : std::nullopt,
        };
        {
            std::lock_guard lock(
                close_save_mutex_);
            close_save_error_.clear();
        }
        auto started = startDocumentWrite(
            ProjectWritePurpose::CloseSave,
            document, path, options);
        if (!started) {
            {
                std::lock_guard lock(
                    close_save_mutex_);
                close_save_error_ =
                    developerError(
                        started.error());
            }
            close_save_state_.store(
                CloseSaveState::Failed,
                std::memory_order_release);
            return CloseSaveStatus::Failed;
        }
        close_save_state_.store(
            CloseSaveState::Saving,
            std::memory_order_release);
        return CloseSaveStatus::Saving;
    }

    void ProjectLifecycle::resetCloseSaveAttempt() {
        application_close_pending_ = false;
        suppress_training_adoption_ = false;
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return;
        }
        {
            std::lock_guard lock(
                close_save_mutex_);
            close_save_error_.clear();
        }
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
    }

    std::string ProjectLifecycle::closeSaveError()
        const {
        std::lock_guard lock(close_save_mutex_);
        return close_save_error_;
    }

    void ProjectLifecycle::markApplicationClosePending() {
        application_close_pending_ = true;
    }

    void ProjectLifecycle::markCloseDiscardRequested() {
        close_discard_requested_ = true;
    }

    void ProjectLifecycle::removeDiscardedAutosaveArtifacts(
        const std::filesystem::path& master) {
        if (auto removed =
                lfs::io::project::
                    remove_autosave_artifacts(
                        master);
            !removed) {
            LOG_WARN(
                "Discarded autosave cleanup failed for {}: {}",
                master.string(),
                developerError(removed.error()));
        }
    }

    bool ProjectLifecycle::isApplicationClosePending()
        const {
        return application_close_pending_;
    }

    void ProjectLifecycle::setSuppressTrainingAdoption(
        const bool suppress) {
        suppress_training_adoption_ = suppress;
    }

    ProjectMenuInfo ProjectLifecycle::menuInfo()
        const {
        const std::lock_guard lock(
            settings_mutex_);
        ProjectMenuInfo result{
            .reopen_last_project =
                settings_.reopen_last_project,
            .auto_save_on_close =
                settings_.auto_save_on_close,
            .autosave_interval_seconds =
                settings_.autosave_interval_seconds,
            .recent_projects = {},
        };
        result.recent_projects.reserve(
            settings_.mru.size());
        for (const auto& entry : settings_.mru) {
            result.recent_projects.push_back({
                .project_uuid =
                    entry.project_uuid.to_string(),
                .last_known_path =
                    entry.last_known_path,
            });
        }
        return result;
    }

    ProjectWritePoll ProjectLifecycle::pollWrite() {
        settleProjectWrite();
        ProjectWritePoll poll;
        if (project_write_job_) {
            const auto job = viewer_.jobs().peek(
                *project_write_job_);
            poll.running =
                !job || job->running();
            if (job) {
                poll.error = job->error;
                poll.error_code = job->error_code;
            }
        }
        if (document_) {
            poll.generation = document_->generation();
            poll.path =
                recovered_master_path_
                    ? recovered_master_path_
                    : document_->source_path();
        }
        if (poll.error.empty()) {
            poll.error = last_project_write_error_;
            poll.error_code =
                last_project_write_error_code_;
        }
        return poll;
    }

    lfs::Result<ProjectInfo>
    ProjectLifecycle::info() {
        settleProjectWrite();
        if (!document_) {
            return fail<ProjectInfo>(
                lfs::ErrorCode::FailedPrecondition,
                "There is no active project document.",
                "Project lifecycle has not created or opened a document",
                "project.document");
        }
        const bool close_save_running =
            close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving;
        if (close_save_running || project_write_job_) {
            auto result =
                cached_project_info_
                    .value_or(ProjectInfo{});
            const auto job =
                viewer_.jobs().peek(
                    *project_write_job_);
            result.project_write_running =
                true;
            result.project_write_stage =
                job ? job->stage
                    : std::string{};
            result.project_write_progress =
                job ? job->progress : 0.0F;
            result.project_write_error =
                job ? job->error
                    : std::string{};
            result.project_write_error_code =
                job ? job->error_code
                    : std::nullopt;
            result.autosave_sequence =
                autosave_sequence_;
            result.recovery_session =
                recovered_master_path_
                    .has_value();
            result.compaction_suggested =
                compaction_suggested_;
            result.physical_bytes =
                storage_stats_.physical_bytes;
            result.estimated_live_bytes =
                storage_stats_
                    .estimated_live_bytes;
            result.dead_bytes =
                storage_stats_.dead_bytes;
            result.dead_ratio =
                storage_stats_.dead_ratio;
            return result;
        }
        const std::lock_guard document_lock(
            document_access_mutex_);
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            // Unadoptable last training generation must
            // not fail read surfaces; MCP save_as
            // preflights through info().
            const auto warning =
                developerError(adopted.error());
            if (warning !=
                last_unadoptable_training_snapshot_warning_) {
                last_unadoptable_training_snapshot_warning_ =
                    warning;
                LOG_WARN(
                    "Discarding unadoptable training snapshot; project info continues with the current document: {}",
                    warning);
            }
        } else {
            last_unadoptable_training_snapshot_warning_
                .clear();
        }
        const bool blank_untitled =
            isBlankUntitledSession();
        const auto* manager =
            viewer_.getSceneManager();
        const auto* trainer_manager =
            viewer_.getTrainerManager();
        const bool training_active =
            viewer_.getTrainer() && trainer_manager &&
            trainer_manager->isTrainingActive();
        const bool training_forces_dirty =
            training_active &&
            !trainer_manager
                 ->isPausedAtCheckpointBaseline();
        const bool parameter_dirty = [&] {
            const auto* parameter_manager =
                viewer_.getParameterManager();
            return parameter_manager &&
                   parameter_manager->isDirty();
        }();
        const auto dirty_chapters =
            blank_untitled
                ? std::vector<std::string>{}
                : document_->dirty_chapters();
        const bool hard_dirty =
            training_forces_dirty ||
            (!blank_untitled &&
             (scene_dirty_.load(
                  std::memory_order_acquire) ||
              payload_dirty_.load(
                  std::memory_order_acquire) ||
              parameter_dirty ||
              hasHardDirtyChapters(*document_)));
        const bool session_dirty =
            !blank_untitled &&
            hasSessionSoftDirtyChapters(*document_);
        ProjectInfo result{
            .path =
                recovered_master_path_
                    ? recovered_master_path_
                    : document_->source_path(),
            .project_uuid =
                document_->project_uuid().to_string(),
            .generation = document_->generation(),
            .dirty = hard_dirty,
            .session_dirty = session_dirty,
            .dirty_chapters = dirty_chapters,
            .hydration_state =
                hydrationName(
                    hydration_.load(
                        std::memory_order_acquire)),
            .payloads = {},
            .contains_embedded_secrets =
                containsEmbeddedSecrets(),
            .reopen_last_project =
                settings_.reopen_last_project,
            .auto_save_on_close =
                settings_.auto_save_on_close,
            .autosave_interval_seconds =
                settings_
                    .autosave_interval_seconds,
            .autosave_dirty_epoch_threshold =
                settings_
                    .autosave_dirty_epoch_threshold,
            .project_write_running = false,
            .project_write_stage = {},
            .project_write_progress = 0.0F,
            .project_write_error =
                last_project_write_error_,
            .project_write_error_code =
                last_project_write_error_code_,
            .autosave_sequence =
                autosave_sequence_,
            .recovery_session =
                recovered_master_path_
                    .has_value(),
            .compaction_suggested =
                compaction_suggested_,
            .physical_bytes =
                storage_stats_.physical_bytes,
            .estimated_live_bytes =
                storage_stats_
                    .estimated_live_bytes,
            .dead_bytes =
                storage_stats_.dead_bytes,
            .dead_ratio =
                storage_stats_.dead_ratio,
            .hydration_error = hydration_error_,
            .recent_projects = {},
        };
        for (const auto& entry : settings_.mru) {
            result.recent_projects.push_back({
                .project_uuid =
                    entry.project_uuid.to_string(),
                .last_known_path =
                    entry.last_known_path,
            });
        }
        for (const auto& state :
             document_->payload_states()) {
            std::string payload_hydration =
                state.loaded
                    ? "loaded"
                    : "unloaded";
            if (manager) {
                if (const auto* node =
                        manager->getScene()
                            .getNodeByUuid(
                                state.instance_uuid)) {
                    switch (
                        node->payload_hydration) {
                    case lfs::core::
                        PayloadHydrationState::
                            NotApplicable:
                        payload_hydration =
                            "not_applicable";
                        break;
                    case lfs::core::
                        PayloadHydrationState::
                            Unloaded:
                        payload_hydration =
                            "unloaded";
                        break;
                    case lfs::core::
                        PayloadHydrationState::
                            Hydrating:
                        payload_hydration =
                            "hydrating";
                        break;
                    case lfs::core::
                        PayloadHydrationState::
                            Loaded:
                        payload_hydration =
                            "loaded";
                        break;
                    case lfs::core::
                        PayloadHydrationState::
                            Failed:
                        payload_hydration =
                            "failed";
                        break;
                    }
                } else {
                    payload_hydration =
                        "invalidated";
                }
            }
            result.payloads.push_back({
                .chapter =
                    state.fourcc.to_string(),
                .node_uuid =
                    state.instance_uuid.to_string(),
                .hydration_state =
                    std::move(
                        payload_hydration),
            });
        }
        cached_project_info_ = result;
        return result;
    }

    void ProjectLifecycle::openStartupProject(
        const std::optional<
            std::filesystem::path>& explicit_path) {
        std::vector<std::filesystem::path> known;
        {
            const std::lock_guard lock(
                settings_mutex_);
            known.reserve(settings_.mru.size());
            for (const auto& entry : settings_.mru) {
                known.push_back(
                    resolveProjectMruPath(
                        entry.last_known_path));
            }
        }
        lfs::io::project::
            sweep_stale_licht_artifacts_for_known_masters(
                known);
        lfs::io::project::sweep_stale_scratch_autosaves(
            recovery_directory_);

        // Never auto-restore from MRU. Startup without an
        // explicit CLI project path leaves a blank session
        // after a clean previous session; the
        // recent-projects chooser (Python panel) offers
        // optional open when the MRU is non-empty. The
        // only exception is a detected unclean close with
        // a recoverable autosave, which routes into the
        // standard recovery prompt.
        if (!explicit_path) {
            offerStartupCrashRecovery();
            return;
        }
        if (auto opened = open(*explicit_path);
            !opened) {
            LOG_ERROR(
                "Failed to open startup project {}: {}",
                lfs::core::path_to_utf8(
                    *explicit_path),
                developerError(
                    opened.error()));
        } else if (auto* gui = viewer_.getGuiManager()) {
            gui->dismissStartupOverlay();
        }
    }

    void ProjectLifecycle::offerStartupCrashRecovery() {
        // Intentional closes remove crash files (#1652),
        // so an Offer here implies the previous session
        // ended uncleanly.
        auto candidate = selectStartupRecoveryCandidate();
        if (!candidate) {
            return;
        }
        LOG_INFO(
            "Unclean shutdown left a recoverable autosave for {}; offering recovery",
            lfs::core::path_to_utf8(
                candidate->untitled_scratch
                    ? candidate->selected_path
                    : candidate->master_path));
        auto* gui = viewer_.getGuiManager();
        if (!gui) {
            return;
        }
        const auto previous_autosave_sequence =
            autosave_sequence_;
        autosave_sequence_ = std::max(
            autosave_sequence_,
            candidate->autosave_sequence);
        enqueueRecoveryPrompt(
            std::move(*candidate),
            ProjectSwitchDisposition::RequireClean,
            previous_autosave_sequence);
        gui->dismissStartupOverlay();
    }

    std::optional<ProjectLifecycle::RecoveryCandidate>
    ProjectLifecycle::selectStartupRecoveryCandidate() {
        std::vector<RecoveryCandidate> offers;
        std::filesystem::path mru_path;
        {
            const std::lock_guard lock(
                settings_mutex_);
            if (!settings_.mru.empty()) {
                mru_path = resolveProjectMruPath(
                    settings_.mru.front()
                        .last_known_path);
            }
        }
        std::error_code ec;
        if (!mru_path.empty() &&
            std::filesystem::is_regular_file(
                mru_path, ec) &&
            !ec) {
            auto inspection =
                lfs::io::project::
                    inspect_autosave_recovery(
                        mru_path);
            if (!inspection) {
                LOG_WARN(
                    "Startup recovery scan skipped for {}: {}",
                    lfs::core::path_to_utf8(mru_path),
                    developerError(
                        inspection.error()));
            } else if (
                inspection->disposition ==
                    lfs::io::project::
                        RecoveryDisposition::Offer &&
                inspection->selected_path) {
                offers.push_back(RecoveryCandidate{
                    .master_path = mru_path,
                    .selected_path =
                        inspection->selected_path
                            ->lexically_normal(),
                    .autosave_sequence =
                        inspection->autosave_sequence,
                    .commit_uuid =
                        inspection->commit_uuid,
                    .snapshot_uuid =
                        inspection->snapshot_uuid,
                    .wallclock_unix_ns =
                        inspection->wallclock_unix_ns,
                    .untitled_scratch = false,
                });
            }
        }
        for (auto& inspection :
             lfs::io::project::scan_scratch_autosaves(
                 recovery_directory_)) {
            if (inspection.disposition !=
                    lfs::io::project::
                        RecoveryDisposition::Offer ||
                !inspection.selected_path) {
                continue;
            }
            offers.push_back(RecoveryCandidate{
                .master_path = {},
                .selected_path =
                    inspection.selected_path
                        ->lexically_normal(),
                .autosave_sequence =
                    inspection.autosave_sequence,
                .commit_uuid = inspection.commit_uuid,
                .snapshot_uuid =
                    inspection.snapshot_uuid,
                .wallclock_unix_ns =
                    inspection.wallclock_unix_ns,
                .untitled_scratch = true,
            });
        }
        offers.erase(
            std::remove_if(
                offers.begin(), offers.end(),
                [this](const RecoveryCandidate& candidate) {
                    return isRecoveryDismissed(
                        DeclinedRecoveryIdentity{
                            .sidecar_path =
                                candidate.selected_path,
                            .autosave_sequence =
                                candidate.autosave_sequence,
                            .commit_uuid =
                                candidate.commit_uuid,
                        });
                }),
            offers.end());
        if (offers.empty()) {
            return std::nullopt;
        }
        return *std::ranges::max_element(
            offers,
            [](const RecoveryCandidate& lhs,
               const RecoveryCandidate& rhs) {
                if (lhs.wallclock_unix_ns !=
                    rhs.wallclock_unix_ns) {
                    return lhs.wallclock_unix_ns <
                           rhs.wallclock_unix_ns;
                }
                return lhs.selected_path.generic_string() <
                       rhs.selected_path.generic_string();
            });
    }

    void ProjectLifecycle::enqueueRecoveryPrompt(
        RecoveryCandidate candidate,
        const ProjectSwitchDisposition disposition,
        const std::uint64_t previous_autosave_sequence) {
        auto* gui = viewer_.getGuiManager();
        if (!gui) {
            return;
        }
        recovery_prompt_pending_ = true;
        pending_recovery_candidate_ = candidate;
        const auto prompt_epoch =
            epoch_.load(std::memory_order_acquire);
        const auto prompt_generation =
            ++recovery_prompt_generation_;
        namespace Keys = lichtfeld::Strings::Recovery;
        const auto display_name =
            candidate.untitled_scratch
                ? std::string(LOC(Keys::UNSAVED_SESSION))
                : candidate.master_path.stem().string();
        const auto saved_at = formatRecoverySavedTime(
            candidate.wallclock_unix_ns,
            candidate.selected_path);
        const auto body = LOCF(
            Keys::BODY, display_name, saved_at);
        lfs::core::ModalRequest request;
        request.title = LOC(Keys::CRASH_TITLE);
        request.body_rml = std::format(
            "<p>{}</p>",
            lfs::vis::gui::escapeRmlText(body));
        request.style = lfs::core::ModalStyle::Warning;
        request.width_dp = 500;
        request.buttons = {
            {LOC(Keys::RECOVER), "primary"},
        };
        if (!candidate.untitled_scratch) {
            request.buttons.push_back(
                {LOC(Keys::OPEN_SAVED), "secondary"});
        }
        request.buttons.push_back(
            {LOC(Keys::SKIP), "secondary"});
        request.on_result =
            [this, candidate, disposition,
             previous_autosave_sequence, prompt_epoch,
             prompt_generation](
                const lfs::core::ModalResult& result) {
                if (epoch_.load(
                        std::memory_order_acquire) !=
                        prompt_epoch ||
                    recovery_prompt_generation_ !=
                        prompt_generation) {
                    return;
                }
                recovery_prompt_pending_ = false;
                pending_recovery_candidate_.reset();
                namespace ResultKeys =
                    lichtfeld::Strings::Recovery;
                lfs::Result<void> opened;
                if (result.button_label ==
                    LOC(ResultKeys::RECOVER)) {
                    opened =
                        candidate.untitled_scratch
                            ? openScratchRecovered(
                                  candidate.selected_path,
                                  disposition)
                            : openRecovered(
                                  candidate.master_path,
                                  candidate.selected_path,
                                  disposition);
                } else if (
                    !candidate.untitled_scratch &&
                    result.button_label ==
                        LOC(ResultKeys::OPEN_SAVED)) {
                    persistRecoveryDismissal(
                        DeclinedRecoveryIdentity{
                            .sidecar_path =
                                candidate.selected_path,
                            .autosave_sequence =
                                candidate.autosave_sequence,
                            .commit_uuid =
                                candidate.commit_uuid,
                        });
                    opened = openMaster(
                        candidate.master_path,
                        disposition);
                } else {
                    handleRecoverySkip(candidate);
                    autosave_sequence_ =
                        previous_autosave_sequence;
                    return;
                }
                if (!opened) {
                    LOG_ERROR(
                        "Recovery decision failed: {}",
                        developerError(
                            opened.error()));
                    publishProjectToast(
                        opened.error(),
                        gui::error_op::kOpenProject);
                }
            };
        request.on_cancel =
            [this, candidate,
             previous_autosave_sequence, prompt_epoch,
             prompt_generation] {
                if (epoch_.load(
                        std::memory_order_acquire) !=
                        prompt_epoch ||
                    recovery_prompt_generation_ !=
                        prompt_generation) {
                    return;
                }
                recovery_prompt_pending_ = false;
                pending_recovery_candidate_.reset();
                handleRecoverySkip(candidate);
                autosave_sequence_ =
                    previous_autosave_sequence;
            };
        gui->enqueueModal(std::move(request));
    }

    void ProjectLifecycle::handleRecoverySkip(
        const RecoveryCandidate& candidate) {
        persistRecoveryDismissal(
            DeclinedRecoveryIdentity{
                .sidecar_path = candidate.selected_path,
                .autosave_sequence =
                    candidate.autosave_sequence,
                .commit_uuid = candidate.commit_uuid,
            });
        if (candidate.untitled_scratch &&
            !candidate.selected_path.empty()) {
            if (auto removed =
                    lfs::io::project::
                        remove_scratch_autosave(
                            candidate.selected_path);
                !removed) {
                LOG_WARN(
                    "Could not remove skipped scratch autosave {}: {}",
                    lfs::core::path_to_utf8(
                        candidate.selected_path),
                    developerError(removed.error()));
            }
        }
        pending_recovery_candidate_.reset();
    }

    bool ProjectLifecycle::isRecoveryDismissed(
        const DeclinedRecoveryIdentity& identity) const {
        const auto matches =
            [&identity](
                const DismissedRecoveryEntry& stored) {
                if (stored.sidecar_path.lexically_normal() !=
                    identity.sidecar_path
                        .lexically_normal()) {
                    return false;
                }
                if (stored.autosave_sequence !=
                    identity.autosave_sequence) {
                    return false;
                }
                if (!stored.commit_uuid.is_nil() &&
                    !identity.commit_uuid.is_nil()) {
                    return stored.commit_uuid ==
                           identity.commit_uuid;
                }
                return true;
            };
        if (declined_recovery_ &&
            matches(*declined_recovery_)) {
            return true;
        }
        const std::lock_guard lock(settings_mutex_);
        return std::ranges::any_of(
            settings_.dismissed_recovery, matches);
    }

    void ProjectLifecycle::persistRecoveryDismissal(
        const DeclinedRecoveryIdentity& identity) {
        declined_recovery_ = identity;
        {
            const std::lock_guard lock(settings_mutex_);
            auto& entries = settings_.dismissed_recovery;
            std::erase_if(
                entries,
                [&identity](
                    const DismissedRecoveryEntry& stored) {
                    return stored.sidecar_path
                                   .lexically_normal() ==
                               identity.sidecar_path
                                   .lexically_normal() &&
                           stored.autosave_sequence ==
                               identity.autosave_sequence &&
                           (stored.commit_uuid.is_nil() ||
                            identity.commit_uuid.is_nil() ||
                            stored.commit_uuid ==
                                identity.commit_uuid);
                });
            entries.push_back(identity);
            constexpr std::size_t kMaxDismissed = 64;
            if (entries.size() > kMaxDismissed) {
                entries.erase(
                    entries.begin(),
                    entries.begin() +
                        static_cast<std::ptrdiff_t>(
                            entries.size() -
                            kMaxDismissed));
            }
        }
        if (auto persisted = persistSettings();
            !persisted) {
            LOG_WARN(
                "Could not persist recovery dismissal: {}",
                developerError(persisted.error()));
        }
    }

    lfs::Result<void>
    ProjectLifecycle::openScratchRecovered(
        const std::filesystem::path& scratch_path,
        const ProjectSwitchDisposition disposition) {
        auto opened = openMaster(
            scratch_path, disposition);
        if (!opened) {
            return opened;
        }
        if (document_) {
            document_->forget_source_path();
            [[maybe_unused]] auto& project =
                document_->edit_project();
        }
        scene_dirty_.store(
            true, std::memory_order_release);
        if (auto locked = lockScratchAutosave();
            !locked) {
            LOG_WARN(
                "Recovered untitled session could not rebind scratch autosave: {}",
                developerError(locked.error()));
        }
        return {};
    }

    std::filesystem::path
    ProjectLifecycle::scratchAutosaveDirectory() const {
        return recovery_directory_;
    }

    void ProjectLifecycle::removeScratchAutosave() {
        const auto path = scratch_autosave_path_;
        scratch_lock_.reset();
        scratch_autosave_path_.reset();
        if (!path || path->empty()) {
            return;
        }
        if (auto removed =
                lfs::io::project::remove_scratch_autosave(
                    *path);
            !removed) {
            LOG_WARN(
                "Could not remove scratch autosave {}: {}",
                lfs::core::path_to_utf8(*path),
                developerError(removed.error()));
        }
    }

    bool ProjectLifecycle::isBlankUntitledSession()
        const {
        if (!document_ || document_->source_path()) {
            return false;
        }
        if (hydration_.load(
                std::memory_order_acquire) !=
            Hydration::Empty) {
            return false;
        }
        const auto* manager =
            viewer_.getSceneManager();
        if (!manager ||
            !manager->getScene().getNodes().empty()) {
            return false;
        }
        return !scene_dirty_.load(
                   std::memory_order_acquire) &&
               !payload_dirty_.load(
                   std::memory_order_acquire);
    }

    lfs::Result<void>
    ProjectLifecycle::ensureScratchAutosaveBinding() {
        if (!document_) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "Untitled autosave has no document.",
                "scratch binding requires a live project document",
                "project.document");
        }
        if (recovery_directory_.empty()) {
            return fail<void>(
                lfs::ErrorCode::Unavailable,
                "Untitled crash recovery has no storage location.",
                "the user-storage recovery directory could not be resolved",
                "project.recovery_directory");
        }
        const auto destination =
            lfs::io::project::scratch_autosave_path(
                recovery_directory_,
                document_->project_uuid());
        if (scratch_autosave_path_ &&
            scratch_autosave_path_->lexically_normal() ==
                destination.lexically_normal()) {
            return {};
        }
        scratch_lock_.reset();
        scratch_autosave_path_ = destination;
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::lockScratchAutosave() {
        if (auto bound = ensureScratchAutosaveBinding();
            !bound) {
            return bound;
        }
        if (scratch_lock_ &&
            scratch_lock_->owns(*scratch_autosave_path_)) {
            return {};
        }
        scratch_lock_.reset();
        auto lease =
            lfs::io::project::WriterLockLease::acquire(
                *scratch_autosave_path_);
        if (!lease) {
            return lfs::Result<void>::failure(
                std::move(lease).error());
        }
        scratch_lock_ = std::move(*lease);
        return {};
    }

} // namespace lfs::vis::project
