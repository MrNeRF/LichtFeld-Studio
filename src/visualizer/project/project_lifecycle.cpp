/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "project_lifecycle.hpp"

#include "core/config_paths.hpp"
#include "core/data_loading_service.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "io/selection_chapter.hpp"
#include "operation/undo_history.hpp"
#include "project/session_state.hpp"
#include "rendering/image_layout.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "visualizer_impl.hpp"

#include <nlohmann/json.hpp>
#include <stb_image_write.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <ranges>
#include <set>
#include <unordered_map>
#include <unordered_set>

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

        [[nodiscard]] std::string developerError(
            const lfs::Error& error) {
            return lfs::format_for_developer(error);
        }

        [[nodiscard]] lfs::Result<
            lfs::training::
                ProjectSnapshotDocumentContext>
        captureTrainingDocumentContext(
            VisualizerImpl& viewer,
            const ProjectDocument& document) {
            auto session =
                viewer.captureProjectSession();
            if (!session) {
                return std::move(session).error();
            }
            return lfs::training::
                ProjectSnapshotDocumentContext{
                    .project_uuid =
                        document.project_uuid(),
                    .source_path =
                        document.source_path(),
                    .project = document.project(),
                    .references =
                        document.references(),
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

        [[nodiscard]] bool isLichtPath(
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

        [[nodiscard]] lfs::Result<
            std::filesystem::path>
        normalizedProjectPath(
            const std::filesystem::path& path) {
            if (path.empty() || !isLichtPath(path)) {
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
            std::ifstream stream(path);
            if (!stream) {
                return fail<ProjectLifecycleSettings>(
                    lfs::ErrorCode::PermissionDenied,
                    "Project lifecycle settings could not be opened.",
                    path.string(), "settings.path");
            }
            const Json json = Json::parse(stream);
            if (!json.is_object() ||
                json.value("version", 0) != 1) {
                return fail<ProjectLifecycleSettings>(
                    lfs::ErrorCode::DataLoss,
                    "Project lifecycle settings are invalid.",
                    "expected an object with version 1",
                    "settings.version");
            }
            settings.reopen_last_project =
                json.value("reopen_last_project", true);
            settings.auto_save_on_close =
                json.value("auto_save_on_close", true);
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
            std::error_code error;
            std::filesystem::create_directories(
                path.parent_path(), error);
            if (error) {
                return fail<void>(
                    lfs::ErrorCode::PermissionDenied,
                    "The project settings directory could not be created.",
                    error.message(), "settings.path");
            }
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
            const Json json{
                {"version", 1},
                {"reopen_last_project",
                 settings.reopen_last_project},
                {"auto_save_on_close",
                 settings.auto_save_on_close},
                {"mru", std::move(entries)},
            };
            const auto temporary =
                path.parent_path() /
                std::format(
                    ".{}.{}.tmp",
                    path.filename().string(),
                    lfs::core::generate_uuid_v4()
                        .to_string());
            {
                std::ofstream stream(
                    temporary,
                    std::ios::binary |
                        std::ios::trunc);
                if (!stream) {
                    return fail<void>(
                        lfs::ErrorCode::PermissionDenied,
                        "Project lifecycle settings could not be written.",
                        temporary.string(), "settings.path");
                }
                stream << json.dump(2) << '\n';
                stream.flush();
                if (!stream) {
                    return fail<void>(
                        lfs::ErrorCode::Unavailable,
                        "Project lifecycle settings could not be flushed.",
                        temporary.string(), "settings.path");
                }
            }
#ifdef _WIN32
            std::filesystem::remove(path, error);
            error.clear();
#endif
            std::filesystem::rename(
                temporary, path, error);
            if (error) {
                std::error_code ignored;
                std::filesystem::remove(
                    temporary, ignored);
                return fail<void>(
                    lfs::ErrorCode::Unavailable,
                    "Project lifecycle settings could not be published.",
                    error.message(), "settings.path");
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

    void rememberProject(
        ProjectLifecycleSettings& settings,
        const lfs::core::Uuid& project_uuid,
        const std::filesystem::path& path) {
        settings.mru.erase(
            std::remove_if(
                settings.mru.begin(), settings.mru.end(),
                [&](const ProjectMruEntry& entry) {
                    return entry.project_uuid ==
                               project_uuid ||
                           entry.last_known_path == path;
                }),
            settings.mru.end());
        settings.mru.insert(
            settings.mru.begin(),
            ProjectMruEntry{
                .project_uuid = project_uuid,
                .last_known_path = path,
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
          settings_path_(
              settings_path.value_or(
                  lfs::core::user_config_dir() /
                  "project_lifecycle.json")) {
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
        epoch_.fetch_add(1, std::memory_order_acq_rel);
        std::lock_guard lock(thread_mutex_);
        for (auto& thread : hydration_threads_) {
            thread.request_stop();
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
        return saveProjectLifecycleSettings(
            settings_path_, settings_);
    }

    lfs::Result<void>
    ProjectLifecycle::preflightSwitch(
        const ProjectSwitchDisposition disposition) {
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The current project is still being saved.",
                "Project switching is blocked until the close save finishes",
                "project.save");
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

    lfs::Result<void>
    ProjectLifecycle::setReopenLastProject(
        const bool enabled) {
        if (settings_.reopen_last_project == enabled) {
            return {};
        }
        const auto previous =
            settings_.reopen_last_project;
        settings_.reopen_last_project = enabled;
        if (auto saved = persistSettings(); !saved) {
            settings_.reopen_last_project = previous;
            return saved;
        }
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::setAutoSaveOnClose(
        const bool enabled) {
        if (settings_.auto_save_on_close == enabled) {
            return {};
        }
        const auto previous =
            settings_.auto_save_on_close;
        settings_.auto_save_on_close = enabled;
        if (auto saved = persistSettings(); !saved) {
            settings_.auto_save_on_close = previous;
            return saved;
        }
        return {};
    }

    void ProjectLifecycle::markSceneMutation(
        const std::uint32_t mutation_flags) {
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
        scene_dirty_.store(
            true, std::memory_order_release);
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
    ProjectLifecycle::adoptCompletedTrainingSnapshot() {
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
        adopted_training_snapshot_count_ =
            metrics.capture.completed_snapshots;
        hydration_.store(
            Hydration::Complete,
            std::memory_order_release);
        hydration_error_.clear();
        rememberProject(
            settings_, document_->project_uuid(),
            metrics.last_path);
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
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::synchronizeDocumentFromViewer() {
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
                scene, bindings);
        if (!captured_scene) {
            return lfs::Status::failure(
                std::move(captured_scene).error());
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
                    scene, selected);
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

            std::unordered_set<lfs::core::Uuid>
                live_geometry;
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
                live_geometry.insert(node->uuid);
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
                if (!live_geometry.contains(
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

        if (auto* parameters =
                viewer_.getParameterManager()) {
            auto snapshot =
                parameters
                    ->capturePendingProjectState();
            if (!snapshot) {
                return lfs::Status::failure(
                    std::move(snapshot).error());
            }
            lfs::io::project::ParametersChapter
                staged_parameters;
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
                viewer_.captureProjectSession();
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

        scene_dirty_.store(
            false, std::memory_order_release);
        payload_dirty_.store(
            false, std::memory_order_release);
        return {};
    }

    lfs::Result<std::vector<std::byte>>
    ProjectLifecycle::capturePreviewPng() const {
        auto captured =
            lfs::vis::capture_viewport_render();
        if (!captured || !captured->image) {
            const auto hydration =
                hydration_.load(
                    std::memory_order_acquire);
            if (hydration == Hydration::ShellReady ||
                hydration == Hydration::Hydrating) {
                // A Phase-A shell intentionally has no renderable payload.
                // Preserve the prior THMB just like every other clean span;
                // the explicit save must still be able to publish without
                // turning an unloaded chapter into an empty one.
                LOG_WARN(
                    "Explicit save during partial hydration is carrying the previous THMB forward");
                return std::vector<std::byte>{};
            }
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::Unavailable,
                "The viewport preview is not available yet.",
                "Render at least one viewport frame before saving",
                "THMB");
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
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            return lfs::Status::failure(
                std::move(adopted).error());
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
            viewer_.getTrainerManager()
                ->isTrainingActive()) {
            auto context =
                captureTrainingDocumentContext(
                    viewer_, *document_);
            if (!context) {
                return lfs::Status::failure(
                    std::move(context).error());
            }
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
            trainer->request_project_save(
                *document_->source_path(),
                std::move(preview),
                std::move(*context));
            return {};
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
        auto saved = document_->save(
            *document_->source_path(),
            ProjectDocumentSaveOptions{
                .commit =
                    {
                        .kind =
                            lfs::io::project::
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
            });
        if (!saved) {
            return lfs::Status::failure(
                std::move(saved).error());
        }
        rememberProject(
            settings_, document_->project_uuid(),
            *document_->source_path());
        if (auto persisted = persistSettings();
            !persisted) {
            LOG_WARN(
                "Project saved, but MRU settings failed: {}",
                developerError(persisted.error()));
        }
        LOG_INFO(
            "Saved .licht generation {} (rewritten={}, reused={}) to {}",
            saved->generation, saved->rewritten_chunks,
            saved->reused_chunks,
            lfs::core::path_to_utf8(
                *document_->source_path()));
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::saveAs(
        const std::filesystem::path& path,
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
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            return lfs::Status::failure(
                std::move(adopted).error());
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
            viewer_.getTrainerManager()
                ->isTrainingActive()) {
            auto context =
                captureTrainingDocumentContext(
                    viewer_, *document_);
            if (!context) {
                return lfs::Status::failure(
                    std::move(context).error());
            }
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
            trainer->request_project_save(
                *normalized,
                std::move(preview),
                std::move(*context));
            return {};
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
        auto saved = document_->save_as(
            *normalized,
            ProjectDocumentSaveOptions{
                .commit =
                    {
                        .kind =
                            lfs::io::project::
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
            });
        if (!saved) {
            return lfs::Status::failure(
                std::move(saved).error());
        }
        rememberProject(
            settings_, document_->project_uuid(),
            *normalized);
        if (auto persisted = persistSettings();
            !persisted) {
            LOG_WARN(
                "Project Save As succeeded, but MRU settings failed: {}",
                developerError(persisted.error()));
        }
        LOG_INFO(
            "Saved project as {} generation {}",
            lfs::core::path_to_utf8(*normalized),
            saved->generation);
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::open(
        const std::filesystem::path& path,
        const ProjectSwitchDisposition disposition) {
        if (auto preflight =
                preflightSwitch(disposition);
            !preflight) {
            return preflight;
        }
        const auto started =
            std::chrono::steady_clock::now();
        auto normalized =
            normalizedProjectPath(path);
        if (!normalized) {
            return lfs::Status::failure(
                std::move(normalized).error());
        }
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
                });
        if (!session) {
            return lfs::Status::failure(
                std::move(session).error());
        }
        auto parameters =
            candidate->parameters().snapshot();
        if (!parameters) {
            return lfs::Status::failure(
                std::move(parameters).error());
        }
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

        viewer_.resetProjectState();
        manager->getScene().commitRestoreStage(
            std::move(*shell));
        manager->changeContentType(
            inferContentType(
                manager->getScene()));
        parameter_manager
            ->installValidatedPendingProjectState(
                *parameters);
        viewer_.stagePreparedProjectSessionRestore(
            std::move(*session));

        document_ = candidate;
        adopted_training_snapshot_count_ = 0;
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        op::undoHistory().clear();
        const auto epoch =
            epoch_.fetch_add(
                1, std::memory_order_acq_rel) +
            1;
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
        rememberProject(
            settings_, candidate->project_uuid(),
            *normalized);
        if (auto persisted = persistSettings();
            !persisted) {
            LOG_WARN(
                "Project opened, but MRU settings failed: {}",
                developerError(persisted.error()));
        }
        if (auto launched =
                launchHydration(
                    candidate, epoch,
                    shell_selection_serial,
                    shell_selected_nodes);
            !launched) {
            markHydrationFailed(
                epoch,
                developerError(launched.error()));
        }
        const auto shell_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                started)
                .count();
        LOG_INFO(
            "Project shell ready in {:.3f} ms: {} (generation {}, {} payload units)",
            shell_ms,
            lfs::core::path_to_utf8(*normalized),
            candidate->generation(),
            candidate->payload_states().size());
        return {};
    }

    lfs::Result<void>
    ProjectLifecycle::launchHydration(
        std::shared_ptr<ProjectDocument> document,
        const std::uint64_t epoch,
        const std::uint64_t selection_mutation_serial,
        std::vector<lfs::core::Uuid>
            selected_node_uuids) {
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
        std::lock_guard lock(thread_mutex_);
        hydration_threads_.emplace_back(
            [this, document = std::move(document),
             source_path, project_uuid,
             minimum_generation,
             epoch, selection_mutation_serial,
             selected_node_uuids =
                 std::move(selected_node_uuids),
             allocator = std::move(allocator)](
                const std::stop_token stop) mutable {
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
                auto* scene_manager =
                    viewer_.getSceneManager();
                if (!scene_manager) {
                    return;
                }
                auto staged =
                    opened_source->stage_hydration(
                        scene_manager->getScene(), {},
                        std::move(allocator));
                if (!staged) {
                    const auto detail =
                        developerError(staged.error());
                    if (!stop.stop_requested()) {
                        markHydrationFailed(
                            epoch, detail);
                    }
                    return;
                }
                auto plan =
                    std::make_shared<
                        std::optional<
                            lfs::io::project::
                                ProjectHydrationPlan>>(
                        std::move(*staged));
                const bool posted =
                    viewer_.postWork({
                        .run =
                            [this, document, epoch,
                             selection_mutation_serial,
                             selected_node_uuids,
                             plan] {
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
                                    return;
                                }
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
                }
            });
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
                    hydration_.store(
                        Hydration::Failed,
                        std::memory_order_release);
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
        document_ =
            std::make_shared<ProjectDocument>(
                std::move(*created));
        adopted_training_snapshot_count_ = 0;
        close_save_state_.store(
            CloseSaveState::Idle,
            std::memory_order_release);
        viewer_.resetProjectState();
        epoch_.fetch_add(
            1, std::memory_order_acq_rel);
        hydration_.store(
            Hydration::Empty,
            std::memory_order_release);
        hydration_error_.clear();
        op::undoHistory().clear();
        scene_dirty_.store(
            false, std::memory_order_release);
        payload_dirty_.store(
            false, std::memory_order_release);
        return {};
    }

    bool ProjectLifecycle::hasDirtyProject() {
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return true;
        }
        if (!document_) {
            return false;
        }
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            LOG_ERROR(
                "Could not adopt the completed training project generation: {}",
                developerError(adopted.error()));
            return true;
        }
        if (viewer_.getTrainer() &&
            viewer_.getTrainerManager() &&
            viewer_.getTrainerManager()
                ->isTrainingActive()) {
            return true;
        }
        const auto* manager =
            viewer_.getSceneManager();
        const bool blank_untitled =
            !document_->source_path() &&
            hydration_.load(
                std::memory_order_acquire) ==
                Hydration::Empty &&
            manager &&
            manager->getScene().getNodes().empty() &&
            !scene_dirty_.load(
                std::memory_order_acquire) &&
            !payload_dirty_.load(
                std::memory_order_acquire);
        if (blank_untitled) {
            return false;
        }
        if (auto synchronized =
                synchronizeDocumentFromViewer();
            !synchronized) {
            LOG_ERROR(
                "Could not evaluate project dirty state: {}",
                developerError(synchronized.error()));
            return true;
        }
        return document_->dirty();
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
            return CloseSaveStatus::NotDirty;
        }
        if (!settings_.auto_save_on_close ||
            !document_ ||
            !document_->source_path()) {
            return CloseSaveStatus::NeedsPrompt;
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
            *document_->source_path();
        const ProjectDocumentSaveOptions options{
            .commit =
                {
                    .kind =
                        lfs::io::project::
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
        };
        close_save_state_.store(
            CloseSaveState::Saving,
            std::memory_order_release);
        {
            std::lock_guard lock(
                close_save_mutex_);
            close_save_error_.clear();
        }
        try {
            close_save_thread_ =
                std::jthread(
                    [this, document, path,
                     options](
                        const std::stop_token stop) {
                        if (stop.stop_requested()) {
                            return;
                        }
                        auto saved =
                            document->save(
                                path, options);
                        const bool succeeded =
                            static_cast<bool>(
                                saved);
                        const auto generation =
                            succeeded
                                ? saved->generation
                                : 0;
                        std::string error;
                        if (!succeeded) {
                            error = developerError(
                                saved.error());
                        }
                        const bool posted =
                            viewer_.postWork({
                                .run =
                                    [this, document,
                                     path, succeeded,
                                     generation,
                                     error =
                                         std::move(
                                             error)] {
                                        if (document_ !=
                                            document) {
                                            {
                                                std::lock_guard lock(
                                                    close_save_mutex_);
                                                close_save_error_ =
                                                    "The active project changed while the close save was running.";
                                            }
                                            close_save_state_.store(
                                                CloseSaveState::Failed,
                                                std::memory_order_release);
                                        } else if (
                                            succeeded) {
                                            rememberProject(
                                                settings_,
                                                document->project_uuid(),
                                                path);
                                            if (auto persisted =
                                                    persistSettings();
                                                !persisted) {
                                                LOG_WARN(
                                                    "Close save succeeded, but MRU settings failed: {}",
                                                    developerError(
                                                        persisted.error()));
                                            }
                                            close_save_state_.store(
                                                CloseSaveState::Succeeded,
                                                std::memory_order_release);
                                            LOG_INFO(
                                                "Save-on-close published .licht generation {}",
                                                generation);
                                        } else {
                                            {
                                                std::lock_guard lock(
                                                    close_save_mutex_);
                                                close_save_error_ =
                                                    error;
                                            }
                                            close_save_state_.store(
                                                CloseSaveState::Failed,
                                                std::memory_order_release);
                                            LOG_ERROR(
                                                "Save-on-close failed: {}",
                                                error);
                                        }
                                        viewer_
                                            .requestApplicationClose();
                                    },
                                .cancel =
                                    [this] {
                                        std::lock_guard lock(
                                            close_save_mutex_);
                                        close_save_error_ =
                                            "The save completion was cancelled during shutdown.";
                                        close_save_state_.store(
                                            CloseSaveState::Failed,
                                            std::memory_order_release);
                                    },
                            });
                        if (!posted) {
                            std::lock_guard lock(
                                close_save_mutex_);
                            close_save_error_ =
                                "The application could not publish the save completion.";
                            close_save_state_.store(
                                CloseSaveState::Failed,
                                std::memory_order_release);
                        }
                    });
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): thread construction failure is published through close-save state.
            {
                std::lock_guard lock(
                    close_save_mutex_);
                close_save_error_ =
                    error.what();
            }
            close_save_state_.store(
                CloseSaveState::Failed,
                std::memory_order_release);
            return CloseSaveStatus::Failed;
        }
        return CloseSaveStatus::Saving;
    }

    void ProjectLifecycle::resetCloseSaveAttempt() {
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

    lfs::Result<ProjectInfo>
    ProjectLifecycle::info() {
        if (close_save_state_.load(
                std::memory_order_acquire) ==
            CloseSaveState::Saving) {
            return fail<ProjectInfo>(
                lfs::ErrorCode::FailedPrecondition,
                "The project is being saved before exit.",
                "Project metadata is unavailable while the close save owns the document",
                "project.save");
        }
        if (!document_) {
            return fail<ProjectInfo>(
                lfs::ErrorCode::FailedPrecondition,
                "There is no active project document.",
                "Project lifecycle has not created or opened a document",
                "project.document");
        }
        if (auto adopted =
                adoptCompletedTrainingSnapshot();
            !adopted) {
            return std::move(adopted).error();
        }
        const auto* manager =
            viewer_.getSceneManager();
        const bool blank_untitled =
            !document_->source_path() &&
            hydration_.load(
                std::memory_order_acquire) ==
                Hydration::Empty &&
            manager &&
            manager->getScene().getNodes().empty() &&
            !scene_dirty_.load(
                std::memory_order_acquire) &&
            !payload_dirty_.load(
                std::memory_order_acquire);
        if (!blank_untitled) {
            if (auto synchronized =
                    synchronizeDocumentFromViewer();
                !synchronized) {
                return std::move(synchronized).error();
            }
        }
        ProjectInfo result{
            .path = document_->source_path(),
            .project_uuid =
                document_->project_uuid().to_string(),
            .generation = document_->generation(),
            .dirty =
                !blank_untitled &&
                document_->dirty(),
            .dirty_chapters =
                blank_untitled
                    ? std::vector<std::string>{}
                    : document_->dirty_chapters(),
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
        return result;
    }

    void ProjectLifecycle::openStartupProject(
        const std::optional<
            std::filesystem::path>& explicit_path) {
        if (explicit_path) {
            if (auto opened = open(*explicit_path);
                !opened) {
                LOG_ERROR(
                    "Failed to open startup project {}: {}",
                    lfs::core::path_to_utf8(
                        *explicit_path),
                    developerError(
                        opened.error()));
            }
            return;
        }
        if (!settings_.reopen_last_project) {
            return;
        }
        for (const auto& entry : settings_.mru) {
            std::error_code error;
            if (!std::filesystem::is_regular_file(
                    entry.last_known_path, error) ||
                error) {
                continue;
            }
            if (auto opened =
                    open(entry.last_known_path);
                !opened) {
                LOG_WARN(
                    "Could not restore last project {}: {}",
                    lfs::core::path_to_utf8(
                        entry.last_known_path),
                    developerError(
                        opened.error()));
            }
            return;
        }
    }

} // namespace lfs::vis::project
