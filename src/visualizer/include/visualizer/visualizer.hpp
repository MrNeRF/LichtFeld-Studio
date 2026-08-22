/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {
    class Scene;
}

namespace lfs::core::param {
    struct TrainingParameters;
}

namespace lfs::vis {
    class SceneManager;
    class RenderingManager;

    enum class GraphicsBackend {
        Vulkan,
    };

    enum class ProjectSwitchDisposition {
        RequireClean,
        DiscardChanges,
    };

    enum class ProjectOpenOutcome {
        Opened,
        RecoveryPromptPending,
    };

    enum class RuntimeServicePhase : std::uint8_t {
        Disabled,
        Starting,
        Running,
        Stopping,
        Failed,
    };

    enum class RuntimeServiceErrorKind : std::uint8_t {
        None,
        InvalidPort,
        BindFailed,
        RuntimeFailure,
    };

    struct RuntimeServiceStatus {
        bool enabled = false;
        bool running = false;
        RuntimeServicePhase phase = RuntimeServicePhase::Disabled;
        bool network_exposed = false;
        int port = 0;
        std::uint64_t request_count = 0;
        std::uint64_t success_count = 0;
        std::uint64_t error_count = 0;
        std::vector<std::string> endpoints;
        bool request_logging = false;
        std::string log_file;
        std::string error;
        RuntimeServiceErrorKind error_kind = RuntimeServiceErrorKind::None;
        std::string error_address;
        int error_port = 0;
    };

    struct RuntimeServiceControls {
        std::function<bool()> toggle_mcp_enabled;
        std::function<bool()> toggle_mcp_binding;
    };

    LFS_VIS_API void setRuntimeServiceControls(RuntimeServiceControls controls);
    LFS_VIS_API bool toggleMcpRuntimeEnabled();
    LFS_VIS_API bool toggleMcpRuntimeBinding();
    [[nodiscard]] LFS_VIS_API std::uint64_t runtimeServiceRevision();

    struct LFS_VIS_API ProjectPayloadInfo {
        std::string chapter;
        std::string node_uuid;
        std::string hydration_state;
    };

    struct LFS_VIS_API ProjectRecentInfo {
        std::string project_uuid;
        std::filesystem::path last_known_path;
    };

    struct LFS_VIS_API ProjectMenuInfo {
        bool reopen_last_project = true;
        bool auto_save_on_close = false;
        std::uint64_t autosave_interval_seconds = 5 * 60;
        std::vector<ProjectRecentInfo> recent_projects;
    };

    struct LFS_VIS_API ProjectWritePoll {
        bool running = false;
        std::uint64_t generation = 0;
        std::optional<std::filesystem::path> path;
        std::string error;
        std::optional<lfs::ErrorCode> error_code;
    };

    struct LFS_VIS_API ProjectInfo {
        std::optional<std::filesystem::path> path;
        std::string project_uuid;
        std::uint64_t generation = 0;
        bool dirty = false;
        bool session_dirty = false;
        std::vector<std::string> dirty_chapters;
        std::string hydration_state = "empty";
        std::vector<ProjectPayloadInfo> payloads;
        bool contains_embedded_secrets = false;
        bool reopen_last_project = true;
        bool auto_save_on_close = false;
        std::uint64_t autosave_interval_seconds =
            5 * 60;
        std::uint64_t autosave_dirty_epoch_threshold =
            20;
        bool project_write_running = false;
        std::string project_write_stage;
        float project_write_progress = 0.0F;
        std::string project_write_error;
        std::optional<lfs::ErrorCode> project_write_error_code;
        std::uint64_t autosave_sequence = 0;
        bool recovery_session = false;
        bool compaction_suggested = false;
        std::uint64_t physical_bytes = 0;
        std::uint64_t estimated_live_bytes = 0;
        std::uint64_t dead_bytes = 0;
        double dead_ratio = 0.0;
        std::string hydration_error;
        std::vector<ProjectRecentInfo> recent_projects;
    };

    struct LFS_VIS_API ViewerOptions {
        std::string title = "LichtFeld Studio";
        int width = 1280;
        int height = 720;
        bool antialiasing = false;
        bool show_startup_overlay = true;
        bool safe_mode = false;
        std::function<RuntimeServiceStatus()> mcp_status_provider;
        bool gut = false;
        GraphicsBackend graphics_backend = GraphicsBackend::Vulkan;
        int monitor_x = 0; // Monitor hint for window placement
        int monitor_y = 0;
        int monitor_width = 0;
        int monitor_height = 0;
        std::optional<std::filesystem::path> startup_project;
        // Embedders and tests may isolate lifecycle settings from the
        // platform user-config directory.
        std::optional<std::filesystem::path>
            project_lifecycle_settings_path;
    };

    class LFS_VIS_API Visualizer {
    public:
        struct WorkItem {
            std::function<void()> run;
            std::function<void()> cancel;
        };

        static std::unique_ptr<Visualizer> create(const ViewerOptions& options = {});

        virtual void run() = 0;
        virtual void setParameters(const lfs::core::param::TrainingParameters& params) = 0;
        virtual std::expected<void, std::string> loadPLY(const std::filesystem::path& path) = 0;
        virtual std::expected<void, std::string> addSplatFile(const std::filesystem::path& path) = 0;
        virtual std::expected<void, std::string> loadDataset(const std::filesystem::path& path) = 0;
        virtual std::expected<void, std::string> loadCheckpointForTraining(const std::filesystem::path& path) = 0;
        virtual void consolidateModels() = 0;
        [[nodiscard]] virtual std::expected<void, std::string> clearScene() = 0;
        virtual core::Scene& getScene() = 0;
        virtual SceneManager* getSceneManager() = 0;
        virtual RenderingManager* getRenderingManager() = 0;

        virtual bool postWork(WorkItem work) = 0;
        [[nodiscard]] virtual bool isOnViewerThread() const { return false; }
        [[nodiscard]] virtual bool acceptsPostedWork() const { return true; }
        virtual void setShutdownRequestedCallback(std::function<void()> callback) = 0;
        virtual std::expected<void, std::string> startTraining() = 0;
        [[nodiscard]] virtual std::optional<int>
        trainingStartOverwriteConflict() {
            return std::nullopt;
        }
        virtual lfs::Result<void>
        projectSave(bool regenerate_preview = true) = 0;
        virtual lfs::Result<void>
        projectSaveAs(const std::filesystem::path& path,
                      bool regenerate_preview = true) = 0;
        virtual lfs::Result<ProjectOpenOutcome>
        projectOpen(
            const std::filesystem::path& path,
            ProjectSwitchDisposition disposition =
                ProjectSwitchDisposition::RequireClean) = 0;
        virtual lfs::Result<void>
        projectCompact() = 0;
        virtual lfs::Result<bool>
        projectIsDirty() = 0;
        virtual lfs::Result<bool>
        projectHasPath() = 0;
        virtual lfs::Result<ProjectInfo>
        projectGetInfo() = 0;
        virtual lfs::Result<ProjectWritePoll>
        projectPollWrite() {
            return ProjectWritePoll{};
        }
        // True when the last ProjectSave or ProjectSaveAs command
        // started a write. A cancelled save dialog returns false.
        [[nodiscard]] virtual bool consumeProjectSaveStarted() {
            return false;
        }
        virtual void projectWaitWrite() {}
        virtual lfs::Result<ProjectMenuInfo>
        projectGetMenuInfo() {
            auto info = projectGetInfo();
            if (!info) {
                return std::move(info).error();
            }
            return ProjectMenuInfo{
                .reopen_last_project =
                    info->reopen_last_project,
                .auto_save_on_close =
                    info->auto_save_on_close,
                .autosave_interval_seconds =
                    info->autosave_interval_seconds,
                .recent_projects =
                    std::move(
                        info->recent_projects),
            };
        }
        virtual lfs::Result<void>
        projectClearRecentFiles() {
            return {};
        }
        virtual lfs::Result<void>
        projectRemoveRecentFile(
            const std::filesystem::path&) {
            return {};
        }

        [[nodiscard]] virtual bool loadFileWipeWouldNeedConfirmation(
            bool is_dataset,
            bool replace,
            bool discard_changes) {
            (void)is_dataset;
            (void)replace;
            (void)discard_changes;
            return false;
        }

        virtual ~Visualizer() = default;
    };

} // namespace lfs::vis
