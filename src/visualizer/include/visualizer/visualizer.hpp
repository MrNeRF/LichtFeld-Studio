/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

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

    struct RuntimeServiceStatus {
        bool enabled = false;
        bool running = false;
        bool network_exposed = false;
        int port = 0;
        std::uint64_t request_count = 0;
        std::uint64_t success_count = 0;
        std::uint64_t error_count = 0;
        std::vector<std::string> endpoints;
        bool request_logging = false;
        std::string log_file;
        std::string error;
    };

    struct RuntimeServiceControls {
        std::function<bool()> toggle_mcp_enabled;
        std::function<bool()> toggle_mcp_binding;
    };

    LFS_VIS_API void setRuntimeServiceControls(RuntimeServiceControls controls);
    LFS_VIS_API bool toggleMcpRuntimeEnabled();
    LFS_VIS_API bool toggleMcpRuntimeBinding();
    [[nodiscard]] LFS_VIS_API std::uint64_t runtimeServiceRevision();

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
        virtual std::expected<std::filesystem::path, std::string> saveCheckpoint(
            const std::optional<std::filesystem::path>& path = std::nullopt) = 0;

        virtual ~Visualizer() = default;
    };

} // namespace lfs::vis
