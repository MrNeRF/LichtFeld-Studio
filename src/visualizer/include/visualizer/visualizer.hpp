/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace lfs::core {
    class Scene;
}

namespace lfs::core::param {
    struct TrainingParameters;
}

namespace lfs::vis {
    class SceneManager;
    class RenderingManager;

    struct LFS_VIS_API ViewerOptions {
        std::string title = "LichtFeld Studio";
        int width = 1280;
        int height = 720;
        bool antialiasing = false;
        bool enable_cuda_interop = true;
        bool gut = false;
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
        virtual void clearScene() = 0;
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
