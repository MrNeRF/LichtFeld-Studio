/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/editor_context.hpp"
#include "core/main_loop.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "ipc/selection_server.hpp"
#include "ipc/view_context.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_base.hpp"
#include "training/training_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

struct SDL_Window;

namespace lfs::vis {
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis {
    class DataLoadingService;

    namespace tools {
        class BrushTool;
        class AlignTool;
        class SelectionTool;
    } // namespace tools

    class VisualizerImpl : public Visualizer {
    public:
        explicit VisualizerImpl(const ViewerOptions& options);
        ~VisualizerImpl() override;

        void run() override;
        void setParameters(const lfs::core::param::TrainingParameters& params) override;
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path) override;
        std::expected<void, std::string> addSplatFile(const std::filesystem::path& path) override;
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path) override;
        std::expected<void, std::string> loadCheckpointForTraining(const std::filesystem::path& path) override;
        void consolidateModels() override;
        void clearScene() override;

        // Getters for GUI (delegating to state manager)
        lfs::training::Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }

        // Component access
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }
        SceneManager* getSceneManager() { return scene_manager_.get(); }
        SDL_Window* getWindow() const { return window_manager_->getWindow(); }
        WindowManager* getWindowManager() { return window_manager_.get(); }
        RenderingManager* getRenderingManager() { return rendering_manager_.get(); }
        gui::GuiManager* getGuiManager() { return gui_manager_.get(); }
        const Viewport& getViewport() const { return viewport_; }
        Viewport& getViewport() { return viewport_; }

        // FPS monitoring
        [[nodiscard]] float getCurrentFPS() const {
            return rendering_manager_ ? rendering_manager_->getCurrentFPS() : 0.0f;
        }

        [[nodiscard]] float getAverageFPS() const {
            return rendering_manager_ ? rendering_manager_->getAverageFPS() : 0.0f;
        }

        // Antialiasing state
        bool isAntiAliasingEnabled() const {
            return rendering_manager_ ? rendering_manager_->getSettings().antialiasing : false;
        }

        tools::BrushTool* getBrushTool() {
            return brush_tool_.get();
        }

        const tools::BrushTool* getBrushTool() const {
            return brush_tool_.get();
        }

        tools::AlignTool* getAlignTool() {
            return align_tool_.get();
        }

        const tools::AlignTool* getAlignTool() const {
            return align_tool_.get();
        }

        tools::SelectionTool* getSelectionTool() {
            return selection_tool_.get();
        }

        const tools::SelectionTool* getSelectionTool() const {
            return selection_tool_.get();
        }

        InputController* getInputController() {
            return input_controller_.get();
        }

        DataLoadingService* getDataLoader() {
            return data_loader_.get();
        }

        EditorContext& getEditorContext() { return editor_context_; }
        const EditorContext& getEditorContext() const { return editor_context_; }

        // Undo/Redo
        void undo();
        void redo();

        // GUI manager
        std::unique_ptr<gui::GuiManager> gui_manager_;
        friend class gui::GuiManager;

        // Allow ToolContext to access GUI manager for logging
        friend class ToolContext;

    private:
        // Main loop callbacks
        bool initialize();
        void update();
        void render();
        void shutdown();
        bool allowclose();

        // Event system
        void setupEventHandlers();
        void setupComponentConnections();
        void handleTrainingCompleted(const lfs::core::events::state::TrainingCompleted& event);
        void handleLoadFileCommand(const lfs::core::events::cmd::LoadFile& cmd);
        void handleLoadConfigFile(const std::filesystem::path& path);
        void handleSwitchToLatestCheckpoint();
        void performReset();

        // Tool initialization
        void initializeTools();

        // Plugin capability invocation (runs on main thread with scene context)
        [[nodiscard]] CapabilityInvokeResult processCapabilityRequest(const std::string& name, const std::string& args);

        // Options
        ViewerOptions options_;

        // Core components
        Viewport viewport_;
        std::unique_ptr<WindowManager> window_manager_;
        std::unique_ptr<InputController> input_controller_;
        std::unique_ptr<RenderingManager> rendering_manager_;
        std::unique_ptr<SceneManager> scene_manager_;
        std::shared_ptr<TrainerManager> trainer_manager_;
        std::unique_ptr<DataLoadingService> data_loader_;
        std::unique_ptr<ParameterManager> parameter_manager_;
        std::unique_ptr<MainLoop> main_loop_;

        // Tools
        std::shared_ptr<tools::BrushTool> brush_tool_;
        std::shared_ptr<tools::AlignTool> align_tool_;
        std::shared_ptr<tools::SelectionTool> selection_tool_;
        std::unique_ptr<ToolContext> tool_context_;

        // Centralized editor state
        EditorContext editor_context_;

        // IPC for MCP selection commands
        std::unique_ptr<SelectionServer> selection_server_;

        // Capability request synchronization (IPC thread waits for main thread to process)
        struct CapabilityRequest {
            std::string name;
            std::string args;
            CapabilityInvokeResult* result = nullptr;
            std::mutex* mtx = nullptr;
            std::condition_variable* cv = nullptr;
            bool* done = nullptr;
        };
        std::optional<CapabilityRequest> pending_capability_request_;
        std::mutex capability_request_mutex_;

        // State tracking
        bool window_initialized_ = false;
        bool gui_initialized_ = false;
        bool tools_initialized_ = false;
        bool pending_auto_train_ = false;
        bool pending_reset_ = false;
        bool gui_frame_rendered_ = false;
        std::vector<std::filesystem::path> pending_view_paths_;
        std::filesystem::path pending_dataset_path_;
    };

} // namespace lfs::vis
