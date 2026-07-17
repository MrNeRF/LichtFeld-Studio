/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/editor_context.hpp"
#include "core/error_codes.hpp"
#include "core/frame_state_machine.hpp"
#include "core/main_loop.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "ipc/view_context.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_base.hpp"
#include "training/training_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

struct SDL_Window;

namespace lfs::python {
    struct SequencerUIStateData;
} // namespace lfs::python

namespace lfs::vis {
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis {
    class DataLoadingService;

    namespace tools {
        class AlignTool;
        class SelectionTool;
    } // namespace tools

    class LFS_VIS_API VisualizerImpl : public Visualizer {
        friend class gui::GuiManager;

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
        [[nodiscard]] std::expected<void, std::string> clearScene() override;
        core::Scene& getScene() override { return scene_manager_->getScene(); }
        bool postWork(WorkItem work) override;
        bool postRenderWork(WorkItem work);
        [[nodiscard]] bool isOnViewerThread() const override {
            return std::this_thread::get_id() == viewer_thread_id_;
        }
        [[nodiscard]] bool acceptsPostedWork() const override;
        [[nodiscard]] bool isProcessingRenderWork() const {
            assert(isOnViewerThread());
            return processing_render_work_;
        }
        void setShutdownRequestedCallback(std::function<void()> callback) override;
        std::expected<void, std::string> startTraining() override;
        std::expected<std::filesystem::path, std::string> saveCheckpoint(
            const std::optional<std::filesystem::path>& path = std::nullopt) override;

        // Getters for GUI (delegating to state manager)
        lfs::training::Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }

        // Component access
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }
        SceneManager* getSceneManager() override { return scene_manager_.get(); }
        SDL_Window* getWindow() const { return window_manager_->getWindow(); }
        WindowManager* getWindowManager() { return window_manager_.get(); }
        RenderingManager* getRenderingManager() override { return rendering_manager_.get(); }
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
        friend class VisualizerImplResetTest_ResetTrainingPreservesExplicitInitPath_Test;

        // Allow ToolContext to access GUI manager for logging
        friend class ToolContext;

    private:
        // Main loop callbacks
        bool initialize();
        void update();
        void render();
        void shutdown();
        bool allowclose();
        void wakeMainLoop() const;

        // Frame exception boundary. Contains an OOM or other error escaping a
        // frame so the loop never aborts: OOM triggers one render-safe pressure
        // episode, other errors escalate under a rate limit.
        void handleFrameException(std::exception_ptr eptr) noexcept;
        void onFrameCompleted() noexcept;
        // Executes the FrameStateMachine's publish effects on the UI thread
        // (pressure toast, OOM-paused modal, renderer-dead modal + OS dialog).
        void applyFrameStateEffects(const FrameStateMachine::Effects& effects) noexcept;
        // Internal-terminal modal carrying the caught fault's code and developer
        // detail; renderer-dead modal (+ AMB-P3-1 OS dialog) for a lost/stalled
        // renderer that can no longer present its own obituary.
        void publishRendererInternalModal(lfs::ErrorCode code, std::string detail) noexcept;
        void publishRendererInternalModal(const lfs::Error& error) noexcept;
        std::vector<lfs::ErrorAction> rendererInternalActions();
        void publishRendererDeadModal(RendererTerminalState cause) noexcept;
        void logRateLimitedFrameError(const std::exception& e) noexcept;

        // Event system
        void setupEventHandlers();
        void setupComponentConnections();
        void handleTrainingCompleted(const lfs::core::events::state::TrainingCompleted& event);
        void handleLoadFileCommand(const lfs::core::events::cmd::LoadFile& cmd);
        void handleLoadConfigFile(const std::filesystem::path& path);
        void handleNewProject();
        void performNewProject();
        void schedulePendingTrainingAction();
        void performPendingTrainingAction();
        void requestApplicationClose();
        void handleSwitchToLatestCheckpoint();
        void performReset();
        void resetProjectState();

        // Tool initialization
        void initializeTools();

        // Subsystem wiring
        void setupPythonBridge();
        void setupViewContextBridge();
        void beginShutdown(std::string_view reason = "Viewer is shutting down");
        void processRenderWorkQueue();
        [[nodiscard]] bool hasPendingRenderWork();
        [[nodiscard]] bool inputFrameRequestsRender() const;

        struct FrameDemand {
            bool viewport_export_locked = false;
            bool scene_dirty = false;
            bool continuous_input = false;
            bool python_animation = false;
            bool python_overlay = false;
            bool python_redraw = false;
            bool gui_animation = false;
            bool input_event = false;
            bool posted_work = false;
            bool render_work = false;
            bool store_dirty = false;
            bool swapchain_resize_pending = false;
            bool swapchain_resize_ready = false;
            bool window_resize_paint_pending = false;
            bool viewport_resize_deferring = false;
            bool viewport_resize_settle_ready = false;

            [[nodiscard]] bool shouldRenderFrame() const {
                return viewport_export_locked || scene_dirty || continuous_input ||
                       python_animation || python_overlay || python_redraw ||
                       gui_animation || input_event || posted_work || render_work ||
                       store_dirty || swapchain_resize_ready || window_resize_paint_pending ||
                       viewport_resize_settle_ready;
            }

            [[nodiscard]] bool needsContinuousLoop() const {
                const bool resize_deferral_throttles_animation =
                    viewport_resize_deferring ||
                    (swapchain_resize_pending && !swapchain_resize_ready);
                return scene_dirty || continuous_input || python_animation ||
                       python_overlay || python_redraw ||
                       (gui_animation && !resize_deferral_throttles_animation) ||
                       render_work || viewport_export_locked || store_dirty ||
                       swapchain_resize_ready || window_resize_paint_pending ||
                       viewport_resize_settle_ready;
            }
        };

        [[nodiscard]] FrameDemand collectFrameDemand(bool viewport_export_locked,
                                                     bool drained_store_dirty = false);
        void waitForNextEvent(bool is_training);

        class CallbackCleanup {
            std::vector<std::function<void()>> cleanups_;

        public:
            void add(std::function<void()> fn) { cleanups_.push_back(std::move(fn)); }
            void clear() {
                for (auto it = cleanups_.rbegin(); it != cleanups_.rend(); ++it)
                    (*it)();
                cleanups_.clear();
            }
            ~CallbackCleanup() { clear(); }
            CallbackCleanup() = default;
            CallbackCleanup(const CallbackCleanup&) = delete;
            CallbackCleanup& operator=(const CallbackCleanup&) = delete;
        };

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

        // Frame exception boundary state (viewer thread only).
        FrameStateMachine frame_state_;
        uint64_t suppressed_frame_errors_ = 0;
        std::chrono::steady_clock::time_point last_frame_error_log_{};

        // Tools
        std::shared_ptr<tools::AlignTool> align_tool_;
        std::shared_ptr<tools::SelectionTool> selection_tool_;
        std::unique_ptr<ToolContext> tool_context_;

        // Centralized editor state
        EditorContext editor_context_;

        mutable std::mutex work_queue_mutex_;
        std::vector<WorkItem> work_queue_;
        std::vector<WorkItem> render_work_queue_;
        std::thread::id viewer_thread_id_;
        bool accepting_work_ = true;
        bool shutdown_started_ = false;
        bool processing_render_work_ = false;

        std::mutex shutdown_callback_mutex_;
        std::function<void()> shutdown_requested_callback_;

        CallbackCleanup callback_cleanup_;

        // State tracking
        bool fully_initialized_ = false;
        bool window_initialized_ = false;
        bool gui_initialized_ = false;
        bool tools_initialized_ = false;
        bool view_context_bridge_initialized_ = false;
        bool pending_auto_train_ = false;
        enum class PendingTrainingAction : std::uint8_t {
            None,
            Reset,
            NewProject,
            Close,
        };
        PendingTrainingAction pending_training_action_ = PendingTrainingAction::None;
        bool pending_training_action_posted_ = false;
        int pending_training_completion_refresh_frames_ = 0;
        bool gui_frame_rendered_ = false;
        bool startup_plugin_preload_started_ = false;
        std::uint64_t startup_plugin_load_status_revision_ = 0;
        bool plugin_preload_timing_active_ = false;
        std::chrono::nanoseconds plugin_preload_max_update_stall_{};
        bool update_work_processed_ = false;
        std::chrono::high_resolution_clock::time_point last_frame_time_ = std::chrono::high_resolution_clock::now();
        bool sequencer_ui_initialized_ = false;
        std::unique_ptr<python::SequencerUIStateData> sequencer_ui_state_;
        std::vector<std::filesystem::path> pending_view_paths_;
        std::filesystem::path pending_dataset_path_;
    };

} // namespace lfs::vis
