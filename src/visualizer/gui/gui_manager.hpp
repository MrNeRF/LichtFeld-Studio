/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/cuda_version.hpp"
#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "gui/async_task_manager.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/menu_bar.hpp"
#include "gui/rml_menu_bar.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "gui/rml_right_panel.hpp"
#include "gui/rml_shell_frame.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "gui/startup_overlay.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/drag_drop_native.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <imgui.h>

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {
        class LFS_VIS_API GuiManager {
        public:
            GuiManager(VisualizerImpl* viewer);
            ~GuiManager();

            // Lifecycle
            void init();
            void shutdown();
            void render();
            void setRmlResizeDeferring(bool defer) { rmlui_manager_.setResizeDeferring(defer); }

            // Sub-manager access
            [[nodiscard]] AsyncTaskManager& asyncTasks() { return async_tasks_; }
            [[nodiscard]] const AsyncTaskManager& asyncTasks() const { return async_tasks_; }
            [[nodiscard]] GizmoManager& gizmo() { return gizmo_manager_; }
            [[nodiscard]] const GizmoManager& gizmo() const { return gizmo_manager_; }
            [[nodiscard]] PanelLayoutManager& panelLayout() { return panel_layout_; }
            [[nodiscard]] const PanelLayoutManager& panelLayout() const { return panel_layout_; }
            [[nodiscard]] GlobalContextMenu& globalContextMenu() { return *global_context_menu_; }

            // State queries
            bool needsAnimationFrame() const;

            // Window visibility
            void showWindow(const std::string& name, bool show = true);

            // Viewport region access
            glm::vec2 getViewportPos() const;
            glm::vec2 getViewportSize() const;
            bool isViewportFocused() const;
            bool isPositionInViewport(double x, double y) const;
            bool isPositionOverFloatingPanel(double x, double y) const;

            bool isForceExit() const { return force_exit_; }
            void setForceExit(bool value) { force_exit_ = value; }

            [[nodiscard]] SequencerController& sequencer() { return sequencer_ui_.controller(); }
            [[nodiscard]] const SequencerController& sequencer() const { return sequencer_ui_.controller(); }

            [[nodiscard]] panels::SequencerUIState& getSequencerUIState() { return sequencer_ui_state_; }
            [[nodiscard]] const panels::SequencerUIState& getSequencerUIState() const { return sequencer_ui_state_; }

            [[nodiscard]] VisualizerImpl* getViewer() const { return viewer_; }
            [[nodiscard]] std::unordered_map<std::string, bool>* getWindowStates() { return &window_states_; }

            void requestExitConfirmation();
            bool isExitConfirmationPending() const;

            bool isCapturingInput() const;
            bool isModalWindowOpen() const;
            [[nodiscard]] bool isStartupVisible() const { return startup_overlay_.isVisible(); }
            void dismissStartupOverlay();
            void captureKey(int key, int mods);
            void captureMouseButton(int button, int mods);

            // Thumbnail system (delegates to MenuBar)
            void requestThumbnail(const std::string& video_id);
            void processThumbnails();
            bool isThumbnailReady(const std::string& video_id) const;
            uint64_t getThumbnailTexture(const std::string& video_id) const;

            int getHighlightedCameraUid() const;

            // Drag-drop state for overlays
            [[nodiscard]] bool isDragHovering() const { return drag_drop_hovering_; }

            // Used by native panel wrappers
            void renderSelectionOverlays(const UIContext& ctx);
            void renderViewportDecorations();

        private:
            void setupEventHandlers();
            void checkCudaVersionAndNotify();
            void applyDefaultStyle();
            void initMenuBar();
            void registerNativePanels();
            void updateInputOverrides(const PanelInputState& input, bool mouse_in_viewport);
            void applyUiScale(float scale);
            void rebuildFonts(float scale);
            void loadImGuiSettings();
            void saveImGuiSettings() const;
            void persistImGuiSettingsIfNeeded();

            // Core dependencies
            VisualizerImpl* viewer_;

            // Owned components
            std::unique_ptr<RmlModalOverlay> rml_modal_overlay_;
            std::unique_ptr<lfs::gui::IVideoExtractorWidget> video_widget_;

            // UI state only
            std::unordered_map<std::string, bool> window_states_;
            bool show_main_panel_ = true;

            // Panel layout and viewport
            PanelLayoutManager panel_layout_;
            ViewportLayout viewport_layout_;
            bool force_exit_ = false;

            std::unique_ptr<MenuBar> menu_bar_;

            panels::SequencerUIState sequencer_ui_state_;
            SequencerUIManager sequencer_ui_;
            GizmoManager gizmo_manager_;

            std::string focus_panel_name_;
            bool ui_hidden_ = false;

            // Font storage
            ImFont* font_regular_ = nullptr;
            ImFont* font_bold_ = nullptr;
            ImFont* font_heading_ = nullptr;
            ImFont* font_small_ = nullptr;
            ImFont* font_section_ = nullptr;
            ImFont* font_monospace_ = nullptr;
            ImFont* mono_fonts_[FontSet::MONO_SIZE_COUNT] = {};
            float mono_font_scales_[FontSet::MONO_SIZE_COUNT] = {};
            std::filesystem::path imgui_ini_path_;
            FontSet buildFontSet() const;

            // Async task management
            AsyncTaskManager async_tasks_;

            StartupOverlay startup_overlay_;
            RmlShellFrame rml_shell_frame_;
            RmlRightPanel rml_right_panel_;
            RmlViewportOverlay rml_viewport_overlay_;
            RmlMenuBar rml_menu_bar_;
            RmlStatusBar rml_status_bar_;
            std::unique_ptr<GlobalContextMenu> global_context_menu_;

            // Native drag-drop handler
            NativeDragDrop drag_drop_;
            bool drag_drop_hovering_ = false;

            // DPI scaling
            float current_ui_scale_ = 1.0f;
            float pending_ui_scale_ = 0.0f;

            // Deferred CUDA version warning (emitted on first drawFrame)
            std::optional<lfs::core::CudaVersionInfo> pending_cuda_warning_;

            // File association prompt (Windows only, one-shot)
            bool file_association_checked_ = false;
            void promptFileAssociation();

            // RmlUI integration
            RmlUIManager rmlui_manager_;

            // Native panel wrapper storage (registered with PanelRegistry)
            std::vector<std::shared_ptr<IPanel>> native_panel_storage_;
            uint64_t panel_frame_serial_ = 0;
            uint8_t ui_layout_settle_frames_ = 0;
            glm::vec2 last_ui_layout_work_pos_{-1.0f, -1.0f};
            glm::vec2 last_ui_layout_work_size_{-1.0f, -1.0f};
            float last_ui_layout_right_panel_w_ = -1.0f;
            float last_ui_layout_scene_ratio_ = -1.0f;
            float last_ui_layout_python_console_w_ = -1.0f;
            bool last_ui_layout_show_main_panel_ = false;
            bool last_ui_layout_ui_hidden_ = false;
            bool last_ui_layout_python_console_visible_ = false;
            std::string last_ui_layout_active_tab_;
        };
    } // namespace gui
} // namespace lfs::vis
