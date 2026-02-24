/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "gui/async_task_manager.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/menu_bar.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "gui/startup_overlay.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/drag_drop_native.hpp"
#include "windows/disk_space_error_dialog.hpp"
#include "windows/video_extractor_dialog.hpp"
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
        class FileBrowser;

        class GuiManager {
        public:
            GuiManager(VisualizerImpl* viewer);
            ~GuiManager();

            // Lifecycle
            void init();
            void shutdown();
            void render();

            // Sub-manager access
            [[nodiscard]] AsyncTaskManager& asyncTasks() { return async_tasks_; }
            [[nodiscard]] const AsyncTaskManager& asyncTasks() const { return async_tasks_; }
            [[nodiscard]] GizmoManager& gizmo() { return gizmo_manager_; }
            [[nodiscard]] const GizmoManager& gizmo() const { return gizmo_manager_; }
            [[nodiscard]] PanelLayoutManager& panelLayout() { return panel_layout_; }
            [[nodiscard]] const PanelLayoutManager& panelLayout() const { return panel_layout_; }

            // State queries
            bool needsAnimationFrame() const;

            // Window visibility
            void showWindow(const std::string& name, bool show = true);

            void setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback);

            // Viewport region access
            ImVec2 getViewportPos() const;
            ImVec2 getViewportSize() const;
            bool isMouseInViewport() const;
            bool isViewportFocused() const;
            bool isPositionInViewport(double x, double y) const;

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
            void updateInputOverrides(bool mouse_in_viewport);

            // Core dependencies
            VisualizerImpl* viewer_;

            // Owned components
            std::unique_ptr<FileBrowser> file_browser_;
            std::unique_ptr<DiskSpaceErrorDialog> disk_space_error_dialog_;
            std::unique_ptr<lfs::gui::VideoExtractorDialog> video_extractor_dialog_;
            std::optional<std::jthread> video_extraction_thread_;

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
            FontSet buildFontSet() const;

            // Async task management
            AsyncTaskManager async_tasks_;

            StartupOverlay startup_overlay_;

            // Native drag-drop handler
            NativeDragDrop drag_drop_;
            bool drag_drop_hovering_ = false;

            // Native panel wrapper storage (registered with PanelRegistry)
            std::vector<std::shared_ptr<IPanel>> native_panel_storage_;
        };
    } // namespace gui
} // namespace lfs::vis
