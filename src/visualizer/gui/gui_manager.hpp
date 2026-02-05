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
#include "gui/panels/menu_bar.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "gui/startup_overlay.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/drag_drop_native.hpp"
#include "windows/disk_space_error_dialog.hpp"
#include "windows/video_extractor_dialog.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <filesystem>
#include <memory>
#include <string>
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
            bool isViewportGizmoDragging() const { return gizmo_manager_.isViewportGizmoDragging(); }
            bool isResizingPanel() const { return panel_layout_.isResizingPanel(); }
            bool isPositionInViewportGizmo(double x, double y) const { return gizmo_manager_.isPositionInViewportGizmo(x, y); }

            void setSelectionSubMode(SelectionSubMode mode) { gizmo_manager_.setSelectionSubMode(mode); }
            [[nodiscard]] SelectionSubMode getSelectionSubMode() const { return gizmo_manager_.getSelectionSubMode(); }
            [[nodiscard]] ToolType getCurrentToolMode() const { return gizmo_manager_.getCurrentToolMode(); }

            [[nodiscard]] TransformSpace getTransformSpace() const { return gizmo_manager_.getTransformSpace(); }
            void setTransformSpace(TransformSpace space) { gizmo_manager_.setTransformSpace(space); }
            [[nodiscard]] PivotMode getPivotMode() const { return gizmo_manager_.getPivotMode(); }
            void setPivotMode(PivotMode mode) { gizmo_manager_.setPivotMode(mode); }
            [[nodiscard]] ImGuizmo::OPERATION getCurrentOperation() const { return gizmo_manager_.getCurrentOperation(); }
            void setCurrentOperation(ImGuizmo::OPERATION op) { gizmo_manager_.setCurrentOperation(op); }

            bool isCropboxGizmoActive() const { return gizmo_manager_.isCropboxGizmoActive(); }
            bool isEllipsoidGizmoActive() const { return gizmo_manager_.isEllipsoidGizmoActive(); }

            bool isForceExit() const { return force_exit_; }
            void setForceExit(bool value) { force_exit_ = value; }

            [[nodiscard]] SequencerController& sequencer() { return sequencer_ui_.controller(); }
            [[nodiscard]] const SequencerController& sequencer() const { return sequencer_ui_.controller(); }

            [[nodiscard]] bool isSequencerVisible() const { return panel_layout_.isShowSequencer(); }
            void setSequencerVisible(bool visible) { panel_layout_.setShowSequencer(visible); }

            [[nodiscard]] panels::SequencerUIState& getSequencerUIState() { return sequencer_ui_state_; }
            [[nodiscard]] const panels::SequencerUIState& getSequencerUIState() const { return sequencer_ui_state_; }

            [[nodiscard]] VisualizerImpl* getViewer() const { return viewer_; }
            [[nodiscard]] std::unordered_map<std::string, bool>* getWindowStates() { return &window_states_; }

            void requestExitConfirmation();
            bool isExitConfirmationPending() const;

            void performExport(lfs::core::ExportFormat format, const std::filesystem::path& path,
                               const std::vector<std::string>& node_names, int sh_degree) {
                async_tasks_.performExport(format, path, node_names, sh_degree);
            }

            bool isCapturingInput() const;
            bool isModalWindowOpen() const;
            [[nodiscard]] bool isStartupVisible() const { return startup_overlay_.isVisible(); }
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

            [[nodiscard]] float getExportProgress() const { return async_tasks_.getExportProgress(); }
            [[nodiscard]] std::string getExportStage() const { return async_tasks_.getExportStage(); }
            [[nodiscard]] lfs::core::ExportFormat getExportFormat() const { return async_tasks_.getExportFormat(); }
            [[nodiscard]] bool isExporting() const { return async_tasks_.isExporting(); }
            void cancelExport() { async_tasks_.cancelExport(); }

            [[nodiscard]] bool isImporting() const { return async_tasks_.isImporting(); }
            [[nodiscard]] bool isImportCompletionShowing() const { return async_tasks_.isImportCompletionShowing(); }
            [[nodiscard]] float getImportProgress() const { return async_tasks_.getImportProgress(); }
            [[nodiscard]] std::string getImportStage() const { return async_tasks_.getImportStage(); }
            [[nodiscard]] std::string getImportDatasetType() const { return async_tasks_.getImportDatasetType(); }
            [[nodiscard]] std::string getImportPath() const { return async_tasks_.getImportPath(); }
            [[nodiscard]] bool getImportSuccess() const { return async_tasks_.getImportSuccess(); }
            [[nodiscard]] std::string getImportError() const { return async_tasks_.getImportError(); }
            [[nodiscard]] size_t getImportNumImages() const { return async_tasks_.getImportNumImages(); }
            [[nodiscard]] size_t getImportNumPoints() const { return async_tasks_.getImportNumPoints(); }
            [[nodiscard]] float getImportSecondsSinceCompletion() const { return async_tasks_.getImportSecondsSinceCompletion(); }
            void dismissImport() { async_tasks_.dismissImport(); }

            [[nodiscard]] bool isExportingVideo() const { return async_tasks_.isExportingVideo(); }
            [[nodiscard]] float getVideoExportProgress() const { return async_tasks_.getVideoExportProgress(); }
            [[nodiscard]] int getVideoExportCurrentFrame() const { return async_tasks_.getVideoExportCurrentFrame(); }
            [[nodiscard]] int getVideoExportTotalFrames() const { return async_tasks_.getVideoExportTotalFrames(); }
            [[nodiscard]] std::string getVideoExportStage() const { return async_tasks_.getVideoExportStage(); }
            void cancelVideoExport() { async_tasks_.cancelVideoExport(); }

        private:
            void setupEventHandlers();
            void checkCudaVersionAndNotify();
            void applyDefaultStyle();
            void initMenuBar();
            void renderPythonPanels(const UIContext& ctx);
            void renderSelectionOverlays(const UIContext& ctx);
            void renderViewportDecorations();
            void updateInputOverrides(bool mouse_in_viewport);

            // Core dependencies
            VisualizerImpl* viewer_;

            // Owned components
            std::unique_ptr<FileBrowser> file_browser_;
            std::unique_ptr<DiskSpaceErrorDialog> disk_space_error_dialog_;
            std::unique_ptr<lfs::gui::VideoExtractorDialog> video_extractor_dialog_;

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
        };
    } // namespace gui
} // namespace lfs::vis
