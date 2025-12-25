/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/commands/composite_command.hpp"
#include "command/commands/cropbox_command.hpp"
#include "command/commands/transform_command.hpp"
#include "core/events.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include "gui/panels/menu_bar.hpp"
#include "gui/panels/transform_panel.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/drag_drop_native.hpp"
#include "windows/exit_confirmation_popup.hpp"
#include "windows/export_dialog.hpp"
#include "windows/notification_popup.hpp"
#include "windows/save_directory_popup.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::core {
    struct SplatData;
}

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {
        class FileBrowser;
        class ScenePanel;
        class ProjectChangedDialogBox;

        class GuiManager {
        public:
            GuiManager(VisualizerImpl* viewer);
            ~GuiManager();

            // Lifecycle
            void init();
            void shutdown();
            void render();

            // State queries
            bool wantsInput() const;
            bool isAnyWindowActive() const;

            // Window visibility
            void showWindow(const std::string& name, bool show = true);
            void toggleWindow(const std::string& name);

            void setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback);

            // Viewport region access
            ImVec2 getViewportPos() const;
            ImVec2 getViewportSize() const;
            bool isMouseInViewport() const;
            bool isViewportFocused() const;
            bool isPositionInViewport(double x, double y) const;
            bool isViewportGizmoDragging() const { return viewport_gizmo_dragging_; }
            bool isResizingPanel() const { return resizing_panel_ || hovering_panel_edge_; }
            bool isPositionInViewportGizmo(double x, double y) const;

            // Crop box gizmo state access
            void setCropGizmoOperation(ImGuizmo::OPERATION op) { crop_gizmo_operation_ = op; }
            void setCropGizmoMode(ImGuizmo::MODE mode) { crop_gizmo_mode_ = mode; }
            ImGuizmo::OPERATION getCropGizmoOperation() const { return crop_gizmo_operation_; }
            ImGuizmo::MODE getCropGizmoMode() const { return crop_gizmo_mode_; }

            // Selection sub-mode shortcuts (Ctrl+1..5)
            void setSelectionSubMode(panels::SelectionSubMode mode);
            panels::SelectionSubMode getSelectionSubMode() const { return gizmo_toolbar_state_.selection_mode; }
            panels::ToolType getCurrentToolMode() const; // Delegates to EditorContext
            const panels::GizmoToolbarState& getGizmoToolbarState() const { return gizmo_toolbar_state_; }
            panels::TransformPanelState& getTransformPanelState() { return transform_panel_state_; }

            bool isForceExit() const { return force_exit_; }

            // Exit confirmation
            void requestExitConfirmation();
            bool isExitConfirmationPending() const;

            // Input capture for key rebinding
            bool isCapturingInput() const;
            bool isModalWindowOpen() const;
            void captureKey(int key, int mods);
            void captureMouseButton(int button, int mods);

        private:
            void setupEventHandlers();
            void applyDefaultStyle();
            void updateViewportRegion();
            void updateViewportFocus();
            void initMenuBar();

            // Core dependencies
            VisualizerImpl* viewer_;

            // Owned components
            std::unique_ptr<FileBrowser> file_browser_;
            std::unique_ptr<ScenePanel> scene_panel_;
            std::unique_ptr<ExportDialog> export_dialog_;
            std::unique_ptr<NotificationPopup> notification_popup_;
            std::unique_ptr<SaveDirectoryPopup> save_directory_popup_;
            std::unique_ptr<ExitConfirmationPopup> exit_confirmation_popup_;

            // UI state only
            std::unordered_map<std::string, bool> window_states_;
            bool show_main_panel_ = true;
            bool show_viewport_gizmo_ = true;

            // Speed overlay state
            bool speed_overlay_visible_ = false;
            std::chrono::steady_clock::time_point speed_overlay_start_time_;
            std::chrono::milliseconds speed_overlay_duration_;
            float current_speed_ = 0.0f;

            // Zoom speed overlay state
            bool zoom_speed_overlay_visible_ = false;
            std::chrono::steady_clock::time_point zoom_speed_overlay_start_time_;
            float zoom_speed_ = 5.0f;

            // Viewport region tracking
            ImVec2 viewport_pos_;
            ImVec2 viewport_size_;
            bool viewport_has_focus_;
            bool force_exit_ = false;

            // Right panel state
            float right_panel_width_ = 300.0f;
            float scene_panel_ratio_ = 0.4f;
            bool resizing_panel_ = false;
            bool hovering_panel_edge_ = false;
            static constexpr float RIGHT_PANEL_MIN_RATIO = 0.01f;
            static constexpr float RIGHT_PANEL_MAX_RATIO = 0.99f;

            // Viewport gizmo layout (must match ViewportGizmo settings)
            static constexpr float VIEWPORT_GIZMO_SIZE = 95.0f;
            static constexpr float VIEWPORT_GIZMO_MARGIN_X = 10.0f;
            static constexpr float VIEWPORT_GIZMO_MARGIN_Y = 10.0f;

            // Status bar layout
            static constexpr float STATUS_BAR_HEIGHT = 22.0f;

            // Crop box gizmo state (shared between panel and rendering)
            ImGuizmo::OPERATION crop_gizmo_operation_ = ImGuizmo::TRANSLATE;
            ImGuizmo::MODE crop_gizmo_mode_ = ImGuizmo::WORLD;

            // Method declarations
            void renderStatusBar(const UIContext& ctx);
            void showSpeedOverlay(float current_speed, float max_speed);
            void showZoomSpeedOverlay(float zoom_speed, float max_zoom_speed);
            void renderCropBoxGizmo(const UIContext& ctx);
            void renderNodeTransformGizmo(const UIContext& ctx);

            std::unique_ptr<MenuBar> menu_bar_;
            bool menu_bar_input_bindings_set_ = false;

            // Node transform gizmo state
            bool show_node_gizmo_ = true;
            ImGuizmo::OPERATION node_gizmo_operation_ = ImGuizmo::TRANSLATE;

            // Gizmo toolbar state
            panels::GizmoToolbarState gizmo_toolbar_state_;
            panels::TransformPanelState transform_panel_state_;

            // Cropbox undo/redo state
            bool cropbox_gizmo_active_ = false;
            std::string cropbox_node_name_;
            std::optional<command::CropBoxState> cropbox_state_before_drag_;

            // Node transform undo/redo state (supports multi-selection)
            bool node_gizmo_active_ = false;
            std::vector<std::string> node_gizmo_node_names_;
            std::vector<glm::mat4> node_transforms_before_drag_;
            glm::vec3 gizmo_pivot_{0.0f};
            glm::mat3 gizmo_cumulative_rotation_{1.0f};

            // Previous tool/selection mode for detecting changes
            panels::ToolType previous_tool_ = panels::ToolType::None;
            panels::SelectionSubMode previous_selection_mode_ = panels::SelectionSubMode::Centers;

            // Tool cleanup
            void deactivateAllTools();

            // Crop box flash effect
            std::chrono::steady_clock::time_point crop_flash_start_;
            bool crop_flash_active_ = false;
            void triggerCropFlash();
            void updateCropFlash();

            bool focus_training_panel_ = false;
            bool ui_hidden_ = false;

            // Font storage
            ImFont* font_regular_ = nullptr;
            ImFont* font_bold_ = nullptr;
            ImFont* font_heading_ = nullptr;
            ImFont* font_small_ = nullptr;
            ImFont* font_section_ = nullptr;

            // Viewport gizmo drag-to-orbit state
            bool viewport_gizmo_dragging_ = false;
            glm::dvec2 gizmo_drag_start_cursor_{0.0, 0.0};

            // Async export state
            struct ExportState {
                std::atomic<bool> active{false};
                std::atomic<bool> cancel_requested{false};
                std::atomic<float> progress{0.0f};
                lfs::core::ExportFormat format{lfs::core::ExportFormat::PLY}; // Protected by mutex
                std::string stage;                                            // Protected by mutex
                std::string error;                                            // Protected by mutex
                std::mutex mutex;
                std::unique_ptr<std::jthread> thread;
            };
            ExportState export_state_;

            void renderExportOverlay();
            void renderEmptyStateOverlay();
            void renderDragDropOverlay();
            void renderStartupOverlay();

            // Startup overlay state
            bool show_startup_overlay_ = true;
            unsigned int startup_logo_texture_ = 0;
            unsigned int startup_core11_texture_ = 0;
            int startup_logo_width_ = 0, startup_logo_height_ = 0;
            int startup_core11_width_ = 0, startup_core11_height_ = 0;
            void startAsyncExport(lfs::core::ExportFormat format,
                                  const std::filesystem::path& path,
                                  std::unique_ptr<lfs::core::SplatData> data);
            void cancelExport();
            bool isExporting() const { return export_state_.active.load(); }

            // Native drag-drop handler
            NativeDragDrop drag_drop_;
            bool drag_drop_hovering_ = false;
        };
    } // namespace gui
} // namespace lfs::vis
