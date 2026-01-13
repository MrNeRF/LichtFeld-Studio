/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/services.hpp"
#include "input/input_bindings.hpp"
#include "input/input_types.hpp"
#include "internal/viewport.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis {

    // Forward declarations
    namespace tools {
        class BrushTool;
        class AlignTool;
        class SelectionTool;
    } // namespace tools
    class ToolContext;

    class InputController {
    public:
        InputController(GLFWwindow* window, Viewport& viewport);
        ~InputController();

        // Setup - MUST be called AFTER ImGui is initialized!
        void initialize();

        // Set brush tool
        void setBrushTool(std::shared_ptr<tools::BrushTool> tool) {
            brush_tool_ = tool;
        }

        // Set align tool
        void setAlignTool(std::shared_ptr<tools::AlignTool> tool) {
            align_tool_ = tool;
        }

        // Set selection tool
        void setSelectionTool(std::shared_ptr<tools::SelectionTool> tool) {
            selection_tool_ = tool;
        }

        // Set tool context for gizmo
        void setToolContext(ToolContext* context) {
            tool_context_ = context;
        }

        // Called every frame by GUI manager to update viewport bounds
        void updateViewportBounds(float x, float y, float w, float h) {
            viewport_bounds_ = {x, y, w, h};
        }

        // Set special input modes
        void setPointCloudMode(bool enabled) {
            point_cloud_mode_ = enabled;
        }

        // Input bindings (customizable hotkeys/mouse)
        input::InputBindings& getBindings() { return bindings_; }
        const input::InputBindings& getBindings() const { return bindings_; }
        void loadInputProfile(const std::string& name) { bindings_.loadProfile(name); }

        // Update function for continuous input (WASD movement and inertia)
        void update(float delta_time);

        // Check if continuous input is active (WASD keys or camera drag)
        [[nodiscard]] bool isContinuousInputActive() const {
            const bool movement_active = keys_movement_[0] || keys_movement_[1] || keys_movement_[2] ||
                                         keys_movement_[3] || keys_movement_[4] || keys_movement_[5];
            const bool camera_drag = drag_mode_ == DragMode::Orbit ||
                                     drag_mode_ == DragMode::Pan ||
                                     drag_mode_ == DragMode::Rotate;
            return movement_active || camera_drag;
        }

        // Node rectangle selection state (for rendering)
        [[nodiscard]] bool isNodeRectDragging() const { return is_node_rect_dragging_; }
        [[nodiscard]] glm::vec2 getNodeRectStart() const { return node_rect_start_; }
        [[nodiscard]] glm::vec2 getNodeRectEnd() const { return node_rect_end_; }

        void handleFileDrop(const std::vector<std::string>& paths);

    private:
        // Store original ImGui callbacks so we can chain
        struct {
            GLFWmousebuttonfun mouse_button = nullptr;
            GLFWcursorposfun cursor_pos = nullptr;
            GLFWscrollfun scroll = nullptr;
            GLFWkeyfun key = nullptr;
            GLFWdropfun drop = nullptr;
            GLFWwindowfocusfun focus = nullptr;
        } imgui_callbacks_;

        // Our callbacks that chain to ImGui
        static void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods);
        static void cursorPosCallback(GLFWwindow* w, double x, double y);
        static void scrollCallback(GLFWwindow* w, double xoff, double yoff);
        static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);
        static void dropCallback(GLFWwindow* w, int count, const char** paths);
        static void windowFocusCallback(GLFWwindow* w, int focused);

        // Internal handlers
        void handleMouseButton(int button, int action, double x, double y);
        void handleMouseMove(double x, double y);
        void handleScroll(double xoff, double yoff);
        void handleKey(int key, int action, int mods);
        void handleGoToCamView(const lfs::core::events::cmd::GoToCamView& event);
        void handleFocusSelection();

        // WASD processing with proper frame timing
        void processWASDMovement();

        // Helpers
        bool isInViewport(double x, double y) const;
        bool shouldCameraHandleInput() const;
        void updateCameraSpeed(bool increase);
        void updateZoomSpeed(bool increase);
        void publishCameraMove();
        bool isNearSplitter(double x) const;
        int getModifierKeys() const;
        glm::vec3 unprojectScreenPoint(double x, double y, float fallback_distance = 5.0f) const;
        input::ToolMode getCurrentToolMode() const;

        // Training pause/resume helpers
        void onCameraMovementStart();
        void onCameraMovementEnd();
        void checkCameraMovementTimeout();

        // Core state
        GLFWwindow* window_;
        Viewport& viewport_;

        // Input bindings for customizable hotkeys
        input::InputBindings bindings_;

        // Tool support
        std::shared_ptr<tools::BrushTool> brush_tool_;
        std::shared_ptr<tools::AlignTool> align_tool_;
        std::shared_ptr<tools::SelectionTool> selection_tool_;
        ToolContext* tool_context_ = nullptr;

        // Viewport bounds for focus detection
        struct {
            float x, y, width, height;
        } viewport_bounds_{0, 0, 1920, 1080};

        // Camera state
        enum class DragMode {
            None,
            Pan,
            Rotate,
            Orbit,
            Gizmo,
            Splitter,
            Brush
        };
        DragMode drag_mode_ = DragMode::None;
        int drag_button_ = -1;
        glm::dvec2 last_mouse_pos_{0, 0};
        float splitter_start_pos_ = 0.5f;
        double splitter_start_x_ = 0.0;

        // Key states
        bool key_r_pressed_ = false;
        bool key_ctrl_pressed_ = false;
        bool key_alt_pressed_ = false;
        bool keys_movement_[6] = {false, false, false, false, false, false}; // fwd, left, back, right, down, up

        // Cached movement key bindings (refreshed when bindings change)
        struct MovementKeys {
            int forward = -1, backward = -1, left = -1, right = -1, up = -1, down = -1;
        } movement_keys_;
        void refreshMovementKeyCache();

        // Special modes
        bool point_cloud_mode_ = false;

        // Throttling for camera events
        std::chrono::steady_clock::time_point last_camera_publish_;
        static constexpr auto camera_publish_interval_ = std::chrono::milliseconds(100);

        // Camera movement tracking for training pause/resume
        bool camera_is_moving_ = false;
        bool training_was_paused_by_camera_ = false;
        std::chrono::steady_clock::time_point last_camera_movement_time_;
        static constexpr auto camera_movement_timeout_ = std::chrono::milliseconds(500);
        bool gt_comparison_active_ = false;

        // Frame timing for WASD movement
        std::chrono::high_resolution_clock::time_point last_frame_time_;

        // Cursor state tracking
        enum class CursorType {
            Default,
            Resize,
            Hand
        };
        CursorType current_cursor_ = CursorType::Default;
        GLFWcursor* resize_cursor_ = nullptr;
        GLFWcursor* hand_cursor_ = nullptr;

        // Double-click detection
        static constexpr double DOUBLE_CLICK_TIME = 0.3;
        static constexpr double DOUBLE_CLICK_DISTANCE = 5.0;

        // Camera frustum interaction
        int last_camview_ = -1;
        int hovered_camera_id_ = -1;
        int last_clicked_camera_id_ = -1;
        std::chrono::steady_clock::time_point last_click_time_;
        glm::dvec2 last_click_pos_{0, 0};

        // General double-click tracking
        std::chrono::steady_clock::time_point last_general_click_time_;
        glm::dvec2 last_general_click_pos_{0, 0};
        int last_general_click_button_ = -1;

        // Rectangle selection for nodes (when no tool is active)
        bool is_node_rect_dragging_ = false;
        glm::vec2 node_rect_start_{0.0f};
        glm::vec2 node_rect_end_{0.0f};

        static InputController* instance_;
    };

} // namespace lfs::vis