/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/services.hpp"
#include "input/input_bindings.hpp"
#include "input/input_types.hpp"
#include "internal/viewport.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <memory>

struct SDL_Window;
struct SDL_Cursor;

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
        InputController(SDL_Window* window, Viewport& viewport);
        ~InputController();

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

        // Event handlers (called by WindowManager)
        void handleMouseButton(int button, int action, double x, double y);
        void handleMouseMove(double x, double y);
        void handleScroll(double xoff, double yoff);
        void handleKey(int key, int action, int mods);
        void handleFileDrop(const std::vector<std::string>& paths);
        void onWindowFocusLost();

    private:
        void handleGoToCamView(const lfs::core::events::cmd::GoToCamView& event);
        void handleFocusSelection();

        // WASD processing with proper frame timing
        void processWASDMovement();

        // Helpers
        bool isInViewport(double x, double y) const;
        bool shouldCameraHandleInput() const;
        void selectCameraByUid(int uid);
        void updateCameraSpeed(bool increase);
        void updateZoomSpeed(bool increase);
        void publishCameraMove();
        bool isNearSplitter(double x) const;
        int getModifierKeys() const;
        bool isKeyPressed(int app_key) const;
        bool isMouseButtonPressed(int app_button) const;
        glm::vec3 unprojectScreenPoint(double x, double y, float fallback_distance = 5.0f) const;
        std::pair<glm::vec3, glm::vec3> computePickRay(double x, double y) const;
        input::ToolMode getCurrentToolMode() const;

        // Training pause/resume helpers
        void onCameraMovementStart();
        void onCameraMovementEnd();
        void checkCameraMovementTimeout();

        // Core state
        SDL_Window* window_;
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
        SDL_Cursor* resize_cursor_ = nullptr;
        SDL_Cursor* hand_cursor_ = nullptr;

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