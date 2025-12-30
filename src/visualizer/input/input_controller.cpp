/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_controller.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/gui_manager.hpp"
#include "io/loader.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "tools/tool_base.hpp"
#include "training/training_manager.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <format>
#include <limits>
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis {

    using namespace lfs::core::events;

    InputController* InputController::instance_ = nullptr;

    InputController::InputController(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
        cmd::GoToCamView::when([this](const auto& e) { handleGoToCamView(e); });

        cmd::ResetCamera::when([this](const auto&) {
            viewport_.camera.resetToHome();
            publishCameraMove();
        });

        internal::WindowFocusLost::when([this](const auto&) {
            drag_mode_ = DragMode::None;
            std::fill(std::begin(keys_movement_), std::end(keys_movement_), false);
            hovered_camera_id_ = -1;

            // Clear ImGui input to prevent tooltip trails
            ImGui::GetIO().ClearInputKeys();
            ImGui::GetIO().ClearInputMouse();
        });
    }

    InputController::~InputController() {
        if (instance_ == this) {
            instance_ = nullptr;
        }

        // Clean up cursor resources
        if (resize_cursor_) {
            glfwDestroyCursor(resize_cursor_);
            resize_cursor_ = nullptr;
        }
        if (hand_cursor_) {
            glfwDestroyCursor(hand_cursor_);
            hand_cursor_ = nullptr;
        }

        // Reset cursor to default before destruction
        if (window_ && current_cursor_ != CursorType::Default) {
            glfwSetCursor(window_, nullptr);
        }
    }

    void InputController::initialize() {
        // Must be called after ImGui_ImplGlfw_InitForOpenGL
        instance_ = this;

        // Store ImGui's callbacks so we can chain to them
        imgui_callbacks_.mouse_button = glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        imgui_callbacks_.cursor_pos = glfwSetCursorPosCallback(window_, cursorPosCallback);
        imgui_callbacks_.scroll = glfwSetScrollCallback(window_, scrollCallback);
        imgui_callbacks_.key = glfwSetKeyCallback(window_, keyCallback);
        imgui_callbacks_.drop = glfwSetDropCallback(window_, dropCallback);
        imgui_callbacks_.focus = glfwSetWindowFocusCallback(window_, windowFocusCallback);

        // Get initial mouse position
        double x, y;
        glfwGetCursorPos(window_, &x, &y);
        last_mouse_pos_ = {x, y};

        // Initialize frame timer
        last_frame_time_ = std::chrono::high_resolution_clock::now();

        // Create cursors
        resize_cursor_ = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
        hand_cursor_ = glfwCreateStandardCursor(GLFW_HAND_CURSOR);

        // Cache movement key bindings and register for updates
        refreshMovementKeyCache();
        bindings_.setOnBindingsChanged([this]() { refreshMovementKeyCache(); });
    }

    void InputController::refreshMovementKeyCache() {
        movement_keys_.forward = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_FORWARD);
        movement_keys_.backward = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_BACKWARD);
        movement_keys_.left = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_LEFT);
        movement_keys_.right = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_RIGHT);
        movement_keys_.up = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_UP);
        movement_keys_.down = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_DOWN);
    }

    // Static callbacks - chain to ImGui then handle ourselves
    void InputController::mouseButtonCallback(GLFWwindow* w, int button, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.mouse_button) {
            instance_->imgui_callbacks_.mouse_button(w, button, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            double x, y;
            glfwGetCursorPos(w, &x, &y);
            instance_->handleMouseButton(button, action, x, y);
        }
    }

    void InputController::cursorPosCallback(GLFWwindow* w, double x, double y) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.cursor_pos) {
            instance_->imgui_callbacks_.cursor_pos(w, x, y);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleMouseMove(x, y);
        }
    }

    void InputController::scrollCallback(GLFWwindow* w, double xoff, double yoff) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.scroll) {
            instance_->imgui_callbacks_.scroll(w, xoff, yoff);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleScroll(xoff, yoff);
        }
    }

    void InputController::keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.key) {
            instance_->imgui_callbacks_.key(w, key, scancode, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleKey(key, action, mods);
        }
    }

    void InputController::dropCallback(GLFWwindow* w, int count, const char** paths) {
        LOG_INFO("GLFW drop callback: {} file(s)", count);

        // Let ImGui handle first (though it probably doesn't use this)
        if (instance_ && instance_->imgui_callbacks_.drop) {
            instance_->imgui_callbacks_.drop(w, count, paths);
        }

        // Then handle file drops
        if (instance_) {
            std::vector<std::string> files(paths, paths + count);
            instance_->handleFileDrop(files);
        } else {
            LOG_ERROR("InputController instance not available for drop callback");
        }
    }

    void InputController::windowFocusCallback(GLFWwindow* w, int focused) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.focus) {
            instance_->imgui_callbacks_.focus(w, focused);
        }

        // Reset states on focus loss
        if (!focused) {
            if (instance_) {
                instance_->drag_mode_ = DragMode::None;
                std::fill(std::begin(instance_->keys_movement_),
                          std::end(instance_->keys_movement_), false);
                instance_->hovered_camera_id_ = -1; // Reset hovered camera

                if (instance_->current_cursor_ != CursorType::Default) {
                    glfwSetCursor(instance_->window_, nullptr);
                    instance_->current_cursor_ = CursorType::Default;
                }
            }
            lfs::core::events::internal::WindowFocusLost{}.emit();
        }
    }

    bool InputController::isNearSplitter(double x) const {
        if (!services().renderingOrNull() || services().renderingOrNull()->getSettings().split_view_mode == SplitViewMode::Disabled) {
            return false;
        }

        float split_pos = services().renderingOrNull()->getSettings().split_position;
        float split_x = viewport_bounds_.x + viewport_bounds_.width * split_pos;

        // Increase the hit area to 10 pixels for easier grabbing
        return std::abs(x - split_x) < 10.0;
    }

    // Core handlers
    void InputController::handleMouseButton(int button, int action, double x, double y) {
        // Forward to GUI for mouse capture (rebinding)
        if (action == GLFW_PRESS && services().guiOrNull() && services().guiOrNull()->isCapturingInput()) {
            services().guiOrNull()->captureMouseButton(button, getModifierKeys());
            return;
        }

        // Check for splitter drag FIRST
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            // Check for double-click on camera frustum
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = std::chrono::duration<double>(now - last_click_time_).count();
            double dist = glm::length(glm::dvec2(x, y) - last_click_pos_);

            constexpr double DOUBLE_CLICK_TIME = 0.5;
            constexpr double DOUBLE_CLICK_DISTANCE = 10.0;

            bool is_double_click = (time_since_last < DOUBLE_CLICK_TIME &&
                                    dist < DOUBLE_CLICK_DISTANCE);

            // If we have a hovered camera, check for double-click
            if (hovered_camera_id_ >= 0) {
                if (is_double_click && hovered_camera_id_ == last_clicked_camera_id_) {
                    cmd::GoToCamView{.cam_id = hovered_camera_id_}.emit();

                    // Reset click tracking to prevent triple-click
                    last_click_time_ = std::chrono::steady_clock::time_point();
                    last_click_pos_ = {-1000, -1000}; // Far away position
                    last_clicked_camera_id_ = -1;
                    return;
                }
                // First click on a camera - record it
                last_click_time_ = now;
                last_click_pos_ = {x, y};
                last_clicked_camera_id_ = hovered_camera_id_;
            } else {
                last_click_time_ = std::chrono::steady_clock::time_point();
                last_click_pos_ = {-1000, -1000};
                last_clicked_camera_id_ = -1;
            }

            // Check for splitter drag
            if (isNearSplitter(x) && services().renderingOrNull()) {
                drag_mode_ = DragMode::Splitter;
                splitter_start_pos_ = services().renderingOrNull()->getSettings().split_position;
                splitter_start_x_ = x;
                glfwSetCursor(window_, resize_cursor_);
                LOG_TRACE("Started splitter drag");
                return;
            }
        }

        if (action == GLFW_RELEASE && drag_mode_ == DragMode::Splitter) {
            drag_mode_ = DragMode::None;
            glfwSetCursor(window_, nullptr); // Reset cursor
            LOG_TRACE("Ended splitter drag");
            return;
        }

        const bool over_gui = ImGui::GetIO().WantCaptureMouse ||
                              (services().guiOrNull() && services().guiOrNull()->isResizingPanel());
        const bool over_gizmo = services().guiOrNull() && services().guiOrNull()->isPositionInViewportGizmo(x, y);

        // Single binding lookup with current tool mode
        const int mods = getModifierKeys();
        const auto mouse_btn = static_cast<input::MouseButton>(button);
        const auto tool_mode = getCurrentToolMode();

        // Check for double-click
        auto now = std::chrono::steady_clock::now();
        const double time_since_last = std::chrono::duration<double>(now - last_general_click_time_).count();
        const double dist = glm::length(glm::dvec2(x, y) - last_general_click_pos_);
        const bool is_general_double_click = (time_since_last < DOUBLE_CLICK_TIME &&
                                              dist < DOUBLE_CLICK_DISTANCE &&
                                              last_general_click_button_ == button);

        // Update click tracking for double-click detection
        last_general_click_time_ = now;
        last_general_click_pos_ = {x, y};
        last_general_click_button_ = button;

        // Check double-click bindings first, then drag bindings
        auto bound_action = input::Action::NONE;
        if (is_general_double_click) {
            bound_action = bindings_.getActionForMouseButton(tool_mode, mouse_btn, mods, true);
        }
        if (bound_action == input::Action::NONE) {
            bound_action = bindings_.getActionForDrag(tool_mode, mouse_btn, mods);
        }
        if (bound_action == input::Action::NONE) {
            bound_action = bindings_.getActionForMouseButton(tool_mode, mouse_btn, mods, false);
        }

        if (action == GLFW_PRESS) {
            // Block if hovering over GUI window
            if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                return;
            }

            // Block if ImGuizmo is being used or hovered
            if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
                return;
            }

            // Only handle if clicking within the viewport
            if (!isInViewport(x, y)) {
                return;
            }

            switch (bound_action) {
            case input::Action::CAMERA_PAN:
                viewport_.camera.initScreenPos(glm::vec2(x, y));
                drag_mode_ = DragMode::Pan;
                drag_button_ = button;
                break;

            case input::Action::CAMERA_ORBIT:
                viewport_.camera.initScreenPos(glm::vec2(x, y));
                drag_mode_ = DragMode::Orbit;
                drag_button_ = button;
                viewport_.camera.startRotateAroundCenter(glm::vec2(x, y), static_cast<float>(glfwGetTime()));
                break;

            case input::Action::CAMERA_SET_PIVOT: {
                const glm::vec3 new_pivot = unprojectScreenPoint(x, y);
                const float current_distance = glm::length(viewport_.camera.getPivot() - viewport_.camera.t);
                const glm::vec3 forward = glm::normalize(viewport_.camera.R * glm::vec3(0, 0, 1));
                viewport_.camera.t = new_pivot - forward * current_distance;
                viewport_.camera.setPivot(new_pivot);
                publishCameraMove();
                break;
            }

            case input::Action::SELECTION_REPLACE:
            case input::Action::SELECTION_ADD:
            case input::Action::SELECTION_REMOVE:
                // Skip selection if clicking on viewport gizmo
                if (!over_gui && !over_gizmo && tool_context_) {
                    if (selection_tool_ && selection_tool_->isEnabled()) {
                        if (selection_tool_->handleMouseButton(button, action, mods, x, y, *tool_context_)) {
                            drag_mode_ = DragMode::Brush;
                            drag_button_ = button;
                        }
                    } else if (brush_tool_ && brush_tool_->isEnabled()) {
                        if (brush_tool_->handleMouseButton(button, action, mods, x, y, *tool_context_)) {
                            drag_mode_ = DragMode::Brush;
                            drag_button_ = button;
                        }
                    }
                }
                break;

            case input::Action::NONE:
            default:
                if (align_tool_ && align_tool_->isEnabled() && tool_context_) {
                    if (!over_gui && align_tool_->handleMouseButton(button, action, x, y, *tool_context_)) {
                        return;
                    }
                }

                // Node picking (controlled by bindings, skips if ImGuizmo active)
                const input::ToolMode input_mode = getCurrentToolMode();
                const input::Action pick_action = bindings_.getActionForMouseButton(
                    input_mode, input::MouseButton::LEFT, mods);
                const input::Action drag_action = bindings_.getActionForDrag(
                    input_mode, input::MouseButton::LEFT, mods);
                const bool has_node_binding = (pick_action == input::Action::NODE_PICK ||
                                               drag_action == input::Action::NODE_RECT_SELECT);

                if (!over_gui && !over_gizmo && button == GLFW_MOUSE_BUTTON_LEFT && tool_context_ &&
                    !ImGuizmo::IsOver() && !ImGuizmo::IsUsing() && has_node_binding) {
                    is_node_rect_dragging_ = true;
                    node_rect_start_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
                    node_rect_end_ = node_rect_start_;
                }
                break;
            }
        } else if (action == GLFW_RELEASE) {
            bool was_dragging = false;

            if (drag_mode_ == DragMode::Pan) {
                drag_mode_ = DragMode::None;
                drag_button_ = -1;
                was_dragging = true;
            } else if (drag_mode_ == DragMode::Orbit) {
                viewport_.camera.endRotateAroundCenter();
                drag_mode_ = DragMode::None;
                drag_button_ = -1;
                was_dragging = true;
            } else if (drag_mode_ == DragMode::Brush) {
                // Release for selection/brush tools
                if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
                    selection_tool_->handleMouseButton(button, action, mods, x, y, *tool_context_);
                } else if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
                    brush_tool_->handleMouseButton(button, action, mods, x, y, *tool_context_);
                }
                drag_mode_ = DragMode::None;
                drag_button_ = -1;
            }

            if (was_dragging) {
                ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
                onCameraMovementEnd();
            }

            // Node picking on release
            if (is_node_rect_dragging_ && button == GLFW_MOUSE_BUTTON_LEFT && tool_context_) {
                is_node_rect_dragging_ = false;
                auto* scene_manager = tool_context_->getSceneManager();
                if (scene_manager) {
                    constexpr float CLICK_THRESHOLD_PX = 5.0f;
                    const float drag_dist = glm::length(node_rect_end_ - node_rect_start_);

                    if (drag_dist < CLICK_THRESHOLD_PX) {
                        // Point pick
                        const glm::vec3 world_pos = unprojectScreenPoint(x, y);
                        const std::string picked = scene_manager->pickNodeAtWorldPosition(world_pos);
                        if (!picked.empty()) {
                            scene_manager->selectNode(picked);
                        } else {
                            scene_manager->clearSelection();
                        }
                    } else {
                        // Rectangle selection
                        const glm::vec2 rect_min(
                            std::min(node_rect_start_.x, node_rect_end_.x),
                            std::min(node_rect_start_.y, node_rect_end_.y));
                        const glm::vec2 rect_max(
                            std::max(node_rect_start_.x, node_rect_end_.x),
                            std::max(node_rect_start_.y, node_rect_end_.y));

                        const std::vector<std::string> picked_nodes = scene_manager->pickNodesInScreenRect(
                            rect_min, rect_max,
                            viewport_.getViewMatrix(),
                            viewport_.getProjectionMatrix(),
                            viewport_.windowSize);

                        if (picked_nodes.empty()) {
                            scene_manager->clearSelection();
                        } else {
                            scene_manager->selectNodes(picked_nodes);
                        }
                    }
                }
            }
        }
    }

    void InputController::handleMouseMove(double x, double y) {
        // Track if we moved significantly
        glm::dvec2 current_pos{x, y};

        // Handle splitter dragging
        if (drag_mode_ == DragMode::Splitter && services().renderingOrNull()) {
            double delta = x - splitter_start_x_;
            float new_pos = splitter_start_pos_ + static_cast<float>(delta / viewport_bounds_.width);

            // FIX: Allow dragging all the way to the edges - no margins!
            new_pos = std::clamp(new_pos, 0.0f, 1.0f);

            ui::SplitPositionChanged{.position = new_pos}.emit();
            last_mouse_pos_ = {x, y};
            return;
        }

        // Camera frustum hover detection with improved throttling
        // (frustum visibility is now controlled by scene graph, not a checkbox)
        if (services().renderingOrNull() &&
            isInViewport(x, y) &&
            drag_mode_ == DragMode::None &&
            !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {

            // Additional throttling based on movement distance
            static glm::dvec2 last_pick_pos{-1, -1};
            static constexpr double MIN_PICK_DISTANCE = 3.0; // pixels

            bool should_pick = false;

            // Check if we moved enough from last pick position
            if (last_pick_pos.x < 0) {
                // First pick
                should_pick = true;
                last_pick_pos = current_pos;
            } else {
                double pick_distance = glm::length(current_pos - last_pick_pos);
                if (pick_distance >= MIN_PICK_DISTANCE) {
                    should_pick = true;
                    last_pick_pos = current_pos;
                }
            }

            if (should_pick) {
                auto result = services().renderingOrNull()->pickCameraFrustum(glm::vec2(x, y));
                if (result >= 0) {
                    const int cam_id = result;
                    if (cam_id != hovered_camera_id_) {
                        hovered_camera_id_ = cam_id;
                        LOG_TRACE("Hovering over camera ID: {}", cam_id);

                        // Change cursor to hand
                        if (current_cursor_ != CursorType::Hand) {
                            glfwSetCursor(window_, hand_cursor_);
                            current_cursor_ = CursorType::Hand;
                        }
                    }
                } else {
                    // No camera under cursor
                    if (hovered_camera_id_ != -1) {
                        hovered_camera_id_ = -1;
                        LOG_TRACE("No longer hovering over camera");
                        if (current_cursor_ == CursorType::Hand) {
                            glfwSetCursor(window_, nullptr);
                            current_cursor_ = CursorType::Default;
                        }
                    }
                }
            }
        } else {
            // Not in conditions for camera picking
            if (hovered_camera_id_ != -1) {
                hovered_camera_id_ = -1;
                if (current_cursor_ == CursorType::Hand) {
                    glfwSetCursor(window_, nullptr);
                    current_cursor_ = CursorType::Default;
                }
            }
        }

        // Determine if we should show resize cursor for splitter
        bool should_show_resize = false;
        if (services().renderingOrNull() && services().renderingOrNull()->getSettings().split_view_mode != SplitViewMode::Disabled) {
            should_show_resize = (drag_mode_ == DragMode::None &&
                                  isInViewport(x, y) &&
                                  isNearSplitter(x) &&
                                  !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow));
        }

        // Only call glfwSetCursor when state actually changes
        if (should_show_resize && current_cursor_ != CursorType::Resize) {
            glfwSetCursor(window_, resize_cursor_);
            current_cursor_ = CursorType::Resize;
        } else if (!should_show_resize && current_cursor_ == CursorType::Resize) {
            glfwSetCursor(window_, nullptr);
            current_cursor_ = CursorType::Default;
        }

        if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
            if (drag_mode_ == DragMode::Brush) {
                brush_tool_->handleMouseMove(x, y, *tool_context_);
                last_mouse_pos_ = {x, y};
                return;
            } else if (brush_tool_->handleMouseMove(x, y, *tool_context_)) {
                last_mouse_pos_ = {x, y};
                return;
            }
        }

        // Skip selection if viewport gizmo is being dragged
        const bool gizmo_dragging = services().guiOrNull() && services().guiOrNull()->isViewportGizmoDragging();
        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_ && !gizmo_dragging) {
            if (drag_mode_ == DragMode::Brush) {
                selection_tool_->handleMouseMove(x, y, *tool_context_);
                last_mouse_pos_ = {x, y};
                return;
            } else if (selection_tool_->handleMouseMove(x, y, *tool_context_)) {
                last_mouse_pos_ = {x, y};
                return;
            }
        }

        glm::vec2 pos(x, y);
        last_mouse_pos_ = current_pos;

        // Node rectangle dragging
        if (is_node_rect_dragging_) {
            node_rect_end_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
            if (tool_context_)
                tool_context_->requestRender();
        }

        // Block camera dragging if ImGuizmo is being used
        if (ImGuizmo::IsUsing()) {
            return;
        }

        // Handle camera dragging
        if (drag_mode_ != DragMode::None &&
            drag_mode_ != DragMode::Gizmo &&
            drag_mode_ != DragMode::Splitter) {

            switch (drag_mode_) {
            case DragMode::Pan:
                viewport_.camera.translate(pos);
                break;
            case DragMode::Rotate:
                viewport_.camera.rotate(pos);
                break;
            case DragMode::Orbit: {
                float current_time = static_cast<float>(glfwGetTime());
                viewport_.camera.updateRotateAroundCenter(pos, current_time);
                break;
            }
            default:
                break;
            }
            // Signal continuous camera movement
            onCameraMovementStart();
            publishCameraMove();
        }
    }

    void InputController::handleScroll([[maybe_unused]] double xoff, double yoff) {
        if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
            if (brush_tool_->handleScroll(xoff, yoff, getModifierKeys(), *tool_context_)) {
                return;
            }
        }

        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
            if (selection_tool_->handleScroll(xoff, yoff, getModifierKeys(), *tool_context_)) {
                return;
            }
        }

        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter) {
            return;
        }

        // Block scroll when hovering over GUI windows (panels)
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            return;
        }

        if (!shouldCameraHandleInput())
            return;

        const float delta = static_cast<float>(yoff);
        if (std::abs(delta) < 0.01f)
            return;

        if (key_r_pressed_) {
            viewport_.camera.rotate_roll(delta);
        } else {
            // In orthographic mode, adjust ortho_scale instead of camera position
            if (services().renderingOrNull()) {
                auto settings = services().renderingOrNull()->getSettings();
                if (settings.orthographic) {
                    // Zoom factor: positive delta = zoom in (increase scale)
                    constexpr float ORTHO_ZOOM_FACTOR = 0.1f;
                    const float scale_factor = 1.0f + delta * ORTHO_ZOOM_FACTOR;
                    settings.ortho_scale = std::clamp(settings.ortho_scale * scale_factor, 1.0f, 10000.0f);
                    services().renderingOrNull()->updateSettings(settings);
                    services().renderingOrNull()->markDirty();
                } else {
                    viewport_.camera.zoom(delta);
                }
            } else {
                viewport_.camera.zoom(delta);
            }
        }

        onCameraMovementStart();
        publishCameraMove();
    }

    void InputController::handleKey(int key, int action, [[maybe_unused]] int mods) {
        // Track modifier keys (always, even if GUI has focus)
        if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
            key_ctrl_pressed_ = (action != GLFW_RELEASE);
        }
        if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) {
            key_alt_pressed_ = (action != GLFW_RELEASE);
        }
        if (key == GLFW_KEY_R) {
            key_r_pressed_ = (action != GLFW_RELEASE);
        }

        // Forward to GUI for key capture (rebinding)
        if (action == GLFW_PRESS && services().guiOrNull() && services().guiOrNull()->isCapturingInput()) {
            services().guiOrNull()->captureKey(key, mods);
            return;
        }

        // Selection tool handles Enter/Escape before global bindings consume them
        if (action == GLFW_PRESS && !ImGui::IsAnyItemActive() &&
            selection_tool_ && selection_tool_->isEnabled() && tool_context_ &&
            selection_tool_->handleKeyPress(key, mods, *tool_context_)) {
            return;
        }

        // Handle key press/repeat actions through bindings
        if ((action == GLFW_PRESS || action == GLFW_REPEAT) && !ImGui::IsAnyItemActive()) {
            const auto tool_mode = getCurrentToolMode();
            const auto bound_action = bindings_.getActionForKey(tool_mode, key, mods);

            // Only speed controls support key repeat
            if (action == GLFW_REPEAT) {
                if (bound_action != input::Action::CAMERA_SPEED_UP &&
                    bound_action != input::Action::CAMERA_SPEED_DOWN &&
                    bound_action != input::Action::ZOOM_SPEED_UP &&
                    bound_action != input::Action::ZOOM_SPEED_DOWN) {
                    return;
                }
            }

            if (bound_action != input::Action::NONE) {
                switch (bound_action) {
                case input::Action::TOGGLE_SPLIT_VIEW:
                    cmd::ToggleSplitView{}.emit();
                    return;

                case input::Action::TOGGLE_GT_COMPARISON:
                    cmd::ToggleGTComparison{}.emit();
                    return;

                case input::Action::CAMERA_RESET_HOME:
                    viewport_.camera.resetToHome();
                    publishCameraMove();
                    return;

                case input::Action::CAMERA_FOCUS_SELECTION:
                    handleFocusSelection();
                    return;

                case input::Action::CYCLE_PLY:
                    cmd::CyclePLY{}.emit();
                    return;

                case input::Action::CYCLE_SELECTION_VIS:
                    if (services().guiOrNull() &&
                        services().guiOrNull()->getCurrentToolMode() == gui::panels::ToolType::Selection) {
                        cmd::CycleSelectionVisualization{}.emit();
                    }
                    return;

                case input::Action::DELETE_SELECTED:
                    cmd::DeleteSelected{}.emit();
                    return;

                case input::Action::DELETE_NODE:
                    // Delete selected PLY node(s)
                    if (tool_context_) {
                        if (auto* sm = tool_context_->getSceneManager()) {
                            const auto selected = sm->getSelectedNodeNames();
                            for (const auto& name : selected) {
                                cmd::RemovePLY{.name = name, .keep_children = false}.emit();
                            }
                        }
                    }
                    return;

                case input::Action::UNDO:
                    cmd::Undo{}.emit();
                    return;

                case input::Action::REDO:
                    cmd::Redo{}.emit();
                    return;

                case input::Action::INVERT_SELECTION:
                    cmd::InvertSelection{}.emit();
                    return;

                case input::Action::DESELECT_ALL:
                    cmd::DeselectAll{}.emit();
                    return;

                case input::Action::SELECT_ALL:
                    cmd::SelectAll{}.emit();
                    return;

                case input::Action::COPY_SELECTION:
                    cmd::CopySelection{}.emit();
                    return;

                case input::Action::PASTE_SELECTION:
                    cmd::PasteSelection{}.emit();
                    return;

                case input::Action::APPLY_CROP_BOX:
                    cmd::ApplyCropBox{}.emit();
                    return;

                case input::Action::CYCLE_BRUSH_MODE:
                    if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
                        brush_tool_->handleKeyPress(key, mods, *tool_context_);
                    }
                    return;

                case input::Action::CAMERA_NEXT_VIEW:
                    if (ImGui::GetIO().WantTextInput)
                        return;
                    if (services().trainerOrNull()) {
                        const int num_cams = static_cast<int>(services().trainerOrNull()->getCamList().size());
                        if (num_cams > 0) {
                            last_camview_ = (last_camview_ < 0) ? 0 : (last_camview_ + 1) % num_cams;
                            cmd::GoToCamView{.cam_id = last_camview_}.emit();
                        }
                    }
                    return;

                case input::Action::CAMERA_PREV_VIEW:
                    if (ImGui::GetIO().WantTextInput)
                        return;
                    if (services().trainerOrNull()) {
                        const int num_cams = static_cast<int>(services().trainerOrNull()->getCamList().size());
                        if (num_cams > 0) {
                            last_camview_ = (last_camview_ < 0) ? num_cams - 1 : (last_camview_ - 1 + num_cams) % num_cams;
                            cmd::GoToCamView{.cam_id = last_camview_}.emit();
                        }
                    }
                    return;

                case input::Action::CAMERA_SPEED_UP:
                    updateCameraSpeed(true);
                    return;

                case input::Action::CAMERA_SPEED_DOWN:
                    updateCameraSpeed(false);
                    return;

                case input::Action::ZOOM_SPEED_UP:
                    updateZoomSpeed(true);
                    return;

                case input::Action::ZOOM_SPEED_DOWN:
                    updateZoomSpeed(false);
                    return;

                case input::Action::SELECT_MODE_CENTERS:
                    if (services().guiOrNull()) {
                        services().guiOrNull()->setSelectionSubMode(gui::panels::SelectionSubMode::Centers);
                    }
                    return;

                case input::Action::SELECT_MODE_RECTANGLE:
                    if (services().guiOrNull()) {
                        services().guiOrNull()->setSelectionSubMode(gui::panels::SelectionSubMode::Rectangle);
                    }
                    return;

                case input::Action::SELECT_MODE_POLYGON:
                    if (services().guiOrNull()) {
                        services().guiOrNull()->setSelectionSubMode(gui::panels::SelectionSubMode::Polygon);
                    }
                    return;

                case input::Action::SELECT_MODE_LASSO:
                    if (services().guiOrNull()) {
                        services().guiOrNull()->setSelectionSubMode(gui::panels::SelectionSubMode::Lasso);
                    }
                    return;

                case input::Action::SELECT_MODE_RINGS:
                    if (services().guiOrNull()) {
                        services().guiOrNull()->setSelectionSubMode(gui::panels::SelectionSubMode::Rings);
                    }
                    return;

                case input::Action::TOGGLE_UI:
                    ui::ToggleUI{}.emit();
                    return;

                case input::Action::TOGGLE_FULLSCREEN:
                    ui::ToggleFullscreen{}.emit();
                    return;

                default:
                    break;
                }
            }
        }

        // Movement keys only work when viewport has focus and gizmo isn't active
        if (!shouldCameraHandleInput() || drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter)
            return;

        // Use cached movement key bindings
        const bool pressed = (action != GLFW_RELEASE);
        if (key == movement_keys_.forward) {
            keys_movement_[0] = pressed;
        } else if (key == movement_keys_.left) {
            keys_movement_[1] = pressed;
        } else if (key == movement_keys_.backward) {
            keys_movement_[2] = pressed;
        } else if (key == movement_keys_.right) {
            keys_movement_[3] = pressed;
        } else if (key == movement_keys_.down) {
            keys_movement_[4] = pressed;
        } else if (key == movement_keys_.up) {
            keys_movement_[5] = pressed;
        }
    }

    void InputController::update(float delta_time) {
        const bool drag_button_released = drag_button_ >= 0 &&
                                          glfwGetMouseButton(window_, drag_button_) != GLFW_PRESS;

        // Handle missed mouse release events (e.g., outside window)
        if (drag_mode_ == DragMode::Orbit && drag_button_released) {
            viewport_.camera.endRotateAroundCenter();
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
        }

        if (drag_mode_ == DragMode::Pan && drag_button_released) {
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
        }

        if (drag_mode_ == DragMode::Splitter &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
            glfwSetCursor(window_, nullptr);
        }

        if (drag_mode_ == DragMode::Brush && drag_button_released) {
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
        }

        // Sync movement key states with actual keyboard (using cached keys)
        const auto& mk = movement_keys_;
        if (keys_movement_[0] && (mk.forward < 0 || glfwGetKey(window_, mk.forward) != GLFW_PRESS)) {
            keys_movement_[0] = false;
        }
        if (keys_movement_[1] && (mk.left < 0 || glfwGetKey(window_, mk.left) != GLFW_PRESS)) {
            keys_movement_[1] = false;
        }
        if (keys_movement_[2] && (mk.backward < 0 || glfwGetKey(window_, mk.backward) != GLFW_PRESS)) {
            keys_movement_[2] = false;
        }
        if (keys_movement_[3] && (mk.right < 0 || glfwGetKey(window_, mk.right) != GLFW_PRESS)) {
            keys_movement_[3] = false;
        }
        if (keys_movement_[4] && (mk.down < 0 || glfwGetKey(window_, mk.down) != GLFW_PRESS)) {
            keys_movement_[4] = false;
        }
        if (keys_movement_[5] && (mk.up < 0 || glfwGetKey(window_, mk.up) != GLFW_PRESS)) {
            keys_movement_[5] = false;
        }

        // Handle continuous movement
        if (shouldCameraHandleInput() && drag_mode_ != DragMode::Gizmo && drag_mode_ != DragMode::Splitter) {
            if (keys_movement_[0]) {
                viewport_.camera.advance_forward(delta_time);
            }
            if (keys_movement_[1]) {
                viewport_.camera.advance_left(delta_time);
            }
            if (keys_movement_[2]) {
                viewport_.camera.advance_backward(delta_time);
            }
            if (keys_movement_[3]) {
                viewport_.camera.advance_right(delta_time);
            }
            if (keys_movement_[4]) {
                viewport_.camera.advance_up(delta_time);
            }
            if (keys_movement_[5]) {
                viewport_.camera.advance_down(delta_time);
            }
        }

        // Publish if moving (removed inertia check)
        bool moving = keys_movement_[0] || keys_movement_[1] || keys_movement_[2] || keys_movement_[3] || keys_movement_[4] || keys_movement_[5];
        if (moving) {
            onCameraMovementStart();
            publishCameraMove();
        }

        // Check if camera movement has timed out and should resume training
        checkCameraMovementTimeout();
    }

    void InputController::handleFileDrop(const std::vector<std::string>& paths) {
        using namespace lfs::core::events;
        ui::FileDropReceived{}.emit();

        LOG_INFO("Processing {} dropped file(s)", paths.size());

        std::vector<std::filesystem::path> splat_files;
        std::optional<std::filesystem::path> dataset_path;
        std::vector<std::string> unrecognized_files;

        for (const auto& path_str : paths) {
            std::filesystem::path filepath = lfs::core::utf8_to_path(path_str);
            LOG_DEBUG("Processing dropped file: {}", lfs::core::path_to_utf8(filepath));

            auto ext = filepath.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".resume") {
                cmd::LoadCheckpointForTraining{.path = filepath}.emit();
                LOG_INFO("Loading checkpoint via drag-and-drop: {}", lfs::core::path_to_utf8(filepath.filename()));
                return;
            } else if (ext == ".json") {
                if (lfs::io::Loader::isDatasetPath(filepath)) {
                    dataset_path = filepath;
                }else {
                    cmd::LoadConfigFile{.path = filepath}.emit();
                    LOG_INFO("Loading config via drag-and-drop: {}", lfs::core::path_to_utf8(filepath.filename()));
                    return;
                }
            } else if (ext == ".ply" || ext == ".sog" || ext == ".spz") {
                splat_files.push_back(filepath);
            } else if (std::filesystem::is_directory(filepath)) {
                // Check for dataset markers
                LOG_DEBUG("Checking directory for dataset markers: {}", lfs::core::path_to_utf8(filepath));
                if (lfs::io::Loader::isDatasetPath(filepath)) {
                    if (!dataset_path) {
                        dataset_path = filepath;
                    }
                } else {
                    // Check if it's a SOG directory (WebP-based format)
                    if (std::filesystem::exists(filepath / "meta.json") &&
                        std::filesystem::exists(filepath / "means_l.webp")) {
                        splat_files.push_back(filepath);
                        LOG_DEBUG("Detected SOG directory: {}", lfs::core::path_to_utf8(filepath));
                    } else {
                        unrecognized_files.push_back(lfs::core::path_to_utf8(filepath));
                    }
                }
            } else {
                unrecognized_files.push_back(lfs::core::path_to_utf8(filepath));
            }
        }

        // Load splat files (PLY, SOG, or SPZ)
        for (const auto& splat : splat_files) {
            cmd::LoadFile{.path = splat, .is_dataset = false}.emit();
            LOG_INFO("Loading {} via drag-and-drop: {}",
                     lfs::core::path_to_utf8(splat.extension()), lfs::core::path_to_utf8(splat.filename()));
        }

        if (dataset_path) {
            LOG_INFO("Showing dataset load popup for: {}", lfs::core::path_to_utf8(*dataset_path));
            cmd::ShowDatasetLoadPopup{.dataset_path = *dataset_path}.emit();
        }

        // Warn about unrecognized files
        if (!unrecognized_files.empty() && splat_files.empty() && !dataset_path) {
            LOG_WARN("Dropped {} file(s) with unrecognized format:", unrecognized_files.size());
            for (const auto& f : unrecognized_files) {
                LOG_WARN("  - {}", f);
            }
            LOG_WARN("Supported formats: .ply, .sog, .spz, .json, .resume, or dataset directories");
        }
    }

    void InputController::handleGoToCamView(const lfs::core::events::cmd::GoToCamView& event) {
        LOG_TIMER_TRACE("HandleGoToCamView");

        if (!services().trainerOrNull()) {
            LOG_ERROR("GoToCamView: trainer_manager_ not initialized");
            return;
        }

        auto cam_data = services().trainerOrNull()->getCamById(event.cam_id);
        if (!cam_data) {
            LOG_ERROR("Camera ID {} not found", event.cam_id);
            return;
        }

        // Get rotation and translation tensors and ensure they're on CPU
        auto R_tensor = cam_data->R().cpu();
        auto T_tensor = cam_data->T().cpu();

        // Get raw CPU pointers - safer and more efficient
        const float* R_data = R_tensor.ptr<float>();
        const float* T_data = T_tensor.ptr<float>();

        if (!R_data || !T_data) {
            LOG_ERROR("Failed to get camera R/T data pointers");
            return;
        }

        // R_data is world_to_cam rotation stored row-major
        // We need cam_to_world for the viewport
        glm::mat3 world_to_cam_R;

        // Load the matrix properly: R_data is row-major, GLM is column-major
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                // R_data[row * 3 + col] is element at [row][col] in row-major
                // GLM[col][row] is element at [row][col] when thinking row-major
                world_to_cam_R[col][row] = R_data[row * 3 + col];
            }
        }

        glm::vec3 world_to_cam_T(T_data[0], T_data[1], T_data[2]);

        // Convert to camera-to-world transform
        glm::mat3 cam_to_world_R = glm::transpose(world_to_cam_R);
        glm::vec3 cam_to_world_T = -cam_to_world_R * world_to_cam_T;

        viewport_.camera.R = cam_to_world_R;
        viewport_.camera.t = cam_to_world_T;

        // Update pivot point to be in front of camera
        viewport_.camera.updatePivotFromCamera();

        // Save as home position if this is the first camera view
        if (!viewport_.camera.home_saved) {
            viewport_.camera.saveHomePosition();
        }

        // Get camera intrinsics using the proper method
        auto [focal_x, focal_y, center_x, center_y] = cam_data->get_intrinsics();
        const float width = static_cast<float>(cam_data->image_width());
        const float height = static_cast<float>(cam_data->image_height());

        // Calculate vertical FOV using the actual focal length
        const float fov_y_rad = 2.0f * std::atan(height / (2.0f * focal_y));
        const float fov_y_deg = glm::degrees(fov_y_rad);

        // Check for principal point offset (should be near center)
        const float cx_expected = width / 2.0f;
        const float cy_expected = height / 2.0f;

        if (std::abs(center_x - cx_expected) > 1.0f || std::abs(center_y - cy_expected) > 1.0f) {
            LOG_WARN("Camera has non-centered principal point: ({:.1f}, {:.1f}) vs expected ({:.1f}, {:.1f})",
                     center_x, center_y, cx_expected, cy_expected);
        }

        // Set the FOV
        ui::RenderSettingsChanged{
            .sh_degree = std::nullopt,
            .fov = fov_y_deg,
            .scaling_modifier = std::nullopt,
            .antialiasing = std::nullopt,
            .background_color = std::nullopt,
            .equirectangular = std::nullopt}
            .emit();

        // Force immediate camera update
        ui::CameraMove{
            .rotation = viewport_.getRotationMatrix(),
            .translation = viewport_.getTranslation()}
            .emit();

        // Set this as the current camera for GT comparison
        if (services().renderingOrNull()) {
            services().renderingOrNull()->setCurrentCameraId(event.cam_id);
        }

        last_camview_ = event.cam_id;
    }

    void InputController::handleFocusSelection() {
        if (!tool_context_)
            return;
        auto* const sm = tool_context_->getSceneManager();
        if (!sm)
            return;

        const auto& scene = sm->getScene();
        const auto& selected = sm->getSelectedNodeNames();

        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());

        // Accumulate world-space AABB from node's local bounds
        const auto accumulateBounds = [&](const SceneNode* node) {
            glm::vec3 local_min, local_max;
            if (!scene.getNodeBounds(node->id, local_min, local_max))
                return;

            const glm::mat4 world_xform = scene.getWorldTransform(node->id);
            for (int i = 0; i < 8; ++i) {
                const glm::vec3 corner(
                    (i & 1) ? local_max.x : local_min.x,
                    (i & 2) ? local_max.y : local_min.y,
                    (i & 4) ? local_max.z : local_min.z);
                const glm::vec3 world_pt = glm::vec3(world_xform * glm::vec4(corner, 1.0f));
                total_min = glm::min(total_min, world_pt);
                total_max = glm::max(total_max, world_pt);
            }
        };

        if (selected.empty()) {
            // Focus on entire scene (skip group nodes)
            for (const auto* node : scene.getNodes()) {
                if (node->type == NodeType::GROUP || node->type == NodeType::CAMERA_GROUP ||
                    node->type == NodeType::IMAGE_GROUP)
                    continue;
                accumulateBounds(node);
            }
        } else {
            // Focus on selected nodes
            for (const auto& name : selected) {
                if (const auto* node = scene.getNode(name))
                    accumulateBounds(node);
            }
        }

        if (total_min.x <= total_max.x) {
            viewport_.camera.focusOnBounds(total_min, total_max);
            publishCameraMove();
        }
    }

    // Helpers
    bool InputController::isInViewport(double x, double y) const {
        return x >= viewport_bounds_.x &&
               x < viewport_bounds_.x + viewport_bounds_.width &&
               y >= viewport_bounds_.y &&
               y < viewport_bounds_.y + viewport_bounds_.height;
    }

    bool InputController::shouldCameraHandleInput() const {
        // Don't handle if gizmo or splitter is active
        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter) {
            return false;
        }

        // Only block when actively using a GUI widget
        return !ImGui::IsAnyItemActive();
    }

    void InputController::updateCameraSpeed(const bool increase) {
        increase ? viewport_.camera.increaseWasdSpeed() : viewport_.camera.decreaseWasdSpeed();
        ui::SpeedChanged{
            .current_speed = viewport_.camera.getWasdSpeed(),
            .max_speed = viewport_.camera.getMaxWasdSpeed()}
            .emit();
    }

    void InputController::updateZoomSpeed(const bool increase) {
        increase ? viewport_.camera.increaseZoomSpeed() : viewport_.camera.decreaseZoomSpeed();
        ui::ZoomSpeedChanged{
            .zoom_speed = viewport_.camera.getZoomSpeed(),
            .max_zoom_speed = viewport_.camera.getMaxZoomSpeed()}
            .emit();
    }

    void InputController::publishCameraMove() {
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }

        // Throttle event emission
        const auto now = std::chrono::steady_clock::now();
        if (now - last_camera_publish_ >= camera_publish_interval_) {
            ui::CameraMove{
                .rotation = viewport_.getRotationMatrix(),
                .translation = viewport_.getTranslation()}
                .emit();
            last_camera_publish_ = now;
        }
    }

    void InputController::onCameraMovementStart() {
        if (!camera_is_moving_) {
            camera_is_moving_ = true;
            last_camera_movement_time_ = std::chrono::steady_clock::now();

            // Pause training f it's running
            if (services().trainerOrNull() && services().trainerOrNull()->isRunning()) {
                services().trainerOrNull()->pauseTrainingTemporary();
                training_was_paused_by_camera_ = true;
                LOG_INFO("Camera movement detected - pausing training temporarily");
            }
        } else {
            // Update movement time
            last_camera_movement_time_ = std::chrono::steady_clock::now();
        }
    }

    void InputController::onCameraMovementEnd() {
        // Don't immediately resume - let the timeout handle it
        last_camera_movement_time_ = std::chrono::steady_clock::now();
    }

    void InputController::checkCameraMovementTimeout() {
        if (!camera_is_moving_) {
            return;
        }

        auto now = std::chrono::steady_clock::now();
        if (now - last_camera_movement_time_ >= camera_movement_timeout_) {
            camera_is_moving_ = false;

            // Resume training if we paused it
            if (training_was_paused_by_camera_ && services().trainerOrNull() && services().trainerOrNull()->isRunning()) {
                services().trainerOrNull()->resumeTrainingTemporary();
                training_was_paused_by_camera_ = false;
                LOG_INFO("Camera movement stopped - resuming training temporarily");
            }
        }
    }

    int InputController::getModifierKeys() const {
        int mods = 0;
        if (glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
            mods |= GLFW_MOD_CONTROL;
        }
        if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
            mods |= GLFW_MOD_SHIFT;
        }
        if (glfwGetKey(window_, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS) {
            mods |= GLFW_MOD_ALT;
        }
        return mods;
    }

    glm::vec3 InputController::unprojectScreenPoint(double x, double y, float fallback_distance) const {
        if (!services().renderingOrNull()) {
            glm::vec3 forward = viewport_.camera.R * glm::vec3(0, 0, 1);
            return viewport_.camera.t + forward * fallback_distance;
        }

        // Convert window coordinates to viewport-local coordinates
        const float local_x = static_cast<float>(x) - viewport_bounds_.x;
        const float local_y = static_cast<float>(y) - viewport_bounds_.y;

        // Try to get depth from rendering manager using viewport-local coordinates
        const float depth = services().renderingOrNull()->getDepthAtPixel(
            static_cast<int>(local_x), static_cast<int>(local_y));

        // If no valid depth, use fallback distance along view direction
        if (depth < 0.0f) {
            glm::vec3 forward = viewport_.camera.R * glm::vec3(0, 0, 1);
            return viewport_.camera.t + forward * fallback_distance;
        }

        // Use viewport dimensions for unprojection
        const float width = viewport_bounds_.width;
        const float height = viewport_bounds_.height;

        // Pinhole camera unprojection matching the rasterizer
        const float fov_y = glm::radians(services().renderingOrNull()->getFovDegrees());
        const float aspect = width / height;
        const float fov_x = 2.0f * std::atan(std::tan(fov_y / 2.0f) * aspect);

        const float fx = width / (2.0f * std::tan(fov_x / 2.0f));
        const float fy = height / (2.0f * std::tan(fov_y / 2.0f));
        const float cx = width / 2.0f;
        const float cy = height / 2.0f;

        // Point in camera space (using viewport-local coordinates)
        const glm::vec4 view_pos(
            (local_x - cx) * depth / fx,
            (local_y - cy) * depth / fy,
            depth,
            1.0f);

        // Build world-to-camera matrix matching rasterizer: w2c = [R^T | -R^T*t]
        const glm::mat3 R = viewport_.getRotationMatrix();
        const glm::vec3 t = viewport_.getTranslation();
        const glm::mat3 R_inv = glm::transpose(R);
        const glm::vec3 t_inv = -R_inv * t;

        glm::mat4 w2c(1.0f);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                w2c[i][j] = R_inv[i][j];
        w2c[3][0] = t_inv.x;
        w2c[3][1] = t_inv.y;
        w2c[3][2] = t_inv.z;

        // Transform from camera space to world space
        return glm::vec3(glm::inverse(w2c) * view_pos);
    }

    input::ToolMode InputController::getCurrentToolMode() const {
        if (selection_tool_ && selection_tool_->isEnabled())
            return input::ToolMode::SELECTION;
        if (brush_tool_ && brush_tool_->isEnabled())
            return input::ToolMode::BRUSH;
        if (align_tool_ && align_tool_->isEnabled())
            return input::ToolMode::ALIGN;
        // Check GUI tool mode for CropBox (and transform tools)
        if (services().guiOrNull()) {
            const auto gui_tool = services().guiOrNull()->getCurrentToolMode();
            if (gui_tool == gui::panels::ToolType::CropBox)
                return input::ToolMode::CROP_BOX;
            if (gui_tool == gui::panels::ToolType::Translate)
                return input::ToolMode::TRANSLATE;
            if (gui_tool == gui::panels::ToolType::Rotate)
                return input::ToolMode::ROTATE;
            if (gui_tool == gui::panels::ToolType::Scale)
                return input::ToolMode::SCALE;
        }
        return input::ToolMode::GLOBAL;
    }

} // namespace lfs::vis