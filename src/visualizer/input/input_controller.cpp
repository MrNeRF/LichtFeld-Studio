/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_controller.hpp"
#include "core/collision.hpp"
#include "core/error_bus.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/source_site.hpp"
#include "core/splat_data_transform.hpp"
#include "gui/bounds_gizmo.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/string_keys.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/utils/native_file_drag.hpp"
#include "input/gamepad_profiles.hpp"
#include "input/input_router.hpp"
#include "input/input_types.hpp"
#include "input/key_codes.hpp"
#include "input/sdl_coordinate_utils.hpp"
#include "input/sdl_key_mapping.hpp"
#include "io/loader.hpp"
#include "io/splat_path.hpp"
#include "io/video/video_extensions.hpp"
#include "operator/operator_context.hpp"
#include "operator/operator_id.hpp"
#include "operator/operator_registry.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/scale_distance_tool.hpp"
#include "tools/selection_tool.hpp"
#include "tools/tool_base.hpp"
#include "tools/unified_tool_registry.hpp"
#include "training/training_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/gui/panel_registry.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/preferences.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_keyboard.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <limits>
#include <utility>

namespace lfs::vis {

    using namespace lfs::core::events;

    namespace {
        constexpr float kWasdShiftSpeedBonus = 20.0f;
        constexpr double kCameraContextMenuDragThreshold = 4.0;
        constexpr double kCameraFrustumClickThreshold = 5.0;
        namespace string_keys = lichtfeld::Strings;

        // Expand [world_min, world_max] by a node's local AABB transformed to world
        // space. With use_percentile, the splat/point-cloud box is the trimmed
        // (1st/99th percentile) AABB so far-flung floaters don't inflate it; meshes
        // and other geometry fall back to the full node bounds.
        void expandNodeWorldBounds(const core::Scene& scene, const core::SceneNode& node,
                                   glm::vec3& world_min, glm::vec3& world_max,
                                   bool use_percentile = false) {
            glm::vec3 local_min{0.0f}, local_max{0.0f};
            bool have_local = false;
            if (use_percentile) {
                if (node.model && node.model->size() > 0)
                    have_local = core::compute_bounds(*node.model, local_min, local_max, 0.0f, true);
                else if (node.point_cloud && node.point_cloud->size() > 0)
                    have_local = core::compute_bounds(*node.point_cloud, local_min, local_max, 0.0f, true);
            }
            if (!have_local && !scene.getNodeBounds(node.id, local_min, local_max))
                return;

            const glm::mat4 world_xform = scene_coords::nodeVisualizerWorldTransform(scene, node.id);
            for (int i = 0; i < 8; ++i) {
                const glm::vec3 corner(
                    (i & 1) ? local_max.x : local_min.x,
                    (i & 2) ? local_max.y : local_min.y,
                    (i & 4) ? local_max.z : local_min.z);
                const glm::vec3 world_pt = glm::vec3(world_xform * glm::vec4(corner, 1.0f));
                world_min = glm::min(world_min, world_pt);
                world_max = glm::max(world_max, world_pt);
            }
        }

        [[nodiscard]] bool isGeometryContainer(const core::SceneNode& node) {
            return node.type == core::NodeType::GROUP || node.type == core::NodeType::PLY_SEQUENCE ||
                   node.type == core::NodeType::DATASET ||
                   node.type == core::NodeType::CAMERA_GROUP ||
                   node.type == core::NodeType::IMAGE_GROUP;
        }

        // Like expandNodeWorldBounds, but hidden nodes and hidden children of a
        // container add nothing: framing follows what the viewport shows.
        void expandVisibleNodeWorldBounds(const core::Scene& scene, const core::SceneNode& node,
                                          glm::vec3& world_min, glm::vec3& world_max,
                                          const bool use_percentile = false) {
            if (!scene.isNodeEffectivelyVisible(node.id))
                return;
            if (!isGeometryContainer(node)) {
                expandNodeWorldBounds(scene, node, world_min, world_max, use_percentile);
                return;
            }
            for (const auto child_id : node.children) {
                if (const auto* const child = scene.getNodeById(child_id))
                    expandVisibleNodeWorldBounds(scene, *child, world_min, world_max, use_percentile);
            }
        }

        [[nodiscard]] bool isTransformGizmoOverOrUsing() {
            return gui::isBoundsGizmoHovered() ||
                   gui::isBoundsGizmoActive() ||
                   gui::isRotationGizmoHovered() ||
                   gui::isRotationGizmoActive() ||
                   gui::isScaleGizmoHovered() ||
                   gui::isScaleGizmoActive() ||
                   gui::isTranslationGizmoHovered() ||
                   gui::isTranslationGizmoActive();
        }

        [[nodiscard]] bool isTransformGizmoUsing() {
            return gui::isBoundsGizmoActive() ||
                   gui::isRotationGizmoActive() ||
                   gui::isScaleGizmoActive() ||
                   gui::isTranslationGizmoActive();
        }

        [[nodiscard]] bool isEnvironmentMapExtension(const std::string_view ext) {
            return ext == ".hdr" || ext == ".exr";
        }

        [[nodiscard]] bool isSelectionModalAction(const input::Action action) {
            return action == input::Action::CONFIRM_POLYGON ||
                   action == input::Action::CANCEL_POLYGON ||
                   action == input::Action::UNDO_POLYGON_VERTEX;
        }

        // Polygon modals take their keys from the selection key map, where the
        // polygon actions are bound, whatever tool mode is active.
        [[nodiscard]] bool isPolygonModalActive() {
            const auto modal_id = op::operators().activeModalId();
            return modal_id == op::to_string(op::BuiltinOp::SelectionStroke) ||
                   modal_id == op::to_string(op::BuiltinOp::CropPolygonDraw);
        }

        [[nodiscard]] bool isAlignModalAction(const input::Action action) {
            return action == input::Action::CANCEL_POLYGON ||
                   action == input::Action::REMOVE_LAST_POINT ||
                   action == input::Action::DELETE_SELECTED ||
                   action == input::Action::APPLY_ALIGNMENT;
        }

        bool dispatchActionToModal(const input::Action action, const int mods,
                                   const double x, const double y) {
            const auto modal_id = op::operators().activeModalId();
            const bool accepted =
                ((modal_id == op::to_string(op::BuiltinOp::SelectionStroke) ||
                  modal_id == op::to_string(op::BuiltinOp::CropPolygonDraw)) &&
                 isSelectionModalAction(action)) ||
                ((modal_id == op::to_string(op::BuiltinOp::AlignPickPoint) ||
                  modal_id == op::to_string(op::BuiltinOp::ScaleDistancePickPoint)) &&
                 isAlignModalAction(action));
            if (!accepted) {
                return false;
            }

            op::ModalEvent evt{};
            evt.type = op::ModalEvent::Type::ACTION;
            evt.data = op::ActionEvent{action, mods, {x, y}};
            return op::operators().dispatchModalEvent(evt) != op::OperatorResult::PASS_THROUGH;
        }

        void applyDroppedEnvironmentMap(const std::filesystem::path& environment_map_path) {
            auto* const rendering_manager = services().renderingOrNull();
            if (!rendering_manager) {
                LOG_WARN("Ignoring dropped environment map because RenderingManager is not available");
                return;
            }

            auto settings = rendering_manager->getSettings();
            settings.environment_mode = EnvironmentBackgroundMode::Equirectangular;
            settings.environment_map_path = lfs::core::path_to_utf8(environment_map_path);
            if (!rendering_manager->updateSettingsFromUser(settings)) {
                return;
            }
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }

            LOG_INFO("Applied environment map via drag-and-drop: {}",
                     lfs::core::path_to_utf8(environment_map_path.filename()));
        }

        bool dispatchKeyToModals(int key, int scancode, int action, int mods,
                                 double x, double y, const bool over_gui) {
            op::ModalEvent evt{};
            evt.type = op::ModalEvent::Type::KEY;
            evt.over_gui = over_gui;
            evt.data = KeyEvent{key, scancode, action, mods};

            if (op::operators().hasModalOperator()) {
                const auto result = op::operators().dispatchModalEvent(evt);
                if (result != op::OperatorResult::PASS_THROUGH) {
                    return true;
                }
            }

            python::ModalEvent py_evt{};
            py_evt.type = python::ModalEvent::Type::Key;
            py_evt.key = key;
            py_evt.action = action;
            py_evt.mods = mods;
            py_evt.x = x;
            py_evt.y = y;
            py_evt.over_gui = over_gui;

            return python::dispatch_modal_event(py_evt);
        }

        bool dispatchMouseButtonToModals(int button, int action, int mods,
                                         double x, double y, const bool over_gui) {
            op::ModalEvent evt{};
            evt.type = op::ModalEvent::Type::MOUSE_BUTTON;
            evt.over_gui = over_gui;
            evt.data = MouseButtonEvent{button, action, mods, {x, y}};

            if (op::operators().hasModalOperator()) {
                const auto result = op::operators().dispatchModalEvent(evt);
                if (result != op::OperatorResult::PASS_THROUGH) {
                    return true;
                }
            }

            python::ModalEvent py_evt{};
            py_evt.type = python::ModalEvent::Type::MouseButton;
            py_evt.button = button;
            py_evt.action = action;
            py_evt.mods = mods;
            py_evt.x = x;
            py_evt.y = y;
            py_evt.over_gui = over_gui;

            return python::dispatch_modal_event(py_evt);
        }

        bool dispatchMouseMoveToModals(double x, double y, double delta_x, double delta_y,
                                       [[maybe_unused]] int mods, const bool over_gui) {
            op::ModalEvent evt{};
            evt.type = op::ModalEvent::Type::MOUSE_MOVE;
            evt.over_gui = over_gui;
            evt.data = MouseMoveEvent{{x, y}, {delta_x, delta_y}};

            if (op::operators().hasModalOperator()) {
                const auto result = op::operators().dispatchModalEvent(evt);
                if (result != op::OperatorResult::PASS_THROUGH) {
                    return true;
                }
            }

            python::ModalEvent py_evt{};
            py_evt.type = python::ModalEvent::Type::MouseMove;
            py_evt.x = x;
            py_evt.y = y;
            py_evt.delta_x = delta_x;
            py_evt.delta_y = delta_y;
            py_evt.over_gui = over_gui;

            return python::dispatch_modal_event(py_evt);
        }

        bool dispatchScrollToModals(double xoff, double yoff, double x, double y,
                                    [[maybe_unused]] int mods, const bool over_gui) {
            op::ModalEvent evt{};
            evt.type = op::ModalEvent::Type::MOUSE_SCROLL;
            evt.over_gui = over_gui;
            evt.data = MouseScrollEvent{xoff, yoff};

            if (op::operators().hasModalOperator()) {
                const auto result = op::operators().dispatchModalEvent(evt);
                if (result != op::OperatorResult::PASS_THROUGH) {
                    return true;
                }
            }

            python::ModalEvent py_evt{};
            py_evt.type = python::ModalEvent::Type::Scroll;
            py_evt.scroll_x = xoff;
            py_evt.scroll_y = yoff;
            py_evt.x = x;
            py_evt.y = y;
            py_evt.over_gui = over_gui;

            return python::dispatch_modal_event(py_evt);
        }

        std::optional<ToolType> ownerToolForActivationAction(const input::Action action) {
            switch (action) {
            case input::Action::TOOL_SELECT:
            case input::Action::SELECT_MODE_CENTERS:
            case input::Action::SELECT_MODE_RECTANGLE:
            case input::Action::SELECT_MODE_POLYGON:
            case input::Action::SELECT_MODE_LASSO:
            case input::Action::SELECT_MODE_RINGS:
            case input::Action::SELECT_MODE_COLOR:
            case input::Action::SELECT_MODE_BOX:
            case input::Action::SELECT_MODE_SPHERE:
                return ToolType::Selection;
            case input::Action::TOOL_TRANSLATE:
                return ToolType::Translate;
            case input::Action::TOOL_ROTATE:
                return ToolType::Rotate;
            case input::Action::TOOL_SCALE:
                return ToolType::Scale;
            case input::Action::TOOL_MIRROR:
                return ToolType::Mirror;
            case input::Action::TOOL_ALIGN:
                return ToolType::Align;
            case input::Action::TOOL_SCALE_DISTANCE:
                return ToolType::ScaleDistance;
            default:
                return std::nullopt;
            }
        }

        bool applySelectionModeShortcut(const input::Action action, gui::GuiManager* gui) {
            if (!gui)
                return false;

            SelectionSubMode submode = SelectionSubMode::Centers;
            const char* submode_id = nullptr;
            switch (action) {
            case input::Action::SELECT_MODE_CENTERS:
                submode = SelectionSubMode::Centers;
                submode_id = "centers";
                break;
            case input::Action::SELECT_MODE_RECTANGLE:
                submode = SelectionSubMode::Rectangle;
                submode_id = "rectangle";
                break;
            case input::Action::SELECT_MODE_POLYGON:
                submode = SelectionSubMode::Polygon;
                submode_id = "polygon";
                break;
            case input::Action::SELECT_MODE_LASSO:
                submode = SelectionSubMode::Lasso;
                submode_id = "lasso";
                break;
            case input::Action::SELECT_MODE_RINGS:
                submode = SelectionSubMode::Rings;
                submode_id = "rings";
                break;
            case input::Action::SELECT_MODE_COLOR:
                submode = SelectionSubMode::Color;
                submode_id = "color";
                break;
            case input::Action::SELECT_MODE_BOX:
                submode = SelectionSubMode::Box;
                submode_id = "box";
                break;
            case input::Action::SELECT_MODE_SPHERE:
                submode = SelectionSubMode::Sphere;
                submode_id = "sphere";
                break;
            default:
                return false;
            }

            gui->gizmo().setSelectionSubMode(submode);
            UnifiedToolRegistry::instance().setActiveSubmode(submode_id);
            return true;
        }

        bool handleToolControlActivationShortcut(const input::Action action, gui::GuiManager* gui) {
            const auto tool = ownerToolForActivationAction(action);
            if (!tool) {
                return false;
            }

            lfs::core::events::tools::SetToolbarTool{.tool_mode = static_cast<int>(*tool)}.emit();
            (void)applySelectionModeShortcut(action, gui);
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }
            return true;
        }

        input::Action resolveCrossToolActivationShortcut(const input::InputBindings& bindings,
                                                         const input::ToolMode current_mode,
                                                         const int physical_key,
                                                         const int logical_key,
                                                         const int mods) {
            for (const auto mode : input::kAllToolModes) {
                if (mode == current_mode) {
                    continue;
                }
                const auto candidate = bindings.getActionForShortcut(
                    mode, physical_key, logical_key, mods);
                if (ownerToolForActivationAction(candidate).has_value()) {
                    return candidate;
                }
            }
            return input::Action::NONE;
        }

        [[nodiscard]] bool shouldDeferClickActionForDrag(const input::Action action) {
            switch (action) {
            case input::Action::CAMERA_SET_PIVOT:
                return true;
            default:
                return false;
            }
        }

        void resetViewportToHome(Viewport& target_viewport, const Viewport& primary_viewport) {
            target_viewport.camera.resetToHome();

            auto* const rendering = services().renderingOrNull();
            if (!rendering)
                return;

            auto settings = rendering->getSettings();
            if (!settings.orthographic)
                return;

            constexpr float kMinDistance = 0.01f;
            constexpr float kMinScale = 1.0f;
            constexpr float kMaxScale = 10000.0f;
            constexpr float kDefaultScale = 100.0f;
            const float distance = glm::length(target_viewport.camera.home_pivot - target_viewport.camera.home_t);
            float scale = kDefaultScale;
            if (target_viewport.windowSize.y > 0 && distance > kMinDistance) {
                const float vfov = lfs::rendering::focalLengthToVFov(settings.focal_length_mm);
                const float half_tan_fov = std::tan(glm::radians(vfov) * 0.5f);
                scale = std::clamp(static_cast<float>(target_viewport.windowSize.y) /
                                       (2.0f * distance * half_tan_fov),
                                   kMinScale, kMaxScale);
            }

            if (&target_viewport == &primary_viewport) {
                settings.ortho_scale = scale;
                if (target_viewport.ortho_scale_override)
                    target_viewport.ortho_scale_override = scale;
                rendering->updateSettings(settings);
            } else {
                target_viewport.ortho_scale_override = scale;
            }
        }
    } // namespace

    InputController* InputController::instance_ = nullptr;

    InputController::InputController(SDL_Window* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
        go_to_cam_view_handler_id_ =
            cmd::GoToCamView::when([this](const auto& e) { handleGoToCamView(e); });

        reset_camera_handler_id_ = cmd::ResetCamera::when([this](const auto&) {
            resetViewportToHome(viewport_, viewport_);
            publishCameraMove();
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }
        });

        dataset_load_completed_handler_id_ = state::DatasetLoadCompleted::when([this](const auto& e) {
            if (e.success) {
                resetViewportToHome(viewport_, viewport_);
                publishCameraMove();
            }
        });

        split_toggle_handler_id_ = cmd::ToggleSplitView::when([this](const auto&) {
            suspendGamepadNavigation();
            clearViewportDragState();
            clearWasdMomentumViewport();
            resetSplitPanelFocus();
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }
        });
        independent_split_toggle_handler_id_ = cmd::ToggleIndependentSplitView::when([this](const auto&) {
            suspendGamepadNavigation();
            clearViewportDragState();
            clearWasdMomentumViewport();
            resetSplitPanelFocus();
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }
        });
        gt_comparison_toggle_handler_id_ = cmd::ToggleGTComparison::when([this](const auto&) {
            suspendGamepadNavigation();
            clearViewportDragState();
            clearWasdMomentumViewport();
            resetSplitPanelFocus();
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }
        });
        scene_cleared_handler_id_ = state::SceneCleared::when([this](const auto&) {
            suspendGamepadNavigation();
            clearViewportDragState();
            clearWasdMomentumViewport();
            scene_extent_ = 0.0f;
            depth_range_initialized_ = false;
            resetSplitPanelFocus();
        });
        scene_loaded_handler_id_ = state::SceneLoaded::when([this](const auto&) {
            suspendGamepadNavigation();
            clearViewportDragState();
            clearWasdMomentumViewport();
            scene_extent_ = 0.0f;
            depth_range_initialized_ = false;
            resetSplitPanelFocus();
        });

        window_focus_lost_handler_id_ = internal::WindowFocusLost::when([this](const auto&) {
            pauseGamepadNavigation();
            cancelValueDrag(false);
            drag_mode_ = DragMode::None;
            syncLookPointerLock(false);
            clearSelectedCameraContextMenuGesture();
            press_selected_camera_frustum_ = false;
            pressed_camera_frustum_id_ = -1;
            pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;
            std::fill(std::begin(keys_movement_), std::end(keys_movement_), false);
            clearWasdMomentumViewport();
            hovered_camera_id_ = -1;
        });
    }

    InputController::~InputController() {
        auto unsubscribe = [](const auto event_tag, std::size_t& handler_id) {
            if (handler_id == 0)
                return;
            ::lfs::event::EventBridge::instance().unsubscribe(typeid(decltype(event_tag)), handler_id);
            handler_id = 0;
        };

        unsubscribe(cmd::GoToCamView{}, go_to_cam_view_handler_id_);
        unsubscribe(cmd::ResetCamera{}, reset_camera_handler_id_);
        unsubscribe(state::DatasetLoadCompleted{}, dataset_load_completed_handler_id_);
        unsubscribe(cmd::ToggleSplitView{}, split_toggle_handler_id_);
        unsubscribe(cmd::ToggleIndependentSplitView{}, independent_split_toggle_handler_id_);
        unsubscribe(cmd::ToggleGTComparison{}, gt_comparison_toggle_handler_id_);
        unsubscribe(state::SceneCleared{}, scene_cleared_handler_id_);
        unsubscribe(state::SceneLoaded{}, scene_loaded_handler_id_);
        unsubscribe(internal::WindowFocusLost{}, window_focus_lost_handler_id_);

        if (instance_ == this) {
            instance_ = nullptr;
        }

        // Clean up cursor resources
        if (resize_cursor_) {
            SDL_DestroyCursor(resize_cursor_);
            resize_cursor_ = nullptr;
        }
        if (hand_cursor_) {
            SDL_DestroyCursor(hand_cursor_);
            hand_cursor_ = nullptr;
        }

        // Reset cursor to default before destruction
        if (window_ && current_cursor_ != CursorType::Default) {
            SDL_SetCursor(SDL_GetDefaultCursor());
        }
    }

    void InputController::setVrViewing(const bool enabled) {
        vr_viewing_ = enabled;
        clearViewportDragState();
        clearWasdMomentumViewport();
        viewport_.camera.clearTransientMotion();
        onWindowFocusLost();
        std::fill(std::begin(keys_movement_), std::end(keys_movement_), false);
    }

    glm::vec3 InputController::vrMovement() const {
        const auto axis = [this](const int positive, const int negative) {
            return static_cast<float>(keys_movement_[positive]) - static_cast<float>(keys_movement_[negative]);
        };
        return {axis(3, 1), axis(4, 5), axis(2, 0)}; // fwd, left, back, right, up, down
    }

    void InputController::handleVrKey(const int physical_key, const int logical_key, const int action,
                                      const int mods, const bool gui_owned) {
        if (action == input::ACTION_RELEASE) {
            armPhysicalMovementKey(physical_key, false);
            return;
        }
        if (gui_owned)
            return;
        if (action == input::ACTION_PRESS) {
            switch (bindings_.getActionForKey(input::ToolMode::VR, logical_key, mods)) {
            case input::Action::EXIT_VR: cmd::SetVrActive{.active = false}.emit(); return;
            case input::Action::VR_RECENTER: cmd::RecenterVr{}.emit(); return;
            case input::Action::VR_SNAP_TURN_LEFT: cmd::SnapTurnVr{.direction = -1}.emit(); return;
            case input::Action::VR_SNAP_TURN_RIGHT: cmd::SnapTurnVr{.direction = 1}.emit(); return;
            default: break;
            }
        }
        if ((mods & (input::KEYMOD_CTRL | input::KEYMOD_ALT | input::KEYMOD_SUPER)) == 0)
            armPhysicalMovementKey(physical_key, true);
    }

    void InputController::initialize() {
        instance_ = this;
        try {
            gamepad_settings_ = input::GamepadNavigationSettings::fromJson(nlohmann::json::parse(UserPreferences::instance().gamepadNavigation()));
        } catch (const std::exception& error) {
            gamepad_configuration_error_ = error.what();
            LOG_WARN("Gamepad configuration was not loaded: {}", error.what());
        }
        applyNewProjectNavigationDefaults();

        // The GUI is initialized before this controller exists.
        if (const auto mode = cameraNavigationModeFromName(loadCameraNavigationPreference()); mode && *mode != CameraNavigationMode::Walk) {
            setCameraNavigationMode(*mode);
        }
        setCameraViewSnapEnabled(loadCameraViewSnapPreference());

        // Get initial mouse position
        float fx, fy;
        input::mouseStateInPixels(window_, &fx, &fy);
        last_mouse_pos_ = {fx, fy};

        // Initialize frame timer
        last_frame_time_ = std::chrono::high_resolution_clock::now();

        // Create cursors
        resize_cursor_ = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_EW_RESIZE);
        hand_cursor_ = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_POINTER);

        refreshMovementKeyCache();
        bindings_.setOnBindingsChanged([this]() { refreshMovementKeyCache(); });
    }

    void InputController::applySplitterCursorOverride() const {
        if (current_cursor_ == CursorType::Resize && resize_cursor_) {
            SDL_SetCursor(resize_cursor_);
        }
    }

    void InputController::refreshMovementKeyCache() {
        for (const auto mode : input::kAllToolModes) {
            auto& mk = movement_keys_per_mode_[static_cast<size_t>(mode)];
            mk.forward = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_FORWARD, mode);
            mk.backward = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_BACKWARD, mode);
            mk.left = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_LEFT, mode);
            mk.right = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_RIGHT, mode);
            mk.up = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_UP, mode);
            mk.down = bindings_.getKeyForAction(input::Action::CAMERA_MOVE_DOWN, mode);
        }
    }

    const InputController::MovementKeys& InputController::currentMovementKeys() const {
        return movement_keys_per_mode_[static_cast<size_t>(getCurrentToolMode())];
    }

    const char* InputController::cameraNavigationModeName(const CameraNavigationMode mode) {
        switch (mode) {
        case CameraNavigationMode::Orbit: return "orbit";
        case CameraNavigationMode::Trackball: return "trackball";
        case CameraNavigationMode::FPV: return "fpv";
        case CameraNavigationMode::Drone: return "drone";
        case CameraNavigationMode::Walk: return "walk";
        }
        return "orbit";
    }

    std::optional<InputController::CameraNavigationMode>
    InputController::cameraNavigationModeFromName(const std::string_view name) {
        if (name == "orbit")
            return CameraNavigationMode::Orbit;
        if (name == "trackball" || name == "turntable")
            return CameraNavigationMode::Trackball;
        if (name == "fpv" || name == "fly")
            return CameraNavigationMode::FPV;
        if (name == "drone")
            return CameraNavigationMode::Drone;
        if (name == "walk")
            return CameraNavigationMode::Walk;
        return std::nullopt;
    }

    core::CapsuleQuery InputController::walkQuery() const {
        std::vector<core::Scene::VisibleMesh> colliders;
        if (const auto* manager = services().sceneOrNull()) {
            const auto& scene = manager->getScene();
            for (const auto* node : scene.getNodes())
                if (node && node->type == core::NodeType::COLLISION && node->collision)
                    colliders.push_back({.mesh = nullptr, .transform = lfs::rendering::dataWorldTransformToVisualizerWorld(scene.getWorldTransform(node->id)), .collision = node->collision});
        }
        if (colliders.empty())
            throw core::collisionError("collision.walk_no_collision", "No collision nodes are loaded");
        return [colliders = std::move(colliders)](glm::vec3 feet, glm::vec3 up, float height, float radius) {
            return std::ranges::any_of(colliders, [&](const auto& collider) {
                return core::collisionCapsuleIntersects(*collider.collision, collider.transform, feet, up, height, radius);
            });
        };
    }

    void InputController::ensureWalkViewport(Viewport& target) {
        if (walk_viewport_ == &target && walk_state_.presentation == target.camera.t)
            return;
        // Validate the handoff before replacing the previous owner's physics.
        core::WalkState candidate{.eye = target.camera.t, .presentation = target.camera.t};
        core::settleWalk(candidate, walk_settings, target.camera.navigationUp(), walkQuery());
        walk_state_ = candidate;
        walk_viewport_ = &target;
    }

    void InputController::setWalkSettings(core::WalkSettings settings) {
        settings.validate();
        if (camera_navigation_mode_ == CameraNavigationMode::Walk) {
            auto& target = walk_viewport_ ? *walk_viewport_ : activeKeyboardViewport();
            ensureWalkViewport(target);
            auto& camera = target.camera;
            const auto up = camera.navigationUp();
            const auto feet = walk_state_.feet;
            if (walkQuery()(feet, up, settings.height, settings.radius))
                throw core::collisionError("collision.walk_blocked", "The edited Walk capsule does not fit");
            walk_state_.eye = feet + up * settings.height;
            walk_state_.presentation = walk_state_.eye;
            camera.pivot += walk_state_.eye - camera.t;
            camera.t = walk_state_.eye;
        }
        pauseGamepadNavigation();
        walk_settings = settings;
        publish_viewport_toolbar_generation();
    }

    bool InputController::setCameraNavigationMode(const CameraNavigationMode mode) {
        return setCameraNavigationModeImpl(mode, true);
    }

    bool InputController::setCameraNavigationModeImpl(const CameraNavigationMode mode, const bool report_failure, Viewport* target) {
        auto& navigation_viewport = target ? *target : activeKeyboardViewport();
        if (camera_navigation_mode_ == mode)
            return true;

        if (mode == CameraNavigationMode::Walk) {
            auto& camera = navigation_viewport.camera;
            core::WalkState candidate{.eye = camera.t, .presentation = camera.t};
            try {
                core::settleWalk(candidate, walk_settings, camera.navigationUp(), walkQuery());
            } catch (const std::exception& error) {
                if (!report_failure)
                    return false;
                const auto message = core::localizedCollisionError(error);
                if (walk_error != message) {
                    lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                        .error = lfs::make_error({.code = lfs::ErrorCode::InvalidArgument,
                                                  .domain = lfs::ErrorDomain::App,
                                                  .severity = lfs::Severity::Warning,
                                                  .user_message = message,
                                                  .detail = error.what(),
                                                  .detection = LFS_SOURCE_SITE_CURRENT()})
                                     .with_context("enter_walk", LFS_SOURCE_SITE_CURRENT()),
                        .surface = lfs::ErrorSurface::Toast,
                        .actions = {},
                        .operation_id = lfs::OperationId::generate(),
                    });
                    walk_error = message;
                    publish_viewport_toolbar_generation();
                }
                python::request_redraw();
                return false;
            }
            walk_state_ = candidate;
            walk_viewport_ = &navigation_viewport;
            camera.t = walk_state_.eye;
            camera.clearTransientMotion();
            walk_error.clear();
        }
        clearViewportDragState();
        pauseGamepadNavigation();
        const bool leaving_drone = camera_navigation_mode_ == CameraNavigationMode::Drone;
        camera_navigation_mode_ = mode;
        if (mode != CameraNavigationMode::Walk)
            walk_viewport_ = nullptr;
        // The pivot is left untouched: orbit modes work with any pivot and FPV
        // re-seeds it on every look drag, so switching modes must not discard a
        // user-set pivot.
        if (leaving_drone) {
            Viewport* finished_viewport = nullptr;
            if (wasd_momentum_viewport_) {
                wasd_momentum_viewport_->camera.finishDrone();
                publishCameraMove(wasd_momentum_viewport_);
                finished_viewport = wasd_momentum_viewport_;
                wasd_momentum_viewport_ = nullptr;
            }
            auto& keyboard_viewport = navigation_viewport;
            if (&keyboard_viewport != finished_viewport) {
                keyboard_viewport.camera.finishDrone();
                publishCameraMove(&keyboard_viewport);
            }
        } else if (mode == CameraNavigationMode::Drone) {
            clearWasdMomentumViewport();
            auto& keyboard_viewport = navigation_viewport;
            keyboard_viewport.camera.enterDrone();
            publishCameraMove(&keyboard_viewport);
        }
        publish_viewport_toolbar_generation();
        python::request_redraw();
        return true;
    }

    void InputController::applyNewProjectNavigationDefaults() {
        auto& preferences = UserPreferences::instance();
        setNavigationUpAxis(navigationUpAxisFromName(preferences.newProjectNavigationUpAxis()).value_or(NavigationUpAxis::LegacyY),
                            preferences.alignNewProjectGrid());
    }

    void InputController::setNavigationUpAxis(const NavigationUpAxis axis, const bool align_grid) {
        pauseGamepadNavigation();
        clearViewportDragState();
        clearWasdMomentumViewport();
        orbit_coast_viewport_ = nullptr;
        pan_coast_viewport_ = nullptr;
        viewport_.camera.setNavigationUpAxis(axis);
        if (viewer_) {
            if (auto* rendering = viewer_->getRenderingManager()) {
                rendering->projectSecondaryViewport().camera.setNavigationUpAxis(axis);
                if (align_grid) {
                    const int plane = axis == NavigationUpAxis::DataZ ? 2 : 1;
                    rendering->setGridPlaneForPanel(SplitViewPanelId::Left, plane);
                    rendering->setGridPlaneForPanel(SplitViewPanelId::Right, plane);
                }
            }
        }
        auto& generation = app_store().render_settings_generation;
        generation.set(generation.get() + 1);
    }

    void InputController::applyNavigationSpeedPreferences(const float zoom_speed,
                                                          const float navigation_speed) {
        viewport_.camera.setZoomSpeed(zoom_speed);
        viewport_.camera.setWasdSpeed(navigation_speed);
        if (viewer_) {
            if (auto* const rendering = viewer_->getRenderingManager()) {
                auto& secondary = rendering->projectSecondaryViewport();
                secondary.camera.setZoomSpeed(zoom_speed);
                secondary.camera.setWasdSpeed(navigation_speed);
            }
        }
    }

    void InputController::onWindowFocusLost() {
        pauseGamepadNavigation();
        if (current_cursor_ != CursorType::Default) {
            SDL_SetCursor(SDL_GetDefaultCursor());
            current_cursor_ = CursorType::Default;
        }
        held_keys_.clear();
        camera_browse_repeat_.stop();
        selection_drag_op_.reset();
        pending_click_drag_ = {};
        forced_mouse_press_action_ = input::Action::NONE;
        text_input_viewport_click_button_ = -1;
        is_node_rect_dragging_ = false;
        node_rect_button_ = -1;
        node_rect_modifiers_ = input::MODIFIER_NONE;
        node_point_pick_enabled_ = false;
        node_rect_select_enabled_ = false;
    }

    bool InputController::hasViewportKeyboardFocus() const {
        return input_router_ && input_router_->isViewportKeyboardFocused();
    }

    bool InputController::isKeyPressed(int app_key) const {
        const SDL_Scancode scancode = input::appKeyToSdlScancode(app_key);
        if (scancode == SDL_SCANCODE_UNKNOWN)
            return false;
        const bool* state = SDL_GetKeyboardState(nullptr);
        assert(scancode < SDL_SCANCODE_COUNT);
        return state[scancode];
    }

    bool InputController::isMouseButtonPressed(int app_button) const {
        SDL_MouseButtonFlags buttons = SDL_GetMouseState(nullptr, nullptr);
        switch (app_button) {
        case static_cast<int>(input::AppMouseButton::LEFT): return (buttons & SDL_BUTTON_LMASK) != 0;
        case static_cast<int>(input::AppMouseButton::RIGHT): return (buttons & SDL_BUTTON_RMASK) != 0;
        case static_cast<int>(input::AppMouseButton::MIDDLE): return (buttons & SDL_BUTTON_MMASK) != 0;
        default: return false;
        }
    }

    int InputController::getModifierKeys() const {
        SDL_Keymod m = SDL_GetModState();
        int mods = 0;
        if (m & SDL_KMOD_CTRL)
            mods |= input::KEYMOD_CTRL;
        if (m & SDL_KMOD_SHIFT)
            mods |= input::KEYMOD_SHIFT;
        if (m & SDL_KMOD_ALT)
            mods |= input::KEYMOD_ALT;
        if (m & SDL_KMOD_MODE)
            mods |= input::KEYMOD_ALTGR;
        if (m & SDL_KMOD_GUI)
            mods |= input::KEYMOD_SUPER;
        return mods;
    }

    bool InputController::hasViewportCursorOverride() const {
        return current_cursor_ == CursorType::Resize;
    }

    std::optional<input::SelectionOp> InputController::selectionDragOperation() const {
        if (!selection_drag_op_ || !op::operators().hasModalOperator())
            return std::nullopt;
        return selection_drag_op_;
    }

    void InputController::refreshSplitDividerCache() const {
        auto* const rendering = services().renderingOrNull();
        if (!rendering) {
            cached_split_divider_screen_x_.reset();
            return;
        }
        cached_split_divider_screen_x_ = rendering->getSplitDividerScreenX(
            {viewport_bounds_.x, viewport_bounds_.y},
            {viewport_bounds_.width, viewport_bounds_.height});
    }

    bool InputController::isNearSplitter(double x, double y) const {
        auto* const rendering = services().renderingOrNull();
        if (!rendering) {
            return false;
        }

        const glm::ivec2 viewport_size{
            std::max(static_cast<int>(std::lround(viewport_bounds_.width)), 0),
            std::max(static_cast<int>(std::lround(viewport_bounds_.height)), 0)};
        const auto content_bounds = rendering->getContentBounds(viewport_size);
        if (content_bounds.width <= 0.0f || content_bounds.height <= 0.0f) {
            return false;
        }

        if (!cached_split_divider_screen_x_) {
            refreshSplitDividerCache();
        }
        if (!cached_split_divider_screen_x_) {
            return false;
        }

        constexpr float SPLITTER_HIT_HALF_WIDTH = 12.0f;
        const float content_left = viewport_bounds_.x + content_bounds.x;
        const float content_top = viewport_bounds_.y + content_bounds.y;
        return x >= content_left &&
               x < content_left + content_bounds.width &&
               y >= content_top &&
               y < content_top + content_bounds.height &&
               std::abs(x - *cached_split_divider_screen_x_) < SPLITTER_HIT_HALF_WIDTH;
    }

    // Core handlers
    void InputController::handleMouseButton(int button, int action, double x, double y) {
        if (vr_viewing_)
            return;
        if (action == input::ACTION_PRESS) {
            if (input_router_ && input_router_->pointerTarget(x, y) == input::InputTarget::Viewport)
                pauseGamepadNavigation();
            gamepad_click_candidate_ = button == static_cast<int>(input::AppMouseButton::LEFT) &&
                                       hasViewportKeyboardFocus() && isInViewport(x, y) && input_router_ &&
                                       input_router_->pointerTarget(x, y) == input::InputTarget::Viewport;
            gamepad_press_pos_ = {x, y};
        } else if (action == input::ACTION_RELEASE && button == static_cast<int>(input::AppMouseButton::LEFT)) {
            if (gamepad_click_candidate_ && hasViewportKeyboardFocus() && isInViewport(x, y) &&
                glm::length(glm::dvec2(x, y) - gamepad_press_pos_) < 4.0) {
                if (const auto interaction = resolvePanelInteraction(x, y); interaction && interaction->valid()) {
                    stopGamepadMotion();
                    gamepad_panel_ = interaction->panel;
                    ++gamepad_click_epoch_;
                }
            }
            gamepad_click_candidate_ = false;
        }
        decayHeldDragMomentum();
        LOG_PERF("InputController::handleMouseButton button={} action={} pos=({},{}) drag_mode={}",
                 button, action, x, y, static_cast<int>(drag_mode_));
        auto* gui = services().guiOrNull();
        const bool is_left_button = button == static_cast<int>(input::AppMouseButton::LEFT);
        // A left release ends a value drag wherever the hidden pointer is.
        if (action == input::ACTION_RELEASE && is_left_button)
            endValueDrag();
        if (action == input::ACTION_PRESS && button == static_cast<int>(input::AppMouseButton::RIGHT) &&
            isValueDragActive()) {
            cancelValueDrag(true);
            return;
        }
        const bool press_consumed_camera_frustum =
            action == input::ACTION_RELEASE &&
            is_left_button &&
            press_selected_camera_frustum_;
        const int pressed_camera_frustum_id = pressed_camera_frustum_id_;
        const glm::dvec2 pressed_camera_frustum_pos = pressed_camera_frustum_pos_;
        const int pressed_camera_frustum_modifiers = pressed_camera_frustum_modifiers_;
        if (action == input::ACTION_PRESS && is_left_button)
            press_selected_camera_frustum_ = false;
        if (action == input::ACTION_PRESS && is_left_button)
            pressed_camera_frustum_id_ = -1;
        if (action == input::ACTION_PRESS && is_left_button)
            pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;
        if (action == input::ACTION_RELEASE && is_left_button) {
            press_selected_camera_frustum_ = false;
            pressed_camera_frustum_id_ = -1;
            pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;
        }
        const bool over_gizmo = gui && gui->gizmo().isPositionInViewportGizmo(x, y);
        const bool over_gui = isPointerOverBlockingUi(x, y);
        const bool over_gui_hover = isPointerOverUiHover(x, y);
        const bool over_transform_gizmo = isTransformGizmoOverOrUsing();
        const int mods = getModifierKeys();

        if (text_input_viewport_click_button_ == button &&
            action == input::ACTION_RELEASE) {
            text_input_viewport_click_button_ = -1;
            return;
        }

        // Consume all mouse events while pie menu is open
        if (gui && gui->gizmo().isPieMenuOpen()) {
            if (action == input::ACTION_PRESS && button == static_cast<int>(input::AppMouseButton::LEFT)) {
                gui->gizmo().onPieMenuClick({static_cast<float>(x), static_cast<float>(y)});
            }
            return;
        }

        // Forward to GUI for mouse capture (rebinding)
        if (gui && gui->isCapturingInput()) {
            if (action == input::ACTION_PRESS) {
                std::optional<int> chord_key;
                if (!held_keys_.empty()) {
                    chord_key = held_keys_.back();
                }
                gui->captureMouseButton(button, mods, x, y, chord_key);
            } else if (action == input::ACTION_RELEASE) {
                gui->captureMouseButtonRelease(button);
            }
            return;
        }

        const bool wants_text_input = input_router_
                                          ? input_router_->isTextInputActive()
                                          : gui::guiFocusState().want_text_input;
        if (action == input::ACTION_PRESS &&
            wants_text_input &&
            !over_gui &&
            isInViewport(x, y)) {
            text_input_viewport_click_button_ = button;
            return;
        }

        const bool selection_pointer_blocked =
            isPolygonModalActive() && (over_gui || (!input_router_ && !isInViewport(x, y)));
        if (!selection_pointer_blocked && action == input::ACTION_PRESS) {
            const auto mouse_btn = static_cast<input::MouseButton>(button);
            const auto tool_mode = isPolygonModalActive() ? input::ToolMode::SELECTION : getCurrentToolMode();
            auto modal_action = bindings_.getActionForMouseButton(tool_mode, mouse_btn, mods, false);
            if (!isSelectionModalAction(modal_action)) {
                modal_action = bindings_.getActionForDrag(tool_mode, mouse_btn, mods, held_keys_);
            }
            if (dispatchActionToModal(modal_action, mods, x, y)) {
                return;
            }
        }

        // Dispatch to modal operators first - if consumed, don't continue
        if (!selection_pointer_blocked &&
            dispatchMouseButtonToModals(button, action, mods, x, y, over_gui_hover)) {
            return;
        }

        if (is_left_button &&
            action == input::ACTION_PRESS &&
            isInViewport(x, y) &&
            isNearSplitter(x, y)) {
            if (isIndependentSplitViewActive()) {
                focusSplitPanel(splitPanelForScreenX(x));
            }
            if (auto* const rendering = services().renderingOrNull()) {
                drag_mode_ = DragMode::Splitter;
                splitter_start_pos_ = rendering->getSplitPosition();
                splitter_start_x_ = x;
                SDL_SetCursor(resize_cursor_);
                current_cursor_ = CursorType::Resize;
                LOG_TRACE("Started splitter drag");
                return;
            }
        }

        if (!over_gui &&
            is_left_button &&
            action == input::ACTION_PRESS) {
            if (isInViewport(x, y) && isIndependentSplitViewActive()) {
                const SplitViewPanelId panel = splitPanelForScreenX(x);
                focusSplitPanel(panel);
                if (selection_tool_) {
                    selection_tool_->setFilterPanel(panel);
                }
            }

            // Check for double-click on camera frustum
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = std::chrono::duration<double>(now - last_click_time_).count();
            double dist = glm::length(glm::dvec2(x, y) - last_click_pos_);

            bool is_double_click = (time_since_last < DOUBLE_CLICK_TIME &&
                                    dist < DOUBLE_CLICK_DISTANCE);

            // If we have a hovered camera, check for double-click. Defer
            // single-click selection until release so orbit drags that start
            // over dataset image frustums do not change the node selection.
            const auto tool_mode = getCurrentToolMode();
            const bool allow_camera_frustum_pick =
                tool_mode == input::ToolMode::GLOBAL ||
                tool_mode == input::ToolMode::SELECTION;
            const bool camera_selection_modifier =
                (mods & (input::KEYMOD_CTRL | input::KEYMOD_SHIFT)) != 0;
            if (allow_camera_frustum_pick &&
                hovered_camera_id_ >= 0 && !over_gizmo && !over_transform_gizmo) {
                if (!camera_selection_modifier && is_double_click &&
                    hovered_camera_id_ == last_clicked_camera_id_) {
                    cmd::GoToCamView{.cam_id = hovered_camera_id_}.emit();
                    if (auto* const viewer = lfs::python::get_visualizer()) {
                        viewer->projectMarkSessionChapterCommitted("VIEW");
                    }

                    // Reset click tracking to prevent triple-click
                    last_click_time_ = std::chrono::steady_clock::time_point();
                    last_click_pos_ = {-1000, -1000}; // Far away position
                    last_clicked_camera_id_ = -1;
                    return;
                }
                last_click_time_ = now;
                last_click_pos_ = {x, y};
                last_clicked_camera_id_ = hovered_camera_id_;
                pressed_camera_frustum_id_ = hovered_camera_id_;
                pressed_camera_frustum_pos_ = {x, y};
                pressed_camera_frustum_modifiers_ = mods;
                press_selected_camera_frustum_ = true;
            } else {
                last_click_time_ = std::chrono::steady_clock::time_point();
                last_click_pos_ = {-1000, -1000};
                last_clicked_camera_id_ = -1;
            }
        }

        if (action == input::ACTION_RELEASE && drag_mode_ == DragMode::Splitter) {
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }
            drag_mode_ = DragMode::None;
            SDL_SetCursor(SDL_GetDefaultCursor());
            current_cursor_ = CursorType::Default;
            LOG_TRACE("Ended splitter drag");
            return;
        }

        // Single binding lookup with current tool mode
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

        // Check double-click bindings first, then drag bindings.
        const auto double_click_action = is_general_double_click
                                             ? bindings_.getActionForMouseButton(tool_mode, mouse_btn, mods, true)
                                             : input::Action::NONE;
        const auto drag_action = bindings_.getActionForDrag(tool_mode, mouse_btn, mods, held_keys_);
        const auto single_click_action = bindings_.getActionForMouseButton(tool_mode, mouse_btn, mods, false);
        auto bound_action = double_click_action;
        if (bound_action == input::Action::NONE) {
            if (single_click_action == input::Action::NODE_PICK &&
                drag_action == input::Action::NODE_RECT_SELECT) {
                bound_action = input::Action::NODE_PICK;
            } else {
                bound_action = drag_action;
            }
        }
        if (bound_action == input::Action::NONE) {
            bound_action = single_click_action;
        }
        if (action == input::ACTION_PRESS &&
            forced_mouse_press_action_ != input::Action::NONE) {
            bound_action = forced_mouse_press_action_;
        }

        const bool is_right_button = button == static_cast<int>(input::AppMouseButton::RIGHT);
        if (action == input::ACTION_PRESS && is_right_button && isTransformGizmoUsing() && gui &&
            gui->gizmo().cancelActiveNodeTransformDrag()) {
            return;
        }
        if (action == input::ACTION_PRESS &&
            is_right_button &&
            pending_camera_context_menu_.active &&
            pending_camera_context_menu_.released) {
            clearSelectedCameraContextMenuGesture();
        }

        if (action == input::ACTION_PRESS) {
            if (over_gui) {
                return;
            }

            // Pivot placement is a viewport-global double-click gesture and must
            // remain available when an editing gizmo is merely hovered. Active
            // gizmo manipulation still owns the pointer until the drag finishes.
            // Gizmos only grab the left button, so hovering one never blocks
            // camera navigation on the other buttons.
            if (isTransformGizmoUsing() ||
                (is_left_button && over_transform_gizmo &&
                 bound_action != input::Action::CAMERA_SET_PIVOT)) {
                return;
            }

            // Only handle if clicking within the viewport
            if (!isInViewport(x, y)) {
                return;
            }

            if (forced_mouse_press_action_ == input::Action::NONE &&
                double_click_action == input::Action::NONE &&
                shouldDeferClickActionForDrag(single_click_action) &&
                drag_action != input::Action::NONE &&
                drag_action != single_click_action) {
                pending_click_drag_ = {
                    .active = true,
                    .button = button,
                    .mods = mods,
                    .click_action = single_click_action,
                    .drag_action = drag_action,
                    .press_pos = {x, y},
                };
                return;
            }

            const auto begin_node_selection_tracking = [&](const bool point_pick_enabled,
                                                           const bool rect_select_enabled) {
                if (!over_gui && !over_gizmo && tool_context_ && !over_transform_gizmo) {
                    is_node_rect_dragging_ = true;
                    node_rect_button_ = button;
                    node_rect_modifiers_ = mods;
                    node_point_pick_enabled_ = point_pick_enabled;
                    node_rect_select_enabled_ = rect_select_enabled;
                    node_rect_panel_ = splitPanelForScreenX(x);
                    node_rect_start_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
                    node_rect_end_ = node_rect_start_;
                }
            };

            switch (bound_action) {
            case input::Action::CAMERA_PAN:
                if (camera_navigation_mode_ == CameraNavigationMode::Walk)
                    break;
                if (const auto interaction = resolvePanelInteraction(x, y); interaction && interaction->valid()) {
                    const int context_camera_id =
                        is_right_button && services().renderingOrNull()
                            ? services().renderingOrNull()->pickCameraFrustum(glm::vec2(x, y))
                            : -1;
                    if (is_right_button &&
                        context_camera_id >= 0 &&
                        canOpenSelectedCameraContextMenu(context_camera_id)) {
                        pending_camera_context_menu_ = {
                            .active = true,
                            .camera_uid = context_camera_id,
                            .press_pos = {x, y},
                            .interaction = *interaction,
                        };
                        hovered_camera_id_ = context_camera_id;
                        break;
                    }

                    beginPanDrag(*interaction, button, x, y);
                } else {
                    break;
                }
                break;

            case input::Action::CAMERA_ORBIT:
                if (const auto interaction = resolvePanelInteraction(x, y); interaction && interaction->valid()) {
                    interaction->viewport->camera.finishGlide();
                    interaction->viewport->camera.initScreenPos(glm::vec2(x, y));
                    drag_viewport_ = interaction->viewport;
                    drag_split_panel_ = interaction->panel;
                    focusSplitPanel(interaction->panel);

                    if (camera_navigation_mode_ == CameraNavigationMode::FPV ||
                        camera_navigation_mode_ == CameraNavigationMode::Walk ||
                        camera_navigation_mode_ == CameraNavigationMode::Drone) {
                        float pivot_distance = glm::length(
                            interaction->viewport->camera.getPivot() - interaction->viewport->camera.t);
                        if (!std::isfinite(pivot_distance) || pivot_distance < 0.1f)
                            pivot_distance = 5.0f;
                        interaction->viewport->camera.updatePivotFromCamera(pivot_distance);
                        drag_mode_ = DragMode::Rotate;
                        lockLookPointer(x, y);
                    } else {
                        interaction->viewport->camera.startRotateAroundCenter(
                            glm::vec2(x, y), static_cast<float>(SDL_GetTicks() / 1000.0f));
                        drag_mode_ = DragMode::Orbit;
                    }
                } else {
                    break;
                }
                drag_button_ = button;
                break;

            case input::Action::CAMERA_SET_PIVOT: {
                if (camera_navigation_mode_ == CameraNavigationMode::Walk)
                    break;
                const auto interaction = resolvePanelInteraction(x, y);
                if (!interaction || !interaction->valid()) {
                    break;
                }
                auto& target_viewport = *interaction->viewport;
                focusSplitPanel(interaction->panel);
                target_viewport.camera.finishGlide();
                float current_distance = glm::length(target_viewport.camera.getPivot() - target_viewport.camera.t);
                if (!std::isfinite(current_distance) || current_distance < 0.1f)
                    current_distance = 5.0f;
                // Background clicks keep the current orbit radius instead of an
                // arbitrary fixed depth.
                const glm::vec3 new_pivot = unprojectScreenPoint(x, y, current_distance);
                const glm::vec3 forward = lfs::rendering::cameraForward(target_viewport.camera.R);

                target_viewport.camera.setPivot(new_pivot);
                target_viewport.camera.startGlide(new_pivot - forward * current_distance);
                onCameraMovementStart();
                publishCameraMove(&target_viewport);
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
                break;
            }

            case input::Action::SELECTION_REPLACE:
            case input::Action::SELECTION_ADD:
            case input::Action::SELECTION_REMOVE:
            case input::Action::SELECTION_INTERSECT:
                if (!over_gui && !over_gizmo && tool_context_ &&
                    !over_transform_gizmo) {
                    if (selection_tool_ && selection_tool_->isEnabled()) {
                        // Invoke selection stroke operator
                        auto* gm = services().guiOrNull();
                        const auto sub_mode = gm ? static_cast<int>(gm->gizmo().getSelectionSubMode()) : 0;
                        int selection_op = 0;
                        switch (bound_action) {
                        case input::Action::SELECTION_ADD: selection_op = 1; break;
                        case input::Action::SELECTION_REMOVE: selection_op = 2; break;
                        case input::Action::SELECTION_INTERSECT: selection_op = 3; break;
                        default: break;
                        }

                        op::OperatorProperties props;
                        props.set("x", x);
                        props.set("y", y);
                        props.set("button", button);
                        props.set("modifiers", mods);
                        props.set("mode", sub_mode);
                        props.set("op", selection_op);
                        props.set("brush_radius", selection_tool_->getBrushRadius());
                        props.set("use_crop_filter", selection_tool_->isCropFilterEnabled());
                        props.set("use_depth_filter", selection_tool_->isDepthFilterEnabled());

                        const auto result = op::operators().invoke(op::BuiltinOp::SelectionStroke, &props);
                        if (result.status == op::OperatorResult::RUNNING_MODAL ||
                            op::operators().hasModalOperator()) {
                            switch (bound_action) {
                            case input::Action::SELECTION_ADD:
                                selection_drag_op_ = input::SelectionOp::Add;
                                break;
                            case input::Action::SELECTION_REMOVE:
                                selection_drag_op_ = input::SelectionOp::Remove;
                                break;
                            case input::Action::SELECTION_INTERSECT:
                                selection_drag_op_ = input::SelectionOp::Intersect;
                                break;
                            default:
                                selection_drag_op_.reset();
                                break;
                            }
                            // Operator is now modal, don't set drag mode - modal dispatch handles it
                        }
                    } else if (((align_tool_ && align_tool_->isEnabled()) ||
                                (scale_distance_tool_ && scale_distance_tool_->isEnabled())) &&
                               !op::operators().hasModalOperator()) {
                        op::OperatorProperties props;
                        props.set("x", x);
                        props.set("y", y);
                        props.set("button", button);
                        props.set("modifiers", mods);
                        const auto id = align_tool_ && align_tool_->isEnabled()
                                            ? op::BuiltinOp::AlignPickPoint
                                            : op::BuiltinOp::ScaleDistancePickPoint;
                        const auto result = op::operators().invoke(id, &props);
                        if (result.status == op::OperatorResult::RUNNING_MODAL) {
                            return;
                        }
                    }
                }
                break;

            case input::Action::NODE_PICK:
                begin_node_selection_tracking(true, drag_action == input::Action::NODE_RECT_SELECT);
                break;

            case input::Action::NODE_RECT_SELECT:
                begin_node_selection_tracking(false, true);
                break;

            case input::Action::NONE:
            default:
                if (((align_tool_ && align_tool_->isEnabled()) ||
                     (scale_distance_tool_ && scale_distance_tool_->isEnabled())) &&
                    tool_context_ && !over_gui &&
                    !op::operators().hasModalOperator()) {
                    op::OperatorProperties props;
                    props.set("x", x);
                    props.set("y", y);
                    props.set("button", button);
                    props.set("modifiers", mods);
                    const auto id = align_tool_ && align_tool_->isEnabled()
                                        ? op::BuiltinOp::AlignPickPoint
                                        : op::BuiltinOp::ScaleDistancePickPoint;
                    const auto result = op::operators().invoke(id, &props);
                    if (result.status == op::OperatorResult::RUNNING_MODAL) {
                        return;
                    }
                }
                break;
            }
        } else if (action == input::ACTION_RELEASE) {
            if (pending_click_drag_.active &&
                pending_click_drag_.button == button) {
                const auto pending = pending_click_drag_;
                pending_click_drag_ = {};
                forced_mouse_press_action_ = pending.click_action;
                handleMouseButton(button, input::ACTION_PRESS,
                                  pending.press_pos.x, pending.press_pos.y);
                forced_mouse_press_action_ = input::Action::NONE;
                return;
            }

            if (is_right_button && pending_camera_context_menu_.active) {
                pending_camera_context_menu_.released = true;
                pending_camera_context_menu_.release_pos = {x, y};
                pending_camera_context_menu_.release_time = std::chrono::steady_clock::now();
                return;
            }

            bool was_dragging = false;
            Viewport* released_viewport = drag_viewport_;
            const SplitViewPanelId released_panel = drag_split_panel_;

            if (drag_mode_ == DragMode::Pan) {
                drag_mode_ = DragMode::None;
                drag_button_ = -1;
                was_dragging = true;
            } else if (drag_mode_ == DragMode::Rotate) {
                drag_mode_ = DragMode::None;
                drag_button_ = -1;
                was_dragging = true;
                syncLookPointerLock();
            } else if (drag_mode_ == DragMode::Orbit) {
                if (drag_viewport_) {
                    drag_viewport_->camera.endRotateAroundCenter();
                } else {
                    viewport_.camera.endRotateAroundCenter();
                }
                if (released_viewport) {
                    snapViewportToNearestAxis(*released_viewport, released_panel);
                } else {
                    snapViewportToNearestAxis(viewport_, released_panel);
                }
                drag_mode_ = DragMode::None;
                drag_button_ = -1;
                was_dragging = true;
            }
            drag_viewport_ = nullptr;

            if (was_dragging) {
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
                auto* const moved_viewport = released_viewport ? released_viewport : &viewport_;
                ui::CameraMove{
                    .rotation = moved_viewport->getRotationMatrix(),
                    .translation = moved_viewport->getTranslation()}
                    .emit();
                onCameraMovementEnd();
            }

            // Short-click on a frustum → select that single camera and stop.
            // Long drags fall through so a rectangle that *started* over a
            // frustum can still sweep-select the surrounding cameras.
            if (press_consumed_camera_frustum) {
                const double drag_dist = glm::length(glm::dvec2(x, y) - pressed_camera_frustum_pos);
                const bool was_click = drag_dist < kCameraFrustumClickThreshold;
                const auto release_tool_mode = getCurrentToolMode();
                const bool allow_camera_frustum_pick =
                    release_tool_mode == input::ToolMode::GLOBAL ||
                    release_tool_mode == input::ToolMode::SELECTION;
                if (allow_camera_frustum_pick &&
                    was_click && pressed_camera_frustum_id >= 0 &&
                    !over_gui && !over_transform_gizmo) {
                    selectCameraByUid(pressed_camera_frustum_id,
                                      (pressed_camera_frustum_modifiers & input::KEYMOD_CTRL) != 0);
                    if (button == node_rect_button_) {
                        is_node_rect_dragging_ = false;
                        node_rect_button_ = -1;
                        node_rect_modifiers_ = input::MODIFIER_NONE;
                        node_point_pick_enabled_ = false;
                        node_rect_select_enabled_ = false;
                    }
                    return;
                }
            }

            // Node picking on release
            if (is_node_rect_dragging_ && button == node_rect_button_) {
                const bool point_pick_enabled = node_point_pick_enabled_;
                const bool rect_select_enabled = node_rect_select_enabled_;
                is_node_rect_dragging_ = false;
                node_rect_button_ = -1;
                const int node_rect_modifiers = node_rect_modifiers_;
                node_rect_modifiers_ = input::MODIFIER_NONE;
                node_point_pick_enabled_ = false;
                node_rect_select_enabled_ = false;
                if (tool_context_ && !isPointerOverBlockingUi(x, y)) {
                    auto* scene_manager = tool_context_->getSceneManager();
                    if (scene_manager) {
                        constexpr float CLICK_THRESHOLD_PX = 5.0f;
                        const float drag_dist = glm::length(node_rect_end_ - node_rect_start_);

                        if (drag_dist < CLICK_THRESHOLD_PX && point_pick_enabled) {
                            // Point pick via ray-AABB intersection
                            const auto [ray_origin, ray_dir] = computePickRay(x, y);
                            const std::string picked = scene_manager->pickNodeByRay(ray_origin, ray_dir);
                            if (!picked.empty()) {
                                const auto* const picked_node = scene_manager->getScene().getNode(picked);
                                const bool ctrl_held =
                                    (node_rect_modifiers & input::KEYMOD_CTRL) != 0;
                                const bool shift_held =
                                    (node_rect_modifiers & input::KEYMOD_SHIFT) != 0;
                                if (picked_node && ctrl_held) {
                                    if (scene_manager->selectionState().isNodeSelected(picked_node->id))
                                        scene_manager->removeFromSelection(picked_node->id);
                                    else
                                        scene_manager->addToSelection(picked_node->id);
                                } else if (picked_node && shift_held) {
                                    scene_manager->addToSelection(picked_node->id);
                                } else if (auto result = cap::selectNode(*scene_manager, picked); !result) {
                                    LOG_WARN("Node pick selection failed: {}", result.error());
                                }
                            } else if ((node_rect_modifiers &
                                        (input::KEYMOD_CTRL | input::KEYMOD_SHIFT)) == 0) {
                                (void)cap::clearNodeSelection(*scene_manager);
                            }
                        } else if (drag_dist >= CLICK_THRESHOLD_PX && rect_select_enabled) {
                            // Rectangle selection — convert window coords to viewport-local
                            glm::vec2 vp_offset(0.0f);
                            if (auto* gm = services().guiOrNull())
                                vp_offset = glm::vec2(gm->getViewportPos().x, gm->getViewportPos().y);

                            float panel_offset_x = 0.0f;
                            float panel_width = viewport_bounds_.width;
                            if (isIndependentSplitViewActive()) {
                                if (auto* const rendering = services().renderingOrNull()) {
                                    const auto panel_info = rendering->resolveViewerPanel(
                                        viewport_,
                                        {viewport_bounds_.x, viewport_bounds_.y},
                                        {viewport_bounds_.width, viewport_bounds_.height},
                                        std::nullopt,
                                        node_rect_panel_);
                                    if (panel_info && panel_info->valid()) {
                                        panel_offset_x = panel_info->x - vp_offset.x;
                                        panel_width = panel_info->width;
                                    }
                                }
                            }

                            const glm::vec2 rect_min(
                                std::min(node_rect_start_.x, node_rect_end_.x) - vp_offset.x - panel_offset_x,
                                std::min(node_rect_start_.y, node_rect_end_.y) - vp_offset.y);
                            const glm::vec2 rect_max(
                                std::max(node_rect_start_.x, node_rect_end_.x) - vp_offset.x - panel_offset_x,
                                std::max(node_rect_start_.y, node_rect_end_.y) - vp_offset.y);

                            Viewport pick_viewport = viewport_;
                            if (auto* const rendering = services().renderingOrNull()) {
                                pick_viewport = rendering->resolvePanelViewport(viewport_, node_rect_panel_);
                            }
                            pick_viewport.windowSize = glm::ivec2(
                                std::max(static_cast<int>(panel_width), 1),
                                std::max(static_cast<int>(viewport_bounds_.height), 1));

                            const std::vector<std::string> picked_nodes = scene_manager->pickNodesInScreenRect(
                                rect_min, rect_max,
                                pick_viewport.getViewMatrix(),
                                pick_viewport.getProjectionMatrix(),
                                pick_viewport.windowSize);

                            // Modifier-driven selection mode, mirroring the
                            // gaussian SelectionTool: shift = add, ctrl =
                            // remove, no mod = replace. An empty rect with no
                            // modifier clears the selection; with a modifier it
                            // is a no-op so transient drags don't wipe state.
                            const bool shift_held = (node_rect_modifiers & input::KEYMOD_SHIFT) != 0;
                            const bool ctrl_held = (node_rect_modifiers & input::KEYMOD_CTRL) != 0;
                            const std::string_view select_mode = ctrl_held    ? "remove"
                                                                 : shift_held ? "add"
                                                                              : "replace";
                            if (picked_nodes.empty()) {
                                if (!shift_held && !ctrl_held) {
                                    (void)cap::clearNodeSelection(*scene_manager);
                                }
                            } else {
                                if (auto result = cap::selectNodes(*scene_manager, picked_nodes, select_mode); !result) {
                                    LOG_WARN("Rectangle node selection failed: {}", result.error());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void InputController::handleMouseMove(double x, double y, const std::optional<glm::dvec2> motion) {
        if (vr_viewing_)
            return;
        if (glm::length(glm::dvec2(x, y) - gamepad_press_pos_) >= 4.0)
            gamepad_click_candidate_ = false;
        decayHeldDragMomentum();
        LOG_PERF("InputController::handleMouseMove pos=({},{}) drag_mode={}",
                 x, y, static_cast<int>(drag_mode_));
        auto* gui = services().guiOrNull();
        // The locked pointer's position stops at the window edge; mouse look
        // accumulates the relative motion instead.
        const bool relative_look = drag_mode_ == DragMode::Rotate && motion.has_value();
        if (relative_look)
            look_pos_ += *motion;
        // The hidden pointer still moves inside the window; it must not hover or
        // drag anything while a value drag owns it.
        if (value_drag_restore_) {
            if (motion) {
                value_drag_dx_ += static_cast<float>(motion->x) / input::windowPixelScale(window_).x;
                if (auto* rendering = services().renderingOrNull())
                    rendering->markDirty(DirtyFlag::OVERLAY, FrameReason::Input, "value_drag");
            }
            return;
        }
        const glm::vec2 camera_drag_pos = relative_look ? glm::vec2(look_pos_) : glm::vec2(x, y);
        const auto park_camera_drag_anchor = [&]() {
            if (drag_mode_ != DragMode::Pan && drag_mode_ != DragMode::Orbit &&
                drag_mode_ != DragMode::Rotate)
                return;
            auto* const target = drag_viewport_ ? drag_viewport_ : &viewport_;
            const float now = static_cast<float>(SDL_GetTicks() / 1000.0f);
            target->camera.followDragPointer(camera_drag_pos, now);
        };

        if (gui && gui->isCapturingInput()) {
            gui->captureMouseMove(x, y);
            last_mouse_pos_ = {x, y};
            park_camera_drag_anchor();
            return;
        }

        // Forward to pie menu if open — consume event to prevent viewport interaction
        if (gui && gui->gizmo().isPieMenuOpen()) {
            gui->gizmo().onPieMenuMouseMove({static_cast<float>(x), static_cast<float>(y)});
            last_mouse_pos_ = {x, y};
            park_camera_drag_anchor();
            return;
        }

        // Track if we moved significantly
        glm::dvec2 current_pos{x, y};
        const double delta_x = x - last_mouse_pos_.x;
        const double delta_y = y - last_mouse_pos_.y;
        if (align_tool_ && align_tool_->isEnabled() && (delta_x != 0.0 || delta_y != 0.0)) {
            if (auto* rendering = services().renderingOrNull())
                rendering->markDirty(DirtyFlag::OVERLAY, FrameReason::Overlay, "align_cursor_moved");
        }
        if (scale_distance_tool_ && scale_distance_tool_->isEnabled() && (delta_x != 0.0 || delta_y != 0.0)) {
            if (auto* rendering = services().renderingOrNull())
                rendering->markDirty(DirtyFlag::OVERLAY, FrameReason::Overlay, "scale_distance_cursor_moved");
        }

        // Dispatch to modal operators first - if consumed, don't continue
        bool over_gui = false;
        bool over_gui_hover = false;
        if (input_router_) {
            const auto targets = input_router_->pointerTargets(x, y);
            over_gui = targets.pointer_target == input::InputTarget::Gui;
            over_gui_hover = targets.hover_target == input::InputTarget::Gui;
        } else {
            over_gui = isPointerOverBlockingUi(x, y);
            over_gui_hover = isPointerOverUiHover(x, y);
        }
        const bool selection_pointer_blocked =
            op::operators().activeModalId() == op::to_string(op::BuiltinOp::SelectionStroke) &&
            (over_gui || (!input_router_ && !isInViewport(x, y)));
        if (!selection_pointer_blocked &&
            dispatchMouseMoveToModals(x, y, delta_x, delta_y, getModifierKeys(), over_gui_hover)) {
            last_mouse_pos_ = current_pos;
            park_camera_drag_anchor();
            return;
        }

        const bool over_viewport_gizmo = gui && gui->gizmo().isPositionInViewportGizmo(x, y);
        const bool over_transform_gizmo = isTransformGizmoOverOrUsing();
        const bool over_splitter =
            drag_mode_ == DragMode::None &&
            isInViewport(x, y) &&
            isNearSplitter(x, y);

        if (pending_click_drag_.active) {
            if (!isMouseButtonPressed(pending_click_drag_.button)) {
                pending_click_drag_ = {};
            } else if (glm::length(current_pos - pending_click_drag_.press_pos) >=
                       DOUBLE_CLICK_DISTANCE) {
                const auto pending = pending_click_drag_;
                pending_click_drag_ = {};
                forced_mouse_press_action_ = pending.drag_action;
                handleMouseButton(pending.button, input::ACTION_PRESS,
                                  pending.press_pos.x, pending.press_pos.y);
                forced_mouse_press_action_ = input::Action::NONE;
            }
        }

        if (drag_mode_ == DragMode::Splitter && services().renderingOrNull()) {
            const auto viewport_size = glm::ivec2(static_cast<int>(viewport_bounds_.width),
                                                  static_cast<int>(viewport_bounds_.height));
            const auto content = services().renderingOrNull()->getContentBounds(viewport_size);
            const double delta = x - splitter_start_x_;
            const float new_pos = std::clamp(splitter_start_pos_ + static_cast<float>(delta / content.width), 0.0f, 1.0f);

            ui::SplitPositionChanged{.position = new_pos}.emit();
            last_mouse_pos_ = {x, y};
            park_camera_drag_anchor();
            return;
        }

        if (pending_camera_context_menu_.active &&
            !pending_camera_context_menu_.released &&
            isMouseButtonPressed(static_cast<int>(input::AppMouseButton::RIGHT)) &&
            glm::length(current_pos - pending_camera_context_menu_.press_pos) >= kCameraContextMenuDragThreshold) {
            const auto interaction = pending_camera_context_menu_.interaction;
            clearSelectedCameraContextMenuGesture();
            if (interaction.valid()) {
                beginPanDrag(interaction,
                             static_cast<int>(input::AppMouseButton::RIGHT),
                             x, y);
            }
        }

        // Camera frustum hover detection with improved throttling
        // (frustum visibility is now controlled by scene graph, not a checkbox)
        if (services().renderingOrNull() &&
            isInViewport(x, y) &&
            drag_mode_ == DragMode::None &&
            !over_gui &&
            !over_splitter &&
            !over_viewport_gizmo &&
            !over_transform_gizmo) {

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
                            SDL_SetCursor(hand_cursor_);
                            current_cursor_ = CursorType::Hand;
                        }
                    }
                } else {
                    // No camera under cursor
                    if (hovered_camera_id_ != -1) {
                        hovered_camera_id_ = -1;
                        LOG_TRACE("No longer hovering over camera");
                        if (current_cursor_ == CursorType::Hand) {
                            SDL_SetCursor(SDL_GetDefaultCursor());
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
                    SDL_SetCursor(SDL_GetDefaultCursor());
                    current_cursor_ = CursorType::Default;
                }
            }
        }

        if (over_splitter) {
            SDL_SetCursor(resize_cursor_);
            current_cursor_ = CursorType::Resize;
        } else if (current_cursor_ == CursorType::Resize) {
            SDL_SetCursor(SDL_GetDefaultCursor());
            current_cursor_ = CursorType::Default;
        }

        glm::vec2 pos(x, y);
        last_mouse_pos_ = current_pos;

        // Node rectangle dragging - cancel if a transform gizmo takes over
        if (is_node_rect_dragging_) {
            if (isTransformGizmoUsing()) {
                is_node_rect_dragging_ = false;
                node_rect_button_ = -1;
                node_rect_modifiers_ = input::MODIFIER_NONE;
                node_point_pick_enabled_ = false;
                node_rect_select_enabled_ = false;
            } else {
                node_rect_end_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
                if (tool_context_)
                    tool_context_->requestRender();
            }
        }

        // Block camera dragging if a transform gizmo is being used
        if (isTransformGizmoUsing()) {
            park_camera_drag_anchor();
            return;
        }

        // Handle camera dragging
        if (drag_mode_ != DragMode::None &&
            drag_mode_ != DragMode::Gizmo &&
            drag_mode_ != DragMode::Splitter) {
            auto* const target_viewport = drag_viewport_ ? drag_viewport_ : &viewport_;

            switch (drag_mode_) {
            case DragMode::Pan: {
                const float current_time = static_cast<float>(SDL_GetTicks() / 1000.0f);
                target_viewport->camera.panDrag(pos, current_time);
                pan_coast_viewport_ = target_viewport;
                break;
            }
            case DragMode::Rotate:
                if (camera_navigation_mode_ == CameraNavigationMode::Drone) {
                    target_viewport->camera.droneLook(camera_drag_pos);
                } else {
                    target_viewport->camera.rotateFpv(camera_drag_pos);
                }
                break;
            case DragMode::Orbit: {
                const float current_time = static_cast<float>(SDL_GetTicks() / 1000.0f);
                target_viewport->camera.orbitDrag(
                    pos, camera_navigation_mode_ == CameraNavigationMode::Trackball, current_time);
                orbit_coast_viewport_ = target_viewport;
                break;
            }
            default:
                break;
            }
            // Signal continuous camera movement
            onCameraMovementStart();
            publishCameraMove(target_viewport);
        }
    }

    void InputController::handleScroll([[maybe_unused]] double xoff, double yoff) {
        if (vr_viewing_)
            return;
        // Capture mode (input settings panel) consumes scroll first so the user
        // can rebind scroll-only actions like Camera Zoom or chord-style Roll.
        if (bindings_.isCapturing()) {
            std::optional<int> chord_key;
            if (!held_keys_.empty()) {
                chord_key = held_keys_.back();
            }
            bindings_.captureScroll(getModifierKeys(), chord_key);
            return;
        }

        float fx, fy;
        input::mouseStateInPixels(window_, &fx, &fy);
        double mouse_x = fx, mouse_y = fy;
        bool over_gui = false;
        bool over_gui_hover = false;
        if (input_router_) {
            const auto targets = input_router_->pointerTargets(mouse_x, mouse_y);
            over_gui = targets.pointer_target == input::InputTarget::Gui;
            over_gui_hover = targets.hover_target == input::InputTarget::Gui;
        } else {
            over_gui = isPointerOverBlockingUi(mouse_x, mouse_y);
            over_gui_hover = isPointerOverUiHover(mouse_x, mouse_y);
        }

        // Dispatch to modal operators first - if consumed, don't continue
        if (dispatchScrollToModals(xoff, yoff, mouse_x, mouse_y, getModifierKeys(), over_gui_hover)) {
            return;
        }

        const int mods = getModifierKeys();
        const input::Action scroll_action = bindings_.getActionForScroll(getCurrentToolMode(), mods, held_keys_);
        if (selection_tool_ && selection_tool_->isEnabled()) {
            if (scroll_action == input::Action::DEPTH_ADJUST_FAR &&
                selection_tool_->isDepthFilterEnabled()) {
                selection_tool_->adjustDepthFar((yoff > 0) ? 1.1f : 0.9f);
                return;
            }
        }

        // Brush radius adjustment for the selection tool. Modal operators
        // for selection strokes pass scroll through, so it's safe to honor
        // BRUSH_RESIZE here even mid-stroke — that's what lets the user grow
        // or shrink the ring while in the middle of an add or subtract drag.
        if (scroll_action == input::Action::BRUSH_RESIZE) {
            if (selection_tool_ && selection_tool_->isEnabled()) {
                const float scale = (yoff > 0) ? 1.1f : 0.9f;
                selection_tool_->setBrushRadius(selection_tool_->getBrushRadius() * scale);
                return;
            }
        }

        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter)
            return;

        if (!isInViewport(mouse_x, mouse_y) || over_gui)
            return;

        const auto interaction = resolvePanelInteraction(mouse_x, mouse_y);
        if (!interaction || !interaction->valid()) {
            return;
        }
        auto& target_viewport = *interaction->viewport;
        focusSplitPanel(interaction->panel);
        target_viewport.camera.finishGlide();

        const float delta = static_cast<float>(yoff);
        if (std::abs(delta) < 0.01f)
            return;

        const bool carry_pivot = camera_navigation_mode_ == CameraNavigationMode::FPV ||
                                 camera_navigation_mode_ == CameraNavigationMode::Drone;
        if (camera_navigation_mode_ == CameraNavigationMode::Walk)
            return;

        pauseGamepadNavigation();
        if (scroll_action == input::Action::CAMERA_ROLL) {
            target_viewport.camera.rotate_roll(delta);
        } else if (scroll_action == input::Action::CAMERA_ZOOM) {
            // In orthographic mode, adjust ortho_scale instead of camera position
            if (services().renderingOrNull()) {
                auto settings = services().renderingOrNull()->getSettings();
                if (settings.orthographic) {
                    constexpr float ORTHO_ZOOM_FACTOR = 0.1f;
                    constexpr float MIN_ORTHO_SCALE = 1.0f;
                    constexpr float MAX_ORTHO_SCALE = 10000.0f;
                    const float scale_factor = 1.0f + delta * ORTHO_ZOOM_FACTOR;
                    const float current = target_viewport.ortho_scale_override.value_or(settings.ortho_scale);
                    const float scale = std::clamp(current * scale_factor, MIN_ORTHO_SCALE, MAX_ORTHO_SCALE);
                    if (&target_viewport != &viewport_ || target_viewport.ortho_scale_override) {
                        target_viewport.ortho_scale_override = scale;
                    }
                    if (&target_viewport == &viewport_) {
                        settings.ortho_scale = scale;
                        services().renderingOrNull()->updateSettings(settings);
                    }
                } else {
                    target_viewport.camera.zoom(delta, carry_pivot);
                }
            } else {
                target_viewport.camera.zoom(delta, carry_pivot);
            }
        } else {
            return;
        }

        onCameraMovementStart();
        publishCameraMove(&target_viewport);
        if (auto* const viewer = lfs::python::get_visualizer()) {
            viewer->projectMarkSessionChapterCommitted("VIEW");
        }
    }

    void InputController::handleKey(const int key, const int action, const int mods) {
        // Compatibility path for tests and callers that don't split physical vs layout-aware keys.
        handleKey(key, key, 0, action, mods);
    }

    void InputController::handleKey(const int physical_key, const int logical_key,
                                    const int scancode, int action, int mods, const bool owned_release, const bool gui_consumed) {
        if ((action == input::ACTION_PRESS || action == input::ACTION_REPEAT) && hasViewportKeyboardFocus())
            pauseGamepadNavigation();
        // Clear before any GUI/modal early return, including owned key releases.
        if (action == input::ACTION_RELEASE)
            camera_browse_repeat_.release(physical_key);
        // Track modifier keys (always, even if GUI has focus)
        if (physical_key == input::KEY_LEFT_CONTROL || physical_key == input::KEY_RIGHT_CONTROL) {
            key_ctrl_pressed_ = (action != input::ACTION_RELEASE);
        }
        if (physical_key == input::KEY_LEFT_ALT || physical_key == input::KEY_RIGHT_ALT) {
            key_alt_pressed_ = (action != input::ACTION_RELEASE);
        }
        if (action == input::ACTION_PRESS && logical_key == input::KEY_ESCAPE && isValueDragActive()) {
            cancelValueDrag(true);
            return;
        }
        const bool wants_text_input = input_router_
                                          ? input_router_->isTextInputActive()
                                          : gui::guiFocusState().want_text_input;
        const bool is_modifier_key =
            physical_key == input::KEY_LEFT_SHIFT || physical_key == input::KEY_RIGHT_SHIFT ||
            physical_key == input::KEY_LEFT_CONTROL || physical_key == input::KEY_RIGHT_CONTROL ||
            physical_key == input::KEY_LEFT_ALT || physical_key == input::KEY_RIGHT_ALT ||
            physical_key == input::KEY_LEFT_SUPER || physical_key == input::KEY_RIGHT_SUPER;
        // The same press is movement, so its printed character must not arm a
        // scroll chord. Release drops nothing that this press did not insert.
        const bool movement_owns_logical_key =
            physicalMovementConsumesLogicalKey(physical_key, logical_key, mods);
        const bool altgr_held = (mods & input::KEYMOD_ALTGR) != 0;
        if (wants_text_input && !bindings_.isCapturing()) {
            // Text keys must not become viewport mouse/scroll chords either.
            held_keys_.clear();
        } else if (!altgr_held && !movement_owns_logical_key && !is_modifier_key &&
                   logical_key != input::KEY_UNKNOWN) {
            if (action == input::ACTION_RELEASE) {
                std::erase(held_keys_, logical_key);
            } else if (!std::ranges::contains(held_keys_, logical_key)) {
                held_keys_.push_back(logical_key);
            }
        }

        auto* gui = services().guiOrNull();

        // Forward to binding capture before Python panels or modal operators consume keys.
        if (action == input::ACTION_PRESS && bindings_.isCapturing()) {
            bindings_.captureKey(physical_key, logical_key, mods);
            return;
        }
        if (vr_viewing_) {
            handleVrKey(physical_key, logical_key, action, mods, gui_consumed || wants_text_input);
            return;
        }

        // AltGr produces layout characters for text input, but this build cannot
        // represent that modifier in saved shortcuts. Never dispatch the bare
        // logical key or arm a physical movement binding for the same event.
        if (altgr_held)
            return;

        const auto tool_mode = getCurrentToolMode();
        SDL_Keymod sdl_modifiers = SDL_KMOD_NONE;
        if (mods & input::MODIFIER_SHIFT)
            sdl_modifiers |= SDL_KMOD_SHIFT;
        const auto key_scancode = static_cast<SDL_Scancode>(scancode);
        const auto printed_keycode = scancode > 0
                                         ? SDL_GetKeyFromScancode(key_scancode, sdl_modifiers, false)
                                         : SDLK_UNKNOWN;
        const auto unshifted_keycode = scancode > 0
                                           ? SDL_GetKeyFromScancode(key_scancode, SDL_KMOD_NONE, false)
                                           : SDLK_UNKNOWN;
        const int printed_punctuation = input::sdlPrintedKeyToAppKey(printed_keycode);
        const bool shift_needed_for_printed_key =
            (mods & input::MODIFIER_SHIFT) != 0 && printed_keycode != unshifted_keycode;
        bool binding_repeats = false;
        auto bound_action = bindings_.getActionForShortcut(
            tool_mode, physical_key, logical_key, mods, printed_punctuation,
            shift_needed_for_printed_key, &binding_repeats);
        if (bound_action == input::Action::NONE) {
            bound_action = resolveCrossToolActivationShortcut(
                bindings_, tool_mode, physical_key, logical_key, mods);
        }

        const bool gui_keyboard_focus = input_router_
                                            ? input_router_->keyboardFocus() == input::InputTarget::Gui
                                            : gui::guiFocusState().want_capture_keyboard;
        if (action == input::ACTION_PRESS && logical_key == input::KEY_ESCAPE &&
            gui && gui->gizmo().cancelActiveGizmoDrag()) {
            return;
        }

        if (action == input::ACTION_PRESS && logical_key == input::KEY_ESCAPE &&
            gui_keyboard_focus) {
            return;
        }

        const bool is_mcp_runtime_action =
            bound_action == input::Action::TOGGLE_MCP_SERVER ||
            bound_action == input::Action::TOGGLE_MCP_BINDING;
        if (lfs::python::has_keyboard_capture_request() && !is_mcp_runtime_action && !owned_release) {
            return;
        }

        const bool project_save = logical_key == input::KEY_S && mods == input::KEYMOD_CTRL;
        if (wants_text_input && !is_mcp_runtime_action && !owned_release) {
            if (project_save && action == input::ACTION_PRESS)
                cmd::ProjectSave{}.emit();
            return;
        }

        if (gui_consumed && !is_mcp_runtime_action && !owned_release)
            return;

        // Physical movement wins over a different logical shortcut. The printed
        // character is not dispatched. Press and repeat arm the key; release
        // clears it. Ctrl/Alt/Super chords stay on the shortcut path.
        if (physicalMovementConsumesLogicalKey(physical_key, logical_key, mods)) {
            if (action == input::ACTION_RELEASE) {
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
            }
            armPhysicalMovementKey(physical_key, action != input::ACTION_RELEASE);
            return;
        }

        if (action == input::ACTION_RELEASE &&
            (bound_action == input::Action::CAMERA_MOVE_FORWARD ||
             bound_action == input::Action::CAMERA_MOVE_BACKWARD ||
             bound_action == input::Action::CAMERA_MOVE_LEFT ||
             bound_action == input::Action::CAMERA_MOVE_RIGHT ||
             bound_action == input::Action::CAMERA_MOVE_UP ||
             bound_action == input::Action::CAMERA_MOVE_DOWN)) {
            if (auto* const viewer = lfs::python::get_visualizer()) {
                viewer->projectMarkSessionChapterCommitted("VIEW");
            }
        }
        // Dispatch to modal operators first - if consumed, don't continue
        float mx_f, my_f;
        input::mouseStateInPixels(window_, &mx_f, &my_f);
        double mx = mx_f, my = my_f;
        const bool over_gui_hover = isPointerOverUiHover(mx, my);
        const auto modal_action =
            isPolygonModalActive()
                ? bindings_.getActionForShortcut(input::ToolMode::SELECTION, physical_key, logical_key, mods,
                                                 printed_punctuation, shift_needed_for_printed_key, nullptr)
                : bound_action;
        if (action == input::ACTION_PRESS &&
            dispatchActionToModal(modal_action, mods, mx, my)) {
            return;
        }
        if (dispatchKeyToModals(logical_key, scancode, action, mods, mx, my, over_gui_hover)) {
            return;
        }
        if (action == input::ACTION_PRESS && camera_navigation_mode_ == CameraNavigationMode::Walk &&
            !gui_keyboard_focus && bindings_.getActionForShortcut(input::ToolMode::WALK, physical_key, logical_key, mods, printed_punctuation, shift_needed_for_printed_key) == input::Action::EXIT_WALK) {
            setCameraNavigationMode(CameraNavigationMode::Orbit);
            return;
        }

        // Handle pie menu key release and escape
        if (gui && gui->gizmo().isPieMenuOpen()) {
            if (action == input::ACTION_RELEASE) {
                const auto pie_key = bindings_.getKeyForAction(input::Action::PIE_MENU, getCurrentToolMode());
                // The press can match the binding by physical key on another layout, so the release must too.
                if (pie_key >= 0 && (logical_key == pie_key || physical_key == pie_key)) {
                    gui->gizmo().onPieMenuKeyRelease();
                    return;
                }
            }
            if (action == input::ACTION_PRESS && logical_key == input::KEY_ESCAPE) {
                gui->gizmo().closePieMenu();
                return;
            }
        }

        const bool viewport_keyboard_focus = input_router_
                                                 ? input_router_->isViewportKeyboardFocused()
                                                 : false;
        const bool modal_open = input_router_
                                    ? input_router_->isModalOpen()
                                    : (gui && gui->isModalWindowOpen());
        if (action != input::ACTION_PRESS && action != input::ACTION_REPEAT)
            return;

        if (modal_open)
            return;

        if (action == input::ACTION_PRESS &&
            logical_key == input::KEY_S &&
            mods == input::KEYMOD_CTRL) {
            cmd::ProjectSave{}.emit();
            return;
        }

        switch (input::shortcutScopeForAction(bound_action)) {
        case input::ShortcutScope::Viewport:
            if (!viewport_keyboard_focus || wants_text_input) {
                return;
            }
            break;
        case input::ShortcutScope::GlobalWhenNotTextEditing:
            if (wants_text_input) {
                return;
            }
            break;
        case input::ShortcutScope::Global:
            break;
        }

        // A held key repeats when its binding or the action asks for it. Camera
        // browsing repeats on its own timer at the preference rate instead.
        if (action == input::ACTION_REPEAT &&
            (bound_action == input::Action::CAMERA_NEXT_VIEW || bound_action == input::Action::CAMERA_PREV_VIEW ||
             !(binding_repeats || input::describe(bound_action).repeats_on_hold))) {
            return;
        }

        if (action == input::ACTION_PRESS) {
            if (handleToolControlActivationShortcut(bound_action, gui))
                return;
        }

        if (bound_action != input::Action::NONE) {
            switch (bound_action) {
            case input::Action::TOGGLE_SPLIT_VIEW:
                cmd::ToggleSplitView{}.emit();
                return;

            case input::Action::TOGGLE_INDEPENDENT_SPLIT_VIEW:
                toggleIndependentSplitView();
                return;

            case input::Action::TOGGLE_GT_COMPARISON:
                if (auto* const rendering = services().renderingOrNull();
                    rendering && (rendering->isGTComparisonActive() || rendering->hasGTComparisonAvailable())) {
                    cmd::ToggleGTComparison{}.emit();
                } else if (const auto* const scene_manager = services().sceneOrNull();
                           scene_manager && plyComparisonNodes(scene_manager->getScene()).size() >= 2) {
                    cmd::ToggleSplitView{}.emit();
                }
                return;

            case input::Action::OPEN_PREFERENCES:
                if (gui)
                    gui->openPreferences();
                return;

            case input::Action::TOGGLE_PYTHON_CONSOLE:
                lfs::python::show_python_console();
                return;

            case input::Action::SELECT_ALL_SCENE_NODES:
                if (gui)
                    (void)gui->selectAllSceneNodesIfFocused();
                return;

            case input::Action::TOGGLE_SCENE_SELECTION_VISIBILITY:
                if (gui)
                    (void)gui->toggleSceneSelectionVisibilityIfFocused();
                return;

            case input::Action::TOGGLE_SCENE_SELECTION_TRAINING:
                if (gui)
                    (void)gui->toggleSceneSelectionTrainingIfFocused();
                return;

            case input::Action::GROUP_SELECTED_SCENE_NODES:
                if (gui)
                    (void)gui->groupSelectedSceneNodesIfFocused();
                return;

            case input::Action::UNGROUP_SELECTED_SCENE_NODE:
                if (gui)
                    (void)gui->ungroupSelectedSceneNodeIfFocused();
                return;

            case input::Action::TOGGLE_MCP_SERVER:
                toggleMcpRuntimeEnabled();
                return;

            case input::Action::TOGGLE_MCP_BINDING:
                toggleMcpRuntimeBinding();
                return;

            case input::Action::TOGGLE_CAMERA_FRUSTUMS:
                if (auto* rendering_manager = services().renderingOrNull()) {
                    auto settings = rendering_manager->getSettings();
                    settings.show_camera_frustums = !settings.show_camera_frustums;
                    rendering_manager->updateSettings(settings);
                    if (auto* const viewer = lfs::python::get_visualizer()) {
                        viewer->projectMarkSessionChapterCommitted("VIEW");
                    }
                }
                return;

            case input::Action::TOGGLE_GRID:
                if (auto* rendering_manager = services().renderingOrNull()) {
                    auto settings = rendering_manager->getSettings();
                    settings.show_grid = !settings.show_grid;
                    rendering_manager->updateSettings(settings);
                    if (auto* const viewer = lfs::python::get_visualizer()) {
                        viewer->projectMarkSessionChapterCommitted("VIEW");
                    }
                }
                return;

            case input::Action::CAMERA_NEXT_VIEW:
            case input::Action::CAMERA_PREV_VIEW: {
                const int delta = bound_action == input::Action::CAMERA_NEXT_VIEW ? 1 : -1;
                if (advanceCameraView(delta)) {
                    if (UserPreferences::instance().cameraBrowseRepeat()) {
                        camera_browse_repeat_modifiers_ = mods;
                        camera_browse_repeat_.start(physical_key, delta, input::CameraBrowseRepeat::Clock::now());
                    }
                }
                return;
            }

            case input::Action::CAMERA_SET_HOME:
                setHome();
                return;

            case input::Action::CAMERA_RESET_HOME:
                goHome();
                return;

            case input::Action::CAMERA_RESTORE_HOME:
                resetHome();
                return;

            case input::Action::CAMERA_FOCUS_SELECTION:
                handleFocusSelection(activeKeyboardViewport());
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
                return;

            case input::Action::CYCLE_PLY:
                cmd::CyclePLY{}.emit();
                return;

            case input::Action::CYCLE_SELECTION_VIS:
                if (gui && gui->gizmo().getCurrentToolMode() == ToolType::Selection) {
                    cmd::CycleSelectionVisualization{}.emit();
                }
                return;

            case input::Action::TOGGLE_SELECTION_DEPTH_FILTER:
                if (selection_tool_ && selection_tool_->isEnabled()) {
                    selection_tool_->toggleDepthFilter();
                    selection_tool_->syncDepthFilterToCamera(activeKeyboardViewport());
                    if (auto* const viewer = lfs::python::get_visualizer()) {
                        viewer->projectMarkSessionChapterCommitted("VIEW");
                    }
                }
                return;

            case input::Action::TOGGLE_SELECTION_CROP_FILTER:
                if (selection_tool_ && selection_tool_->isEnabled()) {
                    selection_tool_->toggleCropFilter();
                    if (auto* const viewer = lfs::python::get_visualizer()) {
                        viewer->projectMarkSessionChapterCommitted("VIEW");
                    }
                }
                return;

            case input::Action::CANCEL_POLYGON:
                if (dispatchActionToModal(bound_action, mods, mx, my)) {
                    return;
                }
                if (op::operators().hasModalOperator()) {
                    op::operators().cancelModalOperator();
                    return;
                }
                if (tool_context_) {
                    if (auto* sm = tool_context_->getSceneManager()) {
                        if (auto* selection_service = sm->getSelectionService();
                            selection_service && selection_service->isInteractiveSelectionActive()) {
                            selection_service->cancelInteractiveSelection();
                            return;
                        }
                    }
                }
                return;

            case input::Action::CONFIRM_POLYGON:
                if (dispatchActionToModal(bound_action, mods, mx, my)) {
                    return;
                }
                if (tool_context_) {
                    if (auto* sm = tool_context_->getSceneManager()) {
                        if (auto* selection_service = sm->getSelectionService();
                            selection_service &&
                            selection_service->isInteractiveSelectionActive() &&
                            selection_service->getInteractiveSelectionShape() == SelectionShape::Polygon) {
                            if (mods & input::KEYMOD_SHIFT) {
                                selection_service->setInteractiveSelectionMode(SelectionMode::Add);
                            } else if (mods & input::KEYMOD_CTRL) {
                                selection_service->setInteractiveSelectionMode(SelectionMode::Remove);
                            } else if (mods & input::KEYMOD_ALT) {
                                selection_service->setInteractiveSelectionMode(SelectionMode::Intersect);
                            } else {
                                selection_service->setInteractiveSelectionMode(SelectionMode::Replace);
                            }
                            (void)selection_service->finishInteractiveSelection();
                            return;
                        }
                    }
                }
                return;

            case input::Action::UNDO_POLYGON_VERTEX:
                if (dispatchActionToModal(bound_action, mods, mx, my)) {
                    return;
                }
                if (tool_context_) {
                    if (auto* sm = tool_context_->getSceneManager()) {
                        if (auto* selection_service = sm->getSelectionService();
                            selection_service &&
                            selection_service->isInteractiveSelectionActive() &&
                            selection_service->getInteractiveSelectionShape() == SelectionShape::Polygon) {
                            if (!selection_service->undoInteractivePolygonVertex()) {
                                selection_service->cancelInteractiveSelection();
                            }
                            return;
                        }
                    }
                }
                return;

            case input::Action::DELETE_SELECTED:
                if (tool_context_) {
                    if (auto* sm = tool_context_->getSceneManager(); sm) {
                        if (selection_tool_ && selection_tool_->isEnabled()) {
                            if (auto* selection_service = sm->getSelectionService();
                                selection_service && selection_service->isInteractiveSelectionActive()) {
                                if (!selection_service->finishInteractiveSelection().success) {
                                    return;
                                }
                            }
                            if (!sm->getScene().hasSelection()) {
                                return;
                            }
                        } else if (!sm->getScene().hasSelection()) {
                            const auto selected = sm->getSelectedNodeNames();
                            if (!selected.empty()) {
                                for (const auto& name : selected)
                                    cmd::RemovePLY{.name = name, .keep_children = false}.emit();
                                return;
                            }
                        }
                    }
                }
                cmd::DeleteSelected{}.emit();
                return;

            case input::Action::DELETE_NODE:
                if (gui && gui->requestDeleteSceneSelectionIfAvailable())
                    return;
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

            case input::Action::CUT_SELECTION:
                cmd::CutSelection{}.emit();
                return;

            case input::Action::TOGGLE_PERFORMANCE_HUD:
                lfs::core::events::ui::ToggleVramHud{}.emit();
                return;

            case input::Action::PASTE_SELECTION:
                cmd::PasteSelection{}.emit();
                return;

            case input::Action::APPLY_CROP_BOX: {
                // Apply the selected polygon or ellipsoid, otherwise the cropbox
                if (tool_context_) {
                    if (auto* sm = tool_context_->getSceneManager()) {
                        if (sm->getSelectedNodeCropPolygonId() != core::NULL_NODE) {
                            cmd::ApplyCropPolygon{}.emit();
                            return;
                        }
                        const auto ellipsoid_id = sm->getSelectedNodeEllipsoidId();
                        if (ellipsoid_id != core::NULL_NODE) {
                            cmd::ApplyEllipsoid{}.emit();
                            return;
                        }
                    }
                }
                cmd::ApplyCropBox{}.emit();
                return;
            }

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

            case input::Action::TOGGLE_UI:
                ui::ToggleUI{}.emit();
                return;

            case input::Action::TOGGLE_FULLSCREEN:
                ui::ToggleFullscreen{}.emit();
                return;

            case input::Action::SEQUENCER_ADD_KEYFRAME:
                cmd::SequencerAddKeyframe{}.emit();
                return;

            case input::Action::SEQUENCER_UPDATE_KEYFRAME:
                cmd::SequencerUpdateKeyframe{}.emit();
                return;

            case input::Action::SEQUENCER_PLAY_PAUSE:
                cmd::SequencerPlayPause{}.emit();
                return;

            case input::Action::PIE_MENU:
                if (gui) {
                    float px, py;
                    input::mouseStateInPixels(window_, &px, &py);
                    gui->gizmo().openPieMenu({px, py});
                }
                return;

            default:
                break;
            }
        }

        // Movement keys work unless text is being edited or a gizmo is active
        if (!shouldCameraHandleInput() || drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter)
            return;

        // Keep physical movement layout-independent, but do not start it from control/meta chords.
        if ((mods & (input::KEYMOD_CTRL | input::KEYMOD_ALT | input::KEYMOD_SUPER)) != 0)
            return;

        armPhysicalMovementKey(physical_key, action != input::ACTION_RELEASE);
    }

    bool InputController::physicalMovementConsumesLogicalKey(
        const int physical_key, const int logical_key, const int mods) const {
        if (physical_key < 0 || physical_key == logical_key)
            return false;
        if ((mods & (input::KEYMOD_CTRL | input::KEYMOD_ALT | input::KEYMOD_SUPER)) != 0)
            return false;
        if (!shouldCameraHandleInput() || drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter)
            return false;
        const auto& mk = currentMovementKeys();
        return physical_key == mk.forward || physical_key == mk.backward ||
               physical_key == mk.left || physical_key == mk.right ||
               physical_key == mk.up || physical_key == mk.down;
    }

    void InputController::armPhysicalMovementKey(const int physical_key, const bool pressed) {
        const auto& mk = currentMovementKeys();
        if (physical_key == mk.forward)
            keys_movement_[0] = pressed;
        else if (physical_key == mk.left)
            keys_movement_[1] = pressed;
        else if (physical_key == mk.backward)
            keys_movement_[2] = pressed;
        else if (physical_key == mk.right)
            keys_movement_[3] = pressed;
        else if (physical_key == mk.down)
            keys_movement_[5] = pressed;
        else if (physical_key == mk.up)
            keys_movement_[4] = pressed;
    }

    void InputController::decayHeldDragMomentum() {
        const auto now = std::chrono::steady_clock::now();
        const float elapsed = std::chrono::duration<float>(now - drag_momentum_updated_at_).count();
        drag_momentum_updated_at_ = now;
        // Pausing a drag must still dissipate its velocity when the frame loop
        // sleeps. Use elapsed time, not the animation timestep (which is clamped).
        if (drag_mode_ == DragMode::Orbit && orbit_coast_viewport_)
            orbit_coast_viewport_->camera.decayOrbitMomentum(elapsed);
        if (drag_mode_ == DragMode::Pan && pan_coast_viewport_)
            pan_coast_viewport_->camera.decayPanMomentum(elapsed);
    }

    bool InputController::advanceCameraView(const int delta) {
        if (const auto* scene_manager = services().sceneOrNull()) {
            const auto cameras_snapshot = scene_manager->getScene().getAllCamerasCached();
            const auto& cameras = *cameras_snapshot;
            if (!cameras.empty()) {
                const auto* rendering = services().renderingOrNull();
                const int current_uid = rendering ? rendering->getCurrentCameraId() : last_camview_;
                const auto current = std::ranges::find_if(cameras, [current_uid](const auto& camera) {
                    return camera->uid() == current_uid;
                });
                const int count = static_cast<int>(cameras.size());
                const int index = current == cameras.end()
                                      ? (delta > 0 ? 0 : count - 1)
                                      : (static_cast<int>(std::distance(cameras.begin(), current)) + delta + count) % count;
                cmd::GoToCamView{.cam_id = cameras[index]->uid()}.emit();
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
                return true;
            }
        }
        return false;
    }

    void InputController::updateCameraBrowseRepeat() {
        if (!camera_browse_repeat_.active())
            return;
        auto& preferences = UserPreferences::instance();
        // The viewport owned the press. A transient unassigned focus does not
        // revoke that hold; explicit GUI focus, text editing and modals do.
        if (!input_router_ || input_router_->keyboardFocus() == input::InputTarget::Gui ||
            input_router_->isTextInputActive() || input_router_->isModalOpen() ||
            bindings_.isCapturing() || lfs::python::has_keyboard_capture_request() ||
            !preferences.cameraBrowseRepeat() ||
            !isKeyPressed(camera_browse_repeat_.key()) || getModifierKeys() != camera_browse_repeat_modifiers_) {
            camera_browse_repeat_.stop();
            return;
        }
        if (const int delta = camera_browse_repeat_.poll(input::CameraBrowseRepeat::Clock::now(),
                                                         preferences.cameraBrowseRepeatRate());
            delta != 0 && !advanceCameraView(delta)) {
            camera_browse_repeat_.stop();
        }
    }

    bool InputController::configureGamepadNavigation(input::GamepadNavigationSettings settings, SDL_JoystickID device) {
        settings.validate();
        auto* window = services().windowOrNull();
        if (!window) {
            stopGamepadMotion();
            return false;
        }
        const auto previous_device = window->gamepads().navigationSelected();
        window->updateGamepadService(settings.enabled || gamepad_settings_.enabled);
        if (!window->gamepads().selectNavigation(device)) {
            if (settings.enabled) {
                window->updateGamepadService(gamepad_settings_.enabled);
                return false;
            }
            (void)window->gamepads().selectNavigation(0);
        }
        suspendGamepadNavigation();
        if (settings.json() != gamepad_settings_.json() || !gamepad_configuration_error_.empty()) {
            if (const auto result = UserPreferences::instance().setGamepadNavigation(settings.json().dump()); !result) {
                (void)window->gamepads().selectNavigation(previous_device);
                window->updateGamepadService(gamepad_settings_.enabled);
                throw std::runtime_error(lfs::format_for_developer(result.error()));
            }
            gamepad_configuration_error_.clear();
        }
        settings.revision = gamepad_settings_.revision + 1;
        gamepad_settings_ = settings;
        window->updateGamepadService(gamepad_settings_.enabled);
        python::request_redraw();
        return true;
    }

    bool InputController::configureGamepadProfiles(const std::string& serialized, const bool disable, const std::string& expected) {
        const auto profiles = input::GamepadProfiles::fromJson(nlohmann::json::parse(serialized));
        auto settings = profiles.activeSettings(disable ? false : gamepad_settings_.enabled);
        if (auto result = UserPreferences::instance().setGamepadProfiles(profiles.value.dump(), settings.enabled, expected); !result)
            throw std::runtime_error(lfs::format_for_developer(result.error()));
        suspendGamepadNavigation();
        settings.revision = gamepad_settings_.revision + 1;
        gamepad_settings_ = std::move(settings);
        if (auto* window = services().windowOrNull())
            window->updateGamepadService(gamepad_settings_.enabled);
        gamepad_configuration_error_.clear();
        python::request_redraw();
        return true;
    }

    void InputController::stopGamepadMotion() {
        if (gamepad_owned_) {
            auto& target = *gamepad_motion_viewport_;
            if (camera_navigation_mode_ == CameraNavigationMode::Drone)
                target.camera.finishDrone();
            target.camera.clearTransientMotion();
            if (wasd_momentum_viewport_ == &target)
                wasd_momentum_viewport_ = nullptr;
            gamepad_owned_ = false;
            publishCameraMove(&target);
            onCameraMovementEnd();
        }
        gamepad_frame_demand_ = false;
    }

    SDL_JoystickID InputController::gamepadNavigationDevice() const {
        auto* window = services().windowOrNull();
        return window ? window->gamepads().navigationSelected() : 0;
    }

    SplitViewPanelId InputController::gamepadNavigationPanel() const {
        return gamepad_panel_;
    }

    Viewport& InputController::gamepadViewport() const {
        if (auto* rendering = services().renderingOrNull())
            return rendering->resolvePanelViewport(viewport_, gamepad_panel_);
        return viewport_;
    }

    void InputController::pauseGamepadNavigation() {
        stopGamepadMotion();
        gamepad_session_.pause();
    }

    void InputController::suspendGamepadNavigation() {
        stopGamepadMotion();
        gamepad_session_.suspend(gamepad_click_epoch_);
        gamepad_click_candidate_ = false;
        gamepad_panel_ = SplitViewPanelId::Left;
    }

    bool InputController::updateGamepadNavigation(float delta_time) {
        if (!gamepad_settings_.enabled) {
            stopGamepadMotion();
            return false;
        }
        // Selection/generation is session-local. A remembered model only chooses
        // settings; it never enables navigation or grants viewport ownership.
        if (auto* window = services().windowOrNull()) {
            const auto& sample = window->gamepads().navigationSample();
            if (sample.id != gamepad_profile_device_ || sample.generation != gamepad_profile_generation_) {
                gamepad_profile_device_ = sample.id;
                gamepad_profile_generation_ = sample.generation;
                suspendGamepadNavigation();
                try {
                    auto profiles = input::GamepadProfiles::fromJson(nlohmann::json::parse(UserPreferences::instance().gamepadProfiles()));
                    const auto& links = profiles.value["device_profiles"];
                    const auto match = links.find(window->gamepads().navigationProfileKey());
                    if (match != links.end() && *match != profiles.value["active"]) {
                        profiles.value["active"] = *match;
                        configureGamepadProfiles(profiles.value.dump());
                    }
                } catch (const std::exception& error) {
                    gamepad_configuration_error_ = error.what();
                    LOG_WARN("Gamepad device profile was not loaded: {}", error.what());
                }
            }
        }
        auto* window = services().windowOrNull();
        if (!window) {
            stopGamepadMotion();
            return false;
        }
        const auto dialog_epoch = gui::nativeFileDialogEpoch();
        if (dialog_epoch != gamepad_dialog_epoch_) {
            pauseGamepadNavigation();
            gamepad_dialog_epoch_ = dialog_epoch;
        }
        auto& target = gamepadViewport();
        auto& camera = target.camera;
        auto* gui = services().guiOrNull();
        const auto gui_state = gui ? gui->inputState() : gui::GuiInputState{};
        const bool keys_held = std::any_of(std::begin(keys_movement_), std::end(keys_movement_), [](bool held) { return held; });
        const auto flags = window_ ? SDL_GetWindowFlags(window_) : 0;
        float pointer_x = 0, pointer_y = 0;
        const bool mouse_down = input::mouseStateInPixels(window_, &pointer_x, &pointer_y) != 0;
        const bool viewport_mouse = mouse_down && input_router_ &&
                                    input_router_->pointerTarget(pointer_x, pointer_y) == input::InputTarget::Viewport;
        input::GamepadNavigationContext context;
        // Native floating panels are SDL windows in this process too. Moving
        // focus to the Sequencer must not look like leaving the application.
        context.allowed = window_ && SDL_GetKeyboardFocus() && !(flags & SDL_WINDOW_MINIMIZED) &&
                          !gui_state.modal_open && !gui::nativeFileDialogActive() &&
                          !keys_held && !viewport_mouse && drag_mode_ == DragMode::None &&
                          !bindings_.isCapturing() && !lfs::python::has_keyboard_capture_request() &&
                          !op::operators().hasModalOperator() &&
                          !gui::preserveNativeDragInput() && window->manualResizeEdgeMask() == 0 &&
                          !gui::PanelRegistry::instance().is_panel_enabled("lfs.preferences") &&
                          (!gui || (!gui->sequencer().isPlaying() && gui->sequencer().state() != PlaybackState::SCRUBBING)) &&
                          window->gamepads().diagnosticSelected() == 0;
        context.target = reinterpret_cast<uintptr_t>(&target);
        context.click_epoch = gamepad_click_epoch_;
        context.sequencer_open = gui && gui->isSequencerVisible() && !gui_state.text_input_active;
        context.sequencer_commands_allowed = !gui || !gui->sequencerUI().isTimelineGestureActive();
        switch (camera_navigation_mode_) {
        case CameraNavigationMode::Orbit: context.mode = input::GamepadNavigationMode::Orbit; break;
        case CameraNavigationMode::Trackball: context.mode = input::GamepadNavigationMode::Trackball; break;
        case CameraNavigationMode::FPV: context.mode = input::GamepadNavigationMode::FPV; break;
        case CameraNavigationMode::Drone: context.mode = input::GamepadNavigationMode::Drone; break;
        case CameraNavigationMode::Walk: context.mode = input::GamepadNavigationMode::Walk; break;
        }
        auto* rendering = services().renderingOrNull();
        const auto projection = rendering ? rendering->getCameraProjectionSettings()
                                          : RenderingManager::CameraProjectionSettings{};
        context.supported_projection = !projection.orthographic ||
                                       camera_navigation_mode_ == CameraNavigationMode::Orbit || camera_navigation_mode_ == CameraNavigationMode::Trackball;
        const auto intent = gamepad_session_.update(window->gamepads().navigationSample(), gamepad_settings_, context, delta_time);
        const auto status = std::string_view(gamepad_session_.status());
        if (status != "ready" && status != "moving") {
            stopGamepadMotion();
            gamepad_frame_demand_ = gamepad_session_.awaitingNeutral();
            return false;
        }
        if (context.sequencer_open && (intent.add_keyframe || intent.remove_selected_keyframe || intent.deselect_keyframe || intent.update_selected_keyframe || intent.keyframe_step)) {
            stopGamepadMotion();
            auto& sequencer_ui = gui->sequencerUI();
            if (intent.add_keyframe)
                sequencer_ui.captureKeyframeFromView(target);
            else if (intent.remove_selected_keyframe)
                sequencer_ui.removeSelectedKeyframe();
            else if (intent.deselect_keyframe)
                sequencer_ui.deselectKeyframe();
            else if (intent.update_selected_keyframe)
                sequencer_ui.updateSelectedKeyframeFromView(target);
            else
                sequencer_ui.selectAdjacentKeyframe(intent.keyframe_step, &target);
            gamepad_frame_demand_ = true;
            return true;
        }
        if (!context.supported_projection) {
            // Editing remains available, but old perspective-camera momentum
            // must not continue after switching to an incompatible projection.
            stopGamepadMotion();
            return false;
        }
        if (intent.exit_walk) {
            setCameraNavigationMode(CameraNavigationMode::Orbit);
            return true;
        }
        if (intent.navigation_step) {
            const bool changed = input::cycleGamepadNavigation(context.mode, intent.navigation_step, projection.orthographic,
                                                               [&](input::GamepadNavigationMode candidate) {
                                                                   const auto mode = cameraNavigationModeFromName(std::string(input::gamepad_modes[static_cast<size_t>(candidate)]));
                                                                   return mode && setCameraNavigationModeImpl(*mode, false, &target);
                                                               });
            if (changed) {
                // This deliberate command came from an armed viewport. Keep its
                // ownership, but require released controls before resuming motion.
                gamepad_frame_demand_ = true;
            }
            return true;
        }
        if (intent.rendering_step && rendering) {
            const auto& settings = rendering->getSettings();
            const int current = settings.point_cloud_mode ? 1 : settings.show_rings        ? 2
                                                            : settings.show_center_markers ? 3
                                                                                           : 0;
            constexpr std::array<std::string_view, 4> modes{"splats", "points", "rings", "centers"};
            rendering->setViewportRenderMode(modes[(current + intent.rendering_step + 4) % 4]);
            return true;
        }
        const bool settling = gamepad_owned_ &&
                              (camera.hasWasdMomentum() || camera.hasDroneMotion());
        if (!intent.active && !settling) {
            if (gamepad_owned_) {
                gamepad_owned_ = false;
                onCameraMovementEnd();
            }
            gamepad_frame_demand_ = false;
            return false;
        }
        const bool acquiring = !gamepad_owned_;
        if (acquiring) {
            clearWasdMomentumViewport();
            if (orbit_coast_viewport_)
                orbit_coast_viewport_->camera.clearOrbitMomentum();
            if (pan_coast_viewport_)
                pan_coast_viewport_->camera.clearPanMomentum();
            orbit_coast_viewport_ = pan_coast_viewport_ = nullptr;
            camera.clearTransientMotion();
            if (camera_navigation_mode_ == CameraNavigationMode::Drone)
                camera.enterDrone();
            gamepad_motion_viewport_ = &target;
            gamepad_owned_ = true;
        }
        const float dt = std::min(delta_time, 0.05f);
        const auto movement = input::NavigationMovement::fromAnalog({intent.move.x, intent.vertical, intent.move.y});
        const float additional_speed = (intent.speed - 1.0f) * camera.getWasdSpeed();
        switch (camera_navigation_mode_) {
        case CameraNavigationMode::Orbit:
        case CameraNavigationMode::Trackball: {
            camera.advanceOrbit(dt, intent.look, camera_navigation_mode_ == CameraNavigationMode::Trackball);
            float pan_factor = 0.5f * intent.speed;
            if (projection.orthographic && target.windowSize.y > 0) {
                const float scale = target.ortho_scale_override.value_or(projection.ortho_scale);
                const float distance = std::max(glm::length(camera.pivot - camera.t), 0.01f);
                const float perspective_scale = target.windowSize.y /
                                                (2.0f * distance * std::tan(glm::radians(lfs::rendering::focalLengthToVFov(projection.focal_length_mm)) * 0.5f));
                pan_factor *= perspective_scale / std::max(scale, 1e-6f);
                const float next_scale = camera.advanceOrthographicDolly(dt, intent.vertical * 0.8f * intent.speed, scale);
                if (next_scale != scale) {
                    if (&target != &viewport_ || target.ortho_scale_override)
                        target.ortho_scale_override = next_scale;
                    if (&target == &viewport_ && rendering)
                        rendering->setOrthographicScale(next_scale);
                }
            } else {
                camera.advanceDolly(dt, intent.vertical * 0.8f * intent.speed);
            }
            camera.advancePan(dt, intent.move * pan_factor);
            break;
        }
        case CameraNavigationMode::FPV:
            camera.advanceLook(dt, intent.look);
            camera.advanceWasd(dt, movement, additional_speed);
            break;
        case CameraNavigationMode::Drone:
            camera.advanceDroneLook(dt, intent.look);
            camera.advanceDrone(dt, movement, additional_speed);
            break;
        case CameraNavigationMode::Walk:
            camera.advanceLook(dt, intent.look);
            try {
                ensureWalkViewport(target);
                core::advanceWalk(walk_state_, walk_settings, camera.navigationUp(),
                                  camera.walkVelocity(movement, walk_settings.speed * intent.speed), dt, walkQuery());
                const auto delta = walk_state_.presentation - camera.t;
                camera.t += delta;
                camera.pivot += delta;
            } catch (const std::exception& e) {
                walk_error = core::localizedCollisionError(e);
                LOG_ERROR("Walk stopped: {}", e.what());
                setCameraNavigationMode(CameraNavigationMode::Orbit);
                return true;
            }
            break;
        }
        gamepad_frame_demand_ = intent.active || camera.hasWasdMomentum() || camera.hasDroneMotion();
        onCameraMovementStart();
        publishCameraMove(&target);
        return true;
    }

    void InputController::update(float delta_time) {
        if (vr_viewing_)
            return;
        updateCameraBrowseRepeat();
        decayHeldDragMomentum();
        // Refresh once per input frame; mouse-move events can then query the
        // divider without taking RenderingManager's settings mutex repeatedly.
        refreshSplitDividerCache();
        maybeInitializeDepthViewRange();

        if (input_router_) {
            const bool any_mouse_buttons_pressed = SDL_GetMouseState(nullptr, nullptr) != 0;
            input_router_->syncPressedMouseButtons(any_mouse_buttons_pressed);
        }

        if (pending_camera_context_menu_.active) {
            const bool right_button_down =
                isMouseButtonPressed(static_cast<int>(input::AppMouseButton::RIGHT));
            if (pending_camera_context_menu_.released) {
                if (right_button_down) {
                    clearSelectedCameraContextMenuGesture();
                } else {
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed =
                        std::chrono::duration<double>(
                            now - pending_camera_context_menu_.release_time)
                            .count();
                    if (elapsed >= DOUBLE_CLICK_TIME) {
                        const int camera_uid = pending_camera_context_menu_.camera_uid;
                        const glm::dvec2 release_pos = pending_camera_context_menu_.release_pos;
                        clearSelectedCameraContextMenuGesture();
                        openSelectedCameraContextMenu(camera_uid,
                                                      static_cast<float>(release_pos.x),
                                                      static_cast<float>(release_pos.y));
                    }
                }
            } else if (!right_button_down) {
                clearSelectedCameraContextMenuGesture();
            }
        }

        const bool drag_button_released = drag_button_ >= 0 &&
                                          !isMouseButtonPressed(drag_button_);

        // Handle missed mouse release events (e.g., outside window)
        if (drag_mode_ == DragMode::Orbit && drag_button_released) {
            auto* const released_viewport = drag_viewport_ ? drag_viewport_ : &viewport_;
            const SplitViewPanelId released_panel = drag_split_panel_;
            if (drag_viewport_) {
                drag_viewport_->camera.endRotateAroundCenter();
                snapViewportToNearestAxis(*drag_viewport_, released_panel);
            } else {
                viewport_.camera.endRotateAroundCenter();
                snapViewportToNearestAxis(viewport_, released_panel);
            }
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
            drag_viewport_ = nullptr;
            press_selected_camera_frustum_ = false;
            pressed_camera_frustum_id_ = -1;
            pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;

            ui::CameraMove{
                .rotation = released_viewport->getRotationMatrix(),
                .translation = released_viewport->getTranslation()}
                .emit();
            onCameraMovementEnd();
        }

        if (drag_mode_ == DragMode::Pan && drag_button_released) {
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
            drag_viewport_ = nullptr;
            press_selected_camera_frustum_ = false;
            pressed_camera_frustum_id_ = -1;
            pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;
        }

        if (drag_mode_ == DragMode::Rotate && drag_button_released) {
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
            drag_viewport_ = nullptr;
            press_selected_camera_frustum_ = false;
            pressed_camera_frustum_id_ = -1;
            pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;
        }
        syncLookPointerLock();

        if (drag_mode_ == DragMode::Splitter &&
            !isMouseButtonPressed(static_cast<int>(input::AppMouseButton::LEFT))) {
            drag_mode_ = DragMode::None;
            drag_button_ = -1;
            press_selected_camera_frustum_ = false;
            pressed_camera_frustum_id_ = -1;
            pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;
            SDL_SetCursor(SDL_GetDefaultCursor());
            current_cursor_ = CursorType::Default;
        }

        // Sync movement key states with actual keyboard (using cached keys)
        if (window_) {
            const auto& mk = currentMovementKeys();
            if (keys_movement_[0] && (mk.forward < 0 || !isKeyPressed(mk.forward))) {
                keys_movement_[0] = false;
            }
            if (keys_movement_[1] && (mk.left < 0 || !isKeyPressed(mk.left))) {
                keys_movement_[1] = false;
            }
            if (keys_movement_[2] && (mk.backward < 0 || !isKeyPressed(mk.backward))) {
                keys_movement_[2] = false;
            }
            if (keys_movement_[3] && (mk.right < 0 || !isKeyPressed(mk.right))) {
                keys_movement_[3] = false;
            }
            if (keys_movement_[4] && (mk.up < 0 || !isKeyPressed(mk.up))) {
                keys_movement_[4] = false;
            }
            if (keys_movement_[5] && (mk.down < 0 || !isKeyPressed(mk.down))) {
                keys_movement_[5] = false;
            }
        }

        if (updateGamepadNavigation(delta_time)) {
            checkCameraMovementTimeout();
            return;
        }

        // Orbit ease-out: while dragging, let a held-still pause fade the stored
        // motion; once released, coast the remembered rotation to a smooth stop.
        if (orbit_coast_viewport_ && drag_mode_ != DragMode::Orbit) {
            auto& orbit_camera = orbit_coast_viewport_->camera;
            if (orbit_camera.hasOrbitMomentum()) {
                orbit_camera.updateOrbitCoast(delta_time);
                onCameraMovementStart();
                publishCameraMove(orbit_coast_viewport_);
                if (!orbit_camera.hasOrbitMomentum()) {
                    ui::CameraMove{
                        .rotation = orbit_coast_viewport_->getRotationMatrix(),
                        .translation = orbit_coast_viewport_->getTranslation()}
                        .emit();
                    orbit_coast_viewport_ = nullptr;
                }
            } else {
                orbit_coast_viewport_ = nullptr;
            }
        }

        // Pan ease-out: mirror the orbit coast for click-drag panning. While the
        // button is held a paused drag fades the stored motion; once released the
        // remembered translation coasts to a smooth stop.
        if (pan_coast_viewport_ && drag_mode_ != DragMode::Pan) {
            auto& pan_camera = pan_coast_viewport_->camera;
            if (pan_camera.hasPanMomentum()) {
                pan_camera.updatePanCoast(delta_time);
                onCameraMovementStart();
                publishCameraMove(pan_coast_viewport_);
                if (!pan_camera.hasPanMomentum()) {
                    ui::CameraMove{
                        .rotation = pan_coast_viewport_->getRotationMatrix(),
                        .translation = pan_coast_viewport_->getTranslation()}
                        .emit();
                    pan_coast_viewport_ = nullptr;
                }
            } else {
                pan_coast_viewport_ = nullptr;
            }
        }

        // Drive the set-pivot recenter glide
        if (auto& glide_viewport = activeKeyboardViewport(); glide_viewport.camera.isGliding()) {
            const bool movement_keys_active = keys_movement_[0] || keys_movement_[1] || keys_movement_[2] ||
                                              keys_movement_[3] || keys_movement_[4] || keys_movement_[5];
            if (movement_keys_active) {
                glide_viewport.camera.finishGlide();
            } else {
                glide_viewport.camera.updateGlide(delta_time);
            }
            onCameraMovementStart();
            publishCameraMove(&glide_viewport);
            if (!glide_viewport.camera.isGliding()) {
                ui::CameraMove{
                    .rotation = glide_viewport.getRotationMatrix(),
                    .translation = glide_viewport.getTranslation()}
                    .emit();
            }
        }

        // Handle continuous WASD movement. The camera carries a damped velocity,
        // so motion eases in on press and glides to rest on release. Drive it
        // every frame while momentum remains so a released key decays to zero.
        auto* active_movement_viewport = &activeKeyboardViewport();
        const bool camera_can_move = shouldCameraHandleInput() &&
                                     drag_mode_ != DragMode::Gizmo && drag_mode_ != DragMode::Splitter;
        const bool keys_active = camera_can_move &&
                                 (keys_movement_[0] || keys_movement_[1] || keys_movement_[2] ||
                                  keys_movement_[3] || keys_movement_[4] || keys_movement_[5]);

        if (keys_active) {
            if (wasd_momentum_viewport_ && wasd_momentum_viewport_ != active_movement_viewport) {
                wasd_momentum_viewport_->camera.clearWasdMomentum();
                wasd_momentum_viewport_->camera.clearDroneMotion();
            }
            wasd_momentum_viewport_ = active_movement_viewport;
        }

        const float movement_speed_bonus =
            (keys_active && (getModifierKeys() & input::KEYMOD_SHIFT) != 0) ? kWasdShiftSpeedBonus : 0.0f;

        if (camera_navigation_mode_ == CameraNavigationMode::Walk) {
            // Neutral input continues the current owner's gravity/settling. Only
            // deliberate keyboard motion may hand Walk over to the keyboard view.
            if (!keys_active && walk_viewport_)
                active_movement_viewport = walk_viewport_;
            auto& camera = active_movement_viewport->camera;
            try {
                ensureWalkViewport(*active_movement_viewport);
                const auto up = camera.navigationUp();
                const auto movement = input::NavigationMovement::fromKeys(
                    keys_active && keys_movement_[0], keys_active && keys_movement_[2],
                    keys_active && keys_movement_[1], keys_active && keys_movement_[3],
                    false, false);
                const auto velocity = camera.walkVelocity(movement, walk_settings.speed);
                core::advanceWalk(walk_state_, walk_settings, up, velocity, delta_time, walkQuery());
                const auto delta = walk_state_.presentation - camera.t;
                camera.t += delta;
                camera.pivot += delta;
                if (glm::length(delta) > 0)
                    publishCameraMove(active_movement_viewport);
            } catch (const std::exception& e) {
                walk_error = core::localizedCollisionError(e);
                LOG_ERROR("Walk stopped: {}", e.what());
                setCameraNavigationMode(CameraNavigationMode::Orbit);
            }
        } else if (camera_navigation_mode_ == CameraNavigationMode::Drone) {
            // The drone must keep integrating with no keys held: braking,
            // leveling and mouse-look smoothing all settle through
            // advanceDrone until hasDroneMotion() reaches false.
            auto* drone_viewport = keys_active ? active_movement_viewport : wasd_momentum_viewport_;
            if (!drone_viewport && active_movement_viewport->camera.hasDroneMotion())
                drone_viewport = active_movement_viewport;
            if (drone_viewport && (keys_active || drone_viewport->camera.hasDroneMotion())) {
                drone_viewport->camera.advanceDrone(
                    delta_time,
                    keys_active && keys_movement_[0],
                    keys_active && keys_movement_[2],
                    keys_active && keys_movement_[1],
                    keys_active && keys_movement_[3],
                    keys_active && keys_movement_[4],
                    keys_active && keys_movement_[5],
                    movement_speed_bonus);

                wasd_momentum_viewport_ = drone_viewport;
                onCameraMovementStart();
                publishCameraMove(drone_viewport);

                if (!keys_active && !drone_viewport->camera.hasDroneMotion()) {
                    ui::CameraMove{
                        .rotation = drone_viewport->getRotationMatrix(),
                        .translation = drone_viewport->getTranslation()}
                        .emit();
                    wasd_momentum_viewport_ = nullptr;
                }
            } else if (!keys_active) {
                wasd_momentum_viewport_ = nullptr;
            }
        } else {
            auto* const movement_viewport = keys_active ? active_movement_viewport : wasd_momentum_viewport_;
            if (movement_viewport && (keys_active || movement_viewport->camera.hasWasdMomentum())) {
                const bool move_forward = keys_active && keys_movement_[0];
                const bool move_backward = keys_active && keys_movement_[2];
                const bool move_left = keys_active && keys_movement_[1];
                const bool move_right = keys_active && keys_movement_[3];
                const bool move_up = keys_active && keys_movement_[4];
                const bool move_down = keys_active && keys_movement_[5];
                auto* const rendering = services().renderingOrNull();
                const auto projection = rendering ? rendering->getCameraProjectionSettings()
                                                  : RenderingManager::CameraProjectionSettings{};
                if (rendering && projection.orthographic && movement_viewport->windowSize.y > 0) {
                    const float distance = std::max(
                        glm::length(movement_viewport->camera.pivot - movement_viewport->camera.t), 0.01f);
                    const float half_tan_fov = std::tan(glm::radians(
                                                            lfs::rendering::focalLengthToVFov(projection.focal_length_mm)) *
                                                        0.5f);
                    const float perspective_scale =
                        static_cast<float>(movement_viewport->windowSize.y) / (2.0f * distance * half_tan_fov);
                    const float current_scale = movement_viewport->ortho_scale_override.value_or(projection.ortho_scale);
                    const float next_scale = movement_viewport->camera.advanceOrthographicWasd(
                        delta_time, move_forward, move_backward, move_left, move_right, move_up, move_down,
                        current_scale, perspective_scale, distance, movement_speed_bonus);
                    if (next_scale != current_scale) {
                        if (movement_viewport != &viewport_ || movement_viewport->ortho_scale_override)
                            movement_viewport->ortho_scale_override = next_scale;
                        if (movement_viewport == &viewport_) {
                            rendering->setOrthographicScale(next_scale);
                        }
                    }
                } else {
                    movement_viewport->camera.advanceWasd(
                        delta_time,
                        keys_active && keys_movement_[0],
                        keys_active && keys_movement_[2],
                        keys_active && keys_movement_[1],
                        keys_active && keys_movement_[3],
                        keys_active && keys_movement_[4],
                        keys_active && keys_movement_[5],
                        movement_speed_bonus);
                }

                onCameraMovementStart();
                publishCameraMove(movement_viewport);

                if (!keys_active && !movement_viewport->camera.hasWasdMomentum()) {
                    ui::CameraMove{
                        .rotation = movement_viewport->getRotationMatrix(),
                        .translation = movement_viewport->getTranslation()}
                        .emit();
                    wasd_momentum_viewport_ = nullptr;
                }
            } else if (!keys_active) {
                wasd_momentum_viewport_ = nullptr;
            }
        }

        // Check if camera movement has timed out and should resume training
        checkCameraMovementTimeout();
    }

    void InputController::handleFileDrop(const std::vector<std::string>& paths) {
        suspendGamepadNavigation();
        using namespace lfs::core::events;
        ui::FileDropReceived{}.emit();

        LOG_INFO("Processing {} dropped file(s)", paths.size());

        std::vector<std::filesystem::path> splat_files;
        std::optional<std::filesystem::path> dataset_path;
        std::optional<std::filesystem::path> environment_map_path;
        std::vector<std::string> skipped_files;
        bool skipped_video_or_project = false;
        bool has_unrecognized_files = false;

        if (paths.size() == 1) {
            const std::filesystem::path dropped_path = lfs::core::utf8_to_path(paths.front());
            auto extension = lfs::core::path_to_utf8(dropped_path.extension());
            std::ranges::transform(
                extension, extension.begin(),
                [](const unsigned char character) {
                    return static_cast<char>(
                        std::tolower(character));
                });
            if (extension == ".licht") {
                auto& panels = gui::PanelRegistry::instance();
                if (panels.is_panel_enabled("lfs.asset_manager")) {
                    if (const auto panel = panels.get_panel_instance("lfs.asset_manager");
                        panel && panel->onViewportDrop(
                                     "application/x-lichtfeld-project-file", paths.front())) {
                        LOG_INFO("Added project to Asset Manager via drag-and-drop: {}",
                                 lfs::core::path_to_utf8(dropped_path.filename()));
                        return;
                    }
                }
                // Unpublished *.tmp.licht names still emit ProjectOpen so
                // lifecycle can reject with unpublishedLichtUserMessage.
                cmd::ProjectOpen{.path = dropped_path}.emit();
                LOG_INFO(
                    "Opening project via drag-and-drop: {}",
                    lfs::core::path_to_utf8(
                        dropped_path.filename()));
                return;
            }
            if (lfs::io::video::is_supported_video_extension(extension)) {
                cmd::ShowVideoExtractor{.video_path = dropped_path}.emit();
                LOG_INFO("Opening video extractor via drag-and-drop: {}",
                         lfs::core::path_to_utf8(dropped_path.filename()));
                return;
            }
        }

        for (const auto& path_str : paths) {
            std::filesystem::path filepath = lfs::core::utf8_to_path(path_str);
            LOG_DEBUG("Processing dropped file: {}", lfs::core::path_to_utf8(filepath));

            auto ext = lfs::core::path_to_utf8(filepath.extension());
            std::ranges::transform(
                ext, ext.begin(), [](const unsigned char character) {
                    return static_cast<char>(std::tolower(character));
                });

            if ((ext == ".licht" || lfs::io::video::is_supported_video_extension(ext)) &&
                !lfs::core::native_fs::is_directory(filepath)) {
                skipped_files.push_back(path_str);
                skipped_video_or_project = true;
                continue;
            }

            if (ext == ".resume") {
                cmd::ShowResumeCheckpointPopup{.checkpoint_path = filepath}.emit();
                continue;
            } else if (lfs::io::is_ssog_path(filepath)) {
                splat_files.push_back(filepath);
            } else if (ext == ".json") {
                if (lfs::io::Loader::isDatasetPath(filepath)) {
                    dataset_path = filepath;
                } else {
                    cmd::LoadConfigFile{.path = filepath}.emit();
                    LOG_INFO("Loading config via drag-and-drop: {}", lfs::core::path_to_utf8(filepath.filename()));
                    continue;
                }
            } else if (isEnvironmentMapExtension(ext)) {
                if (!environment_map_path) {
                    environment_map_path = filepath;
                } else {
                    LOG_DEBUG("Ignoring additional dropped environment map: {}", lfs::core::path_to_utf8(filepath));
                }
            } else if (ext == ".ply" || ext == ".sog" || ext == ".ssog" || ext == ".spz" || ext == ".rad" ||
                       ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz") {
                splat_files.push_back(filepath);
            } else if (ext == ".obj" || ext == ".fbx" || ext == ".gltf" || ext == ".glb" ||
                       ext == ".stl" || ext == ".dae" || ext == ".3ds") {
                splat_files.push_back(filepath);
            } else if (ext == ".bin" || ext == ".txt") {
                // Check if this is a COLMAP file (cameras.bin, images.bin, etc.)
                auto filename = lfs::core::path_to_utf8(filepath.filename());
                std::ranges::transform(
                    filename, filename.begin(), [](const unsigned char character) {
                        return static_cast<char>(std::tolower(character));
                    });
                if (filename == "cameras.bin" || filename == "cameras.txt" ||
                    filename == "images.bin" || filename == "images.txt") {
                    auto parent = filepath.parent_path();
                    if (lfs::io::Loader::isColmapSparsePath(parent)) {
                        cmd::ImportColmapCameras{.sparse_path = parent}.emit();
                        LOG_INFO("Importing COLMAP cameras from: {}", lfs::core::path_to_utf8(parent));
                        continue;
                    }
                }
                skipped_files.push_back(path_str);
                has_unrecognized_files = true;
            } else if (lfs::core::native_fs::is_directory(filepath)) {
                // Check for dataset markers
                LOG_DEBUG("Checking directory for dataset markers: {}", lfs::core::path_to_utf8(filepath));
                if (lfs::io::Loader::isDatasetPath(filepath)) {
                    if (!dataset_path) {
                        dataset_path = filepath;
                    }
                } else if (lfs::io::Loader::isColmapSparsePath(filepath)) {
                    // COLMAP sparse folder - cameras only (no images required)
                    cmd::ImportColmapCameras{.sparse_path = filepath}.emit();
                    LOG_INFO("Importing COLMAP cameras from: {}", lfs::core::path_to_utf8(filepath));
                    continue;
                } else {
                    // Check if it's a SOG directory (WebP-based format)
                    if (lfs::core::native_fs::exists(filepath / "meta.json") &&
                        lfs::core::native_fs::exists(filepath / "means_l.webp")) {
                        splat_files.push_back(filepath);
                        LOG_DEBUG("Detected SOG directory: {}", lfs::core::path_to_utf8(filepath));
                    } else {
                        skipped_files.push_back(path_str);
                        has_unrecognized_files = true;
                    }
                }
            } else {
                skipped_files.push_back(path_str);
                has_unrecognized_files = true;
            }
        }

        // Load splat and mesh files supported by the generic loader path.
        // The loading service owns admission and confirmation so later drops
        // cannot open a separate confirmation ahead of its pending queue.
        if (!splat_files.empty()) {
            cmd::LoadFile{.path = splat_files.front(),
                          .is_dataset = false,
                          .paths = splat_files,
                          .user_batch = true}
                .emit();
        }

        if (dataset_path) {
            LOG_INFO("Showing New Project dialog for: {}", lfs::core::path_to_utf8(*dataset_path));
            cmd::ShowNewProjectDialog{.source_path = *dataset_path}.emit();
        }

        if (environment_map_path) {
            applyDroppedEnvironmentMap(*environment_map_path);
        }

        if (!skipped_files.empty()) {
            std::string reason;
            if (skipped_video_or_project)
                reason = LOC("file_drop.drop_video_project_separately");
            if (has_unrecognized_files) {
                if (!reason.empty())
                    reason += "\n\n";
                reason += LOCF("file_drop.supported_formats_detail",
                               lfs::io::video::supported_video_extensions_display());
            }
            LOG_DEBUG("Dropped {} file(s) that were not opened", skipped_files.size());
            state::FileDropFailed{.files = skipped_files, .error = std::move(reason)}.emit();
        }
    }

    void InputController::handleGoToCamView(const lfs::core::events::cmd::GoToCamView& event) {
        LOG_TIMER_TRACE("HandleGoToCamView");
        auto& target_viewport = activeKeyboardViewport();

        std::shared_ptr<const lfs::core::Camera> cam_data;
        if (auto* scene_mgr = services().sceneOrNull()) {
            cam_data = scene_mgr->getScene().getCameraByUid(event.cam_id);
        }
        if (!cam_data) {
            LOG_ERROR("Camera ID {} not found", event.cam_id);
            return;
        }
        if (input_router_)
            input_router_->focusViewportKeyboard();

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

        // Apply scene transform (handles user rotation/translation of the scene)
        glm::mat4 scene_transform(1.0f);
        if (auto* scene_mgr = services().sceneOrNull()) {
            if (const auto transform =
                    scene_mgr->getScene().getCameraSceneTransformByUid(cam_data->uid())) {
                scene_transform = lfs::rendering::dataWorldTransformToVisualizerWorld(*transform);
            }
        }

        const auto pose = lfs::rendering::visualizerCameraPoseFromDataWorldToCamera(
            lfs::rendering::mat3FromRowMajor3x3(R_data),
            glm::vec3(T_data[0], T_data[1], T_data[2]),
            scene_transform);

        float pivot_distance = glm::length(target_viewport.camera.getPivot() - target_viewport.camera.t);
        if (!std::isfinite(pivot_distance) || pivot_distance < 0.1f)
            pivot_distance = 5.0f;

        target_viewport.setViewMatrix(pose.rotation, pose.translation);

        target_viewport.camera.updatePivotFromCamera(pivot_distance);

        // Save as home position if this is the first camera view
        if (!target_viewport.camera.home_saved) {
            target_viewport.camera.saveHomePosition();
        }

        // Get camera intrinsics using the proper method
        const float focal_y = std::get<1>(cam_data->get_intrinsics());
        const float height = static_cast<float>(cam_data->image_height());

        // Calculate vertical FOV using the actual focal length
        const float fov_y_rad = 2.0f * std::atan(height / (2.0f * focal_y));
        const float fov_y_deg = glm::degrees(fov_y_rad);

        const bool is_equirectangular =
            cam_data->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;

        const auto focal_mm = lfs::rendering::vFovToFocalLength(fov_y_deg);
        auto render_settings_event = ui::RenderSettingsChanged{};
        render_settings_event.focal_length_mm =
            is_equirectangular ? std::nullopt : std::optional(focal_mm);
        render_settings_event.equirectangular = is_equirectangular;
        render_settings_event.emit();

        // In orthographic mode, recalculate ortho_scale to match the equivalent perspective view
        if (auto* rm = services().renderingOrNull()) {
            auto settings = rm->getSettings();
            if (settings.orthographic && !is_equirectangular) {
                const float distance_to_pivot = glm::length(target_viewport.camera.pivot - target_viewport.camera.t);
                const float half_tan_fov = std::tan(glm::radians(fov_y_deg) * 0.5f);
                const float viewport_height = static_cast<float>(viewport_.windowSize.y);
                constexpr float MIN_SCALE = 1.0f;
                constexpr float MAX_SCALE = 10000.0f;
                settings.ortho_scale = std::clamp(
                    viewport_height / (2.0f * distance_to_pivot * half_tan_fov),
                    MIN_SCALE, MAX_SCALE);
                rm->updateSettings(settings);
            }
        }

        // Force immediate camera update
        ui::CameraMove{
            .rotation = target_viewport.getRotationMatrix(),
            .translation = target_viewport.getTranslation()}
            .emit();
        publishCameraMove(&target_viewport);

        auto* const rendering_manager = services().renderingOrNull();

        // Set this as the current camera for GT comparison
        if (rendering_manager) {
            rendering_manager->setCurrentCameraId(event.cam_id);
        }

        if (auto* trainer_mgr = services().trainerOrNull(); trainer_mgr && trainer_mgr->getTrainer()) {
            LOG_INFO("Camera {} view: iter={}, last_loss={:.6f}",
                     event.cam_id,
                     trainer_mgr->getCurrentIteration(),
                     trainer_mgr->getCurrentLoss());
        } else if (rendering_manager) {
            rendering_manager->clearLatestCameraMetrics();
        }

        last_camview_ = event.cam_id;
    }

    void InputController::selectCameraByUid(const int uid, const bool toggle_selection) {
        if (!tool_context_)
            return;
        auto* const sm = tool_context_->getSceneManager();
        if (!sm)
            return;
        for (const auto* node : sm->getScene().getNodes()) {
            if (node->type == core::NodeType::CAMERA && node->camera_uid == uid) {
                if (toggle_selection) {
                    const auto selected_ids = sm->getSelectedNodeIds();
                    if (std::ranges::find(selected_ids, node->id) != selected_ids.end())
                        sm->removeFromSelection(node->id);
                    else
                        sm->addToSelection(node->id);
                } else {
                    sm->selectNode(node->id);
                }
                if (auto* rendering_manager = services().renderingOrNull()) {
                    rendering_manager->markDirty(DirtyFlag::SELECTION | DirtyFlag::OVERLAY, lfs::vis::FrameReason::Selection);
                }
                return;
            }
        }
    }

    bool InputController::handleFocusSelection(Viewport& target_viewport) {
        if (!tool_context_)
            return false;
        auto* const sm = tool_context_->getSceneManager();
        if (!sm)
            return false;

        const auto& scene = sm->getScene();
        const auto& selected = sm->getSelectedNodeNames();

        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());
        bool has_bounds = false;

        // Frame the trimmed splat and point-cloud boxes: a few far floaters would
        // otherwise park the camera beyond the zoom-out limit, where only zooming
        // in still works.
        if (selected.empty()) {
            has_bounds = computeWholeSceneBounds(total_min, total_max, /*use_percentile=*/true);
        } else {
            for (const auto& name : selected) {
                if (const auto* node = scene.getNode(name))
                    expandVisibleNodeWorldBounds(scene, *node, total_min, total_max, /*use_percentile=*/true);
            }
            has_bounds = total_min.x <= total_max.x;
            if (!has_bounds)
                has_bounds = computeWholeSceneBounds(total_min, total_max, /*use_percentile=*/true);
        }

        if (has_bounds) {
            target_viewport.camera.focusOnBounds(total_min, total_max);
            publishCameraMove(&target_viewport);
            return true;
        }
        return false;
    }

    // World AABB of the visible scene, skipping container nodes that carry no
    // geometry. use_percentile yields the trimmed (outlier-excluding) box.
    bool InputController::computeWholeSceneBounds(glm::vec3& out_min, glm::vec3& out_max,
                                                  const bool use_percentile) const {
        if (!tool_context_)
            return false;
        auto* const sm = tool_context_->getSceneManager();
        if (!sm)
            return false;

        const auto& scene = sm->getScene();
        out_min = glm::vec3(std::numeric_limits<float>::max());
        out_max = glm::vec3(std::numeric_limits<float>::lowest());

        for (const auto* node : scene.getNodes()) {
            if (isGeometryContainer(*node) || !scene.isNodeEffectivelyVisible(node->id))
                continue;
            expandNodeWorldBounds(scene, *node, out_min, out_max, use_percentile);
        }

        return out_min.x <= out_max.x;
    }

    // Cached whole-scene radius used to scale WASD speed and cap pan distance by
    // splat size. Uses the trimmed (1st/99th percentile) bounds so a few far-flung
    // floaters can't blow up the extent and make navigation far too fast. Lazily
    // computed after load (bounds may not exist the instant SceneLoaded fires) and
    // invalidated on scene load/clear. Returns 0 while no geometry is present.
    float InputController::sceneExtent() {
        if (scene_extent_ > 0.0f)
            return scene_extent_;

        glm::vec3 min, max;
        if (computeWholeSceneBounds(min, max, /*use_percentile=*/true))
            scene_extent_ = glm::length(max - min) * 0.5f;

        return scene_extent_;
    }

    // Seed the depth-map range from the trimmed scene scale once the extent is
    // known after a load, so the palette spans the actual content instead of the
    // fixed 0.1..100 default that is useless on large RAD scenes. Runs once per
    // scene; the user owns the range after that.
    void InputController::maybeInitializeDepthViewRange() {
        if (depth_range_initialized_)
            return;

        const float radius = sceneExtent();
        if (radius <= 0.0f)
            return;

        auto* const rendering = services().renderingOrNull();
        if (!rendering)
            return;

        constexpr float kDepthViewInitNear = 1.0f;
        auto settings = rendering->getSettings();
        settings.depth_view_min = kDepthViewInitNear;
        settings.depth_view_max = radius;
        sanitizeDepthViewSettings(settings);
        rendering->updateSettings(settings);

        depth_range_initialized_ = true;
    }

    bool InputController::focusSelection() {
        return handleFocusSelection(activeKeyboardViewport());
    }

    void InputController::goHome() {
        auto& target = activeKeyboardViewport();
        resetViewportToHome(target, viewport_);
        publishCameraMove(&target);
        if (auto* const viewer = lfs::python::get_visualizer()) {
            viewer->projectMarkSessionChapterCommitted("VIEW");
        }
    }

    void InputController::setHome() {
        auto& target = activeKeyboardViewport();
        target.camera.saveHomePosition();
        publishCameraMove(&target);
        if (auto* const viewer = lfs::python::get_visualizer()) {
            viewer->projectMarkSessionChapterCommitted("VIEW");
        }
    }

    void InputController::resetHome() {
        auto& target = activeKeyboardViewport();
        target.camera.resetHomePosition();
        resetViewportToHome(target, viewport_);
        publishCameraMove(&target);
        if (auto* const viewer = lfs::python::get_visualizer()) {
            viewer->projectMarkSessionChapterCommitted("VIEW");
        }
    }

    // Helpers
    bool InputController::isInViewport(double x, double y) const {
        return x >= viewport_bounds_.x &&
               x < viewport_bounds_.x + viewport_bounds_.width &&
               y >= viewport_bounds_.y &&
               y < viewport_bounds_.y + viewport_bounds_.height;
    }

    bool InputController::isPointerOverBlockingUi(const double x, const double y) const {
        if (input_router_) {
            return input_router_->pointerTarget(x, y) == input::InputTarget::Gui;
        }

        const auto& focus = gui::guiFocusState();
        if (focus.want_capture_mouse)
            return true;

        auto* gui = services().guiOrNull();
        if (!gui)
            return false;

        return gui->panelLayout().isResizingPanel() ||
               gui->isPositionOverFloatingPanel(x, y);
    }

    bool InputController::isPointerOverUiHover(const double x, const double y) const {
        if (input_router_) {
            return input_router_->hoverTarget(x, y) == input::InputTarget::Gui;
        }

        return isPointerOverBlockingUi(x, y);
    }

    bool InputController::shouldCameraHandleInput() const {
        if (drag_mode_ == DragMode::Gizmo || drag_mode_ == DragMode::Splitter) {
            return false;
        }

        // Navigation keys work while a panel holds keyboard focus, like F and
        // Home. Text fields and modals keep them.
        if (input_router_) {
            return !input_router_->isTextInputActive() && !input_router_->isModalOpen();
        }

        const auto& focus = gui::guiFocusState();
        if (focus.want_text_input)
            return false;

        return !focus.any_item_active;
    }

    bool InputController::isIndependentSplitViewActive() const {
        auto* const rendering = services().renderingOrNull();
        return rendering && rendering->isIndependentSplitViewActive();
    }

    void InputController::toggleIndependentSplitView() {
        if (!isIndependentSplitViewActive()) {
            const auto* const scene_manager = services().sceneOrNull();
            if (!scene_manager || scene_manager->isEmpty()) {
                return;
            }
        }

        cmd::ToggleIndependentSplitView{.viewport = &viewport_}.emit();
        resetSplitPanelFocus();
    }

    SplitViewPanelId InputController::splitPanelForScreenX(const double x) const {
        auto* const rendering = services().renderingOrNull();
        if (!rendering || viewport_bounds_.width <= 0.0f || viewport_bounds_.height <= 0.0f) {
            return SplitViewPanelId::Left;
        }

        const auto panel = rendering->resolveViewerPanel(
            viewport_,
            {viewport_bounds_.x, viewport_bounds_.y},
            {viewport_bounds_.width, viewport_bounds_.height},
            glm::vec2(static_cast<float>(x), viewport_bounds_.y + viewport_bounds_.height * 0.5f));
        return panel ? panel->panel : SplitViewPanelId::Left;
    }

    std::optional<InputController::PanelInteractionState> InputController::resolvePanelInteraction(
        const double x, const double y) {
        if (!isInViewport(x, y)) {
            return std::nullopt;
        }

        auto* const rendering = services().renderingOrNull();
        PanelInteractionState state;
        state.viewport = &viewport_;
        if (!rendering) {
            state.panel = SplitViewPanelId::Left;
            state.local_x = static_cast<float>(x) - viewport_bounds_.x;
            state.local_y = static_cast<float>(y) - viewport_bounds_.y;
            state.width = viewport_bounds_.width;
            state.height = viewport_bounds_.height;
            return state.valid() ? std::optional<PanelInteractionState>(state) : std::nullopt;
        }

        const auto panel = rendering->resolveViewerPanel(
            viewport_,
            {viewport_bounds_.x, viewport_bounds_.y},
            {viewport_bounds_.width, viewport_bounds_.height},
            glm::vec2(static_cast<float>(x), static_cast<float>(y)));
        if (!panel) {
            return std::nullopt;
        }

        state.panel = panel->panel;
        state.viewport = panel->viewport;
        state.local_x = static_cast<float>(x) - panel->x;
        state.local_y = static_cast<float>(y) - panel->y;
        state.width = panel->width;
        state.height = panel->height;

        return state.valid() ? std::optional<PanelInteractionState>(state) : std::nullopt;
    }

    void InputController::focusSplitPanel(const SplitViewPanelId panel) {
        if (auto* const rendering = services().renderingOrNull()) {
            const bool changed =
                rendering->getFocusedSplitPanel() != panel;
            if (changed) {
                pauseGamepadNavigation();
            }
            rendering->setFocusedSplitPanel(panel);
            if (changed) {
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
            }
        }
    }

    void InputController::resetSplitPanelFocus() {
        // Split-layout and scene changes may replace the secondary view's pose.
        // Do not keep integrating physics owned by that old layout.
        walk_viewport_ = nullptr;
        focusSplitPanel(SplitViewPanelId::Left);
        if (selection_tool_) {
            selection_tool_->setFilterPanel(SplitViewPanelId::Left);
        }
    }

    void InputController::clearSelectedCameraContextMenuGesture() {
        pending_camera_context_menu_ = {};
    }

    void InputController::beginPanDrag(const PanelInteractionState& interaction, const int button,
                                       const double x, const double y) {
        LOG_PERF("InputController::beginPanDrag button={} pos=({},{})", button, x, y);
        const float current_time = static_cast<float>(SDL_GetTicks() / 1000.0f);
        interaction.viewport->camera.finishGlide();
        interaction.viewport->camera.setSceneExtent(sceneExtent());
        interaction.viewport->camera.startPan(glm::vec2(x, y), current_time);
        drag_viewport_ = interaction.viewport;
        drag_split_panel_ = interaction.panel;
        focusSplitPanel(interaction.panel);
        drag_mode_ = DragMode::Pan;
        drag_button_ = button;
    }

    void InputController::clearViewportDragState() {
        const bool was_camera_drag =
            drag_mode_ == DragMode::Orbit ||
            drag_mode_ == DragMode::Pan ||
            drag_mode_ == DragMode::Rotate;

        if (drag_mode_ == DragMode::Orbit) {
            if (drag_viewport_) {
                drag_viewport_->camera.endRotateAroundCenter();
            } else {
                viewport_.camera.endRotateAroundCenter();
            }
        }

        drag_mode_ = DragMode::None;
        drag_button_ = -1;
        drag_viewport_ = nullptr;
        drag_split_panel_ = SplitViewPanelId::Left;
        pending_click_drag_ = {};
        forced_mouse_press_action_ = input::Action::NONE;
        is_node_rect_dragging_ = false;
        node_rect_button_ = -1;
        node_rect_modifiers_ = input::MODIFIER_NONE;
        node_point_pick_enabled_ = false;
        node_rect_select_enabled_ = false;
        clearSelectedCameraContextMenuGesture();
        press_selected_camera_frustum_ = false;
        pressed_camera_frustum_id_ = -1;
        pressed_camera_frustum_modifiers_ = input::MODIFIER_NONE;
        syncLookPointerLock();

        if (was_camera_drag) {
            onCameraMovementEnd();
        }
    }

    void InputController::lockLookPointer(const double x, const double y) {
        look_pos_ = {x, y};
        if (!window_ || look_pointer_restore_ || !SDL_SetWindowRelativeMouseMode(window_, true))
            return;
        look_pointer_restore_ =
            glm::vec2(static_cast<float>(x), static_cast<float>(y)) / input::windowPixelScale(window_);
    }

    bool InputController::isValueDragActive() const {
        return value_drag_restore_.has_value();
    }

    bool InputController::beginValueDrag() {
        if (value_drag_restore_)
            return true;
        if (!window_ || look_pointer_restore_)
            return false;
        glm::vec2 pointer;
        SDL_GetMouseState(&pointer.x, &pointer.y);
        if (!SDL_SetWindowRelativeMouseMode(window_, true))
            return false;
        value_drag_restore_ = pointer;
        value_drag_pin_ = pointer;
        value_drag_dx_ = 0.0f;
        value_drag_cancelled_ = false;
        return true;
    }

    float InputController::takeValueDragDelta() {
        return std::exchange(value_drag_dx_, 0.0f);
    }

    std::optional<glm::vec2> InputController::takeValueDragPin() {
        if (value_drag_restore_)
            return value_drag_pin_;
        return std::exchange(value_drag_pin_, std::nullopt);
    }

    void InputController::endValueDrag() {
        releaseValueDrag(true);
    }

    void InputController::cancelValueDrag(const bool restore_pointer) {
        if (!value_drag_restore_)
            return;
        value_drag_cancelled_ = true;
        releaseValueDrag(restore_pointer);
    }

    void InputController::releaseValueDrag(const bool restore_pointer) {
        if (!value_drag_restore_)
            return;
        SDL_SetWindowRelativeMouseMode(window_, false);
        if (restore_pointer)
            SDL_WarpMouseInWindow(window_, value_drag_restore_->x, value_drag_restore_->y);
        value_drag_restore_.reset();
        if (auto* rendering = services().renderingOrNull())
            rendering->markDirty(DirtyFlag::OVERLAY, FrameReason::Input, "value_drag");
    }

    void InputController::syncLookPointerLock(const bool restore_position) {
        if (!look_pointer_restore_ || drag_mode_ == DragMode::Rotate)
            return;
        SDL_SetWindowRelativeMouseMode(window_, false);
        if (restore_position)
            SDL_WarpMouseInWindow(window_, look_pointer_restore_->x, look_pointer_restore_->y);
        look_pointer_restore_.reset();
    }

    void InputController::clearWasdMomentumViewport() {
        if (!wasd_momentum_viewport_) {
            return;
        }
        wasd_momentum_viewport_->camera.clearWasdMomentum();
        wasd_momentum_viewport_->camera.clearDroneMotion();
        wasd_momentum_viewport_ = nullptr;
    }

    bool InputController::canOpenSelectedCameraContextMenu(const int hovered_camera_uid) const {
        if (hovered_camera_uid < 0 || !tool_context_) {
            return false;
        }

        const auto* const scene_manager = tool_context_->getSceneManager();
        if (!scene_manager) {
            return false;
        }

        const auto& scene = scene_manager->getScene();
        for (const auto* const node : scene.getNodes()) {
            if (node && node->type == core::NodeType::CAMERA && node->camera_uid == hovered_camera_uid)
                return true;
        }
        return false;
    }

    void InputController::applyCameraTrainingStateToSelection(
        const std::vector<std::string>& selected_names, const bool enabled) {
        if (!tool_context_) {
            return;
        }

        auto* const scene_manager = tool_context_->getSceneManager();
        if (!scene_manager) {
            return;
        }

        auto& scene = scene_manager->getScene();
        for (const auto& selected_name : selected_names) {
            const auto* const node = scene.getNode(selected_name);
            if (node && node->type == core::NodeType::CAMERA) {
                scene.setCameraTrainingEnabled(node->name, enabled);
            }
        }
    }

    void InputController::openSelectedCameraContextMenu(const int hovered_camera_uid,
                                                        const float screen_x,
                                                        const float screen_y) {
        auto* const gui = services().guiOrNull();
        if (!gui || !canOpenSelectedCameraContextMenu(hovered_camera_uid)) {
            return;
        }

        auto* const scene_manager = tool_context_ ? tool_context_->getSceneManager() : nullptr;
        if (!scene_manager) {
            return;
        }

        auto& scene = scene_manager->getScene();
        const core::SceneNode* hovered_node = nullptr;
        for (const auto* const node : scene.getNodes()) {
            if (node && node->type == core::NodeType::CAMERA && node->camera_uid == hovered_camera_uid) {
                hovered_node = node;
                break;
            }
        }
        if (!hovered_node) {
            return;
        }

        const auto selected_names = scene_manager->getSelectedNodeNames();
        bool hovered_in_selection = false;
        bool all_selected_cameras = selected_names.size() > 1;
        for (const auto& selected_name : selected_names) {
            const auto* const node = scene.getNode(selected_name);
            if (!node || node->type != core::NodeType::CAMERA) {
                all_selected_cameras = false;
                continue;
            }
            if (node->camera_uid == hovered_camera_uid)
                hovered_in_selection = true;
        }
        const bool use_multi_menu = all_selected_cameras && hovered_in_selection;

        std::vector<gui::ContextMenuItem> items;
        if (use_multi_menu) {
            items.push_back({
                .label = LOC(string_keys::Scene::ENABLE_ALL_TRAINING),
                .action = "enable_all_train",
            });
            items.push_back({
                .label = LOC(string_keys::Scene::DISABLE_ALL_TRAINING),
                .action = "disable_all_train",
            });

            gui->globalContextMenu().request(
                std::move(items), screen_x, screen_y,
                [this, selected_names](std::string_view action) {
                    if (action == "enable_all_train") {
                        applyCameraTrainingStateToSelection(selected_names, true);
                    } else if (action == "disable_all_train") {
                        applyCameraTrainingStateToSelection(selected_names, false);
                    }
                });
            return;
        }

        items.push_back({
            .label = LOC(string_keys::Scene::GO_TO_CAMERA_VIEW),
            .action = "go_to_camera",
        });
        items.push_back({
            .label = LOC(string_keys::Scene::GO_TO_IMAGE),
            .action = "go_to_image",
        });
        items.push_back({
            .label = LOC(string_keys::Scene::OPEN_IN_GT_COMPARE),
            .action = "open_in_gt_compare",
        });
        const std::string image_path = hovered_node->image_path;
        if (!image_path.empty()) {
            items.push_back({
                .label = LOC(string_keys::Scene::SHOW_IN_FILE_MANAGER),
                .action = "show_in_file_manager",
            });
        }
        items.push_back({
            .label = hovered_node->training_enabled
                         ? LOC(string_keys::Scene::DISABLE_FOR_TRAINING)
                         : LOC(string_keys::Scene::ENABLE_FOR_TRAINING),
            .action = hovered_node->training_enabled ? "disable_train" : "enable_train",
            .separator_before = true,
        });

        const std::string camera_name = hovered_node->name;
        gui->globalContextMenu().request(
            std::move(items), screen_x, screen_y,
            [this, camera_name, hovered_camera_uid, image_path](std::string_view action) {
                if (action == "go_to_camera") {
                    cmd::GoToCamView{.cam_id = hovered_camera_uid}.emit();
                    if (auto* const viewer = lfs::python::get_visualizer()) {
                        viewer->projectMarkSessionChapterCommitted("VIEW");
                    }
                    return;
                }
                if (action == "go_to_image") {
                    cmd::OpenCameraPreview{.cam_id = hovered_camera_uid}.emit();
                    return;
                }
                if (action == "open_in_gt_compare") {
                    cmd::GoToCamView{.cam_id = hovered_camera_uid}.emit();
                    if (auto* const viewer = lfs::python::get_visualizer()) {
                        viewer->projectMarkSessionChapterCommitted("VIEW");
                    }
                    auto* const rm = services().renderingOrNull();
                    if (!rm || !rm->isGTComparisonActive())
                        cmd::ToggleGTComparison{}.emit();
                    return;
                }
                if (action == "show_in_file_manager") {
                    if (!image_path.empty() &&
                        !lfs::core::reveal_in_file_manager(lfs::core::utf8_to_path(image_path))) {
                        LOG_WARN("Failed to reveal image in file manager: {}", image_path);
                    }
                    return;
                }

                if (!tool_context_) {
                    return;
                }
                auto* const scene_manager = tool_context_->getSceneManager();
                if (!scene_manager) {
                    return;
                }

                if (action == "enable_train") {
                    scene_manager->getScene().setCameraTrainingEnabled(camera_name, true);
                } else if (action == "disable_train") {
                    scene_manager->getScene().setCameraTrainingEnabled(camera_name, false);
                }
            });
    }

    bool InputController::snapViewportToNearestAxis(Viewport& target_viewport,
                                                    const SplitViewPanelId panel) {
        if (!camera_view_snap_enabled_) {
            return false;
        }

        constexpr float kAxisSnapAngleDegrees = 10.0f;
        int snapped_axis = -1;
        if (!target_viewport.camera.snapToNearestAxisView(
                kAxisSnapAngleDegrees, &snapped_axis, nullptr)) {
            return false;
        }

        target_viewport.camera.clearOrbitMomentum();
        if (&target_viewport == orbit_coast_viewport_) {
            orbit_coast_viewport_ = nullptr;
        }

        if (auto* const rendering = services().renderingOrNull()) {
            rendering->setGridPlaneForPanel(panel, snapped_axis);
        }

        return true;
    }

    Viewport& InputController::activeKeyboardViewport() {
        if (auto* const rendering = services().renderingOrNull()) {
            return rendering->resolveFocusedViewport(viewport_);
        }
        return viewport_;
    }

    const Viewport& InputController::activeKeyboardViewport() const {
        if (auto* const rendering = services().renderingOrNull()) {
            return rendering->resolveFocusedViewport(viewport_);
        }
        return viewport_;
    }

    void InputController::updateCameraSpeed(const bool increase) {
        auto& target_viewport = activeKeyboardViewport();
        increase ? target_viewport.camera.increaseWasdSpeed() : target_viewport.camera.decreaseWasdSpeed();
        ui::SpeedChanged{
            .current_speed = target_viewport.camera.getWasdSpeed(),
            .max_speed = target_viewport.camera.getMaxWasdSpeed()}
            .emit();
        if (auto* const viewer = lfs::python::get_visualizer()) {
            viewer->projectMarkSessionChapterCommitted("VIEW");
        }
    }

    void InputController::updateZoomSpeed(const bool increase) {
        auto& target_viewport = activeKeyboardViewport();
        increase ? target_viewport.camera.increaseZoomSpeed() : target_viewport.camera.decreaseZoomSpeed();
        ui::ZoomSpeedChanged{
            .zoom_speed = target_viewport.camera.getZoomSpeed(),
            .max_zoom_speed = target_viewport.camera.getMaxZoomSpeed()}
            .emit();
        if (auto* const viewer = lfs::python::get_visualizer()) {
            viewer->projectMarkSessionChapterCommitted("VIEW");
        }
    }

    void InputController::publishCameraMove(Viewport* target_viewport) {
        LOG_PERF("InputController::publishCameraMove drag_mode={}", static_cast<int>(drag_mode_));
        auto* const active_viewport = target_viewport ? target_viewport : &viewport_;
        if (selection_tool_ && selection_tool_->isEnabled()) {
            selection_tool_->syncDepthFilterToCamera(*active_viewport);
        }

        if (auto* const rendering = services().renderingOrNull()) {
            rendering->markCameraPoseChanged();
        }

        // Throttle event emission
        const auto now = std::chrono::steady_clock::now();
        if (now - last_camera_publish_ >= camera_publish_interval_) {
            ui::CameraMove{
                .rotation = active_viewport->getRotationMatrix(),
                .translation = active_viewport->getTranslation()}
                .emit();
            last_camera_publish_ = now;
        }
    }

    void InputController::onCameraMovementStart() {
        const auto now = std::chrono::steady_clock::now();
        if (!camera_is_moving_) {
            camera_is_moving_ = true;
            last_camera_movement_time_ = now;

            if (auto* const rendering = services().renderingOrNull();
                rendering && rendering->isGTComparisonActive()) {
                cmd::ToggleGTComparison{}.emit();
            }
            // Training continues during camera motion. Viewer re-renders zero-copy
            // bindings per frame (try-lock skip-and-retain on refine exclusive).
        } else {
            last_camera_movement_time_ = now;
        }
    }

    void InputController::onCameraMovementEnd() {
        // Don't clear camera_is_moving_ immediately — let the timeout handle it.
        last_camera_movement_time_ = std::chrono::steady_clock::now();
    }

    void InputController::checkCameraMovementTimeout() {
        if (!camera_is_moving_) {
            return;
        }

        auto now = std::chrono::steady_clock::now();
        if (now - last_camera_movement_time_ >= camera_movement_timeout_) {
            camera_is_moving_ = false;
        }
    }

    glm::vec3 InputController::unprojectScreenPoint(double x, double y, float fallback_distance) const {
        auto* const rendering = services().renderingOrNull();
        const auto interaction = rendering
                                     ? rendering->resolveViewerPanel(
                                           viewport_,
                                           {viewport_bounds_.x, viewport_bounds_.y},
                                           {viewport_bounds_.width, viewport_bounds_.height},
                                           glm::vec2(static_cast<float>(x), static_cast<float>(y)))
                                     : std::nullopt;
        const auto* const target_viewport = (interaction && interaction->valid()) ? interaction->viewport : &viewport_;
        if (!rendering || !interaction || !interaction->valid()) {
            const glm::vec3 forward = lfs::rendering::cameraForward(target_viewport->camera.R);
            return target_viewport->camera.t + forward * fallback_distance;
        }

        const float local_x = static_cast<float>(x) - interaction->x;
        const float local_y = static_cast<float>(y) - interaction->y;
        const auto render_settings = rendering->getSettings();
        const float focal_length_mm = render_settings.focal_length_mm;
        const float ortho_scale = target_viewport->ortho_scale_override.value_or(render_settings.ortho_scale);
        Viewport projection_viewport = *target_viewport;
        projection_viewport.windowSize = glm::ivec2(
            std::max(static_cast<int>(interaction->width), 1),
            std::max(static_cast<int>(interaction->height), 1));

        const int sample_x = static_cast<int>(local_x);
        const int sample_y = static_cast<int>(local_y);
        float depth = -1.0f;
        if (auto* const scene_manager = services().sceneOrNull()) {
            depth = rendering->renderExpectedDepthAtPixel(
                RenderingManager::ExpectedDepthSampleRequest{
                    .scene_manager = scene_manager,
                    .viewport = &projection_viewport,
                    .render_size = projection_viewport.windowSize,
                    .pixel = {sample_x, sample_y},
                    .focal_length_mm = focal_length_mm,
                    .orthographic = render_settings.orthographic,
                    .ortho_scale = ortho_scale,
                    .panel = interaction->panel,
                });
        }
        if (depth <= 0.0f) {
            depth = rendering->getDepthAtPixel(sample_x, sample_y, interaction->panel);
        }

        if (depth > 0.0f) {
            const glm::vec3 world = projection_viewport.unprojectPixel(
                local_x, local_y, depth, focal_length_mm, render_settings.orthographic, ortho_scale);
            if (Viewport::isValidWorldPosition(world)) {
                return world;
            }
        }

        const glm::vec3 fallback_world =
            projection_viewport.unprojectPixel(
                local_x, local_y, fallback_distance, focal_length_mm, render_settings.orthographic, ortho_scale);
        if (Viewport::isValidWorldPosition(fallback_world)) {
            return fallback_world;
        }

        const glm::vec3 forward = lfs::rendering::cameraForward(target_viewport->camera.R);
        return target_viewport->camera.t + forward * fallback_distance;
    }

    std::pair<glm::vec3, glm::vec3> InputController::computePickRay(double x, double y) const {
        const auto* const rendering = services().renderingOrNull();
        const auto interaction = rendering
                                     ? rendering->resolveViewerPanel(
                                           viewport_,
                                           {viewport_bounds_.x, viewport_bounds_.y},
                                           {viewport_bounds_.width, viewport_bounds_.height},
                                           glm::vec2(static_cast<float>(x), static_cast<float>(y)))
                                     : std::nullopt;
        const auto* const target_viewport = (interaction && interaction->valid()) ? interaction->viewport : &viewport_;
        const glm::mat3 R = target_viewport->getRotationMatrix();
        const glm::vec3 camera_pos = target_viewport->getTranslation();

        if (!rendering || !interaction || !interaction->valid()) {
            const glm::vec3 forward = lfs::rendering::cameraForward(R);
            return {camera_pos, forward};
        }

        const float local_x = static_cast<float>(x) - interaction->x;
        const float local_y = static_cast<float>(y) - interaction->y;
        const glm::vec3 world_dir = lfs::rendering::computePickRayDirection(
            R,
            glm::ivec2(
                std::max(static_cast<int>(interaction->width), 1),
                std::max(static_cast<int>(interaction->height), 1)),
            local_x,
            local_y,
            rendering->getFocalLengthMm());
        return {camera_pos, world_dir};
    }

    input::ToolMode InputController::getCurrentToolMode() const {
        if (vr_viewing_)
            return input::ToolMode::VR;
        if (UnifiedToolRegistry::instance().getActiveTool() == "builtin.cropbox")
            return input::ToolMode::CROP_BOX;
        if (selection_tool_ && selection_tool_->isEnabled())
            return input::ToolMode::SELECTION;
        if (align_tool_ && align_tool_->isEnabled())
            return input::ToolMode::ALIGN;
        if (scale_distance_tool_ && scale_distance_tool_->isEnabled())
            return input::ToolMode::ALIGN;
        // Check GUI tool mode for transform tools
        if (services().guiOrNull()) {
            const auto gui_tool = services().guiOrNull()->gizmo().getCurrentToolMode();
            if (gui_tool == ToolType::Translate)
                return input::ToolMode::TRANSLATE;
            if (gui_tool == ToolType::Rotate)
                return input::ToolMode::ROTATE;
            if (gui_tool == ToolType::Scale)
                return input::ToolMode::SCALE;
        }
        return input::ToolMode::GLOBAL;
    }

} // namespace lfs::vis
