/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"
#include "core/selection_domain.hpp"
#include "core/services.hpp"
#include "input/key_codes.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/core/editor_context.hpp"

namespace lfs::vis::op {

    namespace {
        constexpr int kSelectionModeColor = static_cast<int>(lfs::vis::SelectionSubMode::Color);

        [[nodiscard]] bool isCameraSubmodeSupported(const int mode) {
            switch (static_cast<lfs::vis::SelectionSubMode>(mode)) {
            case lfs::vis::SelectionSubMode::Rectangle:
            case lfs::vis::SelectionSubMode::Polygon:
            case lfs::vis::SelectionSubMode::Lasso:
                return true;
            case lfs::vis::SelectionSubMode::Centers:
            case lfs::vis::SelectionSubMode::Rings:
            case lfs::vis::SelectionSubMode::Color:
            case lfs::vis::SelectionSubMode::Box:
            case lfs::vis::SelectionSubMode::Sphere:
                return false;
            }
            return false;
        }

        [[nodiscard]] bool hasVisibleCameraNode(const SceneManager& scene_manager) {
            const auto& scene = scene_manager.getScene();
            for (const auto* const node : scene.getNodes()) {
                if (node && node->type == lfs::core::NodeType::CAMERA && node->camera &&
                    scene.isNodeEffectivelyVisible(node->id)) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] lfs::vis::SelectionShape toSelectionShape(const int mode) {
            switch (static_cast<lfs::vis::SelectionSubMode>(mode)) {
            case lfs::vis::SelectionSubMode::Rectangle: return lfs::vis::SelectionShape::Rectangle;
            case lfs::vis::SelectionSubMode::Polygon: return lfs::vis::SelectionShape::Polygon;
            case lfs::vis::SelectionSubMode::Lasso: return lfs::vis::SelectionShape::Lasso;
            case lfs::vis::SelectionSubMode::Rings: return lfs::vis::SelectionShape::Rings;
            case lfs::vis::SelectionSubMode::Box: return lfs::vis::SelectionShape::Box;
            case lfs::vis::SelectionSubMode::Sphere: return lfs::vis::SelectionShape::Sphere;
            case lfs::vis::SelectionSubMode::Centers:
            case lfs::vis::SelectionSubMode::Color:
            default: return lfs::vis::SelectionShape::Brush;
            }
        }

        [[nodiscard]] lfs::vis::SelectionMode toSelectionMode(const int mode) {
            switch (mode) {
            case 1: return lfs::vis::SelectionMode::Add;
            case 2: return lfs::vis::SelectionMode::Remove;
            case 3: return lfs::vis::SelectionMode::Intersect;
            default: return lfs::vis::SelectionMode::Replace;
            }
        }

        [[nodiscard]] lfs::vis::SelectionMode selectionModeFromModifiers(const int mods) {
            if (mods & input::KEYMOD_SHIFT) {
                return lfs::vis::SelectionMode::Add;
            }
            if (mods & input::KEYMOD_CTRL) {
                return lfs::vis::SelectionMode::Remove;
            }
            if (mods & input::KEYMOD_ALT) {
                return lfs::vis::SelectionMode::Intersect;
            }
            return lfs::vis::SelectionMode::Replace;
        }

    } // namespace

    const OperatorDescriptor SelectionStrokeOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::SelectionStroke,
        .python_class_id = {},
        .label = "Selection Stroke",
        .description = "Paint or drag to select gaussians",
        .icon = "selection",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE,
    };

    bool SelectionStrokeOperator::poll(const OperatorContext& ctx,
                                       const OperatorProperties* props) const {
        switch (resolveSelectionDomain(ctx.scene())) {
        case SelectionDomain::Gaussians:
            return ctx.scene().getScene().getTotalGaussianCount() > 0;
        case SelectionDomain::PointCloud: {
            const auto* const node = resolveActivePointCloudNode(ctx.scene());
            return node && node->point_cloud && node->point_cloud->size() > 0;
        }
        case SelectionDomain::Cameras:
            if (props && !isCameraSubmodeSupported(props->get_or<int>("mode", 0))) {
                return false;
            }
            if (!hasVisibleCameraNode(ctx.scene())) {
                return false;
            }
            if (const auto* const rendering_manager = services().renderingOrNull()) {
                return rendering_manager->getSettings().show_camera_frustums;
            }
            return false;
        }
        return false;
    }

    OperatorResult SelectionStrokeOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto* const service = ctx.scene().getSelectionService();
        if (!service) {
            return OperatorResult::CANCELLED;
        }

        const int mode_int = props.get_or<int>("mode", 0);
        if (resolveSelectionDomain(ctx.scene()) == SelectionDomain::Cameras &&
            !isCameraSubmodeSupported(mode_int)) {
            return OperatorResult::CANCELLED;
        }
        mode_ = toSelectionMode(props.get_or<int>("op", 0));
        brush_radius_ = props.get_or<float>("brush_radius", 20.0f);
        stroke_button_ = props.get_or<int>("button", static_cast<int>(input::AppMouseButton::LEFT));
        filters_.crop_filter = props.get_or<bool>("use_crop_filter", false);
        filters_.depth_filter = props.get_or<bool>("use_depth_filter", false);
        filters_.restrict_to_selected_nodes = props.get_or<bool>("restrict_to_selected_nodes", true);

        if (mode_int == kSelectionModeColor) {
            const float click_x = static_cast<float>(props.get_or<double>("x", 0.0));
            const float click_y = static_cast<float>(props.get_or<double>("y", 0.0));
            const auto result = service->selectByColorAt(click_x, click_y, mode_, filters_);
            return result.success ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
        }

        shape_ = toSelectionShape(mode_int);
        const glm::vec2 start_pos(props.get_or<double>("x", 0.0), props.get_or<double>("y", 0.0));
        if (!service->beginInteractiveSelection(shape_, mode_, start_pos, brush_radius_, filters_)) {
            return OperatorResult::CANCELLED;
        }

        return OperatorResult::RUNNING_MODAL;
    }

    OperatorResult SelectionStrokeOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
        auto* const service = ctx.scene().getSelectionService();
        if (!service) {
            return OperatorResult::CANCELLED;
        }

        const auto* event = ctx.event();
        if (!event) {
            return OperatorResult::RUNNING_MODAL;
        }

        if (event->type == ModalEvent::Type::MOUSE_MOVE) {
            const auto* move = event->as<MouseMoveEvent>();
            if (!move) {
                return OperatorResult::RUNNING_MODAL;
            }

            service->updateInteractiveSelection(glm::vec2(move->position));
            if (shape_ == lfs::vis::SelectionShape::Box ||
                shape_ == lfs::vis::SelectionShape::Sphere) {
                service->refreshInteractivePreview();
            }
            if (shape_ == lfs::vis::SelectionShape::Polygon) {
                return service->isInteractivePolygonVertexDragActive()
                           ? OperatorResult::RUNNING_MODAL
                           : OperatorResult::PASS_THROUGH;
            }
            return OperatorResult::RUNNING_MODAL;
        }

        if (event->type == ModalEvent::Type::MOUSE_BUTTON) {
            const auto* mb = event->as<MouseButtonEvent>();
            if (!mb) {
                return OperatorResult::RUNNING_MODAL;
            }

            const bool is_stroke_button = mb->button == stroke_button_;

            if (shape_ == lfs::vis::SelectionShape::Polygon) {
                if (is_stroke_button && mb->action == input::ACTION_PRESS) {
                    if (service->isInteractiveSelectionClosed()) {
                        if (mb->mods & input::KEYMOD_SHIFT) {
                            (void)service->insertInteractivePolygonVertex(glm::vec2(mb->position));
                            return OperatorResult::RUNNING_MODAL;
                        }
                        if (mb->mods & input::KEYMOD_CTRL) {
                            (void)service->removeInteractivePolygonVertex(glm::vec2(mb->position));
                            return OperatorResult::RUNNING_MODAL;
                        }
                    }

                    if (service->beginInteractivePolygonVertexDrag(glm::vec2(mb->position))) {
                        return OperatorResult::RUNNING_MODAL;
                    }
                    service->appendInteractivePolygonVertex(glm::vec2(mb->position));
                    return OperatorResult::RUNNING_MODAL;
                }

                if (is_stroke_button && mb->action == input::ACTION_RELEASE &&
                    service->isInteractivePolygonVertexDragActive()) {
                    service->endInteractivePolygonVertexDrag();
                    return OperatorResult::RUNNING_MODAL;
                }

                return OperatorResult::PASS_THROUGH;
            }

            if (is_stroke_button && mb->action == input::ACTION_RELEASE) {
                service->updateInteractiveSelection(glm::vec2(mb->position));
                const auto result = service->finishInteractiveSelection();
                return result.success ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
            }
        }

        // Always let scroll events through so the user can resize the selection
        // ring mid-stroke (BRUSH_RESIZE is bound to Ctrl/Shift+scroll). The
        // shape doesn't matter — the operator never uses scroll input itself.
        if (event->type == ModalEvent::Type::MOUSE_SCROLL) {
            return OperatorResult::PASS_THROUGH;
        }

        if (event->type == ModalEvent::Type::ACTION) {
            const auto* action = event->as<ActionEvent>();
            if (!action) {
                return OperatorResult::RUNNING_MODAL;
            }

            switch (action->action) {
            case input::Action::CONFIRM_POLYGON: {
                if (shape_ != lfs::vis::SelectionShape::Polygon) {
                    return OperatorResult::PASS_THROUGH;
                }

                mode_ = selectionModeFromModifiers(action->mods);
                service->setInteractiveSelectionMode(mode_);
                const auto result = service->finishInteractiveSelection();
                return result.success ? OperatorResult::FINISHED : OperatorResult::RUNNING_MODAL;
            }

            case input::Action::UNDO_POLYGON_VERTEX:
                if (shape_ == lfs::vis::SelectionShape::Polygon) {
                    if (!service->undoInteractivePolygonVertex()) {
                        return OperatorResult::CANCELLED;
                    }
                    return OperatorResult::RUNNING_MODAL;
                }
                return OperatorResult::CANCELLED;

            case input::Action::CANCEL_POLYGON:
                return OperatorResult::CANCELLED;

            default:
                return OperatorResult::PASS_THROUGH;
            }
        }

        if (event->type == ModalEvent::Type::KEY) {
            const auto* key = event->as<KeyEvent>();
            if (!key) {
                return OperatorResult::RUNNING_MODAL;
            }
            return OperatorResult::PASS_THROUGH;
        }

        return OperatorResult::RUNNING_MODAL;
    }

    void SelectionStrokeOperator::cancel(OperatorContext& ctx) {
        if (auto* const service = ctx.scene().getSelectionService()) {
            service->cancelInteractiveSelection();
        }
    }

    void registerSelectionOperators() {
        operators().registerOperator(BuiltinOp::SelectionStroke, SelectionStrokeOperator::DESCRIPTOR,
                                     [] { return std::make_unique<SelectionStrokeOperator>(); });
    }

    void unregisterSelectionOperators() {
        operators().unregisterOperator(BuiltinOp::SelectionStroke);
    }

} // namespace lfs::vis::op
