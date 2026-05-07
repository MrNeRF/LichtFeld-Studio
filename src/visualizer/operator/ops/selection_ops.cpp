/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"
#include "input/key_codes.hpp"
#include "operator/operator_registry.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

    namespace {

        [[nodiscard]] lfs::vis::SelectionShape toSelectionShape(const int mode) {
            switch (mode) {
            case 1: return lfs::vis::SelectionShape::Rectangle;
            case 2: return lfs::vis::SelectionShape::Polygon;
            case 3: return lfs::vis::SelectionShape::Lasso;
            case 4: return lfs::vis::SelectionShape::Rings;
            default: return lfs::vis::SelectionShape::Brush;
            }
        }

        [[nodiscard]] lfs::vis::SelectionMode toSelectionMode(const int mode) {
            switch (mode) {
            case 1: return lfs::vis::SelectionMode::Add;
            case 2: return lfs::vis::SelectionMode::Remove;
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
                                       const OperatorProperties* /*props*/) const {
        return ctx.scene().getScene().getTotalGaussianCount() > 0;
    }

    OperatorResult SelectionStrokeOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto* const service = ctx.scene().getSelectionService();
        if (!service) {
            return OperatorResult::CANCELLED;
        }

        shape_ = toSelectionShape(props.get_or<int>("mode", 0));
        mode_ = toSelectionMode(props.get_or<int>("op", 0));
        brush_radius_ = props.get_or<float>("brush_radius", 20.0f);
        stroke_button_ = props.get_or<int>("button", static_cast<int>(input::AppMouseButton::LEFT));
        filters_.crop_filter = props.get_or<bool>("use_crop_filter", false);
        filters_.depth_filter = props.get_or<bool>("use_depth_filter", false);
        filters_.restrict_to_selected_nodes = props.get_or<bool>("restrict_to_selected_nodes", true);

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
                const auto result = service->finishInteractiveSelection();
                return result.success ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
            }
        }

        if (event->type == ModalEvent::Type::MOUSE_SCROLL &&
            shape_ == lfs::vis::SelectionShape::Polygon) {
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
