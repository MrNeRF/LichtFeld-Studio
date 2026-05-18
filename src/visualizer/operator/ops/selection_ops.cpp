/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"
#include "core/cuda/selection_ops.hpp"
#include "core/services.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_manager.hpp"
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

        const int mode_int = props.get_or<int>("mode", 0);
        mode_ = toSelectionMode(props.get_or<int>("op", 0));
        brush_radius_ = props.get_or<float>("brush_radius", 20.0f);
        stroke_button_ = props.get_or<int>("button", static_cast<int>(input::AppMouseButton::LEFT));
        filters_.crop_filter = props.get_or<bool>("use_crop_filter", false);
        filters_.depth_filter = props.get_or<bool>("use_depth_filter", false);
        filters_.restrict_to_selected_nodes = props.get_or<bool>("restrict_to_selected_nodes", true);

        // Color selection mode (5): pick Gaussian under cursor and select by color similarity
        if (mode_int == 5) {
            const float click_x = static_cast<float>(props.get_or<double>("x", 0.0));
            const float click_y = static_cast<float>(props.get_or<double>("y", 0.0));

            // GPU pick pass to find which Gaussian is under the cursor
            const auto picked = service->pickGaussianAt(click_x, click_y);
            if (!picked) {
                return OperatorResult::CANCELLED;
            }
            const int hovered_id = *picked;

            auto& scene = ctx.scene().getScene();
            auto* model = scene.getCombinedModel();
            if (!model) {
                return OperatorResult::CANCELLED;
            }

            const auto& sh0 = model->sh0();
            if (!sh0.is_valid() || static_cast<size_t>(hovered_id) >= sh0.size(0)) {
                return OperatorResult::CANCELLED;
            }

            // Decode the reference Gaussian's SH DC color on CPU
            auto sh0_cpu = sh0.cpu();
            const float* sh0_data = sh0_cpu.ptr<float>();
            if (!sh0_data) {
                return OperatorResult::CANCELLED;
            }

            constexpr float SH_C0 = 0.28209479177387814f;
            const float ref_r = std::clamp(0.5f + sh0_data[hovered_id * 3] * SH_C0, 0.0f, 1.0f);
            const float ref_g = std::clamp(0.5f + sh0_data[hovered_id * 3 + 1] * SH_C0, 0.0f, 1.0f);
            const float ref_b = std::clamp(0.5f + sh0_data[hovered_id * 3 + 2] * SH_C0, 0.0f, 1.0f);

            constexpr float COLOR_THRESHOLD = 0.2f;
            const auto group_id = scene.getActiveSelectionGroup();
            auto mask = core::cuda::select_by_color(sh0, ref_r, ref_g, ref_b, COLOR_THRESHOLD, group_id);

            // Apply with undo support (overwrite the ring selection with color selection)
            auto snapshot = std::make_unique<op::SceneSnapshot>(ctx.scene(), "selection.by_color");
            snapshot->captureSelection();
            scene.setSelectionMask(std::make_shared<core::Tensor>(std::move(mask)));
            snapshot->captureAfter();
            op::pushSceneSnapshotIfChanged(std::move(snapshot));

            if (auto* rm = services().renderingOrNull()) {
                rm->markDirty(DirtyFlag::SELECTION);
            }
            return OperatorResult::FINISHED;
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
