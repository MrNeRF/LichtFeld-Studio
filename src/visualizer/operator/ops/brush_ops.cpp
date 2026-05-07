/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "brush_ops.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "gui/gui_manager.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/cuda_kernels.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"
#include "visualizer_impl.hpp"
#include <optional>

namespace lfs::vis::op {

    namespace {

        struct ViewportBounds {
            float x = 0, y = 0, width = 0, height = 0;
        };

        struct BrushRenderTarget {
            SplitViewPanelId panel = SplitViewPanelId::Left;
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
            int render_width = 0;
            int render_height = 0;

            [[nodiscard]] bool valid() const {
                return width > 0.0f &&
                       height > 0.0f &&
                       render_width > 0 &&
                       render_height > 0;
            }
        };

        ViewportBounds getViewportBounds() {
            auto* gm = services().guiOrNull();
            if (!gm) {
                return {};
            }
            const auto pos = gm->getViewportPos();
            const auto size = gm->getViewportSize();
            return {pos.x, pos.y, size.x, size.y};
        }

        std::optional<BrushRenderTarget> resolveBrushRenderTarget(
            RenderingManager& rendering_manager,
            gui::GuiManager& gui_manager,
            const double screen_x,
            const double screen_y) {
            auto* const viewer = gui_manager.getViewer();
            if (!viewer) {
                return std::nullopt;
            }

            const auto bounds = getViewportBounds();
            if (bounds.width <= 0.0f || bounds.height <= 0.0f) {
                return std::nullopt;
            }

            const auto panel = rendering_manager.resolveViewerPanel(
                viewer->getViewport(),
                {bounds.x, bounds.y},
                {bounds.width, bounds.height},
                glm::vec2(static_cast<float>(screen_x), static_cast<float>(screen_y)));
            if (!panel || !panel->valid()) {
                return std::nullopt;
            }

            return BrushRenderTarget{
                .panel = panel->panel,
                .x = panel->x,
                .y = panel->y,
                .width = panel->width,
                .height = panel->height,
                .render_width = panel->render_width,
                .render_height = panel->render_height,
            };
        }

    } // namespace

    const OperatorDescriptor BrushStrokeOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::BrushStroke,
        .python_class_id = {},
        .label = "Brush Stroke",
        .description = "Paint to select or adjust saturation of gaussians",
        .icon = "brush",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE,
    };

    bool BrushStrokeOperator::poll(const OperatorContext& ctx,
                                   const OperatorProperties* /*props*/) const {
        return ctx.scene().getScene().getTotalGaussianCount() > 0;
    }

    OperatorResult BrushStrokeOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto mode_int = props.get_or<int>("mode", 0);
        mode_ = static_cast<BrushMode>(mode_int);

        const auto action_int = props.get_or<int>("action", 0);
        action_ = static_cast<BrushAction>(action_int);

        brush_radius_ = props.get_or<float>("brush_radius", 20.0f);
        saturation_amount_ = props.get_or<float>("saturation_amount", 0.5f);
        stroke_button_ = props.get_or<int>("button", static_cast<int>(input::AppMouseButton::LEFT));

        const auto x = props.get_or<double>("x", 0.0);
        const auto y = props.get_or<double>("y", 0.0);

        last_stroke_pos_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));

        if (mode_ == BrushMode::Select) {
            beginSelectionStroke(ctx);
            updateSelectionAtPoint(x, y, ctx);
        } else {
            beginSaturationStroke(ctx);
            updateSaturationAtPoint(x, y, ctx);
        }

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::SELECTION);
        }

        return OperatorResult::RUNNING_MODAL;
    }

    OperatorResult BrushStrokeOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
        const auto* event = ctx.event();
        if (!event) {
            return OperatorResult::RUNNING_MODAL;
        }

        if (event->type == ModalEvent::Type::MOUSE_MOVE) {
            const auto* move = event->as<MouseMoveEvent>();
            if (!move) {
                return OperatorResult::RUNNING_MODAL;
            }

            const double x = move->position.x;
            const double y = move->position.y;

            if (mode_ == BrushMode::Select) {
                updateSelectionAtPoint(x, y, ctx);
            } else {
                updateSaturationAtPoint(x, y, ctx);
            }

            last_stroke_pos_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));

            if (services().renderingOrNull()) {
                services().renderingOrNull()->markDirty(DirtyFlag::SELECTION);
            }
            return OperatorResult::RUNNING_MODAL;
        }

        if (event->type == ModalEvent::Type::MOUSE_BUTTON) {
            const auto* mb = event->as<MouseButtonEvent>();
            if (!mb) {
                return OperatorResult::RUNNING_MODAL;
            }

            if (mb->button == stroke_button_ && mb->action == input::ACTION_RELEASE) {
                if (mode_ == BrushMode::Select) {
                    finalizeSelectionStroke(ctx);
                } else {
                    finalizeSaturationStroke(ctx);
                }
                clearBrushState();
                return OperatorResult::FINISHED;
            }

            if (mb->button == static_cast<int>(input::AppMouseButton::RIGHT) &&
                mb->button != stroke_button_ &&
                mb->action == input::ACTION_PRESS) {
                return OperatorResult::CANCELLED;
            }
        }

        if (event->type == ModalEvent::Type::KEY) {
            const auto* ke = event->as<KeyEvent>();
            if (ke && ke->action == input::ACTION_PRESS) {
                return OperatorResult::PASS_THROUGH;
            }
        }

        return OperatorResult::RUNNING_MODAL;
    }

    void BrushStrokeOperator::cancel(OperatorContext& ctx) {
        // Restore original state
        if (mode_ == BrushMode::Select) {
            if (selection_before_ && selection_before_->is_valid()) {
                ctx.scene().getScene().setSelectionMask(selection_before_);
            }
        } else {
            if (sh0_before_ && sh0_before_->is_valid() && !saturation_node_name_.empty()) {
                auto* mutable_node = ctx.scene().getScene().getMutableNode(saturation_node_name_);
                if (mutable_node && mutable_node->model) {
                    mutable_node->model->sh0() = sh0_before_->clone();
                }
            }
        }

        clearBrushState();

        cumulative_selection_ = lfs::core::Tensor();
        selection_before_.reset();
        sh0_before_.reset();
        saturation_node_name_.clear();

        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty(DirtyFlag::SELECTION);
        }
    }

    void BrushStrokeOperator::beginSelectionStroke(OperatorContext& ctx) {
        auto& scene = ctx.scene().getScene();
        const size_t n = scene.getTotalGaussianCount();
        if (n == 0) {
            return;
        }

        const auto existing = scene.getSelectionMask();
        selection_before_ = (existing && existing->is_valid())
                                ? std::make_shared<lfs::core::Tensor>(existing->clone())
                                : nullptr;

        // Start with existing selection (if any) for Add mode, or empty for Remove
        if (existing && existing->is_valid() && existing->size(0) == n) {
            cumulative_selection_ = existing->to(lfs::core::DataType::Bool);
        } else {
            cumulative_selection_ = lfs::core::Tensor::zeros({n}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        }
    }

    void BrushStrokeOperator::beginSaturationStroke(OperatorContext& ctx) {
        auto& scene = ctx.scene().getScene();

        auto visible_nodes = scene.getVisibleNodes();
        if (visible_nodes.empty()) {
            return;
        }

        saturation_node_name_ = visible_nodes[0]->name;
        auto* mutable_node = scene.getMutableNode(saturation_node_name_);
        if (!mutable_node || !mutable_node->model) {
            return;
        }

        const auto& sh0 = mutable_node->model->sh0();
        if (sh0.is_valid()) {
            sh0_before_ = std::make_shared<lfs::core::Tensor>(sh0.clone());
        } else {
            sh0_before_.reset();
        }
    }

    void BrushStrokeOperator::updateSelectionAtPoint(double x, double y, OperatorContext& /*ctx*/) {
        if (!cumulative_selection_.is_valid()) {
            return;
        }

        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return;
        }

        const auto target = resolveBrushRenderTarget(*rm, *gm, x, y);
        if (!target || !target->valid()) {
            return;
        }

        rm->setFocusedSplitPanel(target->panel);

        const float scale_x = static_cast<float>(target->render_width) / target->width;
        const float rel_x = static_cast<float>(x) - target->x;
        const float rel_y = static_cast<float>(y) - target->y;

        const float image_x = rel_x * scale_x;
        const float image_y = rel_y * (static_cast<float>(target->render_height) / target->height);
        const float scaled_radius = brush_radius_ * scale_x;
        const bool add_mode = (action_ == BrushAction::Add);

        rm->setCursorPreviewState(
            true, image_x, image_y, scaled_radius, add_mode, &cumulative_selection_, false, 0.0f, target->panel);
    }

    void BrushStrokeOperator::updateSaturationAtPoint(double x, double y, OperatorContext& ctx) {
        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        auto* selection_service = ctx.scene().getSelectionService();
        if (!rm || !gm || !gm->getViewer() || !selection_service) {
            return;
        }

        if (saturation_node_name_.empty()) {
            return;
        }

        auto* mutable_node = ctx.scene().getScene().getMutableNode(saturation_node_name_);
        if (!mutable_node || !mutable_node->model) {
            return;
        }

        auto& sh0 = mutable_node->model->sh0();
        if (!sh0.is_valid()) {
            return;
        }

        const auto target = resolveBrushRenderTarget(*rm, *gm, x, y);
        if (!target || !target->valid()) {
            return;
        }

        rm->setFocusedSplitPanel(target->panel);

        const float scale_x = static_cast<float>(target->render_width) / target->width;
        const float scale_y = static_cast<float>(target->render_height) / target->height;
        const float rel_x = static_cast<float>(x) - target->x;
        const float rel_y = static_cast<float>(y) - target->y;

        const float image_x = rel_x * scale_x;
        const float image_y = rel_y * scale_y;
        const float scaled_radius = brush_radius_ * scale_x;

        // Reshape SH0 from [N, 1, 3] to [N, 3] for the kernel
        auto sh0_reshaped = sh0.reshape({static_cast<int>(sh0.size(0)), 3});

        const auto screen_positions = selection_service->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) {
            return;
        }

        const int num_gaussians = static_cast<int>(screen_positions->size(0));
        if (num_gaussians == 0) {
            return;
        }

        lfs::launchAdjustSaturation(
            sh0_reshaped.ptr<float>(),
            screen_positions->ptr<float>(),
            image_x,
            image_y,
            scaled_radius,
            saturation_amount_,
            num_gaussians,
            nullptr);

        rm->markDirty(DirtyFlag::SPLATS);
        rm->setCursorPreviewState(
            true, image_x, image_y, scaled_radius, true, nullptr, true, saturation_amount_, target->panel);
    }

    void BrushStrokeOperator::finalizeSelectionStroke(OperatorContext& ctx) {
        if (!cumulative_selection_.is_valid()) {
            return;
        }

        auto& scene_manager = ctx.scene();
        auto& scene = scene_manager.getScene();

        auto entry = std::make_unique<SceneSnapshot>(scene_manager, "select.brush");
        entry->captureSelection();

        auto mask = cumulative_selection_.to(lfs::core::DataType::UInt8);
        scene.setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));

        entry->captureAfter();
        pushSceneSnapshotIfChanged(std::move(entry));

        cumulative_selection_ = lfs::core::Tensor();
    }

    void BrushStrokeOperator::finalizeSaturationStroke(OperatorContext& ctx) {
        if (sh0_before_ && sh0_before_->is_valid() && !saturation_node_name_.empty()) {
            auto entry = std::make_unique<TensorUndoEntry>(
                "brush.saturation",
                UndoMetadata{
                    .id = "tensor.saturation",
                    .label = "Saturation Brush",
                    .source = "operator",
                    .scope = "tensor",
                },
                saturation_node_name_ + ".sh0",
                sh0_before_->clone(),
                [&scene_manager = ctx.scene(), node_name = saturation_node_name_]() -> lfs::core::Tensor* {
                    auto* node = scene_manager.getScene().getMutableNode(node_name);
                    if (!node || !node->model) {
                        return nullptr;
                    }
                    return &node->model->sh0();
                });
            entry->captureAfter();
            if (entry->hasChanges()) {
                undoHistory().push(std::move(entry));
            }
        }
        sh0_before_.reset();
        saturation_node_name_.clear();
    }

    void BrushStrokeOperator::clearBrushState() {
        if (auto* rm = services().renderingOrNull()) {
            rm->clearCursorPreviewState();
            rm->markDirty(DirtyFlag::SELECTION);
        }
    }

    void registerBrushOperators() {
        operators().registerOperator(BuiltinOp::BrushStroke, BrushStrokeOperator::DESCRIPTOR,
                                     [] { return std::make_unique<BrushStrokeOperator>(); });
    }

    void unregisterBrushOperators() {
        operators().unregisterOperator(BuiltinOp::BrushStroke);
    }

} // namespace lfs::vis::op
