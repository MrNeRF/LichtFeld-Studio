/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "gui/gui_manager.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rasterizer/rasterization/include/forward.h"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer_impl.hpp"
#include <cuda_runtime.h>
#include <glm/gtc/matrix_transform.hpp>
#include <optional>

namespace lfs::vis::op {

    namespace {

        struct ViewportBounds {
            float x = 0, y = 0, width = 0, height = 0;
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

        glm::vec3 screenToWorld(float abs_x, float abs_y) {
            auto* rm = services().renderingOrNull();
            auto* gm = services().guiOrNull();
            if (!rm || !gm || !gm->getViewer()) {
                return glm::vec3(-1e10f);
            }

            const auto& viewport = gm->getViewer()->getViewport();
            const auto bounds = getViewportBounds();
            const float local_x = abs_x - bounds.x;
            const float local_y = abs_y - bounds.y;

            const float depth = rm->getDepthAtPixel(
                static_cast<int>(local_x), static_cast<int>(local_y));

            if (depth > 0.0f) {
                const glm::vec3 world = viewport.unprojectPixel(
                    local_x, local_y, depth, rm->getFocalLengthMm());
                if (world.x > -1e9f) {
                    return world;
                }
            }

            const float pivot_dist = glm::length(viewport.camera.pivot - viewport.camera.t);
            const float fallback_dist = pivot_dist > 0.1f ? pivot_dist : 10.0f;
            const glm::vec3 forward = viewport.camera.R * glm::vec3(0, 0, 1);
            return viewport.camera.t + forward * fallback_dist;
        }

        glm::vec2 worldToScreen(const glm::vec3& world_pos, const Viewport& viewport, float focal_mm) {
            const glm::mat4 view = viewport.getViewMatrix();
            const glm::mat4 proj = viewport.getProjectionMatrix(focal_mm);
            const glm::vec4 clip = proj * view * glm::vec4(world_pos, 1.0f);
            if (clip.w <= 0.0f) {
                return glm::vec2(-1e6f);
            }
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            const auto bounds = getViewportBounds();
            return glm::vec2(
                bounds.x + (ndc.x * 0.5f + 0.5f) * bounds.width,
                bounds.y + (1.0f - (ndc.y * 0.5f + 0.5f)) * bounds.height);
        }

        std::optional<lfs::core::Tensor> projectPolygonToRenderSpace(
            const std::vector<glm::vec3>& world_points,
            const glm::mat4& vp,
            int render_w, int render_h) {
            const size_t num_verts = world_points.size();
            auto poly_cpu = lfs::core::Tensor::empty(
                {num_verts, 2}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto* data = poly_cpu.ptr<float>();
            for (size_t i = 0; i < num_verts; ++i) {
                const glm::vec4 clip = vp * glm::vec4(world_points[i], 1.0f);
                if (clip.w <= 0.0f) {
                    return std::nullopt;
                }
                const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                data[i * 2] = (ndc.x * 0.5f + 0.5f) * static_cast<float>(render_w);
                data[i * 2 + 1] = (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(render_h);
            }
            return poly_cpu;
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

    bool SelectionStrokeOperator::poll(const OperatorContext& ctx) const {
        return ctx.scene().getScene().getTotalGaussianCount() > 0;
    }

    OperatorResult SelectionStrokeOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto mode_int = props.get_or<int>("mode", 0);
        mode_ = static_cast<SelectionMode>(mode_int);

        const auto op_int = props.get_or<int>("op", 0);
        op_ = static_cast<SelectionOp>(op_int);

        brush_radius_ = props.get_or<float>("brush_radius", 20.0f);
        use_depth_filter_ = props.get_or<bool>("use_depth_filter", false);

        const auto x = props.get_or<double>("x", 0.0);
        const auto y = props.get_or<double>("y", 0.0);

        beginStroke(ctx);

        if (mode_ == SelectionMode::Brush || mode_ == SelectionMode::Rings) {
            last_stroke_pos_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
            updateBrushSelection(x, y, ctx);
        } else if (mode_ == SelectionMode::Rectangle) {
            rect_start_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
            rect_end_ = rect_start_;
        } else if (mode_ == SelectionMode::Lasso) {
            lasso_points_.clear();
            lasso_points_.emplace_back(static_cast<float>(x), static_cast<float>(y));
        } else if (mode_ == SelectionMode::Polygon) {
            polygon_world_points_.clear();
            polygon_closed_ = false;
            polygon_world_points_.push_back(screenToWorld(static_cast<float>(x), static_cast<float>(y)));
            updatePolygonPreview(ctx);
        }

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty(DirtyFlag::SELECTION);
        }

        return OperatorResult::RUNNING_MODAL;
    }

    OperatorResult SelectionStrokeOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
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

            if (mode_ == SelectionMode::Brush || mode_ == SelectionMode::Rings) {
                updateBrushSelection(x, y, ctx);
            } else if (mode_ == SelectionMode::Rectangle) {
                rect_end_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));
                updateRectPreview(ctx);
            } else if (mode_ == SelectionMode::Lasso) {
                const glm::vec2 new_point(static_cast<float>(x), static_cast<float>(y));
                if (lasso_points_.empty() || glm::distance(lasso_points_.back(), new_point) > 3.0f) {
                    lasso_points_.push_back(new_point);
                    updateLassoPreview(ctx);
                }
            }

            if (mode_ == SelectionMode::Polygon) {
                return OperatorResult::PASS_THROUGH;
            }

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

            if (mode_ == SelectionMode::Polygon) {
                if (mb->button == static_cast<int>(input::AppMouseButton::LEFT) && mb->action == input::ACTION_PRESS) {
                    const glm::vec2 click_pos(static_cast<float>(mb->position.x),
                                              static_cast<float>(mb->position.y));

                    if (polygon_world_points_.size() >= 3 && !polygon_closed_) {
                        auto* gm = services().guiOrNull();
                        auto* rm = services().renderingOrNull();
                        if (gm && gm->getViewer() && rm) {
                            const auto& vp = gm->getViewer()->getViewport();
                            const glm::vec2 first_screen = worldToScreen(
                                polygon_world_points_.front(), vp, rm->getFocalLengthMm());
                            if (glm::distance(click_pos, first_screen) < POLYGON_CLOSE_THRESHOLD) {
                                polygon_closed_ = true;
                                updatePolygonPreview(ctx);
                                if (rm) {
                                    rm->markDirty(DirtyFlag::SELECTION);
                                }
                                return OperatorResult::RUNNING_MODAL;
                            }
                        }
                    }

                    if (!polygon_closed_) {
                        polygon_world_points_.push_back(screenToWorld(click_pos.x, click_pos.y));
                        updatePolygonPreview(ctx);
                        if (services().renderingOrNull()) {
                            services().renderingOrNull()->markDirty(DirtyFlag::SELECTION);
                        }
                    }
                    return OperatorResult::RUNNING_MODAL;
                }

                if (mb->button == static_cast<int>(input::AppMouseButton::RIGHT) && mb->action == input::ACTION_PRESS) {
                    if (polygon_world_points_.size() > 1) {
                        polygon_world_points_.pop_back();
                        polygon_closed_ = false;
                        updatePolygonPreview(ctx);
                        if (services().renderingOrNull()) {
                            services().renderingOrNull()->markDirty(DirtyFlag::SELECTION);
                        }
                        return OperatorResult::RUNNING_MODAL;
                    }
                    return OperatorResult::CANCELLED;
                }
                return OperatorResult::PASS_THROUGH;
            }

            if (mb->button == static_cast<int>(input::AppMouseButton::LEFT) && mb->action == input::ACTION_RELEASE) {
                clearPreview(ctx);

                if (mode_ == SelectionMode::Rectangle) {
                    computeRectSelection(ctx);
                } else if (mode_ == SelectionMode::Lasso) {
                    computeLassoSelection(ctx);
                }

                finalizeSelection(ctx);
                return OperatorResult::FINISHED;
            }

            if (mb->button == static_cast<int>(input::AppMouseButton::RIGHT) && mb->action == input::ACTION_PRESS) {
                return OperatorResult::CANCELLED;
            }
        }

        if (event->type == ModalEvent::Type::MOUSE_SCROLL && mode_ == SelectionMode::Polygon) {
            return OperatorResult::PASS_THROUGH;
        }

        if (event->type == ModalEvent::Type::KEY) {
            const auto* ke = event->as<KeyEvent>();
            if (!ke || ke->action != input::ACTION_PRESS) {
                return OperatorResult::RUNNING_MODAL;
            }

            if (ke->key == input::KEY_ESCAPE) {
                return OperatorResult::CANCELLED;
            }

            if (mode_ == SelectionMode::Polygon && ke->key == input::KEY_ENTER) {
                if (polygon_world_points_.size() >= 3) {
                    polygon_closed_ = true;
                    // Update selection op based on current modifiers
                    if (ke->mods & input::KEYMOD_SHIFT) {
                        op_ = SelectionOp::Add;
                    } else if (ke->mods & input::KEYMOD_CTRL) {
                        op_ = SelectionOp::Remove;
                    } else {
                        op_ = SelectionOp::Replace;
                    }
                    clearPreview(ctx);
                    computePolygonSelection(ctx);
                    finalizeSelection(ctx);
                    return OperatorResult::FINISHED;
                }
                return OperatorResult::RUNNING_MODAL;
            }
        }

        return OperatorResult::RUNNING_MODAL;
    }

    void SelectionStrokeOperator::cancel(OperatorContext& ctx) {
        clearPreview(ctx);

        // Restore original selection
        if (selection_before_ && selection_before_->is_valid()) {
            ctx.scene().getScene().setSelectionMask(selection_before_);
        }

        stroke_selection_ = lfs::core::Tensor();
        selection_before_.reset();
        lasso_points_.clear();
        polygon_world_points_.clear();
        polygon_closed_ = false;

        if (auto* rm = services().renderingOrNull()) {
            rm->clearPreviewSelection();
            rm->clearBrushState();
            rm->clearPolygonPreview();
            rm->markDirty(DirtyFlag::SELECTION);
        }
    }

    void SelectionStrokeOperator::beginStroke(OperatorContext& ctx) {
        auto& scene = ctx.scene().getScene();
        const size_t n = scene.getTotalGaussianCount();
        if (n == 0) {
            return;
        }

        const auto existing = scene.getSelectionMask();
        selection_before_ = (existing && existing->is_valid())
                                ? std::make_shared<lfs::core::Tensor>(existing->clone())
                                : nullptr;

        stroke_selection_ = lfs::core::Tensor::zeros({n}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
    }

    void SelectionStrokeOperator::updateBrushSelection(double x, double y, OperatorContext& ctx) {
        if (!stroke_selection_.is_valid()) {
            return;
        }

        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        const auto bounds = getViewportBounds();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : static_cast<int>(viewport.windowSize.x);
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : static_cast<int>(viewport.windowSize.y);

        if (bounds.width <= 0 || bounds.height <= 0) {
            return;
        }

        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;
        const bool add_mode = (op_ != SelectionOp::Remove);

        const auto node_mask = ctx.scene().getSelectedNodeMask();
        const auto transform_indices = ctx.scene().getScene().getTransformIndices();

        if (mode_ == SelectionMode::Rings) {
            const float rel_x = static_cast<float>(x) - bounds.x;
            const float rel_y = static_cast<float>(y) - bounds.y;
            const float image_x = rel_x * scale_x;
            const float image_y = rel_y * scale_y;

            const int hovered_id = rm->getHoveredGaussianId();
            if (hovered_id >= 0) {
                lfs::rendering::set_selection_element(stroke_selection_.ptr<bool>(), hovered_id, add_mode);
                if (transform_indices && !node_mask.empty()) {
                    lfs::rendering::filter_selection_by_node_mask(stroke_selection_, *transform_indices, node_mask);
                }
                if (use_depth_filter_) {
                    rm->applyCropFilter(stroke_selection_);
                }
            }
            rm->setBrushState(true, image_x, image_y, 0.0f, add_mode, nullptr, false, 0.0f);
        } else {
            constexpr float STEP_FACTOR = 0.5f;
            const glm::vec2 current_pos(static_cast<float>(x), static_cast<float>(y));
            const glm::vec2 delta = current_pos - last_stroke_pos_;
            const float scaled_radius = brush_radius_ * scale_x;
            const int num_steps = std::max(1, static_cast<int>(std::ceil(glm::length(delta) / (brush_radius_ * STEP_FACTOR))));

            for (int i = 0; i < num_steps; ++i) {
                const float t = (num_steps == 1) ? 1.0f : static_cast<float>(i + 1) / static_cast<float>(num_steps);
                const glm::vec2 pos = last_stroke_pos_ + delta * t;
                const float image_x = (pos.x - bounds.x) * scale_x;
                const float image_y = (pos.y - bounds.y) * scale_y;
                rm->brushSelect(image_x, image_y, scaled_radius, stroke_selection_);
            }

            if (transform_indices && !node_mask.empty()) {
                lfs::rendering::filter_selection_by_node_mask(stroke_selection_, *transform_indices, node_mask);
            }
            if (use_depth_filter_) {
                rm->applyCropFilter(stroke_selection_);
            }

            const float final_x = (current_pos.x - bounds.x) * scale_x;
            const float final_y = (current_pos.y - bounds.y) * scale_y;
            rm->setBrushState(true, final_x, final_y, scaled_radius, add_mode, nullptr);
            last_stroke_pos_ = current_pos;
        }

        rm->setPreviewSelection(&stroke_selection_, add_mode);
    }

    void SelectionStrokeOperator::updateRectPreview(OperatorContext& /*ctx*/) {
        if (!stroke_selection_.is_valid()) {
            return;
        }

        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return;
        }

        const auto screen_positions = rm->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) {
            return;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        const auto bounds = getViewportBounds();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : static_cast<int>(viewport.windowSize.x);
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : static_cast<int>(viewport.windowSize.y);

        if (bounds.width <= 0 || bounds.height <= 0) {
            return;
        }

        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const float x0 = (std::min(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float y0 = (std::min(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;
        const float x1 = (std::max(rect_start_.x, rect_end_.x) - bounds.x) * scale_x;
        const float y1 = (std::max(rect_start_.y, rect_end_.y) - bounds.y) * scale_y;

        lfs::rendering::rect_select_tensor(*screen_positions, x0, y0, x1, y1, stroke_selection_);

        const bool add_mode = (op_ != SelectionOp::Remove);
        rm->setPreviewSelection(&stroke_selection_, add_mode);
    }

    void SelectionStrokeOperator::updateLassoPreview(OperatorContext& /*ctx*/) {
        if (!stroke_selection_.is_valid() || lasso_points_.size() < 3) {
            return;
        }

        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return;
        }

        const auto screen_positions = rm->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) {
            return;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        const auto bounds = getViewportBounds();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : static_cast<int>(viewport.windowSize.x);
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : static_cast<int>(viewport.windowSize.y);

        if (bounds.width <= 0 || bounds.height <= 0) {
            return;
        }

        const float scale_x = static_cast<float>(render_w) / bounds.width;
        const float scale_y = static_cast<float>(render_h) / bounds.height;

        const size_t num_verts = lasso_points_.size();
        auto poly_cpu = lfs::core::Tensor::empty({num_verts, 2}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* data = poly_cpu.ptr<float>();
        for (size_t i = 0; i < num_verts; ++i) {
            data[i * 2] = (lasso_points_[i].x - bounds.x) * scale_x;
            data[i * 2 + 1] = (lasso_points_[i].y - bounds.y) * scale_y;
        }
        const auto poly_gpu = poly_cpu.cuda();

        lfs::rendering::polygon_select_tensor(*screen_positions, poly_gpu, stroke_selection_);

        const bool add_mode = (op_ != SelectionOp::Remove);
        rm->setPreviewSelection(&stroke_selection_, add_mode);

        std::vector<std::pair<float, float>> preview(num_verts);
        for (size_t i = 0; i < num_verts; ++i)
            preview[i] = {data[i * 2], data[i * 2 + 1]};
        rm->setLassoPreview(preview, add_mode);
    }

    void SelectionStrokeOperator::computeRectSelection(OperatorContext& ctx) {
        updateRectPreview(ctx);
    }

    void SelectionStrokeOperator::computeLassoSelection(OperatorContext& ctx) {
        updateLassoPreview(ctx);
    }

    void SelectionStrokeOperator::updatePolygonPreview(OperatorContext& /*ctx*/) {
        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        const auto& cached = rm->getCachedResult();
        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : static_cast<int>(viewport.windowSize.x);
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : static_cast<int>(viewport.windowSize.y);

        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix(rm->getFocalLengthMm());
        const glm::mat4 vp = proj * view;

        const bool add_mode = (op_ != SelectionOp::Remove);
        rm->setPolygonPreview(polygon_world_points_, polygon_closed_, add_mode);

        if (stroke_selection_.is_valid() && polygon_world_points_.size() >= 3) {
            const auto screen_positions = rm->getScreenPositions();
            if (screen_positions && screen_positions->is_valid()) {
                const auto poly_cpu = projectPolygonToRenderSpace(polygon_world_points_, vp, render_w, render_h);
                if (poly_cpu) {
                    const auto poly_gpu = poly_cpu->cuda();
                    lfs::rendering::polygon_select_tensor(*screen_positions, poly_gpu, stroke_selection_);
                    rm->setPreviewSelection(&stroke_selection_, add_mode);
                }
            }
        }
    }

    void SelectionStrokeOperator::computePolygonSelection(OperatorContext& /*ctx*/) {
        if (!stroke_selection_.is_valid() || polygon_world_points_.size() < 3) {
            return;
        }

        auto* rm = services().renderingOrNull();
        auto* gm = services().guiOrNull();
        if (!rm || !gm || !gm->getViewer()) {
            return;
        }

        const auto screen_positions = rm->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid()) {
            return;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        const auto& cached = rm->getCachedResult();

        const int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : static_cast<int>(viewport.windowSize.x);
        const int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : static_cast<int>(viewport.windowSize.y);

        const glm::mat4 view = viewport.getViewMatrix();
        const glm::mat4 proj = viewport.getProjectionMatrix(rm->getFocalLengthMm());
        const glm::mat4 vp = proj * view;

        const auto poly_cpu = projectPolygonToRenderSpace(polygon_world_points_, vp, render_w, render_h);
        if (!poly_cpu) {
            return;
        }
        const auto poly_gpu = poly_cpu->cuda();

        lfs::rendering::polygon_select_tensor(*screen_positions, poly_gpu, stroke_selection_);
    }

    void SelectionStrokeOperator::finalizeSelection(OperatorContext& ctx) {
        if (!stroke_selection_.is_valid()) {
            return;
        }

        auto& scene_manager = ctx.scene();
        auto& scene = scene_manager.getScene();

        const auto node_mask = scene_manager.getSelectedNodeMask();
        if (node_mask.empty()) {
            return;
        }

        const uint8_t group_id = scene.getActiveSelectionGroup();
        const auto existing_mask = scene.getSelectionMask();
        const size_t n = stroke_selection_.numel();

        // Build locked groups bitmask
        uint32_t locked_bitmask[8] = {0};
        for (const auto& group : scene.getSelectionGroups()) {
            if (group.locked) {
                locked_bitmask[group.id / 32] |= (1u << (group.id % 32));
            }
        }

        uint32_t* d_locked = nullptr;
        cudaMalloc(&d_locked, sizeof(locked_bitmask));
        cudaMemcpy(d_locked, locked_bitmask, sizeof(locked_bitmask), cudaMemcpyHostToDevice);

        auto output_mask = lfs::core::Tensor::empty({n}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);

        const lfs::core::Tensor EMPTY_MASK;
        const lfs::core::Tensor& existing_ref = (existing_mask && existing_mask->is_valid())
                                                    ? *existing_mask
                                                    : EMPTY_MASK;
        const auto transform_indices = scene.getTransformIndices();
        const bool add_mode = (op_ != SelectionOp::Remove);
        const bool replace_mode = (op_ == SelectionOp::Replace);

        // Apply crop filter if enabled
        if (use_depth_filter_) {
            if (auto* rm = services().renderingOrNull()) {
                rm->applyCropFilter(stroke_selection_);
            }
        }

        lfs::rendering::apply_selection_group_tensor_mask(
            stroke_selection_, existing_ref, output_mask, group_id, d_locked,
            add_mode, transform_indices.get(), node_mask, replace_mode);
        cudaFree(d_locked);

        auto entry = std::make_unique<SceneSnapshot>(scene_manager, "select.stroke");
        entry->captureSelection();

        auto new_selection = std::make_shared<lfs::core::Tensor>(std::move(output_mask));
        scene.setSelectionMask(new_selection);

        entry->captureAfter();
        undoHistory().push(std::move(entry));

        stroke_selection_ = lfs::core::Tensor();
        lasso_points_.clear();
        polygon_world_points_.clear();
        polygon_closed_ = false;

        if (auto* rm = services().renderingOrNull()) {
            rm->clearPreviewSelection();
            rm->clearBrushState();
            rm->clearPolygonPreview();
            rm->clearLassoPreview();
            rm->markDirty(DirtyFlag::SELECTION);
        }
    }

    void SelectionStrokeOperator::clearPreview(OperatorContext& /*ctx*/) {
        if (auto* rm = services().renderingOrNull()) {
            rm->clearPreviewSelection();
            rm->clearPolygonPreview();
            rm->clearLassoPreview();
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
