/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "select_ops.hpp"
#include "core/cuda/selection_ops.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "scene/scene_manager.hpp"

#include <optional>

namespace lfs::vis::op {

    namespace {
        // Rows held by a locked group other than the target group; bulk selection leaves them alone.
        std::optional<lfs::core::Tensor> locked_rows(const lfs::core::Scene& scene,
                                                     const lfs::core::Tensor& values,
                                                     const uint8_t group_id) {
            std::optional<lfs::core::Tensor> rows;
            for (const auto& group : scene.getSelectionGroups()) {
                if (!group.locked || group.id == group_id)
                    continue;
                auto in_group = values.eq(static_cast<float>(group.id));
                rows = rows ? rows->logical_or(in_group) : std::move(in_group);
            }
            return rows;
        }

        bool has_non_identity_transform(const std::vector<glm::mat4>& transforms) {
            for (const auto& transform : transforms) {
                for (int column = 0; column < 4; ++column) {
                    for (int row = 0; row < 4; ++row) {
                        const float expected = column == row ? 1.0f : 0.0f;
                        if (transform[column][row] != expected)
                            return true;
                    }
                }
            }
            return false;
        }

        struct SelectionTransformContext {
            std::shared_ptr<core::Tensor> indices;
            std::vector<glm::mat4> transforms;

            [[nodiscard]] const core::Tensor* indices_ptr() const {
                return indices ? indices.get() : nullptr;
            }

            [[nodiscard]] const std::vector<glm::mat4>* transforms_ptr() const {
                return indices ? &transforms : nullptr;
            }
        };

        SelectionTransformContext selection_transform_context(core::Scene& scene, const size_t model_size) {
            SelectionTransformContext context;
            context.transforms = scene.getVisibleNodeTransforms();
            if (!has_non_identity_transform(context.transforms))
                return context;

            // Transform metadata can be temporarily unavailable while the
            // scene cache is rebuilding. Preserve the legacy local-space
            // behavior in that case instead of rejecting the operation.
            auto indices = scene.getTransformIndices();
            if (indices && indices->is_valid() && indices->numel() == model_size) {
                context.indices = std::move(indices);
            }
            return context;
        }
    } // namespace

    OperationResult SelectAll::execute(SceneManager& scene,
                                       const OperatorProperties& /*props*/,
                                       const std::any& /*input*/) {
        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        size_t count = model->size();
        auto group_id = scene.getScene().getActiveSelectionGroup();

        auto mask = lfs::core::Tensor::full({count}, static_cast<float>(group_id),
                                            lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        if (const auto existing = scene.getScene().getSelectionMask();
            existing && existing->is_valid() && existing->numel() == count) {
            const auto values = existing->cuda().to(lfs::core::DataType::UInt8);
            if (const auto rows = locked_rows(scene.getScene(), values, group_id))
                mask = values.where(*rows, mask);
        }
        scene.getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));

        return OperationResult::success();
    }

    bool SelectAll::poll(SceneManager& scene) const {
        return scene.getScene().getTotalGaussianCount() > 0;
    }

    OperationResult SelectNone::execute(SceneManager& scene,
                                        const OperatorProperties& /*props*/,
                                        const std::any& /*input*/) {
        scene.getScene().clearUnlockedSelection();
        return OperationResult::success();
    }

    OperationResult SelectInvert::execute(SceneManager& scene,
                                          const OperatorProperties& /*props*/,
                                          const std::any& /*input*/) {
        auto mask = scene.getScene().getVisibleSelectionMask();
        if (!mask) {
            auto* model = scene.getScene().getCombinedModel();
            if (!model) {
                return OperationResult::failure("No model loaded");
            }
            auto group_id = scene.getScene().getActiveSelectionGroup();
            auto new_mask = lfs::core::Tensor::full({model->size()}, static_cast<float>(group_id),
                                                    lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
            scene.getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(new_mask)));
            return OperationResult::success();
        }

        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        // Invert only the active group; rows of other groups keep their membership.
        auto group_id = scene.getScene().getActiveSelectionGroup();
        const auto values = mask->cuda().to(lfs::core::DataType::UInt8);
        const auto other_group = values.gt(0.0f).logical_and(values.eq(static_cast<float>(group_id)).logical_not());
        const auto newly_active = values.eq(0.0f);

        auto new_mask = values.where(other_group, lfs::core::Tensor::zeros_like(values));
        new_mask.masked_fill_(newly_active, static_cast<float>(group_id));

        scene.getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(new_mask)));

        return OperationResult::success();
    }

    bool SelectInvert::poll(SceneManager& scene) const {
        return scene.getScene().getTotalGaussianCount() > 0;
    }

    OperationResult SelectGrow::execute(SceneManager& scene,
                                        const OperatorProperties& props,
                                        const std::any& /*input*/) {
        auto mask = scene.getScene().getVisibleSelectionMask();
        if (!mask || !scene.getScene().hasSelection()) {
            return OperationResult::skipped("No selection to grow");
        }

        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        auto& core_scene = scene.getScene();
        const auto transform_context = selection_transform_context(core_scene, model->means().size(0));

        const int iterations = props.get_or<int>("iterations", 1);
        const float radius = props.get_or<float>("radius", 0.1f);
        assert(iterations > 0);
        assert(radius > 0.0f);
        const auto group_id = scene.getScene().getActiveSelectionGroup();

        auto current = *mask;
        for (int i = 0; i < iterations; ++i) {
            current = core::cuda::selection_grow(
                current, model->means(), radius, group_id,
                transform_context.indices_ptr(), transform_context.transforms_ptr());
        }

        scene.getScene().setSelectionMask(std::make_shared<core::Tensor>(std::move(current)));
        return OperationResult::success();
    }

    bool SelectGrow::poll(SceneManager& scene) const {
        return scene.getScene().hasSelection();
    }

    OperationResult SelectShrink::execute(SceneManager& scene,
                                          const OperatorProperties& props,
                                          const std::any& /*input*/) {
        auto mask = scene.getScene().getVisibleSelectionMask();
        if (!mask || !scene.getScene().hasSelection()) {
            return OperationResult::skipped("No selection to shrink");
        }

        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        auto& core_scene = scene.getScene();
        const auto transform_context = selection_transform_context(core_scene, model->means().size(0));

        const int iterations = props.get_or<int>("iterations", 1);
        const float radius = props.get_or<float>("radius", 0.1f);
        assert(iterations > 0);
        assert(radius > 0.0f);

        auto current = *mask;
        for (int i = 0; i < iterations; ++i) {
            current = core::cuda::selection_shrink(
                current, model->means(), radius,
                transform_context.indices_ptr(), transform_context.transforms_ptr());
        }

        scene.getScene().setSelectionMask(std::make_shared<core::Tensor>(std::move(current)));
        return OperationResult::success();
    }

    bool SelectShrink::poll(SceneManager& scene) const {
        return scene.getScene().hasSelection();
    }

    OperationResult SelectByOpacity::execute(SceneManager& scene,
                                             const OperatorProperties& props,
                                             const std::any& /*input*/) {
        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        const float min_opacity = props.get_or<float>("min_opacity", 0.0f);
        const float max_opacity = props.get_or<float>("max_opacity", 1.0f);
        const auto group_id = scene.getScene().getActiveSelectionGroup();

        auto new_mask = core::cuda::select_by_opacity(model->opacity_raw(), min_opacity, max_opacity, group_id);
        scene.getScene().setSelectionMask(std::make_shared<core::Tensor>(std::move(new_mask)));

        return OperationResult::success();
    }

    bool SelectByOpacity::poll(SceneManager& scene) const {
        return scene.getScene().getTotalGaussianCount() > 0;
    }

    OperationResult SelectByScale::execute(SceneManager& scene,
                                           const OperatorProperties& props,
                                           const std::any& /*input*/) {
        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        const float max_scale = props.get_or<float>("max_scale", 1.0f);
        const auto group_id = scene.getScene().getActiveSelectionGroup();

        auto new_mask = core::cuda::select_by_scale(model->scaling_raw(), max_scale, group_id);
        scene.getScene().setSelectionMask(std::make_shared<core::Tensor>(std::move(new_mask)));

        return OperationResult::success();
    }

    bool SelectByScale::poll(SceneManager& scene) const {
        return scene.getScene().getTotalGaussianCount() > 0;
    }

} // namespace lfs::vis::op
