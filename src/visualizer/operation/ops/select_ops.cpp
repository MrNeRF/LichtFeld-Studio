/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "select_ops.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

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
        scene.getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(mask)));

        return OperationResult::success();
    }

    bool SelectAll::poll(SceneManager& scene) const {
        return scene.getScene().getCombinedModel() != nullptr;
    }

    OperationResult SelectNone::execute(SceneManager& scene,
                                        const OperatorProperties& /*props*/,
                                        const std::any& /*input*/) {
        scene.getScene().clearSelection();
        return OperationResult::success();
    }

    OperationResult SelectInvert::execute(SceneManager& scene,
                                          const OperatorProperties& /*props*/,
                                          const std::any& /*input*/) {
        auto mask = scene.getScene().getSelectionMask();
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

        auto group_id = scene.getScene().getActiveSelectionGroup();
        auto is_selected = mask->gt(0.0f);
        auto inverted = is_selected.logical_not();

        auto new_mask = lfs::core::Tensor::zeros({model->size()}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        new_mask.masked_fill_(inverted, static_cast<float>(group_id));

        scene.getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(new_mask)));

        return OperationResult::success();
    }

    bool SelectInvert::poll(SceneManager& scene) const {
        return scene.getScene().getCombinedModel() != nullptr;
    }

    OperationResult SelectGrow::execute(SceneManager& scene,
                                        const OperatorProperties& props,
                                        const std::any& /*input*/) {
        auto mask = scene.getScene().getSelectionMask();
        if (!mask || !scene.getScene().hasSelection()) {
            return OperationResult::skipped("No selection to grow");
        }

        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        auto iterations = props.get_or<int>("iterations", 1);
        auto radius = props.get_or<float>("radius", 0.1f);

        LOG_INFO("SelectGrow: iterations={}, radius={}", iterations, radius);

        return OperationResult::success();
    }

    bool SelectGrow::poll(SceneManager& scene) const {
        return scene.getScene().hasSelection();
    }

    OperationResult SelectShrink::execute(SceneManager& scene,
                                          const OperatorProperties& props,
                                          const std::any& /*input*/) {
        auto mask = scene.getScene().getSelectionMask();
        if (!mask || !scene.getScene().hasSelection()) {
            return OperationResult::skipped("No selection to shrink");
        }

        auto* model = scene.getScene().getCombinedModel();
        if (!model) {
            return OperationResult::failure("No model loaded");
        }

        auto iterations = props.get_or<int>("iterations", 1);
        auto radius = props.get_or<float>("radius", 0.1f);

        LOG_INFO("SelectShrink: iterations={}, radius={}", iterations, radius);

        return OperationResult::success();
    }

    bool SelectShrink::poll(SceneManager& scene) const {
        return scene.getScene().hasSelection();
    }

} // namespace lfs::vis::op
