/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "select_ops.hpp"
#include "core/cuda/selection_ops.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/selection_domain.hpp"
#include "core/tensor.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"
#include <unordered_set>
#include <vector>

namespace lfs::vis::op {
    namespace {
        [[nodiscard]] std::vector<std::string> visibleCameraNodeNames(const SceneManager& scene) {
            std::vector<std::string> names;
            const auto& core_scene = scene.getScene();
            for (const auto* const node : core_scene.getNodes()) {
                if (node && node->type == lfs::core::NodeType::CAMERA && node->camera &&
                    core_scene.isNodeEffectivelyVisible(node->id)) {
                    names.push_back(node->name);
                }
            }
            return names;
        }
    } // namespace

    OperationResult SelectAll::execute(SceneManager& scene,
                                       const OperatorProperties& /*props*/,
                                       const std::any& /*input*/) {
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::PointCloud:
            if (const auto result = scene.selectAllPoints(); !result) {
                return OperationResult::failure(result.error());
            }
            return OperationResult::success();
        case SelectionDomain::Cameras:
            if (auto result = cap::selectNodes(scene, visibleCameraNodeNames(scene), "replace"); !result) {
                return OperationResult::failure(result.error());
            }
            return OperationResult::success();
        case SelectionDomain::Gaussians:
            break;
        }

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
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::Gaussians:
            return scene.getScene().getCombinedModel() != nullptr;
        case SelectionDomain::PointCloud:
            return resolveActivePointCloudNode(scene) != nullptr;
        case SelectionDomain::Cameras:
            return !visibleCameraNodeNames(scene).empty();
        }
        return false;
    }

    OperationResult SelectNone::execute(SceneManager& scene,
                                        const OperatorProperties& /*props*/,
                                        const std::any& /*input*/) {
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::PointCloud:
            if (const auto result = scene.clearPointSelection(); !result) {
                return OperationResult::failure(result.error());
            }
            return OperationResult::success();
        case SelectionDomain::Cameras:
            if (auto result = cap::clearNodeSelection(scene); !result) {
                return OperationResult::failure(result.error());
            }
            return OperationResult::success();
        case SelectionDomain::Gaussians:
            break;
        }

        scene.getScene().clearSelection();
        return OperationResult::success();
    }

    bool SelectNone::poll(SceneManager& scene) const {
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::Gaussians:
            return scene.getScene().hasSelection();
        case SelectionDomain::PointCloud:
            return scene.getActivePointSelectionCount() > 0;
        case SelectionDomain::Cameras:
            return !scene.getSelectedNodeNames().empty();
        }
        return false;
    }

    OperationResult SelectInvert::execute(SceneManager& scene,
                                          const OperatorProperties& /*props*/,
                                          const std::any& /*input*/) {
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::PointCloud:
            if (const auto result = scene.invertPointSelection(); !result) {
                return OperationResult::failure(result.error());
            }
            return OperationResult::success();
        case SelectionDomain::Cameras: {
            const auto visible_names = visibleCameraNodeNames(scene);
            std::unordered_set<std::string> selected_set;
            for (const auto& name : scene.getSelectedNodeNames()) {
                selected_set.insert(name);
            }
            std::vector<std::string> inverted;
            inverted.reserve(visible_names.size());
            for (const auto& name : visible_names) {
                if (!selected_set.contains(name)) {
                    inverted.push_back(name);
                }
            }
            if (inverted.empty()) {
                if (auto result = cap::clearNodeSelection(scene); !result) {
                    return OperationResult::failure(result.error());
                }
            } else if (auto result = cap::selectNodes(scene, inverted, "replace"); !result) {
                return OperationResult::failure(result.error());
            }
            return OperationResult::success();
        }
        case SelectionDomain::Gaussians:
            break;
        }

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

        auto group_id = scene.getScene().getActiveSelectionGroup();
        auto is_selected = mask->gt(0.0f);
        auto inverted = is_selected.logical_not();

        auto new_mask = lfs::core::Tensor::zeros({model->size()}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        new_mask.masked_fill_(inverted, static_cast<float>(group_id));

        scene.getScene().setSelectionMask(std::make_shared<lfs::core::Tensor>(std::move(new_mask)));

        return OperationResult::success();
    }

    bool SelectInvert::poll(SceneManager& scene) const {
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::Gaussians:
            return scene.getScene().getCombinedModel() != nullptr;
        case SelectionDomain::PointCloud:
            return resolveActivePointCloudNode(scene) != nullptr;
        case SelectionDomain::Cameras:
            return !visibleCameraNodeNames(scene).empty();
        }
        return false;
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

        const int iterations = props.get_or<int>("iterations", 1);
        const float radius = props.get_or<float>("radius", 0.1f);
        assert(iterations > 0);
        assert(radius > 0.0f);
        const auto group_id = scene.getScene().getActiveSelectionGroup();

        auto current = *mask;
        for (int i = 0; i < iterations; ++i) {
            current = core::cuda::selection_grow(current, model->means(), radius, group_id);
        }

        scene.getScene().setSelectionMask(std::make_shared<core::Tensor>(std::move(current)));
        return OperationResult::success();
    }

    bool SelectGrow::poll(SceneManager& scene) const {
        return resolveSelectionDomain(scene) == SelectionDomain::Gaussians &&
               scene.getScene().hasSelection();
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

        const int iterations = props.get_or<int>("iterations", 1);
        const float radius = props.get_or<float>("radius", 0.1f);
        assert(iterations > 0);
        assert(radius > 0.0f);

        auto current = *mask;
        for (int i = 0; i < iterations; ++i) {
            current = core::cuda::selection_shrink(current, model->means(), radius);
        }

        scene.getScene().setSelectionMask(std::make_shared<core::Tensor>(std::move(current)));
        return OperationResult::success();
    }

    bool SelectShrink::poll(SceneManager& scene) const {
        return resolveSelectionDomain(scene) == SelectionDomain::Gaussians &&
               scene.getScene().hasSelection();
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
        return resolveSelectionDomain(scene) == SelectionDomain::Gaussians &&
               scene.getScene().getCombinedModel() != nullptr;
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
        return resolveSelectionDomain(scene) == SelectionDomain::Gaussians &&
               scene.getScene().getCombinedModel() != nullptr;
    }

} // namespace lfs::vis::op
