/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_service.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

#include <cassert>

namespace lfs::vis {

    SelectionService::SelectionService(SceneManager* scene_manager, RenderingManager* rendering_manager)
        : scene_manager_(scene_manager),
          rendering_manager_(rendering_manager) {
        assert(scene_manager);
        assert(rendering_manager);
    }

    SelectionService::~SelectionService() = default;

    SelectionResult SelectionService::selectRect(float x0, float y0, float x1, float y1, SelectionMode mode,
                                                 [[maybe_unused]] int camera_index) {
        if (!scene_manager_ || !rendering_manager_)
            return {false, 0, "Missing managers"};

        auto& scene = scene_manager_->getScene();
        const size_t total = scene.getTotalGaussianCount();
        if (total == 0)
            return {false, 0, "No gaussians"};

        const auto screen_positions = rendering_manager_->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid())
            return {false, 0, "No screen positions"};

        auto selection = core::Tensor::zeros({total}, core::Device::CUDA, core::DataType::UInt8);
        rendering::rect_select_tensor(*screen_positions, x0, y0, x1, y1, selection);

        const auto old_mask = scene.getSelectionMask();
        auto new_mask = std::make_shared<core::Tensor>(applyModeLogic(selection, mode));
        scene.setSelectionMask(new_mask);

        rendering_manager_->markDirty(DirtyFlag::SELECTION);

        return {true, static_cast<size_t>(new_mask->to(core::DataType::Float32).sum_scalar()), ""};
    }

    SelectionResult SelectionService::selectPolygon(const core::Tensor& vertices, SelectionMode mode,
                                                    [[maybe_unused]] int camera_index) {
        if (!scene_manager_ || !rendering_manager_)
            return {false, 0, "Missing managers"};

        auto& scene = scene_manager_->getScene();
        const size_t total = scene.getTotalGaussianCount();
        if (total == 0)
            return {false, 0, "No gaussians"};

        const auto screen_positions = rendering_manager_->getScreenPositions();
        if (!screen_positions || !screen_positions->is_valid())
            return {false, 0, "No screen positions"};

        auto selection = core::Tensor::zeros({total}, core::Device::CUDA, core::DataType::UInt8);
        const auto& gpu_vertices = vertices.device() == core::Device::CUDA ? vertices : vertices.cuda();
        rendering::polygon_select_tensor(*screen_positions, gpu_vertices, selection);

        const auto old_mask = scene.getSelectionMask();
        auto new_mask = std::make_shared<core::Tensor>(applyModeLogic(selection, mode));
        scene.setSelectionMask(new_mask);

        rendering_manager_->markDirty(DirtyFlag::SELECTION);

        return {true, static_cast<size_t>(new_mask->to(core::DataType::Float32).sum_scalar()), ""};
    }

    SelectionResult SelectionService::applyMask(const std::vector<uint8_t>& mask, SelectionMode mode) {
        if (!scene_manager_)
            return {false, 0, "Missing scene manager"};

        const size_t total = scene_manager_->getScene().getTotalGaussianCount();
        if (total == 0 || mask.size() != total)
            return {false, 0, "Mask size mismatch"};

        auto tensor_mask = core::Tensor::from_vector(std::vector<bool>(mask.begin(), mask.end()), {total}, core::Device::CUDA);
        return applyMask(tensor_mask, mode);
    }

    SelectionResult SelectionService::applyMask(const core::Tensor& mask, SelectionMode mode) {
        if (!scene_manager_ || !rendering_manager_)
            return {false, 0, "Missing managers"};

        auto& scene = scene_manager_->getScene();
        const size_t total = scene.getTotalGaussianCount();
        if (total == 0 || mask.numel() != total)
            return {false, 0, "Mask size mismatch"};

        const auto old_mask = scene.getSelectionMask();
        auto new_mask = std::make_shared<core::Tensor>(applyModeLogic(mask, mode));
        scene.setSelectionMask(new_mask);

        rendering_manager_->markDirty(DirtyFlag::SELECTION);

        return {true, static_cast<size_t>(new_mask->to(core::DataType::Float32).sum_scalar()), ""};
    }

    void SelectionService::beginStroke() {
        if (!scene_manager_)
            return;

        const size_t n = scene_manager_->getScene().getTotalGaussianCount();
        if (n == 0)
            return;

        const auto existing = scene_manager_->getScene().getSelectionMask();
        selection_before_stroke_ =
            (existing && existing->is_valid()) ? std::make_shared<core::Tensor>(existing->clone()) : nullptr;

        stroke_selection_ = core::Tensor::zeros({n}, core::Device::CUDA, core::DataType::Bool);
        stroke_active_ = true;
    }

    core::Tensor* SelectionService::getStrokeSelection() {
        return stroke_active_ ? &stroke_selection_ : nullptr;
    }

    SelectionResult SelectionService::finalizeStroke(SelectionMode mode, const std::vector<bool>& node_mask) {
        if (!stroke_active_ || !stroke_selection_.is_valid())
            return {false, 0, "No active stroke"};

        if (!scene_manager_ || !rendering_manager_)
            return {false, 0, "Missing managers"};

        auto& scene = scene_manager_->getScene();
        const uint8_t group_id = scene.getActiveSelectionGroup();
        const auto existing_mask = scene.getSelectionMask();
        const size_t n = stroke_selection_.numel();

        uint32_t locked_bitmask[LOCKED_GROUPS_SIZE] = {0};
        for (const auto& group : scene.getSelectionGroups()) {
            if (group.locked) {
                locked_bitmask[group.id / 32] |= (1u << (group.id % 32));
            }
        }

        uint32_t* d_locked = nullptr;
        cudaMalloc(&d_locked, sizeof(locked_bitmask));
        cudaMemcpy(d_locked, locked_bitmask, sizeof(locked_bitmask), cudaMemcpyHostToDevice);

        auto output_mask = core::Tensor::empty({n}, core::Device::CUDA, core::DataType::UInt8);

        static const core::Tensor EMPTY_MASK;
        const auto& existing_ref = (existing_mask && existing_mask->is_valid()) ? *existing_mask : EMPTY_MASK;
        const auto transform_indices = scene.getTransformIndices();
        const bool add_mode = (mode != SelectionMode::Remove);
        const bool replace_mode = (mode == SelectionMode::Replace);

        rendering::apply_selection_group_tensor_mask(stroke_selection_, existing_ref, output_mask, group_id, d_locked,
                                                     add_mode, transform_indices.get(), node_mask, replace_mode);
        cudaFree(d_locked);

        auto new_selection = std::make_shared<core::Tensor>(std::move(output_mask));
        scene.setSelectionMask(new_selection);

        selection_before_stroke_.reset();
        stroke_selection_ = core::Tensor();
        stroke_active_ = false;

        rendering_manager_->clearPreviewSelection();
        rendering_manager_->clearBrushState();
        rendering_manager_->markDirty(DirtyFlag::SELECTION);

        return {true, static_cast<size_t>(new_selection->to(core::DataType::Float32).sum_scalar()), ""};
    }

    void SelectionService::cancelStroke() {
        selection_before_stroke_.reset();
        stroke_selection_ = core::Tensor();
        stroke_active_ = false;

        if (rendering_manager_) {
            rendering_manager_->clearPreviewSelection();
            rendering_manager_->clearBrushState();
        }
    }

    size_t SelectionService::getTotalGaussianCount() const {
        return scene_manager_ ? scene_manager_->getScene().getTotalGaussianCount() : 0;
    }

    bool SelectionService::hasScreenPositions() const {
        if (!rendering_manager_)
            return false;
        const auto sp = rendering_manager_->getScreenPositions();
        return sp && sp->is_valid();
    }

    std::shared_ptr<core::Tensor> SelectionService::getScreenPositions() const {
        return rendering_manager_ ? rendering_manager_->getScreenPositions() : nullptr;
    }

    core::Tensor SelectionService::applyModeLogic(const core::Tensor& stroke, SelectionMode mode) const {
        if (!scene_manager_)
            return stroke;

        const auto old_mask = scene_manager_->getScene().getSelectionMask();

        if (mode == SelectionMode::Add && old_mask && old_mask->is_valid()) {
            return (*old_mask) | stroke;
        }
        if (mode == SelectionMode::Remove && old_mask && old_mask->is_valid()) {
            const auto ones = core::Tensor::ones({stroke.numel()}, core::Device::CUDA, core::DataType::UInt8);
            return (*old_mask) * (ones - stroke);
        }
        return stroke;
    }

} // namespace lfs::vis
