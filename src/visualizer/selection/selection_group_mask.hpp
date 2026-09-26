/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace lfs::vis::selection {

    inline constexpr size_t kSelectionGroupCount = 256;
    using LockedGroupMask = std::array<bool, kSelectionGroupCount>;

    [[nodiscard]] inline LockedGroupMask build_locked_group_mask(const core::Scene& scene) {
        LockedGroupMask locked_mask{};
        for (const auto& group : scene.getSelectionGroups()) {
            if (group.locked) {
                locked_mask[group.id] = true;
            }
        }
        return locked_mask;
    }

    inline const core::Tensor& update_locked_group_mask(
        const core::Scene& scene,
        const core::Tensor& selection,
        core::Tensor& device_mask,
        LockedGroupMask& cached_host_mask,
        bool& cached_host_mask_valid) {
        const auto backend = core::gpu_backend_of(selection);
        if (!backend)
            throw std::invalid_argument("Selection group locks require a GPU selection tensor");
        const auto locked_mask = build_locked_group_mask(scene);
        if (!device_mask.is_valid() || core::gpu_backend_of(device_mask) != backend ||
            device_mask.dtype() != core::DataType::Bool ||
            device_mask.numel() != kSelectionGroupCount || !device_mask.is_contiguous() ||
            !cached_host_mask_valid || locked_mask != cached_host_mask) {
            const core::GpuBackendScope scope(*backend);
            // Publish a new version: an earlier queued selection may still read
            // the previous tensor. Unchanged lock state reuses its cached tensor.
            core::Tensor updated = core::Tensor::from_vector(std::vector<bool>(locked_mask.begin(), locked_mask.end()), {kSelectionGroupCount}, core::Device::GPU);
            device_mask = std::move(updated);
            cached_host_mask = locked_mask;
            cached_host_mask_valid = true;
        }
        return device_mask;
    }

} // namespace lfs::vis::selection
