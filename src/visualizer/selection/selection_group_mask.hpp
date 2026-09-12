/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <expected>
#include <string>

#include <cuda_runtime.h>

namespace lfs::vis::selection {

    inline constexpr size_t LOCKED_GROUPS_WORDS = 8;
    inline constexpr size_t kSelectionGroupCount = LOCKED_GROUPS_WORDS * 32;
    using LockedGroupMaskWords = std::array<uint32_t, LOCKED_GROUPS_WORDS>;

    [[nodiscard]] inline LockedGroupMaskWords build_locked_group_mask(const core::Scene& scene) {
        LockedGroupMaskWords locked_bitmask{};
        for (const auto& group : scene.getSelectionGroups()) {
            if (group.locked) {
                const size_t group_id = static_cast<size_t>(group.id);
                assert(group_id < kSelectionGroupCount && "selection group id exceeds locked-group mask capacity");
                if (group_id >= kSelectionGroupCount)
                    continue;
                locked_bitmask[group_id / 32] |= (1u << (group_id % 32));
            }
        }
        return locked_bitmask;
    }

    inline std::expected<uint32_t*, std::string> upload_locked_group_mask(
        const core::Scene& scene,
        core::Tensor& device_mask,
        LockedGroupMaskWords& cached_host_mask,
        bool& cached_host_mask_valid) {
        bool device_recreated = false;
        if (!device_mask.is_valid() ||
            device_mask.device() != core::Device::GPU ||
            device_mask.dtype() != core::DataType::Int32 ||
            device_mask.numel() != LOCKED_GROUPS_WORDS) {
            device_mask = core::Tensor::zeros({LOCKED_GROUPS_WORDS}, core::Device::GPU, core::DataType::Int32);
            device_recreated = true;
        }

        const auto locked_bitmask = build_locked_group_mask(scene);
        const bool needs_upload =
            device_recreated || !cached_host_mask_valid || locked_bitmask != cached_host_mask;
        const bool host_locked_pointer =
            !lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA) ||
            lfs::core::gpu_backend_of(device_mask) == lfs::core::GpuBackend::Vulkan;

        if (needs_upload) {
            cached_host_mask = locked_bitmask;
            if (host_locked_pointer) {
                cached_host_mask_valid = true;
            } else if (const auto err = cudaMemcpyAsync(device_mask.ptr<uint32_t>(),
                                                        cached_host_mask.data(),
                                                        sizeof(cached_host_mask),
                                                        cudaMemcpyHostToDevice,
                                                        device_mask.stream());
                       err != cudaSuccess) {
                cached_host_mask_valid = false;
                return std::unexpected(cudaGetErrorString(err));
            } else {
                cached_host_mask_valid = true;
            }
        }

        if (host_locked_pointer) {
            return cached_host_mask.data();
        }
        return device_mask.ptr<uint32_t>();
    }

    inline std::expected<uint32_t*, std::string> upload_locked_group_mask(const core::Scene& scene,
                                                                          core::Tensor& device_mask) {
        LockedGroupMaskWords cached_host_mask{};
        bool cached_host_mask_valid = false;
        return upload_locked_group_mask(scene, device_mask, cached_host_mask, cached_host_mask_valid);
    }

} // namespace lfs::vis::selection
