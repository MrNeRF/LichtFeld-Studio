/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <string>

#include <cuda_runtime.h>

namespace lfs::vis::selection {

    inline constexpr size_t LOCKED_GROUPS_WORDS = 8;
    using LockedGroupMaskWords = std::array<uint32_t, LOCKED_GROUPS_WORDS>;

    [[nodiscard]] inline LockedGroupMaskWords build_locked_group_mask(const core::Scene& scene) {
        LockedGroupMaskWords locked_bitmask{};
        for (const auto& group : scene.getSelectionGroups()) {
            if (group.locked) {
                locked_bitmask[group.id / 32] |= (1u << (group.id % 32));
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
            device_mask.device() != core::Device::CUDA ||
            device_mask.dtype() != core::DataType::Int32 ||
            device_mask.numel() != LOCKED_GROUPS_WORDS) {
            device_mask = core::Tensor::zeros({LOCKED_GROUPS_WORDS}, core::Device::CUDA, core::DataType::Int32);
            device_recreated = true;
        }

        const auto locked_bitmask = build_locked_group_mask(scene);
        const bool needs_upload =
            device_recreated || !cached_host_mask_valid || locked_bitmask != cached_host_mask;

        if (needs_upload) {
            if (const auto err = cudaMemcpy(device_mask.ptr<uint32_t>(),
                                            locked_bitmask.data(),
                                            sizeof(locked_bitmask),
                                            cudaMemcpyHostToDevice);
                err != cudaSuccess) {
                return std::unexpected(cudaGetErrorString(err));
            }
            cached_host_mask = locked_bitmask;
            cached_host_mask_valid = true;
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
