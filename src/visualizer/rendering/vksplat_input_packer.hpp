/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/tensor_vulkan_interop.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

namespace lfs::vis::vksplat {
    // The caller owns one copy per retired render slot and marks changed inputs.
    [[nodiscard]] LFS_VIS_API lfs::core::Tensor prepareDeletedMask(
        lfs::core::TensorVulkanInterop& interop,
        const lfs::core::Tensor& input,
        lfs::core::GpuBackend backend,
        bool upload_requested,
        lfs::core::Tensor& slot_copy);

    struct LFS_VIS_API RawDeviceInputLayout {
        std::size_t num_splats = 0;
        std::size_t xyz_bytes = 0;
        std::size_t sh0_bytes = 0;
        std::size_t shN_bytes = 0;
        std::size_t rotations_bytes = 0;
        std::size_t scaling_bytes = 0;
        std::size_t opacity_bytes = 0;
        std::uint32_t shN_layout_rest = 0;
        bool omits_shN = false;
        bool shN_f16 = false;
        bool shN_q16 = false;
        std::size_t shN_element_bytes = sizeof(float);
        std::uint32_t shN_n_cells = 0;
        std::size_t shN_bounds_bytes = 0;
        bool attrs_f16 = false;
        std::size_t non_sh_bytes = 0;
    };

    [[nodiscard]] LFS_VIS_API std::expected<RawDeviceInputLayout, std::string> rawDeviceInputLayout(
        const lfs::core::SplatData& splat_data, int upload_sh_degree = -1);
} // namespace lfs::vis::vksplat
