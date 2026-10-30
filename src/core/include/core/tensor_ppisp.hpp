/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/device/ppisp_params.hpp"
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include <cstddef>
#include <type_traits>

namespace lfs::core {
    struct PpispRegion {
        int x_offset = 0;
        int y_offset = 0;
        int full_width = 0;
        int full_height = 0;
    };

    // Shared with the Metal and Vulkan PPISP parameter blocks.
    static_assert(std::is_standard_layout_v<PpispParams>);
    static_assert(sizeof(PpispParams) == 176);
    static_assert(offsetof(PpispParams, exposure_factor) == 0);
    static_assert(offsetof(PpispParams, vignetting) == 4);
    static_assert(offsetof(PpispParams, color_matrix) == 64);
    static_assert(offsetof(PpispParams, crf) == 100);
    static_assert(offsetof(PpispParams, y_offset) == 160);
    static_assert(offsetof(PpispParams, full_height) == 164);
    static_assert(offsetof(PpispParams, x_offset) == 168);
    static_assert(offsetof(PpispParams, full_width) == 172);

    // Float RGB CHW. Non-contiguous inputs are materialized on their own backend.
    LFS_CORE_API Tensor ppisp_apply(const Tensor& rgb, const PpispParams& params);
} // namespace lfs::core
