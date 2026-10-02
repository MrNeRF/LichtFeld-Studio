/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Exact order statistics on Metal, the radix select of densification_kernels.cu
// (kernels in mrnf.metal, host side in mrnf_metal.cpp).
namespace lfs::training::metal {

    // A rank that selects total / 2 of the counted elements.
    inline constexpr uint32_t kSelectMedian = 0xffffffffu;

    // Selection s picks sorted[ranks[s]] of values[offsets[s] + i * stride],
    // i < count, in cub::DeviceRadixSort's float order; positive_only counts
    // only values > 0. Up to 8 selections share one launch. The returned
    // Int32 workspace holds selection s's (total, rank, prefix, value bits)
    // at element 4 s.
    core::Tensor radix_select(const core::Tensor& values, size_t count, uint32_t stride, bool positive_only,
                              std::span<const uint32_t> offsets, std::span<const uint32_t> ranks);

    // Reads the selected values of a workspace back to the host.
    std::vector<float> selected_values(const core::Tensor& workspace, size_t selections);

} // namespace lfs::training::metal
