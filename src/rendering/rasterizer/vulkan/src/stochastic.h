/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer.h"
#include <array>
#include <cstdint>

inline constexpr std::size_t VULKAN_GS_STOCHASTIC_HISTORY_SIZE = 4;

// Separate allocations keep uint4 and float4 storage strides explicit. These
// buffers belong to one logical viewport, not the rotating output-image ring.
struct VulkanGSStochasticBuffers {
    // Two words per occluder: the splat ID, then a 16-bit depth key and four
    // 4-bit occlusion ratios.
    Buffer<std::uint32_t> occluders;
    Buffer<float> outgoing_weights;
    std::array<Buffer<float>, 2> color;
    Buffer<float> rest_average;
};

struct VulkanGSStochasticResolveUniforms {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t mode;
    std::uint32_t history_valid;
    std::uint32_t sample_count;
    std::uint32_t orthographic;
    std::uint32_t previous_orthographic;
    std::uint32_t temporal_projection_valid;
    float intrinsics[4];
    float previous_intrinsics[4];
    float view_to_previous[3][4];
    float origins[4];
    float background[4];
};
static_assert(sizeof(VulkanGSStochasticResolveUniforms) == 144);
