/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor_readback.hpp"
#include "lfs/training/ops/gsplat.hpp"
namespace lfs::training::vulkan {
    // What backward needs from the forward it follows.
    struct GsplatFrame {
        core::Tensor means, scales, quats, opacities, camera, bg_color, bg_image;
        core::Tensor radii, means2d, colors, tile_offsets, gaussian_ids, alpha, last_ids, depths;
        gpu_ops::GsplatRenderMode mode = gpu_ops::GsplatRenderMode::RGB;
        uint32_t count = 0, width = 0, height = 0, degree = 0, layout_rest = 0, intersections = 0;
    };

    struct VulkanGsplatState : gpu_ops::BackendState {
        GsplatFrame frame;
        bool live = false;
        bool indirect = true;
        core::Tensor control;
        std::string message;
        core::Tensor camera, radial, tangential, prism, image, alpha, last_ids, depth;
        core::Tensor radii, means2d, colors, depth_keys, tile_counts, tile_offsets, ends;
        core::Tensor keys_a, keys_b, ids_a, ids_b, grads;
        core::TensorReadback intersection_readback;
    };

} // namespace lfs::training::vulkan
