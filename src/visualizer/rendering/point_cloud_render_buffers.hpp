/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include "core/tensor_vulkan_interop.hpp"

namespace lfs::core {
    // Viewer-only definition of the opaque payload declared in PointCloud.
    // Prepared off the viewer thread; tensor storage, buffer leases and upload
    // completion remain alive through every renderer submission using them.
    // Only valid for these exact tensors.
    struct PointCloudRenderBuffers {
        Tensor positions;
        Tensor colors;
        TensorVulkanBuffer positions_buffer;
        TensorVulkanBuffer colors_buffer;
        VulkanTimelinePoint ready;
    };
} // namespace lfs::core
