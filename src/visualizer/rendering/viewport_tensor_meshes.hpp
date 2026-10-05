/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_kernel_module.hpp"
#include "viewport_frame_desc.hpp"

#include <memory>

namespace lfs::vis {
    class TensorFrameUploads;

    // Viewport meshes for the tensor compositor, ported from VulkanMeshPass:
    // PBR shading, optional shadow map and wireframe overlay, hidden behind
    // nearer splats through the splat depth.
    class TensorMeshPass {
    public:
        TensorMeshPass();
        ~TensorMeshPass();
        TensorMeshPass(const TensorMeshPass&) = delete;
        TensorMeshPass& operator=(const TensorMeshPass&) = delete;

        // Draws desc.mesh_items into `destination` within `rect` (framebuffer pixels).
        void record(lfs::core::Tensor& destination, const ViewportFrameDesc& desc,
                    const lfs::core::GpuKernelModule::Scissor& rect, TensorFrameUploads& uploads);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
