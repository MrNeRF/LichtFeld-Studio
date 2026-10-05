/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_kernel_module.hpp"
#include "rendering/rendering.hpp"
#include "viewport_frame_desc.hpp"

#include <memory>

namespace lfs::vis {
    class TensorFrameUploads;

    // Viewport meshes for the tensor compositor, ported from VulkanMeshPass:
    // PBR shading, optional shadow map and wireframe overlay, hidden behind
    // nearer splats through the splat depth.
    class LFS_VIS_API TensorMeshPass {
    public:
        TensorMeshPass();
        ~TensorMeshPass();
        TensorMeshPass(const TensorMeshPass&) = delete;
        TensorMeshPass& operator=(const TensorMeshPass&) = delete;

        // Draws desc.mesh_items into `destination` within `rect` (framebuffer pixels).
        void record(lfs::core::Tensor& destination, const ViewportFrameDesc& desc,
                    const lfs::core::GpuKernelModule::Scissor& rect, TensorFrameUploads& uploads);

        // Renders an isolated mesh layer. Color is CPU float32 [4,H,W] with
        // binary coverage alpha; depth is positive linear view depth [H,W],
        // with +INF outside mesh coverage.
        [[nodiscard]] lfs::Result<lfs::rendering::MeshLayer> renderOffscreen(
            const ViewportMeshPassDesc& desc, const glm::mat4& projection,
            int width, int height);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
