/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "viewport_draw_types.hpp"
#include <core/error.hpp>
#include <core/export.hpp>
#include <core/tensor.hpp>
#include <rendering/rendering.hpp>

#include <cstddef>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace lfs::vis {

    class GraphicsContext;

    // Owned by the rendering layer, which is what consumes it during the composite.
    using MeshLayer = lfs::rendering::MeshLayer;

    [[nodiscard]] LFS_VIS_API float linearizeMeshViewDepth(
        float z_ndc,
        const glm::mat4& projection) noexcept;

    // The full-image projection restricted to the pixel rectangle at origin, so a render of just
    // that rectangle samples the pixel centers of the full-size render.
    [[nodiscard]] LFS_VIS_API glm::mat4 cropProjectionToRect(
        const glm::mat4& projection,
        glm::ivec2 full_size,
        glm::ivec2 origin,
        glm::ivec2 size) noexcept;

    // Writes the mesh pixels in front of the splat depth into an 8-bit HWC image at origin, as the
    // viewport's depth-tested mesh pass does. splat_depth starts at the layer's first pixel and
    // advances splat_depth_stride per row; without it every mesh pixel is written.
    LFS_VIS_API void compositeMeshLayer(
        const MeshLayer& layer,
        const float* splat_depth,
        std::size_t splat_depth_stride,
        lfs::core::Tensor& image,
        glm::ivec2 origin);

    class LFS_VIS_API MeshOffscreenRenderer {
    public:
        MeshOffscreenRenderer();
        ~MeshOffscreenRenderer();

        MeshOffscreenRenderer(const MeshOffscreenRenderer&) = delete;
        MeshOffscreenRenderer& operator=(const MeshOffscreenRenderer&) = delete;
        MeshOffscreenRenderer(MeshOffscreenRenderer&&) noexcept;
        MeshOffscreenRenderer& operator=(MeshOffscreenRenderer&&) noexcept;

        [[nodiscard]] lfs::Result<MeshLayer> render(
            GraphicsContext& context,
            const ViewportMeshPassDesc& params,
            const glm::mat4& projection,
            int width,
            int height);

        void shutdown();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
