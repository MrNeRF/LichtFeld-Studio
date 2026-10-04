/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "viewport_draw_types.hpp"
#include <core/error.hpp>
#include <core/export.hpp>
#include <core/tensor.hpp>
#include <rendering/rendering.hpp>

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
