/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_offscreen_renderer.hpp"

#include "viewport_tensor_meshes.hpp"

namespace lfs::vis {

    float linearizeMeshViewDepth(const float z_ndc, const glm::mat4& projection) noexcept {
        if (projection[2][3] == -1.0f)
            return projection[3][2] / (z_ndc + projection[2][2]);
        return -(z_ndc - projection[3][2]) / projection[2][2];
    }

    struct MeshOffscreenRenderer::Impl {
        TensorMeshPass mesh_pass;
    };

    MeshOffscreenRenderer::MeshOffscreenRenderer()
        : impl_(std::make_unique<Impl>()) {}
    MeshOffscreenRenderer::~MeshOffscreenRenderer() = default;
    MeshOffscreenRenderer::MeshOffscreenRenderer(MeshOffscreenRenderer&&) noexcept = default;
    MeshOffscreenRenderer& MeshOffscreenRenderer::operator=(MeshOffscreenRenderer&&) noexcept = default;

    lfs::Result<MeshLayer> MeshOffscreenRenderer::render(
        GraphicsContext&, const ViewportMeshPassDesc& params,
        const glm::mat4& projection, const int width, const int height) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->mesh_pass.renderOffscreen(params, projection, width, height);
    }

    void MeshOffscreenRenderer::shutdown() {
        impl_.reset();
    }

} // namespace lfs::vis
