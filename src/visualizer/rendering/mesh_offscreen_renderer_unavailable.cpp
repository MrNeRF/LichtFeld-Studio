/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_offscreen_renderer.hpp"

namespace lfs::vis {

    struct MeshOffscreenRenderer::Impl {};

    MeshOffscreenRenderer::MeshOffscreenRenderer()
        : impl_(std::make_unique<Impl>()) {}
    MeshOffscreenRenderer::~MeshOffscreenRenderer() = default;
    MeshOffscreenRenderer::MeshOffscreenRenderer(MeshOffscreenRenderer&&) noexcept = default;
    MeshOffscreenRenderer& MeshOffscreenRenderer::operator=(MeshOffscreenRenderer&&) noexcept = default;

    lfs::Result<MeshLayer> MeshOffscreenRenderer::render(
        GraphicsContext&, const ViewportMeshPassDesc&, const glm::mat4&, int, int) {
        return lfs::make_error({
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Rendering,
            .user_message =
                "Mesh viewport rendering is unavailable on the native Metal path in Phase 2",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

    void MeshOffscreenRenderer::shutdown() {}

} // namespace lfs::vis
