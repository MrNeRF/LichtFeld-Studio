/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/mesh2splat.hpp"

namespace lfs::rendering {
    std::expected<std::unique_ptr<core::SplatData>, std::string>
    mesh_to_splat(const core::MeshData&, const core::Mesh2SplatOptions&,
                  core::Mesh2SplatProgressCallback) {
        return std::unexpected(
            "Mesh2Splat is unavailable in the Metal-only build: the current converter requires Vulkan");
    }
} // namespace lfs::rendering
