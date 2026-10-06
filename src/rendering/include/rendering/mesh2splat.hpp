/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/mesh2splat.hpp"
#include "core/splat_data.hpp"

#include <expected>
#include <memory>

namespace lfs::core {
    struct MeshData;
}

namespace lfs::rendering {

    [[nodiscard]] std::expected<std::unique_ptr<core::SplatData>, std::string>
    mesh_to_splat(const core::MeshData& mesh,
                  const core::Mesh2SplatOptions& options = {},
                  core::Mesh2SplatProgressCallback progress = nullptr);

    // The converter as a tensor program on `backend`, which must support raster
    // draws. mesh_to_splat uses it in builds without the Vulkan converter.
    [[nodiscard]] lfs::Result<std::unique_ptr<core::SplatData>>
    mesh_to_splat_tensor(const core::MeshData& mesh,
                         const core::Mesh2SplatOptions& options,
                         core::GpuBackend backend,
                         core::Mesh2SplatProgressCallback progress = nullptr);

} // namespace lfs::rendering
