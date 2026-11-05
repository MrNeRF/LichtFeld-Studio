/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/mesh2splat.hpp"
#include "core/splat_data.hpp"

#include <cstdint>
#include <expected>
#include <memory>

namespace lfs::core {
    struct MeshData;
}

namespace lfs::rendering {

    // `min_output_entries` reserves space for that many splats when more than the estimate from the resolution will come.
    [[nodiscard]] std::expected<std::unique_ptr<core::SplatData>, std::string>
    mesh_to_splat(const core::MeshData& mesh,
                  const core::Mesh2SplatOptions& options = {},
                  core::Mesh2SplatProgressCallback progress = nullptr,
                  uint64_t min_output_entries = 0);

} // namespace lfs::rendering
