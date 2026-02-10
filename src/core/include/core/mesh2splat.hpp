/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/mesh_data.hpp"
#include "core/splat_data.hpp"

#include <expected>
#include <functional>
#include <memory>
#include <string>

#include <glm/glm.hpp>

namespace lfs::core {

    struct Mesh2SplatOptions {
        int resolution_target = 1024;
        float sigma = 0.65f;
        glm::vec3 light_dir{0.0f, 0.0f, 1.0f};
        float light_intensity = 0.7f;
        float ambient = 0.4f;
    };

    using Mesh2SplatProgressCallback = std::function<bool(float progress, const std::string& stage)>;

    [[nodiscard]] LFS_CORE_API std::expected<std::unique_ptr<SplatData>, std::string>
    mesh_to_splat(const MeshData& mesh,
                  const Mesh2SplatOptions& options = {},
                  Mesh2SplatProgressCallback progress = nullptr);

} // namespace lfs::core
