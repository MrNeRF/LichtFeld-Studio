/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <array>
#include <optional>

namespace lfs::core {
    class Camera;

    // An empty rotation leaves decoded camera-space normals in their original frame.
    LFS_CORE_API std::optional<std::array<float, 9>> normal_prior_rotation(
        const Camera& camera, bool world_space, const std::array<float, 9>& prior_world_rotation);
} // namespace lfs::core
