/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <glm/glm.hpp>

namespace lfs::vis {

    // [N, 3] points of any dtype and device through a world transform, as a CPU Float32 [N, 3].
    LFS_VIS_API lfs::core::Tensor transformPointsToWorld(const lfs::core::Tensor& means, const glm::mat4& world_transform);

    // [N, 3] colors, bytes scaled by 1/255 or floats as they are, as a CPU Float32 [N, 3].
    LFS_VIS_API lfs::core::Tensor pointColorsAsFloat(const lfs::core::Tensor& colors);

} // namespace lfs::vis
