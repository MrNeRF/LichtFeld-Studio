/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cmath>
#include <glm/glm.hpp>

namespace lfs::rendering {
#define LFS_LOSS_INLINE          inline
#define LFS_LOSS_VEC3            glm::vec3
#define LFS_LOSS_ISFINITE(value) std::isfinite(value)
    using glm::abs;
    using glm::clamp;
    using glm::exp;
    using glm::max;
    using glm::mix;
#include "../shader/src/slang/gt_loss_heatmap.inc"
#undef LFS_LOSS_INLINE
#undef LFS_LOSS_VEC3
#undef LFS_LOSS_ISFINITE
} // namespace lfs::rendering
