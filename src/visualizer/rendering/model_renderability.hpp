/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/splat_data.hpp"

namespace lfs::vis {

    [[nodiscard]] inline bool hasRenderableGaussians(const lfs::core::SplatData* model) {
        return model && model->means_raw().is_valid() && model->size() > 0;
    }

} // namespace lfs::vis
