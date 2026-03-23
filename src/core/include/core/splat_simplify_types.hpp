/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <string>

namespace lfs::core {

    struct SplatSimplifyOptions {
        double ratio = 0.1;
        int knn_k = 16;
        double merge_cap = 0.5;
        float opacity_prune_threshold = 0.1f;
    };

    using SplatSimplifyProgressCallback = std::function<bool(float progress, const std::string& stage)>;

} // namespace lfs::core
