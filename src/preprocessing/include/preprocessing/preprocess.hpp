/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/parameters.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace lfs::preprocessing {

    [[nodiscard]] lfs::Result<std::filesystem::path> ensure_lpips_weights(bool allow_download);

    using PreprocessProgressCallback = std::function<void(
        std::size_t done, std::size_t total, std::string_view filename)>;

    struct PreprocessRunResult {
        bool ok = true;
        std::string error;
        std::size_t processed = 0;
        std::size_t skipped = 0;
    };

    int run_preprocess(const lfs::core::param::PreprocessParameters& params);

    PreprocessRunResult run_preprocess_ex(
        const lfs::core::param::PreprocessParameters& params,
        const PreprocessProgressCallback& progress = {});

    std::filesystem::path ensure_sam2_weights(bool no_download = false);

    // Downloads on first use, like the other cached models.
    std::filesystem::path ensure_romav1_weights(bool no_download = false);

    // Where those weights are cached, whether or not they are there yet.
    std::filesystem::path romav1_weights_cache_path();

} // namespace lfs::preprocessing
