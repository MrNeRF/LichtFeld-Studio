/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

#include <functional>
#include <string>
#include <string_view>

namespace lfs::preprocessing {

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

} // namespace lfs::preprocessing
