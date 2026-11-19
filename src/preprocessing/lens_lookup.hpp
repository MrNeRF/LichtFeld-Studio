/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "preprocessing/lens_priors.hpp"

#include <filesystem>
#include <optional>

namespace lfs::preprocessing {
    [[nodiscard]] std::optional<LensCamera> lookup_lens_source(const core::param::PreprocessParameters& params,
                                                               const std::filesystem::path& image_path);
} // namespace lfs::preprocessing
