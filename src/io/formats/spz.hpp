/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/exporter.hpp"
#include <expected>

namespace lfs::io {

    using lfs::core::SplatData;

    // Load SPZ (Niantic compressed gaussian splat format)
    std::expected<SplatData, std::string> load_spz(const std::filesystem::path& filepath);

} // namespace lfs::io
