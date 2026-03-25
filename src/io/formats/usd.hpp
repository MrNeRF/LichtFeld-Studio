/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <expected>
#include <filesystem>

namespace lfs::io {

    using lfs::core::SplatData;

    // Load OpenUSD Gaussian ParticleField data (.usd/.usda/.usdc/.usdz)
    std::expected<SplatData, std::string> load_usd(const std::filesystem::path& filepath);
    std::expected<void, std::string> validate_usd(const std::filesystem::path& filepath);

} // namespace lfs::io
