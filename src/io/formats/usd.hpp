/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/splat_data.hpp"
#include <expected>
#include <filesystem>

namespace lfs::io {

    using lfs::core::SplatData;

    struct UsdProjectUnitLoad {
        SplatData data;
        double meters_per_unit = 1.0;
    };

    // Load OpenUSD Gaussian ParticleField data (.usd/.usda/.usdc/.usdz)
    std::expected<SplatData, std::string> load_usd(const std::filesystem::path& filepath);
    // Project import keeps authored stage units in the payload and returns the
    // stage's metres-per-unit metadata for PROJ.world_unit_scale.
    lfs::Result<UsdProjectUnitLoad>
    load_usd_project_units(const std::filesystem::path& filepath);
    std::expected<void, std::string> validate_usd(const std::filesystem::path& filepath);

} // namespace lfs::io
