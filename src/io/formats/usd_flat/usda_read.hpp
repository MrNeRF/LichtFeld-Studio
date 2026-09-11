/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "flat_stage.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace lfs::io::usd_flat {

    lfs::Result<FlatStage> read_usda(const std::filesystem::path& path);
    lfs::Result<FlatStage> read_usda_bytes(const std::uint8_t* data, std::size_t size);

} // namespace lfs::io::usd_flat
