/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "flat_stage.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace lfs::io::usd_flat {

    lfs::Result<FlatStage> read_usdc(const std::filesystem::path& path);
    lfs::Result<FlatStage> read_usdc_bytes(const std::uint8_t* data, std::size_t size);
    lfs::Status write_usdc(const FlatStage& stage, const std::filesystem::path& path);

} // namespace lfs::io::usd_flat
