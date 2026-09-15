/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "flat_stage.hpp"
#include <filesystem>

namespace lfs::io::usd_flat {

    lfs::Result<FlatStage> read_usdz(const std::filesystem::path& path);

} // namespace lfs::io::usd_flat
