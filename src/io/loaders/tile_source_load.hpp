/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "io/loader_interface.hpp"
#include "io/splat_tile_source.hpp"
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace lfs::io {

    // How a hierarchical splat source opens: full detail flat (editable, no streaming)
    // when its loading peak fits free VRAM and it stays below a splat limit, otherwise
    // the coarsest cut, from which the viewer streams.
    struct TileLoadPlan {
        bool flat = false;
        std::vector<std::uint32_t> tiles; // tiles to load now
        std::uint64_t full_splats = 0;
        std::uint64_t full_bytes = 0; // flat loading peak
    };

    [[nodiscard]] TileLoadPlan plan_tile_load(const SplatTileSource& source, std::string_view label);

    // A tile source failure as a load error: the readable detail for the user, the full
    // developer report in the log.
    [[nodiscard]] std::unexpected<Error> tile_load_error(const lfs::Error& error, const std::filesystem::path& path = {});

    // Decodes the tiles in parallel and merges them into one model.
    [[nodiscard]] Result<std::unique_ptr<SplatData>> load_tiles_merged(const SplatTileSource& source,
                                                                       const std::vector<std::uint32_t>& tiles,
                                                                       const LoadOptions& options);

} // namespace lfs::io
