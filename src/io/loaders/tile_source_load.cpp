/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tile_source_load.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <tbb/parallel_for.h>
#include <unordered_map>

namespace lfs::io {

    namespace {
        // Full detail loads flat (editable, no streaming) when its loading peak fits free
        // VRAM and it stays below this many splats.
        constexpr std::uint64_t kFlatMaxSplats = 8'000'000;

        std::uint64_t splat_bytes(const std::uint64_t splats, const int sh_degree) {
            const int rest = (sh_degree + 1) * (sh_degree + 1) - 1;
            return splats * (14 + 3 * rest) * sizeof(float);
        }
    } // namespace

    TileLoadPlan plan_tile_load(const SplatTileSource& source, const std::string_view label) {
        const auto everything = [](std::uint32_t) { return true; };
        auto full = select_splat_tiles(source, {.sse_per_error = 1.0f, .max_sse = 0.0f}, everything);
        // Loading flat holds every tile at its own SH degree and then their merged copy,
        // which pads every splat to the highest degree among them.
        std::uint64_t tile_bytes = 0;
        int max_sh_degree = 0;
        for (const auto index : full.render) {
            const auto& tile = source.tiles()[index];
            tile_bytes += splat_bytes(tile.splat_count, tile.sh_degree);
            max_sh_degree = std::max(max_sh_degree, tile.sh_degree);
        }
        TileLoadPlan plan;
        plan.full_splats = full.render_splats;
        plan.full_bytes = tile_bytes + splat_bytes(full.render_splats, max_sh_degree);
        const auto memory = core::gpu_backend_memory_info(core::default_gpu_backend());
        plan.flat = plan.full_splats <= kFlatMaxSplats && plan.full_bytes <= memory.free_bytes;
        // Streaming starts from the coarsest content; the viewer refines from there.
        plan.tiles = plan.flat ? std::move(full.render)
                               : select_splat_tiles(source, {.max_sse = std::numeric_limits<float>::infinity()},
                                                    everything)
                                     .render;
        LOG_INFO("{}: {} ({} full-detail splats, {:.1f} of {:.1f} GiB free VRAM)", label,
                 plan.flat ? "loading full detail" : "streaming", plan.full_splats,
                 static_cast<double>(plan.full_bytes) / (1 << 30), static_cast<double>(memory.free_bytes) / (1 << 30));
        return plan;
    }

    std::unexpected<Error> tile_load_error(const lfs::Error& error, const std::filesystem::path& path) {
        LOG_ERROR("{}", lfs::format_for_developer(error));
        return make_error(ErrorCode::CORRUPTED_DATA, std::string(error.detail()), path);
    }

    Result<std::unique_ptr<SplatData>> load_tiles_merged(const SplatTileSource& source,
                                                         const std::vector<std::uint32_t>& tiles,
                                                         const LoadOptions& options) {
        std::vector<std::optional<SplatData>> loaded(tiles.size());
        std::vector<std::optional<lfs::Error>> errors(tiles.size());
        tbb::parallel_for(std::size_t{0}, tiles.size(), [&](const std::size_t i) {
            if (is_load_cancel_requested(options))
                return;
            if (auto data = load_splat_tile_gpu(source, tiles[i]))
                loaded[i] = std::move(*data);
            else
                errors[i] = data.error();
        });
        throw_if_load_cancel_requested(options);
        for (const auto& error : errors)
            if (error)
                return tile_load_error(*error);
        std::unordered_map<std::uint32_t, const SplatData*> by_tile;
        for (std::size_t i = 0; i < tiles.size(); ++i)
            by_tile.emplace(tiles[i], loaded[i] ? &*loaded[i] : nullptr);
        auto merged = merge_splat_tiles(source, tiles, [&](const std::uint32_t tile) { return by_tile.at(tile); });
        if (!merged)
            return make_error(ErrorCode::EMPTY_DATASET, "The tile source has no splat content");
        return merged;
    }

} // namespace lfs::io
