/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tiles3d_loader.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "io/splat_tile_source.hpp"
#include <algorithm>
#include <chrono>
#include <format>
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

        Result<std::unique_ptr<SplatData>> load_merged(const SplatTileSource& source,
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
                    return make_error(ErrorCode::CORRUPTED_DATA, lfs::format_for_developer(*error));
            std::unordered_map<std::uint32_t, const SplatData*> by_tile;
            for (std::size_t i = 0; i < tiles.size(); ++i)
                by_tile.emplace(tiles[i], loaded[i] ? &*loaded[i] : nullptr);
            auto merged = merge_splat_tiles(source, tiles, [&](const std::uint32_t tile) { return by_tile.at(tile); });
            if (!merged)
                return make_error(ErrorCode::EMPTY_DATASET, "3D Tiles tileset has no splat content");
            return merged;
        }
    } // namespace

    bool Tiles3dLoader::canLoad(const std::filesystem::path& path) const {
        return is_tiles3d_path(path);
    }

    Result<LoadResult> Tiles3dLoader::load(const std::filesystem::path& path, const LoadOptions& options) {
        const auto start = std::chrono::steady_clock::now();
        if (options.progress)
            options.progress(0.0f, "Reading 3D Tiles tileset");
        auto source = open_tiles3d(path);
        if (!source)
            return make_error(ErrorCode::CORRUPTED_DATA, lfs::format_for_developer(source.error()), path);

        LoadResult result;
        result.scene_center = core::Tensor::zeros({3}, core::Device::CPU);
        result.loader_used = name();
        result.is_tileset = true;
        if (options.validate_only)
            return result;

        const auto everything = [](std::uint32_t) { return true; };
        const auto full = select_splat_tiles(**source, {.sse_per_error = 1.0f, .max_sse = 0.0f}, everything);
        // Loading flat holds every tile at its own SH degree and then their merged copy,
        // which pads every splat to the highest degree among them.
        std::uint64_t tile_bytes = 0;
        int max_sh_degree = 0;
        for (const auto index : full.render) {
            const auto& tile = (*source)->tiles()[index];
            tile_bytes += splat_bytes(tile.splat_count, tile.sh_degree);
            max_sh_degree = std::max(max_sh_degree, tile.sh_degree);
        }
        const std::uint64_t full_bytes = tile_bytes + splat_bytes(full.render_splats, max_sh_degree);
        const auto memory = core::gpu_backend_memory_info(core::default_gpu_backend());
        const bool flat = full.render_splats <= kFlatMaxSplats && full_bytes <= memory.free_bytes;

        // Streaming starts from the coarsest content; the viewer refines from there.
        const auto& tiles = flat ? full.render
                                 : select_splat_tiles(**source, {.max_sse = std::numeric_limits<float>::infinity()},
                                                      everything)
                                       .render;
        LOG_INFO("3D Tiles: {} ({} full-detail splats, {:.1f} of {:.1f} GiB free VRAM)",
                 flat ? "loading full detail" : "streaming", full.render_splats,
                 static_cast<double>(full_bytes) / (1 << 30), static_cast<double>(memory.free_bytes) / (1 << 30));
        if (options.progress)
            options.progress(10.0f, std::format("Loading {} tiles", tiles.size()));
        auto merged = load_merged(**source, tiles, options);
        if (!merged)
            return std::unexpected(merged.error());

        result.data = std::shared_ptr<SplatData>(std::move(*merged));
        if (!flat)
            result.tile_source = std::move(*source);
        result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        if (options.progress)
            options.progress(100.0f, "3D Tiles loaded");
        return result;
    }
} // namespace lfs::io
