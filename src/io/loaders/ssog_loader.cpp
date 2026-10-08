/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "ssog_loader.hpp"
#include "formats/ssog.hpp"
#include "tile_source_load.hpp"
#include <chrono>
#include <format>
namespace lfs::io {
    bool SsogLoader::canLoad(const std::filesystem::path& path) const {
        return is_ssog_path(path);
    }
    Result<LoadResult> SsogLoader::load(const std::filesystem::path& path, const LoadOptions& options) {
        const auto start = std::chrono::steady_clock::now();
        if (options.progress)
            options.progress(0, "Loading SSOG");
        std::shared_ptr<SplatData> data;
        std::shared_ptr<SplatTileSource> tile_source;
        std::optional<std::vector<uint8_t>> license_bytes;
        if (options.validate_only) {
            if (auto result = validate_ssog(path); !result)
                return std::unexpected(result.error());
        } else {
            auto source = open_ssog_tiles(path, &license_bytes);
            if (!source)
                return tile_load_error(source.error(), path);
            const auto plan = plan_tile_load(**source, "SSOG");
            if (plan.flat) {
                auto result = load_ssog(path, {}, &license_bytes);
                if (!result)
                    return std::unexpected(result.error());
                data = std::make_shared<SplatData>(std::move(*result));
            } else {
                // Too large to hold whole: show the coarsest levels and stream the rest.
                if (options.progress)
                    options.progress(10, std::format("Loading {} SSOG tiles", plan.tiles.size()));
                auto merged = load_tiles_merged(**source, plan.tiles, options);
                if (!merged)
                    return std::unexpected(merged.error());
                data = std::shared_ptr<SplatData>(std::move(*merged));
                tile_source = std::move(*source);
            }
        }
        if (options.progress)
            options.progress(100, "SSOG complete");
        LoadResult result;
        result.data = std::move(data);
        result.scene_center = core::Tensor::zeros({3}, core::Device::CPU);
        result.loader_used = name();
        result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        result.license_bytes = std::move(license_bytes);
        result.tile_source = std::move(tile_source);
        return result;
    }
} // namespace lfs::io
