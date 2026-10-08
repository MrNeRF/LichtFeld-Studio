/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tiles3d_loader.hpp"
#include "core/splat_data.hpp"
#include "io/splat_tile_source.hpp"
#include "tile_source_load.hpp"
#include <chrono>
#include <format>

namespace lfs::io {

    bool Tiles3dLoader::canLoad(const std::filesystem::path& path) const {
        return is_tiles3d_path(path);
    }

    Result<LoadResult> Tiles3dLoader::load(const std::filesystem::path& path, const LoadOptions& options) {
        const auto start = std::chrono::steady_clock::now();
        if (options.progress)
            options.progress(0.0f, "Reading 3D Tiles tileset");
        auto source = open_tiles3d(path);
        if (!source)
            return tile_load_error(source.error(), path);

        LoadResult result;
        result.scene_center = core::Tensor::zeros({3}, core::Device::CPU);
        result.loader_used = name();
        result.is_tileset = true;
        if (options.validate_only)
            return result;

        const auto plan = plan_tile_load(**source, "3D Tiles");
        if (options.progress)
            options.progress(10.0f, std::format("Loading {} tiles", plan.tiles.size()));
        auto merged = load_tiles_merged(**source, plan.tiles, options);
        if (!merged)
            return std::unexpected(merged.error());

        result.data = std::shared_ptr<SplatData>(std::move(*merged));
        if (!plan.flat)
            result.tile_source = std::move(*source);
        result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        if (options.progress)
            options.progress(100.0f, "3D Tiles loaded");
        return result;
    }
} // namespace lfs::io
