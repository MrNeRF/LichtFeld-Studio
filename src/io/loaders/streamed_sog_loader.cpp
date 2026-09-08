/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "streamed_sog_loader.hpp"
#include "formats/streamed_sog.hpp"
#include <chrono>
namespace lfs::io {
    bool StreamedSogLoader::canLoad(const std::filesystem::path& path) const {
        return is_streamed_sog_path(path);
    }
    Result<LoadResult> StreamedSogLoader::load(const std::filesystem::path& path, const LoadOptions& options) {
        const auto start = std::chrono::steady_clock::now();
        if (!canLoad(path))
            return make_error(ErrorCode::PATH_NOT_FOUND, "Streamed SOG manifest does not exist", path);
        if (options.progress)
            options.progress(0, "Loading streamed SOG");
        std::shared_ptr<SplatData> data;
        if (options.validate_only) {
            if (auto result = validate_streamed_sog(path); !result)
                return make_error(ErrorCode::INVALID_HEADER, result.error(), path);
        } else {
            auto result = load_streamed_sog(path);
            if (!result)
                return make_error(ErrorCode::CORRUPTED_DATA, result.error(), path);
            data = std::make_shared<SplatData>(std::move(*result));
        }
        if (options.progress)
            options.progress(100, "Streamed SOG complete");
        LoadResult result;
        result.data = std::move(data);
        result.scene_center = core::Tensor::zeros({3}, core::Device::CPU);
        result.loader_used = name();
        result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        return result;
    }
} // namespace lfs::io
