/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "ssog_loader.hpp"
#include "formats/ssog.hpp"
#include <chrono>
namespace lfs::io {
    bool SsogLoader::canLoad(const std::filesystem::path& path) const {
        return is_ssog_path(path);
    }
    Result<LoadResult> SsogLoader::load(const std::filesystem::path& path, const LoadOptions& options) {
        const auto start = std::chrono::steady_clock::now();
        if (options.progress)
            options.progress(0, "Loading SSOG");
        std::shared_ptr<SplatData> data;
        if (options.validate_only) {
            if (auto result = validate_ssog(path); !result)
                return std::unexpected(result.error());
        } else {
            auto result = load_ssog(path);
            if (!result)
                return std::unexpected(result.error());
            data = std::make_shared<SplatData>(std::move(*result));
        }
        if (options.progress)
            options.progress(100, "SSOG complete");
        LoadResult result;
        result.data = std::move(data);
        result.scene_center = core::Tensor::zeros({3}, core::Device::CPU);
        result.loader_used = name();
        result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        return result;
    }
} // namespace lfs::io
