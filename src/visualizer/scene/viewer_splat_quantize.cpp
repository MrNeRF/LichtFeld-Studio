/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/viewer_splat_quantize.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "io/loader.hpp"
#include "lfs/training/sh_value_storage.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace lfs::vis {
    namespace {

        [[nodiscard]] bool isPlyPath(const std::filesystem::path& path) {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return extension == ".ply";
        }

    } // namespace

    void quantizeViewerLoadedPlyShN(const std::filesystem::path& path,
                                    lfs::core::SplatData& model) {
        if (!isPlyPath(path)) {
            return;
        }
        if (model.shN_value_quantized() ||
            !model.shN_raw().is_valid() || model.shN_raw().numel() == 0) {
            return;
        }
        if (!model.has_tensor_allocator() || !lfs::io::splatTensorsRendererReady(model)) {
            LOG_WARN("Viewer PLY SH q16 skipped for '{}': Vulkan-external storage is unavailable or degraded",
                     lfs::core::path_to_utf8(path));
            return;
        }

        const std::size_t shN_before_bytes = model.shN_raw().bytes();
        bool converted = false;
        try {
            converted = lfs::training::sh_value::apply_shN_value_quant(model);
        } catch (const std::exception& e) {
            LOG_WARN("Viewer PLY SH q16 failed for '{}'; retaining fp32 SH: {}",
                     lfs::core::path_to_utf8(path), e.what());
            return;
        }
        if (!converted) {
            return;
        }
        if (!model.shN_value_quantized() || !lfs::io::splatTensorsRendererReady(model)) {
            throw std::runtime_error(
                "Viewer PLY SH q16 produced a non-bindable codes/bounds pair");
        }

        const std::size_t shN_after_bytes =
            model.shN_raw().bytes() + model.shN_value_bounds().bytes();
        LOG_INFO(
            "Viewer PLY SH q16: path='{}' gaussians={} before_bytes={} after_bytes={} saved_mib={:.3f}",
            lfs::core::path_to_utf8(path), model.size(), shN_before_bytes, shN_after_bytes,
            (static_cast<double>(shN_before_bytes) - static_cast<double>(shN_after_bytes)) /
                (1024.0 * 1024.0));
    }

} // namespace lfs::vis
