/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/viewer_splat_quantize.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/sh_value_quant.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/sh_value_storage.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace lfs::vis {
    namespace {

        [[nodiscard]] bool isPlyPath(const std::filesystem::path& path) {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return extension == ".ply";
        }

        [[nodiscard]] bool rendererReady(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid() || tensor.numel() == 0) {
                return true;
            }
            return tensor.is_external_storage() &&
                   tensor.external_storage_kind() == "vulkan_external_buffer";
        }

        [[nodiscard]] bool shNStorageRendererReady(const lfs::core::SplatData& model) {
            return rendererReady(model.shN_raw()) &&
                   (!model.shN_value_quantized() || rendererReady(model.shN_value_bounds()));
        }

        [[nodiscard]] bool baseAttrsRendererReady(const lfs::core::SplatData& model) {
            return rendererReady(model.means_raw()) &&
                   rendererReady(model.sh0_raw()) &&
                   rendererReady(model.scaling_raw()) &&
                   rendererReady(model.rotation_raw()) &&
                   rendererReady(model.opacity_raw());
        }

        [[nodiscard]] std::size_t q16ShNCodesBytes(const lfs::core::SplatData& model) {
            const auto n = static_cast<std::size_t>(model.size());
            const auto cap = model.means_raw().is_valid()
                                 ? std::max(model.means_raw().capacity(), n)
                                 : n;
            const auto rest = static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
            return lfs::core::sh_value_quant::sh_value_u16_count(cap, rest) * sizeof(std::uint16_t);
        }

        [[noreturn]] void throwUnrenderable(const std::filesystem::path& path,
                                            const lfs::core::SplatData& model) {
            std::string message = std::format(
                "Viewer cannot bind splat tensors for '{}' in Vulkan-external storage",
                lfs::core::path_to_utf8(path));
            if (!lfs::core::sh_value_quant::enabled()) {
                message += "; SH value quantization is disabled";
            }
            if (const auto violation = lfs::core::shareable_allocation_violation(
                    q16ShNCodesBytes(model), "SplatData.shN")) {
                message += "; ";
                message += *violation;
            }
            throw std::runtime_error(std::move(message));
        }

        void encodeViewerSplatShNIfNeeded(const std::filesystem::path& path,
                                          lfs::core::SplatData& model) {
            if (!model.has_tensor_allocator()) {
                LOG_WARN("Viewer SH q16 skipped for '{}': Vulkan-external storage is unavailable or degraded",
                         lfs::core::path_to_utf8(path));
                return;
            }

            const bool shN_only_not_ready =
                baseAttrsRendererReady(model) && !shNStorageRendererReady(model);
            const bool should_encode =
                (isPlyPath(path) || shN_only_not_ready) &&
                !model.shN_value_quantized() &&
                model.shN_raw().is_valid() &&
                model.shN_raw().numel() > 0;

            if (!should_encode) {
                return;
            }

            const std::size_t shN_before_bytes = model.shN_raw().bytes();
            const bool converted = lfs::training::sh_value::apply_shN_value_quant(model);
            if (converted) {
                const std::size_t shN_after_bytes =
                    model.shN_raw().bytes() + model.shN_value_bounds().bytes();
                LOG_INFO(
                    "Viewer SH q16: path='{}' gaussians={} before_bytes={} after_bytes={} saved_mib={:.3f}",
                    lfs::core::path_to_utf8(path), model.size(), shN_before_bytes, shN_after_bytes,
                    (static_cast<double>(shN_before_bytes) - static_cast<double>(shN_after_bytes)) /
                        (1024.0 * 1024.0));
            }
        }

    } // namespace

    bool viewerSplatTensorsRendererReady(const lfs::core::SplatData& model) {
        return baseAttrsRendererReady(model) && shNStorageRendererReady(model);
    }

    void ensureViewerSplatShNExportable(const std::filesystem::path& path,
                                        lfs::core::SplatData& model) {
        try {
            encodeViewerSplatShNIfNeeded(path, model);
        } catch (const std::exception& error) {
            LOG_WARN("Viewer SH q16 failed for '{}': {}",
                     lfs::core::path_to_utf8(path), error.what());
        }
    }

    void quantizeViewerLoadedPlyShN(const std::filesystem::path& path,
                                    lfs::core::SplatData& model) {
        encodeViewerSplatShNIfNeeded(path, model);

        if (baseAttrsRendererReady(model) && !shNStorageRendererReady(model)) {
            throwUnrenderable(path, model);
        }
    }

} // namespace lfs::vis
