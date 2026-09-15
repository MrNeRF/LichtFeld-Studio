/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <filesystem>

namespace lfs::core {
    class SplatData;
}

namespace lfs::vis {

    // Same Vulkan-external storage check VkSplat uses before it will bind a model
    // without the input-copy fallback (means/sh0/rotation/scaling/opacity/shN, and
    // shN bounds when the rest buffer is q16).
    [[nodiscard]] LFS_VIS_API bool viewerSplatTensorsRendererReady(const lfs::core::SplatData& model);

    // Encode rest SH into exportable q16 Vulkan-external storage when the current
    // buffer cannot be bound. Reuses SplatData::apply_shN_value_quant. Does not
    // throw: hydration must still complete if a single model cannot encode.
    void ensureViewerSplatShNExportable(const std::filesystem::path& path, lfs::core::SplatData& model);

    void quantizeViewerLoadedPlyShN(const std::filesystem::path& path, lfs::core::SplatData& model);

} // namespace lfs::vis
