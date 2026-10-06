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

    // Same renderer-visible storage check the scene renderer uses before binding a model
    // without the input-copy fallback (means/sh0/rotation/scaling/opacity/shN, and
    // shN bounds when the rest buffer is q16).
    [[nodiscard]] LFS_VIS_API bool viewerSplatTensorsRendererReady(const lfs::core::SplatData& model);

    // Encode rest SH into renderer-visible q16 storage when the current
    // buffer cannot be bound. Reuses SplatData::apply_shN_value_quant. Does not
    // throw: hydration must still complete if a single model cannot encode.
    void ensureViewerSplatShNExportable(const std::filesystem::path& path, lfs::core::SplatData& model);

    void quantizeViewerLoadedPlyShN(const std::filesystem::path& path, lfs::core::SplatData& model);

} // namespace lfs::vis
