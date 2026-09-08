/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "io/exporter.hpp"

namespace lfs::io {
    Result<SplatData> load_ssog(
        const std::filesystem::path&, const SsogLoadOptions& = {});
    bool is_ssog_path(const std::filesystem::path&);
    // Structural validation, including unit metadata ranges; does not decode textures or use CUDA.
    Result<void> validate_ssog(const std::filesystem::path&);
} // namespace lfs::io
