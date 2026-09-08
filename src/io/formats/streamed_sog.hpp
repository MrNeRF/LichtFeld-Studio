/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "io/exporter.hpp"

namespace lfs::io {
    std::expected<SplatData, std::string> load_streamed_sog(
        const std::filesystem::path&, const StreamedSogLoadOptions& = {});
    bool is_streamed_sog_path(const std::filesystem::path&);
    // Structural validation, including unit metadata ranges; does not decode textures or use CUDA.
    std::expected<void, std::string> validate_streamed_sog(const std::filesystem::path&);
} // namespace lfs::io
