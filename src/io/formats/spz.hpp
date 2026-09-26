/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/exporter.hpp"
#include <expected>

namespace lfs::io {

    using lfs::core::SplatData;

    // Load SPZ (Niantic compressed gaussian splat format), bare or embedded in a .glb
    std::expected<SplatData, std::string> load_spz(const std::filesystem::path& filepath);

    // True for a .glb whose primitive carries KHR_gaussian_splatting_compression_spz_2
    bool is_spz_glb(const std::filesystem::path& filepath);

} // namespace lfs::io
