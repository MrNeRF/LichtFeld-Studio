/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "io/exporter.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace lfs::io {

    using lfs::core::SplatData;

    struct RadDecodedChunk {
        std::uint64_t base = 0;
        std::uint64_t count = 0;
        int max_sh_degree = 0;
        std::uint32_t sh_coeffs_rest = 0;
        bool lod_opacity_encoded = false;
        std::vector<float> means;
        std::vector<float> opacity_raw;
        std::vector<float> sh0_raw;
        std::vector<float> scaling_raw;
        std::vector<float> rotation_raw;
        std::vector<float> shN_canonical;
        std::vector<std::uint16_t> child_count;
        std::vector<std::uint32_t> child_start;
    };

    // Load RAD (Random Access Dynamic) format - chunked hierarchical Gaussian splat format
    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath);
    std::expected<RadDecodedChunk, std::string> load_rad_chunk(
        const std::filesystem::path& filepath,
        const lfs::core::SplatLodTree::ChunkFileRange& range,
        int max_sh_degree,
        bool lod_opacity_encoded);

} // namespace lfs::io
