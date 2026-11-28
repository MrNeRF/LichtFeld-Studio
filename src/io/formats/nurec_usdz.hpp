/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::io {

    using lfs::core::SplatData;

    // Cheap archive sniff used to route .usdz files to the correct loader.
    std::expected<bool, std::string> is_nurec_usdz(const std::filesystem::path& filepath);

    // Load NuRec payloads packaged inside USDZ archives.
    std::expected<SplatData, std::string> load_nurec_usdz(const std::filesystem::path& filepath);
    std::expected<void, std::string> validate_nurec_usdz(const std::filesystem::path& filepath);

    // Test hook: the most bytes one zlib or libarchive call gets when a payload is compressed, decompressed or written.
    // By default that is each library's own limit (4 GiB for zlib, 2 GiB for libarchive's writes), and 0 restores it.
    // A payload past them goes in pieces; a test lowers the piece to cover those without gigabytes.
    void set_nurec_piece_bytes_for_testing(std::size_t bytes) noexcept;

} // namespace lfs::io
