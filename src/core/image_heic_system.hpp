/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "image_codecs.hpp"

namespace lfs::core::image_codecs {
    // Internal adapter: libheif still owns container validation and color policy.
    // A null path selects the borrowed memory span; neither input outlives this call.
    // A null image requests metadata/capability probing without decoding pixels.
    bool read_heic_system(const std::filesystem::path* path, std::span<const std::uint8_t> bytes,
                          const Probe& expected, int source_bits, int orientation, bool icc_over_hdr, Image* image, std::string& error);
} // namespace lfs::core::image_codecs
