/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "image_codecs.hpp"

namespace lfs::core::image_codecs {
    bool probe_exr(const std::filesystem::path& path, Probe& result, std::string& error);
    // Zero selects automatic scheduling; one keeps decoding sequential. The
    // internal override permits reproducible codec tests and measurements.
    bool decode_exr(const std::filesystem::path& path, Image& result, std::string& error, unsigned max_workers = 0);
} // namespace lfs::core::image_codecs
