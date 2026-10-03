/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "image_codecs.hpp"

namespace lfs::core::image_codecs {
    bool probe_exr(const std::filesystem::path& path, Probe& result, std::string& error);
    bool decode_exr(const std::filesystem::path& path, Image& result, std::string& error);
} // namespace lfs::core::image_codecs
