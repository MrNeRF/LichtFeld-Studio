/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_codecs.hpp"

// Like lfs_video, this consumer is compiled into a static library and imports
// the writers from a shared library with hidden default visibility.
bool exercise_image_codec_exports() {
    namespace codec = lfs::core::image_codecs;
    std::string error;
    const std::filesystem::path path;
    const bool original_jpeg = !codec::write_jpeg(path, nullptr, 1, 1, 3, 90, std::nullopt, error);
    const bool full_chroma_jpeg = !codec::write_jpeg(path, nullptr, 1, 1, 3, 95, std::nullopt, error, true);
    const bool png = !codec::write_png(path, nullptr, 1, 1, 3, 8, 6, std::nullopt, error);
    const bool image = !codec::write_image_u8("invalid.png", nullptr, 1, 1, 3, 95, std::nullopt, error);
    return original_jpeg && full_chroma_jpeg && png && image && !error.empty();
}
