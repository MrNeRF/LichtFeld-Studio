/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_avif.hpp"
#include "core/image_codecs.hpp"
#include "core/image_exr.hpp"
#include "core/image_jxl.hpp"

// Like lfs_video, this consumer is compiled into a static library and imports
// the writers from a shared library with hidden default visibility.
bool exercise_image_codec_exports() {
    namespace codec = lfs::core::image_codecs;
    std::string error;
    const std::filesystem::path path;
    std::vector<uint8_t> encoded;
    const auto memory = codec::encode_jpeg_memory({}, 1, 1, 3, 3, encoded);
    const auto png_memory = codec::encode_png_memory({}, 1, 1, 3, 8, 3, encoded);
    const auto tiff_memory = codec::encode_tiff_memory({}, 1, 1, 3, 8, 3, encoded);
    const auto exr_memory = codec::encode_exr_memory({}, encoded);
    const auto jxl_memory = codec::encode_jxl_memory({}, 1, 1, 3, codec::SampleType::UInt8, 3, encoded);
    const auto jxl_file = codec::write_jxl(path, {}, 1, 1, 3, codec::SampleType::UInt8, 3);
    const auto avif_memory = codec::encode_avif_memory({}, 1, 1, 3, 3, encoded);
    const auto avif_file = codec::write_avif(path, {}, 1, 1, 3, 3);
    codec::Image jxl_image;
    const bool jxl = !codec::decode_jxl(std::span<const uint8_t>{}, jxl_image, error);
    const bool original_jpeg = !codec::write_jpeg(path, nullptr, 1, 1, 3, 90, std::nullopt, error);
    const bool full_chroma_jpeg = !codec::write_jpeg(path, nullptr, 1, 1, 3, 95, std::nullopt, error, true);
    const bool png = !codec::write_png(path, nullptr, 1, 1, 3, 8, 6, std::nullopt, error);
    const bool image = !codec::write_image_u8("invalid.png", nullptr, 1, 1, 3, 95, std::nullopt, error);
    const lfs::core::AtomicFileOptions options{.overwrite = false, .durable = false, .create_directories = false};
    const auto atomic = codec::write_image_u8(path, nullptr, 1, 1, 3, 95, std::nullopt, options);
    return !avif_memory && !avif_file && codec::avif_encoder_available() && !codec::avif_encoder_version().empty() && jxl && !jxl_memory && !jxl_file && codec::jxl_encoder_version() > 0 && !exr_memory && !memory && !png_memory && !tiff_memory && original_jpeg && full_chroma_jpeg && png && image && !atomic && !error.empty();
}
