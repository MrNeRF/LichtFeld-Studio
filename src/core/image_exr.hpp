/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "image_codecs.hpp"
#include <array>
#include <functional>
#include <span>
#include <string_view>
#include <utility>

namespace lfs::core::image_codecs {
    LFS_IMAGE_CODECS_API bool probe_exr(const std::filesystem::path& path, Probe& result, std::string& error);
    // Zero selects automatic scheduling; one keeps decoding sequential. The
    // internal override permits reproducible codec tests and measurements.
    LFS_IMAGE_CODECS_API bool decode_exr(const std::filesystem::path& path, Image& result, std::string& error, unsigned max_workers = 0);
    enum class ExrSampleType { Half,
                               Float };
    enum class ExrCompression { None,
                                Zip };
    struct ExrEncodeView {
        int width = 0, height = 0, channels = 0;
        size_t row_stride = 0;
        std::span<const std::uint8_t> pixels;
    };
    using ExrWriteCallback = int64_t (*)(void*, const void*, uint64_t, uint64_t) noexcept;
    // The caller owns storage and commit policy. The codec never opens a path.
    struct ExrWriteStream {
        std::string_view name;
        void* user = nullptr;
        ExrWriteCallback write = nullptr;
    };
    struct ExrEncodeOptions {
        ExrSampleType precision = ExrSampleType::Half;
        ExrCompression compression = ExrCompression::Zip;
        bool premultiply_alpha = false;
        std::array<float, 8> chromaticities{};
        std::span<const std::pair<std::string, std::string>> text;
        std::function<bool()> cancelled;
    };
    [[nodiscard]] LFS_IMAGE_CODECS_API Result<void> encode_exr(const ExrEncodeView&, const ExrWriteStream&, const ExrEncodeOptions&);
} // namespace lfs::core::image_codecs
