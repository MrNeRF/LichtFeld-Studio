// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/file_frame_sink.hpp"
#include "core/image_codecs.hpp"
#include "core/path_utils.hpp"
#include "media/video_frame_extractor.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace lfs::media {
    namespace {
        SinkResult sinkError(ErrorCode code, std::string message) {
            return SinkResult::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace

    FileFrameSink::FileFrameSink(FileFrameSinkOptions options) : options_(std::move(options)) {}
    SinkResult FileFrameSink::begin(const SinkSession&) {
        if (active_)
            return sinkError(ErrorCode::FailedPrecondition, "File sink already active");
        if (options_.output_directory.empty() ||
            (options_.format != FrameFileFormat::PNG && options_.format != FrameFileFormat::JPEG))
            return sinkError(ErrorCode::InvalidArgument, "Invalid file sink directory or format");
        if (options_.format == FrameFileFormat::JPEG)
            options_.jpeg_quality = options_.jpeg_quality == 0 ? 90 : std::clamp(options_.jpeg_quality, 1, 100);
        std::error_code directory_error;
        std::filesystem::create_directories(options_.output_directory, directory_error);
        if (directory_error) {
            return SinkResult::failure(make_error({.code = directory_error == std::errc::permission_denied ? ErrorCode::PermissionDenied : ErrorCode::Unavailable,
                                                   .domain = ErrorDomain::IO,
                                                   .detail = "Create output directory failed: " + core::path_to_utf8(options_.output_directory),
                                                   .detection = LFS_SOURCE_SITE_CURRENT(),
                                                   .native = NativeError{ErrorDomain::IO, directory_error.value(), directory_error.category().name()}}));
        }
        filenames_.clear();
        active_ = true;
        return {};
    }
    SinkResult FileFrameSink::write(const FrameView& frame) {
        if (!active_)
            return sinkError(ErrorCode::FailedPrecondition, "File sink is not active");
        const auto required = frame.requiredBytes();
        if (!required)
            return SinkResult::failure(required.error());
        const auto row_bytes = static_cast<std::size_t>(frame.layout.width) * 3;
        const auto height = static_cast<std::size_t>(frame.layout.height);
        if (row_bytes > std::numeric_limits<std::size_t>::max() / height)
            return sinkError(ErrorCode::InvalidArgument, "File sink packed buffer size overflows");
        if (frame.info.legacy_source_frame < 1)
            return sinkError(ErrorCode::InvalidArgument, "File sink requires a positive source frame number");
        const auto extension = options_.format == FrameFileFormat::PNG ? ".png" : ".jpg";
        const auto filename = options_.output_directory /
                              (io::formatFrameFilenameStem(options_.filename_pattern, frame.info.legacy_source_frame) + extension);
        if (!filenames_.insert(filename).second)
            return sinkError(ErrorCode::AlreadyExists, "Duplicate file sink filename");
        // Legacy extraction is already packed: no additional pixel copy.
        std::vector<std::uint8_t> packed;
        const auto* pixels = frame.pixels.data();
        if (frame.layout.row_stride != row_bytes) {
            packed.resize(row_bytes * height);
            for (std::size_t row = 0; row < height; ++row)
                std::memcpy(packed.data() + row * row_bytes,
                            pixels + row * frame.layout.row_stride, row_bytes);
            pixels = packed.data();
        }
        std::string error;
        const bool success = options_.format == FrameFileFormat::JPEG
                                 ? core::image_codecs::write_jpeg(filename, pixels, frame.layout.width, frame.layout.height, 3,
                                                                  options_.jpeg_quality, std::nullopt, error, options_.jpeg_quality > 90)
                                 : core::image_codecs::write_png(filename, pixels, frame.layout.width, frame.layout.height, 3, 8, 6, std::nullopt, error);
        return success ? SinkResult{} : sinkError(ErrorCode::Unavailable, std::move(error));
    }
    SinkResult FileFrameSink::complete(const SinkSummary&) {
        if (!active_)
            return sinkError(ErrorCode::FailedPrecondition, "File sink is not active");
        active_ = false;
        return {};
    }
    void FileFrameSink::abort(const SinkSummary&) noexcept { active_ = false; }
} // namespace lfs::media
