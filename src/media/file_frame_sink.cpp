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
            options_.output_directory.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos ||
            options_.filename_pattern.find('\0') != std::string::npos ||
            (options_.format != FrameFileFormat::PNG && options_.format != FrameFileFormat::JPEG && options_.format != FrameFileFormat::EXR))
            return sinkError(ErrorCode::InvalidArgument, "Invalid file sink directory or format");
        if (options_.format == FrameFileFormat::EXR &&
            ((options_.exr.precision != ExrPrecision::Half && options_.exr.precision != ExrPrecision::Float) ||
             (options_.exr.compression != ExrCompression::None && options_.exr.compression != ExrCompression::ZIP) ||
             options_.exr.provenance.find('\0') != std::string::npos || options_.exr.provenance.size() > 1024 * 1024))
            return sinkError(ErrorCode::InvalidArgument, "Invalid EXR sink options");
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
        if (options_.format != FrameFileFormat::EXR && frame.layout.format != FramePixelFormat::RGB8)
            return sinkError(ErrorCode::Unsupported, "PNG/JPEG file sink requires RGB8; use EXR for float samples");
        const auto row_bytes = static_cast<std::size_t>(frame.layout.width) * pixelBytes(frame.layout.format);
        const auto height = static_cast<std::size_t>(frame.layout.height);
        if (row_bytes > std::numeric_limits<std::size_t>::max() / height)
            return sinkError(ErrorCode::InvalidArgument, "File sink packed buffer size overflows");
        if (frame.info.legacy_source_frame < 1)
            return sinkError(ErrorCode::InvalidArgument, "File sink requires a positive source frame number");
        const auto extension = options_.format == FrameFileFormat::PNG ? ".png" : options_.format == FrameFileFormat::EXR ? ".exr"
                                                                                                                          : ".jpg";
        const auto filename = options_.output_directory /
                              (io::formatFrameFilenameStem(options_.filename_pattern, frame.info.legacy_source_frame) + extension);
        if (!filenames_.insert(filename).second)
            return sinkError(ErrorCode::AlreadyExists, "Duplicate file sink filename");
        if (options_.format == FrameFileFormat::EXR)
            return ImageOutput::writeExr(filename, frame, options_.exr);
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
