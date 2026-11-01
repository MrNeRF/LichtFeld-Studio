// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_ingest.hpp"
#include "source_metadata_internal.hpp"
#include "xmp_sidecar_io.hpp"
#include <limits>

namespace lfs::media {
    using detail::sidecarError;
    using detail::sidecarIoError;
    namespace {
        bool validPath(const std::filesystem::path& path) {
            return !path.empty() && path.has_filename() && path.native().find(std::filesystem::path::value_type{}) == std::filesystem::path::string_type::npos;
        }
    } // namespace
    Result<SourceSidecarMetadata> MediaIngest::sourceSidecarMetadata(const std::filesystem::path& image, const std::filesystem::path& sidecar,
                                                                     const SourceMetadataIndexLimits& limits, std::size_t max_bytes) {
        try {
            const auto pair = [&] { return " (image=" + core::path_to_utf8(image) + ", sidecar=" + core::path_to_utf8(sidecar) + ")"; };
            if (!validPath(image) || !validPath(sidecar))
                return sidecarError(ErrorCode::InvalidArgument, "Image and XMP sidecar paths must name files without null bytes" + pair());
            if (!max_bytes || max_bytes > core::image_codecs::max_xmp_packet_bytes)
                return sidecarError(ErrorCode::InvalidArgument, "XMP sidecar byte limit must be between 1 and 32 MiB (limit=" + std::to_string(max_bytes) + ")" + pair());
            std::error_code error;
            const auto image_path = core::native_fs::absolute(image, error);
            if (error)
                return sidecarIoError(error, "Resolve sidecar image path", image);
            const auto sidecar_path = core::native_fs::absolute(sidecar, error);
            if (error)
                return sidecarIoError(error, "Resolve XMP sidecar path", sidecar);
            auto extension = sidecar_path.extension().native();
            for (auto& ch : extension)
                if (ch >= 'A' && ch <= 'Z')
                    ch += 'a' - 'A';
            auto full_name = image_path.filename();
            full_name += ".xmp";
            auto stem_name = image_path.stem();
            stem_name += ".xmp";
            auto actual_name = sidecar_path.filename();
            actual_name.replace_extension(".xmp");
            if (image_path == sidecar_path ||
                extension != std::filesystem::path(".xmp").native() ||
                (actual_name != full_name && actual_name != stem_name))
                return sidecarError(ErrorCode::InvalidArgument, "XMP sidecar must share the image directory and use its full filename or stem plus .xmp" + pair());
            const bool same_directory = core::native_fs::equivalent(image_path.parent_path(), sidecar_path.parent_path(), error);
            if (error)
                return sidecarIoError(error, "Compare image and XMP sidecar directories", sidecar_path);
            if (!same_directory)
                return sidecarError(ErrorCode::InvalidArgument, "Image and XMP sidecar must share the same directory" + pair());
            const auto image_status = core::native_fs::symlink_status(image_path, error);
            if (error)
                return sidecarIoError(error, "Inspect sidecar image", image_path);
            if (!core::native_fs::is_regular_file(image_status))
                return sidecarError(ErrorCode::FailedPrecondition, "Sidecar image must be a regular non-symlink file (path=" + core::path_to_utf8(image_path) + ", type=" + std::to_string(static_cast<int>(image_status.type())) + ")");
            const auto sidecar_status = core::native_fs::symlink_status(sidecar_path, error);
            if (error)
                return sidecarIoError(error, "Inspect XMP sidecar", sidecar_path);
            if (!core::native_fs::is_regular_file(sidecar_status))
                return sidecarError(ErrorCode::FailedPrecondition, "XMP sidecar must be a regular non-symlink file (path=" + core::path_to_utf8(sidecar_path) + ", type=" + std::to_string(static_cast<int>(sidecar_status.type())) + ")");
            detail::SidecarFile image_file;
            const auto image_opened = image_file.open(image_path, std::numeric_limits<std::size_t>::max(), true);
            if (!image_opened)
                return image_opened.error();
            detail::SidecarFile file;
            const auto size = file.open(sidecar_path, max_bytes);
            if (!size)
                return size.error();
            if (file.sameFile(image_file))
                return sidecarError(ErrorCode::InvalidArgument, "Image and XMP sidecar must be different files" + pair());
            auto bytes = detail::readSidecarBytes(file, *size);
            if (!bytes)
                return Error(bytes.error()).with_context("Read XMP sidecar (path=" + core::path_to_utf8(sidecar_path) + ")", LFS_SOURCE_SITE_CURRENT());
            core::image_codecs::XmpBlock block;
            block.envelope = core::image_codecs::XmpEnvelope::SidecarFile;
            block.bytes = std::move(*bytes);
            core::image_codecs::SourceXmp xmp;
            xmp.status = core::image_codecs::MetadataStatus::Collected;
            xmp.scope = "explicit_xmp_sidecar";
            xmp.blocks.push_back(std::move(block));
            auto catalog = catalogSourceMetadata({}, limits, detail::sourceMetadataLocator(sidecar), std::move(xmp));
            if (!catalog)
                return catalog.error();
            return SourceSidecarMetadata{detail::sourceMetadataLocator(image), actual_name == full_name ? XmpSidecarNaming::ImageFilename : XmpSidecarNaming::ImageStem, std::move(*catalog)};
        } catch (const std::bad_alloc&) {
            return sidecarError(ErrorCode::ResourceExhausted, "XMP sidecar collection allocation failed");
        } catch (const std::filesystem::filesystem_error& error) {
            return sidecarIoError(error.code(), "Resolve XMP sidecar pair", error.path1());
        }
    }
} // namespace lfs::media
