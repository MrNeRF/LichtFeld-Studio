// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/atomic_file.hpp"
#include "core/export.hpp"
#include "media/image_output.hpp"
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>

namespace lfs::media {
    namespace detail {
        struct JpegMetadata;
        class ResumableFileSink;
    } // namespace detail
    enum class FrameFileFormat { PNG,
                                 JPEG,
                                 EXR,
                                 TIFF16,
                                 JXL,
                                 AVIF };
    enum class FrameFileNumbering { SourceFrame,
                                    DeliveryIndex };
    struct FileFrameSinkOptions {
        std::filesystem::path output_directory;
        std::string filename_pattern = "frame_%d";
        FrameFileFormat format = FrameFileFormat::PNG;
        int jpeg_quality = 95;
        ExrOutputOptions exr;
        // Previous extraction output (including metadata), only after success.
        bool remove_stale_frames = false;
        // The extractor owns metadata when it generates a new manifest.
        bool preserve_metadata = false;
        // Appended to preserve existing positional aggregate initializers.
        core::AtomicFileOptions image_output{.overwrite = true, .durable = false, .create_directories = false};
        int first_source_frame_number = 1;
        Tiff16OutputOptions tiff16;
        JxlOutputOptions jxl;
        AvifOutputOptions avif;
        // Extraction retains its source-frame numbering. Rendered output uses
        // the zero-based delivery ordinal without inventing a source frame.
        FrameFileNumbering numbering = FrameFileNumbering::SourceFrame;
        // Required EXIF output geometry and XMP static source provenance; JPEG only.
        bool embed_source_metadata = false;
    };
    class LFS_MEDIA_API FileFrameSink final : public FrameSink {
    public:
        explicit FileFrameSink(FileFrameSinkOptions options);
        SinkResult begin(const SinkSession&) override;
        SinkResult write(const FrameView&) override;
        SinkResult complete(const SinkSummary&) override;
        void abort(const SinkSummary&) noexcept override;

    private:
        friend class detail::ResumableFileSink;
        SinkResult beginPrepared(const SinkSession&, std::shared_ptr<const detail::JpegMetadata>);
        FileFrameSinkOptions options_;
        bool active_ = false;
        std::shared_ptr<const detail::JpegMetadata> jpeg_metadata_;
        std::unordered_set<std::filesystem::path> filenames_;
        struct ExistingFrame {
            std::filesystem::path path;
            std::filesystem::file_time_type modified;
            std::uintmax_t size;
        };
        std::vector<ExistingFrame> existing_frames_;
    };
} // namespace lfs::media
