/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/atomic_file.hpp"
#include "core/export.hpp"
#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "media/hdr_tonemap.hpp"
#include "media/image_output.hpp"

namespace lfs::media {
    class FrameSink;
}

namespace lfs::io {

    // Supports %d, %0Nd zero-padding, %% escaping, and legacy %000 zero-padding.
    [[nodiscard]] LFS_MEDIA_API std::string formatFrameFilenameStem(std::string_view pattern, int frame_number);
    // True when the extractor writes `filename` (name only) for `pattern`: a frame stem with any
    // positive frame number and padding width, followed by .png, .jpg or .exr.
    [[nodiscard]] LFS_MEDIA_API bool isGeneratedFrameFilename(std::string_view pattern, std::string_view filename,
                                                              bool allow_zero = false);
    // Frames and metadata an earlier extraction wrote into `directory`: files the metadata manifest
    // lists and files `pattern` produces. Other files, such as existing photos, are never included.
    [[nodiscard]] LFS_MEDIA_API std::vector<std::filesystem::path> generatedExtractionFiles(
        const std::filesystem::path& directory, std::string_view pattern);
    [[nodiscard]] LFS_MEDIA_API std::size_t calculateFpsSampleCount(double start_time, double end_time,
                                                                    double target_fps);
    [[nodiscard]] LFS_MEDIA_API double fpsSampleTime(double start_time, double end_time,
                                                     double target_fps, std::size_t sample_index);
    [[nodiscard]] LFS_MEDIA_API bool frameCoversSampleTime(double frame_time, double frame_duration,
                                                           double sample_time);
    [[nodiscard]] LFS_MEDIA_API bool shouldFillRetainedFpsTail(bool reached_eof, bool reached_end);
    // End time to request for a trim range edited against the preview player's duration. An end
    // at the player's (float) duration can mean "to the end of the video" (-1) for an automatic
    // interval range. Timed sampling and explicit user bounds retain their numeric endpoint.
    [[nodiscard]] inline double extractionEndTime(const float trim_end, const double player_duration,
                                                  const bool allow_stream_end = true) {
        return trim_end < 0.0f || (allow_stream_end && trim_end >= static_cast<float>(player_duration))
                   ? -1.0
                   : static_cast<double>(trim_end);
    }

    enum class ExtractionMode {
        FPS,     // Extract at specific FPS
        INTERVAL // Extract every N frames
    };

    enum class ImageFormat {
        PNG,
        JPG,
        EXR,
        TIFF16,
        JXL,
        AVIF
    };

    enum class ResolutionMode {
        Original,
        Scale,
        Custom
    };

    enum class ExtractionOutcome {
        Completed,
        Cancelled,
        Failed,
    };

    enum class SharpnessAlgorithm {
        LAPLACIAN, // Laplacian variance — fast, blur detection
        TENENGRAD, // Sobel energy — directional blur detection
        COMBINED   // Tenengrad + Laplacian — best overall
    };

    struct SharpnessParams {
        bool enabled = false;
        SharpnessAlgorithm algorithm = SharpnessAlgorithm::COMBINED;
        double threshold = 0.0;            // 0 = disabled (no threshold filtering)
        int window_candidates_target = 10; // <0=auto, 0=all, >0=fixed candidates per interval
        bool window_mode = false;
    };

    class LFS_MEDIA_API VideoFrameExtractor {
    public:
        VideoFrameExtractor();
        ~VideoFrameExtractor();

        struct WrittenFrame {
            std::filesystem::path path;
            int width = 0, height = 0;
            double timestamp = 0;
            int source_frame = 0;
        };

        struct Params {
            std::filesystem::path video_path;
            std::filesystem::path output_dir;
            ExtractionMode mode = ExtractionMode::FPS;
            double fps = 1.0;
            int frame_interval = 1;
            int first_source_frame_number = 1;
            ImageFormat format = ImageFormat::PNG;
            int jpg_quality = 95;
            SharpnessParams sharpness;
            std::function<void(int, int, int)> progress_callback; // (current, total, discarded)
            std::function<bool()> cancel_requested;
            std::function<void(const WrittenFrame&)> frame_written;

            // Trim range
            double start_time = 0.0;
            double end_time = -1.0; // -1 means end of video

            // Resolution
            ResolutionMode resolution_mode = ResolutionMode::Original;
            float scale = 1.0f;
            int custom_width = 0;
            int custom_height = 0;

            // Output naming
            std::string filename_pattern = "frame_%d"; // %d = 1-based source video frame number
            bool generate_metadata = false;
            int rotation = 0; // 0, 90, 180, 270
            bool convert_hdr_to_sdr = false;
            bool allow_hardware_decode = true;
            media::FramePixelFormat output_format = media::FramePixelFormat::RGB8;
            media::FrameColor input_color;
            media::FloatColorProfile float_color_profile = media::FloatColorProfile::Sdr;
            media::ExrPrecision exr_precision = media::ExrPrecision::Half;
            media::ExrCompression exr_compression = media::ExrCompression::ZIP;
            bool overwrite_metadata = false;
            bool allow_hardware_encode = true;
            media::ColorTransfer tiff16_transfer = media::ColorTransfer::Srgb;
            bool jxl_lossless = true;
            float jxl_distance = 1.f;
            int jxl_effort = 4;
            media::JxlPrecision jxl_precision = media::JxlPrecision::Float32;
            media::ColorTransfer jxl_transfer = media::ColorTransfer::Linear;
            bool avif_lossless = true;
            int avif_quality = 90;
            int avif_speed = 8;
            int avif_bit_depth = 8;
            bool preserve_source_metadata = false;
            bool embed_source_metadata = false;
            // Publication policy for embedded JPEG/PNG metadata; legacy field name retained.
            core::AtomicFileOptions jpeg_metadata_output{.overwrite = true, .durable = false, .create_directories = false};
        };

        struct ValidatedLayout {
            int width = 0;
            int height = 0;
            std::size_t rgb_bytes = 0;
        };

        [[nodiscard]] static bool validateParams(const Params& params, int source_width,
                                                 int source_height, double stream_time_base,
                                                 ValidatedLayout& layout, std::string& error);
        bool extract(const Params& params, std::string& error);
        // Synchronous CPU delivery. output_dir, file naming and generate_metadata
        // do not cause filesystem output; the supplied sink owns that policy.
        bool extractToSink(const Params& params, media::FrameSink& sink, std::string& error);
        // CPU sink delivery with compatibility filename deduplication and schema-2 metadata.
        bool extractFilesToSink(const Params& params, media::FrameSink& sink, std::string& error);
        [[nodiscard]] ExtractionOutcome lastOutcome() const;
        [[nodiscard]] std::optional<Error> lastError() const;

    private:
        class Impl;
        Impl* impl_;
    };

} // namespace lfs::io
