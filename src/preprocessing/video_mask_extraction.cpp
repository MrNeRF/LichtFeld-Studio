/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/video_mask_extraction.hpp"
#include "core/cuda_error.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include <array>
#include <exception>

namespace lfs::preprocessing {

    std::filesystem::path video_frames_directory(const std::filesystem::path& output, const bool masking) {
        return masking ? output / "images" : output;
    }

    std::filesystem::path video_mask_path(const std::filesystem::path& dataset,
                                          const std::filesystem::path& image_name) {
        auto mask = dataset / "masks" / image_name;
        mask += ".png";
        return mask;
    }

    std::expected<VideoMaskReport, Error> mask_video_frames(const std::span<const VideoMaskFrame> frames,
                                                            const ImageCircle& circle, const VideoMaskOptions& masking) {
        struct Stream {
            cudaStream_t value = nullptr;
            Stream() { LFS_CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
            ~Stream() {
                LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(value), value, "video masking stream drain");
                core::CudaMemoryPool::instance().release_stream(value);
                LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(value), value, "video masking stream teardown");
            }
        } stream;
        const core::CUDAStreamGuard guard(stream.value);
        return mask_video_sequence(frames, circle, masking, stream.value);
    }

    io::ExtractionOutcome extract_video_frames(const io::VideoFrameExtractor::Params& params,
                                               const VideoMaskExtractionOptions& options, std::string& error, std::string* warning) {
        if (warning)
            warning->clear();
        io::VideoFrameExtractor extractor;
        if (!options.enabled) {
            extractor.extract(params, error);
            return extractor.lastOutcome();
        }
        try {
            auto extraction = params;
            extraction.output_dir = video_frames_directory(params.output_dir, true);
            std::vector<VideoMaskFrame> frames;
            extraction.frame_written = [&](const io::VideoFrameExtractor::WrittenFrame& frame) {
                frames.push_back({.image_path = frame.path,
                                  .mask_path = video_mask_path(params.output_dir, frame.path.filename()),
                                  .width = frame.width,
                                  .height = frame.height,
                                  .time_seconds = frame.timestamp});
                if (params.frame_written)
                    params.frame_written(frame);
            };
            if (!extractor.extract(extraction, error))
                return extractor.lastOutcome();
            if (frames.empty()) {
                error = "No extracted frames to mask";
                return io::ExtractionOutcome::Failed;
            }
            auto masking = options.masking;
            masking.cancel_requested = params.cancel_requested;
            auto result = mask_video_frames(frames, options.circle, masking);
            if (!result) {
                error = result.error().user_message();
                return result.error().code() == ErrorCode::Cancelled ? io::ExtractionOutcome::Cancelled
                                                                     : io::ExtractionOutcome::Failed;
            }
            if (warning)
                *warning = result->warning;
            LOG_INFO("SAM2 video masking completed: {} frames, {:.3f} seconds", frames.size(), result->mask_wall_seconds);
            return io::ExtractionOutcome::Completed;
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): converts the exception text to the returned extraction error.
            error = e.what();
            return io::ExtractionOutcome::Failed;
        }
    }

} // namespace lfs::preprocessing
