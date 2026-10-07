// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace lfs::media {
    enum class VideoEncodeBackend { Software,
                                    Cuda,
                                    VideoToolbox };
    enum class VideoEncodeLayout { YUV420P,
                                   NV12 };
    struct VideoEncodeOptions {
        int width = 0;
        int height = 0;
        int framerate = 30;
        int crf = 18;
        // Requested backend; an unavailable hardware encoder falls back to software.
        VideoEncodeBackend preferred_backend = VideoEncodeBackend::Software;
        // Prepared by the producer; no application metadata factory is needed.
        std::string comment;
    };
    struct VideoEncodePlane {
        std::uint8_t* data = nullptr;
        // Pitch and visible width are bytes; NV12 UV contains interleaved pairs.
        std::size_t row_stride = 0;
        int width = 0;
        int height = 0;
    };
    struct VideoEncodeTarget {
        VideoEncodeLayout layout = VideoEncodeLayout::YUV420P;
        VideoEncodeBackend backend = VideoEncodeBackend::Software;
        std::array<VideoEncodePlane, 3> planes;
    };
    // Borrowed writable planes are valid only during write(). CUDA storage is
    // owned by the codec's context: writers use transfers valid across contexts
    // and finish their work before returning. CPU writers must not dereference
    // CUDA addresses. A rejected write is not submitted or assigned an output PTS.
    class VideoEncodeWriter {
    public:
        virtual ~VideoEncodeWriter() = default;
        virtual Result<void> write(const VideoEncodeTarget& target) = 0;
    };
    // Codec/file/session ownership stays in media. No FFmpeg, Tensor or native
    // GPU runtime types are part of the producer-facing contract. Calls must be
    // serialized, and a writer must not reenter its session.
    class LFS_MEDIA_API VideoEncodeSession {
    public:
        VideoEncodeSession();
        // Releases resources. Call close() explicitly to flush packets/trailer.
        ~VideoEncodeSession();
        VideoEncodeSession(const VideoEncodeSession&) = delete;
        VideoEncodeSession& operator=(const VideoEncodeSession&) = delete;
        VideoEncodeSession(VideoEncodeSession&&);
        VideoEncodeSession& operator=(VideoEncodeSession&&);

        [[nodiscard]] Result<void> open(const std::filesystem::path& path, const VideoEncodeOptions& options);
        [[nodiscard]] Result<void> writeFrame(VideoEncodeWriter& writer);
        [[nodiscard]] Result<void> close();
        [[nodiscard]] bool isOpen() const;
        [[nodiscard]] VideoEncodeBackend backend() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::media
