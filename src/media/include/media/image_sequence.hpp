// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/file_frame_sink.hpp"
#include <memory>

namespace lfs::media {
    struct ImageSequenceOptions {
        FileFrameSinkOptions files{.image_output = {.overwrite = false, .durable = false, .create_directories = false}};
        FrameLayout layout;
        int framerate = 30;
        int frame_count = 0;
        std::function<bool()> cancelled;
    };
    // Metadata-only preflight. Does not create directories or open a provider.
    [[nodiscard]] LFS_MEDIA_API SinkResult validateImageSequenceOptions(const ImageSequenceOptions&);

    // Ordered, synchronous output for rendered frames. Uses the existing image
    // sink and atomic writers; no synthetic decoder/source timestamps. Partial
    // sequences retain completed images and cannot be reported as complete.
    class LFS_MEDIA_API ImageSequence {
    public:
        ImageSequence();
        ~ImageSequence();
        ImageSequence(const ImageSequence&) = delete;
        ImageSequence& operator=(const ImageSequence&) = delete;
        [[nodiscard]] SinkResult open(const ImageSequenceOptions&);
        [[nodiscard]] SinkResult write(const FrameView&);
        [[nodiscard]] SinkResult close();
        void abort() noexcept;
        [[nodiscard]] bool isOpen() const noexcept;
        [[nodiscard]] std::size_t framesAccepted() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::media
