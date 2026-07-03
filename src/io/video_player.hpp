/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace lfs::io {

    class VideoPlayer {
    public:
        VideoPlayer();
        ~VideoPlayer();

        VideoPlayer(const VideoPlayer&) = delete;
        VideoPlayer& operator=(const VideoPlayer&) = delete;

        bool open(const std::filesystem::path& path);
        void close();
        bool isOpen() const;

        void play();
        void pause();
        void togglePlayPause();
        bool isPlaying() const;

        void seek(double seconds);
        void seekFrame(int64_t frame_number);
        void stepForward();
        void stepBackward();

        // Update playback, returns true if frame changed
        bool update(double delta_seconds);

        // Get current frame as RGB data (call after update returns true)
        const uint8_t* currentFrameData() const;
        int width() const;
        int height() const;
        int sourceWidth() const;
        int sourceHeight() const;

        double currentTime() const;
        double duration() const;
        int64_t currentFrameNumber() const;
        int64_t totalFrames() const;
        double fps() const;

        // Thumbnail generation (seeks and decodes single frame)
        std::vector<uint8_t> getThumbnail(double time, int max_width);

        // Detected rotation from video metadata (0, 90, 180, 270)
        int rotation() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io
