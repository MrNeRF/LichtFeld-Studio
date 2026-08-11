/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
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

        void togglePlayPause();
        bool isPlaying() const;

        void seek(double seconds);
        bool rerenderCurrentFrame();
        void stepForward();
        void stepBackward();

        // Update playback, returns true if frame changed
        bool update(double delta_seconds);

        // Get current frame as RGB or RGBA data (call after update returns true).
        const uint8_t* currentFrameData() const;
        int currentFrameChannels() const;
        bool currentFrameHasGpuRotation() const;
        int width() const;
        int height() const;
        int sourceWidth() const;
        int sourceHeight() const;

        double currentTime() const;
        double duration() const;
        double fps() const;

        // Detected rotation from video metadata (0, 90, 180, 270)
        int rotation() const;

        // HDR detection
        bool isHdr() const;
        bool isHdrConversionSupported();
        std::string hdrInfo() const;
        void setHdrToSdr(bool enabled);
        void setPreviewRotation(int degrees);
        std::string takeError();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io
