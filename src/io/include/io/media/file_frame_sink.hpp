// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "io/media/frame_sink.hpp"
#include <filesystem>
#include <unordered_set>

namespace lfs::media {
    enum class FrameFileFormat { PNG,
                                 JPEG };
    struct FileFrameSinkOptions {
        std::filesystem::path output_directory;
        std::string filename_pattern = "frame_%d";
        FrameFileFormat format = FrameFileFormat::PNG;
        int jpeg_quality = 95;
    };
    class FileFrameSink final : public FrameSink {
    public:
        explicit FileFrameSink(FileFrameSinkOptions options);
        SinkResult begin(const SinkSession&) override;
        SinkResult write(const FrameView&) override;
        SinkResult complete(const SinkSummary&) override;
        void abort(const SinkSummary&) noexcept override;

    private:
        FileFrameSinkOptions options_;
        bool active_ = false;
        std::unordered_set<std::filesystem::path> filenames_;
    };
} // namespace lfs::media
