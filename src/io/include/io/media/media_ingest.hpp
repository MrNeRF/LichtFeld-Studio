// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "io/media/file_frame_sink.hpp"
#include <functional>

namespace lfs::media {
    enum class SelectionMode { FPS,
                               Interval };
    enum class ResizeMode { Original,
                            Scale,
                            Custom };
    enum class SharpnessMethod { Laplacian,
                                 Tenengrad,
                                 Combined };
    struct Selection {
        SelectionMode mode = SelectionMode::FPS;
        double fps = 1;
        int interval = 1;
    };
    struct Geometry {
        ResizeMode mode = ResizeMode::Original;
        float scale = 1;
        int width = 0;
        int height = 0;
        int clockwise_rotation = 0;
    };
    struct Sharpness {
        bool enabled = false;
        SharpnessMethod method = SharpnessMethod::Combined;
        double threshold = 0;
        int window_candidates = 10;
        bool window = false;
    };
    struct IngestProgress {
        int processed = 0;
        int estimated = 0;
        int discarded = 0;
    };
    struct IngestRequest {
        std::filesystem::path input;
        Selection selection;
        Geometry geometry;
        Sharpness sharpness;
        double start_seconds = 0;
        double end_seconds = -1;
        bool convert_hdr_to_sdr = false;
        std::function<void(const IngestProgress&)> progress;
        std::function<bool()> cancelled;
    };
    struct FileExtraction {
        FileFrameSinkOptions files;
        bool write_metadata = false;
    };
    struct IngestReport {
        std::size_t frames_accepted = 0;
        int discarded = 0;
    };
    struct IngestCapabilities {
        bool software_decode = true;
        bool rgb8 = true;
        bool png = true;
        bool jpeg = true;
        bool hardware_decode = false;
        bool hdr_to_sdr = false;
    };
    // CPU profile: synchronous calls, no app runtime or GPU context. Failed
    // results retain accepted count as an error field; sinks retain partial data.
    class MediaIngest {
    public:
        [[nodiscard]] static IngestCapabilities capabilities() noexcept;
        [[nodiscard]] static Result<IngestReport> extract(const IngestRequest&, FrameSink&);
        [[nodiscard]] static Result<IngestReport> extractFiles(const IngestRequest&, const FileExtraction&);
    };
} // namespace lfs::media
