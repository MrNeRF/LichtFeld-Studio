/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/video_frame_extractor.hpp"
#include "io/video_player.hpp"

#include <array>
#include <atomic>
#include <filesystem>
#include <functional>
#include <glad/glad.h>
#include <string>

struct ImDrawList;
struct ImVec2;

namespace lfs::gui {

    struct VideoExtractionParams {
        std::filesystem::path video_path;
        std::filesystem::path output_dir;
        io::ExtractionMode mode = io::ExtractionMode::FPS;
        double fps = 1.0;
        int frame_interval = 1;
        io::ImageFormat format = io::ImageFormat::PNG;
        int jpg_quality = 95;

        double start_time = 0.0;
        double end_time = -1.0;

        io::ResolutionMode resolution_mode = io::ResolutionMode::Original;
        float scale = 1.0f;
        int custom_width = 0;
        int custom_height = 0;

        std::string filename_pattern = "frame_%d";
    };

    class VideoExtractorDialog {
    public:
        VideoExtractorDialog();
        ~VideoExtractorDialog();

        bool render();
        void setOnStartExtraction(std::function<void(const VideoExtractionParams&)> callback);

        void updateProgress(int current, int total);
        void setExtractionComplete();
        void setExtractionError(const std::string& error);

        [[nodiscard]] bool isVideoPlaying() const;

    private:
        void renderVideoPreview();
        void renderTransportControls();
        void renderTimeline();
        void renderTrimControls();
        void renderFileSelection();
        void renderExtractionSettings();
        void renderFormatSettings();
        void renderResolutionSettings();
        void renderOutputSettings();
        void updatePreviewTexture();
        void openVideo(const std::filesystem::path& path);
        void renderExtractionMarkers(ImDrawList* dl, ImVec2 pos, float width, float height, double duration);
        [[nodiscard]] int calculateEstimatedFrames() const;

        std::filesystem::path video_path_;
        std::filesystem::path output_dir_;

        int mode_selection_ = 0;
        float fps_ = 1.0f;
        int frame_interval_ = 1;

        int format_selection_ = 0;
        int jpg_quality_ = 95;

        int resolution_mode_ = 0;
        int scale_selection_ = 3;
        int custom_width_ = 1920;
        int custom_height_ = 1080;

        std::array<char, 64> filename_pattern_{"frame_%d"};

        float trim_start_ = 0.0f;
        float trim_end_ = -1.0f;

        std::function<void(const VideoExtractionParams&)> on_start_extraction_;

        std::atomic<bool> extracting_{false};
        std::atomic<int> current_frame_{0};
        std::atomic<int> total_frames_{0};
        std::string error_message_;
        bool show_completion_message_ = false;

        std::unique_ptr<io::VideoPlayer> player_;
        GLuint preview_texture_ = 0;
        int preview_texture_width_ = 0;
        int preview_texture_height_ = 0;
        bool texture_needs_update_ = true;

        bool scrubbing_ = false;
    };

} // namespace lfs::gui
