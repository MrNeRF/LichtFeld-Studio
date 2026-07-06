/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/video_frame_extractor.hpp"
#include "io/video_player.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer/gui/vulkan_ui_texture.hpp"

#include <RmlUi/Core/EventListener.h>
#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace Rml {
    class Element;
    class ElementDocument;
    class ElementFormControlSelect;
} // namespace Rml

namespace lfs::vis::gui {
    class RmlPanelHost;
    class RmlUIManager;
    struct PanelDrawContext;
    struct PanelInputState;
} // namespace lfs::vis::gui

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

        io::SharpnessAlgorithm sharpness_algorithm = io::SharpnessAlgorithm::COMBINED;
        double sharpness_threshold = 10.0;
        int window_candidates_target = 10;
        bool sharpness_window_mode = false;
        bool sharpness_enabled = false;
        bool generate_metadata = false;
        int rotation = 0; // 0, 90, 180, 270
    };

    class LFS_VIS_API VideoExtractorDialog : public IVideoExtractorWidget {
    public:
        VideoExtractorDialog();
        ~VideoExtractorDialog() override;

        [[nodiscard]] bool isVideoPlaying() const override;
        void shutdown() override;
        [[nodiscard]] bool supportsDirectDraw() const override { return true; }
        void preloadDirect(float w, float h, const lfs::vis::gui::PanelDrawContext& ctx,
                           float clip_y_min, float clip_y_max,
                           const lfs::vis::gui::PanelInputState* input) override;
        void drawDirect(float x, float y, float w, float h,
                        const lfs::vis::gui::PanelDrawContext& ctx) override;
        bool drawDirectCached(float x, float y, float w, float h,
                              const lfs::vis::gui::PanelDrawContext& ctx) override;
        [[nodiscard]] float getDirectDrawHeight() const override;
        void setInputClipY(float y_min, float y_max) override;
        void setInput(const lfs::vis::gui::PanelInputState* input) override;
        void setForcedHeight(float h) override;
        [[nodiscard]] bool wantsKeyboard() const override;
        [[nodiscard]] bool needsAnimationFrame() const override;
        void reloadRmlResources() override;

    private:
        struct EventListener final : Rml::EventListener {
            VideoExtractorDialog* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        enum class TimelineDragTarget {
            None,
            Playhead,
            TrimStart,
            TrimEnd,
        };

        enum class ExtractionStatusMessage {
            None,
            Complete,
            Stopped,
        };

        struct ExtractionStatusSnapshot {
            std::string error_message;
            ExtractionStatusMessage status_message = ExtractionStatusMessage::None;
        };

        void startExtraction(const VideoExtractionParams& params);
        void joinExtractionThread();

        void updateProgress(int current, int total, int discarded = 0);
        void setExtractionComplete();
        void setExtractionStopped();
        void setExtractionError(const std::string& error);
        [[nodiscard]] ExtractionStatusSnapshot getExtractionStatusSnapshot() const;
        void clearExtractionStatus();
        void clearStatusMessage();
        void clearErrorMessage();

        void updatePreviewTexture();
        [[nodiscard]] bool openVideo(const std::filesystem::path& path);
        [[nodiscard]] int calculateEstimatedFrames() const;

        bool ensureInitialized(lfs::vis::gui::RmlUIManager* manager);
        void clearElementCache();
        void cacheElements();
        void bindEventListeners();
        void syncPanel();
        void syncLocale();
        void syncVideoPreview();
        void syncTimeline();
        void syncControls();
        void syncExtractionStatus();
        void syncOutputPreview();
        void handleEvent(Rml::Event& event);
        void handleClick(const std::string& id);
        void handleChange(const std::string& id);
        void handleTimelineEvent(Rml::Event& event);
        void seekFromTimeline(float mouse_x);
        void setTrimFromTimeline(TimelineDragTarget target, float mouse_x);
        void applyTextInput(const std::string& id);
        void beginExtractionFromUi();
        void requestStopExtraction();
        void markContentDirty();
        void disablePanel();
        [[nodiscard]] bool hasDynamicState() const;
        [[nodiscard]] double trimDuration() const;

        std::filesystem::path video_path_;
        std::filesystem::path output_dir_;

        int mode_selection_ = 0;
        float fps_ = 1.0f;
        int frame_interval_ = 1;

        int format_selection_ = 0;
        int jpg_quality_ = 95;
        int window_candidates_target_ = 10;

        int resolution_mode_ = 0;
        int scale_selection_ = 3;
        int custom_width_ = 1920;
        int custom_height_ = 1080;

        std::array<char, 64> filename_pattern_{"frame_%d"};

        float trim_start_ = 0.0f;
        float trim_end_ = -1.0f;

        std::atomic<bool> extracting_{false};
        std::atomic<bool> stop_extraction_requested_{false};
        std::atomic<int> current_frame_{0};
        std::atomic<int> total_frames_{0};
        std::atomic<int> discarded_frames_{0};
        std::atomic<bool> extraction_status_dirty_{false};
        mutable std::mutex extraction_status_mutex_;
        std::string error_message_;
        ExtractionStatusMessage status_message_ = ExtractionStatusMessage::None;

        std::unique_ptr<io::VideoPlayer> player_;
        std::unique_ptr<lfs::vis::gui::VulkanUiTexture> preview_texture_;
        int preview_texture_width_ = 0;
        int preview_texture_height_ = 0;
        bool texture_needs_update_ = true;

        TimelineDragTarget timeline_drag_target_ = TimelineDragTarget::None;

        std::optional<std::jthread> extraction_thread_;

        std::unique_ptr<lfs::vis::gui::RmlPanelHost> host_;
        EventListener listener_;
        Rml::ElementDocument* document_ = nullptr;
        std::string last_language_;
        std::string preview_src_;
        bool elements_cached_ = false;
        bool controls_dirty_ = true;

        Rml::Element* title_el_ = nullptr;
        Rml::Element* close_btn_el_ = nullptr;
        Rml::Element* preview_shell_el_ = nullptr;
        Rml::Element* preview_image_el_ = nullptr;
        Rml::Element* preview_empty_el_ = nullptr;
        Rml::Element* step_back_btn_el_ = nullptr;
        Rml::Element* play_btn_el_ = nullptr;
        Rml::Element* play_icon_el_ = nullptr;
        Rml::Element* step_forward_btn_el_ = nullptr;
        Rml::Element* time_label_el_ = nullptr;
        Rml::Element* timeline_el_ = nullptr;
        Rml::Element* timeline_trim_el_ = nullptr;
        Rml::Element* timeline_progress_el_ = nullptr;
        Rml::Element* timeline_playhead_el_ = nullptr;
        Rml::Element* timeline_start_el_ = nullptr;
        Rml::Element* timeline_end_el_ = nullptr;
        Rml::Element* timeline_markers_el_ = nullptr;
        Rml::Element* trim_start_input_el_ = nullptr;
        Rml::Element* trim_end_input_el_ = nullptr;
        Rml::Element* trim_start_set_el_ = nullptr;
        Rml::Element* trim_end_set_el_ = nullptr;
        Rml::Element* trim_reset_el_ = nullptr;
        Rml::Element* estimated_frames_el_ = nullptr;
        Rml::Element* video_value_el_ = nullptr;
        Rml::Element* output_value_el_ = nullptr;
        Rml::Element* browse_video_el_ = nullptr;
        Rml::Element* browse_output_el_ = nullptr;
        Rml::ElementFormControlSelect* mode_select_el_ = nullptr;
        Rml::Element* fps_row_el_ = nullptr;
        Rml::Element* fps_slider_el_ = nullptr;
        Rml::Element* fps_value_el_ = nullptr;
        Rml::Element* interval_row_el_ = nullptr;
        Rml::Element* interval_input_el_ = nullptr;
        Rml::Element* interval_value_el_ = nullptr;
        Rml::ElementFormControlSelect* format_select_el_ = nullptr;
        Rml::Element* quality_row_el_ = nullptr;
        Rml::Element* quality_slider_el_ = nullptr;
        Rml::Element* quality_value_el_ = nullptr;
        Rml::ElementFormControlSelect* resolution_select_el_ = nullptr;
        Rml::Element* scale_row_el_ = nullptr;
        Rml::ElementFormControlSelect* scale_select_el_ = nullptr;
        Rml::Element* custom_resolution_row_el_ = nullptr;
        Rml::Element* custom_width_input_el_ = nullptr;
        Rml::Element* custom_height_input_el_ = nullptr;
        Rml::Element* output_resolution_el_ = nullptr;
        Rml::Element* pattern_input_el_ = nullptr;
        Rml::Element* pattern_example_el_ = nullptr;
        Rml::Element* start_btn_el_ = nullptr;
        Rml::Element* stop_btn_el_ = nullptr;
        Rml::Element* cancel_btn_el_ = nullptr;
        Rml::Element* select_hint_el_ = nullptr;
        Rml::Element* progress_section_el_ = nullptr;
        Rml::Element* progress_text_el_ = nullptr;
        Rml::Element* progress_bar_el_ = nullptr;
        Rml::Element* complete_section_el_ = nullptr;
        Rml::Element* complete_text_el_ = nullptr;
        Rml::Element* ok_btn_el_ = nullptr;
        Rml::Element* stopped_section_el_ = nullptr;
        Rml::Element* stopped_text_el_ = nullptr;
        Rml::Element* stopped_ok_btn_el_ = nullptr;
        Rml::Element* error_section_el_ = nullptr;
        Rml::Element* error_text_el_ = nullptr;
        Rml::Element* dismiss_btn_el_ = nullptr;
        Rml::Element* sharpness_toggle_el_ = nullptr;
        Rml::Element* sharpness_options_el_ = nullptr;
        Rml::Element* sharpness_threshold_row_el_ = nullptr;
        Rml::ElementFormControlSelect* sharpness_algorithm_select_el_ = nullptr;
        Rml::ElementFormControlSelect* sharpness_mode_select_el_ = nullptr;
        Rml::Element* sharpness_mode_desc_el_ = nullptr;
        Rml::Element* sharpness_threshold_slider_el_ = nullptr;
        Rml::Element* sharpness_threshold_value_el_ = nullptr;
        Rml::Element* sharpness_window_row_el_ = nullptr;
        Rml::Element* window_candidates_select_el_ = nullptr;
        Rml::Element* window_candidates_readout_el_ = nullptr;
        Rml::Element* generate_metadata_el_ = nullptr;
        Rml::Element* overwrite_overlay_el_ = nullptr;
        int rotation_deg_ = 0;
        std::vector<uint8_t> rotated_buf_;
        Rml::Element* rotation_cw_btn_el_ = nullptr;
        Rml::Element* rotation_ccw_btn_el_ = nullptr;
        Rml::Element* rotation_value_el_ = nullptr;
        VideoExtractionParams pending_params_;
        bool pending_params_set_ = false;
    };

} // namespace lfs::gui
