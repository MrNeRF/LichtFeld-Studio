/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/video_frame_extractor.hpp"
#include "io/video_player.hpp"
#include "preprocessing/video_mask_extraction.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer/gui/vulkan_ui_texture.hpp"

#include <RmlUi/Core/EventListener.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
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
        int first_source_frame_number = 1;

        io::SharpnessAlgorithm sharpness_algorithm = io::SharpnessAlgorithm::COMBINED;
        double sharpness_threshold = 10.0;
        int window_candidates_target = 10;
        bool sharpness_window_mode = false;
        bool sharpness_enabled = false;
        preprocessing::VideoMaskExtractionOptions masking;
        bool generate_metadata = false;
        bool convert_hdr_to_sdr = false;
        int rotation = 0; // 0, 90, 180, 270
    };

    class LFS_VIS_API VideoExtractorDialog : public IVideoExtractorWidget {
    public:
        VideoExtractorDialog();
        ~VideoExtractorDialog() override;

        [[nodiscard]] bool isVideoPlaying() const override;
        [[nodiscard]] bool openVideoPath(const std::filesystem::path& path) override;
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
        void setFloating(bool floating) override;
        SDL_Window* nativeWindow() const override;
        bool detachToDesktop(float x, float y, float w, float h) override;
        bool restoreDesktopWindow() override;
        void releaseDesktopWindow() override;
        void resetDesktopRestoreState() override {
            desktop_reopen_bounds_.reset();
            desktop_restore_checked_ = false;
        }
        bool detachHost(float x, float y, float w, float h, bool desktop_coordinates);
        [[nodiscard]] bool wantsKeyboard() const override;
        [[nodiscard]] bool needsAnimationFrame() const override;
        void reloadRmlResources() override;

    private:
        friend class VideoExtractorDialogTestAccess;
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
            std::string warning_message;
            ExtractionStatusMessage status_message = ExtractionStatusMessage::None;
        };

        enum class BatchItemStatus {
            Pending,
            Running,
            Complete,
            Failed,
            Stopped,
        };

        struct BatchVideoItem {
            std::filesystem::path video_path;
            std::filesystem::path output_dir;
            double start_time = 0.0;
            double end_time = -1.0;
            bool end_time_is_auto = true;
            int rotation = 0;
            double source_fps = 0.0;
            double duration = 0.0;
            bool is_hdr = false;
            bool hdr_to_sdr_supported = false;
            bool has_existing_output = false;
            BatchItemStatus status = BatchItemStatus::Pending;
            std::string error;
            std::string warning;
        };

        struct QueuedBatchProbe {
            std::filesystem::path video_path;
            uint64_t generation = 0;
        };

        struct BatchProbeFailure {
            std::filesystem::path video_path;
            std::string error;
            uint64_t generation = 0;
        };

        struct CompletedBatchProbe {
            BatchVideoItem item;
            uint64_t generation = 0;
        };

        void startExtraction(const VideoExtractionParams& params);
        void startBatchExtraction(std::vector<VideoExtractionParams> params);
        void joinExtractionThread();

        void updateProgress(int current, int total, int discarded = 0);
        void setExtractionComplete(const std::string& warning = {});
        void setExtractionStopped();
        void setExtractionError(const std::string& error, bool extraction_failure = true);
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
        bool syncHdrControls();
        void syncExtractionStatus();
        void syncOutputPreview();
        void syncOverwritePrompt();
        void handleEvent(Rml::Event& event);
        void handleClick(const std::string& id);
        void handleChange(const std::string& id, bool explicit_edit = true);
        void handleTimelineEvent(Rml::Event& event);
        void seekFromTimeline(float mouse_x);
        void setTrimFromTimeline(TimelineDragTarget target, float mouse_x);
        void applyTextInput(const std::string& id);
        [[nodiscard]] bool resolveFilenamePattern(std::string& pattern, std::string& error) const;
        [[nodiscard]] bool resolveFilenamePattern(const std::filesystem::path& video_path,
                                                  std::string& pattern, std::string& error) const;
        [[nodiscard]] bool collectMaskingOptions(preprocessing::VideoMaskExtractionOptions& options, std::string& error) const;
        [[nodiscard]] bool buildBatchParams(std::vector<VideoExtractionParams>& params,
                                            std::string& error);
        void beginBatchExtractionFromUi();
        void launchBatchExtraction(std::vector<VideoExtractionParams> params);
        void addVideosFromBrowse(std::vector<std::filesystem::path> paths);
        [[nodiscard]] bool addBatchVideos(const std::vector<std::filesystem::path>& paths);
        void runBatchProbeWorker(std::stop_token stop_token);
        void applyCompletedBatchProbes();
        void joinBatchProbeThread();
        void selectBatchItem(std::size_t index);
        void saveSelectedBatchItem();
        [[nodiscard]] bool probeBatchVideo(BatchVideoItem& item, std::string& error) const;
        void refreshBatchOutputDirectories();
        void syncBatchList();
        void removeSelectedBatchItem();
        void clearBatchVideos();
        [[nodiscard]] std::string batchStatusLabel(const BatchVideoItem& item) const;
        [[nodiscard]] bool removeExistingBatchOutput(const std::vector<VideoExtractionParams>& params);
        void beginExtractionFromUi();
        void launchExtraction(const VideoExtractionParams& params);
        void requestStopExtraction();
        void markContentDirty();
        void disablePanel();
        [[nodiscard]] bool hasDynamicState() const;
        [[nodiscard]] double trimDuration() const;

        std::filesystem::path video_path_;
        std::filesystem::path output_dir_;
        std::vector<BatchVideoItem> batch_items_;
        std::optional<std::size_t> selected_batch_item_;
        std::vector<VideoExtractionParams> pending_batch_params_;
        bool pending_batch_metadata_will_be_replaced_ = false;
        int pending_batch_conflict_count_ = 0;
        bool continue_batch_on_failure_ = true;
        bool batch_output_root_explicit_ = false;
        std::atomic<bool> batch_list_dirty_{true};
        std::atomic<int> active_batch_index_{-1};
        std::atomic<int> completed_batch_items_{0};
        std::atomic<int> failed_batch_items_{0};
        mutable std::mutex batch_mutex_;

        std::optional<std::jthread> batch_probe_thread_;
        std::deque<QueuedBatchProbe> queued_batch_probes_;
        std::vector<std::filesystem::path> scheduled_batch_probe_paths_;
        std::vector<CompletedBatchProbe> completed_batch_probe_items_;
        std::vector<BatchProbeFailure> completed_batch_probe_failures_;
        mutable std::mutex batch_probe_mutex_;
        std::atomic<bool> batch_probe_active_{false};
        std::atomic<bool> batch_probe_results_ready_{false};
        uint64_t batch_probe_generation_ = 0;

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

        int naming_selection_ = 2;
        std::string filename_pattern_ = "frame_%05d";
        int first_source_frame_number_ = 1;

        float trim_start_ = 0.0f;
        float trim_end_ = -1.0f;
        bool trim_end_is_auto_ = true;

        std::atomic<bool> extracting_{false};
        std::atomic<bool> stop_extraction_requested_{false};
        std::atomic<int> current_frame_{0};
        std::atomic<int> total_frames_{0};
        std::atomic<int> discarded_frames_{0};
        std::atomic<bool> extraction_status_dirty_{false};
        mutable std::mutex extraction_status_mutex_;
        std::string error_message_;
        std::string warning_message_;
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
        bool floating_ = false;
        bool desktop_restore_checked_ = false;
        std::optional<std::array<float, 4>> desktop_reopen_bounds_;
        lfs::vis::gui::RmlUIManager* desktop_manager_ = nullptr;

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
        Rml::Element* batch_video_list_el_ = nullptr;
        Rml::Element* batch_summary_el_ = nullptr;
        Rml::Element* remove_video_el_ = nullptr;
        Rml::Element* clear_videos_el_ = nullptr;
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
        Rml::Element* masking_toggle_el_ = nullptr;
        Rml::Element* masking_options_el_ = nullptr;
        Rml::Element* mask_circle_el_ = nullptr;
        Rml::Element* mask_recenter_el_ = nullptr;
        Rml::Element* mask_fov_el_ = nullptr;
        Rml::Element* mask_model_el_ = nullptr;
        Rml::Element* mask_require_operator_el_ = nullptr;
        Rml::Element* mask_dilation_el_ = nullptr;
        Rml::Element* mask_view_fov_el_ = nullptr;
        Rml::Element* mask_manual_seed_el_ = nullptr;
        Rml::Element* mask_seed_options_el_ = nullptr;
        Rml::Element* mask_seed_x_el_ = nullptr;
        Rml::Element* mask_seed_y_el_ = nullptr;
        Rml::Element* mask_seed_frame_el_ = nullptr;
        std::atomic<bool> masking_active_{false};
        Rml::ElementFormControlSelect* naming_select_el_ = nullptr;
        Rml::Element* custom_pattern_row_el_ = nullptr;
        Rml::Element* naming_help_el_ = nullptr;
        Rml::Element* naming_error_el_ = nullptr;
        Rml::Element* pattern_input_el_ = nullptr;
        Rml::Element* first_source_frame_number_input_el_ = nullptr;
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
        Rml::Element* continue_batch_on_failure_el_ = nullptr;
        Rml::Element* hdr_to_sdr_el_ = nullptr;
        Rml::Element* hdr_to_sdr_row_el_ = nullptr;
        Rml::Element* overwrite_overlay_el_ = nullptr;
        Rml::Element* overwrite_message_el_ = nullptr;
        Rml::Element* overwrite_cancel_btn_el_ = nullptr;
        Rml::Element* overwrite_delete_btn_el_ = nullptr;
        Rml::Element* overwrite_proceed_btn_el_ = nullptr;
        int rotation_deg_ = 0;
        bool hdr_to_sdr_ = false;
        std::vector<uint8_t> rotated_buf_;
        Rml::Element* rotation_cw_btn_el_ = nullptr;
        Rml::Element* rotation_ccw_btn_el_ = nullptr;
        Rml::Element* rotation_value_el_ = nullptr;
        Rml::Element* hdr_badge_el_ = nullptr;
        Rml::Element* hdr_badge_type_el_ = nullptr;
        Rml::Element* hdr_badge_label_el_ = nullptr;
        VideoExtractionParams pending_params_;
        bool pending_params_set_ = false;
        bool pending_has_frame_conflicts_ = false;
        bool pending_metadata_will_be_replaced_ = false;
    };

} // namespace lfs::gui
