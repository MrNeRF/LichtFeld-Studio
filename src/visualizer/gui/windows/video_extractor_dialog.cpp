/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_extractor_dialog.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/include/core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/panel_height_mode.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/native_file_dialog.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string_view>

using namespace lichtfeld::Strings;

namespace lfs::gui {

    namespace {

        constexpr std::array<float, 4> SCALE_VALUES{0.25f, 0.5f, 0.75f, 1.0f};
        constexpr float MIN_TRIM_SECONDS = 0.1f;
        constexpr int MAX_TIMELINE_MARKERS = 180;

        [[nodiscard]] std::string formatTime(const double seconds) {
            if (!std::isfinite(seconds) || seconds < 0.0)
                return "--:--.--";

            const int mins = static_cast<int>(seconds) / 60;
            const double secs = seconds - static_cast<double>(mins * 60);
            return std::format("{}:{:05.2f}", mins, secs);
        }

        [[nodiscard]] std::string cacheAttrName(const std::string_view kind,
                                                const std::string_view name) {
            std::string attr = "data-lfs-video-extractor-";
            attr += kind;
            attr += "-";
            attr += name;
            return attr;
        }

        [[nodiscard]] bool setCachedInnerRml(Rml::Element* const el, const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("rml", "value");
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetInnerRML(value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        [[nodiscard]] bool setCachedText(Rml::Element* const el, const std::string& value) {
            return setCachedInnerRml(el, Rml::StringUtilities::EncodeRml(value));
        }

        [[nodiscard]] bool setCachedProperty(Rml::Element* const el,
                                             const std::string_view name,
                                             const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("prop", name);
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetProperty(std::string(name), value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        [[nodiscard]] bool setCachedAttribute(Rml::Element* const el,
                                              const std::string_view name,
                                              const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("attr", name);
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetAttribute(std::string(name), value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        [[nodiscard]] std::string controlValue(const Rml::Element* const el) {
            if (!el)
                return {};

            if (const auto* const input = dynamic_cast<const Rml::ElementFormControlInput*>(el))
                return input->GetValue();

            return el->GetAttribute<Rml::String>("value", "");
        }

        [[nodiscard]] bool setCachedControlValue(Rml::Element* const el,
                                                 const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("control", "value");
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            if (auto* const input = dynamic_cast<Rml::ElementFormControlInput*>(el))
                input->SetValue(value);
            else
                el->SetAttribute("value", value);

            el->SetAttribute(attr_name, value);
            return true;
        }

        void invalidateCachedControlValue(Rml::Element* const el) {
            if (!el)
                return;

            const std::string attr_name = cacheAttrName("control", "value");
            el->RemoveAttribute(attr_name.c_str());
        }

        [[nodiscard]] bool setCachedDisabled(Rml::Element* const el, const bool disabled) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("attr", "disabled");
            const char* const value = disabled ? "1" : "0";
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            if (disabled)
                el->SetAttribute("disabled", "disabled");
            else
                el->RemoveAttribute("disabled");
            el->SetAttribute(attr_name, value);
            return true;
        }

        [[nodiscard]] bool setCachedSelect(Rml::ElementFormControlSelect* const el,
                                           const int selection) {
            if (!el || el->GetSelection() == selection)
                return false;
            el->SetSelection(selection);
            return true;
        }

        template <typename... Args>
        [[nodiscard]] std::string localizedFormat(const char* const key, Args... args) {
            char buffer[256]{};
            std::snprintf(buffer, sizeof(buffer), LOC(key), args...);
            return buffer;
        }

        [[nodiscard]] float readFloatValue(const Rml::Element* const el, const float fallback) {
            if (!el)
                return fallback;
            const std::string value = controlValue(el);
            char* end = nullptr;
            const float parsed = std::strtof(value.c_str(), &end);
            return end && end != value.c_str() && std::isfinite(parsed) ? parsed : fallback;
        }

        [[nodiscard]] int readIntValue(const Rml::Element* const el, const int fallback) {
            if (!el)
                return fallback;
            const std::string value = controlValue(el);
            char* end = nullptr;
            const long parsed = std::strtol(value.c_str(), &end, 10);
            if (!end || end == value.c_str())
                return fallback;
            return static_cast<int>(std::clamp<long>(parsed, 1, 65535));
        }

    } // namespace

    VideoExtractorDialog::VideoExtractorDialog()
        : player_(std::make_unique<io::VideoPlayer>()) {
        listener_.owner = this;
    }

    VideoExtractorDialog::~VideoExtractorDialog() {
        joinExtractionThread();
        preview_texture_.reset();
    }

    void VideoExtractorDialog::shutdown() {
        joinExtractionThread();
        preview_texture_.reset();
        host_.reset();
        document_ = nullptr;
        clearElementCache();
    }

    bool VideoExtractorDialog::isVideoPlaying() const {
        return player_ && player_->isPlaying();
    }

    void VideoExtractorDialog::preloadDirect(const float w, const float h,
                                             const lfs::vis::gui::PanelDrawContext& ctx,
                                             const float clip_y_min,
                                             const float clip_y_max,
                                             const lfs::vis::gui::PanelInputState* input) {
        setInputClipY(clip_y_min, clip_y_max);
        setInput(input);
        if (!ensureInitialized(ctx.ui ? ctx.ui->rml_manager : nullptr))
            return;

        syncPanel();
        host_->prepareDirect(w, h);
    }

    void VideoExtractorDialog::drawDirect(const float x, const float y,
                                          const float w, const float h,
                                          const lfs::vis::gui::PanelDrawContext& ctx) {
        if (!ensureInitialized(ctx.ui ? ctx.ui->rml_manager : nullptr))
            return;

        syncPanel();
        host_->drawDirect(x, y, w, h);
    }

    bool VideoExtractorDialog::drawDirectCached(const float x, const float y,
                                                const float w, const float h,
                                                const lfs::vis::gui::PanelDrawContext& ctx) {
        if (!ensureInitialized(ctx.ui ? ctx.ui->rml_manager : nullptr))
            return false;

        syncPanel();
        if (hasDynamicState() || host_->needsAnimationFrame()) {
            host_->drawDirect(x, y, w, h);
            return true;
        }
        return host_->drawDirectCached(x, y, w, h);
    }

    float VideoExtractorDialog::getDirectDrawHeight() const {
        return host_ ? host_->getContentHeight() : 0.0f;
    }

    void VideoExtractorDialog::setInputClipY(const float y_min, const float y_max) {
        if (host_)
            host_->setInputClipY(y_min, y_max);
    }

    void VideoExtractorDialog::setInput(const lfs::vis::gui::PanelInputState* input) {
        if (host_)
            host_->setInput(input);
    }

    void VideoExtractorDialog::setForcedHeight(const float h) {
        if (host_)
            host_->setForcedHeight(h);
    }

    bool VideoExtractorDialog::wantsKeyboard() const {
        return host_ && host_->wantsKeyboard();
    }

    bool VideoExtractorDialog::needsAnimationFrame() const {
        return hasDynamicState() || (host_ && host_->needsAnimationFrame());
    }

    void VideoExtractorDialog::reloadRmlResources() {
        if (host_)
            host_->reloadDocument();
        document_ = nullptr;
        clearElementCache();
        elements_cached_ = false;
        last_language_.clear();
        controls_dirty_ = true;
    }

    void VideoExtractorDialog::startExtraction(const VideoExtractionParams& params) {
        joinExtractionThread();

        extraction_thread_.emplace([this, params]() {
            io::VideoFrameExtractor extractor;

            io::VideoFrameExtractor::Params extract_params;
            extract_params.video_path = params.video_path;
            extract_params.output_dir = params.output_dir;
            extract_params.mode = params.mode;
            extract_params.fps = params.fps;
            extract_params.frame_interval = params.frame_interval;
            extract_params.format = params.format;
            extract_params.jpg_quality = params.jpg_quality;
            extract_params.start_time = params.start_time;
            extract_params.end_time = params.end_time;
            extract_params.resolution_mode = params.resolution_mode;
            extract_params.scale = params.scale;
            extract_params.custom_width = params.custom_width;
            extract_params.custom_height = params.custom_height;
            extract_params.filename_pattern = params.filename_pattern;
            extract_params.cancel_requested = [this]() {
                return stop_extraction_requested_.load();
            };

            extract_params.progress_callback = [this](const int current, const int total) {
                updateProgress(current, total);
            };

            std::string error;
            if (!extractor.extract(extract_params, error)) {
                if (stop_extraction_requested_.load()) {
                    LOG_INFO("Video frame extraction stopped");
                    setExtractionStopped();
                } else {
                    LOG_ERROR("Video frame extraction failed: {}", error);
                    setExtractionError(error);
                }
            } else {
                LOG_INFO("Video frame extraction completed successfully");
                setExtractionComplete();
            }
        });
    }

    void VideoExtractorDialog::joinExtractionThread() {
        if (extraction_thread_ && extraction_thread_->joinable()) {
            if (extracting_.load())
                stop_extraction_requested_.store(true);
            extraction_thread_->join();
        }
        extraction_thread_.reset();
        stop_extraction_requested_.store(false);
    }

    void VideoExtractorDialog::updateProgress(const int current, const int total) {
        current_frame_.store(current);
        total_frames_.store(total);
        extraction_status_dirty_.store(true);
    }

    void VideoExtractorDialog::setExtractionComplete() {
        extracting_.store(false);
        stop_extraction_requested_.store(false);
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        status_message_ = ExtractionStatusMessage::Complete;
        extraction_status_dirty_.store(true);
    }

    void VideoExtractorDialog::setExtractionStopped() {
        extracting_.store(false);
        stop_extraction_requested_.store(false);
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        status_message_ = ExtractionStatusMessage::Stopped;
        extraction_status_dirty_.store(true);
    }

    void VideoExtractorDialog::setExtractionError(const std::string& error) {
        extracting_.store(false);
        stop_extraction_requested_.store(false);
        std::lock_guard lock(extraction_status_mutex_);
        error_message_ = error;
        status_message_ = ExtractionStatusMessage::None;
        extraction_status_dirty_.store(true);
    }

    VideoExtractorDialog::ExtractionStatusSnapshot VideoExtractorDialog::getExtractionStatusSnapshot() const {
        std::lock_guard lock(extraction_status_mutex_);
        return {
            .error_message = error_message_,
            .status_message = status_message_,
        };
    }

    void VideoExtractorDialog::clearExtractionStatus() {
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        status_message_ = ExtractionStatusMessage::None;
        markContentDirty();
    }

    void VideoExtractorDialog::clearStatusMessage() {
        std::lock_guard lock(extraction_status_mutex_);
        status_message_ = ExtractionStatusMessage::None;
        markContentDirty();
    }

    void VideoExtractorDialog::clearErrorMessage() {
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        markContentDirty();
    }

    bool VideoExtractorDialog::openVideo(const std::filesystem::path& path) {
        if (!player_->open(path)) {
            setExtractionError(std::format("Failed to open {}", lfs::core::path_to_utf8(path)));
            return false;
        }

        video_path_ = path;
        trim_start_ = 0.0f;
        trim_end_ = static_cast<float>(player_->duration());
        custom_width_ = std::max(16, player_->sourceWidth());
        custom_height_ = std::max(16, player_->sourceHeight());
        texture_needs_update_ = true;
        preview_src_.clear();

        if (output_dir_.empty()) {
            std::filesystem::path output_name = video_path_.stem();
            output_name += "_frames";
            output_dir_ = video_path_.parent_path() / output_name;
        }

        controls_dirty_ = true;
        clearExtractionStatus();
        markContentDirty();
        return true;
    }

    int VideoExtractorDialog::calculateEstimatedFrames() const {
        if (!player_ || !player_->isOpen())
            return 0;

        const double duration = trimDuration();
        if (duration <= 0.0)
            return 0;

        if (mode_selection_ == 0)
            return static_cast<int>(std::ceil(duration * static_cast<double>(fps_)));

        const double video_fps = std::max(player_->fps(), 0.001);
        return static_cast<int>(std::ceil((duration * video_fps) /
                                          static_cast<double>(std::max(1, frame_interval_))));
    }

    void VideoExtractorDialog::updatePreviewTexture() {
        if (!player_->isOpen())
            return;

        const uint8_t* const data = player_->currentFrameData();
        if (!data)
            return;

        const int width = player_->width();
        const int height = player_->height();
        if (!preview_texture_)
            preview_texture_ = std::make_unique<lfs::vis::gui::VulkanUiTexture>();

        if (preview_texture_->upload(data, width, height, 3)) {
            preview_texture_width_ = width;
            preview_texture_height_ = height;
        } else {
            preview_texture_.reset();
            preview_texture_width_ = 0;
            preview_texture_height_ = 0;
            preview_src_.clear();
        }
        texture_needs_update_ = false;
    }

    bool VideoExtractorDialog::ensureInitialized(lfs::vis::gui::RmlUIManager* manager) {
        if (!manager)
            return false;

        if (!host_) {
            host_ = std::make_unique<lfs::vis::gui::RmlPanelHost>(
                manager, "video_extractor", "rmlui/video_extractor.rml");
            host_->setHeightMode(lfs::vis::gui::PanelHeightMode::Fill);
            host_->setForeground(true);
        }

        if (!host_->ensureDocumentLoaded())
            return false;

        if (document_ && elements_cached_)
            return true;

        document_ = host_->getDocument();
        if (!document_) {
            LOG_ERROR("VideoExtractorDialog: missing Rml document");
            return false;
        }

        cacheElements();
        return elements_cached_;
    }

    void VideoExtractorDialog::clearElementCache() {
        title_el_ = nullptr;
        close_btn_el_ = nullptr;
        preview_shell_el_ = nullptr;
        preview_image_el_ = nullptr;
        preview_empty_el_ = nullptr;
        step_back_btn_el_ = nullptr;
        play_btn_el_ = nullptr;
        play_icon_el_ = nullptr;
        step_forward_btn_el_ = nullptr;
        time_label_el_ = nullptr;
        timeline_el_ = nullptr;
        timeline_trim_el_ = nullptr;
        timeline_progress_el_ = nullptr;
        timeline_playhead_el_ = nullptr;
        timeline_start_el_ = nullptr;
        timeline_end_el_ = nullptr;
        timeline_markers_el_ = nullptr;
        trim_start_input_el_ = nullptr;
        trim_end_input_el_ = nullptr;
        trim_start_set_el_ = nullptr;
        trim_end_set_el_ = nullptr;
        trim_reset_el_ = nullptr;
        estimated_frames_el_ = nullptr;
        video_value_el_ = nullptr;
        output_value_el_ = nullptr;
        browse_video_el_ = nullptr;
        browse_output_el_ = nullptr;
        mode_select_el_ = nullptr;
        fps_row_el_ = nullptr;
        fps_slider_el_ = nullptr;
        fps_value_el_ = nullptr;
        interval_row_el_ = nullptr;
        interval_input_el_ = nullptr;
        interval_value_el_ = nullptr;
        format_select_el_ = nullptr;
        quality_row_el_ = nullptr;
        quality_slider_el_ = nullptr;
        quality_value_el_ = nullptr;
        resolution_select_el_ = nullptr;
        scale_row_el_ = nullptr;
        scale_select_el_ = nullptr;
        custom_resolution_row_el_ = nullptr;
        custom_width_input_el_ = nullptr;
        custom_height_input_el_ = nullptr;
        output_resolution_el_ = nullptr;
        pattern_input_el_ = nullptr;
        pattern_example_el_ = nullptr;
        start_btn_el_ = nullptr;
        stop_btn_el_ = nullptr;
        cancel_btn_el_ = nullptr;
        select_hint_el_ = nullptr;
        progress_section_el_ = nullptr;
        progress_text_el_ = nullptr;
        progress_bar_el_ = nullptr;
        complete_section_el_ = nullptr;
        complete_text_el_ = nullptr;
        ok_btn_el_ = nullptr;
        stopped_section_el_ = nullptr;
        stopped_text_el_ = nullptr;
        stopped_ok_btn_el_ = nullptr;
        error_section_el_ = nullptr;
        error_text_el_ = nullptr;
        dismiss_btn_el_ = nullptr;
        elements_cached_ = false;
    }

    void VideoExtractorDialog::cacheElements() {
        clearElementCache();

        title_el_ = document_->GetElementById("title-text");
        close_btn_el_ = document_->GetElementById("close-btn");
        preview_shell_el_ = document_->GetElementById("preview-shell");
        preview_image_el_ = document_->GetElementById("preview-image");
        preview_empty_el_ = document_->GetElementById("preview-empty");
        step_back_btn_el_ = document_->GetElementById("btn-step-back");
        play_btn_el_ = document_->GetElementById("btn-play");
        play_icon_el_ = document_->GetElementById("play-icon");
        step_forward_btn_el_ = document_->GetElementById("btn-step-forward");
        time_label_el_ = document_->GetElementById("time-label");
        timeline_el_ = document_->GetElementById("timeline");
        timeline_trim_el_ = document_->GetElementById("timeline-trim");
        timeline_progress_el_ = document_->GetElementById("timeline-progress");
        timeline_playhead_el_ = document_->GetElementById("timeline-playhead");
        timeline_start_el_ = document_->GetElementById("timeline-start");
        timeline_end_el_ = document_->GetElementById("timeline-end");
        timeline_markers_el_ = document_->GetElementById("timeline-markers");
        trim_start_input_el_ = document_->GetElementById("trim-start-input");
        trim_end_input_el_ = document_->GetElementById("trim-end-input");
        trim_start_set_el_ = document_->GetElementById("btn-trim-start-set");
        trim_end_set_el_ = document_->GetElementById("btn-trim-end-set");
        trim_reset_el_ = document_->GetElementById("btn-trim-reset");
        estimated_frames_el_ = document_->GetElementById("estimated-frames");
        video_value_el_ = document_->GetElementById("video-value");
        output_value_el_ = document_->GetElementById("output-value");
        browse_video_el_ = document_->GetElementById("btn-browse-video");
        browse_output_el_ = document_->GetElementById("btn-browse-output");
        mode_select_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("mode-select"));
        fps_row_el_ = document_->GetElementById("fps-row");
        fps_slider_el_ = document_->GetElementById("fps-slider");
        fps_value_el_ = document_->GetElementById("fps-value");
        interval_row_el_ = document_->GetElementById("interval-row");
        interval_input_el_ = document_->GetElementById("interval-input");
        interval_value_el_ = document_->GetElementById("interval-value");
        format_select_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("format-select"));
        quality_row_el_ = document_->GetElementById("quality-row");
        quality_slider_el_ = document_->GetElementById("quality-slider");
        quality_value_el_ = document_->GetElementById("quality-value");
        resolution_select_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("resolution-select"));
        scale_row_el_ = document_->GetElementById("scale-row");
        scale_select_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("scale-select"));
        custom_resolution_row_el_ = document_->GetElementById("custom-resolution-row");
        custom_width_input_el_ = document_->GetElementById("custom-width-input");
        custom_height_input_el_ = document_->GetElementById("custom-height-input");
        output_resolution_el_ = document_->GetElementById("output-resolution");
        pattern_input_el_ = document_->GetElementById("pattern-input");
        pattern_example_el_ = document_->GetElementById("pattern-example");
        start_btn_el_ = document_->GetElementById("btn-start");
        stop_btn_el_ = document_->GetElementById("btn-stop");
        cancel_btn_el_ = document_->GetElementById("btn-cancel");
        select_hint_el_ = document_->GetElementById("select-hint");
        progress_section_el_ = document_->GetElementById("progress-section");
        progress_text_el_ = document_->GetElementById("progress-text");
        progress_bar_el_ = document_->GetElementById("progress-bar");
        complete_section_el_ = document_->GetElementById("complete-section");
        complete_text_el_ = document_->GetElementById("complete-text");
        ok_btn_el_ = document_->GetElementById("btn-complete-ok");
        stopped_section_el_ = document_->GetElementById("stopped-section");
        stopped_text_el_ = document_->GetElementById("stopped-text");
        stopped_ok_btn_el_ = document_->GetElementById("btn-stopped-ok");
        error_section_el_ = document_->GetElementById("error-section");
        error_text_el_ = document_->GetElementById("error-text");
        dismiss_btn_el_ = document_->GetElementById("btn-error-dismiss");

        elements_cached_ =
            title_el_ && close_btn_el_ && preview_shell_el_ && preview_image_el_ &&
            preview_empty_el_ && step_back_btn_el_ && play_btn_el_ && play_icon_el_ &&
            step_forward_btn_el_ && time_label_el_ && timeline_el_ && timeline_trim_el_ &&
            timeline_progress_el_ && timeline_playhead_el_ && timeline_start_el_ &&
            timeline_end_el_ && timeline_markers_el_ && trim_start_input_el_ &&
            trim_end_input_el_ && trim_start_set_el_ && trim_end_set_el_ && trim_reset_el_ &&
            estimated_frames_el_ && video_value_el_ && output_value_el_ && browse_video_el_ &&
            browse_output_el_ && mode_select_el_ && fps_row_el_ && fps_slider_el_ &&
            fps_value_el_ && interval_row_el_ && interval_input_el_ && interval_value_el_ &&
            format_select_el_ && quality_row_el_ && quality_slider_el_ && quality_value_el_ &&
            resolution_select_el_ && scale_row_el_ && scale_select_el_ &&
            custom_resolution_row_el_ && custom_width_input_el_ && custom_height_input_el_ &&
            output_resolution_el_ && pattern_input_el_ && pattern_example_el_ &&
            start_btn_el_ && stop_btn_el_ && cancel_btn_el_ && select_hint_el_ && progress_section_el_ &&
            progress_text_el_ && progress_bar_el_ && complete_section_el_ &&
            complete_text_el_ && ok_btn_el_ && stopped_section_el_ && stopped_text_el_ &&
            stopped_ok_btn_el_ && error_section_el_ && error_text_el_ && dismiss_btn_el_;

        if (!elements_cached_) {
            LOG_ERROR("VideoExtractorDialog: missing required Rml elements");
            return;
        }

        bindEventListeners();
        controls_dirty_ = true;
    }

    void VideoExtractorDialog::bindEventListeners() {
        const auto listen_click = [this](Rml::Element* const el) {
            if (el)
                el->AddEventListener(Rml::EventId::Click, &listener_);
        };
        const auto listen_change = [this](Rml::Element* const el) {
            if (el) {
                el->AddEventListener(Rml::EventId::Change, &listener_);
                el->AddEventListener(Rml::EventId::Blur, &listener_);
            }
        };
        const auto listen_input = [this](Rml::Element* const el) {
            if (el)
                el->AddEventListener("input", &listener_);
        };

        listen_click(close_btn_el_);
        listen_click(step_back_btn_el_);
        listen_click(play_btn_el_);
        listen_click(step_forward_btn_el_);
        listen_click(trim_start_set_el_);
        listen_click(trim_end_set_el_);
        listen_click(trim_reset_el_);
        listen_click(browse_video_el_);
        listen_click(browse_output_el_);
        listen_click(start_btn_el_);
        listen_click(stop_btn_el_);
        listen_click(cancel_btn_el_);
        listen_click(ok_btn_el_);
        listen_click(stopped_ok_btn_el_);
        listen_click(dismiss_btn_el_);

        listen_change(mode_select_el_);
        listen_change(fps_slider_el_);
        listen_input(fps_slider_el_);
        listen_change(interval_input_el_);
        listen_change(format_select_el_);
        listen_change(quality_slider_el_);
        listen_input(quality_slider_el_);
        listen_change(resolution_select_el_);
        listen_change(scale_select_el_);
        listen_change(custom_width_input_el_);
        listen_change(custom_height_input_el_);
        listen_change(pattern_input_el_);
        listen_change(trim_start_input_el_);
        listen_change(trim_end_input_el_);

        timeline_el_->AddEventListener(Rml::EventId::Mousedown, &listener_);
        if (auto* const body = document_->GetElementById("body")) {
            body->AddEventListener(Rml::EventId::Mousemove, &listener_);
            body->AddEventListener(Rml::EventId::Mouseup, &listener_);
        }
    }

    void VideoExtractorDialog::syncPanel() {
        if (!elements_cached_)
            return;

        syncLocale();
        syncVideoPreview();
        syncTimeline();
        syncControls();
        syncExtractionStatus();
        syncOutputPreview();

        if (controls_dirty_) {
            controls_dirty_ = false;
            markContentDirty();
        }
    }

    void VideoExtractorDialog::syncLocale() {
        const std::string language = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        if (language == last_language_)
            return;

        last_language_ = language;
        bool changed = false;
        changed |= setCachedText(title_el_, LOC(VideoExtractor::TITLE));
        changed |= setCachedAttribute(preview_empty_el_, "data-empty-text", LOC(VideoExtractor::SELECT_PREVIEW));
        changed |= setCachedText(preview_empty_el_, LOC(VideoExtractor::SELECT_PREVIEW));
        changed |= setCachedAttribute(step_back_btn_el_, "title", LOC(VideoExtractor::STEP_BACKWARD));
        changed |= setCachedAttribute(step_forward_btn_el_, "title", LOC(VideoExtractor::STEP_FORWARD));
        changed |= setCachedAttribute(trim_start_set_el_, "title", LOC(VideoExtractor::SET_START));
        changed |= setCachedAttribute(trim_end_set_el_, "title", LOC(VideoExtractor::SET_END));
        changed |= setCachedAttribute(browse_output_el_, "title", LOC(VideoExtractor::SELECT_FOLDER));
        changed |= setCachedAttribute(fps_slider_el_, "title", LOC(VideoExtractor::FPS_TOOLTIP));
        changed |= setCachedAttribute(interval_input_el_, "title", LOC(VideoExtractor::INTERVAL_TOOLTIP));
        changed |= setCachedAttribute(quality_slider_el_, "title", LOC(VideoExtractor::QUALITY_LABEL));
        changed |= setCachedAttribute(custom_width_input_el_, "title", LOC(VideoExtractor::WIDTH));
        changed |= setCachedAttribute(custom_height_input_el_, "title", LOC(VideoExtractor::HEIGHT));
        changed |= setCachedAttribute(pattern_input_el_, "title", LOC(VideoExtractor::PATTERN_TOOLTIP));
        changed |= setCachedText(stop_btn_el_, LOC(VideoExtractor::STOP));
        changed |= setCachedText(cancel_btn_el_, LOC(VideoExtractor::CANCEL));
        markContentDirty();
        controls_dirty_ |= changed;
    }

    void VideoExtractorDialog::syncVideoPreview() {
        bool changed = false;
        if (player_->isOpen()) {
            if (player_->update(0.0))
                texture_needs_update_ = true;
            if (texture_needs_update_)
                updatePreviewTexture();
        }

        const auto shell_region = preview_shell_el_->GetBox().GetSize(Rml::BoxArea::Content);
        if (shell_region.x > 8.0f) {
            const float preview_h = std::clamp(shell_region.x * 9.0f / 16.0f,
                                               150.0f,
                                               360.0f);
            const std::string preview_h_css = std::format("{:.1f}px", preview_h);
            changed |= setCachedProperty(preview_shell_el_, "height", preview_h_css);
            changed |= setCachedProperty(preview_shell_el_, "min-height", preview_h_css);
        }

        const bool has_preview = player_->isOpen() && preview_texture_ && preview_texture_->valid() &&
                                 preview_texture_width_ > 0 && preview_texture_height_ > 0;

        changed |= setCachedProperty(preview_empty_el_, "display", has_preview ? "none" : "block");
        changed |= setCachedProperty(preview_image_el_, "display", has_preview ? "block" : "none");

        if (has_preview) {
            const std::string src = preview_texture_->rmlSrcUrl(preview_texture_width_,
                                                                preview_texture_height_);
            if (!src.empty() && src != preview_src_) {
                preview_image_el_->SetAttribute("src", src);
                preview_src_ = src;
                changed = true;
            }

            const auto region = preview_shell_el_->GetBox().GetSize(Rml::BoxArea::Content);
            if (region.x > 8.0f && region.y > 8.0f) {
                const float video_aspect =
                    static_cast<float>(preview_texture_width_) /
                    static_cast<float>(std::max(1, preview_texture_height_));
                const float region_aspect = region.x / std::max(1.0f, region.y);
                float display_w = region.x;
                float display_h = region.y;
                if (video_aspect > region_aspect) {
                    display_w = region.x;
                    display_h = region.x / video_aspect;
                } else {
                    display_h = region.y;
                    display_w = region.y * video_aspect;
                }
                changed |= setCachedProperty(preview_image_el_, "width",
                                             std::format("{:.1f}px", display_w));
                changed |= setCachedProperty(preview_image_el_, "height",
                                             std::format("{:.1f}px", display_h));
            }
        } else if (!preview_src_.empty()) {
            preview_src_.clear();
            preview_image_el_->SetAttribute("src", "");
            changed = true;
        }

        const char* const play_icon = player_->isPlaying()
                                          ? "../icon/sequencer/pause.png"
                                          : "../icon/sequencer/play.png";
        changed |= setCachedAttribute(play_icon_el_, "src", play_icon);
        changed |= setCachedAttribute(play_btn_el_, "title",
                                      player_->isPlaying() ? LOC(VideoExtractor::PAUSE)
                                                           : LOC(VideoExtractor::PLAY));

        if (changed)
            markContentDirty();
    }

    void VideoExtractorDialog::syncTimeline() {
        bool changed = false;
        const bool has_video = player_->isOpen();
        const double duration = has_video ? std::max(player_->duration(), 0.001) : 1.0;
        const double current = has_video ? std::clamp(player_->currentTime(), 0.0, duration) : 0.0;
        const float start = has_video ? std::clamp(trim_start_, 0.0f, static_cast<float>(duration)) : 0.0f;
        const float end = has_video
                              ? std::clamp(trim_end_ < 0.0f ? static_cast<float>(duration) : trim_end_,
                                           start, static_cast<float>(duration))
                              : 1.0f;

        changed |= setCachedText(time_label_el_, has_video
                                                     ? std::format("{} / {}", formatTime(current), formatTime(duration))
                                                     : "--:--.-- / --:--.--");

        const float start_pct = has_video ? (start / static_cast<float>(duration)) * 100.0f : 0.0f;
        const float end_pct = has_video ? (end / static_cast<float>(duration)) * 100.0f : 100.0f;
        const float current_pct = has_video ? (static_cast<float>(current / duration) * 100.0f) : 0.0f;
        const float trim_progress_pct = has_video
                                            ? std::clamp(current_pct, start_pct, end_pct)
                                            : 0.0f;
        changed |= setCachedProperty(timeline_trim_el_, "left", std::format("{:.3f}%", start_pct));
        changed |= setCachedProperty(timeline_trim_el_, "width", std::format("{:.3f}%", std::max(0.0f, end_pct - start_pct)));
        changed |= setCachedProperty(timeline_progress_el_, "left", std::format("{:.3f}%", start_pct));
        changed |= setCachedProperty(timeline_progress_el_, "width", std::format("{:.3f}%", std::max(0.0f, trim_progress_pct - start_pct)));
        changed |= setCachedProperty(timeline_playhead_el_, "left", std::format("{:.3f}%", current_pct));
        changed |= setCachedProperty(timeline_start_el_, "left", std::format("{:.3f}%", start_pct));
        changed |= setCachedProperty(timeline_end_el_, "left", std::format("{:.3f}%", end_pct));

        std::string markers;
        const int frame_count = calculateEstimatedFrames();
        if (has_video && frame_count > 0) {
            if (frame_count <= MAX_TIMELINE_MARKERS) {
                markers.reserve(static_cast<size_t>(frame_count) * 64);
                const double step = mode_selection_ == 0
                                        ? 1.0 / std::max(0.001, static_cast<double>(fps_))
                                        : static_cast<double>(std::max(1, frame_interval_)) /
                                              std::max(0.001, player_->fps());
                for (int i = 0; i < frame_count; ++i) {
                    const double time = static_cast<double>(start) + static_cast<double>(i) * step;
                    if (time > end)
                        break;
                    const double pct = (time / duration) * 100.0;
                    markers += std::format("<span class=\"timeline-marker\" style=\"left: {:.3f}%;\"></span>", pct);
                }
            } else {
                markers = std::format(
                    "<span class=\"timeline-marker-band\" style=\"left: {:.3f}%; width: {:.3f}%;\"></span>",
                    start_pct, std::max(0.0f, end_pct - start_pct));
            }
        }
        changed |= setCachedInnerRml(timeline_markers_el_, markers);

        changed |= setCachedControlValue(trim_start_input_el_, std::format("{:.1f}", start));
        changed |= setCachedControlValue(trim_end_input_el_, std::format("{:.1f}", end));
        changed |= setCachedText(estimated_frames_el_, localizedFormat(VideoExtractor::ESTIMATED_FRAMES, frame_count));

        if (changed)
            markContentDirty();
    }

    void VideoExtractorDialog::syncControls() {
        bool changed = false;
        const bool has_video = player_->isOpen();
        const bool extracting = extracting_.load();
        const bool stop_requested = stop_extraction_requested_.load();
        const bool can_start = has_video && !output_dir_.empty() && !extracting;

        changed |= setCachedDisabled(step_back_btn_el_, !has_video);
        changed |= setCachedDisabled(play_btn_el_, !has_video);
        changed |= setCachedDisabled(step_forward_btn_el_, !has_video);
        changed |= setCachedDisabled(timeline_el_, !has_video);
        changed |= setCachedDisabled(trim_start_input_el_, !has_video);
        changed |= setCachedDisabled(trim_end_input_el_, !has_video);
        changed |= setCachedDisabled(trim_start_set_el_, !has_video);
        changed |= setCachedDisabled(trim_end_set_el_, !has_video);
        changed |= setCachedDisabled(trim_reset_el_, !has_video);
        changed |= setCachedDisabled(start_btn_el_, !can_start);
        changed |= setCachedDisabled(stop_btn_el_, !extracting || stop_requested);
        changed |= setCachedDisabled(cancel_btn_el_, false);

        changed |= setCachedSelect(mode_select_el_, mode_selection_);
        changed |= setCachedSelect(format_select_el_, format_selection_);
        changed |= setCachedSelect(resolution_select_el_, resolution_mode_);
        changed |= setCachedSelect(scale_select_el_, scale_selection_);

        changed |= setCachedProperty(fps_row_el_, "display", mode_selection_ == 0 ? "flex" : "none");
        changed |= setCachedProperty(interval_row_el_, "display", mode_selection_ == 1 ? "flex" : "none");
        changed |= setCachedProperty(quality_row_el_, "display", format_selection_ == 1 ? "flex" : "none");
        changed |= setCachedProperty(scale_row_el_, "display", resolution_mode_ == 1 ? "flex" : "none");
        changed |= setCachedProperty(custom_resolution_row_el_, "display", resolution_mode_ == 2 ? "flex" : "none");
        changed |= setCachedProperty(stop_btn_el_, "display", extracting ? "block" : "none");

        changed |= setCachedControlValue(fps_slider_el_, std::format("{:.1f}", fps_));
        changed |= setCachedText(fps_value_el_, std::format("{:.1f} {}", fps_, LOC(VideoExtractor::FPS_LABEL)));
        changed |= setCachedControlValue(interval_input_el_, std::to_string(frame_interval_));
        changed |= setCachedText(interval_value_el_, LOC(VideoExtractor::FRAMES_UNIT));
        changed |= setCachedControlValue(quality_slider_el_, std::to_string(jpg_quality_));
        changed |= setCachedText(quality_value_el_, std::format("{}%", jpg_quality_));
        changed |= setCachedControlValue(custom_width_input_el_, std::to_string(custom_width_));
        changed |= setCachedControlValue(custom_height_input_el_, std::to_string(custom_height_));
        changed |= setCachedControlValue(pattern_input_el_, filename_pattern_.data());

        const std::string video_display = video_path_.empty()
                                              ? LOC(VideoExtractor::NO_FILE)
                                              : lfs::core::path_to_utf8(video_path_.filename());
        const std::string output_display = output_dir_.empty()
                                               ? LOC(VideoExtractor::NO_DIR)
                                               : lfs::core::path_to_utf8(output_dir_);
        changed |= setCachedText(video_value_el_, video_display);
        changed |= setCachedText(output_value_el_, output_display);
        changed |= setCachedProperty(select_hint_el_, "display", can_start ? "none" : "inline-block");
        changed |= setCachedText(select_hint_el_, LOC(VideoExtractor::SELECT_BOTH));

        if (changed)
            markContentDirty();
    }

    void VideoExtractorDialog::syncExtractionStatus() {
        bool changed = extraction_status_dirty_.exchange(false);
        const bool extracting = extracting_.load();
        const int current = current_frame_.load();
        const int total = total_frames_.load();

        changed |= setCachedProperty(progress_section_el_, "display", extracting ? "flex" : "none");
        if (extracting) {
            const float progress = total > 0 ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
            changed |= setCachedAttribute(progress_bar_el_, "value", std::format("{:.4f}", progress));
            changed |= setCachedText(progress_text_el_, total > 0
                                                            ? localizedFormat(VideoExtractor::EXTRACTING, current, total)
                                                            : LOC(VideoExtractor::STARTING));
        }

        const auto snapshot = getExtractionStatusSnapshot();
        changed |= setCachedProperty(complete_section_el_, "display",
                                     snapshot.status_message == ExtractionStatusMessage::Complete && !extracting
                                         ? "flex"
                                         : "none");
        if (snapshot.status_message == ExtractionStatusMessage::Complete && !extracting) {
            changed |= setCachedText(complete_text_el_,
                                     std::format("{} {}", LOC(VideoExtractor::COMPLETE),
                                                 localizedFormat(VideoExtractor::EXTRACTED, current)));
        }

        changed |= setCachedProperty(stopped_section_el_, "display",
                                     snapshot.status_message == ExtractionStatusMessage::Stopped && !extracting
                                         ? "flex"
                                         : "none");
        if (snapshot.status_message == ExtractionStatusMessage::Stopped && !extracting)
            changed |= setCachedText(stopped_text_el_, LOC(VideoExtractor::STOPPED));

        const bool has_error = !snapshot.error_message.empty();
        changed |= setCachedProperty(error_section_el_, "display", has_error ? "flex" : "none");
        if (has_error) {
            char buffer[512]{};
            std::snprintf(buffer, sizeof(buffer), LOC(VideoExtractor::ERROR_MSG), snapshot.error_message.c_str());
            changed |= setCachedText(error_text_el_, buffer);
        }

        if (changed)
            markContentDirty();
    }

    void VideoExtractorDialog::syncOutputPreview() {
        bool changed = false;
        int out_w = player_->isOpen() ? player_->sourceWidth() : 0;
        int out_h = player_->isOpen() ? player_->sourceHeight() : 0;

        if (player_->isOpen()) {
            if (resolution_mode_ == 1) {
                const float scale = SCALE_VALUES[std::clamp(scale_selection_, 0, static_cast<int>(SCALE_VALUES.size() - 1))];
                out_w = std::max(1, static_cast<int>(std::round(static_cast<float>(out_w) * scale)));
                out_h = std::max(1, static_cast<int>(std::round(static_cast<float>(out_h) * scale)));
            } else if (resolution_mode_ == 2) {
                out_w = custom_width_;
                out_h = custom_height_;
            }
        }

        changed |= setCachedText(output_resolution_el_,
                                 player_->isOpen()
                                     ? localizedFormat(VideoExtractor::OUTPUT_RES, out_w, out_h)
                                     : std::format("{} --", LOC(VideoExtractor::OUTPUT)));
        const char* const ext = format_selection_ == 0 ? ".png" : ".jpg";
        const std::string preview = io::formatFrameFilenameStem(filename_pattern_.data(), 1);
        changed |= setCachedText(pattern_example_el_,
                                 localizedFormat(VideoExtractor::EXAMPLE, preview.c_str(), ext));

        if (changed)
            markContentDirty();
    }

    void VideoExtractorDialog::EventListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->handleEvent(event);
    }

    void VideoExtractorDialog::handleEvent(Rml::Event& event) {
        auto* target = event.GetTargetElement();
        while (target && target->GetId().empty())
            target = target->GetParentNode();

        auto* current = event.GetCurrentElement();
        const std::string id = current && !current->GetId().empty()
                                   ? current->GetId()
                                   : (target ? target->GetId() : std::string{});

        const auto event_id = event.GetId();
        if (event_id == Rml::EventId::Mousedown ||
            event_id == Rml::EventId::Mousemove ||
            event_id == Rml::EventId::Mouseup) {
            handleTimelineEvent(event);
            return;
        }

        if (event_id == Rml::EventId::Click) {
            handleClick(id);
            event.StopPropagation();
            return;
        }

        if (event_id == Rml::EventId::Change || event_id == Rml::EventId::Blur ||
            event.GetType() == "input") {
            handleChange(id);
            event.StopPropagation();
        }
    }

    void VideoExtractorDialog::handleClick(const std::string& id) {
        if (id == "close-btn" || id == "btn-cancel") {
            disablePanel();
        } else if (id == "btn-stop") {
            requestStopExtraction();
        } else if (id == "btn-browse-video") {
            const auto path = lfs::vis::gui::OpenVideoFileDialog(video_path_);
            if (!path.empty())
                (void)openVideo(path);
        } else if (id == "btn-browse-output") {
            const auto path = lfs::vis::gui::PickFolderDialog(output_dir_);
            if (!path.empty()) {
                output_dir_ = path;
                clearExtractionStatus();
                controls_dirty_ = true;
                markContentDirty();
            }
        } else if (id == "btn-step-back" && player_->isOpen()) {
            player_->stepBackward();
            texture_needs_update_ = true;
            markContentDirty();
        } else if (id == "btn-play" && player_->isOpen()) {
            player_->togglePlayPause();
            markContentDirty();
        } else if (id == "btn-step-forward" && player_->isOpen()) {
            player_->stepForward();
            texture_needs_update_ = true;
            markContentDirty();
        } else if (id == "btn-trim-start-set" && player_->isOpen()) {
            trim_start_ = std::clamp(static_cast<float>(player_->currentTime()), 0.0f,
                                     trim_end_ - MIN_TRIM_SECONDS);
            controls_dirty_ = true;
            markContentDirty();
        } else if (id == "btn-trim-end-set" && player_->isOpen()) {
            trim_end_ = std::clamp(static_cast<float>(player_->currentTime()),
                                   trim_start_ + MIN_TRIM_SECONDS,
                                   static_cast<float>(player_->duration()));
            controls_dirty_ = true;
            markContentDirty();
        } else if (id == "btn-trim-reset" && player_->isOpen()) {
            trim_start_ = 0.0f;
            trim_end_ = static_cast<float>(player_->duration());
            controls_dirty_ = true;
            markContentDirty();
        } else if (id == "btn-start") {
            beginExtractionFromUi();
        } else if (id == "btn-complete-ok" || id == "btn-stopped-ok") {
            clearStatusMessage();
            current_frame_.store(0);
            total_frames_.store(0);
        } else if (id == "btn-error-dismiss") {
            clearErrorMessage();
        }
    }

    void VideoExtractorDialog::handleChange(const std::string& id) {
        Rml::Element* changed_control = nullptr;

        if (id == "mode-select") {
            mode_selection_ = mode_select_el_ ? mode_select_el_->GetSelection() : mode_selection_;
        } else if (id == "format-select") {
            format_selection_ = format_select_el_ ? format_select_el_->GetSelection() : format_selection_;
        } else if (id == "resolution-select") {
            resolution_mode_ = resolution_select_el_ ? resolution_select_el_->GetSelection() : resolution_mode_;
        } else if (id == "scale-select") {
            scale_selection_ = scale_select_el_ ? scale_select_el_->GetSelection() : scale_selection_;
        } else if (id == "fps-slider") {
            fps_ = std::clamp(readFloatValue(fps_slider_el_, fps_), 0.1f, 30.0f);
            changed_control = fps_slider_el_;
        } else if (id == "interval-input") {
            frame_interval_ = std::clamp(readIntValue(interval_input_el_, frame_interval_), 1, 100);
            changed_control = interval_input_el_;
        } else if (id == "quality-slider") {
            jpg_quality_ = std::clamp(readIntValue(quality_slider_el_, jpg_quality_), 50, 100);
            changed_control = quality_slider_el_;
        } else {
            applyTextInput(id);
            if (id == "trim-start-input")
                changed_control = trim_start_input_el_;
            else if (id == "trim-end-input")
                changed_control = trim_end_input_el_;
            else if (id == "custom-width-input")
                changed_control = custom_width_input_el_;
            else if (id == "custom-height-input")
                changed_control = custom_height_input_el_;
            else if (id == "pattern-input")
                changed_control = pattern_input_el_;
        }

        invalidateCachedControlValue(changed_control);
        controls_dirty_ = true;
        markContentDirty();
    }

    void VideoExtractorDialog::handleTimelineEvent(Rml::Event& event) {
        if (!player_->isOpen() || !timeline_el_)
            return;

        const auto type = event.GetId();
        if (type == Rml::EventId::Mousedown && event.GetCurrentElement() == timeline_el_) {
            if (event.GetParameter<int>("button", 0) != 0)
                return;

            const float mouse_x = event.GetParameter<float>("mouse_x", 0.0f);
            const float width = std::max(timeline_el_->GetBox().GetSize(Rml::BoxArea::Border).x, 1.0f);
            const float local_pct =
                std::clamp((mouse_x - timeline_el_->GetAbsoluteLeft()) / width, 0.0f, 1.0f);
            const float duration = static_cast<float>(std::max(player_->duration(), 0.001));
            const float time = local_pct * duration;
            const float start_dist = std::abs(time - trim_start_);
            const float end_dist = std::abs(time - trim_end_);
            const float handle_threshold = std::max(0.35f, duration * 0.015f);

            if (start_dist <= handle_threshold && start_dist <= end_dist)
                timeline_drag_target_ = TimelineDragTarget::TrimStart;
            else if (end_dist <= handle_threshold)
                timeline_drag_target_ = TimelineDragTarget::TrimEnd;
            else
                timeline_drag_target_ = TimelineDragTarget::Playhead;

            if (timeline_drag_target_ == TimelineDragTarget::Playhead)
                seekFromTimeline(mouse_x);
            else
                setTrimFromTimeline(timeline_drag_target_, mouse_x);
            event.StopPropagation();
        } else if (type == Rml::EventId::Mousemove &&
                   timeline_drag_target_ != TimelineDragTarget::None) {
            const float mouse_x = event.GetParameter<float>("mouse_x", 0.0f);
            if (timeline_drag_target_ == TimelineDragTarget::Playhead)
                seekFromTimeline(mouse_x);
            else
                setTrimFromTimeline(timeline_drag_target_, mouse_x);
            event.StopPropagation();
        } else if (type == Rml::EventId::Mouseup &&
                   timeline_drag_target_ != TimelineDragTarget::None) {
            timeline_drag_target_ = TimelineDragTarget::None;
            event.StopPropagation();
        }
    }

    void VideoExtractorDialog::seekFromTimeline(const float mouse_x) {
        if (!player_->isOpen() || !timeline_el_)
            return;
        const float width = std::max(timeline_el_->GetBox().GetSize(Rml::BoxArea::Border).x, 1.0f);
        const float t = std::clamp((mouse_x - timeline_el_->GetAbsoluteLeft()) / width, 0.0f, 1.0f);
        player_->seek(t * player_->duration());
        texture_needs_update_ = true;
        markContentDirty();
    }

    void VideoExtractorDialog::setTrimFromTimeline(const TimelineDragTarget target,
                                                   const float mouse_x) {
        if (!player_->isOpen() || !timeline_el_)
            return;
        const float duration = static_cast<float>(std::max(player_->duration(), 0.001));
        const float width = std::max(timeline_el_->GetBox().GetSize(Rml::BoxArea::Border).x, 1.0f);
        const float t = std::clamp((mouse_x - timeline_el_->GetAbsoluteLeft()) / width, 0.0f, 1.0f);
        const float time = t * duration;

        if (target == TimelineDragTarget::TrimStart)
            trim_start_ = std::clamp(time, 0.0f, trim_end_ - MIN_TRIM_SECONDS);
        else if (target == TimelineDragTarget::TrimEnd)
            trim_end_ = std::clamp(time, trim_start_ + MIN_TRIM_SECONDS, duration);

        controls_dirty_ = true;
        markContentDirty();
    }

    void VideoExtractorDialog::applyTextInput(const std::string& id) {
        if (id == "trim-start-input" && player_->isOpen()) {
            trim_start_ = std::clamp(readFloatValue(trim_start_input_el_, trim_start_), 0.0f,
                                     trim_end_ - MIN_TRIM_SECONDS);
        } else if (id == "trim-end-input" && player_->isOpen()) {
            trim_end_ = std::clamp(readFloatValue(trim_end_input_el_, trim_end_),
                                   trim_start_ + MIN_TRIM_SECONDS,
                                   static_cast<float>(player_->duration()));
        } else if (id == "custom-width-input") {
            custom_width_ = std::max(16, readIntValue(custom_width_input_el_, custom_width_));
        } else if (id == "custom-height-input") {
            custom_height_ = std::max(16, readIntValue(custom_height_input_el_, custom_height_));
        } else if (id == "pattern-input" && pattern_input_el_) {
            const std::string pattern = controlValue(pattern_input_el_);
            const std::string normalized = pattern.empty() ? "frame_%d" : pattern;
            std::fill(filename_pattern_.begin(), filename_pattern_.end(), '\0');
            std::strncpy(filename_pattern_.data(), normalized.c_str(), filename_pattern_.size() - 1);
        }
    }

    void VideoExtractorDialog::beginExtractionFromUi() {
        applyTextInput("trim-start-input");
        applyTextInput("trim-end-input");
        applyTextInput("custom-width-input");
        applyTextInput("custom-height-input");
        applyTextInput("pattern-input");

        if (!player_->isOpen() || output_dir_.empty() || extracting_.load())
            return;

        VideoExtractionParams params;
        params.video_path = video_path_;
        params.output_dir = output_dir_;
        params.mode = mode_selection_ == 0 ? io::ExtractionMode::FPS : io::ExtractionMode::INTERVAL;
        params.fps = static_cast<double>(fps_);
        params.frame_interval = frame_interval_;
        params.format = format_selection_ == 0 ? io::ImageFormat::PNG : io::ImageFormat::JPG;
        params.jpg_quality = jpg_quality_;
        params.start_time = static_cast<double>(trim_start_);
        params.end_time = static_cast<double>(trim_end_);

        static constexpr std::array<io::ResolutionMode, 3> RES_MODES{
            io::ResolutionMode::Original,
            io::ResolutionMode::Scale,
            io::ResolutionMode::Custom};
        params.resolution_mode = RES_MODES[std::clamp(resolution_mode_, 0, 2)];
        params.scale = SCALE_VALUES[std::clamp(scale_selection_, 0, static_cast<int>(SCALE_VALUES.size() - 1))];
        params.custom_width = custom_width_;
        params.custom_height = custom_height_;
        params.filename_pattern = filename_pattern_.data();

        stop_extraction_requested_.store(false);
        extracting_.store(true);
        current_frame_.store(0);
        total_frames_.store(0);
        clearExtractionStatus();
        startExtraction(params);
        controls_dirty_ = true;
        markContentDirty();
    }

    void VideoExtractorDialog::requestStopExtraction() {
        if (!extracting_.load())
            return;
        stop_extraction_requested_.store(true);
        controls_dirty_ = true;
        markContentDirty();
    }

    void VideoExtractorDialog::markContentDirty() {
        if (host_)
            host_->markContentDirty();
    }

    void VideoExtractorDialog::disablePanel() {
        lfs::vis::gui::PanelRegistry::instance().set_panel_enabled("native.video_extractor", false);
    }

    bool VideoExtractorDialog::hasDynamicState() const {
        return extracting_.load() ||
               extraction_status_dirty_.load() ||
               timeline_drag_target_ != TimelineDragTarget::None ||
               (player_ && player_->isPlaying());
    }

    double VideoExtractorDialog::trimDuration() const {
        if (!player_ || !player_->isOpen())
            return 0.0;
        const double duration = std::max(player_->duration(), 0.0);
        const double start = std::clamp(static_cast<double>(trim_start_), 0.0, duration);
        const double end = std::clamp(trim_end_ < 0.0f ? duration : static_cast<double>(trim_end_),
                                      start, duration);
        return std::max(0.0, end - start);
    }

} // namespace lfs::gui
