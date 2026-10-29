/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_extractor_dialog.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/include/core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/number_format.hpp"
#include "core/path_utils.hpp"
#include "gui/panel_height_mode.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "io/video/video_metadata.hpp"
#include "python/python_runtime.hpp"
#include <cctype>

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <set>
#include <string_view>
#include <unordered_map>

using namespace lichtfeld::Strings;

namespace lfs::gui {

    namespace {

        constexpr std::array<float, 4> SCALE_VALUES{0.25f, 0.5f, 0.75f, 1.0f};
        constexpr float MIN_TRIM_SECONDS = 0.1f;
        constexpr int MAX_TIMELINE_MARKERS = 180;

        [[nodiscard]] std::string outputDirectoryKey(std::string name) {
            std::transform(name.begin(), name.end(), name.begin(), [](const unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return name;
        }

        [[nodiscard]] float maximumExtractionFps(const io::VideoPlayer& player) {
            const double source_fps = player.isOpen() ? player.fps() : 0.0;
            if (!std::isfinite(source_fps) || source_fps <= 0.0 ||
                source_fps > std::numeric_limits<float>::max())
                return 30.0f;
            return std::max(0.1f, static_cast<float>(source_fps));
        }

        [[nodiscard]] float maximumRequestedExtractionFps(const io::VideoPlayer& player,
                                                          const bool has_batch,
                                                          const float requested_fps) {
            if (!has_batch)
                return maximumExtractionFps(player);

            // A batch has one shared requested rate. Do not let selecting a lower-rate clip
            // silently lower it; buildBatchParams clamps it independently for every source.
            return std::max({30.0f, requested_fps, maximumExtractionFps(player)});
        }

        [[nodiscard]] float sliderExtractionFps(const float value, const io::VideoPlayer& player,
                                                const bool has_batch, const float requested_fps) {
            const double rounded = std::round(static_cast<double>(value) * 10.0) / 10.0;
            return static_cast<float>(std::clamp(
                rounded, 0.1, static_cast<double>(maximumRequestedExtractionFps(player, has_batch, requested_fps))));
        }

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
            // A text field being typed in keeps its text; its blur writes the clamped value.
            if (auto* const document = el->GetOwnerDocument();
                document && document->GetFocusLeafNode() == el && el->GetAttribute<Rml::String>("type", "") == "text")
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

        [[nodiscard]] bool setCachedChecked(Rml::Element* const el, const bool checked) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("attr", "checked");
            const char* const value = checked ? "1" : "0";
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            if (checked)
                el->SetAttribute("checked", "checked");
            else
                el->RemoveAttribute("checked");
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

        template <size_t N>
        [[nodiscard]] bool setCachedSelectLabels(
            Rml::ElementFormControlSelect* const select,
            const std::array<const char*, N>& keys) {
            if (!select)
                return false;

            bool changed = false;
            for (size_t i = 0; i < N; ++i) {
                auto* const option = select->GetOption(static_cast<int>(i));
                if (!option)
                    continue;

                const std::string localized = LOC(keys[i]);
                if (option->GetInnerRML() == localized)
                    continue;

                option->SetInnerRML(localized);
                changed = true;
            }

            if (!changed)
                return false;

            auto* const selected = select->GetOption(select->GetSelection());
            if (!selected)
                return true;

            const std::string selected_rml = selected->GetInnerRML();
            for (int i = 0; i < select->GetNumChildren(true); ++i) {
                auto* const child = select->GetChild(i);
                if (child && child->GetTagName() == "selectvalue") {
                    child->SetInnerRML(selected_rml);
                    break;
                }
            }

            return true;
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

        [[nodiscard]] int readNonNegativeIntValue(const Rml::Element* const el, const int fallback) {
            if (!el)
                return fallback;
            const std::string value = controlValue(el);
            int parsed = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (error != std::errc{} || end != value.data() + value.size())
                return fallback;
            return std::max(0, parsed);
        }

        // Frames and, when requested, metadata an extraction writes into its output directory.
        // Other files,
        // such as photos already in the folder, are never part of the list.
        std::vector<std::filesystem::path> generatedExtractionFiles(const VideoExtractionParams& params) {
            std::vector<std::filesystem::path> files;
            const auto frames = preprocessing::video_frames_directory(params.output_dir, params.masking.enabled);
            const auto scan = [&](const std::filesystem::path& directory, const bool masks) {
                std::error_code ec;
                for (std::filesystem::directory_iterator it = lfs::core::native_fs::directory_iterator(directory, ec), end;
                     !ec && it != end; it.increment(ec)) {
                    std::error_code type_ec;
                    if (!it->is_regular_file(type_ec))
                        continue;
                    auto name = lfs::core::path_to_utf8(it->path().filename());
                    if (masks) {
                        if (!name.ends_with(".png"))
                            continue;
                        name.resize(name.size() - 4);
                    }
                    if ((!masks && params.generate_metadata && name == "extraction_metadata.json") ||
                        io::isGeneratedFrameFilename(params.filename_pattern, name, true))
                        files.push_back(lfs::core::native_fs::entry_path(directory, it->path()));
                }
            };
            scan(frames, false);
            if (params.masking.enabled)
                scan(params.output_dir / "masks", true);
            return files;
        }

    } // namespace

    VideoExtractorDialog::VideoExtractorDialog()
        : player_(std::make_unique<io::VideoPlayer>()) {
        listener_.owner = this;
    }

    VideoExtractorDialog::~VideoExtractorDialog() {
        joinBatchProbeThread();
        joinExtractionThread();
        preview_texture_.reset();
    }

    void VideoExtractorDialog::shutdown() {
        joinBatchProbeThread();
        joinExtractionThread();
        preview_texture_.reset();
        host_.reset();
        document_ = nullptr;
        clearElementCache();
    }

    bool VideoExtractorDialog::isVideoPlaying() const {
        return player_ && player_->isPlaying();
    }

    bool VideoExtractorDialog::openVideoPath(const std::filesystem::path& path) {
        if (!batch_items_.empty() || batch_probe_active_.load() || batch_probe_results_ready_.load()) {
            if (pending_params_set_)
                return false;
            return addBatchVideos({path});
        }
        return openVideo(path);
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

    void VideoExtractorDialog::setFloating(const bool floating) {
        floating_ = floating;
        if (host_)
            host_->setFloating(floating_);
    }

    bool VideoExtractorDialog::wantsKeyboard() const {
        return host_ && host_->wantsKeyboard();
    }

    SDL_Window* VideoExtractorDialog::nativeWindow() const {
        return host_ ? host_->nativeWindow() : nullptr;
    }

    bool VideoExtractorDialog::detachHost(float x, float y, float w, float h, bool desktop_coordinates) {
        if (!host_ || host_->nativeWindow() || !host_->externalAllowed())
            return false;
        auto* manager = host_->manager();
        preview_texture_.reset();
        preview_src_.clear();
        host_.reset();
        document_ = nullptr;
        clearElementCache();
        host_ = std::make_unique<lfs::vis::gui::RmlPanelHost>(manager, "video_extractor", "rmlui/video_extractor.rml");
        host_->setHeightMode(lfs::vis::gui::PanelHeightMode::Fill);
        host_->setForeground(true);
        host_->setFloating(floating_);
        if (!host_->configureExternal(x, y, w, h, desktop_coordinates) || !ensureInitialized(manager))
            return false;
        texture_needs_update_ = controls_dirty_ = true;
        syncPanel();
        return nativeWindow() != nullptr;
    }

    bool VideoExtractorDialog::detachToDesktop(float x, float y, float w, float h) {
        return detachHost(x, y, w, h, false);
    }

    bool VideoExtractorDialog::restoreDesktopWindow() {
        if (!host_ && desktop_reopen_bounds_ && desktop_manager_)
            ensureInitialized(desktop_manager_);
        if (desktop_restore_checked_ || !host_)
            return false;
        desktop_restore_checked_ = true;
        if (desktop_reopen_bounds_) {
            const auto bounds = *desktop_reopen_bounds_;
            desktop_reopen_bounds_.reset();
            return detachHost(bounds[0], bounds[1], bounds[2], bounds[3], true);
        }
        float bounds[4]{};
        return host_->restoredExternalBounds(bounds) && detachHost(bounds[0], bounds[1], bounds[2], bounds[3], true);
    }

    void VideoExtractorDialog::releaseDesktopWindow() {
        auto* native = nativeWindow();
        if (!native)
            return;
        int x = 0, y = 0, w = 0, h = 0;
        SDL_GetWindowPosition(native, &x, &y);
        SDL_GetWindowSize(native, &w, &h);
        desktop_reopen_bounds_ = std::array<float, 4>{float(x), float(y), float(w), float(h)};
        desktop_manager_ = host_->manager();
        preview_texture_.reset();
        preview_src_.clear();
        host_.reset();
        document_ = nullptr;
        clearElementCache();
        desktop_restore_checked_ = false;
        texture_needs_update_ = controls_dirty_ = true;
    }

    bool VideoExtractorDialog::needsAnimationFrame() const {
        const std::string current_language =
            lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        if (!current_language.empty() && current_language != last_language_)
            return true;
        return hasDynamicState() || (host_ && host_->needsAnimationFrame());
    }

    void VideoExtractorDialog::reloadRmlResources() {
        if (host_)
            host_->reloadDocument();
        document_ = nullptr;
        clearElementCache();
        elements_cached_ = false;
        preview_src_.clear();
        last_language_.clear();
        controls_dirty_ = true;
        batch_list_dirty_.store(true);
    }

    void VideoExtractorDialog::startExtraction(const VideoExtractionParams& params) {
        joinExtractionThread();

        extraction_thread_.emplace([this, params]() {
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
            extract_params.first_source_frame_number = params.first_source_frame_number;
            extract_params.sharpness.enabled = params.sharpness_enabled;
            extract_params.sharpness.algorithm = params.sharpness_algorithm;
            extract_params.sharpness.threshold = params.sharpness_threshold;
            extract_params.sharpness.window_candidates_target = params.window_candidates_target;
            extract_params.sharpness.window_mode = params.sharpness_window_mode;
            extract_params.generate_metadata = params.generate_metadata;
            extract_params.convert_hdr_to_sdr = params.convert_hdr_to_sdr;
            extract_params.rotation = params.rotation;
            extract_params.cancel_requested = [this]() {
                return stop_extraction_requested_.load();
            };

            extract_params.progress_callback = [this](const int current, const int total, const int discarded) {
                updateProgress(current, total, discarded);
            };

            std::string error, warning;
            auto masking = params.masking;
            masking.masking.progress = [this](const preprocessing::VideoMaskProgress& p) {
                masking_active_.store(true);
                updateProgress(static_cast<int>(p.completed), static_cast<int>(p.total), 0);
            };
            masking_active_.store(false);
            const auto outcome = preprocessing::extract_video_frames(extract_params, masking, error, &warning);
            masking_active_.store(false);
            if (outcome != io::ExtractionOutcome::Completed) {
                if (outcome == io::ExtractionOutcome::Cancelled) {
                    LOG_INFO("Video frame extraction stopped");
                    setExtractionStopped();
                } else {
                    LOG_ERROR("Video frame extraction failed: {}", error);
                    setExtractionError(error);
                }
            } else {
                setExtractionComplete(warning);
            }
        });
    }

    void VideoExtractorDialog::startBatchExtraction(std::vector<VideoExtractionParams> params) {
        joinExtractionThread();
        const bool continue_on_failure = continue_batch_on_failure_;
        extraction_thread_.emplace([this, params = std::move(params), continue_on_failure]() {
            const auto make_extractor_params = [this](const VideoExtractionParams& item,
                                                      const int item_index) {
                io::VideoFrameExtractor::Params extract_params;
                extract_params.video_path = item.video_path;
                extract_params.output_dir = item.output_dir;
                extract_params.mode = item.mode;
                extract_params.fps = item.fps;
                extract_params.frame_interval = item.frame_interval;
                extract_params.format = item.format;
                extract_params.jpg_quality = item.jpg_quality;
                extract_params.start_time = item.start_time;
                extract_params.end_time = item.end_time;
                extract_params.resolution_mode = item.resolution_mode;
                extract_params.scale = item.scale;
                extract_params.custom_width = item.custom_width;
                extract_params.custom_height = item.custom_height;
                extract_params.filename_pattern = item.filename_pattern;
                extract_params.first_source_frame_number = item.first_source_frame_number;
                extract_params.sharpness.enabled = item.sharpness_enabled;
                extract_params.sharpness.algorithm = item.sharpness_algorithm;
                extract_params.sharpness.threshold = item.sharpness_threshold;
                extract_params.sharpness.window_candidates_target = item.window_candidates_target;
                extract_params.sharpness.window_mode = item.sharpness_window_mode;
                extract_params.generate_metadata = item.generate_metadata;
                extract_params.convert_hdr_to_sdr = item.convert_hdr_to_sdr;
                extract_params.rotation = item.rotation;
                extract_params.cancel_requested = [this]() { return stop_extraction_requested_.load(); };
                extract_params.progress_callback = [this, item_index](const int current, const int total,
                                                                      const int discarded) {
                    active_batch_index_.store(item_index);
                    updateProgress(current, total, discarded);
                };
                return extract_params;
            };

            std::string warnings;
            for (std::size_t i = 0; i < params.size(); ++i) {
                if (stop_extraction_requested_.load()) {
                    {
                        std::lock_guard lock(batch_mutex_);
                        for (std::size_t remaining = i; remaining < batch_items_.size(); ++remaining)
                            batch_items_[remaining].status = BatchItemStatus::Stopped;
                    }
                    batch_list_dirty_.store(true);
                    setExtractionStopped();
                    return;
                }
                active_batch_index_.store(static_cast<int>(i));
                current_frame_.store(0);
                total_frames_.store(0);
                discarded_frames_.store(0);
                {
                    std::lock_guard lock(batch_mutex_);
                    if (i < batch_items_.size()) {
                        batch_items_[i].status = BatchItemStatus::Running;
                        batch_items_[i].error.clear();
                        batch_items_[i].warning.clear();
                    }
                }
                batch_list_dirty_.store(true);
                extraction_status_dirty_.store(true);

                std::string error, warning;
                auto masking = params[i].masking;
                masking.masking.progress = [this](const preprocessing::VideoMaskProgress& p) {
                    masking_active_.store(true);
                    updateProgress(static_cast<int>(p.completed), static_cast<int>(p.total), 0);
                };
                masking_active_.store(false);
                const auto outcome = preprocessing::extract_video_frames(
                    make_extractor_params(params[i], static_cast<int>(i)), masking, error, &warning);
                masking_active_.store(false);
                if (outcome == io::ExtractionOutcome::Completed) {
                    std::lock_guard lock(batch_mutex_);
                    if (i < batch_items_.size()) {
                        batch_items_[i].status = BatchItemStatus::Complete;
                        batch_items_[i].warning = warning;
                    }
                    if (!warning.empty()) {
                        if (!warnings.empty())
                            warnings += "\n";
                        warnings += lfs::core::path_to_utf8(params[i].video_path.filename()) + ": " + warning;
                    }
                    completed_batch_items_.fetch_add(1);
                    batch_list_dirty_.store(true);
                    continue;
                }

                if (outcome == io::ExtractionOutcome::Cancelled ||
                    stop_extraction_requested_.load()) {
                    {
                        std::lock_guard lock(batch_mutex_);
                        if (i < batch_items_.size())
                            batch_items_[i].status = BatchItemStatus::Stopped;
                        for (std::size_t remaining = i + 1; remaining < batch_items_.size(); ++remaining)
                            batch_items_[remaining].status = BatchItemStatus::Stopped;
                    }
                    batch_list_dirty_.store(true);
                    setExtractionStopped();
                    return;
                }

                LOG_ERROR("Batch video frame extraction failed for {}: {}",
                          lfs::core::path_to_utf8(params[i].video_path), error);
                {
                    std::lock_guard lock(batch_mutex_);
                    if (i < batch_items_.size()) {
                        batch_items_[i].status = BatchItemStatus::Failed;
                        batch_items_[i].error = error;
                    }
                }
                failed_batch_items_.fetch_add(1);
                batch_list_dirty_.store(true);
                if (!continue_on_failure) {
                    {
                        std::lock_guard lock(batch_mutex_);
                        for (std::size_t remaining = i + 1; remaining < batch_items_.size(); ++remaining)
                            batch_items_[remaining].status = BatchItemStatus::Stopped;
                    }
                    batch_list_dirty_.store(true);
                    setExtractionError(error);
                    return;
                }
            }
            setExtractionComplete(warnings);
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

    void VideoExtractorDialog::updateProgress(const int current, const int total, const int discarded) {
        current_frame_.store(current);
        total_frames_.store(total);
        discarded_frames_.store(discarded);
        extraction_status_dirty_.store(true);
    }

    void VideoExtractorDialog::setExtractionComplete(const std::string& warning) {
        extracting_.store(false);
        stop_extraction_requested_.store(false);
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        warning_message_ = warning;
        status_message_ = ExtractionStatusMessage::Complete;
        extraction_status_dirty_.store(true);
    }

    void VideoExtractorDialog::setExtractionStopped() {
        extracting_.store(false);
        stop_extraction_requested_.store(false);
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        warning_message_.clear();
        status_message_ = ExtractionStatusMessage::Stopped;
        extraction_status_dirty_.store(true);
    }

    void VideoExtractorDialog::setExtractionError(const std::string& error,
                                                  const bool extraction_failure) {
        if (extraction_failure) {
            extracting_.store(false);
            stop_extraction_requested_.store(false);
        }
        std::lock_guard lock(extraction_status_mutex_);
        error_message_ = error;
        warning_message_.clear();
        status_message_ = ExtractionStatusMessage::None;
        extraction_status_dirty_.store(true);
    }

    VideoExtractorDialog::ExtractionStatusSnapshot VideoExtractorDialog::getExtractionStatusSnapshot() const {
        std::lock_guard lock(extraction_status_mutex_);
        return {
            .error_message = error_message_,
            .warning_message = warning_message_,
            .status_message = status_message_,
        };
    }

    void VideoExtractorDialog::clearExtractionStatus() {
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        warning_message_.clear();
        status_message_ = ExtractionStatusMessage::None;
        markContentDirty();
    }

    void VideoExtractorDialog::clearStatusMessage() {
        std::lock_guard lock(extraction_status_mutex_);
        warning_message_.clear();
        status_message_ = ExtractionStatusMessage::None;
        markContentDirty();
    }

    void VideoExtractorDialog::clearErrorMessage() {
        std::lock_guard lock(extraction_status_mutex_);
        error_message_.clear();
        markContentDirty();
    }

    bool VideoExtractorDialog::openVideo(const std::filesystem::path& path) {
        if (extracting_.load() || pending_params_set_)
            return false;

        // A finished worker must publish its status before the new source clears it.
        joinExtractionThread();
        if (!player_->open(path)) {
            video_path_.clear();
            output_dir_.clear();
            controls_dirty_ = true;
            syncHdrControls();
            setExtractionError(std::format("Failed to open {}", lfs::core::path_to_utf8(path)));
            return false;
        }

        const bool video_path_changed = path != video_path_;
        video_path_ = path;
        trim_start_ = 0.0f;
        const double duration = player_->duration();
        trim_end_ = duration > 0.0 ? static_cast<float>(duration) : -1.0f;
        trim_end_is_auto_ = true;
        custom_width_ = std::max(16, player_->sourceWidth());
        custom_height_ = std::max(16, player_->sourceHeight());
        rotation_deg_ = player_->rotation();
        player_->setPreviewRotation(rotation_deg_);
        hdr_to_sdr_ = player_->isHdr() && player_->isHdrConversionSupported();
        player_->setHdrToSdr(hdr_to_sdr_);
        if (hdr_to_sdr_)
            player_->rerenderCurrentFrame();
        if (rotation_value_el_)
            rotation_value_el_->SetInnerRML(std::to_string(rotation_deg_) + "°");
        syncHdrControls();
        texture_needs_update_ = true;
        preview_src_.clear();

        if (video_path_changed || output_dir_.empty()) {
            std::filesystem::path output_name = video_path_.stem();
            output_name += "_frames";
            output_dir_ = video_path_.parent_path() / output_name;
        }

        controls_dirty_ = true;
        clearExtractionStatus();
        markContentDirty();
        return true;
    }

    void VideoExtractorDialog::refreshBatchOutputDirectories() {
        std::set<std::string> used_names;
        for (auto& item : batch_items_) {
            std::string stem = lfs::core::path_to_utf8(item.video_path.stem());
            if (stem.empty())
                stem = "video";
            std::string unique_stem = stem;
            for (int suffix = 2; !used_names.insert(outputDirectoryKey(unique_stem)).second; ++suffix)
                unique_stem = std::format("{}_{}", stem, suffix);
            item.output_dir = output_dir_.empty() ? std::filesystem::path{}
                                                  : output_dir_ / lfs::core::utf8_to_path(unique_stem);
        }
    }

    void VideoExtractorDialog::saveSelectedBatchItem() {
        if (!selected_batch_item_ || *selected_batch_item_ >= batch_items_.size() || !player_->isOpen())
            return;
        auto& item = batch_items_[*selected_batch_item_];
        item.start_time = trim_start_;
        item.end_time = trim_end_;
        item.end_time_is_auto = trim_end_is_auto_;
        item.rotation = rotation_deg_;
        item.duration = player_->duration();
        item.source_fps = player_->fps();
        item.is_hdr = player_->isHdr();
        item.hdr_to_sdr_supported = item.is_hdr && player_->isHdrConversionSupported();
    }

    bool VideoExtractorDialog::probeBatchVideo(BatchVideoItem& item, std::string& error) const {
        io::video::VideoMetadata metadata;
        if (!io::video::probeVideoFileMetadata(item.video_path, metadata, error))
            return false;
        item.rotation = metadata.rotation;
        item.source_fps = metadata.fps;
        item.duration = metadata.duration;
        item.is_hdr = metadata.is_hdr;
        item.hdr_to_sdr_supported = item.is_hdr && io::isHdrTonemapSupported(metadata.hdr_format);
        return true;
    }

    void VideoExtractorDialog::selectBatchItem(const std::size_t index) {
        if (extracting_.load() || pending_params_set_ || index >= batch_items_.size())
            return;
        saveSelectedBatchItem();
        const bool preserve_shared_settings = selected_batch_item_.has_value() ||
                                              (player_->isOpen() &&
                                               video_path_.lexically_normal() == batch_items_[index].video_path);
        const std::filesystem::path output_root = output_dir_;
        const int custom_width = custom_width_;
        const int custom_height = custom_height_;
        const bool hdr_to_sdr = hdr_to_sdr_;
        if (!openVideo(batch_items_[index].video_path)) {
            output_dir_ = output_root;
            refreshBatchOutputDirectories();
            return;
        }

        // openVideo supplies a single-video default output directory. A batch must preserve its
        // explicit root instead, including an intentionally empty root before the user chooses it.
        output_dir_ = output_root;
        refreshBatchOutputDirectories();
        if (preserve_shared_settings) {
            custom_width_ = custom_width;
            custom_height_ = custom_height;
            hdr_to_sdr_ = hdr_to_sdr;
            player_->setHdrToSdr(hdr_to_sdr_ && player_->isHdrConversionSupported());
        }

        selected_batch_item_ = index;
        auto& item = batch_items_[index];
        item.duration = player_->duration();
        item.source_fps = player_->fps();
        item.is_hdr = player_->isHdr();
        item.hdr_to_sdr_supported = item.is_hdr && player_->isHdrConversionSupported();
        trim_start_ = static_cast<float>(item.start_time);
        trim_end_ = item.end_time < 0.0 ? static_cast<float>(item.duration)
                                        : static_cast<float>(item.end_time);
        trim_end_is_auto_ = item.end_time_is_auto;
        rotation_deg_ = item.rotation;
        player_->setPreviewRotation(rotation_deg_);
        if (player_->currentFrameHasGpuRotation())
            player_->rerenderCurrentFrame();
        texture_needs_update_ = true;
        controls_dirty_ = true;
        batch_list_dirty_.store(true);
        markContentDirty();
    }

    bool VideoExtractorDialog::addBatchVideos(const std::vector<std::filesystem::path>& paths) {
        if (extracting_.load() || pending_params_set_ || paths.empty())
            return false;

        const bool adding_first_batch_item = batch_items_.empty();
        std::set<std::filesystem::path> existing;
        for (const auto& item : batch_items_)
            existing.insert(item.video_path.lexically_normal());

        bool start_worker = false;
        bool queued_any = false;
        {
            std::lock_guard lock(batch_probe_mutex_);
            for (const auto& path : paths) {
                const auto normalized = path.lexically_normal();
                const bool already_scheduled = std::find(scheduled_batch_probe_paths_.begin(),
                                                         scheduled_batch_probe_paths_.end(), normalized) !=
                                               scheduled_batch_probe_paths_.end();
                if (normalized.empty() || !existing.insert(normalized).second || already_scheduled)
                    continue;
                queued_batch_probes_.push_back({.video_path = normalized, .generation = batch_probe_generation_});
                scheduled_batch_probe_paths_.push_back(normalized);
                queued_any = true;
            }
            if (queued_any && !batch_probe_active_.exchange(true))
                start_worker = true;
        }
        if (!queued_any)
            return false;

        if (adding_first_batch_item && !batch_output_root_explicit_) {
            // Batch extraction requires an intentional output root. Do not silently reuse the
            // legacy single-video "<video>_frames" default as a root for every selected video.
            output_dir_.clear();
        }
        if (start_worker) {
            if (batch_probe_thread_)
                batch_probe_thread_.reset();
            batch_probe_thread_.emplace(
                [this](const std::stop_token stop_token) { runBatchProbeWorker(stop_token); });
        }
        controls_dirty_ = true;
        batch_list_dirty_.store(true);
        markContentDirty();
        return true;
    }

    void VideoExtractorDialog::runBatchProbeWorker(const std::stop_token stop_token) {
        for (;;) {
            QueuedBatchProbe request;
            bool finished = false;
            {
                std::lock_guard lock(batch_probe_mutex_);
                if (stop_token.stop_requested() || queued_batch_probes_.empty()) {
                    batch_probe_active_.store(false);
                    finished = true;
                } else {
                    request = std::move(queued_batch_probes_.front());
                    queued_batch_probes_.pop_front();
                }
            }
            if (finished) {
                lfs::python::request_redraw();
                return;
            }

            BatchVideoItem item{.video_path = request.video_path};
            std::string error;
            const bool success = probeBatchVideo(item, error);

            std::lock_guard lock(batch_probe_mutex_);
            if (stop_token.stop_requested() || request.generation != batch_probe_generation_)
                continue;
            if (success) {
                completed_batch_probe_items_.push_back({.item = std::move(item), .generation = request.generation});
            } else {
                completed_batch_probe_failures_.push_back(
                    {.video_path = request.video_path, .error = std::move(error), .generation = request.generation});
            }
            batch_probe_results_ready_.store(true);
        }
    }

    void VideoExtractorDialog::applyCompletedBatchProbes() {
        // Publish an entire queued batch at once. Besides keeping the list stable while it is
        // populated, this aggregates failures into one useful message.
        if (batch_probe_active_.load())
            return;

        std::vector<CompletedBatchProbe> items;
        std::vector<BatchProbeFailure> failures;
        {
            std::lock_guard lock(batch_probe_mutex_);
            items.swap(completed_batch_probe_items_);
            failures.swap(completed_batch_probe_failures_);
            batch_probe_results_ready_.store(false);
            for (const auto& result : items)
                std::erase(scheduled_batch_probe_paths_, result.item.video_path);
            for (const auto& failure : failures)
                std::erase(scheduled_batch_probe_paths_, failure.video_path);
        }

        for (auto& result : items)
            batch_items_.push_back(std::move(result.item));
        if (!items.empty()) {
            refreshBatchOutputDirectories();
            if (!selected_batch_item_ && !batch_items_.empty())
                selectBatchItem(0);
        }

        if (!failures.empty()) {
            for (const auto& failure : failures) {
                LOG_WARN("Video extractor: could not add batch video {}: {}",
                         lfs::core::path_to_utf8(failure.video_path.filename()), failure.error);
            }
            const std::string message = failures.size() == 1
                                            ? std::format("{}: {}",
                                                          lfs::core::path_to_utf8(failures.front().video_path.filename()),
                                                          failures.front().error)
                                            : std::format("{} videos could not be opened; see log for details.",
                                                          failures.size());
            setExtractionError(message, false);
        }
        if (!items.empty() || !failures.empty()) {
            controls_dirty_ = true;
            batch_list_dirty_.store(true);
            markContentDirty();
        }
    }

    void VideoExtractorDialog::joinBatchProbeThread() {
        if (!batch_probe_thread_)
            return;
        batch_probe_thread_->request_stop();
        batch_probe_thread_.reset();
        batch_probe_active_.store(false);
    }

    void VideoExtractorDialog::addVideosFromBrowse(std::vector<std::filesystem::path> paths) {
        if (paths.empty())
            return;

        const bool has_pending_batch = batch_probe_active_.load() || batch_probe_results_ready_.load();
        const bool has_single_video = batch_items_.empty() && player_->isOpen() && !video_path_.empty();
        const bool repeats_single_video = paths.size() == 1 && has_single_video &&
                                          paths.front().lexically_normal() == video_path_.lexically_normal();
        if (!has_single_video || repeats_single_video) {
            if (paths.size() == 1 && batch_items_.empty() && !has_pending_batch)
                openVideo(paths.front());
            else
                (void)addBatchVideos(paths);
            return;
        }

        // A second browse selection promotes the existing single source to a batch item rather
        // than replacing it. selectBatchItem() preserves the shared output settings when the
        // completed probe selects this same source again.
        paths.insert(paths.begin(), video_path_);
        (void)addBatchVideos(paths);
    }

    void VideoExtractorDialog::removeSelectedBatchItem() {
        if (extracting_.load() || !selected_batch_item_ || *selected_batch_item_ >= batch_items_.size())
            return;
        const std::size_t removed = *selected_batch_item_;
        batch_items_.erase(batch_items_.begin() + static_cast<std::ptrdiff_t>(removed));
        selected_batch_item_.reset();
        refreshBatchOutputDirectories();
        if (!batch_items_.empty())
            selectBatchItem(std::min(removed, batch_items_.size() - 1));
        else {
            player_->close();
            video_path_.clear();
            preview_src_.clear();
            texture_needs_update_ = true;
        }
        controls_dirty_ = true;
        batch_list_dirty_.store(true);
        markContentDirty();
    }

    void VideoExtractorDialog::clearBatchVideos() {
        if (extracting_.load())
            return;
        {
            std::lock_guard lock(batch_probe_mutex_);
            ++batch_probe_generation_;
            queued_batch_probes_.clear();
            scheduled_batch_probe_paths_.clear();
            completed_batch_probe_items_.clear();
            completed_batch_probe_failures_.clear();
            batch_probe_results_ready_.store(false);
        }
        batch_items_.clear();
        selected_batch_item_.reset();
        player_->close();
        video_path_.clear();
        preview_src_.clear();
        texture_needs_update_ = true;
        controls_dirty_ = true;
        batch_list_dirty_.store(true);
        markContentDirty();
    }

    std::string VideoExtractorDialog::batchStatusLabel(const BatchVideoItem& item) const {
        if (!item.warning.empty())
            return LOC("video_extractor.masking_warning_label");
        switch (item.status) {
        case BatchItemStatus::Pending: return LOC("video_extractor.batch_pending");
        case BatchItemStatus::Running: return LOC("video_extractor.batch_running");
        case BatchItemStatus::Complete: return LOC("video_extractor.batch_complete");
        case BatchItemStatus::Failed: return LOC("video_extractor.batch_failed");
        case BatchItemStatus::Stopped: return LOC("video_extractor.batch_stopped");
        }
        return {};
    }

    void VideoExtractorDialog::syncBatchList() {
        if (!batch_video_list_el_ || !batch_list_dirty_.exchange(false))
            return;
        int probing_count = 0;
        if (batch_probe_active_.load()) {
            std::lock_guard lock(batch_probe_mutex_);
            probing_count = static_cast<int>(scheduled_batch_probe_paths_.size());
        }
        std::string rows;
        int completed = 0;
        int failed = 0;
        {
            std::lock_guard lock(batch_mutex_);
            std::unordered_map<std::string, int> filename_counts;
            for (const auto& item : batch_items_)
                ++filename_counts[lfs::core::path_to_utf8(item.video_path.filename())];
            std::unordered_map<std::string, int> filename_occurrences;
            rows.reserve(batch_items_.size() * 180);
            for (std::size_t i = 0; i < batch_items_.size(); ++i) {
                const auto& item = batch_items_[i];
                completed += item.status == BatchItemStatus::Complete;
                failed += item.status == BatchItemStatus::Failed;
                const std::string filename = lfs::core::path_to_utf8(item.video_path.filename());
                std::string display_name = filename;
                if (filename_counts[filename] > 1)
                    display_name += std::format(" ({})", ++filename_occurrences[filename]);
                const std::string selected = selected_batch_item_ && *selected_batch_item_ == i
                                                 ? " batch-video-item--selected"
                                                 : "";
                std::string status = batchStatusLabel(item);
                if (item.has_existing_output)
                    status = std::format("{} · {}", status, LOC("video_extractor.batch_existing_output"));
                std::string tooltip = std::format("{}: {}\n{}: {}", LOC(VideoExtractor::VIDEO),
                                                  lfs::core::path_to_utf8(item.video_path),
                                                  LOC(VideoExtractor::OUTPUT),
                                                  lfs::core::path_to_utf8(item.output_dir));
                if (!item.warning.empty())
                    tooltip += std::format("\n{}", item.warning);
                if (!item.error.empty())
                    tooltip += std::format("\n{}", item.error);
                rows += std::format("<button id=\"batch-video-{}\" class=\"batch-video-item{}\" "
                                    "title=\"{}\" type=\"button\">"
                                    "<span class=\"batch-video-item__name\">{}</span>"
                                    "<span class=\"batch-video-item__status\">{}</span></button>",
                                    i, selected, Rml::StringUtilities::EncodeRml(tooltip),
                                    Rml::StringUtilities::EncodeRml(display_name),
                                    Rml::StringUtilities::EncodeRml(status));
            }
        }
        if (rows.empty())
            rows = std::format("<span class=\"inline-note\">{}</span>", LOC(VideoExtractor::NO_FILE));
        const bool changed = setCachedInnerRml(batch_video_list_el_, rows);
        const int total = static_cast<int>(batch_items_.size());
        const std::string summary = probing_count > 0
                                        ? LOCF("video_extractor.batch_reading", probing_count)
                                    : total == 0
                                        ? std::string{}
                                        : LOCF("video_extractor.batch_summary", total, completed, failed);
        if (setCachedText(batch_summary_el_, summary) || changed)
            markContentDirty();
    }

    int VideoExtractorDialog::calculateEstimatedFrames() const {
        if (!player_ || !player_->isOpen())
            return 0;

        const double duration = trimDuration();
        if (duration <= 0.0)
            return 0;

        if (mode_selection_ == 0)
            return static_cast<int>(std::min<std::size_t>(
                io::calculateFpsSampleCount(trim_start_, trim_start_ + duration, fps_),
                static_cast<std::size_t>(std::numeric_limits<int>::max())));

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

        int width = player_->width();
        int height = player_->height();
        const int channels = player_->currentFrameChannels();
        if (channels != 3 && channels != 4)
            return;
        const uint8_t* upload_data = data;
        std::vector<uint8_t> rotated_buf;

        if (rotation_deg_ != 0 && !player_->currentFrameHasGpuRotation()) {
            rotated_buf.resize(static_cast<size_t>(width) * height * channels);
            if (rotation_deg_ == 180) {
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        const int si = (y * width + x) * channels;
                        const int di = ((height - 1 - y) * width + (width - 1 - x)) * channels;
                        for (int channel = 0; channel < channels; ++channel)
                            rotated_buf[di + channel] = data[si + channel];
                    }
            } else {
                const int dst_w = height;
                const int dst_h = width;
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        const int si = (y * width + x) * channels;
                        const int di = (rotation_deg_ == 90)
                                           ? (x * height + (height - 1 - y)) * channels // CW
                                           : ((width - 1 - x) * height + y) * channels; // CCW
                        for (int channel = 0; channel < channels; ++channel)
                            rotated_buf[di + channel] = data[si + channel];
                    }
                width = dst_w;
                height = dst_h;
            }
            upload_data = rotated_buf.data();
        }

        if (!preview_texture_)
            preview_texture_ = std::make_unique<lfs::vis::gui::VulkanUiTexture>();
        if (preview_texture_->upload(upload_data, width, height, channels)) {
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
            host_->setFloating(floating_);
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
        batch_video_list_el_ = nullptr;
        batch_summary_el_ = nullptr;
        remove_video_el_ = nullptr;
        clear_videos_el_ = nullptr;
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
        masking_toggle_el_ = nullptr;
        masking_options_el_ = nullptr;
        mask_circle_el_ = nullptr;
        mask_recenter_el_ = nullptr;
        mask_fov_el_ = nullptr;
        mask_model_el_ = nullptr;
        mask_require_operator_el_ = nullptr;
        mask_dilation_el_ = nullptr;
        mask_view_fov_el_ = nullptr;
        mask_manual_seed_el_ = nullptr;
        mask_seed_options_el_ = nullptr;
        mask_seed_x_el_ = nullptr;
        mask_seed_y_el_ = nullptr;
        mask_seed_frame_el_ = nullptr;

        naming_select_el_ = nullptr;
        custom_pattern_row_el_ = nullptr;
        naming_help_el_ = nullptr;
        naming_error_el_ = nullptr;
        pattern_input_el_ = nullptr;
        first_source_frame_number_input_el_ = nullptr;
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
        continue_batch_on_failure_el_ = nullptr;
        hdr_to_sdr_el_ = nullptr;
        hdr_to_sdr_row_el_ = nullptr;
        overwrite_overlay_el_ = nullptr;
        overwrite_message_el_ = nullptr;
        overwrite_cancel_btn_el_ = nullptr;
        overwrite_delete_btn_el_ = nullptr;
        overwrite_proceed_btn_el_ = nullptr;
        rotation_cw_btn_el_ = nullptr;
        rotation_ccw_btn_el_ = nullptr;
        rotation_value_el_ = nullptr;
        hdr_badge_el_ = nullptr;
        hdr_badge_type_el_ = nullptr;
        hdr_badge_label_el_ = nullptr;
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
        batch_video_list_el_ = document_->GetElementById("batch-video-list");
        batch_summary_el_ = document_->GetElementById("batch-summary");
        remove_video_el_ = document_->GetElementById("btn-remove-video");
        clear_videos_el_ = document_->GetElementById("btn-clear-videos");
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
        masking_toggle_el_ = document_->GetElementById("masking-toggle");
        masking_options_el_ = document_->GetElementById("masking-options");
        mask_circle_el_ = document_->GetElementById("mask-circle");
        mask_recenter_el_ = document_->GetElementById("mask-recenter");
        mask_fov_el_ = document_->GetElementById("mask-fov");
        mask_model_el_ = document_->GetElementById("mask-model");
        mask_require_operator_el_ = document_->GetElementById("mask-require-operator");
        mask_dilation_el_ = document_->GetElementById("mask-dilation");
        mask_view_fov_el_ = document_->GetElementById("mask-view-fov");
        mask_manual_seed_el_ = document_->GetElementById("mask-manual-seed");
        mask_seed_options_el_ = document_->GetElementById("mask-seed-options");
        mask_seed_x_el_ = document_->GetElementById("mask-seed-x");
        mask_seed_y_el_ = document_->GetElementById("mask-seed-y");
        mask_seed_frame_el_ = document_->GetElementById("mask-seed-frame");

        naming_select_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("naming-select"));
        custom_pattern_row_el_ = document_->GetElementById("custom-pattern-row");
        naming_help_el_ = document_->GetElementById("naming-help");
        naming_error_el_ = document_->GetElementById("naming-error");
        pattern_input_el_ = document_->GetElementById("pattern-input");
        first_source_frame_number_input_el_ =
            document_->GetElementById("first-source-frame-number-input");
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
        sharpness_toggle_el_ = document_->GetElementById("sharpness-toggle");
        sharpness_options_el_ = document_->GetElementById("sharpness-options");
        sharpness_threshold_row_el_ = document_->GetElementById("sharpness-threshold-row");
        sharpness_algorithm_select_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(
            document_->GetElementById("sharpness-algorithm-select"));
        sharpness_mode_select_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(
            document_->GetElementById("sharpness-mode-select"));
        sharpness_mode_desc_el_ = document_->GetElementById("sharpness-mode-desc");
        sharpness_threshold_slider_el_ = document_->GetElementById("sharpness-threshold-slider");
        sharpness_threshold_value_el_ = document_->GetElementById("sharpness-threshold-value");
        sharpness_window_row_el_ = document_->GetElementById("sharpness-window-row");
        window_candidates_select_el_ = document_->GetElementById("window-candidates-select");
        window_candidates_readout_el_ = document_->GetElementById("window-candidates-readout");
        generate_metadata_el_ = document_->GetElementById("generate-metadata");
        continue_batch_on_failure_el_ = document_->GetElementById("continue-batch-on-failure");
        overwrite_overlay_el_ = document_->GetElementById("overwrite-overlay");
        overwrite_message_el_ = document_->GetElementById("overwrite-message");
        overwrite_cancel_btn_el_ = document_->GetElementById("overwrite-cancel");
        overwrite_delete_btn_el_ = document_->GetElementById("overwrite-delete");
        overwrite_proceed_btn_el_ = document_->GetElementById("overwrite-proceed");

        elements_cached_ =
            title_el_ && close_btn_el_ && preview_shell_el_ && preview_image_el_ &&
            preview_empty_el_ && step_back_btn_el_ && play_btn_el_ && play_icon_el_ &&
            step_forward_btn_el_ && time_label_el_ && timeline_el_ && timeline_trim_el_ &&
            timeline_progress_el_ && timeline_playhead_el_ && timeline_start_el_ &&
            timeline_end_el_ && timeline_markers_el_ && trim_start_input_el_ &&
            trim_end_input_el_ && trim_start_set_el_ && trim_end_set_el_ && trim_reset_el_ &&
            estimated_frames_el_ && video_value_el_ && batch_video_list_el_ && batch_summary_el_ &&
            remove_video_el_ && clear_videos_el_ && output_value_el_ && browse_video_el_ &&
            browse_output_el_ && mode_select_el_ && fps_row_el_ && fps_slider_el_ &&
            fps_value_el_ && interval_row_el_ && interval_input_el_ && interval_value_el_ &&
            format_select_el_ && quality_row_el_ && quality_slider_el_ && quality_value_el_ &&
            resolution_select_el_ && scale_row_el_ && scale_select_el_ &&
            custom_resolution_row_el_ && custom_width_input_el_ && custom_height_input_el_ &&
            output_resolution_el_ && naming_select_el_ && custom_pattern_row_el_ &&
            naming_help_el_ && naming_error_el_ && pattern_input_el_ && first_source_frame_number_input_el_ &&
            pattern_example_el_ &&
            start_btn_el_ && stop_btn_el_ && cancel_btn_el_ && select_hint_el_ && progress_section_el_ &&
            progress_text_el_ && progress_bar_el_ && complete_section_el_ &&
            complete_text_el_ && ok_btn_el_ && stopped_section_el_ && stopped_text_el_ &&
            stopped_ok_btn_el_ && error_section_el_ && error_text_el_ && dismiss_btn_el_ &&
            sharpness_toggle_el_ && sharpness_options_el_ && sharpness_algorithm_select_el_ &&
            sharpness_mode_select_el_ && sharpness_threshold_slider_el_ && sharpness_threshold_value_el_;

        if (!elements_cached_) {
            LOG_ERROR("VideoExtractorDialog: missing required Rml elements");
            return;
        }

        rotation_cw_btn_el_ = document_->GetElementById("btn-rotation-cw");
        rotation_ccw_btn_el_ = document_->GetElementById("btn-rotation-ccw");
        rotation_value_el_ = document_->GetElementById("rotation-value");
        hdr_badge_el_ = document_->GetElementById("hdr-badge");
        hdr_badge_type_el_ = document_->GetElementById("hdr-badge-type");
        hdr_badge_label_el_ = document_->GetElementById("hdr-badge-label");
        hdr_to_sdr_el_ = document_->GetElementById("hdr-to-sdr");
        hdr_to_sdr_row_el_ = document_->GetElementById("hdr-to-sdr-row");

        bindEventListeners();
        controls_dirty_ = true;
        batch_list_dirty_.store(true);
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
        listen_click(batch_video_list_el_);
        listen_click(remove_video_el_);
        listen_click(clear_videos_el_);
        listen_click(browse_output_el_);
        listen_click(start_btn_el_);
        listen_click(stop_btn_el_);
        listen_click(cancel_btn_el_);
        listen_click(ok_btn_el_);
        listen_click(stopped_ok_btn_el_);
        listen_click(dismiss_btn_el_);

        listen_click(overwrite_cancel_btn_el_);
        listen_click(overwrite_delete_btn_el_);
        listen_click(overwrite_proceed_btn_el_);

        listen_change(mode_select_el_);
        listen_change(fps_slider_el_);
        listen_input(fps_slider_el_);
        listen_change(fps_value_el_);
        listen_change(interval_input_el_);
        listen_change(format_select_el_);
        listen_change(quality_slider_el_);
        listen_input(quality_slider_el_);
        listen_change(resolution_select_el_);
        for (auto* control : {masking_toggle_el_, mask_circle_el_, mask_recenter_el_, mask_fov_el_,
                              mask_model_el_, mask_require_operator_el_, mask_dilation_el_, mask_view_fov_el_,
                              mask_manual_seed_el_, mask_seed_x_el_, mask_seed_y_el_, mask_seed_frame_el_})
            listen_change(control);

        listen_change(scale_select_el_);
        listen_change(custom_width_input_el_);
        listen_change(custom_height_input_el_);
        listen_change(naming_select_el_);
        listen_change(pattern_input_el_);
        listen_change(first_source_frame_number_input_el_);
        listen_change(trim_start_input_el_);
        listen_change(trim_end_input_el_);
        listen_change(sharpness_toggle_el_);
        listen_change(sharpness_algorithm_select_el_);
        listen_change(sharpness_mode_select_el_);
        listen_change(sharpness_threshold_slider_el_);
        listen_input(sharpness_threshold_slider_el_);
        listen_change(window_candidates_select_el_);
        listen_change(generate_metadata_el_);
        listen_change(continue_batch_on_failure_el_);
        listen_change(hdr_to_sdr_el_);
        listen_click(rotation_cw_btn_el_);
        listen_click(rotation_ccw_btn_el_);

        timeline_el_->AddEventListener(Rml::EventId::Mousedown, &listener_);
        if (auto* const body = document_->GetElementById("body")) {
            body->AddEventListener(Rml::EventId::Mousemove, &listener_);
            body->AddEventListener(Rml::EventId::Mouseup, &listener_);
        }
    }

    void VideoExtractorDialog::syncPanel() {
        const lfs::vis::gui::ScopedVulkanUiTextureContext textures(host_ ? host_->nativeGraphics() : nullptr);
        if (!elements_cached_)
            return;

        applyCompletedBatchProbes();
        syncLocale();
        syncBatchList();
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
        batch_list_dirty_.store(true);
        bool changed = false;
        changed |= setCachedText(title_el_, LOC(VideoExtractor::TITLE));
        changed |= setCachedAttribute(preview_empty_el_, "data-empty-text", LOC(VideoExtractor::SELECT_PREVIEW));
        changed |= setCachedText(preview_empty_el_, LOC(VideoExtractor::SELECT_PREVIEW));
        changed |= setCachedAttribute(step_back_btn_el_, "title", LOC(VideoExtractor::STEP_BACKWARD));
        changed |= setCachedAttribute(step_forward_btn_el_, "title", LOC(VideoExtractor::STEP_FORWARD));
        changed |= setCachedAttribute(trim_start_set_el_, "title", LOC(VideoExtractor::SET_START));
        changed |= setCachedAttribute(trim_end_set_el_, "title", LOC(VideoExtractor::SET_END));
        changed |= setCachedAttribute(browse_output_el_, "title", LOC(VideoExtractor::SELECT_FOLDER));
        changed |= setCachedText(browse_video_el_, LOC(VideoExtractor::ADD_VIDEOS));
        changed |= setCachedText(remove_video_el_, LOC(VideoExtractor::REMOVE_VIDEO));
        changed |= setCachedText(clear_videos_el_, LOC(VideoExtractor::CLEAR_VIDEOS));
        changed |= setCachedAttribute(fps_slider_el_, "title", LOC(VideoExtractor::FPS_TOOLTIP));
        changed |= setCachedAttribute(fps_value_el_, "title", LOC(VideoExtractor::FPS_TOOLTIP));
        changed |= setCachedAttribute(interval_input_el_, "title", LOC(VideoExtractor::INTERVAL_TOOLTIP));
        changed |= setCachedAttribute(quality_slider_el_, "title", LOC(VideoExtractor::QUALITY_LABEL));
        changed |= setCachedAttribute(custom_width_input_el_, "title", LOC(VideoExtractor::WIDTH));
        changed |= setCachedAttribute(custom_height_input_el_, "title", LOC(VideoExtractor::HEIGHT));
        changed |= setCachedAttribute(pattern_input_el_, "title", LOC(VideoExtractor::PATTERN_TOOLTIP));
        changed |= setCachedSelectLabels(naming_select_el_,
                                         std::array{"video_extractor.naming_video_frame", "video_extractor.naming_video_number",
                                                    "video_extractor.naming_frame_number", "video_extractor.naming_number",
                                                    "video_extractor.naming_custom"});
        changed |= setCachedSelectLabels(
            mode_select_el_,
            std::array{"video_extractor.mode_fps", "video_extractor.mode_interval"});
        changed |= setCachedSelectLabels(
            format_select_el_,
            std::array{"video_extractor.format_png", "video_extractor.format_jpeg"});
        changed |= setCachedSelectLabels(
            resolution_select_el_,
            std::array{"video_extractor.res_original", "video_extractor.res_scale",
                       "video_extractor.res_custom"});
        changed |= setCachedSelectLabels(
            sharpness_algorithm_select_el_,
            std::array{"video_extractor.sharpness_algo_combined",
                       "video_extractor.sharpness_algo_tenengrad",
                       "video_extractor.sharpness_algo_laplacian"});
        changed |= setCachedSelectLabels(
            sharpness_mode_select_el_,
            std::array{"video_extractor.sharpness_mode_threshold",
                       "video_extractor.sharpness_mode_window"});
        changed |= setCachedSelectLabels(
            dynamic_cast<Rml::ElementFormControlSelect*>(window_candidates_select_el_),
            std::array{"video_extractor.candidates_auto", "video_extractor.candidates_fast",
                       "video_extractor.candidates_balanced", "video_extractor.candidates_quality",
                       "video_extractor.candidates_high_quality",
                       "video_extractor.candidates_very_high", "video_extractor.candidates_all"});
        changed |= setCachedText(stop_btn_el_, LOC(VideoExtractor::STOP));
        changed |= setCachedText(cancel_btn_el_, LOC(VideoExtractor::CANCEL));
        syncOverwritePrompt();
        changed |= setCachedText(overwrite_cancel_btn_el_, LOC(VideoExtractor::OVERWRITE_CANCEL));
        changed |= setCachedText(overwrite_delete_btn_el_, LOC(VideoExtractor::OVERWRITE_DELETE));
        changed |= setCachedText(overwrite_proceed_btn_el_, LOC(VideoExtractor::OVERWRITE_PROCEED));
        markContentDirty();
        controls_dirty_ |= changed;
    }

    void VideoExtractorDialog::syncOverwritePrompt() {
        if (!pending_batch_params_.empty()) {
            const char* const prompt = pending_batch_metadata_will_be_replaced_
                                           ? "video_extractor.batch_overwrite_prompt_metadata"
                                           : "video_extractor.batch_overwrite_prompt";
            if (setCachedText(overwrite_message_el_, LOCF(prompt, pending_batch_conflict_count_)))
                markContentDirty();
            return;
        }
        if (pending_params_set_ && pending_params_.masking.enabled) {
            if (setCachedText(overwrite_message_el_, LOC("video_extractor.masking_overwrite_prompt")))
                markContentDirty();
            return;
        }
        const char* const prompt = pending_metadata_will_be_replaced_
                                       ? (pending_has_frame_conflicts_
                                              ? VideoExtractor::OVERWRITE_PROMPT_FRAMES_METADATA
                                              : VideoExtractor::OVERWRITE_PROMPT_METADATA)
                                       : VideoExtractor::OVERWRITE_PROMPT;
        if (setCachedText(overwrite_message_el_, LOC(prompt)))
            markContentDirty();
    }

    void VideoExtractorDialog::syncVideoPreview() {
        bool changed = false;
        if (player_->isOpen()) {
            if (player_->update(0.0))
                texture_needs_update_ = true;
            if (texture_needs_update_)
                updatePreviewTexture();
        }
        if (std::string player_error = player_->takeError(); !player_error.empty())
            setExtractionError(player_error, false);

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
            const std::string& src = preview_texture_->rmlSrcUrl(preview_texture_width_,
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

    bool VideoExtractorDialog::syncHdrControls() {
        const bool has_video = player_ && player_->isOpen();
        const bool is_hdr = has_video && player_->isHdr();
        const bool conversion_supported =
            is_hdr && player_->isHdrConversionSupported();
        if (batch_items_.empty() && (!has_video || !conversion_supported))
            hdr_to_sdr_ = false;

        bool changed = false;
        changed |= setCachedChecked(hdr_to_sdr_el_, hdr_to_sdr_ && conversion_supported);
        changed |= setCachedProperty(hdr_to_sdr_row_el_, "display",
                                     is_hdr ? "inline-flex" : "none");
        changed |= setCachedDisabled(hdr_to_sdr_el_,
                                     extracting_.load() || !conversion_supported);

        if (!has_video) {
            changed |= setCachedProperty(hdr_badge_el_, "display", "none");
            changed |= setCachedText(hdr_badge_label_el_, "");
            changed |= setCachedText(hdr_badge_type_el_, "");
            changed |= setCachedProperty(hdr_badge_type_el_, "display", "none");
            if (hdr_badge_el_) {
                hdr_badge_el_->SetClass("hdr-badge--hdr", false);
                hdr_badge_el_->SetClass("hdr-badge--sdr", false);
            }
            return changed;
        }

        changed |= setCachedProperty(hdr_badge_el_, "display", "inline-flex");
        if (hdr_badge_el_) {
            hdr_badge_el_->SetClass("hdr-badge--hdr", is_hdr);
            hdr_badge_el_->SetClass("hdr-badge--sdr", !is_hdr);
        }
        changed |= setCachedText(hdr_badge_label_el_, is_hdr ? "HDR" : "SDR");
        changed |= setCachedText(hdr_badge_type_el_,
                                 is_hdr ? player_->hdrInfo() : "");
        changed |= setCachedProperty(hdr_badge_type_el_, "display",
                                     is_hdr ? "inline" : "none");
        return changed;
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
                    if (time >= end)
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

        // These editable values are parsed again on blur and Start; preserve float precision.
        changed |= setCachedControlValue(trim_start_input_el_, std::format("{}", start));
        changed |= setCachedControlValue(trim_end_input_el_, std::format("{}", end));
        changed |= setCachedText(estimated_frames_el_, LOCF(VideoExtractor::ESTIMATED_FRAMES, lfs::core::format_count(frame_count)));

        if (changed)
            markContentDirty();
    }

    void VideoExtractorDialog::syncControls() {
        bool changed = false;
        const bool has_video = player_->isOpen();
        const bool has_batch = !batch_items_.empty();
        const bool extracting = extracting_.load();
        const bool probing_batch = batch_probe_active_.load();
        const bool stop_requested = stop_extraction_requested_.load();
        const bool can_start = (has_batch || has_video) && !output_dir_.empty() && !extracting && !probing_batch;

        changed |= setCachedDisabled(step_back_btn_el_, !has_video);
        changed |= setCachedDisabled(play_btn_el_, !has_video);
        changed |= setCachedDisabled(step_forward_btn_el_, !has_video);
        changed |= setCachedDisabled(timeline_el_, !has_video);
        changed |= setCachedDisabled(trim_start_input_el_, !has_video);
        changed |= setCachedDisabled(trim_end_input_el_, !has_video);
        changed |= setCachedDisabled(trim_start_set_el_, !has_video);
        changed |= setCachedDisabled(trim_end_set_el_, !has_video);
        changed |= setCachedDisabled(trim_reset_el_, !has_video);
        std::string resolved_pattern, naming_error;
        const bool naming_valid = resolveFilenamePattern(resolved_pattern, naming_error);
        changed |= setCachedDisabled(start_btn_el_, !can_start || !naming_valid);
        changed |= setCachedDisabled(stop_btn_el_, !extracting || stop_requested);
        changed |= setCachedDisabled(cancel_btn_el_, false);
        changed |= setCachedDisabled(browse_video_el_, extracting || pending_params_set_);
        changed |= setCachedDisabled(remove_video_el_, extracting || pending_params_set_ || !selected_batch_item_);
        changed |= setCachedDisabled(clear_videos_el_, extracting || pending_params_set_ || probing_batch ||
                                                           batch_items_.empty());
        changed |= setCachedDisabled(browse_output_el_, extracting || pending_params_set_);

        changed |= setCachedSelect(naming_select_el_, naming_selection_);
        changed |= setCachedDisabled(naming_select_el_, extracting);
        changed |= setCachedDisabled(pattern_input_el_, extracting);
        changed |= setCachedDisabled(first_source_frame_number_input_el_, extracting);
        changed |= setCachedProperty(custom_pattern_row_el_, "display", naming_selection_ == 4 ? "flex" : "none");
        changed |= setCachedProperty(naming_help_el_, "display", naming_selection_ == 4 ? "block" : "none");
        changed |= setCachedText(naming_error_el_, naming_error);
        changed |= setCachedProperty(naming_error_el_, "display", naming_valid ? "none" : "block");
        changed |= setCachedSelect(mode_select_el_, mode_selection_);
        changed |= setCachedSelect(format_select_el_, format_selection_);
        changed |= setCachedSelect(resolution_select_el_, resolution_mode_);
        changed |= setCachedSelect(scale_select_el_, scale_selection_);

        changed |= setCachedProperty(fps_row_el_, "display", mode_selection_ == 0 ? "flex" : "none");
        changed |= setCachedProperty(interval_row_el_, "display", mode_selection_ == 1 ? "flex" : "none");
        changed |= setCachedProperty(quality_row_el_, "display", format_selection_ == 1 ? "flex" : "none");
        changed |= syncHdrControls();
        changed |= setCachedProperty(scale_row_el_, "display", resolution_mode_ == 1 ? "flex" : "none");
        changed |= setCachedProperty(custom_resolution_row_el_, "display", resolution_mode_ == 2 ? "flex" : "none");
        changed |= setCachedProperty(stop_btn_el_, "display", extracting ? "block" : "none");
        changed |= setCachedChecked(continue_batch_on_failure_el_, continue_batch_on_failure_);
        changed |= setCachedDisabled(continue_batch_on_failure_el_, extracting);

        const bool masking_on = masking_toggle_el_ && masking_toggle_el_->HasAttribute("checked");
        const bool circle_on = mask_circle_el_ && mask_circle_el_->HasAttribute("checked");
        const bool manual_seed = mask_manual_seed_el_ && mask_manual_seed_el_->HasAttribute("checked");
        changed |= setCachedProperty(masking_options_el_, "display", masking_on ? "block" : "none");
        changed |= setCachedProperty(mask_seed_options_el_, "display", manual_seed ? "flex" : "none");
        for (auto* control : {masking_toggle_el_, mask_circle_el_, mask_manual_seed_el_, mask_seed_x_el_,
                              mask_seed_y_el_, mask_seed_frame_el_, mask_model_el_, mask_require_operator_el_, mask_dilation_el_})
            changed |= setCachedDisabled(control, extracting);
        changed |= setCachedDisabled(mask_recenter_el_, extracting || !circle_on);
        changed |= setCachedDisabled(mask_fov_el_, extracting || !circle_on);
        changed |= setCachedDisabled(mask_view_fov_el_, extracting || !circle_on || !mask_recenter_el_->HasAttribute("checked"));

        const bool sharpness_on = sharpness_toggle_el_ && sharpness_toggle_el_->HasAttribute("checked");
        changed |= setCachedProperty(sharpness_options_el_, "display", sharpness_on ? "block" : "none");

        // Show threshold slider only in threshold mode (hidden in window mode)
        const bool window_mode = sharpness_mode_select_el_ &&
                                 sharpness_mode_select_el_->GetSelection() == 1;
        changed |= setCachedProperty(sharpness_threshold_row_el_, "display",
                                     (sharpness_on && !window_mode) ? "flex" : "none");
        changed |= setCachedProperty(sharpness_window_row_el_, "display",
                                     (sharpness_on && window_mode) ? "flex" : "none");
        if (sharpness_mode_desc_el_) {
            changed |= setCachedText(sharpness_mode_desc_el_,
                                     window_mode ? LOC(VideoExtractor::SHARPNESS_MODE_DESC_WINDOW)
                                                 : LOC(VideoExtractor::SHARPNESS_MODE_DESC_THRESHOLD));
        }

        if (window_candidates_readout_el_) {
            // Calculate estimated window frames
            int est_window = 0;
            if (mode_selection_ == 0 && player_ && player_->fps() > 0 && fps_ > 0)
                est_window = static_cast<int>(std::round(player_->fps() / fps_));
            else if (mode_selection_ == 1)
                est_window = frame_interval_;

            std::string opt = window_candidates_select_el_
                                  ? window_candidates_select_el_->GetAttribute<Rml::String>("value", "10")
                                  : "10";
            int candidates = 0;
            if (window_candidates_target_ < 0) {
                // Auto mode: sqrt-based
                candidates = std::clamp(static_cast<int>(std::round(std::sqrt(static_cast<double>(est_window))) * 2), 5, 20);
            } else if (window_candidates_target_ == 0) {
                // All frames
                candidates = est_window;
            } else {
                candidates = std::min(window_candidates_target_, std::max(1, est_window));
            }
            changed |= setCachedText(window_candidates_readout_el_,
                                     LOCF(VideoExtractor::CANDIDATES_READOUT_FMT, lfs::core::format_count(candidates), lfs::core::format_count(est_window)));
        }

        if (sharpness_threshold_slider_el_ && sharpness_threshold_value_el_) {
            const int val = readIntValue(sharpness_threshold_slider_el_, 10);
            changed |= setCachedText(sharpness_threshold_value_el_, std::to_string(val) + "%");
        }

        // RmlUi range controls emit change while their value attribute is being updated.
        // Synchronizing the slider must not change the requested FPS.
        changed |= setCachedDisabled(fps_slider_el_, !has_video || extracting);
        changed |= setCachedDisabled(fps_value_el_, !has_video || extracting);
        const float maximum_fps = maximumRequestedExtractionFps(*player_, has_batch, fps_);
        const float requested_fps = std::clamp(fps_, 0.1f, maximum_fps);
        changed |= setCachedAttribute(fps_slider_el_, "max", std::format("{}", maximum_fps));
        changed |= setCachedControlValue(fps_slider_el_, std::format("{}", requested_fps));
        fps_ = requested_fps;
        // At the source limit, match the range control's step-rounded display
        // without changing the exact FPS used for extraction.
        changed |= setCachedControlValue(fps_value_el_, fps_ == maximum_fps
                                                            ? std::format("{}", readFloatValue(fps_slider_el_, fps_))
                                                            : std::format("{}", fps_));
        changed |= setCachedControlValue(interval_input_el_, std::to_string(frame_interval_));
        changed |= setCachedText(interval_value_el_, LOC(VideoExtractor::FRAMES_UNIT));
        changed |= setCachedControlValue(quality_slider_el_, std::to_string(jpg_quality_));
        changed |= setCachedText(quality_value_el_, std::format("{}%", jpg_quality_));
        changed |= setCachedControlValue(custom_width_input_el_, std::to_string(custom_width_));
        changed |= setCachedControlValue(custom_height_input_el_, std::to_string(custom_height_));
        changed |= setCachedControlValue(pattern_input_el_, filename_pattern_);
        changed |= setCachedControlValue(first_source_frame_number_input_el_,
                                         std::to_string(first_source_frame_number_));

        const std::string video_display = batch_items_.empty()
                                              ? (video_path_.empty() ? LOC(VideoExtractor::NO_FILE)
                                                                     : lfs::core::path_to_utf8(video_path_.filename()))
                                              : LOCF("video_extractor.batch_video_count", batch_items_.size());
        const std::string output_display = output_dir_.empty()
                                               ? LOC(VideoExtractor::NO_DIR)
                                               : lfs::core::path_to_utf8(output_dir_);
        changed |= setCachedText(video_value_el_, video_display);
        changed |= setCachedText(output_value_el_, output_display);
        const bool inputs_missing = (batch_items_.empty() ? !player_->isOpen() : false) || output_dir_.empty();
        changed |= setCachedProperty(select_hint_el_, "display", inputs_missing ? "inline-block" : "none");
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
            const int active_index = active_batch_index_.load();
            const int batch_total = static_cast<int>(batch_items_.size());
            const std::string batch_prefix = active_index >= 0 && batch_total > 0
                                                 ? std::format("{}/{} · ", active_index + 1, batch_total)
                                                 : std::string{};
            const float item_progress = total > 0
                                            ? std::clamp(static_cast<float>(current) / static_cast<float>(total), 0.0f, 1.0f)
                                            : 0.0f;
            const float progress = batch_total > 0 && active_index >= 0
                                       ? (static_cast<float>(active_index) + item_progress) /
                                             static_cast<float>(batch_total)
                                       : item_progress;
            changed |= setCachedAttribute(progress_bar_el_, "value", std::format("{:.4f}", progress));
            if (masking_active_.load()) {
                changed |= setCachedText(progress_text_el_, batch_prefix + LOCF("video_extractor.masking_progress", current, total));
            } else if (total > 0) {
                const int discarded = discarded_frames_.load();
                const std::string discard_str = discarded > 0
                                                    ? LOCF(VideoExtractor::DISCARDED_FORMAT, lfs::core::format_count(discarded))
                                                    : "";
                changed |= setCachedText(progress_text_el_,
                                         std::format("{}{}/{}{}", batch_prefix, lfs::core::format_count(current),
                                                     lfs::core::format_count(total), discard_str));
            } else {
                changed |= setCachedText(progress_text_el_, batch_prefix + LOC(VideoExtractor::STARTING));
            }
        }

        const auto snapshot = getExtractionStatusSnapshot();
        changed |= setCachedProperty(complete_section_el_, "display",
                                     snapshot.status_message == ExtractionStatusMessage::Complete && !extracting
                                         ? "flex"
                                         : "none");
        if (snapshot.status_message == ExtractionStatusMessage::Complete && !extracting) {
            std::string complete_msg;
            if (!batch_items_.empty()) {
                complete_msg = LOCF("video_extractor.batch_summary", batch_items_.size(),
                                    completed_batch_items_.load(), failed_batch_items_.load());
            } else {
                const int saved = current - discarded_frames_.load();
                const int discarded = discarded_frames_.load();
                complete_msg = std::format("{} {}", LOC(VideoExtractor::COMPLETE),
                                           LOCF(VideoExtractor::EXTRACTED, lfs::core::format_count(saved)));
                if (discarded > 0)
                    complete_msg += LOCF(VideoExtractor::DISCARDED_FORMAT, lfs::core::format_count(discarded));
            }
            if (!snapshot.warning_message.empty())
                complete_msg = LOCF("video_extractor.masking_warning", snapshot.warning_message);
            changed |= setCachedAttribute(complete_section_el_, "class", snapshot.warning_message.empty() ? "status-section status-section--success" : "status-section status-section--warning");
            changed |= setCachedText(complete_text_el_, complete_msg);
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
            changed |= setCachedText(error_text_el_,
                                     LOCF(VideoExtractor::ERROR_MSG, snapshot.error_message));
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
            // Frames are scaled first, then rotated.
            if (rotation_deg_ == 90 || rotation_deg_ == 270)
                std::swap(out_w, out_h);
        }

        changed |= setCachedText(output_resolution_el_,
                                 player_->isOpen()
                                     ? LOCF(VideoExtractor::OUTPUT_RES, out_w, out_h)
                                     : std::format("{} --", LOC(VideoExtractor::OUTPUT)));
        const char* const ext = format_selection_ == 0 ? ".png" : ".jpg";
        std::string resolved_pattern, naming_error;
        const bool valid = resolveFilenamePattern(resolved_pattern, naming_error);
        const std::string preview = valid
                                        ? io::formatFrameFilenameStem(resolved_pattern, first_source_frame_number_)
                                        : "";
        changed |= setCachedText(pattern_example_el_, valid
                                                          ? LOCF(player_->isOpen() ? VideoExtractor::EXAMPLE : "video_extractor.naming_example_pending", preview.c_str(), ext)
                                                          : "");

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
        const std::string target_id = target ? target->GetId() : std::string{};
        const std::string id = target_id.starts_with("batch-video-")
                                   ? target_id
                                   : (current && !current->GetId().empty()
                                          ? current->GetId()
                                          : target_id);

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
            handleChange(id, event_id != Rml::EventId::Blur);
            if (id == "fps-slider" && event_id == Rml::EventId::Change) {
                const float value = event.GetParameter<float>("value", fps_);
                if (std::isfinite(value))
                    fps_ = sliderExtractionFps(value, *player_, !batch_items_.empty(), fps_);
            }
            event.StopPropagation();
        }
    }

    void VideoExtractorDialog::handleClick(const std::string& id) {
        if (id == "close-btn" || id == "btn-cancel") {
            disablePanel();
        } else if (id == "btn-stop") {
            requestStopExtraction();
        } else if (id == "btn-browse-video") {
            if (extracting_.load() || pending_params_set_)
                return;
            addVideosFromBrowse(lfs::vis::gui::OpenVideoFilesDialog(video_path_));
        } else if (id == "btn-remove-video") {
            removeSelectedBatchItem();
        } else if (id == "btn-clear-videos") {
            clearBatchVideos();
        } else if (id.starts_with("batch-video-")) {
            std::size_t index = 0;
            const std::string_view index_text(id.data() + std::string_view("batch-video-").size(),
                                              id.size() - std::string_view("batch-video-").size());
            const auto [end, error] = std::from_chars(index_text.data(), index_text.data() + index_text.size(), index);
            if (error == std::errc{} && end == index_text.data() + index_text.size())
                selectBatchItem(index);
        } else if (id == "btn-browse-output") {
            if (extracting_.load() || pending_params_set_)
                return;
            const auto path = lfs::vis::gui::PickFolderDialog(output_dir_);
            if (!path.empty()) {
                output_dir_ = path;
                batch_output_root_explicit_ = true;
                refreshBatchOutputDirectories();
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
                                     std::max(0.0f, trim_end_ - MIN_TRIM_SECONDS));
            controls_dirty_ = true;
            markContentDirty();
        } else if (id == "btn-trim-end-set" && player_->isOpen()) {
            trim_end_is_auto_ = false;
            trim_end_ = std::clamp(static_cast<float>(player_->currentTime()),
                                   std::min(trim_start_ + MIN_TRIM_SECONDS, static_cast<float>(player_->duration())),
                                   static_cast<float>(player_->duration()));
            controls_dirty_ = true;
            markContentDirty();
        } else if (id == "btn-trim-reset" && player_->isOpen()) {
            trim_end_is_auto_ = true;
            trim_start_ = 0.0f;
            const double duration = player_->duration();
            trim_end_ = duration > 0.0 ? static_cast<float>(duration) : -1.0f;
            controls_dirty_ = true;
            markContentDirty();
        } else if (id == "btn-start") {
            if (batch_probe_active_.load() || batch_probe_results_ready_.load())
                return;
            if (batch_items_.empty())
                beginExtractionFromUi();
            else
                beginBatchExtractionFromUi();
        } else if (id == "overwrite-delete") {
            if (!pending_batch_params_.empty()) {
                if (removeExistingBatchOutput(pending_batch_params_))
                    launchBatchExtraction(std::move(pending_batch_params_));
                else {
                    pending_batch_params_.clear();
                    pending_params_set_ = false;
                    pending_batch_metadata_will_be_replaced_ = false;
                    pending_batch_conflict_count_ = 0;
                    if (overwrite_overlay_el_)
                        overwrite_overlay_el_->SetClass("hidden", true);
                }
                return;
            }
            if (pending_params_set_) {
                // Remove only what this extraction would write; other files stay.
                for (const auto& path : generatedExtractionFiles(pending_params_)) {
                    std::error_code ec;
                    if (!lfs::core::native_fs::remove(path, ec) && ec) {
                        pending_params_set_ = false;
                        if (overwrite_overlay_el_)
                            overwrite_overlay_el_->SetClass("hidden", true);
                        setExtractionError(LOCF(VideoExtractor::OVERWRITE_DELETE_ERROR,
                                                lfs::core::path_to_utf8(path), ec.message()),
                                           false);
                        pending_has_frame_conflicts_ = false;
                        pending_metadata_will_be_replaced_ = false;
                        controls_dirty_ = true;
                        markContentDirty();
                        return;
                    }
                }
                launchExtraction(pending_params_);
            }
        } else if (id == "overwrite-proceed") {
            if (!pending_batch_params_.empty())
                launchBatchExtraction(std::move(pending_batch_params_));
            else if (pending_params_set_)
                launchExtraction(pending_params_);
        } else if (id == "overwrite-cancel") {
            pending_params_set_ = false;
            pending_batch_params_.clear();
            pending_batch_metadata_will_be_replaced_ = false;
            pending_batch_conflict_count_ = 0;
            pending_has_frame_conflicts_ = false;
            pending_metadata_will_be_replaced_ = false;
            if (overwrite_overlay_el_)
                overwrite_overlay_el_->SetClass("hidden", true);
            controls_dirty_ = true;
            markContentDirty();
        } else if (id == "btn-complete-ok" || id == "btn-stopped-ok") {
            clearStatusMessage();
            current_frame_.store(0);
            total_frames_.store(0);
        } else if (id == "btn-error-dismiss") {
            clearErrorMessage();
        } else if (id == "btn-rotation-cw") {
            rotation_deg_ = (rotation_deg_ + 90) % 360;
            player_->setPreviewRotation(rotation_deg_);
            if (rotation_value_el_)
                rotation_value_el_->SetInnerRML(std::to_string(rotation_deg_) + "°");
            if (player_->isOpen() && player_->currentFrameHasGpuRotation()) {
                LOG_INFO("HDR preview: re-rendering current frame with libplacebo GPU rotation {} deg", rotation_deg_);
                player_->rerenderCurrentFrame();
            }
            texture_needs_update_ = true;
            markContentDirty();
        } else if (id == "btn-rotation-ccw") {
            rotation_deg_ = (rotation_deg_ + 270) % 360;
            player_->setPreviewRotation(rotation_deg_);
            if (rotation_value_el_)
                rotation_value_el_->SetInnerRML(std::to_string(rotation_deg_) + "°");
            if (player_->isOpen() && player_->currentFrameHasGpuRotation()) {
                LOG_INFO("HDR preview: re-rendering current frame with libplacebo GPU rotation {} deg", rotation_deg_);
                player_->rerenderCurrentFrame();
            }
            texture_needs_update_ = true;
            markContentDirty();
        }
    }

    void VideoExtractorDialog::handleChange(const std::string& id, const bool explicit_edit) {
        Rml::Element* changed_control = nullptr;

        if (id == "naming-select") {
            naming_selection_ = naming_select_el_ ? std::clamp(naming_select_el_->GetSelection(), 0, 4) : naming_selection_;
        } else if (id == "mode-select") {
            mode_selection_ = mode_select_el_ ? mode_select_el_->GetSelection() : mode_selection_;
        } else if (id == "format-select") {
            format_selection_ = format_select_el_ ? format_select_el_->GetSelection() : format_selection_;
        } else if (id == "resolution-select") {
            resolution_mode_ = resolution_select_el_ ? resolution_select_el_->GetSelection() : resolution_mode_;
        } else if (id == "scale-select") {
            scale_selection_ = scale_select_el_ ? scale_select_el_->GetSelection() : scale_selection_;
        } else if (id == "fps-slider") {
            fps_ = sliderExtractionFps(readFloatValue(fps_slider_el_, fps_), *player_,
                                       !batch_items_.empty(), fps_);
            changed_control = fps_slider_el_;
        } else if (id == "fps-value") {
            const std::string value = controlValue(fps_value_el_);
            float parsed = 0.0f;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (error == std::errc{} && end == value.data() + value.size() && std::isfinite(parsed)) {
                fps_ = std::clamp(parsed, 0.1f,
                                  maximumRequestedExtractionFps(*player_, !batch_items_.empty(), fps_));
            }
            changed_control = fps_value_el_;
        } else if (id == "interval-input") {
            frame_interval_ = std::clamp(readIntValue(interval_input_el_, frame_interval_), 1, 100);
            changed_control = interval_input_el_;
        } else if (id == "quality-slider") {
            jpg_quality_ = std::clamp(readIntValue(quality_slider_el_, jpg_quality_), 50, 100);
            changed_control = quality_slider_el_;
        } else if (id == "window-candidates-select") {
            if (window_candidates_select_el_) {
                const auto* select = dynamic_cast<const Rml::ElementFormControlSelect*>(window_candidates_select_el_);
                if (select) {
                    const int values[] = {-1, 3, 5, 10, 20, 50, 0};
                    window_candidates_target_ = values[std::clamp(select->GetSelection(), 0, 6)];
                }
            }
            changed_control = window_candidates_select_el_;
        } else if (id == "masking-toggle" || id == "mask-circle" || id == "mask-recenter" ||
                   id == "mask-view-fov" || id == "mask-dilation" || id == "mask-require-operator" ||
                   id == "mask-model" || id == "mask-fov" || id == "mask-manual-seed" ||
                   id == "mask-seed-x" || id == "mask-seed-y" || id == "mask-seed-frame") {
            changed_control = document_ ? document_->GetElementById(id) : nullptr;
        } else if (id == "sharpness-toggle") {
            changed_control = sharpness_toggle_el_;
        } else if (id == "sharpness-algorithm-select") {
            changed_control = sharpness_algorithm_select_el_;
        } else if (id == "sharpness-mode-select") {
            changed_control = sharpness_mode_select_el_;
        } else if (id == "sharpness-threshold-slider") {
            // value read in syncControls
            changed_control = sharpness_threshold_slider_el_;
        } else if (id == "generate-metadata") {
            changed_control = generate_metadata_el_;
        } else if (id == "continue-batch-on-failure") {
            continue_batch_on_failure_ = continue_batch_on_failure_el_ &&
                                         continue_batch_on_failure_el_->HasAttribute("checked");
            changed_control = continue_batch_on_failure_el_;
        } else if (id == "hdr-to-sdr") {
            if (extracting_.load()) {
                (void)setCachedChecked(hdr_to_sdr_el_, hdr_to_sdr_);
                return;
            }
            hdr_to_sdr_ = hdr_to_sdr_el_ && hdr_to_sdr_el_->HasAttribute("checked");
            player_->setHdrToSdr(hdr_to_sdr_);
            if (player_->isOpen()) {
                player_->rerenderCurrentFrame();
                texture_needs_update_ = true;
                preview_src_.clear();
            }
            changed_control = hdr_to_sdr_el_;
        } else {
            applyTextInput(id);
            if (id == "trim-start-input")
                changed_control = trim_start_input_el_;
            else if (id == "trim-end-input") {
                if (explicit_edit)
                    trim_end_is_auto_ = false;
                changed_control = trim_end_input_el_;
            } else if (id == "custom-width-input")
                changed_control = custom_width_input_el_;
            else if (id == "custom-height-input")
                changed_control = custom_height_input_el_;
            else if (id == "pattern-input")
                changed_control = pattern_input_el_;
            else if (id == "first-source-frame-number-input")
                changed_control = first_source_frame_number_input_el_;
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
            trim_start_ = std::clamp(time, 0.0f, std::max(0.0f, trim_end_ - MIN_TRIM_SECONDS));
        else if (target == TimelineDragTarget::TrimEnd) {
            trim_end_is_auto_ = false;
            trim_end_ = std::clamp(time, std::min(trim_start_ + MIN_TRIM_SECONDS, duration), duration);
        }

        controls_dirty_ = true;
        markContentDirty();
    }

    void VideoExtractorDialog::applyTextInput(const std::string& id) {
        if (id == "trim-start-input" && player_->isOpen()) {
            trim_start_ = std::clamp(readFloatValue(trim_start_input_el_, trim_start_), 0.0f,
                                     std::max(0.0f, trim_end_ - MIN_TRIM_SECONDS));
        } else if (id == "trim-end-input" && player_->isOpen()) {
            trim_end_ = std::clamp(readFloatValue(trim_end_input_el_, trim_end_),
                                   std::min(trim_start_ + MIN_TRIM_SECONDS, static_cast<float>(player_->duration())),
                                   static_cast<float>(player_->duration()));
        } else if (id == "custom-width-input") {
            custom_width_ = std::max(16, readIntValue(custom_width_input_el_, custom_width_));
        } else if (id == "custom-height-input") {
            custom_height_ = std::max(16, readIntValue(custom_height_input_el_, custom_height_));
        } else if (id == "pattern-input" && pattern_input_el_ && naming_selection_ == 4) {
            filename_pattern_ = controlValue(pattern_input_el_);
        } else if (id == "first-source-frame-number-input") {
            first_source_frame_number_ = readNonNegativeIntValue(
                first_source_frame_number_input_el_, first_source_frame_number_);
        }
    }

    bool VideoExtractorDialog::resolveFilenamePattern(std::string& pattern, std::string& error) const {
        return resolveFilenamePattern(video_path_, pattern, error);
    }

    bool VideoExtractorDialog::resolveFilenamePattern(const std::filesystem::path& source_path,
                                                      std::string& pattern, std::string& error) const {
        static constexpr std::array<std::string_view, 4> presets{
            "{video}_frame_%05d", "{video}_%05d", "frame_%05d", "%05d"};
        const std::string_view source = naming_selection_ >= 0 && naming_selection_ < 4
                                            ? presets[naming_selection_]
                                            : std::string_view(filename_pattern_);
        const std::string video_name = source_path.empty() ? "video" : lfs::core::path_to_utf8(source_path.stem());
        pattern.clear();
        error.clear();
        for (size_t i = 0; i < source.size();) {
            if (source.substr(i, 2) == "{{" || source.substr(i, 2) == "}}") {
                pattern += source[i];
                i += 2;
            } else if (source.substr(i, 7) == "{video}") {
                // Insert the source stem literally: percent signs must not become frame tokens.
                for (const char c : video_name) {
                    pattern += c;
                    if (c == '%')
                        pattern += '%';
                }
                i += 7;
            } else if (source[i] == '{' || source[i] == '}') {
                error = LOC("video_extractor.naming_error_placeholder");
                return false;
            } else {
                pattern += source[i++];
            }
        }
        // Keep platform-specific filesystem limits with the existing writer. In particular,
        // a UTF-8 byte limit would incorrectly reject valid Unicode names on Windows.
        for (const int number : {1, std::numeric_limits<int>::max()}) {
            const std::string stem = io::formatFrameFilenameStem(pattern, number);
            bool invalid = stem.empty() || stem.find('\0') != std::string::npos || stem.find('/') != std::string::npos;
#ifdef _WIN32
            // The writer appends the image extension, so a trailing dot or space
            // in the stem is internal to the final filename and is valid.
            invalid |= std::any_of(stem.begin(), stem.end(), [](const unsigned char c) {
                return c < 32 || std::string_view("<>:\"\\|?*").find(static_cast<char>(c)) != std::string_view::npos;
            });
            std::string base = stem.substr(0, stem.find('.'));
            while (!base.empty() && base.back() == ' ')
                base.pop_back();
            for (char& c : base)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            const std::string_view suffix = base.size() >= 3 ? std::string_view(base).substr(3) : std::string_view{};
            invalid |= base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" ||
                       base == "CONIN$" || base == "CONOUT$" ||
                       ((base.starts_with("COM") || base.starts_with("LPT")) &&
                        ((suffix.size() == 1 && suffix[0] >= '1' && suffix[0] <= '9') ||
                         suffix == "\xc2\xb9" || suffix == "\xc2\xb2" || suffix == "\xc2\xb3"));
#endif
            if (invalid) {
                error = LOC("video_extractor.naming_error_filename");
                return false;
            }
        }
        return true;
    }

    bool VideoExtractorDialog::collectMaskingOptions(preprocessing::VideoMaskExtractionOptions& options,
                                                     std::string& error) const {
        options = {};
        options.enabled = masking_toggle_el_ && masking_toggle_el_->HasAttribute("checked");
        if (!options.enabled)
            return true;
        const auto checked = [](const Rml::Element* el) { return el && el->HasAttribute("checked"); };
        const auto number = [](const Rml::Element* el, auto& value) {
            const auto text = controlValue(el);
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            return ec == std::errc{} && end == text.data() + text.size() && std::isfinite(double(value));
        };
        options.masking.require_operator = checked(mask_require_operator_el_);
        if (!number(mask_dilation_el_, options.masking.dilation_radius) ||
            options.masking.dilation_radius < 0 || options.masking.dilation_radius > 128) {
            error = LOC("video_extractor.masking_invalid_dilation");
            return false;
        }
        options.masking.weights = lfs::core::utf8_to_path(controlValue(mask_model_el_));
        if (!options.masking.weights.empty() && !lfs::core::native_fs::is_regular_file(options.masking.weights)) {
            error = LOC("video_extractor.masking_invalid_model");
            return false;
        }
        if (checked(mask_circle_el_)) {
            options.circle = {0.5, 0.5, 0.5, 0.5};
            options.masking.recenter_fisheye = checked(mask_recenter_el_);
            if (!number(mask_fov_el_, options.masking.fisheye_fov_degrees) ||
                !(options.masking.fisheye_fov_degrees > 0 && options.masking.fisheye_fov_degrees < 360)) {
                error = LOC("video_extractor.masking_invalid_fov");
                return false;
            }
        }
        if (options.masking.recenter_fisheye &&
            (!number(mask_view_fov_el_, options.masking.view_fov_degrees) ||
             !(options.masking.view_fov_degrees > 0 && options.masking.view_fov_degrees < 360))) {
            error = LOC("video_extractor.masking_invalid_fov");
            return false;
        }
        if (checked(mask_manual_seed_el_)) {
            preprocessing::VideoMaskSeedPrompt seed;
            float x = 0, y = 0;
            if (!number(mask_seed_x_el_, x) || !number(mask_seed_y_el_, y) ||
                !number(mask_seed_frame_el_, seed.frame) || !(x >= 0 && x <= 1 && y >= 0 && y <= 1) || seed.frame < 0) {
                error = LOC("video_extractor.masking_invalid_seed");
                return false;
            }
            seed.points.push_back({x, y, 1});
            options.masking.prompt = std::move(seed);
        }
        return true;
    }

    bool VideoExtractorDialog::buildBatchParams(std::vector<VideoExtractionParams>& params,
                                                std::string& error) {
        applyTextInput("trim-start-input");
        applyTextInput("trim-end-input");
        saveSelectedBatchItem();
        params.clear();
        error.clear();
        if (batch_items_.empty()) {
            error = LOC("video_extractor.batch_no_videos");
            return false;
        }
        if (output_dir_.empty()) {
            error = LOC("video_extractor.batch_no_output_root");
            return false;
        }

        applyTextInput("custom-width-input");
        applyTextInput("custom-height-input");
        applyTextInput("pattern-input");
        applyTextInput("first-source-frame-number-input");
        refreshBatchOutputDirectories();

        VideoExtractionParams shared;
        if (!collectMaskingOptions(shared.masking, error))
            return false;
        shared.mode = mode_selection_ == 0 ? io::ExtractionMode::FPS : io::ExtractionMode::INTERVAL;
        shared.fps = static_cast<double>(fps_);
        shared.frame_interval = frame_interval_;
        shared.format = format_selection_ == 0 ? io::ImageFormat::PNG : io::ImageFormat::JPG;
        shared.jpg_quality = jpg_quality_;
        static constexpr std::array<io::ResolutionMode, 3> resolution_modes{
            io::ResolutionMode::Original, io::ResolutionMode::Scale, io::ResolutionMode::Custom};
        shared.resolution_mode = resolution_modes[std::clamp(resolution_mode_, 0, 2)];
        shared.scale = SCALE_VALUES[std::clamp(scale_selection_, 0, static_cast<int>(SCALE_VALUES.size() - 1))];
        shared.custom_width = custom_width_;
        shared.custom_height = custom_height_;
        shared.first_source_frame_number = first_source_frame_number_;
        shared.sharpness_enabled = sharpness_toggle_el_ && sharpness_toggle_el_->HasAttribute("checked");
        if (sharpness_algorithm_select_el_) {
            static constexpr io::SharpnessAlgorithm algorithms[]{
                io::SharpnessAlgorithm::COMBINED, io::SharpnessAlgorithm::TENENGRAD,
                io::SharpnessAlgorithm::LAPLACIAN};
            shared.sharpness_algorithm = algorithms[std::clamp(sharpness_algorithm_select_el_->GetSelection(), 0, 2)];
        }
        shared.sharpness_window_mode = sharpness_mode_select_el_ && sharpness_mode_select_el_->GetSelection() == 1;
        shared.window_candidates_target = window_candidates_target_;
        shared.sharpness_threshold = static_cast<double>(readIntValue(sharpness_threshold_slider_el_, 10));
        shared.generate_metadata = generate_metadata_el_ && generate_metadata_el_->HasAttribute("checked");

        params.reserve(batch_items_.size());
        for (const auto& item : batch_items_) {
            VideoExtractionParams item_params = shared;
            item_params.video_path = item.video_path;
            item_params.output_dir = item.output_dir;
            item_params.start_time = item.start_time;
            const bool allow_stream_end = item.end_time_is_auto &&
                                          item_params.mode == io::ExtractionMode::INTERVAL &&
                                          !(item_params.sharpness_enabled && item_params.sharpness_window_mode);
            // An unselected item keeps its auto end; resolve it like the player's trim end.
            const double trim_end = item.end_time < 0.0 ? item.duration : item.end_time;
            item_params.end_time = item.duration > 0.0
                                       ? io::extractionEndTime(static_cast<float>(trim_end), item.duration,
                                                               allow_stream_end)
                                       : item.end_time;
            item_params.rotation = item.rotation;
            item_params.convert_hdr_to_sdr =
                hdr_to_sdr_ && item.is_hdr && item.hdr_to_sdr_supported;
            if (item.source_fps > 0.0 && item_params.mode == io::ExtractionMode::FPS)
                item_params.fps = std::min(item_params.fps, item.source_fps);
            if (!resolveFilenamePattern(item.video_path, item_params.filename_pattern, error)) {
                error = std::format("{}: {}", lfs::core::path_to_utf8(item.video_path.filename()), error);
                return false;
            }
            params.push_back(std::move(item_params));
        }
        return true;
    }

    bool VideoExtractorDialog::removeExistingBatchOutput(const std::vector<VideoExtractionParams>& params) {
        for (const auto& item_params : params) {
            for (const auto& path : generatedExtractionFiles(item_params)) {
                std::error_code ec;
                if (!lfs::core::native_fs::remove(path, ec) && ec) {
                    setExtractionError(LOCF(VideoExtractor::OVERWRITE_DELETE_ERROR,
                                            lfs::core::path_to_utf8(path), ec.message()),
                                       false);
                    return false;
                }
            }
        }
        return true;
    }

    void VideoExtractorDialog::beginBatchExtractionFromUi() {
        if (extracting_.load() || batch_probe_active_.load() || batch_probe_results_ready_.load())
            return;
        std::vector<VideoExtractionParams> params;
        std::string error;
        if (!buildBatchParams(params, error)) {
            setExtractionError(error, false);
            controls_dirty_ = true;
            markContentDirty();
            return;
        }

        bool metadata_will_be_replaced = false;
        int conflict_count = 0;
        for (std::size_t i = 0; i < params.size(); ++i) {
            const auto& item_params = params[i];
            const auto existing = generatedExtractionFiles(item_params);
            {
                std::lock_guard lock(batch_mutex_);
                if (i < batch_items_.size())
                    batch_items_[i].has_existing_output = !existing.empty();
            }
            if (existing.empty())
                continue;
            ++conflict_count;
            metadata_will_be_replaced |= item_params.generate_metadata &&
                                         std::any_of(existing.begin(), existing.end(), [](const std::filesystem::path& path) {
                                             return lfs::core::path_to_utf8(path.filename()) == "extraction_metadata.json";
                                         });
        }
        batch_list_dirty_.store(true);
        if (conflict_count > 0) {
            pending_batch_params_ = std::move(params);
            pending_params_set_ = true;
            pending_batch_metadata_will_be_replaced_ = metadata_will_be_replaced;
            pending_batch_conflict_count_ = conflict_count;
            syncOverwritePrompt();
            if (overwrite_overlay_el_)
                overwrite_overlay_el_->SetClass("hidden", false);
            return;
        }
        launchBatchExtraction(std::move(params));
    }

    void VideoExtractorDialog::launchBatchExtraction(std::vector<VideoExtractionParams> params) {
        pending_params_set_ = false;
        pending_batch_params_.clear();
        pending_has_frame_conflicts_ = false;
        pending_metadata_will_be_replaced_ = false;
        pending_batch_metadata_will_be_replaced_ = false;
        pending_batch_conflict_count_ = 0;
        if (overwrite_overlay_el_)
            overwrite_overlay_el_->SetClass("hidden", true);
        stop_extraction_requested_.store(false);
        extracting_.store(true);
        current_frame_.store(0);
        total_frames_.store(0);
        discarded_frames_.store(0);
        completed_batch_items_.store(0);
        failed_batch_items_.store(0);
        active_batch_index_.store(-1);
        continue_batch_on_failure_ = continue_batch_on_failure_el_ && continue_batch_on_failure_el_->HasAttribute("checked");
        {
            std::lock_guard lock(batch_mutex_);
            for (auto& item : batch_items_) {
                item.has_existing_output = false;
                item.status = BatchItemStatus::Pending;
                item.warning.clear();
                item.error.clear();
            }
        }
        batch_list_dirty_.store(true);
        clearExtractionStatus();
        startBatchExtraction(std::move(params));
        controls_dirty_ = true;
        markContentDirty();
    }

    void VideoExtractorDialog::beginExtractionFromUi() {
        applyTextInput("trim-start-input");
        applyTextInput("trim-end-input");
        applyTextInput("custom-width-input");
        applyTextInput("custom-height-input");
        applyTextInput("pattern-input");
        applyTextInput("first-source-frame-number-input");

        if (!player_->isOpen() || output_dir_.empty() || extracting_.load() ||
            batch_probe_active_.load() || batch_probe_results_ready_.load())
            return;

        VideoExtractionParams params;
        std::string masking_error;
        if (!collectMaskingOptions(params.masking, masking_error)) {
            setExtractionError(masking_error);
            return;
        }
        params.video_path = video_path_;
        params.output_dir = output_dir_;
        params.mode = mode_selection_ == 0 ? io::ExtractionMode::FPS : io::ExtractionMode::INTERVAL;
        params.fps = static_cast<double>(fps_);
        params.frame_interval = frame_interval_;
        params.format = format_selection_ == 0 ? io::ImageFormat::PNG : io::ImageFormat::JPG;
        params.jpg_quality = jpg_quality_;
        params.start_time = static_cast<double>(trim_start_);
        static constexpr std::array<io::ResolutionMode, 3> RES_MODES{
            io::ResolutionMode::Original,
            io::ResolutionMode::Scale,
            io::ResolutionMode::Custom};
        params.resolution_mode = RES_MODES[std::clamp(resolution_mode_, 0, 2)];
        params.scale = SCALE_VALUES[std::clamp(scale_selection_, 0, static_cast<int>(SCALE_VALUES.size() - 1))];
        params.custom_width = custom_width_;
        params.custom_height = custom_height_;
        params.first_source_frame_number = first_source_frame_number_;
        std::string naming_error;
        if (!resolveFilenamePattern(params.filename_pattern, naming_error)) {
            controls_dirty_ = true;
            markContentDirty();
            return;
        }
        params.sharpness_enabled = sharpness_toggle_el_ && sharpness_toggle_el_->HasAttribute("checked");
        if (sharpness_algorithm_select_el_) {
            static constexpr io::SharpnessAlgorithm ALGO_MAP[] = {
                io::SharpnessAlgorithm::COMBINED,
                io::SharpnessAlgorithm::TENENGRAD,
                io::SharpnessAlgorithm::LAPLACIAN};
            params.sharpness_algorithm = ALGO_MAP[std::clamp(sharpness_algorithm_select_el_->GetSelection(), 0, 2)];
        }
        params.sharpness_window_mode = sharpness_mode_select_el_ && sharpness_mode_select_el_->GetSelection() == 1;
        params.end_time = io::extractionEndTime(
            trim_end_, player_->duration(),
            trim_end_is_auto_ && params.mode == io::ExtractionMode::INTERVAL &&
                !(params.sharpness_enabled && params.sharpness_window_mode));
        params.window_candidates_target = window_candidates_target_;
        params.sharpness_threshold = static_cast<double>(readIntValue(sharpness_threshold_slider_el_, 10));
        params.generate_metadata = generate_metadata_el_ && generate_metadata_el_->HasAttribute("checked");
        params.convert_hdr_to_sdr = hdr_to_sdr_;
        params.rotation = rotation_deg_;

        // Ask before replacing frames matching this naming or metadata this run will write.
        const auto existing_files = generatedExtractionFiles(params);
        if (!existing_files.empty()) {
            pending_params_ = params;
            pending_params_set_ = true;
            pending_has_frame_conflicts_ = std::any_of(
                existing_files.begin(), existing_files.end(), [](const std::filesystem::path& path) {
                    return lfs::core::path_to_utf8(path.filename()) != "extraction_metadata.json";
                });
            pending_metadata_will_be_replaced_ = params.generate_metadata && std::any_of(
                                                                                 existing_files.begin(), existing_files.end(), [](const std::filesystem::path& path) {
                                                                                     return lfs::core::path_to_utf8(path.filename()) == "extraction_metadata.json";
                                                                                 });
            syncOverwritePrompt();
            if (overwrite_overlay_el_)
                overwrite_overlay_el_->SetClass("hidden", false);
            return;
        }

        launchExtraction(params);
    }

    void VideoExtractorDialog::launchExtraction(const VideoExtractionParams& params) {
        pending_params_set_ = false;
        pending_has_frame_conflicts_ = false;
        pending_metadata_will_be_replaced_ = false;
        if (overwrite_overlay_el_)
            overwrite_overlay_el_->SetClass("hidden", true);
        stop_extraction_requested_.store(false);
        extracting_.store(true);
        current_frame_.store(0);
        total_frames_.store(0);
        discarded_frames_.store(0);
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
