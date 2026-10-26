/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/sequencer_ui_state.hpp"
#include "io/video/video_frame_rate.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::vis::gui {

    // A dialog draft owns only export preferences. Editing or discarding it must
    // not change playback, projection, keyframes, or the persisted UI state.
    struct SequencerExportSettings {
        io::video::VideoPreset preset;
        std::string width;
        std::string height;
        bool aspect_locked = true;
        int aspect_width;
        int aspect_height;
        std::string framerate;
        std::string quality;
        int framing;
        int camera_width;
        int camera_height;
        bool equirectangular;

        explicit SequencerExportSettings(const panels::SequencerUIState& state)
            : preset(state.preset),
              width(std::to_string(state.outputWidth())),
              height(std::to_string(state.outputHeight())),
              aspect_width(state.outputWidth()),
              aspect_height(state.outputHeight()),
              framerate(io::video::videoFrameRateLabel({state.framerate, state.framerate_denominator})),
              quality(std::to_string(state.quality)),
              framing(state.export_fit),
              camera_width(std::clamp(state.camera_width, 1, 8192)),
              camera_height(std::clamp(state.camera_height, 1, 8192)),
              equirectangular(state.equirectangular) {}

        void selectPreset(const io::video::VideoPreset next) {
            preset = next;
            if (next != io::video::VideoPreset::CUSTOM) {
                const auto info = io::video::getPresetInfo(next);
                width = std::to_string(info.width);
                height = std::to_string(info.height);
                captureAspect();
            }
            // Resolution presets do not silently change the output frame rate
            // or quality. Both remain explicit controls in this dialog.
        }

        [[nodiscard]] static std::optional<int> parseDimension(const std::string_view text) {
            const auto value = parseInteger(text);
            if (!value || *value < 16 || *value > 8192 || *value % 2 != 0)
                return std::nullopt;
            return value;
        }

        void captureAspect() {
            const auto w = parseDimension(width);
            const auto h = parseDimension(height);
            if (w && h) {
                aspect_width = *w;
                aspect_height = *h;
            }
        }

        void setAspectLocked(const bool locked) {
            if (locked && !aspect_locked) {
                if (!parseDimension(width) || !parseDimension(height))
                    return;
                captureAspect();
            }
            aspect_locked = locked;
        }

        // Keep the original ratio as the anchor: repeatedly editing either
        // side must not accumulate rounding drift. Only the dependent side is
        // rounded, to the nearest even pixel required by the video encoder.
        void editDimension(const bool editing_width, std::string value) {
            (editing_width ? width : height) = std::move(value);
            const auto dimension = parseDimension(editing_width ? width : height);
            if (aspect_locked && dimension && aspect_width > 0 && aspect_height > 0) {
                const int numerator = editing_width ? aspect_height : aspect_width;
                const int denominator = editing_width ? aspect_width : aspect_height;
                const auto product = static_cast<int64_t>(*dimension) * numerator;
                const auto rounded = ((product + denominator) / (2LL * denominator)) * 2;
                // Retain an out-of-range result visibly; validation disables
                // submission rather than clamping and changing the ratio.
                (editing_width ? height : width) = std::to_string(rounded);
            }
            reconcilePreset();
        }

        void reconcilePreset() {
            if (preset == io::video::VideoPreset::CUSTOM)
                return;
            const auto info = io::video::getPresetInfo(preset);
            if (parseDimension(width) != info.width || parseDimension(height) != info.height)
                preset = io::video::VideoPreset::CUSTOM;
        }

        void swapDimensions() {
            std::swap(width, height);
            std::swap(aspect_width, aspect_height);
            reconcilePreset();
        }

        [[nodiscard]] bool needsFraming() const {
            const auto w = parseDimension(width);
            const auto h = parseDimension(height);
            return !equirectangular && w && h &&
                   static_cast<int64_t>(*w) * camera_height != static_cast<int64_t>(*h) * camera_width;
        }

        [[nodiscard]] std::string aspectLabel() const {
            const auto w = parseDimension(width);
            const auto h = parseDimension(height);
            if (!w || !h)
                return "";
            const int divisor = std::gcd(*w, *h);
            return std::to_string(*w / divisor) + ":" + std::to_string(*h / divisor);
        }

        inline static constexpr auto COMMON_FRAME_RATES = io::video::VIDEO_FRAME_RATES;

        bool selectFrameratePreset(const std::string_view value) {
            for (const auto& preset : COMMON_FRAME_RATES) {
                if (value == preset.label) {
                    framerate = preset.label;
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] std::string frameratePreset() const {
            const auto value = io::video::parseVideoFrameRate(framerate);
            for (const auto& preset : COMMON_FRAME_RATES)
                if (value && *value == preset.rate)
                    return preset.label;
            return "";
        }

        [[nodiscard]] std::optional<int> qualitySliderValue() const {
            const auto crf = parseInteger(quality);
            if (!crf || *crf > 51)
                return std::nullopt;
            return 51 - *crf;
        }

        bool selectQualitySlider(std::string_view value) {
            // RmlUi's range input returns fixed-point values (e.g. 33.000000),
            // even with step=1. printf may use a comma under LC_NUMERIC.
            // Accept integral values only, without float casts or rounding.
            if (const auto dot = value.find_first_of(".,"); dot != std::string_view::npos) {
                if (dot + 1 == value.size() || value.find_first_not_of('0', dot + 1) != std::string_view::npos)
                    return false;
                value = value.substr(0, dot);
            }
            const auto slider = parseInteger(value);
            if (!slider || *slider > 51)
                return false;
            quality = std::to_string(51 - *slider);
            return true;
        }

        [[nodiscard]] static std::optional<int> parseInteger(std::string_view text) {
            const auto first = text.find_first_not_of(" \t");
            if (first == std::string_view::npos)
                return std::nullopt;
            text = text.substr(first, text.find_last_not_of(" \t") - first + 1);
            if (text.front() < '0' || text.front() > '9')
                return std::nullopt;
            int value = 0;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (error != std::errc{} || end != text.data() + text.size())
                return std::nullopt;
            return value;
        }

        [[nodiscard]] std::optional<io::video::VideoExportOptions> validated() const {
            if (static_cast<int>(preset) < 0 || static_cast<int>(preset) >= io::video::getPresetCount())
                return std::nullopt;
            const auto w = parseDimension(width);
            const auto h = parseDimension(height);
            const auto fps = io::video::parseVideoFrameRate(framerate);
            const auto crf = parseInteger(quality);
            if (!w || !h || !fps || !crf)
                return std::nullopt;
            const auto info = io::video::getPresetInfo(preset);
            io::video::VideoExportOptions result;
            result.preset = *w == info.width && *h == info.height
                                ? preset
                                : io::video::VideoPreset::CUSTOM;
            result.width = *w;
            result.height = *h;
            result.framerate = fps->numerator;
            result.framerate_denominator = fps->denominator;
            if (framing < 0 || framing > 2)
                return std::nullopt;
            result.framing = needsFraming() ? static_cast<io::video::VideoFrameFit>(framing) : io::video::VideoFrameFit::FIT;
            result.camera_width = equirectangular ? 0 : camera_width;
            result.camera_height = equirectangular ? 0 : camera_height;
            result.crf = *crf;
            if (!io::video::validateVideoEncodingOptions(result))
                return std::nullopt;
            return result;
        }

        static bool apply(const io::video::VideoExportOptions& options, panels::SequencerUIState& state) {
            const bool changed = state.preset != options.preset || state.custom_width != options.width ||
                                 state.custom_height != options.height || state.framerate != options.framerate ||
                                 state.framerate_denominator != options.framerate_denominator ||
                                 state.export_fit != static_cast<int>(options.framing) || state.quality != options.crf;
            state.preset = options.preset;
            state.custom_width = options.width;
            state.custom_height = options.height;
            state.framerate = options.framerate;
            state.framerate_denominator = options.framerate_denominator;
            state.export_fit = static_cast<int>(options.framing);
            state.quality = options.crf;
            return changed;
        }
    };

} // namespace lfs::vis::gui
