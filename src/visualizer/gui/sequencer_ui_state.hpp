/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "io/video/video_export_options.hpp"

#include <string>
#include <utility>

namespace lfs::vis::gui::panels {

    struct SequencerUIState {
        bool show_camera_path = true;
        bool snap_to_grid = false;
        float snap_interval = 0.5f;
        float playback_speed = 1.0f;
        bool follow_playback = false;
        bool show_pip_preview = true;
        float pip_preview_scale = 1.0f;
        bool show_film_strip = true;
        bool equirectangular = false;
        float sequence_fps = 24.0f;
        lfs::io::video::VideoPreset preset = lfs::io::video::VideoPreset::YOUTUBE_1080P;
        int custom_width = 1920;
        int custom_height = 1080;
        int framerate = 30;
        int quality = 18;
        lfs::io::video::VideoReconstructionSelection reconstruction{};

        // Snapshot the saved selection for both the Sequencer button and Python.
        [[nodiscard]] lfs::core::events::cmd::SequencerExportVideo videoExportRequest(
            const int width, const int height, const int fps, const int crf,
            std::string path = {}, const bool include_provenance = true) const {
            return {
                .width = width,
                .height = height,
                .framerate = fps,
                .crf = crf,
                .path = std::move(path),
                .include_provenance = include_provenance,
                .reconstruction_backend_id = reconstruction.backend_id,
                .reconstruction_preset_id = reconstruction.preset_id,
                .reconstruction_fallback = std::string(
                    lfs::io::video::videoReconstructionFallbackId(reconstruction.fallback)),
            };
        }

        [[nodiscard]] int outputWidth() const {
            if (preset == lfs::io::video::VideoPreset::CUSTOM)
                return custom_width;
            return lfs::io::video::getPresetInfo(preset).width;
        }

        [[nodiscard]] int outputHeight() const {
            if (preset == lfs::io::video::VideoPreset::CUSTOM)
                return custom_height;
            return lfs::io::video::getPresetInfo(preset).height;
        }
    };

} // namespace lfs::vis::gui::panels
