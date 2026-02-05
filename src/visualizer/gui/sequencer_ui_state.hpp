/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/video/video_export_options.hpp"

namespace lfs::vis::gui::panels {

    struct SequencerUIState {
        bool show_camera_path = true;
        bool snap_to_grid = false;
        float snap_interval = 0.5f;
        float playback_speed = 1.0f;
        bool follow_playback = false;
        float pip_preview_scale = 1.0f;
        lfs::io::video::VideoPreset preset = lfs::io::video::VideoPreset::YOUTUBE_1080P;
        int custom_width = 1920;
        int custom_height = 1080;
        int framerate = 30;
        int quality = 18;
    };

} // namespace lfs::vis::gui::panels
