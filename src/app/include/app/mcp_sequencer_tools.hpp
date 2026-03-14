/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <string>

namespace lfs::mcp {
    class ToolRegistry;
}

namespace lfs::python {
    struct SequencerUIStateData;
}

namespace lfs::vis {
    class SequencerController;
    class Visualizer;
}

namespace lfs::app {

    struct SequencerToolBackend {
        std::function<std::expected<void, std::string>()> ensure_ready;
        std::function<vis::SequencerController*()> controller;
        std::function<bool()> is_visible;
        std::function<void(bool)> set_visible;
        std::function<python::SequencerUIStateData*()> ui_state;
        std::function<void()> add_keyframe;
        std::function<void()> update_selected_keyframe;
        std::function<void(size_t)> select_keyframe;
        std::function<void(size_t)> go_to_keyframe;
        std::function<void(size_t)> delete_keyframe;
        std::function<void(size_t, int)> set_keyframe_easing;
        std::function<void()> play_pause;
        std::function<void()> clear;
        std::function<bool(const std::string&)> save_path;
        std::function<bool(const std::string&)> load_path;
        std::function<void(float)> set_playback_speed;
    };

    void register_gui_sequencer_tools(
        mcp::ToolRegistry& registry,
        vis::Visualizer* viewer,
        SequencerToolBackend backend);

} // namespace lfs::app
