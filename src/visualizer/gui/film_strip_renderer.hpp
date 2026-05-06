/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/vulkan_ui_texture.hpp"
#include "sequencer/keyframe.hpp"
#include <array>
#include <core/export.hpp>
#include <cstdint>
#include <optional>
#include <vector>

namespace lfs::vis {
    class SequencerController;
    class RenderingManager;
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis::gui {

    class LFS_VIS_API FilmStripRenderer {
    public:
        static constexpr int THUMB_WIDTH = 128;
        static constexpr int THUMB_HEIGHT = 72;
        static constexpr int MAX_SLOTS = 64;
        static constexpr int MAX_RENDERS_PER_FRAME = 6;
        static constexpr int BURST_RENDERS_PER_FRAME = 20;
        static constexpr int BURST_FRAMES = 3;
        static constexpr float STRIP_HEIGHT = 56.0f;
        static constexpr float THUMB_PADDING = 4.0f;
        static constexpr float SPROCKET_W = 4.0f;
        static constexpr float SPROCKET_H = 3.0f;
        static constexpr float SPROCKET_SPACING = 10.0f;
        static constexpr float SPROCKET_ROUNDING = 1.0f;
        static constexpr float SPROCKET_INSET = 0.5f;

        struct RenderOptions {
            float panel_x = 0.0f;
            float panel_width = 0.0f;
            float timeline_x = 0.0f;
            float timeline_width = 0.0f;
            float strip_y = 0.0f;
            float mouse_x = 0.0f;
            float mouse_y = 0.0f;
            float zoom_level = 1.0f;
            float pan_offset = 0.0f;
            float display_end_time = 0.0f;
            std::optional<sequencer::KeyframeId> selected_keyframe_id;
            std::optional<sequencer::KeyframeId> hovered_keyframe_id;
            std::optional<float> selected_keyframe_time;
            std::optional<float> hovered_keyframe_time;
        };

        struct HoverState {
            float exact_time = 0.0f;
            float sample_time = 0.0f;
            float interval_start_time = 0.0f;
            float interval_end_time = 0.0f;
            float guide_x = 0.0f;
            float thumb_min_x = 0.0f;
            float thumb_max_x = 0.0f;
            bool over_thumbnail = false;
        };

        struct ThumbInfo {
            float time = 0.0f;
            float interval_start_time = 0.0f;
            float interval_end_time = 0.0f;
            float screen_x = 0.0f;
            float screen_width = 0.0f;
            float screen_center_x = 0.0f;
            int slot_idx = -1;
            float priority = 0.0f;
            bool contains_selected = false;
            bool contains_hovered_keyframe = false;
            bool hovered = false;
            bool stale = true;
        };

        struct ExactMarkerInfo {
            float time = 0.0f;
            float screen_x = 0.0f;
            bool selected = false;
            bool hovered = false;
        };

        void render(const SequencerController& controller,
                    RenderingManager* rm, SceneManager* sm,
                    const RenderOptions& options);

        [[nodiscard]] const std::optional<HoverState>& hoverState() const { return hover_state_; }
        [[nodiscard]] const std::vector<ThumbInfo>& thumbs() const { return thumbs_; }
        [[nodiscard]] const std::vector<ExactMarkerInfo>& markers() const { return exact_markers_; }
        [[nodiscard]] std::uintptr_t textureIdForSlot(const int slot_idx) const;
        [[nodiscard]] std::string srcUrlForSlot(const int slot_idx) const;
        [[nodiscard]] bool slotIsCurrentGeneration(const int slot_idx) const;

        void invalidateAll();
        void destroyGraphicsResources();

    private:
        struct Slot {
            VulkanUiTexture texture;
            float time = -1.0f;
            uint32_t frame_used = 0;
            uint32_t generation = 0;
            bool valid = false;
        };

        struct RenderRequest {
            size_t index = 0;
            int visible_index = -1;
            float time = 0.0f;
            float tolerance = 0.0f;
            float priority = 0.0f;
            int preferred_slot = -1;
        };

        int findSlot(float time, float tolerance,
                     const std::array<bool, MAX_SLOTS>& claimed_slots) const;
        int allocateSlot(const std::array<bool, MAX_SLOTS>& claimed_slots);
        bool renderThumbnail(int slot_idx, float time,
                             const SequencerController& controller,
                             RenderingManager* rm, SceneManager* sm);

        std::array<Slot, MAX_SLOTS> slots_;
        uint32_t frame_counter_ = 0;

        std::vector<ThumbInfo> thumbs_;
        std::vector<ExactMarkerInfo> exact_markers_;
        std::vector<RenderRequest> render_requests_;
        std::vector<int> visible_slot_assignments_;
        uint32_t generation_ = 0;
        int burst_remaining_ = 0;
        std::optional<HoverState> hover_state_;
        std::optional<float> last_hover_focus_time_;
    };

} // namespace lfs::vis::gui
