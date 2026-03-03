/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/gl_resources.hpp"
#include <array>
#include <cstdint>

namespace lfs::vis {
    class SequencerController;
    class RenderingManager;
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis::gui {

    class FilmStripRenderer {
    public:
        static constexpr int THUMB_WIDTH = 128;
        static constexpr int THUMB_HEIGHT = 72;
        static constexpr int MAX_SLOTS = 32;
        static constexpr int MAX_RENDERS_PER_FRAME = 2;
        static constexpr float STRIP_HEIGHT = 56.0f;
        static constexpr float THUMB_PADDING = 4.0f;

        void render(const SequencerController& controller,
                    RenderingManager* rm, SceneManager* sm,
                    float panel_x, float panel_width,
                    float timeline_x, float timeline_width,
                    float strip_y,
                    float zoom_level, float pan_offset,
                    float display_end_time);

        void invalidateAll();
        void destroyGLResources();

    private:
        struct Slot {
            rendering::Texture texture;
            float time = -1.0f;
            uint32_t frame_used = 0;
            bool valid = false;
        };

        void initGL();
        int findSlot(float time, float tolerance) const;
        int allocateSlot(uint32_t current_frame);
        bool renderThumbnail(int slot_idx, float time,
                             const SequencerController& controller,
                             RenderingManager* rm, SceneManager* sm);

        std::array<Slot, MAX_SLOTS> slots_;
        rendering::FBO fbo_;
        rendering::RBO depth_rbo_;
        bool gl_initialized_ = false;
        bool gl_init_failed_ = false;
        uint32_t frame_counter_ = 0;
    };

} // namespace lfs::vis::gui
