/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "rendering/scene_output.hpp"
#include <cstdint>
#include <glm/vec2.hpp>

namespace lfs::vis::vr {

    // One rendered eye. The renderer keeps the image until it renders the next
    // VR frame or releases its VR targets.
    struct EyeFrame {
        SceneImageHandle image;
        SceneImageViewHandle image_view;
        SceneImageLayout image_layout;
        std::uint64_t image_generation = 0;
        // Signaled at completion_value once the image is written.
        SceneTimelineHandle completion_timeline;
        std::uint64_t completion_value = 0;
        glm::ivec2 size{0, 0};       // eye extent
        glm::ivec2 alloc_size{0, 0}; // image extent (0 = size)
        bool flip_y = false;
        bool rendered = false;
    };

} // namespace lfs::vis::vr
