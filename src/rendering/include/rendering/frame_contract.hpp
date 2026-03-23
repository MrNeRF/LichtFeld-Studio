/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "render_constants.hpp"
#include <glm/glm.hpp>

namespace lfs::rendering {

    // Renderer-facing frame contract for the refactor.

    struct FrameView {
        glm::mat3 rotation{1.0f};
        glm::vec3 translation{0.0f};
        glm::ivec2 size{0, 0};
        float focal_length_mm = DEFAULT_FOCAL_LENGTH_MM;
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;
        float ortho_scale = DEFAULT_ORTHO_SCALE;
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
    };

    struct TextureHandle {
        unsigned int id = 0;
        glm::ivec2 size{0, 0};
        glm::vec2 texcoord_scale{1.0f, 1.0f};

        [[nodiscard]] bool valid() const {
            return id != 0 && size.x > 0 && size.y > 0;
        }
    };

    struct GpuFrame {
        TextureHandle color;
        TextureHandle depth;
        bool depth_is_ndc = false;
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;

        [[nodiscard]] bool valid() const {
            return color.valid();
        }
    };

} // namespace lfs::rendering
