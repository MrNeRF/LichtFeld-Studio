/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "coordinate_conventions.hpp"
#include "render_constants.hpp"
#include <glm/glm.hpp>
#include <optional>

namespace lfs::rendering {

    // Renderer-facing frame contract for the refactor.
    // Rotation/translation are visualizer-space camera-to-world transforms.

    enum class TextureOrigin {
        BottomLeft,
        TopLeft,
    };

    [[nodiscard]] inline bool presentationFlipYFromTextureOrigin(const TextureOrigin origin) {
        return origin == TextureOrigin::TopLeft;
    }

    struct FrameView {
        glm::mat3 rotation{1.0f};
        glm::vec3 translation{0.0f};
        glm::ivec2 size{0, 0};
        float focal_length_mm = DEFAULT_FOCAL_LENGTH_MM;
        std::optional<CameraIntrinsics> intrinsics_override;
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;
        float ortho_scale = DEFAULT_ORTHO_SCALE;
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};

        [[nodiscard]] glm::mat4 getViewMatrix() const {
            return makeViewMatrix(rotation, translation);
        }
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
        // Presentation orientation for the screen quad. Geometry/camera conventions live elsewhere.
        bool flip_y = false;
        bool depth_is_ndc = false;
        bool color_has_alpha = false;
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;

        [[nodiscard]] bool valid() const {
            return color.valid();
        }
    };

} // namespace lfs::rendering
