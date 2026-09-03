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
        glm::ivec2 subregion_origin{0, 0};
        glm::ivec2 subregion_full_size{0, 0};
        float focal_length_mm = DEFAULT_FOCAL_LENGTH_MM;
        std::optional<CameraIntrinsics> intrinsics_override;
        // Unjittered intrinsics of the displayed camera, for depth-window containment only.
        // Captured before applySceneViewJitter and replaced wholesale wherever
        // intrinsics_override is replaced. Never carries jitter. Empty means "image-centred
        // with the draw focals", which is correct for every synthetic-intrinsics viewport.
        std::optional<CameraIntrinsics> containment_intrinsics;
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;
        float ortho_scale = DEFAULT_ORTHO_SCALE;
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};

        [[nodiscard]] glm::mat4 getViewMatrix() const {
            return makeViewMatrix(rotation, translation);
        }

        // A clipped render target keeps the camera calibrated to the full
        // viewport. The renderer uses the target size for its tile grid and
        // this size for projection/camera intrinsics.
        [[nodiscard]] glm::ivec2 cameraSize() const {
            return subregion_full_size.x > 0 && subregion_full_size.y > 0
                       ? subregion_full_size
                       : size;
        }

        [[nodiscard]] CameraIntrinsics getCameraIntrinsics() const {
            const auto camera_size = cameraSize();
            if (!orthographic && intrinsics_override) {
                return *intrinsics_override;
            }
            if (orthographic) {
                const float scale = std::isfinite(ortho_scale) && ortho_scale > 1.0e-5f
                                        ? ortho_scale
                                        : DEFAULT_ORTHO_SCALE;
                return {.focal_x = scale,
                        .focal_y = scale,
                        .center_x = static_cast<float>(camera_size.x) * 0.5f,
                        .center_y = static_cast<float>(camera_size.y) * 0.5f};
            }
            const auto [fx, fy] = computePixelFocalLengths(camera_size, focal_length_mm);
            return {.focal_x = fx,
                    .focal_y = fy,
                    .center_x = static_cast<float>(camera_size.x) * 0.5f,
                    .center_y = static_cast<float>(camera_size.y) * 0.5f};
        }

        [[nodiscard]] glm::mat4 getProjectionMatrix(
            const float near_plane_override = DEFAULT_NEAR_PLANE,
            const float far_plane_override = DEFAULT_FAR_PLANE) const {
            return createProjectionMatrix(
                cameraSize(),
                focalLengthToVFov(focal_length_mm),
                orthographic,
                ortho_scale,
                near_plane_override,
                far_plane_override);
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
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;
        bool orthographic = false;

        [[nodiscard]] bool valid() const {
            return color.valid();
        }
    };

} // namespace lfs::rendering
