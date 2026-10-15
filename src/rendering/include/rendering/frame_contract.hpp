/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "coordinate_conventions.hpp"
#include "render_constants.hpp"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <limits>
#include <optional>
#include <vector>

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
        // Export pixels per source viewport pixel. Keep screen-space splat
        // filtering and extent limits in the source viewport's pixel units.
        float rasterization_scale = 1.0f;
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

    // An orthographic image does not change when the eye slides along the view axis, but the rasterizer culls
    // everything behind the eye. Returns how far to move the eye back so the node-local bounds, placed by every
    // model transform (identity when there are none), lie in front of it.
    [[nodiscard]] inline float orthographicEyeSetback(const FrameView& frame_view,
                                                      const glm::vec3& local_min,
                                                      const glm::vec3& local_max,
                                                      const std::vector<glm::mat4>* model_transforms) {
        const glm::vec3 forward = dataCameraToWorldFromVisualizerRotation(frame_view.rotation)[2];
        float nearest = std::numeric_limits<float>::max();
        float extent = 0.0f;
        const auto visit = [&](const glm::mat4& transform) {
            glm::vec3 world_min(std::numeric_limits<float>::max());
            glm::vec3 world_max(std::numeric_limits<float>::lowest());
            for (int corner = 0; corner < 8; ++corner) {
                const glm::vec3 local{(corner & 1) ? local_max.x : local_min.x,
                                      (corner & 2) ? local_max.y : local_min.y,
                                      (corner & 4) ? local_max.z : local_min.z};
                const glm::vec3 world = glm::vec3(transform * glm::vec4(local, 1.0f));
                nearest = std::min(nearest, glm::dot(world - frame_view.translation, forward));
                world_min = glm::min(world_min, world);
                world_max = glm::max(world_max, world);
            }
            extent = std::max(extent, glm::length(world_max - world_min));
        };
        if (model_transforms && !model_transforms->empty()) {
            for (const auto& transform : *model_transforms)
                visit(transform);
        } else {
            visit(glm::mat4(1.0f));
        }
        const float margin = DEFAULT_NEAR_PLANE + 0.01f * extent;
        return std::isfinite(nearest) && std::isfinite(extent) ? std::max(0.0f, margin - nearest) : 0.0f;
    }

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
