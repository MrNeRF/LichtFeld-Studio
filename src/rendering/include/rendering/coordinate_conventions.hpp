/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "render_constants.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <iterator>
#include <optional>
#include <utility>

namespace lfs::rendering {

    struct CameraPose {
        glm::mat3 rotation{1.0f};
        glm::vec3 translation{0.0f};
    };

    // Single coordinate-conversion boundary for the application.
    //
    // Visualizer cameras use local axes:
    //   +X = right, +Y = up, +Z = back, -Z = forward.
    //
    // Dataset / import / export cameras keep the existing local axes:
    //   +X = right, +Y = down, +Z = forward.
    //
    // The rasterizer camera basis is different again:
    //   +X = right, +Y = up, +Z = forward.
    //
    // Visualizer <-> dataset flips Y/Z in both local camera space and the
    // visualizer-facing world basis. Raw scene/model data stays in dataset
    // coordinates and gets this basis change once at the visualizer boundary.
    //
    // Visualizer <-> raster flips local Z only.
    inline const glm::mat3 VISUALIZER_TO_DATA_CAMERA_AXES{1, 0, 0, 0, -1, 0, 0, 0, -1};
    inline const glm::mat3 DATA_TO_VISUALIZER_CAMERA_AXES = VISUALIZER_TO_DATA_CAMERA_AXES;
    inline const glm::mat4 DATA_TO_VISUALIZER_CAMERA_AXES_4{1, 0, 0, 0,
                                                            0, -1, 0, 0,
                                                            0, 0, -1, 0,
                                                            0, 0, 0, 1};
    inline const glm::mat4 VISUALIZER_TO_DATA_CAMERA_AXES_4 = DATA_TO_VISUALIZER_CAMERA_AXES_4;
    inline const glm::mat3 DATA_TO_VISUALIZER_WORLD_AXES = DATA_TO_VISUALIZER_CAMERA_AXES;
    inline const glm::mat3 VISUALIZER_TO_DATA_WORLD_AXES = DATA_TO_VISUALIZER_WORLD_AXES;
    inline const glm::mat4 DATA_TO_VISUALIZER_WORLD_AXES_4 = DATA_TO_VISUALIZER_CAMERA_AXES_4;
    inline const glm::mat4 VISUALIZER_TO_DATA_WORLD_AXES_4 = DATA_TO_VISUALIZER_WORLD_AXES_4;
    inline const glm::mat3 VISUALIZER_TO_RASTER_CAMERA_AXES{1, 0, 0, 0, 1, 0, 0, 0, -1};
    inline const glm::mat3 RASTER_TO_VISUALIZER_CAMERA_AXES = VISUALIZER_TO_RASTER_CAMERA_AXES;

    inline glm::vec3 cameraRight(const glm::mat3& rotation) {
        return glm::normalize(rotation[0]);
    }

    inline glm::vec3 cameraUp(const glm::mat3& rotation) {
        return glm::normalize(rotation[1]);
    }

    inline glm::vec3 cameraBackward(const glm::mat3& rotation) {
        return glm::normalize(rotation[2]);
    }

    inline glm::vec3 cameraForward(const glm::mat3& rotation) {
        return -cameraBackward(rotation);
    }

    inline glm::mat3 visualizerRotationFromDataCameraToWorld(const glm::mat3& data_camera_to_world) {
        return data_camera_to_world * DATA_TO_VISUALIZER_CAMERA_AXES;
    }

    inline glm::mat3 dataCameraToWorldFromVisualizerRotation(const glm::mat3& visualizer_rotation) {
        return visualizer_rotation * VISUALIZER_TO_DATA_CAMERA_AXES;
    }

    inline glm::mat3 rasterCameraToWorldFromVisualizerRotation(const glm::mat3& visualizer_rotation) {
        return visualizer_rotation * VISUALIZER_TO_RASTER_CAMERA_AXES;
    }

    inline glm::vec3 visualizerWorldPointFromDataWorld(const glm::vec3& data_point) {
        return DATA_TO_VISUALIZER_WORLD_AXES * data_point;
    }

    inline glm::vec3 dataWorldPointFromVisualizerWorld(const glm::vec3& visualizer_point) {
        return VISUALIZER_TO_DATA_WORLD_AXES * visualizer_point;
    }

    inline glm::mat4 dataWorldTransformToVisualizerWorld(const glm::mat4& data_world_transform) {
        return DATA_TO_VISUALIZER_WORLD_AXES_4 * data_world_transform;
    }

    inline glm::mat4 visualizerWorldTransformToDataWorld(const glm::mat4& visualizer_world_transform) {
        return VISUALIZER_TO_DATA_WORLD_AXES_4 * visualizer_world_transform;
    }

    inline glm::vec3 chooseFallbackUp(const glm::vec3& forward);

    inline glm::mat3 mat3FromRowMajor3x3(const float* const row_major) {
        glm::mat3 matrix(1.0f);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                matrix[col][row] = row_major[row * 3 + col];
            }
        }
        return matrix;
    }

    inline glm::mat3 orthonormalizedRotation(const glm::mat4& transform) {
        glm::vec3 x = glm::vec3(transform[0]);
        glm::vec3 y = glm::vec3(transform[1]);
        glm::vec3 z = glm::vec3(transform[2]);

        const auto normalized_or = [](const glm::vec3& value, const glm::vec3& fallback) {
            const float length = glm::length(value);
            return length > 1e-6f ? value / length : fallback;
        };

        x = normalized_or(x, glm::vec3(1.0f, 0.0f, 0.0f));
        y = y - glm::dot(y, x) * x;
        y = normalized_or(y, chooseFallbackUp(x));
        z = z - glm::dot(z, x) * x - glm::dot(z, y) * y;
        z = normalized_or(z, glm::cross(x, y));

        if (glm::dot(glm::cross(x, y), z) < 0.0f) {
            z = -z;
        }

        return glm::mat3(x, y, z);
    }

    inline glm::vec3 dataCameraPositionFromWorldToCamera(const glm::mat3& data_world_to_camera,
                                                         const glm::vec3& data_world_to_camera_translation) {
        const glm::mat3 data_camera_to_world = glm::transpose(data_world_to_camera);
        return -data_camera_to_world * data_world_to_camera_translation;
    }

    inline CameraPose visualizerCameraPoseFromDataCameraToWorld(
        const glm::mat3& data_camera_to_world,
        const glm::vec3& data_camera_position,
        const glm::mat4& visualizer_scene_transform = glm::mat4(1.0f)) {
        return {
            .rotation = orthonormalizedRotation(visualizer_scene_transform) *
                        visualizerRotationFromDataCameraToWorld(data_camera_to_world),
            .translation = glm::vec3(visualizer_scene_transform * glm::vec4(data_camera_position, 1.0f)),
        };
    }

    inline CameraPose visualizerCameraPoseFromDataWorldToCamera(
        const glm::mat3& data_world_to_camera,
        const glm::vec3& data_world_to_camera_translation,
        const glm::mat4& visualizer_scene_transform = glm::mat4(1.0f)) {
        return visualizerCameraPoseFromDataCameraToWorld(
            glm::transpose(data_world_to_camera),
            dataCameraPositionFromWorldToCamera(data_world_to_camera, data_world_to_camera_translation),
            visualizer_scene_transform);
    }

    inline glm::mat4 makeViewMatrix(const glm::mat3& rotation, const glm::vec3& translation) {
        const glm::mat3 rotation_inv = glm::transpose(rotation);
        const glm::vec3 translation_inv = -rotation_inv * translation;

        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view[i][j] = rotation_inv[i][j];
            }
        }
        view[3][0] = translation_inv.x;
        view[3][1] = translation_inv.y;
        view[3][2] = translation_inv.z;
        return view;
    }

    inline glm::vec3 chooseFallbackUp(const glm::vec3& forward) {
        const glm::vec3 candidates[] = {
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(1.0f, 0.0f, 0.0f)};

        glm::vec3 best = candidates[0];
        float best_alignment = std::abs(glm::dot(glm::normalize(forward), best));
        for (size_t i = 1; i < std::size(candidates); ++i) {
            const float alignment = std::abs(glm::dot(glm::normalize(forward), candidates[i]));
            if (alignment < best_alignment) {
                best = candidates[i];
                best_alignment = alignment;
            }
        }
        return best;
    }

    inline bool isFiniteVec3(const glm::vec3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    }

    inline std::optional<glm::mat3> tryMakeVisualizerLookAtRotation(const glm::vec3& eye,
                                                                    const glm::vec3& target,
                                                                    const glm::vec3& requested_up) {
        if (!isFiniteVec3(eye) || !isFiniteVec3(target) || !isFiniteVec3(requested_up)) {
            return std::nullopt;
        }

        const glm::vec3 view = target - eye;
        const float view_length = glm::length(view);
        if (view_length <= 1e-6f) {
            return std::nullopt;
        }

        const glm::vec3 forward = view / view_length;

        glm::vec3 up = requested_up;
        const float up_length = glm::length(up);
        if (up_length <= 1e-6f) {
            up = chooseFallbackUp(forward);
        } else {
            up /= up_length;
        }

        glm::vec3 right = glm::cross(forward, up);
        float right_length = glm::length(right);
        if (right_length <= 1e-6f) {
            up = chooseFallbackUp(forward);
            right = glm::cross(forward, up);
            right_length = glm::length(right);
            if (right_length <= 1e-6f) {
                return std::nullopt;
            }
        }
        right /= right_length;

        const glm::vec3 backward = -forward;
        glm::vec3 camera_up = glm::cross(backward, right);
        const float camera_up_length = glm::length(camera_up);
        if (camera_up_length <= 1e-6f) {
            return std::nullopt;
        }
        camera_up /= camera_up_length;

        return glm::mat3(right, camera_up, backward);
    }

    inline glm::mat3 makeVisualizerLookAtRotation(const glm::vec3& eye,
                                                  const glm::vec3& target,
                                                  const glm::vec3& requested_up = glm::vec3(0.0f, 1.0f, 0.0f)) {
        if (const auto rotation = tryMakeVisualizerLookAtRotation(eye, target, requested_up)) {
            return *rotation;
        }
        return glm::mat3(1.0f);
    }

    inline std::pair<float, float> computePixelFocalLengths(const glm::ivec2& viewport_size,
                                                            const float focal_length_mm) {
        const float width = static_cast<float>(viewport_size.x);
        const float height = static_cast<float>(viewport_size.y);
        const float fov_y = focalLengthToVFovRad(focal_length_mm);
        const float aspect = width / height;
        const float fov_x = 2.0f * std::atan(std::tan(fov_y * 0.5f) * aspect);
        const float fx = width / (2.0f * std::tan(fov_x * 0.5f));
        const float fy = height / (2.0f * std::tan(fov_y * 0.5f));
        return {fx, fy};
    }

    inline glm::vec3 computePickRayDirection(const glm::mat3& rotation,
                                             const glm::ivec2& viewport_size,
                                             const float screen_x,
                                             const float screen_y,
                                             const float focal_length_mm) {
        const auto [fx, fy] = computePixelFocalLengths(viewport_size, focal_length_mm);
        const float width = static_cast<float>(viewport_size.x);
        const float height = static_cast<float>(viewport_size.y);
        const float cx = width * 0.5f;
        const float cy = height * 0.5f;

        const glm::vec3 camera_dir = glm::normalize(glm::vec3(
            (screen_x - cx) / fx,
            (cy - screen_y) / fy,
            -1.0f));
        return glm::normalize(rotation * camera_dir);
    }

    inline glm::vec3 unprojectScreenPoint(const glm::mat3& rotation,
                                          const glm::vec3& translation,
                                          const glm::ivec2& viewport_size,
                                          const float screen_x,
                                          const float screen_y,
                                          const float depth,
                                          const float focal_length_mm) {
        const auto [fx, fy] = computePixelFocalLengths(viewport_size, focal_length_mm);
        const float width = static_cast<float>(viewport_size.x);
        const float height = static_cast<float>(viewport_size.y);
        const float cx = width * 0.5f;
        const float cy = height * 0.5f;

        const glm::vec3 view_pos(
            (screen_x - cx) * depth / fx,
            (cy - screen_y) * depth / fy,
            -depth);
        return rotation * view_pos + translation;
    }

    inline std::optional<glm::vec2> projectWorldPoint(const glm::mat3& rotation,
                                                      const glm::vec3& translation,
                                                      const glm::ivec2& viewport_size,
                                                      const glm::vec3& world_point,
                                                      const float focal_length_mm,
                                                      const bool orthographic = false,
                                                      const float ortho_scale = DEFAULT_ORTHO_SCALE) {
        const glm::vec3 view = glm::transpose(rotation) * (world_point - translation);
        if (!isFiniteVec3(view) || view.z >= -1e-6f) {
            return std::nullopt;
        }

        const float width = static_cast<float>(viewport_size.x);
        const float height = static_cast<float>(viewport_size.y);
        const float cx = width * 0.5f;
        const float cy = height * 0.5f;

        if (orthographic) {
            if (!std::isfinite(ortho_scale) || ortho_scale <= 0.0f) {
                return std::nullopt;
            }

            return glm::vec2(
                cx + view.x * ortho_scale,
                cy - view.y * ortho_scale);
        }

        const auto [fx, fy] = computePixelFocalLengths(viewport_size, focal_length_mm);
        const float depth = -view.z;

        return glm::vec2(
            cx + view.x * fx / depth,
            cy - view.y * fy / depth);
    }

} // namespace lfs::rendering
