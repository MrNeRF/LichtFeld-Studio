/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace lfs::training::kernels {

    // An extruded crop polygon in its local frame: the outline in XY (3 to 64 vertices, even-odd rule) over
    // |z| <= half_depth. The vertices are read on the host when a kernel launches.
    struct RoiPolygon {
        const glm::vec2* vertices = nullptr;
        int count = 0;
        float half_depth = 0.0f;
    };

    void launch_roi_weight_map(
        const float* world_to_camera, // [1,4,4] or [4,4] CUDA, row-major
        const float* camera_position, // [3] CUDA
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        const glm::mat4& world_to_cropbox,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        bool inverse,
        float outside_weight,
        const float* camera_rays, // [H,W,3] CUDA camera-space rays (NaN: no ray), or null for pinhole
        // With both set (a camera inside the box), a pixel is in the ROI when the surface it shows
        // lies in the box, not when its ray crosses it: every ray of such a camera starts inside.
        const float* surface_depth, // [H,W] CUDA accumulated depth in the rays' units, or null
        const float* surface_alpha, // [H,W] CUDA accumulated alpha, or null
        float* output,              // [H,W] CUDA
        cudaStream_t stream = nullptr);

    // launch_roi_weight_map for a crop polygon: a pixel is in the ROI when its ray passes through the prism
    // (or, for a camera inside the prism, when the surface it shows lies in it). The test is exact for
    // concave outlines: the ray is clipped to the |z| slab and the segment left must touch the outline.
    void launch_roi_weight_map(
        const float* world_to_camera,
        const float* camera_position,
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        const glm::mat4& world_to_polygon,
        const RoiPolygon& polygon,
        bool inverse,
        float outside_weight,
        const float* camera_rays,
        const float* surface_depth,
        const float* surface_alpha,
        float* output,
        cudaStream_t stream = nullptr);

    // Largest stretch of the model-to-box transform: box units per model unit.
    [[nodiscard]] float box_units_per_model_unit(const glm::mat4& model_to_cropbox);

    // Marks splats whose centre ray from this camera misses the box (meets it when inverse): they
    // are neither inside the box nor in front of or behind it in this view. With scaling_raw
    // ([N,3] log scales) the box grows by each splat's 3-sigma footprint. Deleted rows stay false.
    void launch_roi_view_reject_mask(
        const float* means,       // [N,3] CUDA, model frame
        const float* scaling_raw, // [N,3] CUDA or null
        size_t n,
        const glm::mat4& model_to_cropbox,
        const glm::mat4& world_to_cropbox,
        const float* camera_position, // [3] CUDA, world
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        bool inverse,
        const bool* deleted, // [N] CUDA or null
        bool* rejected,      // [N] CUDA
        cudaStream_t stream = nullptr);

    // launch_roi_view_reject_mask for a crop polygon. The footprint margin grows the outline by a disc and the
    // slab by the same distance; inverse polygons shrink only the slab, as eroding a concave outline per splat
    // is far costlier than the test itself.
    void launch_roi_view_reject_mask(
        const float* means,
        const float* scaling_raw,
        size_t n,
        const glm::mat4& model_to_polygon,
        const glm::mat4& world_to_polygon,
        const float* camera_position,
        const RoiPolygon& polygon,
        bool inverse,
        const bool* deleted,
        bool* rejected,
        cudaStream_t stream = nullptr);

    // Marks splats that no listed camera sees inside, in front of or behind the box (outside it
    // when inverse). in_roi, when given, marks centres inside the ROI (outside the box when
    // inverse). Deleted rows stay false in both.
    void launch_roi_relevance_reject_mask(
        const float* means,       // [N,3] CUDA, model frame
        const float* scaling_raw, // [N,3] CUDA or null
        size_t n,
        const glm::mat4& model_to_cropbox,
        const float* cameras_in_box, // [M,3] CUDA camera centres in box space
        int num_cameras,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        bool inverse,
        const bool* deleted, // [N] CUDA or null
        bool* rejected,      // [N] CUDA
        bool* in_roi,        // [N] CUDA or null
        cudaStream_t stream = nullptr);

    // launch_roi_relevance_reject_mask for a crop polygon, with the margin of launch_roi_view_reject_mask.
    void launch_roi_relevance_reject_mask(
        const float* means,
        const float* scaling_raw,
        size_t n,
        const glm::mat4& model_to_polygon,
        const float* cameras_in_polygon, // [M,3] CUDA camera centres in polygon space
        int num_cameras,
        const RoiPolygon& polygon,
        bool inverse,
        const bool* deleted,
        bool* rejected,
        bool* in_roi,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
