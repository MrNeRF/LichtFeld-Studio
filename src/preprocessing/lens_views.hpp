/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/lens_priors.hpp"

#include <cuda_runtime.h>

namespace lfs::preprocessing {

    // [size, size, 3] face pixels resampled bilinearly from the [height, width, 3] lens image; mid-grey where
    // the lens holds no ray.
    void sample_lens_face(const float* lens_rgb, const LensCamera& camera, const LensFace& face,
                          float* face_rgb, cudaStream_t stream);

    // For every lens pixel: the face's distance along the ray, its normal rotated into the camera frame and a
    // weight fading to 0 at the face border; distance is NaN where the face does not see the pixel or its
    // mask is off. face_points/face_normals are [size, size, 3] in the face frame, face_mask [size, size].
    void project_face_to_lens(const LensCamera& camera, const LensFace& face,
                              const float* face_points, const float* face_normals, const float* face_mask,
                              float* distance, float* normal, float* weight, cudaStream_t stream);

    // Weighted blend of `faces` planes ([faces, pixels] distance/weight, [faces, pixels, 3] normal) with one
    // scale per face; pixels no face reaches get distance 0 and a zero normal.
    void blend_lens_faces(const float* distance, const float* normal, const float* weight,
                          const float* scales, int faces, int pixels,
                          float* out_distance, float* out_normal, cudaStream_t stream);

} // namespace lfs::preprocessing
