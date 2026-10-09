/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/lens_priors.hpp"

namespace lfs::preprocessing {

    // [size, size, 3] face pixels resampled bilinearly from the [height, width, 3] lens image; mid-grey where
    // the lens holds no ray.
    void sample_lens_face(const core::Tensor& lens_rgb, const LensCamera& camera, const LensFace& face,
                          core::Tensor& face_rgb);

    // For every lens pixel: the face's distance along the ray, its normal rotated into the camera frame and a
    // weight fading to 0 at the face border; distance is NaN where the face does not see the pixel or its
    // mask is off. face_points/face_normals are [size, size, 3] in the face frame, face_mask [size, size].
    void project_face_to_lens(const LensCamera& camera, const LensFace& face,
                              const core::Tensor& face_points, const core::Tensor& face_normals, const core::Tensor& face_mask,
                              core::Tensor& distance, core::Tensor& normal, core::Tensor& weight);

    // Weighted blend of `faces` planes ([faces, pixels] distance/weight, [faces, pixels, 3] normal) with one
    // scale per face; pixels no face reaches get distance 0 and a zero normal.
    void blend_lens_faces(const core::Tensor& distance, const core::Tensor& normal, const core::Tensor& weight,
                          const core::Tensor& scales, int faces, int pixels,
                          core::Tensor& out_distance, core::Tensor& out_normal);

} // namespace lfs::preprocessing
