/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/geometry_types.hpp"

#include "core/tensor.hpp"

#include <functional>
#include <vector>

namespace lfs::preprocessing {

    // A lens as the rasterizer models it, at width x height pixels.
    struct LensCamera {
        training::kernels::DepthCameraProjection projection;
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        int width = 0;
        int height = 0;
    };

    // A square pinhole view inside the lens; `rotation` (row-major) maps camera rays into the face frame.
    struct LensFace {
        float rotation[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        float focal = 0.0f;
        int size = 0;
    };

    // [height, width, 3] unit camera ray through each pixel center; NaN where the lens holds no ray.
    void lens_rays(const LensCamera& camera, core::Tensor& rays);

    // A monocular estimator's answer for one square face, in the face frame.
    struct LensFaceOutputs {
        std::vector<float> points;  // [size * size * 3]
        std::vector<float> normals; // [size * size * 3]
        std::vector<float> mask;    // [size * size], >= 0.5 where the estimate is valid
    };

    using LensFaceInference = std::function<LensFaceOutputs(const std::vector<float>& face_rgb, int size)>;

    struct LensPriors {
        int width = 0;
        int height = 0;
        std::vector<float> distance; // [height * width] distance along the ray, 0 where unknown
        std::vector<float> normal;   // [height * width * 3] camera frame
        int faces = 0;
    };

    // Square pinhole faces covering every ray of the lens: one for a moderate field of view, a front face and
    // a ring for a wide fisheye, a cube for a panorama.
    [[nodiscard]] std::vector<LensFace> plan_lens_faces(
        const LensCamera& camera, int face_size);

    // Runs `infer` on faces resampled from the [source.height, source.width, 3] image and gathers the answers
    // onto `grid` (the same lens at the output resolution). Each face is rescaled to agree with the faces
    // gathered before it where they overlap, since a monocular estimate fixes its scale per image.
    [[nodiscard]] LensPriors estimate_lens_priors(const std::vector<float>& source_rgb,
                                                  const LensCamera& source,
                                                  const LensCamera& grid,
                                                  int face_size,
                                                  const LensFaceInference& infer);

} // namespace lfs::preprocessing
