/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace lfs::gpu_ops {

    struct RoiParams {
        HW image;
        Intrinsics intrinsics;
        // Column-major, matching the scene crop-box transform.
        std::array<float, 16> world_to_cropbox;
        std::array<float, 3> minimum, maximum;
        float outside_weight;
        bool inverse;
        // A crop polygon prism instead of the box: the outline vertices (x, y pairs) in the crop
        // frame over |z| <= maximum[2], which minimum and maximum bound. Empty for a box.
        // polygon_device holds the same [count, 2] floats on the device for backends that read
        // them there.
        std::span<const float> polygon{};
        Tensor polygon_device{};
    };

    struct TrainingImageOps {
        void (*heatmap)(In loss, Out latest, Out ema, int slot, float ema_alpha);
        // camera_rays: optional [H,W,3] camera-space rays (non-finite: no ray) for lens cameras;
        // pinholes use the intrinsics. With surface_depth and surface_alpha ([H,W], a camera
        // inside the box), a pixel is in the ROI when the surface it shows lies in the box.
        void (*roi)(In view, In camera_position, In camera_rays, In surface_depth, In surface_alpha,
                    Out weights, const RoiParams&);
        void (*resize_background)(In source, Out destination);
        void (*random_background)(Out destination, uint64_t seed);
        // `photometric_weight` is empty or a Float32 [H,W] map with zero for
        // ignored pixels. Canny excludes neighborhoods touching zero weights.
        void (*canny)(In image, In photometric_weight, Out edges);
        void (*normalize_scalar)(Out values, In scalar, float skip_below);
        // Optional upload/layout specialization; callers retain their default path.
        Tensor (*upload_image_chw)(In cpu_hwc) = nullptr;
        Tensor (*quantize_to_8bit_grid)(In image) = nullptr;
        Tensor (*quantize_to_grid)(In image, float levels) = nullptr;
        Tensor (*flip_error_map)(In reference, In test, float pixels_per_degree) = nullptr;
        Tensor (*flip_error_image)(In error_map) = nullptr;
        // Whether roi takes RoiParams::polygon. Callers must not pass a polygon otherwise.
        bool roi_polygon = false;
    };

} // namespace lfs::gpu_ops
