/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "nvimgcodec.h"
#include "out_of_bound_roi_fill.h"

#include <utility>

namespace nvimgcodec {

// nvimgcodecOrientation_t.rotated is the counter-clockwise rotation in degrees applied to the
// raw codestream pixels to produce the displayed image. Together with flip_x (mirror across
// the vertical axis) and flip_y (mirror across the horizontal axis) it describes the EXIF
// orientation correction. flip_x and flip_y are applied AFTER the rotation, in display space.

// Identity orientation (no rotation, no flips). Useful when callers need to express "as if
// EXIF orientation were not applied" without re-constructing the struct at each call site.
inline const nvimgcodecOrientation_t kIdentityOrientation{
    NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr,
    0, 0, 0};

// Returns (display_width, display_height) given raw codestream dims and an orientation.
inline std::pair<uint32_t, uint32_t> oriented_dims(uint32_t raw_w, uint32_t raw_h,
                                                   const nvimgcodecOrientation_t& orientation) {
    const bool swap = (orientation.rotated == 90 || orientation.rotated == 270);
    return swap ? std::pair<uint32_t, uint32_t>(raw_h, raw_w)
                : std::pair<uint32_t, uint32_t>(raw_w, raw_h);
}

// Validates a region against display dims if EXIF orientation will be applied, otherwise
// against the raw codestream dims. The check matches is_region_out_of_bounds in semantics.
inline bool is_region_out_of_bounds_effective(const nvimgcodecRegion_t& region,
                                              const nvimgcodecOrientation_t& orientation,
                                              uint32_t raw_w, uint32_t raw_h,
                                              bool apply_exif_orientation) {
    if (apply_exif_orientation) {
        const auto [disp_w, disp_h] = oriented_dims(raw_w, raw_h, orientation);
        return is_region_out_of_bounds(region, disp_w, disp_h);
    }
    return is_region_out_of_bounds(region, raw_w, raw_h);
}

}  // namespace nvimgcodec
