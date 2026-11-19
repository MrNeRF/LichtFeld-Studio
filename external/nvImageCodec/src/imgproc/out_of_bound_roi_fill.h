
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <string>

namespace nvimgcodec {

void fill_out_of_bounds_region(
    const nvimgcodecImageInfo_t& target_image_info,
    uint32_t original_image_width, uint32_t original_image_height, // dimensions of image before applying roi
    const nvimgcodecRegion_t& region
);

inline bool is_region_out_of_bounds(const nvimgcodecRegion_t& region, uint32_t original_image_width, uint32_t original_image_height) {
    if (region.ndim == 0)
        return false;
    assert(region.ndim == 2);
    return region.start[0] < 0 ||
            region.start[1] < 0 ||
            region.end[0] < 0 || // to prevent overflow when casting to uint
            static_cast<uint32_t>(region.end[0]) > original_image_height ||
            region.end[1] < 0 || // to prevent overflow when casting to uint
            static_cast<uint32_t>(region.end[1]) > original_image_width;
}

// If region out of bounds fill is supported will return an empty string. Otherwise will return an error message.
// Should only be called if region is out of bounds (which can be checked with is_region_out_of_bounds())
std::string verify_region_fill_support(const nvimgcodecRegion_t& region, const nvimgcodecImageInfo_t& target_image_info);

}  // namespace nvimgcodec