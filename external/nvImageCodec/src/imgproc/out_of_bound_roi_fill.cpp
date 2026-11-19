
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

#include "nvimgcodec.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace nvimgcodec {

// Helper functions that return as size_t, so there is no overflow during computations later on
inline size_t get_num_channels_in_plane(const nvimgcodecImageInfo_t& image_info, uint32_t plane_idx) {
    return static_cast<size_t>(image_info.plane_info[plane_idx].num_channels);
}

inline size_t get_num_pixels_in_row(const nvimgcodecImageInfo_t& image_info, uint32_t plane_idx) {
    return image_info.plane_info[plane_idx].width * get_num_channels_in_plane(image_info, plane_idx);
}

inline size_t get_num_bytes_per_pixel(const nvimgcodecImageInfo_t& image_info, uint32_t plane_idx) {
    return static_cast<unsigned int>(image_info.plane_info[plane_idx].sample_type) >> (8 + 3);
}

inline size_t get_plane_byte_size(const nvimgcodecImageInfo_t& image_info, uint32_t plane_idx) {
    return image_info.plane_info[plane_idx].row_stride * image_info.plane_info[plane_idx].height;
}

inline uint32_t get_fill_bits(const nvimgcodecRegion_t& region, uint32_t comp_idx) {
    if (comp_idx >= NVIMGCODEC_MAX_ROI_FILL_CHANNELS) {
        comp_idx = NVIMGCODEC_MAX_ROI_FILL_CHANNELS - 1;
    } 

    const auto& fill_sample = region.out_of_bounds_samples[comp_idx];
    switch (fill_sample.type) {
        case 0: // no type set
            return 0;

        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
            return fill_sample.value.as_uint;

        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
        {
            uint32_t bits;
            std::memcpy(&bits, &fill_sample.value.as_int, sizeof(bits));
            return bits;
        }

        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
        {
            uint32_t bits;
            std::memcpy(&bits, &fill_sample.value.as_float, sizeof(bits));
            return bits;
        }

        default:
            // verify_region_fill_support() rejects unsupported sample data types
            // before we ever reach this path, so the default case is unreachable
            // in practice. Keep the assert for debug builds and return a defined
            // sentinel (0) so release builds do not return an indeterminate value.
            assert(false);
            return 0;
    }
}

void fill_out_of_bounds_region_device(
    const nvimgcodecImageInfo_t& image_info,
    int original_image_width, int original_image_height,
    const nvimgcodecRegion_t& region
) {
    assert(image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE);
    assert(region.ndim == 2);
    for (uint32_t i = 0; i < image_info.num_planes; ++i) {
        auto bytes_per_pixel = get_num_bytes_per_pixel(image_info, i);
        assert(bytes_per_pixel == 1 || bytes_per_pixel == 2 || bytes_per_pixel == 4);
    }

    const int roi_y_begin = region.start[0];
    const int roi_x_begin = region.start[1];
    const int roi_y_end = region.end[0];
    const int roi_x_end = region.end[1];

    const int in_bounds_roi_size = std::min(roi_y_end, original_image_height) - std::max(roi_y_begin, 0);

    auto plane_start = reinterpret_cast<uint8_t*>(image_info.buffer);

    uint32_t component_idx = 0;
    for (uint32_t i = 0; i < image_info.num_planes; ++i) {
        for (uint32_t c = 1; c < image_info.plane_info[i].num_channels; ++c) {
            assert(get_fill_bits(region, c + component_idx) == get_fill_bits(region, component_idx));
        }

        // width and height are in pixels, stride is in bytes
        auto memset2D = [&](uint8_t* ptr, size_t width, size_t height) {
            auto dev_ptr = reinterpret_cast<CUdeviceptr>(ptr);
            auto stride = image_info.plane_info[i].row_stride;

            switch (get_num_bytes_per_pixel(image_info, i)) {
            case 1:
                cuMemsetD2D8Async(dev_ptr, stride, get_fill_bits(region, component_idx), width, height, image_info.cuda_stream);
                break;
            case 2:
                cuMemsetD2D16Async(dev_ptr, stride, get_fill_bits(region, component_idx), width, height, image_info.cuda_stream);
                break;
            case 4:
                cuMemsetD2D32Async(dev_ptr, stride, get_fill_bits(region, component_idx), width, height, image_info.cuda_stream);
                break;
            default:
                assert(false);
            }
        };

        // bottom out of found ROI
        if (roi_y_end > original_image_height) {
            const int bottom_oob_roi_height = roi_y_end - original_image_height;

            auto bottom_roi_start_ptr = plane_start +
                get_plane_byte_size(image_info, i) - image_info.plane_info[i].row_stride * bottom_oob_roi_height;

            memset2D(bottom_roi_start_ptr, get_num_pixels_in_row(image_info, i), bottom_oob_roi_height);
        }

        auto first_real_image_row = plane_start;
        // top out of found ROI
        if (roi_y_begin < 0) {
            memset2D(plane_start, get_num_pixels_in_row(image_info, i), (-roi_y_begin));

            first_real_image_row += image_info.plane_info[i].row_stride * (-roi_y_begin);
        }

        // left out of found ROI
        if (roi_x_begin < 0) {
            size_t num_pixels_to_set_in_row = (-roi_x_begin) * get_num_channels_in_plane(image_info, i);
            memset2D(first_real_image_row, num_pixels_to_set_in_row, in_bounds_roi_size);
        }

        // right out of found ROI
        if (roi_x_end > original_image_width) {
            const int right_oob_roi_width = roi_x_end - original_image_width;
            size_t num_pixels_to_set_in_row = right_oob_roi_width * get_num_channels_in_plane(image_info, i);

            auto right_oob_roi_start = first_real_image_row +
                (image_info.plane_info[i].width - right_oob_roi_width) *
                get_num_channels_in_plane(image_info, i) *
                get_num_bytes_per_pixel(image_info, i);

            memset2D(right_oob_roi_start, num_pixels_to_set_in_row, in_bounds_roi_size);
        }

        plane_start += get_plane_byte_size(image_info, i);
        component_idx += image_info.plane_info[i].num_channels;
    }
}

void fill_out_of_bounds_region(
    const nvimgcodecImageInfo_t& target_image_info,
    uint32_t original_image_width, uint32_t original_image_height,
    const nvimgcodecRegion_t& region
) {
    if (region.ndim == 0) {
        return;
    }

    assert(region.ndim == 2);
    assert(original_image_width <= static_cast<uint32_t>(std::numeric_limits<int>::max()));
    assert(original_image_height <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

    switch (target_image_info.buffer_kind) {
    case NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE:
        fill_out_of_bounds_region_device(
            target_image_info, static_cast<int32_t>(original_image_width), static_cast<int32_t>(original_image_height), region
        );
        break;
    case NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST:
        throw std::runtime_error("Out of bound ROI fill is not supported in host buffers");
        break;
    default:
        throw std::runtime_error("Unsupported buffer kind: " + std::to_string(target_image_info.buffer_kind));
    }
}

std::string verify_region_fill_support(const nvimgcodecRegion_t& region, const nvimgcodecImageInfo_t& target_image_info)
{
    if (region.ndim != 2) {
        return "Region out of bounds fill is supported only for 2 dimensions.";
    }

    uint32_t component_idx = 0;
    for (uint32_t p = 0; p < target_image_info.num_planes; ++p) {
        auto bytes_per_pixel = get_num_bytes_per_pixel(target_image_info, p);
        if (bytes_per_pixel != 1 && bytes_per_pixel != 2 && bytes_per_pixel != 4) {
            return "Out of bounds region fill is supported only for 1, 2 or 4 byte pixel type.";
        }

        for (uint32_t c = 1; c < target_image_info.plane_info[p].num_channels; ++c) {
            if (get_fill_bits(region, c + component_idx) != get_fill_bits(region, component_idx)) {
                return "Out of bounds region fill value must be the same for all channels in the same plane.";
            }
        }
        component_idx += target_image_info.plane_info[p].num_channels;
    }

    if (region.out_of_bounds_policy != NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT) {
        return "Unsupported out_of_bounds_policy: " + std::to_string(region.out_of_bounds_policy);
    }

    static const std::array supported_sample_data_type {
        static_cast<nvimgcodecSampleDataType_t>(0), // if type was not set, will assume sample value = 0
        NVIMGCODEC_SAMPLE_DATA_TYPE_INT8,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8,
        NVIMGCODEC_SAMPLE_DATA_TYPE_INT16,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16,
        NVIMGCODEC_SAMPLE_DATA_TYPE_INT32,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32,
        NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32,
    };
    for (const auto& fill_sample : region.out_of_bounds_samples) {
        if (std::count(supported_sample_data_type.begin(), supported_sample_data_type.end(), fill_sample.type) == 0) {
            return "Unsupported fill sample type: " + std::to_string(fill_sample.type);
        }
    }

    return "";
}

}  // namespace nvimgcodec