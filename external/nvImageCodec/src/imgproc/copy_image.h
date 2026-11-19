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

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include "nvimgcodec.h"
#include "imgproc/exception.h"
#include "imgproc/safe_arithmetic.h"
#include "imgproc/type_utils.h"

namespace nvimgcodec {

// True when every plane of `info` is row-packed (row_stride equals the natural
// row size), i.e. the buffer has no row padding. Combined with the C image_info
// contract that planes sit back-to-back at the cumulative `row_stride * height`
// offset, this means the whole buffer is contiguous. Each plane's stride is
// checked, not only plane 0 (planes may differ, e.g. subsampled chroma).
inline bool IsBufferContiguous(const nvimgcodecImageInfo_t& info)
{
    for (uint32_t c = 0; c < info.num_planes; ++c) {
        const auto& plane = info.plane_info[c];
        const size_t row_size = static_cast<size_t>(plane.width) * plane.num_channels * TypeSize(plane.sample_type);
        if (plane.row_stride != row_size)
            return false;
    }
    return true;
}

// Copy an image between two buffers that describe the same logical image
// (identical per-plane width/height/num_channels/sample_type) but may differ in
// row padding and host/device residency. The caller is responsible for the
// device context (DeviceGuard), stream synchronization, and event recording;
// this only enqueues the copy on `stream`.
//
// When both buffers are fully contiguous the whole image transfers in a single
// cudaMemcpyAsync regardless of plane count. Otherwise each plane is copied with
// cudaMemcpy2DAsync using its own width/height (so non-uniform planes such as
// subsampled chroma are handled), with each plane's start offset taken as the
// running sum of row_stride * height per the C image_info buffer-layout contract.
inline void CopyImage(
    const nvimgcodecImageInfo_t& dst_info,
    const nvimgcodecImageInfo_t& src_info,
    cudaMemcpyKind direction,
    cudaStream_t stream)
{
    assert(src_info.num_planes == dst_info.num_planes);
    assert(GetImageSize(src_info) == GetImageSize(dst_info));

    if (IsBufferContiguous(src_info) && IsBufferContiguous(dst_info)) {
        // No padding on either side and matching layouts, so the whole image is
        // a single contiguous block on both sides - one copy moves all planes.
        CHECK_CUDA(cudaMemcpyAsync(
            dst_info.buffer, src_info.buffer, GetImageSize(src_info), direction, stream));
        return;
    }

    size_t src_offsets[NVIMGCODEC_MAX_NUM_PLANES];
    size_t dst_offsets[NVIMGCODEC_MAX_NUM_PLANES];
    if (!SafePlaneByteOffsets(src_info, src_offsets) || !SafePlaneByteOffsets(dst_info, dst_offsets))
        FatalError(INTERNAL_ERROR, "plane byte offsets overflow size_t");
    for (uint32_t c = 0; c < src_info.num_planes; ++c) {
        const auto& src_plane = src_info.plane_info[c];
        const auto& dst_plane = dst_info.plane_info[c];
        const size_t row_size = static_cast<size_t>(src_plane.width) * src_plane.num_channels * TypeSize(src_plane.sample_type);
        CHECK_CUDA(cudaMemcpy2DAsync(
            static_cast<uint8_t*>(dst_info.buffer) + dst_offsets[c], dst_plane.row_stride,
            static_cast<const uint8_t*>(src_info.buffer) + src_offsets[c], src_plane.row_stride,
            row_size, src_plane.height,
            direction, stream));
    }
}

}  // namespace nvimgcodec
