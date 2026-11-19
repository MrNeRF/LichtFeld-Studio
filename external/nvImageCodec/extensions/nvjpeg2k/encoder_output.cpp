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

#include "encoder_output.h"

#include <cstdio>

namespace nvjpeg2k {
namespace detail {

nvimgcodecStatus_t unmap_output(
    nvimgcodecIoStreamDesc_t* io_stream, void*& mapped_data, size_t mapped_capacity) noexcept
{
    if (!mapped_data)
        return NVIMGCODEC_STATUS_SUCCESS;

    void* data = mapped_data;
    mapped_data = nullptr;
    if (!io_stream || !io_stream->unmap)
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    return io_stream->unmap(io_stream->instance, data, mapped_capacity);
}

nvimgcodecStatus_t finalize_output(nvimgcodecIoStreamDesc_t* io_stream, void*& mapped_data,
    size_t mapped_capacity, size_t produced_length, const void* fallback_data) noexcept
{
    if (!io_stream)
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;

    const bool direct_output = fallback_data == nullptr;
    if (direct_output && !mapped_data)
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;

    // Unmap before seek/write/flush. A flush is allowed to resize the backing
    // allocation and would otherwise invalidate the outstanding mapping.
    nvimgcodecStatus_t status = unmap_output(io_stream, mapped_data, mapped_capacity);
    if (status != NVIMGCODEC_STATUS_SUCCESS)
        return status;

    if (produced_length > mapped_capacity)
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;

    if (direct_output) {
        if (!io_stream->seek)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        status = io_stream->seek(io_stream->instance, static_cast<ptrdiff_t>(produced_length), SEEK_SET);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
            return status;
    } else {
        if (!io_stream->seek || !io_stream->write)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        status = io_stream->seek(io_stream->instance, 0, SEEK_SET);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
            return status;

        size_t output_size = 0;
        status = io_stream->write(io_stream->instance, &output_size,
            const_cast<void*>(fallback_data), produced_length);
        if (status != NVIMGCODEC_STATUS_SUCCESS)
            return status;
        if (output_size != produced_length)
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }

    if (!io_stream->flush)
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    return io_stream->flush(io_stream->instance);
}

} // namespace detail
} // namespace nvjpeg2k
