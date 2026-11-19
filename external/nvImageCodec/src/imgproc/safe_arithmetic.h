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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "nvimgcodec.h"

namespace nvimgcodec {

// Compute out = a * b in size_t, returning false if the product would
// overflow. Uses __builtin_mul_overflow on GCC/Clang (the only compilers
// supported on Linux/macOS) and falls back to a portable division-based
// check on MSVC, which has no equivalent builtin for size_t.
//
// Use whenever attacker-controlled metadata (image dimensions, channel
// counts, tile geometry, ...) is multiplied to produce a buffer size or
// element count: an unchecked product can wrap and produce an undersized
// allocation that subsequent decode writes overrun.
inline bool SafeMulSizeT(size_t a, size_t b, size_t& out)
{
#if defined(__GNUC__) || defined(__clang__)
    return !__builtin_mul_overflow(a, b, &out);
#else
    // Parens around `max` suppress the Windows <windows.h> max() macro.
    if (b != 0 && a > (std::numeric_limits<size_t>::max)() / b)
        return false;
    out = a * b;
    return true;
#endif
}

// Compute out = a * b * c in size_t, failing on overflow at either step.
inline bool SafeMul3SizeT(size_t a, size_t b, size_t c, size_t& out)
{
    size_t partial = 0;
    return SafeMulSizeT(a, b, partial) && SafeMulSizeT(partial, c, out);
}

// Compute out = a + b in size_t, returning false if the sum would overflow.
inline bool SafeAddSizeT(size_t a, size_t b, size_t& out)
{
#if defined(__GNUC__) || defined(__clang__)
    return !__builtin_add_overflow(a, b, &out);
#else
    // Parens around `max` suppress the Windows <windows.h> max() macro.
    if (a > (std::numeric_limits<size_t>::max)() - b)
        return false;
    out = a + b;
    return true;
#endif
}

// Byte size of one plane: row_stride * height (row_stride is already in bytes).
// Returns false on overflow.
inline bool SafePlaneByteSize(const nvimgcodecImagePlaneInfo_t& plane, size_t& out)
{
    return SafeMulSizeT(plane.row_stride, plane.height, out);
}

// Fill out_offsets[p] with the byte offset of plane p inside a planar buffer:
// the running sum of row_stride * height over all preceding planes i < p, per
// the C image_info buffer-layout contract (planes sit back-to-back, each plane
// honouring its own row_stride and height). out_offsets must hold at least
// info.num_planes entries. Returns false if any product or running sum would
// overflow size_t (e.g. attacker-controlled dimensions), leaving out_offsets
// partially written.
inline bool SafePlaneByteOffsets(const nvimgcodecImageInfo_t& info, size_t* out_offsets)
{
    size_t running = 0;
    for (uint32_t p = 0; p < info.num_planes; ++p) {
        out_offsets[p] = running;
        size_t plane_bytes = 0;
        if (!SafePlaneByteSize(info.plane_info[p], plane_bytes))
            return false;
        if (!SafeAddSizeT(running, plane_bytes, running))
            return false;
    }
    return true;
}

}  // namespace nvimgcodec
