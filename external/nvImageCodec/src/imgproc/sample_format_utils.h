/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace nvimgcodec {

constexpr bool IsPlanar(nvimgcodecSampleFormat_t fmt)
{
    switch (fmt)
    {
    case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
    case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:
    case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
    case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
    case NVIMGCODEC_SAMPLEFORMAT_P_YA:
    case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        return true;
    default:
        return false;
    }
}

constexpr bool IsRgb(nvimgcodecSampleFormat_t fmt) {
    switch (fmt) {
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
            return true;
        default:
            return false;
    }
}

constexpr bool IsBgr(nvimgcodecSampleFormat_t fmt) {
    switch (fmt) {
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
            return true;
        default:
            return false;
    }
}

constexpr bool IsGray(nvimgcodecSampleFormat_t fmt) {
    switch (fmt) {
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y:
            return true;
        default:
            return false;
    }
}

constexpr int NumberOfChannels(const nvimgcodecImageInfo_t& info)
{
    return IsPlanar(info.sample_format) ? info.num_planes : info.plane_info[0].num_channels;
}

constexpr bool IsSampleFormatInterleaved(nvimgcodecSampleFormat_t sample_format)
{
    //First bit of sample format says if this is interleaved or not
    return static_cast<int>(sample_format) % 2 == 0 ;
}

}  // namespace nvimgcodec