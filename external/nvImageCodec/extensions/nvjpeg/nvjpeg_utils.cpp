/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvjpeg_utils.h"

namespace nvjpeg {

NvjpegVersion get_nvjpeg_version() {
    NvjpegVersion v;
    if (NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MAJOR_VERSION, &v.major_ver) &&
        NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MINOR_VERSION, &v.minor_ver) &&
        NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(PATCH_LEVEL, &v.patch_ver)) {
        v.valid = true;
    }
    return v;
}

unsigned int get_nvjpeg_flags(const NvjpegVersion& version, bool fancy_upsampling, unsigned int extra_flags) {
    unsigned int nvjpeg_flags = extra_flags;
#ifdef NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION
    if (version && fancy_upsampling && version >= NvjpegVersion(12, 1, 0)) {
        nvjpeg_flags |= NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION;
    }
#endif
    return nvjpeg_flags;
}

}