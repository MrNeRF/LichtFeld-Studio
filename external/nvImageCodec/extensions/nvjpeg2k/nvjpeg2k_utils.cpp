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

#include "nvjpeg2k_utils.h"
#include "nvjpeg2k.h"

namespace nvjpeg2k {

    int flat_version(int major, int minor, int patch) {
        return ((major) * 1000000 + (minor) * 1000 + (patch));
    }

    int get_version() {
        int major = -1, minor = -1, patch = -1;
        if (NVJPEG2K_STATUS_SUCCESS == nvjpeg2kGetProperty(MAJOR_VERSION, &major) &&
            NVJPEG2K_STATUS_SUCCESS == nvjpeg2kGetProperty(MINOR_VERSION, &minor) &&
            NVJPEG2K_STATUS_SUCCESS == nvjpeg2kGetProperty(PATCH_LEVEL, &patch)) {
            return flat_version(major, minor, patch);
        } else {
            return -1;
        }
    }

    bool is_version_at_least(int major, int minor, int patch) {
        return get_version() >= flat_version(major, minor, patch);
    }

} // namespace nvjpeg2k