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
#include <sstream>
#include <string>

namespace nvjpeg {

    int nvjpeg_flat_version(int major, int minor, int patch) {
        return ((major) * 1000000 + (minor) * 1000 + (patch));
    }

    int nvjpeg_get_version() {
        int major = -1, minor = -1, patch = -1;
        if (NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MAJOR_VERSION, &major) &&
            NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MINOR_VERSION, &minor) &&
            NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(PATCH_LEVEL, &patch)) {
            return nvjpeg_flat_version(major, minor, patch);
        } else {
            return -1;
        }
    }

    bool nvjpeg_at_least(int major, int minor, int patch) {
        return nvjpeg_get_version() >= nvjpeg_flat_version(major, minor, patch);
    }

    unsigned int get_nvjpeg_flags(const char* module_name, const char* options) {

        // if available, we prefer this to be the default (it matches libjpeg implementation)
        bool fancy_upsampling = true;
        unsigned int nvjpeg_extra_flags = 0;
        std::istringstream iss(options ? options : "");
        std::string token;
        while (std::getline(iss, token, ' ')) {
            std::string::size_type colon = token.find(':');
            std::string::size_type equal = token.find('=');
            if (colon == std::string::npos || equal == std::string::npos || colon > equal)
                continue;
            std::string module = token.substr(0, colon);
            if (module != "" && module != module_name)
                continue;
            std::string option = token.substr(colon + 1, equal - colon - 1);
            std::string value_str = token.substr(equal + 1);

            std::istringstream value(value_str);
            if (option == "fancy_upsampling") {
                value >> fancy_upsampling;
            } else if (option == "extra_flags") {
                value >> nvjpeg_extra_flags;
            }
        }

        unsigned int nvjpeg_flags = 0;
        nvjpeg_flags |= nvjpeg_extra_flags;
#ifdef NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION
        if (fancy_upsampling && nvjpeg_at_least(12, 1, 0)) {
            nvjpeg_flags |= NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION;
        }
#endif
        return nvjpeg_flags;
    }

} // namespace nvjpeg