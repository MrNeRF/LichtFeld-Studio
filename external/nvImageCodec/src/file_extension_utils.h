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

#include <nvimgcodec.h>
#include <cctype>
#include <string>

namespace nvimgcodec {

inline std::string normalizeFileExtension(const char* file_extension)
{
    if (file_extension == nullptr || file_extension[0] == '\0') {
        return {};
    }
    std::string ext(file_extension);
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (ext[0] != '.') {
        ext.insert(ext.begin(), '.');
    }
    return ext;
}

// Fits a C API slot of NVIMGCODEC_MAX_FILE_EXTENSION_SIZE (including terminator).
inline bool fileExtensionFitsCApiSlot(const std::string& normalized)
{
    return !normalized.empty() && normalized.size() < NVIMGCODEC_MAX_FILE_EXTENSION_SIZE;
}

// Fits a C API slot of NVIMGCODEC_MAX_CODEC_NAME_SIZE (including terminator).
inline bool codecNameFitsCApiSlot(const char* codec)
{
    return codec != nullptr && codec[0] != '\0' && std::char_traits<char>::length(codec) < NVIMGCODEC_MAX_CODEC_NAME_SIZE;
}

} // namespace nvimgcodec
