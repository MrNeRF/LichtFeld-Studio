/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <string>

namespace nvimgcodec {

    std::string file_ext_to_codec(const std::string& file_ext) {
        static std::map<std::string, std::string> ext2codec = {
            {".bmp", "bmp"},
            {".j2c", "jpeg2k"},
            {".j2k", "jpeg2k"},
            {".jp2", "jpeg2k"},
            {".tiff", "tiff"},
            {".tif", "tiff"},
            {".jpg", "jpeg"},
            {".jpeg", "jpeg"},
            {".ppm", "pnm"},
            {".pgm", "pnm"},
            {".pbm", "pnm"},
            {".pnm", "pnm"},
            {".webp", "webp"},
            {".png", "png"}};
        std::string codec_name{};
        auto it = ext2codec.find(file_ext);
        if (it != ext2codec.end()) {
            codec_name = it->second;
        }
        return codec_name;
    }

} // namespace nvimgcodec
