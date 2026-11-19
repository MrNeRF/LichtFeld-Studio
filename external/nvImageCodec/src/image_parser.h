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

#pragma once

#include "iimage_parser.h"
#include <memory>
#include <nvimgcodec.h>
#include <string>

namespace nvimgcodec {
    class ImageParser : public IImageParser {
    public:
        explicit ImageParser(const nvimgcodecParserDesc_t* desc);
        ~ImageParser() override;
        std::string getParserId() const override;
        std::string getCodecName() const override;
        nvimgcodecStatus_t getCodeStreamInfo(nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecCodeStreamInfo_t* codestream_info) override;
        nvimgcodecStatus_t getImageInfo(nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageInfo_t* image_info) override;

    private:
        const nvimgcodecParserDesc_t* parser_desc_;
        nvimgcodecParser_t parser_;
    };

} // namespace nvimgcodec