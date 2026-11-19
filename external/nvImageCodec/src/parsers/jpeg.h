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

#include <nvimgcodec.h>
#include <array>
#include <vector>
namespace nvimgcodec {

class JPEGParserPlugin
{
  public:
    explicit JPEGParserPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecParserDesc_t* getParserDesc();
    static constexpr const char* codecName() { return "jpeg"; }
    static constexpr std::array<const char*, 2> fileExtensions() { return {{".jpg", ".jpeg"}}; }

  private:
    struct Parser
    {
        Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework);
        nvimgcodecStatus_t getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream);
        nvimgcodecStatus_t getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);
        static nvimgcodecStatus_t static_destroy(nvimgcodecParser_t parser);
        static nvimgcodecStatus_t static_get_codestream_info(
            nvimgcodecParser_t parser, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream);
        static nvimgcodecStatus_t static_get_image_info(
            nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);

        const char* plugin_id_;
        const nvimgcodecFrameworkDesc_t* framework_;
    };

    nvimgcodecStatus_t canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream);
    nvimgcodecStatus_t create(nvimgcodecParser_t* parser);

    static nvimgcodecStatus_t static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream);
    static nvimgcodecStatus_t static_create(void* instance, nvimgcodecParser_t* parser);

    static constexpr const char* plugin_id_ = "jpeg_parser";
    const nvimgcodecFrameworkDesc_t* framework_;
    nvimgcodecParserDesc_t parser_desc_;
};

nvimgcodecStatus_t get_jpeg_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);

} // namespace nvimgcodec