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

#include <array>
#include <nvimgcodec.h>
#include <vector>

namespace nvimgcodec {

    class JPEG2KParserPlugin {
    public:
        explicit JPEG2KParserPlugin(const nvimgcodecFrameworkDesc_t* framework);
        nvimgcodecParserDesc_t* getParserDesc();

    private:
        struct Parser {
            Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework);

            nvimgcodecStatus_t parseJP2(nvimgcodecIoStreamDesc_t* io_stream);
            nvimgcodecStatus_t parseCodeStream(nvimgcodecIoStreamDesc_t* io_stream);
            nvimgcodecStatus_t getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream);
            nvimgcodecStatus_t getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);

            static nvimgcodecStatus_t static_destroy(nvimgcodecParser_t parser);
            static nvimgcodecStatus_t static_get_codestream_info(
                nvimgcodecParser_t parser, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream);
            static nvimgcodecStatus_t static_get_image_info(
                nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);

            const char* plugin_id_;
            const nvimgcodecFrameworkDesc_t* framework_;
            uint16_t num_components = 0, CSiz = 0;
            uint32_t height = 0, width = 0;
            // True once a JP2 ihdr box has been parsed in the current getImageInfo call.
            // Distinguishes "no ihdr present (bare codestream)" from a JP2 whose ihdr legitimately
            // reports a value that would otherwise collide with a sentinel.
            bool has_ihdr = false;
            uint8_t bits_per_component = 0;
            nvimgcodecColorSpec_t color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;
            uint32_t XSiz = 0, YSiz = 0, XOSiz = 0, YOSiz = 0, XTSiz = 0, YTSiz = 0, XTOSiz = 0, YTOSiz = 0;
            std::array<uint8_t, NVIMGCODEC_MAX_NUM_PLANES> XRSiz{}, YRSiz{}, Ssiz{};
        };

        nvimgcodecStatus_t canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream);
        nvimgcodecStatus_t create(nvimgcodecParser_t* parser);

        static nvimgcodecStatus_t static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream);
        static nvimgcodecStatus_t static_create(void* instance, nvimgcodecParser_t* parser);

        static constexpr const char* plugin_id_ = "jpeg2k_parser";
        const nvimgcodecFrameworkDesc_t* framework_;
        nvimgcodecParserDesc_t parser_desc_;
    };

    nvimgcodecStatus_t get_jpeg2k_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);

} // namespace nvimgcodec