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
#include "image_parser.h"
#include <cassert>
#include <iostream>

namespace nvimgcodec {

    ImageParser::ImageParser(const nvimgcodecParserDesc_t* desc)
        : parser_desc_(desc) {
        parser_desc_->create(parser_desc_->instance, &parser_);
    }

    ImageParser::~ImageParser() {
        parser_desc_->destroy(parser_);
    }
    std::string ImageParser::getParserId() const {
        return parser_desc_->id;
    }

    std::string ImageParser::getCodecName() const {
        return parser_desc_->codec;
    }

    nvimgcodecStatus_t ImageParser::getCodeStreamInfo(nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecCodeStreamInfo_t* codestream_info) {
        assert(code_stream);
        assert(parser_desc_->getCodeStreamInfo);
        return parser_desc_->getCodeStreamInfo(parser_, codestream_info, code_stream);
    }

    nvimgcodecStatus_t ImageParser::getImageInfo(nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageInfo_t* image_info) {
        assert(code_stream);
        assert(parser_desc_->getImageInfo);
        return parser_desc_->getImageInfo(parser_, image_info, code_stream);
    }

} // namespace nvimgcodec