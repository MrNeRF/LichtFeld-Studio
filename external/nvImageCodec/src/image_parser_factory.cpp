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
#include "image_parser_factory.h"
#include <cassert>
#include <iostream>
#include "image_parser.h"

namespace nvimgcodec {

ImageParserFactory::ImageParserFactory(const nvimgcodecParserDesc_t* desc)
    : parser_desc_(desc)
{
}
std::string ImageParserFactory::getParserId() const
{
    return parser_desc_->id;
}

std::string ImageParserFactory::getCodecName() const
{
    return parser_desc_->codec;
}

const nvimgcodecParserDesc_t* ImageParserFactory::getParserDesc() const
{
    return parser_desc_;
}

std::unique_ptr<IImageParser> ImageParserFactory::createParser() const
{
    return std::make_unique<ImageParser>(parser_desc_);
}

bool ImageParserFactory::canParse(nvimgcodecCodeStreamDesc_t* code_stream)
{
    assert(code_stream);
    int result = 0;
    parser_desc_->canParse(parser_desc_->instance, &result, code_stream);
    return result;
}

} // namespace nvimgcodec