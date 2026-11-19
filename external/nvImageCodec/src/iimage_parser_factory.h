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
#include <memory>
#include <string>

namespace nvimgcodec {

class IImageParser;
class IImageParserFactory
{
  public:
    virtual ~IImageParserFactory() = default;
    virtual std::string getParserId() const = 0;
    virtual std::string getCodecName() const = 0;
    virtual const nvimgcodecParserDesc_t* getParserDesc() const = 0;
    virtual bool canParse(nvimgcodecCodeStreamDesc_t* code_stream) = 0;
    virtual std::unique_ptr<IImageParser> createParser() const = 0;
};

} // namespace nvimgcodec