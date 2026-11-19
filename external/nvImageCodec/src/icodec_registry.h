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
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nvimgcodec {

class ICodec;
class IImageParser;

class ICodecRegistry
{
  public:
    virtual ~ICodecRegistry() = default;
    virtual void registerCodec(std::unique_ptr<ICodec> codec) = 0;
    virtual std::unique_ptr<IImageParser> getParser(
        nvimgcodecCodeStreamDesc_t* code_stream) const = 0;
    virtual ICodec* getCodecByName(const char* name) = 0;
    virtual size_t getCodecsCount() const = 0;
    virtual ICodec* getCodecByIndex(size_t i) = 0;

    virtual void registerFileExtension(const std::string& file_extension, const std::string& codec_name, bool is_default) = 0;
    virtual void unregisterFileExtension(const std::string& file_extension, const std::string& codec_name) = 0;
    virtual std::vector<std::string> getCodecNames(const std::string& file_extension) const = 0;
    virtual std::vector<std::string> getFileExtensions() const = 0;
    virtual std::vector<std::string> getFileExtensions(const std::string& codec_name) const = 0;
};
} // namespace nvimgcodec