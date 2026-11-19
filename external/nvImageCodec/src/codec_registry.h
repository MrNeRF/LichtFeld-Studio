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

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "icodec.h"
#include "icodec_registry.h"

namespace nvimgcodec {

class IImageParser;
class ILogger;

class CodecRegistry : public ICodecRegistry
{
  public:
    CodecRegistry(ILogger* logger);
    void registerCodec(std::unique_ptr<ICodec> codec) override;
    std::unique_ptr<IImageParser> getParser(
        nvimgcodecCodeStreamDesc_t* code_stream) const override;
    ICodec* getCodecByName(const char* name) override;

    size_t getCodecsCount() const override;
    ICodec* getCodecByIndex(size_t index) override;

    void registerFileExtension(const std::string& file_extension, const std::string& codec_name, bool is_default) override;
    void unregisterFileExtension(const std::string& file_extension, const std::string& codec_name) override;
    std::vector<std::string> getCodecNames(const std::string& file_extension) const override;
    std::vector<std::string> getFileExtensions() const override;
    std::vector<std::string> getFileExtensions(const std::string& codec_name) const override;

  private:
    void setDefaultCodecForExtension(const std::string& file_extension, ICodec* codec);

    ILogger* logger_;
    std::deque<ICodec*> codec_ptrs_;
    std::map<std::string, std::unique_ptr<ICodec>> by_name_;
};
} // namespace nvimgcodec