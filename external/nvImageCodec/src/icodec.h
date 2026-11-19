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
#include <vector>

namespace nvimgcodec {

class IImageParser;
class IImageEncoder;
class IImageDecoder;
class IImageParserFactory;
class IImageEncoderFactory;
class IImageDecoderFactory;

class ICodec
{
  public:
    virtual ~ICodec() = default;

    virtual const std::string& name() const = 0;
    virtual bool hasRegistrations() const = 0;

    virtual std::unique_ptr<IImageParser> createParser(nvimgcodecCodeStreamDesc_t* code_stream) const = 0;
    virtual std::vector<IImageParserFactory*> getParserFactories() const = 0;
    virtual std::vector<IImageDecoderFactory*> getDecoderFactories() const = 0;
    virtual std::vector<IImageEncoderFactory*> getEncoderFactories() const = 0;

    virtual void registerEncoderFactory(std::unique_ptr<IImageEncoderFactory> factory, float priority) = 0;
    virtual bool unregisterEncoderFactory(const nvimgcodecEncoderDesc_t* desc) = 0;
    virtual void registerDecoderFactory(std::unique_ptr<IImageDecoderFactory> factory, float priority) = 0;
    virtual bool unregisterDecoderFactory(const nvimgcodecDecoderDesc_t* desc) = 0;
    virtual void registerParserFactory(std::unique_ptr<IImageParserFactory> factory, float priority) = 0;
    virtual bool unregisterParserFactory(const nvimgcodecParserDesc_t* desc) = 0;

    virtual void registerFileExtension(const std::string& file_extension) = 0;
    virtual void unregisterFileExtension(const std::string& file_extension) = 0;
    virtual const std::vector<std::string>& getFileExtensions() const = 0;
    virtual bool hasFileExtension(const std::string& file_extension) const = 0;
    virtual bool isDefaultFileExtension(const std::string& file_extension) const = 0;
    virtual void setDefaultFileExtension(const std::string& file_extension, bool is_default) = 0;
};
} // namespace nvimgcodec
