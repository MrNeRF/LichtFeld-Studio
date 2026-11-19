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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "iimage_decoder_factory.h"
#include "iimage_encoder_factory.h"
#include "iimage_parser_factory.h"

#include "icodec.h"

namespace nvimgcodec {

class IImageParser;
class IImageEncoder;
class IImageDecoder;
class ILogger;

class Codec : public ICodec
{
  public:
    explicit Codec(ILogger* logger, const char* name);
    const std::string& name() const override;
    bool hasRegistrations() const override;
    std::unique_ptr<IImageParser> createParser(nvimgcodecCodeStreamDesc_t* code_stream) const override;
    std::vector<IImageParserFactory*> getParserFactories() const override;
    std::vector<IImageDecoderFactory*> getDecoderFactories() const override;
    std::vector<IImageEncoderFactory*> getEncoderFactories() const override;
    void registerEncoderFactory(std::unique_ptr<IImageEncoderFactory> factory, float priority) override;
    bool unregisterEncoderFactory(const nvimgcodecEncoderDesc_t* desc) override;
    void registerDecoderFactory(std::unique_ptr<IImageDecoderFactory> factory, float priority) override;
    bool unregisterDecoderFactory(const nvimgcodecDecoderDesc_t* desc) override;
    void registerParserFactory(std::unique_ptr<IImageParserFactory> factory, float priority) override;
    bool unregisterParserFactory(const nvimgcodecParserDesc_t* desc) override;

    void registerFileExtension(const std::string& file_extension) override;
    void unregisterFileExtension(const std::string& file_extension) override;
    const std::vector<std::string>& getFileExtensions() const override;
    bool hasFileExtension(const std::string& file_extension) const override;
    bool isDefaultFileExtension(const std::string& file_extension) const override;
    void setDefaultFileExtension(const std::string& file_extension, bool is_default) override;

  private:
    ILogger* logger_;
    std::string name_;
    std::multimap<float, std::unique_ptr<IImageParserFactory>> parsers_;
    std::multimap<float, std::unique_ptr<IImageEncoderFactory>> encoders_;
    std::multimap<float, std::unique_ptr<IImageDecoderFactory>> decoders_;
    std::vector<std::string> file_extensions_;
    std::map<std::string, int> file_extension_refcount_;
    std::set<std::string> default_file_extensions_;
};
} // namespace nvimgcodec
