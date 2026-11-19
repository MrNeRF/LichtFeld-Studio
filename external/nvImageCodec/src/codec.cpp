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
#include "codec.h"
#include <iostream>
#include "image_decoder.h"
#include "image_encoder.h"
#include "image_parser.h"
#include "log.h"

namespace nvimgcodec {
    
Codec::Codec(ILogger* logger, const char* name)
    : logger_(logger)
    , name_(name)
{
}

bool Codec::hasRegistrations() const
{
    return !parsers_.empty() || !encoders_.empty() || !decoders_.empty() || !file_extensions_.empty();
}

std::unique_ptr<IImageParser> Codec::createParser(nvimgcodecCodeStreamDesc_t* code_stream) const
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::getParser " << name_);
    for (const auto& entry : parsers_) {
        NVIMGCODEC_LOG_TRACE(logger_, "- probing parser:" << entry.second->getParserId());
        if (entry.second->canParse(code_stream)) {
            NVIMGCODEC_LOG_TRACE(logger_, "- - can parse");
            return entry.second->createParser();
        }
    }

    return nullptr;
}

std::vector<IImageParserFactory*> Codec::getParserFactories() const
{
    std::vector<IImageParserFactory*> factories;
    factories.reserve(parsers_.size());
    for (const auto& entry : parsers_) {
        factories.push_back(entry.second.get());
    }
    return factories;
}

std::vector<IImageDecoderFactory*> Codec::getDecoderFactories() const
{
    std::vector<IImageDecoderFactory*> factories;
    factories.reserve(decoders_.size());
    for (const auto& entry : decoders_) {
        factories.push_back(entry.second.get());
    }
    return factories;
}

std::vector<IImageEncoderFactory*> Codec::getEncoderFactories() const
{
    std::vector<IImageEncoderFactory*> factories;
    factories.reserve(encoders_.size());
    for (const auto& entry : encoders_) {
        factories.push_back(entry.second.get());
    }
    return factories;
}

const std::string& Codec::name() const
{
    return name_;
}

void Codec::registerParserFactory(
    std::unique_ptr<IImageParserFactory> parserFactory, float priority)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::registerParser");
    parsers_.emplace(priority, std::move(parserFactory));
}

bool Codec::unregisterParserFactory(const nvimgcodecParserDesc_t* desc)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::unregisterParser");
    for (auto it = parsers_.begin(); it != parsers_.end(); ++it) {
        if (it->second->getParserDesc() == desc) {
            parsers_.erase(it);
            return true;
        }
    }
    return false;
}

void Codec::registerEncoderFactory(
    std::unique_ptr<IImageEncoderFactory> encoderFactory, float priority)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::registerEncoder");
    encoders_.emplace(priority, std::move(encoderFactory));
}

bool Codec::unregisterEncoderFactory(const nvimgcodecEncoderDesc_t* desc)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::unregisterEncoder");
    for (auto it = encoders_.begin(); it != encoders_.end(); ++it) {
        if (it->second->getEncoderDesc() == desc) {
            encoders_.erase(it);
            return true;
        }
    }
    return false;
}

void Codec::registerDecoderFactory(
    std::unique_ptr<IImageDecoderFactory> decoderFactory, float priority)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::registerDecoder");
    decoders_.emplace(priority, std::move(decoderFactory));
}

bool Codec::unregisterDecoderFactory(const nvimgcodecDecoderDesc_t* desc)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::unregisterDecoder");
    for (auto it = decoders_.begin(); it != decoders_.end(); ++it) {
        if (it->second->getDecoderDesc() == desc) {
            decoders_.erase(it);
            return true;
        }
    }
    return false;
}

void Codec::registerFileExtension(const std::string& file_extension)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::registerFileExtension " << name_ << " " << file_extension);
    auto& count = file_extension_refcount_[file_extension];
    if (count == 0) {
        file_extensions_.push_back(file_extension);
    }
    ++count;
}

void Codec::unregisterFileExtension(const std::string& file_extension)
{
    NVIMGCODEC_LOG_TRACE(logger_, "Codec::unregisterFileExtension " << name_ << " " << file_extension);
    auto it = file_extension_refcount_.find(file_extension);
    if (it == file_extension_refcount_.end()) {
        return;
    }
    --it->second;
    if (it->second <= 0) {
        file_extension_refcount_.erase(it);
        default_file_extensions_.erase(file_extension);
        for (auto ext_it = file_extensions_.begin(); ext_it != file_extensions_.end(); ++ext_it) {
            if (*ext_it == file_extension) {
                file_extensions_.erase(ext_it);
                break;
            }
        }
    }
}

const std::vector<std::string>& Codec::getFileExtensions() const
{
    return file_extensions_;
}

bool Codec::hasFileExtension(const std::string& file_extension) const
{
    auto it = file_extension_refcount_.find(file_extension);
    return it != file_extension_refcount_.end() && it->second > 0;
}

bool Codec::isDefaultFileExtension(const std::string& file_extension) const
{
    return default_file_extensions_.count(file_extension) != 0;
}

void Codec::setDefaultFileExtension(const std::string& file_extension, bool is_default)
{
    if (is_default) {
        default_file_extensions_.insert(file_extension);
    } else {
        default_file_extensions_.erase(file_extension);
    }
}

} // namespace nvimgcodec
