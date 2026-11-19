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

#include "codec_registry.h"
#include "codec.h"
#include "iimage_parser.h"
#include "log.h"

#include <iostream>
#include <set>
#include <stdexcept>

namespace nvimgcodec {

CodecRegistry::CodecRegistry(ILogger* logger)
    : logger_(logger)
{
}

void CodecRegistry::registerCodec(std::unique_ptr<ICodec> codec)
{
    if (by_name_.find(codec->name()) != by_name_.end())
        throw std::invalid_argument("A different codec with the same name already registered.");
    // TODO(janton): Figure out a way to give JPEG a higher priority without this
    //               Perhaps codec should come with a priority too
    if (codec->name() == "jpeg") {
        codec_ptrs_.push_front(codec.get());
    } else {
        codec_ptrs_.push_back(codec.get());
    }
    by_name_.insert(std::make_pair(codec->name(), std::move(codec)));
}

std::unique_ptr<IImageParser> CodecRegistry::getParser(
    nvimgcodecCodeStreamDesc_t* code_stream) const
{
    NVIMGCODEC_LOG_TRACE(logger_, "CodecRegistry::getParser");
    for (auto* codec : codec_ptrs_) {
        std::unique_ptr<IImageParser> parser = codec->createParser(code_stream);
        if (parser) {
            return parser;
        }
    }

    return nullptr;
}

ICodec* CodecRegistry::getCodecByName(const char* name)
{
    auto it = by_name_.find(name);
    if (it != by_name_.end())
        return it->second.get();
    else
        return nullptr;
}

size_t CodecRegistry::getCodecsCount() const
{
    return codec_ptrs_.size();
}

ICodec* CodecRegistry::getCodecByIndex(size_t index)
{
    return codec_ptrs_[index];
}

void CodecRegistry::registerFileExtension(const std::string& file_extension, const std::string& codec_name, bool is_default)
{
    NVIMGCODEC_LOG_TRACE(logger_, "CodecRegistry::registerFileExtension " << file_extension << " -> " << codec_name);
    ICodec* codec = getCodecByName(codec_name.c_str());
    if (codec == nullptr) {
        NVIMGCODEC_LOG_WARNING(logger_, "Codec " << codec_name << " not registered");
        return;
    }

    bool another_has_extension = false;
    for (auto* other : codec_ptrs_) {
        if (other != codec && other->hasFileExtension(file_extension)) {
            another_has_extension = true;
            break;
        }
    }

    codec->registerFileExtension(file_extension);
    if (is_default || !another_has_extension) {
        setDefaultCodecForExtension(file_extension, codec);
    }
}

void CodecRegistry::unregisterFileExtension(const std::string& file_extension, const std::string& codec_name)
{
    NVIMGCODEC_LOG_TRACE(logger_, "CodecRegistry::unregisterFileExtension " << file_extension << " -> " << codec_name);
    ICodec* codec = getCodecByName(codec_name.c_str());
    if (codec == nullptr) {
        return;
    }

    const bool was_default = codec->isDefaultFileExtension(file_extension);
    codec->unregisterFileExtension(file_extension);
    if (was_default && !codec->hasFileExtension(file_extension)) {
        ICodec* next = nullptr;
        for (auto* other : codec_ptrs_) {
            if (other->hasFileExtension(file_extension)) {
                next = other;
                break;
            }
        }
        if (next != nullptr) {
            setDefaultCodecForExtension(file_extension, next);
        }
    }
}

std::vector<std::string> CodecRegistry::getCodecNames(const std::string& file_extension) const
{
    std::vector<std::string> names;
    std::string default_name;
    for (auto* codec : codec_ptrs_) {
        if (!codec->hasFileExtension(file_extension)) {
            continue;
        }
        if (codec->isDefaultFileExtension(file_extension)) {
            default_name = codec->name();
        } else {
            names.push_back(codec->name());
        }
    }
    if (!default_name.empty()) {
        names.insert(names.begin(), std::move(default_name));
    }
    return names;
}

std::vector<std::string> CodecRegistry::getFileExtensions() const
{
    std::vector<std::string> extensions;
    std::set<std::string> seen;
    for (auto* codec : codec_ptrs_) {
        for (const auto& extension : codec->getFileExtensions()) {
            if (seen.insert(extension).second) {
                extensions.push_back(extension);
            }
        }
    }
    return extensions;
}

std::vector<std::string> CodecRegistry::getFileExtensions(const std::string& codec_name) const
{
    auto it = by_name_.find(codec_name);
    if (it == by_name_.end()) {
        return {};
    }
    return it->second->getFileExtensions();
}

void CodecRegistry::setDefaultCodecForExtension(const std::string& file_extension, ICodec* codec)
{
    for (auto* other : codec_ptrs_) {
        other->setDefaultFileExtension(file_extension, other == codec);
    }
}

} // namespace nvimgcodec