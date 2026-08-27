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
#include "image_decoder.h"
#include "image_encoder.h"
#include "image_parser.h"
#include "log.h"
#include <iostream>

namespace nvimgcodec {

    Codec::Codec(ILogger* logger, const char* name)
        : logger_(logger), name_(name) {
    }

    std::unique_ptr<IImageParser> Codec::createParser(nvimgcodecCodeStreamDesc_t* code_stream) const {
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

    int Codec::getDecodersNum() const {
        return decoders_.size();
    }

    IImageDecoderFactory* Codec::getDecoderFactory(int index) const {
        if (size_t(index) >= decoders_.size()) {
            return nullptr;
        }
        auto it = decoders_.begin();
        for (int i = 0; i < index; ++i)
            it++;
        return it != decoders_.end() ? it->second.get() : nullptr;
    }

    int Codec::getEncodersNum() const {
        return encoders_.size();
    }

    IImageEncoderFactory* Codec::getEncoderFactory(int index) const {
        if (size_t(index) >= encoders_.size()) {
            return nullptr;
        }
        auto it = encoders_.begin();
        for (int i = 0; i < index; ++i)
            it++;
        return it != encoders_.end() ? it->second.get() : nullptr;
    }

    const std::string& Codec::name() const {
        return name_;
    }

    void Codec::registerParserFactory(
        std::unique_ptr<IImageParserFactory> parserFactory, float priority) {
        NVIMGCODEC_LOG_TRACE(logger_, "Codec::registerParser");
        parsers_.emplace(priority, std::move(parserFactory));
    }

    void Codec::unregisterParserFactory(const std::string parser_id) {
        NVIMGCODEC_LOG_TRACE(logger_, "Codec::unregisterParser");
        for (auto it = parsers_.begin(); it != parsers_.end(); ++it) {
            if (it->second->getParserId() == parser_id) {
                parsers_.erase(it);
                return;
            }
        }
    }

    void Codec::registerEncoderFactory(
        std::unique_ptr<IImageEncoderFactory> encoderFactory, float priority) {
        NVIMGCODEC_LOG_TRACE(logger_, "Codec::registerEncoder");
        encoders_.emplace(priority, std::move(encoderFactory));
    }

    void Codec::unregisterEncoderFactory(const std::string encoder_id) {
        NVIMGCODEC_LOG_TRACE(logger_, "Codec::unregisterEncoder");
        for (auto it = encoders_.begin(); it != encoders_.end(); ++it) {
            if (it->second->getEncoderId() == encoder_id) {
                encoders_.erase(it);
                return;
            }
        }
    }

    void Codec::registerDecoderFactory(
        std::unique_ptr<IImageDecoderFactory> decoderFactory, float priority) {
        NVIMGCODEC_LOG_TRACE(logger_, "Codec::registerDecoder");
        decoders_.emplace(priority, std::move(decoderFactory));
    }

    void Codec::unregisterDecoderFactory(const std::string decoder_id) {
        NVIMGCODEC_LOG_TRACE(logger_, "Codec::unregisterDecoder");
        for (auto it = decoders_.begin(); it != decoders_.end(); ++it) {
            if (it->second->getDecoderId() == decoder_id) {
                decoders_.erase(it);
                return;
            }
        }
    }

} // namespace nvimgcodec
