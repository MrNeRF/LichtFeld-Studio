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

#include "iimage_encoder.h"
#include "iimage_encoder_factory.h"
#include "image_generic_codec.h"
#include <nvimgcodec.h>

namespace nvimgcodec {

    class ImageGenericEncoder : public ImageGenericCodec<ImageGenericEncoder, IImageEncoderFactory, IImageEncoder> {
    public:
        ImageGenericEncoder(ILogger* logger,
                            ICodecRegistry* codec_registry,
                            const nvimgcodecExecutionParams_t* exec_params,
                            const char* options = nullptr)
            : Base(logger, codec_registry, exec_params, options) {}

        ~ImageGenericEncoder() override = default;

        void canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
                       const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format) noexcept;
        ProcessingResultsPromise::FutureImpl encode(
            const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params) noexcept;

        using Base = ImageGenericCodec<ImageGenericEncoder, IImageEncoderFactory, IImageEncoder>;
        using Factory = IImageEncoderFactory;
        using Processor = IImageEncoder;
        using Entry = SampleEntry<typename Base::ProcessorEntry>;

        static const std::string& process_name() {
            static std::string name{"encode"};
            return name;
        };

        static size_t getProcessorCount(ICodec* codec) {
            return codec->getEncodersNum();
        }

        static IImageEncoderFactory* getFactory(ICodec* codec, int i) {
            return codec->getEncoderFactory(i);
        }

        static std::string getId(Factory* factory) {
            return factory->getEncoderId();
        }

        static std::unique_ptr<IImageEncoder> createInstance(
            const Factory* factory, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
            return factory->createEncoder(exec_params, options);
        }

        static bool hasBatchedAPI(Processor* processor) {
            return false; // no batched API for encoders
        }

        static int getMiniBatchSize(Processor* processor) {
            return -1;
        }

        static int getProcessorsNum(ICodec* codec) {
            return codec->getEncodersNum();
        }

        nvimgcodecProcessingStatus_t canProcessImpl(Entry& sample, ProcessorEntry* processor, int tid) noexcept;
        bool processImpl(Entry& sample, int tid) noexcept;
        bool processBatchImpl(ProcessorEntry& processor) noexcept;
        void sortSamples();
        bool copyToTempBuffers(Entry& sample, int tid);
    };

} // namespace nvimgcodec
