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
#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include "iimage_decoder.h"

namespace nvimgcodec {
class IDecodeState;
class IImage;
class ICodeStream;

class ImageDecoder : public IImageDecoder
{
  public:
    // Throws if the plugin fails to create the decoder.
    ImageDecoder(const nvimgcodecDecoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    ~ImageDecoder() override;
    nvimgcodecBackendKind_t getBackendKind() const override;
    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const override;
    bool canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params,
        nvimgcodecProcessingStatus_t* status, int thread_idx) const override;
    bool decode(nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecDecodeParams_t* params, int thread_idx) override;
    bool decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
        const nvimgcodecDecodeParams_t* params, int batch_size, int thread_idx) override;
    bool hasDecodeBatch() const override;
    int getMiniBatchSize() const override;
    const char* decoderId() const override;

  private:
    const nvimgcodecDecoderDesc_t* decoder_desc_;
    const nvimgcodecExecutionParams_t* exec_params_;
    nvimgcodecDecoder_t decoder_ = nullptr;
};

} // namespace nvimgcodec