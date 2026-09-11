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
#include "image_decoder.h"
#include "decode_state_batch.h"
#include "icode_stream.h"
#include "iimage.h"
#include "imgproc/exception.h"
#include "processing_results.h"
#include <algorithm>
#include <cassert>
#include <nvtx3/nvtx3.hpp>

namespace nvimgcodec {

    ImageDecoder::ImageDecoder(const nvimgcodecDecoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options)
        : decoder_desc_(desc), exec_params_(exec_params) {
        auto ret = decoder_desc_->create(decoder_desc_->instance, &decoder_, exec_params, options);
        if (NVIMGCODEC_STATUS_SUCCESS != ret) {
            decoder_ = nullptr;
        }
    }

    ImageDecoder::~ImageDecoder() {
        if (decoder_)
            decoder_desc_->destroy(decoder_);
    }

    nvimgcodecBackendKind_t ImageDecoder::getBackendKind() const {
        return decoder_desc_->backend_kind;
    }

    nvimgcodecStatus_t ImageDecoder::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const {
        assert(decoder_desc_ && decoder_desc_->getMetadata);
        return decoder_desc_->getMetadata(decoder_, code_stream, metadata, metadata_count);
    }

    bool ImageDecoder::canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
                                 const nvimgcodecDecodeParams_t* params, nvimgcodecProcessingStatus_t* status, int thread_idx) const {
        assert(decoder_desc_ && decoder_desc_->canDecode);
        assert(code_stream);
        assert(image);
        try {
            *status = decoder_desc_->canDecode(decoder_, image, code_stream, params, thread_idx);
            return *status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return false;
        }
    }

    bool ImageDecoder::decode(
        nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
        assert(decoder_desc_);
        // Defense in depth: a batch-only decoder (e.g. nvjpeg lossless) leaves the single-sample
        // `decode` callback NULL. It must never be dispatched here; fail gracefully rather than
        // call a NULL function pointer (release builds strip the assert above).
        if (decoder_desc_->decode == nullptr)
            return false;
        try {
            auto ret = decoder_desc_->decode(decoder_, image, code_stream, params, thread_idx);
            return ret == NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return false;
        }
    }

    bool ImageDecoder::hasDecodeBatch() const {
        assert(decoder_desc_);
        return decoder_desc_->decodeBatch != nullptr;
    }

    int ImageDecoder::getMiniBatchSize() const {
        assert(decoder_desc_);
        try {
            int batch_size = -1;
            if (decoder_desc_->getMiniBatchSize) {
                if (decoder_desc_->getMiniBatchSize(decoder_, &batch_size) == NVIMGCODEC_STATUS_SUCCESS)
                    return batch_size;
            }
            return -1;
        } catch (std::exception& e) {
            return -1;
        }
    }

    bool ImageDecoder::decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
                                   const nvimgcodecDecodeParams_t* params, int batch_size, int thread_idx) {
        assert(decoder_desc_);
        if (decoder_desc_->decodeBatch == nullptr)
            return false;
        try {
            return decoder_desc_->decodeBatch(decoder_, images, code_streams, batch_size, params, thread_idx);
        } catch (std::exception& e) {
            return false;
        }
    }

    const char* ImageDecoder::decoderId() const {
        return decoder_desc_->id;
    }

} // namespace nvimgcodec
