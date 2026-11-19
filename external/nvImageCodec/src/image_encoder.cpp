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
#include "image_encoder.h"
#include <cassert>
#include "encode_state_batch.h"
#include "icode_stream.h"
#include "iimage.h"
#include "imgproc/exception.h"

namespace nvimgcodec {

ImageEncoder::ImageEncoder(const nvimgcodecEncoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : encoder_desc_(desc)
{
    auto ret = encoder_desc_->create(encoder_desc_->instance, &encoder_, exec_params, options);
    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        encoder_ = nullptr;
        throw Exception(INTERNAL_ERROR, std::string("Could not create encoder ") + encoder_desc_->id + ", status: " + std::to_string(ret));
    }
}

ImageEncoder::~ImageEncoder()
{
    if (encoder_)
        encoder_desc_->destroy(encoder_);
}

nvimgcodecBackendKind_t ImageEncoder::getBackendKind() const
{
    return encoder_desc_->backend_kind;
}

std::unique_ptr<IEncodeState> ImageEncoder::createEncodeStateBatch() const
{
    return std::make_unique<EncodeStateBatch>();
}

bool ImageEncoder::canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
    const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* status, int thread_idx) const
{
    assert(encoder_desc_ && encoder_desc_->canEncode);
    assert(code_stream);
    assert(image);
    try {
        *status = encoder_desc_->canEncode(encoder_, code_stream, image, params, thread_idx);
        return *status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    } catch (std::exception& e) {
        *status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        return false;
    }
}

bool ImageEncoder::encode(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image,
    const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* status, int thread_idx)
{
    assert(encoder_desc_ && encoder_desc_->encode);
    try {
        auto ret = encoder_desc_->encode(encoder_, code_stream, image, params, thread_idx);
        return ret == NVIMGCODEC_STATUS_SUCCESS;
    } catch (std::exception& e) {
        return false;
    }
}

const char* ImageEncoder::encoderId() const
{
    return encoder_desc_->id;
}

} // namespace nvimgcodec
