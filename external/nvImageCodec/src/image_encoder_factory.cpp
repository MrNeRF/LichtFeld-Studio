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
#include "image_encoder_factory.h"
#include <cassert>
#include "code_stream.h"
#include "image.h"
#include "image_encoder.h"
namespace nvimgcodec {

ImageEncoderFactory::ImageEncoderFactory(const nvimgcodecEncoderDesc_t* desc)
    : encoder_desc_(desc)
{
}

std::string ImageEncoderFactory::getCodecName() const
{
    return encoder_desc_->codec;
}

std::string ImageEncoderFactory::getEncoderId() const
{
    return encoder_desc_->id;
}

nvimgcodecBackendKind_t ImageEncoderFactory::getBackendKind() const
{
    return encoder_desc_->backend_kind;
}

const nvimgcodecEncoderDesc_t* ImageEncoderFactory::getEncoderDesc() const
{
    return encoder_desc_;
}

std::unique_ptr<IImageEncoder> ImageEncoderFactory::createEncoder(
    const nvimgcodecExecutionParams_t* exec_params, const char* options) const
{
    return std::make_unique<ImageEncoder>(encoder_desc_, exec_params, options);
}
} // namespace nvimgcodec