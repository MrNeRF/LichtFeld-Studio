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
#include "image_decoder_factory.h"

#include <cassert>
#include "code_stream.h"
#include "image.h"
#include "image_decoder.h"

namespace nvimgcodec {

ImageDecoderFactory::ImageDecoderFactory(const nvimgcodecDecoderDesc_t* desc)
    : decoder_desc_(desc)
{
}

std::string ImageDecoderFactory::getDecoderId() const
{
    return decoder_desc_->id;
}

std::string ImageDecoderFactory::getCodecName() const
{
    return decoder_desc_->codec;
}

nvimgcodecBackendKind_t ImageDecoderFactory::getBackendKind() const
{
    return decoder_desc_->backend_kind;
}

const nvimgcodecDecoderDesc_t* ImageDecoderFactory::getDecoderDesc() const
{
    return decoder_desc_;
}

std::unique_ptr<IImageDecoder> ImageDecoderFactory::createDecoder(
    const nvimgcodecExecutionParams_t* exec_params, const char* options) const
{
    return std::make_unique<ImageDecoder>(decoder_desc_, exec_params, options);
}

} // namespace nvimgcodec