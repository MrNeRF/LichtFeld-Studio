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
#include "image.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include "idecode_state.h"
#include "iencode_state.h"
#include "processing_results.h"

namespace nvimgcodec {

Image::Image()
    : index_(0)
    , image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr}
    , image_desc_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_DESC, sizeof(nvimgcodecImageDesc_t), nullptr, this, Image::static_get_image_info,
          Image::static_image_ready}
    , promise_(nullptr)
{
}

Image::~Image()
{
}

void Image::setIndex(int index)
{
    index_ = index;
}

int Image::getIndex()
{
    return index_;
}

void Image::setImageInfo(const nvimgcodecImageInfo_t* image_info)
{
    image_info_ = *image_info;
}

void Image::getImageInfo(nvimgcodecImageInfo_t* image_info)
{
    *image_info = image_info_;
}

nvimgcodecImageDesc_t* Image::getImageDesc()
{
    return &image_desc_;
}

void Image::setPromise(std::shared_ptr<ProcessingResultsPromise> promise)
{
    promise_ = promise;
}

std::shared_ptr<ProcessingResultsPromise> Image::getPromise()
{
    return promise_;
}

nvimgcodecStatus_t Image::imageReady(nvimgcodecProcessingStatus_t processing_status)
{
    promise_->set(index_, {processing_status, {}});
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t Image::static_get_image_info(void* instance, nvimgcodecImageInfo_t* result)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getImageInfo(result);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t Image::static_image_ready(
    void* instance, nvimgcodecProcessingStatus_t processing_status)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->imageReady(processing_status);
    return NVIMGCODEC_STATUS_SUCCESS;
}


} // namespace nvimgcodec
