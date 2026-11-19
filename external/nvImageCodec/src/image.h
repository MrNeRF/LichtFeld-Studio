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

#include <memory>
#include <nvimgcodec.h>
#include "iimage.h"

namespace nvimgcodec {

class Image : public IImage
{
  public:
    explicit Image();
    ~Image() override;
    void setIndex(int index) override;
    int getIndex() override;
    void setImageInfo(const nvimgcodecImageInfo_t* image_info) override;
    void getImageInfo(nvimgcodecImageInfo_t* image_info) override;
    std::shared_ptr<ProcessingResultsPromise> getPromise() override;
    void setPromise(std::shared_ptr<ProcessingResultsPromise> promise) override;
    nvimgcodecImageDesc_t* getImageDesc() override;
  private:
    nvimgcodecStatus_t imageReady(nvimgcodecProcessingStatus_t processing_status);

    static nvimgcodecStatus_t static_get_image_info(void* instance, nvimgcodecImageInfo_t* result);
    static nvimgcodecStatus_t static_image_ready(
        void* instance, nvimgcodecProcessingStatus_t processing_status);
    int index_;
    nvimgcodecImageInfo_t image_info_;
    nvimgcodecImageDesc_t image_desc_;
    std::shared_ptr<ProcessingResultsPromise> promise_;
};
} // namespace nvimgcodec
