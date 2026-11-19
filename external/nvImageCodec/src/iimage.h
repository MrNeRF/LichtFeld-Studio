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

namespace nvimgcodec {
class ProcessingResultsPromise;

class IImage
{
  public:
    virtual ~IImage() = default;
    virtual int getIndex()  = 0;
    virtual void setIndex(int index)  = 0;
    virtual void getImageInfo(nvimgcodecImageInfo_t* image_info)  = 0;
    virtual void setImageInfo(const nvimgcodecImageInfo_t* image_info)  = 0;
    virtual nvimgcodecImageDesc_t* getImageDesc()  = 0;
    virtual std::shared_ptr<ProcessingResultsPromise> getPromise() = 0;
    virtual void setPromise(std::shared_ptr<ProcessingResultsPromise> promise) = 0;
};
} // namespace nvimgcodec
