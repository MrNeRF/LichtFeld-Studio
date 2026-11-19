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
#include <nvimgcodec.h>
#include <memory>
#include <string>

namespace nvimgcodec {

class CodecRegistry;
class ICodec;

class ICodeStream
{
  public:
    virtual ~ICodeStream() = default;
    virtual void parseFromFile(const std::string& file_name) = 0;
    virtual void parseFromMem(const unsigned char* data, size_t size) = 0;
    virtual void setOutputToFile(const char* file_name) = 0;
    virtual void setOutputToHostMem(void* ctx, nvimgcodecResizeBufferFunc_t get_buffer_func) = 0;
    virtual nvimgcodecStatus_t getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info) = 0;
    virtual nvimgcodecStatus_t getImageInfo(nvimgcodecImageInfo_t* image_info) = 0;
    virtual nvimgcodecStatus_t setImageInfo(const nvimgcodecImageInfo_t* image_info) = 0;
    virtual std::string getCodecName() const = 0;
    virtual ICodec* getCodec() const = 0;
    virtual nvimgcodecIoStreamDesc_t* getInputStreamDesc() = 0;
    virtual nvimgcodecCodeStreamDesc_t* getCodeStreamDesc() = 0;
};
} // namespace nvimgcodec