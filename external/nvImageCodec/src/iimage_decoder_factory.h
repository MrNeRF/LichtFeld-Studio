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
#include <memory>
#include <string>

namespace nvimgcodec {

class IImageDecoder;

class IImageDecoderFactory
{
  public:
    virtual ~IImageDecoderFactory() = default;
    virtual std::string getDecoderId() const = 0;
    virtual std::string getCodecName() const = 0;
    virtual nvimgcodecBackendKind_t getBackendKind() const = 0;
    virtual const nvimgcodecDecoderDesc_t* getDecoderDesc() const = 0;
    // Throws if the decoder cannot be created.
    virtual std::unique_ptr<IImageDecoder> createDecoder(
        const nvimgcodecExecutionParams_t* exec_params, const char* options) const = 0;
};
} // namespace nvimgcodec