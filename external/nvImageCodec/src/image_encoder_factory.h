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
#include "iimage_encoder_factory.h"

namespace nvimgcodec {

class IImageEncoder;

class ImageEncoderFactory : public IImageEncoderFactory
{
  public:
    explicit ImageEncoderFactory(const nvimgcodecEncoderDesc_t* desc);
    std::string getEncoderId() const override;
    std::string getCodecName() const override;
    nvimgcodecBackendKind_t getBackendKind() const override;
    const nvimgcodecEncoderDesc_t* getEncoderDesc() const override;
    std::unique_ptr<IImageEncoder> createEncoder(
        const nvimgcodecExecutionParams_t* exec_params, const char* options) const override;

  private:
    const nvimgcodecEncoderDesc_t* encoder_desc_;
};
} // namespace nvimgcodec