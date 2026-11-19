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

#include "processing_results.h"
#include <memory>
#include <nvimgcodec.h>
#include <string>
#include <vector>

namespace nvimgcodec {

    class IEncodeState;
    class IImage;
    class ICodeStream;

    class IImageEncoder {
    public:
        virtual ~IImageEncoder() = default;
        virtual nvimgcodecBackendKind_t getBackendKind() const = 0;
        virtual std::unique_ptr<IEncodeState> createEncodeStateBatch() const = 0;
        virtual bool canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                               const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* status, int thread_idx) const = 0;
        virtual bool encode(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image,
                            const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* status, int thread_idx) = 0;
        virtual const char* encoderId() const = 0;
    };

} // namespace nvimgcodec