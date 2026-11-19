/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "imgproc/device_buffer.h"  // for DeleterStrategy

namespace nvimgcodec {

struct PinnedBuffer
{
    explicit PinnedBuffer(nvimgcodecPinnedAllocator_t* pinned_allocator = nullptr);
    void resize(size_t new_size, cudaStream_t new_stream);
    void alloc_impl(size_t new_size, cudaStream_t new_stream);

    // Rebind the buffer's deleter so that it no longer references the stream
    // this buffer was allocated on. After this call returns, the caller may
    // destroy the original `cuda_stream` at any time without affecting correct
    // cleanup of this buffer. Safe and idempotent.
    void detach_from_stream() noexcept;

    nvimgcodecPinnedAllocator_t* allocator;
    std::shared_ptr<void> d_ptr;
    void* data = nullptr;
    size_t capacity = 0;
    size_t size = 0;
    cudaStream_t stream = 0;
    std::shared_ptr<DeleterStrategy> deleter_strategy;
};

} // end nvimgcodec
