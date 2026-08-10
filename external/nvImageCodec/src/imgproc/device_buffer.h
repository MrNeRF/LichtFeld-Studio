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

#include <functional>
#include <memory>
#include <nvimgcodec.h>

namespace nvimgcodec {

    enum class DeleterBackend {
        Unknown,
        DeviceCustom,
        DeviceAsync,
        DeviceSync,
        PinnedCustom,
        PinnedHost
    };

    // Holds the currently-active cleanup callback for a buffer's underlying
    // storage. The callback is installed at allocation time and may be swapped
    // later (e.g., by `detach_from_stream()`) without invalidating the
    // `shared_ptr` deleter that references this strategy.
    //
    // By convention, `fn` must not throw: it is invoked from a `shared_ptr`
    // deleter and therefore from destructors. All implementations in this file
    // swallow CUDA/allocator errors internally.
    struct DeleterStrategy {
        std::function<void(void*)> fn;
        // Device id captured at allocation time. Used by `detach_from_stream()`
        // so the detached deleter can run a `DeviceGuard` without needing to
        // re-query the (potentially already-destroyed) user stream.
        int device_id = -1;
        DeleterBackend backend = DeleterBackend::Unknown;
    };

    struct DeviceBuffer {
        explicit DeviceBuffer(nvimgcodecDeviceAllocator_t* device_allocator = nullptr);
        void resize(size_t new_size, cudaStream_t new_stream);
        void alloc_impl(size_t new_size, cudaStream_t new_stream);

        // Rebind the buffer's deleter so that it no longer references the stream
        // this buffer was allocated on. After this call returns, the caller may
        // destroy the original `cuda_stream` at any time without affecting correct
        // cleanup of this buffer; the detached deleter uses plain `cudaFree` under
        // a `DeviceGuard` (or the user-supplied allocator with the legacy default
        // stream). Safe and idempotent to call regardless of which allocation path
        // produced the current deleter.
        void detach_from_stream() noexcept;

        nvimgcodecDeviceAllocator_t* allocator;
        std::shared_ptr<void> d_ptr;
        void* data = nullptr;
        size_t size = 0;
        size_t capacity = 0;
        cudaStream_t stream = 0;
        // Referenced by `d_ptr`'s deleter; mutating `deleter_strategy->fn`
        // changes the behavior of the already-installed `shared_ptr` deleter.
        std::shared_ptr<DeleterStrategy> deleter_strategy;
    };

} // namespace nvimgcodec
