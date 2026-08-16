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

#include "imgproc/pinned_buffer.h"
#include "imgproc/device_guard.h"
#include "imgproc/exception.h"
#include "imgproc/stream_device.h"
#include <cassert>

namespace nvimgcodec {

    namespace {

        struct StrategyDeleter {
            std::shared_ptr<DeleterStrategy> strategy;

            void operator()(void* ptr) const noexcept {
                if (strategy && strategy->fn) {
                    strategy->fn(ptr);
                }
            }
        };

        std::shared_ptr<void> MakeBufferOwner(void* data, const std::shared_ptr<DeleterStrategy>& strategy) {
            // The unique_ptr `rollback` provides exception-safety while constructing the shared_ptr:
            // if the shared_ptr's control-block allocation throws, ~rollback frees `data` via the
            // strategy. Once `owner` is constructed, both wrappers point at the same `data`, so we
            // must detach `rollback` to prevent a double-free. Coverity flags release() as a leak
            // because it cannot see the alias already held by `owner`.
            std::unique_ptr<void, StrategyDeleter> rollback(data, StrategyDeleter{strategy});
            std::shared_ptr<void> owner(rollback.get(), StrategyDeleter{strategy});
            // coverity[leaked_storage : FALSE]
            rollback.release();
            return owner;
        }

    } // namespace

    PinnedBuffer::PinnedBuffer(nvimgcodecPinnedAllocator_t* pinned_allocator)
        : allocator(pinned_allocator) {
    }

    void PinnedBuffer::resize(size_t new_size, cudaStream_t new_stream) {
        if (new_size <= capacity && stream == new_stream) {
            stream = new_stream;
            size = new_size;
        } else {
            alloc_impl(new_size, new_stream);
        }
    }

    void PinnedBuffer::alloc_impl(size_t new_size, cudaStream_t new_stream) {
        auto strategy = std::make_shared<DeleterStrategy>();

        if (allocator && allocator->pinned_malloc) {
            if (!allocator->pinned_free) {
                FatalError(INVALID_PARAMETER, "pinned allocator missing free callback");
            }
            const int dev_id = get_stream_device_id(new_stream);
            void* new_data = nullptr;
            nvimgcodec::DeviceGuard device_guard(dev_id);
            const int alloc_status = allocator->pinned_malloc(allocator->pinned_ctx, &new_data, new_size, new_stream);
            if (alloc_status != 0 || (new_size && !new_data)) {
                FatalError(ALLOCATION_ERROR, "pinned allocator failed");
            }
            strategy->device_id = dev_id;
            strategy->backend = DeleterBackend::PinnedCustom;
            strategy->fn = [alloc = this->allocator, new_size, new_stream, dev_id](void* ptr) {
                try {
                    nvimgcodec::DeviceGuard g(dev_id);
                    alloc->pinned_free(alloc->pinned_ctx, ptr, new_size, new_stream);
                } catch (...) {
                    // Swallow: deleters must not throw.
                }
                // User allocator may invoke CUDA internally; clear any sticky
                // error it left so later CUDA calls are not misattributed.
                (void)cudaGetLastError();
            };
            auto new_owner = MakeBufferOwner(new_data, strategy);
            data = new_data;
            size = new_size;
            capacity = new_size;
            stream = new_stream;
            deleter_strategy = strategy;
            d_ptr = std::move(new_owner);
        } else {
            int dev_id = nvimgcodec::get_stream_device_id(new_stream);
            void* new_data = nullptr;
            nvimgcodec::DeviceGuard device_guard(dev_id);
            CHECK_CUDA(cudaMallocHost(&new_data, new_size));
            strategy->device_id = dev_id;
            strategy->backend = DeleterBackend::PinnedHost;
            strategy->fn = [](void* ptr) {
                cudaError_t e = cudaFreeHost(ptr);
                if (e != cudaSuccess) {
                    (void)cudaGetLastError();
                }
            };
            auto new_owner = MakeBufferOwner(new_data, strategy);
            data = new_data;
            size = new_size;
            capacity = new_size;
            stream = new_stream;
            deleter_strategy = strategy;
            d_ptr = std::move(new_owner);
        }
    }

    void PinnedBuffer::detach_from_stream() noexcept {
        if (!deleter_strategy) {
            return;
        }
        const int dev_id = deleter_strategy->device_id;

        if (deleter_strategy->backend == DeleterBackend::PinnedCustom && allocator && allocator->pinned_free) {
            auto alloc = this->allocator;
            const size_t cached_size = size;
            deleter_strategy->backend = DeleterBackend::PinnedCustom;
            deleter_strategy->fn = [alloc, cached_size, dev_id](void* ptr) {
                try {
                    nvimgcodec::DeviceGuard g(dev_id);
                    // Pass the legacy default stream so we no longer reference
                    // the user's original stream.
                    alloc->pinned_free(alloc->pinned_ctx, ptr, cached_size, cudaStream_t{0});
                } catch (...) {
                    // Swallow: deleters must not throw.
                }
                // User allocator or DeviceGuard may invoke CUDA internally; clear
                // any sticky error so later CUDA calls are not misattributed.
                (void)cudaGetLastError();
            };
        }
        // For the non-allocator path the deleter already uses `cudaFreeHost`,
        // which does not reference any stream; no rebind needed.
    }

} // namespace nvimgcodec
