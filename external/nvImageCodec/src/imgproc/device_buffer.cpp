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

#include "imgproc/device_buffer.h"
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

    DeviceBuffer::DeviceBuffer(nvimgcodecDeviceAllocator_t* device_allocator)
        : allocator(device_allocator) {
    }

    void DeviceBuffer::resize(size_t new_size, cudaStream_t new_stream) {
        if (new_size <= capacity) {
            if (new_stream != stream) {
                const int old_dev_id = deleter_strategy ? deleter_strategy->device_id : get_stream_device_id(stream);
                const int new_dev_id = get_stream_device_id(new_stream);
                if (old_dev_id != new_dev_id) {
                    alloc_impl(new_size, new_stream);
                    return;
                }

                {
                    // Concrete stream handles carry their context, but default/legacy/per-thread
                    // stream sentinels are resolved against the current device.
                    nvimgcodec::DeviceGuard device_guard(old_dev_id);
                    CHECK_CUDA(cudaStreamSynchronize(stream));
                }

                if (deleter_strategy) {
                    deleter_strategy->device_id = new_dev_id;
                    if (deleter_strategy->backend == DeleterBackend::DeviceCustom && allocator && allocator->device_free) {
                        auto alloc = this->allocator;
                        const size_t cached_capacity = capacity;
                        deleter_strategy->fn = [alloc, cached_capacity, new_stream, new_dev_id](void* ptr) {
                            try {
                                nvimgcodec::DeviceGuard g(new_dev_id);
                                alloc->device_free(alloc->device_ctx, ptr, cached_capacity, new_stream);
                            } catch (...) {
                                // Swallow: deleters must not throw.
                            }
                            // User allocator may invoke CUDA internally; clear any sticky
                            // error it left so later CUDA calls are not misattributed.
                            (void)cudaGetLastError();
                        };
                    } else if (deleter_strategy->backend == DeleterBackend::DeviceAsync) {
                        deleter_strategy->fn = [new_stream, new_dev_id](void* ptr) {
                            try {
                                nvimgcodec::DeviceGuard g(new_dev_id);
                                cudaError_t e = cudaFreeAsync(ptr, new_stream);
                                if (e != cudaSuccess) {
                                    (void)cudaGetLastError();
                                }
                            } catch (...) {
                                // Swallow: deleters must not throw.
                                (void)cudaGetLastError();
                            }
                        };
                    } else if (deleter_strategy->backend == DeleterBackend::DeviceSync) {
                        deleter_strategy->fn = [new_dev_id](void* ptr) {
                            try {
                                nvimgcodec::DeviceGuard g(new_dev_id);
                                cudaError_t e = cudaFree(ptr);
                                if (e != cudaSuccess) {
                                    (void)cudaGetLastError();
                                }
                            } catch (...) {
                                // Swallow: deleters must not throw.
                                (void)cudaGetLastError();
                            }
                        };
                    }
                }
                stream = new_stream;
            }
            size = new_size;
            return;
        }
        alloc_impl(new_size, new_stream);
    }

    void DeviceBuffer::alloc_impl(size_t new_size, cudaStream_t new_stream) {
        auto strategy = std::make_shared<DeleterStrategy>();
        const int dev_id = get_stream_device_id(new_stream);

        if (allocator && allocator->device_malloc) {
            if (!allocator->device_free) {
                FatalError(INVALID_PARAMETER, "device allocator missing free callback");
            }
            void* new_data = nullptr;
            nvimgcodec::DeviceGuard device_guard(dev_id);
            const int alloc_status = allocator->device_malloc(allocator->device_ctx, &new_data, new_size, new_stream);
            if (alloc_status != 0 || (new_size && !new_data)) {
                FatalError(ALLOCATION_ERROR, "device allocator failed");
            }
            strategy->device_id = dev_id;
            strategy->backend = DeleterBackend::DeviceCustom;
            strategy->fn = [alloc = this->allocator, new_size, new_stream, dev_id](void* ptr) {
                try {
                    nvimgcodec::DeviceGuard g(dev_id);
                    alloc->device_free(alloc->device_ctx, ptr, new_size, new_stream);
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
        } else if (nvimgcodec::can_use_async_mem_ops(new_stream)) {
            void* new_data = nullptr;
            nvimgcodec::DeviceGuard device_guard(dev_id);
            CHECK_CUDA(cudaMallocAsync(&new_data, new_size, new_stream));
            strategy->device_id = dev_id;
            strategy->backend = DeleterBackend::DeviceAsync;
            strategy->fn = [new_stream, dev_id](void* ptr) {
                try {
                    nvimgcodec::DeviceGuard g(dev_id);
                    cudaError_t e = cudaFreeAsync(ptr, new_stream);
                    if (e != cudaSuccess) {
                        (void)cudaGetLastError();
                    }
                } catch (...) {
                    // Swallow: deleters must not throw.
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
        } else {
            void* new_data = nullptr;
            nvimgcodec::DeviceGuard device_guard(dev_id);
            CHECK_CUDA(cudaMalloc(&new_data, new_size));
            strategy->device_id = dev_id;
            strategy->backend = DeleterBackend::DeviceSync;
            strategy->fn = [dev_id](void* ptr) {
                try {
                    nvimgcodec::DeviceGuard g(dev_id);
                    cudaError_t e = cudaFree(ptr);
                    if (e != cudaSuccess) {
                        (void)cudaGetLastError();
                    }
                } catch (...) {
                    // Swallow: deleters must not throw.
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

    void DeviceBuffer::detach_from_stream() noexcept {
        if (!deleter_strategy) {
            return;
        }
        const int dev_id = deleter_strategy->device_id;

        if (deleter_strategy->backend == DeleterBackend::DeviceCustom && allocator && allocator->device_free) {
            auto alloc = this->allocator;
            const size_t cached_size = size;
            deleter_strategy->backend = DeleterBackend::DeviceCustom;
            deleter_strategy->fn = [alloc, cached_size, dev_id](void* ptr) {
                try {
                    nvimgcodec::DeviceGuard g(dev_id);
                    // Use the legacy default stream (0) so we no longer reference
                    // the user's original stream. The custom allocator contract
                    // allows any valid stream; the default stream synchronizes.
                    alloc->device_free(alloc->device_ctx, ptr, cached_size, cudaStream_t{0});
                } catch (...) {
                    // Swallow: deleters must not throw.
                }
                // User allocator or DeviceGuard may invoke CUDA internally; clear
                // any sticky error so later CUDA calls are not misattributed.
                (void)cudaGetLastError();
            };
        } else {
            deleter_strategy->backend = DeleterBackend::DeviceSync;
            deleter_strategy->fn = [dev_id](void* ptr) {
                try {
                    nvimgcodec::DeviceGuard g(dev_id);
                    cudaError_t e = cudaFree(ptr);
                    if (e != cudaSuccess) {
                        (void)cudaGetLastError();
                    }
                } catch (...) {
                    // Swallow: deleters must not throw.
                    (void)cudaGetLastError();
                }
            };
        }
    }

} // namespace nvimgcodec
