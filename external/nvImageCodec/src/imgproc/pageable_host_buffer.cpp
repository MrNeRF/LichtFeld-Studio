/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "imgproc/pageable_host_buffer.h"
#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h>  // _aligned_malloc / _aligned_free
#endif

namespace nvimgcodec {

namespace {

// Portable aligned host allocation. POSIX uses posix_memalign; Windows uses
// _aligned_malloc (which must be paired with _aligned_free).
void* aligned_host_alloc(size_t size, size_t alignment)
{
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, size) != 0) {
        return nullptr;
    }
    return p;
#endif
}

void aligned_host_free(void* p) noexcept
{
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}

} // namespace

void PageableHostBuffer::resize(size_t new_size)
{
    if (new_size <= capacity) {
        size = new_size;
    } else {
        alloc_impl(new_size);
    }
}

void PageableHostBuffer::alloc_impl(size_t new_size)
{
    // 64-byte alignment matches typical SIMD/cache-line expectations.
    constexpr size_t kAlignment = 64;

    // Allocate the deleter strategy before the buffer so that a throwing
    // make_shared cannot leak a successful aligned_host_alloc.
    auto strategy = std::make_shared<DeleterStrategy>();
    strategy->device_id = -1;
    strategy->fn = [](void* ptr) { aligned_host_free(ptr); };

    const size_t padded = (new_size + kAlignment - 1) & ~(kAlignment - 1);
    const size_t actual = padded ? padded : kAlignment;

    void* p = aligned_host_alloc(actual, kAlignment);
    if (p == nullptr) {
        throw std::bad_alloc();
    }

    data = p;
    size = new_size;
    capacity = actual;
    deleter_strategy = strategy;
    d_ptr.reset(data, [strategy](void* ptr) {
        if (strategy && strategy->fn) {
            strategy->fn(ptr);
        }
    });
}

} // end nvimgcodec
