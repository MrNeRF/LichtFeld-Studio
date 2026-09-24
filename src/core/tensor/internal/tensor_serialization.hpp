/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/tensor_serialization.hpp"
#include "tensor_impl.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

namespace lfs::core {

    namespace serialization_detail {
        LFS_CORE_API void require_remaining_bytes(std::istream& is,
                                                  uint64_t required,
                                                  std::string_view field);
        // Consume one serialized tensor without allocating host or device storage.
        // Uses a seek over the payload so framed .licht streams do not decompress it.
        LFS_LOCAL_SYMBOL void skip_serialized_tensor(std::istream& is);

        struct TensorLoadTiming {
            double alloc_ms = 0.0;
            double read_ms = 0.0;
        };

        class TensorLoadTimingScope {
        public:
            LFS_CORE_API explicit TensorLoadTimingScope(TensorLoadTiming& timing) noexcept;
            TensorLoadTimingScope(const TensorLoadTimingScope&) = delete;
            TensorLoadTimingScope& operator=(const TensorLoadTimingScope&) = delete;
            LFS_CORE_API ~TensorLoadTimingScope();

        private:
            TensorLoadTiming* previous_ = nullptr;
        };

        // Public operator>> always pins. Splat deserialize uses this sibling so
        // host tensors at or above 256 MiB skip cudaHostAlloc / the pinned cache.
        LFS_CORE_API void read_serialized_tensor(std::istream& is, Tensor& tensor,
                                                 bool use_pinned);
        LFS_CORE_API void read_serialized_tensor_pageable_if_large(std::istream& is,
                                                                   Tensor& tensor);
        // When `is` is backed by a contiguous memory streambuf, copies the
        // payload with cudaMemcpyAsync into a device tensor on `stream`.
        // Otherwise falls back to read_serialized_tensor_pageable_if_large.
        LFS_LOCAL_SYMBOL void read_serialized_tensor_device_from_span_or_host(
            std::istream& is, Tensor& tensor, cudaStream_t stream);
    } // namespace serialization_detail

} // namespace lfs::core
