/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tensor_impl.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

namespace lfs::core {

    constexpr uint32_t TENSOR_FILE_MAGIC = 0x4C465354;
    constexpr uint32_t TENSOR_FILE_VERSION = 1;
    constexpr uint64_t MAX_SERIALIZED_TENSOR_BYTES = 64ULL * 1024ULL * 1024ULL * 1024ULL;

    struct TensorFileHeader {
        uint32_t magic;
        uint32_t version;
        uint8_t dtype;
        uint8_t device;
        uint16_t rank;
        uint64_t numel;
    };

    namespace serialization_detail {
        LFS_CORE_API void read_exact(std::istream& is,
                                     void* destination,
                                     std::size_t bytes,
                                     std::string_view field);
        LFS_CORE_API void require_remaining_bytes(std::istream& is,
                                                  uint64_t required,
                                                  std::string_view field);
        // Consume one serialized tensor without allocating host or device storage.
        // Uses a seek over the payload so framed .licht streams do not decompress it.
        LFS_CORE_API void skip_serialized_tensor(std::istream& is);

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
        LFS_CORE_API void read_serialized_tensor_device_from_span_or_host(
            std::istream& is, Tensor& tensor, cudaStream_t stream);
    } // namespace serialization_detail

    LFS_CORE_API std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    LFS_CORE_API std::istream& operator>>(std::istream& is, Tensor& tensor);

    LFS_CORE_API void save_tensor(const Tensor& tensor, const std::string& filename);
    LFS_CORE_API Tensor load_tensor(const std::string& filename);

} // namespace lfs::core
