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
    } // namespace serialization_detail

    LFS_CORE_API std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    LFS_CORE_API std::istream& operator>>(std::istream& is, Tensor& tensor);

    LFS_CORE_API void save_tensor(const Tensor& tensor, const std::string& filename);
    LFS_CORE_API Tensor load_tensor(const std::string& filename);

} // namespace lfs::core
