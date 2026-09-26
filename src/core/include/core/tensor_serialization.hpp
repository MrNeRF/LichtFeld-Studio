/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include <cstdint>
#include <iosfwd>
#include <memory>
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

    LFS_CORE_API std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    LFS_CORE_API std::istream& operator>>(std::istream& is, Tensor& tensor);

    LFS_CORE_API void save_tensor(const Tensor& tensor, const std::string& filename);
    LFS_CORE_API Tensor load_tensor(const std::string& filename);

    namespace serialization_detail {
        LFS_CORE_API void read_exact(std::istream& stream, void* destination,
                                     std::size_t bytes, std::string_view field);
    }

    class LFS_CORE_API TensorArchiveReader {
    public:
        struct Timing {
            double allocation_ms = 0.0;
            double read_ms = 0.0;
        };

        explicit TensorArchiveReader(std::istream& stream);
        ~TensorArchiveReader();
        TensorArchiveReader(const TensorArchiveReader&) = delete;
        TensorArchiveReader& operator=(const TensorArchiveReader&) = delete;

        void read_exact(void* destination, std::size_t bytes, std::string_view field);
        void require_remaining_bytes(std::uint64_t bytes, std::string_view field);
        void skip_tensor();
        void read_tensor(Tensor& tensor);
        void read_host_tensor(Tensor& tensor, bool pin_memory);
        [[nodiscard]] Timing timing() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::core
