/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"
#include "tensor_impl.hpp"
#include <fstream>
#include <limits>
#include <string_view>

namespace lfs::core {

    constexpr uint32_t TENSOR_FILE_MAGIC = 0x4C465354;
    constexpr uint32_t TENSOR_FILE_VERSION = 1;
    constexpr uint16_t MAX_SERIALIZED_TENSOR_RANK = 8;
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

        inline void read_exact(std::istream& is,
                               void* const destination,
                               const std::size_t bytes,
                               const std::string_view field) {
            if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                throw std::runtime_error("Serialized " + std::string(field) + " exceeds stream limits");
            }
            if (bytes == 0)
                return;
            is.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
            if (!is) {
                throw std::runtime_error("Truncated serialized " + std::string(field));
            }
        }

        inline void require_remaining_bytes(std::istream& is,
                                            const uint64_t required,
                                            const std::string_view field) {
            const auto current = is.tellg();
            if (current == std::streampos(-1))
                return;

            is.seekg(0, std::ios::end);
            const auto end = is.tellg();
            is.seekg(current);
            if (!is || end == std::streampos(-1) || end < current ||
                static_cast<uint64_t>(end - current) < required) {
                throw std::runtime_error("Truncated serialized " + std::string(field));
            }
        }

    } // namespace serialization_detail

    inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        if (!tensor.is_valid()) {
            throw std::runtime_error("Cannot serialize invalid tensor");
        }

        const TensorFileHeader header{
            TENSOR_FILE_MAGIC,
            TENSOR_FILE_VERSION,
            static_cast<uint8_t>(tensor.dtype()),
            static_cast<uint8_t>(tensor.device()),
            static_cast<uint16_t>(tensor.ndim()),
            tensor.numel()};
        os.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const size_t dim : tensor.shape().dims()) {
            const uint64_t d = dim;
            os.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }

        Tensor src = tensor.device() == Device::CUDA ? tensor.cpu() : tensor;
        if (!src.is_contiguous()) {
            src = src.contiguous();
        }
        os.write(reinterpret_cast<const char*>(src.data_ptr()), src.bytes());

        if (!os) {
            throw std::runtime_error("Failed to write tensor");
        }
        return os;
    }

    inline std::istream& operator>>(std::istream& is, Tensor& tensor) {
        TensorFileHeader header{};
        serialization_detail::read_exact(is, &header, sizeof(header), "tensor header");

        if (header.magic != TENSOR_FILE_MAGIC) {
            throw std::runtime_error("Invalid tensor file: wrong magic number");
        }
        if (header.version != TENSOR_FILE_VERSION) {
            throw std::runtime_error("Unsupported tensor file version");
        }
        if (header.rank > MAX_SERIALIZED_TENSOR_RANK) {
            throw std::runtime_error("Invalid tensor file: rank exceeds supported maximum");
        }
        if (header.dtype > static_cast<uint8_t>(DataType::Bool)) {
            throw std::runtime_error("Invalid tensor file: unsupported dtype");
        }
        if (header.device > static_cast<uint8_t>(Device::CUDA)) {
            throw std::runtime_error("Invalid tensor file: unsupported device");
        }

        std::vector<size_t> dims(header.rank);
        uint64_t checked_numel = 1;
        for (uint16_t i = 0; i < header.rank; ++i) {
            uint64_t d = 0;
            serialization_detail::read_exact(is, &d, sizeof(d), "tensor dimension");
            if (d > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("Invalid tensor file: dimension exceeds platform size");
            }
            if (d != 0 && checked_numel > std::numeric_limits<uint64_t>::max() / d) {
                throw std::runtime_error("Invalid tensor file: shape element count overflows");
            }
            checked_numel *= d;
            dims[i] = static_cast<size_t>(d);
        }

        const DataType dtype = static_cast<DataType>(header.dtype);
        if (checked_numel != header.numel) {
            throw std::runtime_error("Shape elements mismatch");
        }
        const auto item_size = dtype_size(dtype);
        if (item_size == 0 ||
            header.numel > std::numeric_limits<uint64_t>::max() / item_size) {
            throw std::runtime_error("Invalid tensor file: byte size overflows");
        }
        const uint64_t payload_bytes = header.numel * item_size;
        if (payload_bytes > MAX_SERIALIZED_TENSOR_BYTES) {
            throw std::runtime_error("Invalid tensor file: payload exceeds byte budget");
        }
        serialization_detail::require_remaining_bytes(is, payload_bytes, "tensor payload");

        const TensorShape shape(dims);
        Tensor loaded = Tensor::empty(shape, Device::CPU, dtype);
        serialization_detail::read_exact(is, loaded.data_ptr(), loaded.bytes(), "tensor payload");
        tensor = std::move(loaded);
        return is;
    }

    inline void save_tensor(const Tensor& tensor, const std::string& filename) {
        std::ofstream file;
        if (!open_file_for_write(utf8_to_path(filename), std::ios::binary, file)) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        file << tensor;
    }

    inline Tensor load_tensor(const std::string& filename) {
        std::ifstream file;
        if (!open_file_for_read(utf8_to_path(filename), std::ios::binary, file)) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        Tensor tensor;
        file >> tensor;
        return tensor;
    }

} // namespace lfs::core
