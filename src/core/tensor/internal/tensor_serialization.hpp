/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"
#include "tensor_impl.hpp"
#include <fstream>

namespace lfs::core {

    constexpr uint32_t TENSOR_FILE_MAGIC = 0x4C465354;
    constexpr uint32_t TENSOR_FILE_VERSION = 1;

    struct TensorFileHeader {
        uint32_t magic;
        uint32_t version;
        uint8_t dtype;
        uint8_t device;
        uint16_t rank;
        uint64_t numel;
    };

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
        os.write(reinterpret_cast<const char*>(src.ptr<uint8_t>()), src.bytes());

        if (!os) {
            throw std::runtime_error("Failed to write tensor");
        }
        return os;
    }

    inline std::istream& operator>>(std::istream& is, Tensor& tensor) {
        TensorFileHeader header;
        is.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (header.magic != TENSOR_FILE_MAGIC) {
            throw std::runtime_error("Invalid tensor file: wrong magic number");
        }
        if (header.version != TENSOR_FILE_VERSION) {
            throw std::runtime_error("Unsupported tensor file version");
        }

        std::vector<size_t> dims(header.rank);
        for (uint16_t i = 0; i < header.rank; ++i) {
            uint64_t d;
            is.read(reinterpret_cast<char*>(&d), sizeof(d));
            dims[i] = static_cast<size_t>(d);
        }

        const TensorShape shape(dims);
        const DataType dtype = static_cast<DataType>(header.dtype);

        if (shape.elements() != header.numel) {
            throw std::runtime_error("Shape elements mismatch");
        }

        tensor = Tensor::empty(shape, Device::CPU, dtype);
        is.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.bytes());

        if (!is) {
            throw std::runtime_error("Failed to read tensor");
        }
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
