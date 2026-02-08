/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Forward declarations for Tensor library (use when only Tensor*/& needed)

#include <cstddef>
#include <cstdint>

namespace lfs::core {

    class Tensor;

    enum class Device : uint8_t {
        CPU = 0,
        CUDA = 1
    };

    enum class DataType : uint8_t {
        Float32 = 0,
        Float16 = 1,
        Int32 = 2,
        Int64 = 3,
        UInt8 = 4,
        Bool = 5
    };

    constexpr size_t dtype_size(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
        case DataType::Bool: return 1;
        default: return 0;
        }
    }

    inline const char* dtype_name(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        case DataType::Bool: return "bool";
        default: return "unknown";
        }
    }

    constexpr bool is_bool_like(DataType dt) {
        return dt == DataType::Bool || dt == DataType::UInt8;
    }

    inline const char* device_name(Device device) {
        return device == Device::CPU ? "cpu" : "cuda";
    }

} // namespace lfs::core
