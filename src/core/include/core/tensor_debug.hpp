/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include <cmath>
#include <string>

namespace lfs::core::debug {

    // Validation result for NaN/Inf checking
    struct TensorValidation {
        bool has_nan = false;
        bool has_inf = false;
        size_t nan_count = 0;
        size_t inf_count = 0;
        float min_val = 0.0f;
        float max_val = 0.0f;
        float mean_val = 0.0f;

        [[nodiscard]] bool is_valid() const { return !has_nan && !has_inf; }

        [[nodiscard]] LFS_CORE_API std::string to_string() const;
    };

    LFS_CORE_API TensorValidation validate_tensor_cpu(const Tensor& tensor);
    LFS_CORE_API TensorValidation validate_tensor_gpu(const Tensor& tensor);

    // Auto-select validation based on device and size
    inline TensorValidation validate_tensor(const Tensor& tensor) {
        constexpr size_t GPU_THRESHOLD = 10000;
        if (tensor.is_empty())
            return {};
        if (tensor.device() == Device::CUDA && tensor.numel() > GPU_THRESHOLD) {
            return validate_tensor_gpu(tensor);
        }
        return validate_tensor_cpu(tensor);
    }

    inline void log_tensor_validation(const Tensor& tensor, const char* name,
                                      const char* file, const int line) {
        const auto result = validate_tensor(tensor);
        if (!result.is_valid()) {
            LOG_WARN("Tensor '{}' at {}:{} - {}", name, file, line, result.to_string());
        }
    }

    // Tensor comparison result
    struct TensorDiff {
        bool shapes_match = true;
        bool dtypes_match = true;
        float max_abs_diff = 0.0f;
        float mean_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        size_t num_different = 0;
        size_t total_elements = 0;

        [[nodiscard]] bool is_close(const float atol = 1e-5f, const float rtol = 1e-4f) const {
            return shapes_match && dtypes_match && max_abs_diff <= atol + rtol * max_abs_diff;
        }

        [[nodiscard]] LFS_CORE_API std::string to_string() const;
    };

    LFS_CORE_API TensorDiff diff_tensors(const Tensor& expected, const Tensor& actual, float tolerance = 1e-5f);

    inline void log_tensor_diff(const Tensor& expected, const Tensor& actual,
                                const char* name, const float tolerance = 1e-5f) {
        const auto diff = diff_tensors(expected, actual, tolerance);
        if (!diff.is_close(tolerance)) {
            LOG_WARN("Tensor diff '{}': {}", name, diff.to_string());
        }
    }

    // Tensor statistics
    struct TensorStats {
        float min = 0.0f;
        float max = 0.0f;
        float mean = 0.0f;
        float std = 0.0f;
        size_t numel = 0;
        TensorShape shape;
        DataType dtype = DataType::Float32;
        bool is_cuda = false;

        [[nodiscard]] LFS_CORE_API std::string to_string() const;
    };

    LFS_CORE_API TensorStats get_tensor_stats(const Tensor& tensor);

    inline void log_tensor_info(const Tensor& tensor, const char* name) {
        LOG_DEBUG("Tensor '{}': {}", name, get_tensor_stats(tensor).to_string());
    }

} // namespace lfs::core::debug

#ifdef TENSOR_VALIDATION_ENABLED
#define VALIDATE_TENSOR(tensor) \
    lfs::core::debug::log_tensor_validation(tensor, #tensor, __FILE__, __LINE__)
#define VALIDATE_TENSOR_NAMED(tensor, name) \
    lfs::core::debug::log_tensor_validation(tensor, name, __FILE__, __LINE__)
#else
#define VALIDATE_TENSOR(tensor)             ((void)0)
#define VALIDATE_TENSOR_NAMED(tensor, name) ((void)0)
#endif

#define INSPECT_TENSOR(tensor)                  lfs::core::debug::log_tensor_info(tensor, #tensor)
#define DIFF_TENSORS(expected, actual)          lfs::core::debug::log_tensor_diff(expected, actual, #expected " vs " #actual)
#define DIFF_TENSORS_TOL(expected, actual, tol) lfs::core::debug::log_tensor_diff(expected, actual, #expected " vs " #actual, tol)
