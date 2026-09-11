/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_debug.hpp"
#include "core/tensor_trace.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

namespace lfs::core::debug {

    std::string TensorValidation::to_string() const {
        if (is_valid()) {
            return std::format("valid (min={:.6f}, max={:.6f}, mean={:.6f})",
                               min_val, max_val, mean_val);
        }
        return std::format("INVALID (nan={}, inf={}, min={:.6f}, max={:.6f})",
                           nan_count, inf_count, min_val, max_val);
    }

    std::string TensorDiff::to_string() const {
        if (!shapes_match)
            return "shape mismatch";
        if (!dtypes_match)
            return "dtype mismatch";
        return std::format("max_diff={:.6e}, mean_diff={:.6e}, diff_count={}/{}",
                           max_abs_diff, mean_abs_diff, num_different, total_elements);
    }

    std::string TensorStats::to_string() const {
        return std::format("shape={}, dtype={}, device={}, min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}",
                           shape.str(), dtype_name(dtype), is_cuda ? "cuda" : "cpu", min, max, mean, std);
    }

    TensorOpTracer& TensorOpTracer::instance() {
        static TensorOpTracer inst;
        return inst;
    }

    TensorValidation validate_tensor_cpu(const Tensor& tensor) {
        TensorValidation result;

        if (tensor.is_empty() || tensor.dtype() != DataType::Float32) {
            return result;
        }

        // Copy to CPU if needed
        const Tensor cpu_tensor = tensor.device() == Device::CUDA ? tensor.cpu() : tensor;
        const float* data = cpu_tensor.ptr<float>();
        const size_t n = cpu_tensor.numel();

        result.min_val = std::numeric_limits<float>::max();
        result.max_val = std::numeric_limits<float>::lowest();
        double sum = 0.0;

        for (size_t i = 0; i < n; ++i) {
            const float val = data[i];
            if (std::isnan(val)) {
                result.has_nan = true;
                ++result.nan_count;
            } else if (std::isinf(val)) {
                result.has_inf = true;
                ++result.inf_count;
            } else {
                result.min_val = std::min(result.min_val, val);
                result.max_val = std::max(result.max_val, val);
                sum += val;
            }
        }

        const size_t valid_count = n - result.nan_count - result.inf_count;
        result.mean_val = valid_count > 0 ? static_cast<float>(sum / valid_count) : 0.0f;

        return result;
    }

    // GPU validation is implemented in tensor_debug.cu
    // Forward declaration - implemented in CUDA file
    extern TensorValidation validate_tensor_gpu_impl(const float* data, size_t n);

    TensorValidation validate_tensor_gpu(const Tensor& tensor) {
        if (tensor.is_empty() || tensor.dtype() != DataType::Float32 || tensor.device() != Device::CUDA) {
            return validate_tensor_cpu(tensor);
        }
        return validate_tensor_gpu_impl(tensor.ptr<float>(), tensor.numel());
    }

    TensorDiff diff_tensors(const Tensor& expected, const Tensor& actual, float tolerance) {
        TensorDiff result;
        result.total_elements = expected.numel();

        if (expected.shape() != actual.shape()) {
            result.shapes_match = false;
            return result;
        }

        if (expected.dtype() != actual.dtype()) {
            result.dtypes_match = false;
            return result;
        }

        if (expected.is_empty()) {
            return result;
        }

        // Copy to CPU for comparison
        const Tensor exp_cpu = expected.device() == Device::CUDA ? expected.cpu() : expected;
        const Tensor act_cpu = actual.device() == Device::CUDA ? actual.cpu() : actual;

        if (expected.dtype() == DataType::Float32) {
            const float* exp_data = exp_cpu.ptr<float>();
            const float* act_data = act_cpu.ptr<float>();
            const size_t n = expected.numel();

            double sum_abs_diff = 0.0;

            for (size_t i = 0; i < n; ++i) {
                const float diff = std::abs(exp_data[i] - act_data[i]);
                sum_abs_diff += diff;

                if (diff > result.max_abs_diff) {
                    result.max_abs_diff = diff;
                }

                if (diff > tolerance) {
                    ++result.num_different;
                }

                if (exp_data[i] != 0.0f) {
                    const float rel_diff = diff / std::abs(exp_data[i]);
                    result.max_rel_diff = std::max(result.max_rel_diff, rel_diff);
                }
            }

            result.mean_abs_diff = static_cast<float>(sum_abs_diff / n);
        }

        return result;
    }

    TensorStats get_tensor_stats(const Tensor& tensor) {
        TensorStats stats;
        stats.shape = tensor.shape();
        stats.dtype = tensor.dtype();
        stats.numel = tensor.numel();
        stats.is_cuda = tensor.device() == Device::CUDA;

        if (tensor.is_empty() || tensor.dtype() != DataType::Float32) {
            return stats;
        }

        const Tensor cpu_tensor = tensor.device() == Device::CUDA ? tensor.cpu() : tensor;
        const float* data = cpu_tensor.ptr<float>();
        const size_t n = cpu_tensor.numel();

        stats.min = std::numeric_limits<float>::max();
        stats.max = std::numeric_limits<float>::lowest();
        double sum = 0.0;
        double sum_sq = 0.0;

        for (size_t i = 0; i < n; ++i) {
            const float val = data[i];
            if (!std::isnan(val) && !std::isinf(val)) {
                stats.min = std::min(stats.min, val);
                stats.max = std::max(stats.max, val);
                sum += val;
                sum_sq += val * val;
            }
        }

        stats.mean = static_cast<float>(sum / n);
        const double variance = (sum_sq / n) - (stats.mean * stats.mean);
        stats.std = static_cast<float>(std::sqrt(std::max(0.0, variance)));

        return stats;
    }

} // namespace lfs::core::debug
