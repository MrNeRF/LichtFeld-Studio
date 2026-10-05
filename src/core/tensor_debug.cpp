/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_debug.hpp"
#include "core/tensor_backend.hpp"
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

    bool TensorDiff::is_close(const float atol, const float rtol) const {
        LFS_ASSERT_MSG(std::isfinite(atol) && std::isfinite(rtol) && atol >= 0 && rtol >= 0,
                       "tensor diff tolerances must be finite and non-negative");
        if (!shapes_match || !dtypes_match)
            return false;
        if (!difference_.is_valid())
            return max_abs_diff <= atol || max_rel_diff <= rtol;
        const auto close = difference_.isfinite().logical_and(difference_.le(scale_.mul(rtol).add(atol)));
        return close.count_nonzero() == total_elements;
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
                           shape.str(), dtype_name(dtype), backend ? gpu_backend_name(*backend) : "cpu", min, max, mean, std);
    }

    namespace {
        Tensor floating_values(const Tensor& tensor) {
            LFS_ASSERT_MSG(tensor.is_valid(), "tensor inspection requires a valid tensor");
            return tensor.to(DataType::Float32).contiguous();
        }

        TensorValidation validate_values(const Tensor& tensor) {
            TensorValidation result;
            if (tensor.is_empty())
                return result;
            const auto values = floating_values(tensor);
            const auto finite = values.isfinite();
            result.nan_count = values.isnan().count_nonzero();
            result.inf_count = values.isinf().count_nonzero();
            result.has_nan = result.nan_count != 0;
            result.has_inf = result.inf_count != 0;
            const auto count = values.numel() - result.nan_count - result.inf_count;
            if (count != 0) {
                const auto invalid = finite.logical_not();
                result.min_val = values.masked_fill(invalid, std::numeric_limits<float>::infinity()).min().item<float>();
                result.max_val = values.masked_fill(invalid, -std::numeric_limits<float>::infinity()).max().item<float>();
                result.mean_val = values.masked_fill(invalid, 0.0f).div(static_cast<float>(count)).sum().item<float>();
            }
            return result;
        }
    } // namespace

    TensorValidation validate_tensor_cpu(const Tensor& tensor) {
        return validate_values(tensor);
    }

    TensorValidation validate_tensor_gpu(const Tensor& tensor) {
        return validate_values(tensor);
    }

    TensorDiff diff_tensors(const Tensor& expected, const Tensor& actual, float tolerance) {
        LFS_ASSERT_MSG(std::isfinite(tolerance) && tolerance >= 0.0f,
                       "tensor diff tolerance must be finite and non-negative");
        TensorDiff result;
        result.total_elements = expected.numel();
        result.shapes_match = expected.shape() == actual.shape();
        result.dtypes_match = expected.dtype() == actual.dtype();
        if (!result.shapes_match || !result.dtypes_match || expected.is_empty())
            return result;
        LFS_ASSERT_MSG(expected.device() == actual.device() && gpu_backend_of(expected) == gpu_backend_of(actual),
                       "tensor diff operands must share a device and backend");
        const auto a = floating_values(expected);
        const auto b = floating_values(actual);
        const auto equal = a.eq(b);
        const auto raw_difference = a.sub(b).abs();
        const auto difference = raw_difference.masked_fill(raw_difference.isfinite().logical_not(),
                                                           std::numeric_limits<float>::infinity())
                                    .masked_fill(equal, 0.0f);
        result.difference_ = difference;
        result.scale_ = a.abs().masked_fill(a.isfinite().logical_not(), 0.0f);
        result.max_abs_diff = difference.max().item<float>();
        result.mean_abs_diff = difference.div(static_cast<float>(result.total_elements)).sum().item<float>();
        result.num_different = difference.gt(tolerance).count_nonzero();
        const auto relative = difference.div(result.scale_.clamp_min(std::numeric_limits<float>::min()));
        result.max_rel_diff = relative.masked_fill(equal, 0.0f).max().item<float>();
        return result;
    }

    TensorStats get_tensor_stats(const Tensor& tensor) {
        TensorStats stats;
        stats.shape = tensor.shape();
        stats.dtype = tensor.dtype();
        stats.numel = tensor.numel();
        stats.backend = gpu_backend_of(tensor);
        stats.is_cuda = stats.backend == GpuBackend::CUDA;
        if (tensor.is_empty())
            return stats;
        const auto validation = validate_values(tensor);
        stats.min = validation.min_val;
        stats.max = validation.max_val;
        stats.mean = validation.mean_val;
        const auto count = stats.numel - validation.nan_count - validation.inf_count;
        if (count != 0) {
            const auto values = floating_values(tensor);
            const float scale = std::max(std::abs(stats.min), std::abs(stats.max));
            if (scale > 0.0f) {
                // Split the scaling so GPU fast math never forms a subnormal
                // reciprocal when the largest value is near Float32's limit.
                int exponent = 0;
                std::frexp(scale, &exponent);
                const int first = exponent / 2;
                const auto scaled = values.mul(std::ldexp(1.0f, -first)).mul(std::ldexp(1.0f, -(exponent - first)));
                const auto centered = scaled.sub(std::ldexp(stats.mean, -exponent)).masked_fill(values.isfinite().logical_not(), 0.0f);
                const float deviation = centered.square().div(static_cast<float>(count)).sum().sqrt().item<float>();
                stats.std = static_cast<float>(std::min(static_cast<double>(scale), std::ldexp(static_cast<double>(deviation), exponent)));
            }
        }
        return stats;
    }

} // namespace lfs::core::debug
