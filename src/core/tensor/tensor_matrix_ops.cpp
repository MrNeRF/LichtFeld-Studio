/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/tensor_trace.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"

namespace lfs::core {

    namespace {

        // CPU matrix multiply: C = A @ B
        // A: [m, k], B: [k, n], C: [m, n]
        void cpu_matmul(const float* a, const float* b, float* c,
                        size_t m, size_t k, size_t n) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; ++l) {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }

    } // namespace

    Tensor Tensor::mm(const Tensor& other) const {
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       std::format("mm requires valid input tensors "
                                   "(left={}, right={})",
                                   str(), other.str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 && other.dtype_ == DataType::Float32,
                       std::format("mm requires Float32 inputs "
                                   "(left_dtype={}({}), right_dtype={}({}), "
                                   "left_shape={}, right_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(other.dtype_), static_cast<int>(other.dtype_),
                                   shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(shape_.rank() == 2 && other.shape_.rank() == 2,
                       std::format("mm requires rank-2 inputs "
                                   "(left_rank={}, right_rank={}, left_shape={}, right_shape={})",
                                   shape_.rank(), other.shape_.rank(),
                                   shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(shape_[1] == other.shape_[0],
                       std::format("mm dimension mismatch: {}x{} @ {}x{}",
                                   shape_[0], shape_[1], other.shape_[0], other.shape_[1]));
        LFS_ASSERT_MSG(device_ == other.device_,
                       std::format("mm inputs must be on the same device "
                                   "(left_device={}, right_device={})",
                                   device_name(device_), device_name(other.device_)));

        const size_t m = shape_[0];
        const size_t k = shape_[1];
        const size_t n = other.shape_[1];

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        // GPU: use tiled CUDA sgemm kernel
        if (device_ == Device::CUDA) {
            auto result = empty({m, n}, Device::CUDA, dtype_);
            tensor_ops::launch_sgemm(a.ptr<float>(), b.ptr<float>(), result.ptr<float>(),
                                     m, n, k, result.stream());
            return result;
        }

        auto result = empty({m, n}, Device::CPU, dtype_);
        cpu_matmul(a.ptr<float>(), b.ptr<float>(), result.ptr<float>(), m, k, n);
        return result;
    }

    Tensor Tensor::bmm(const Tensor& other) const {
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       std::format("bmm requires valid input tensors "
                                   "(left={}, right={})",
                                   str(), other.str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 && other.dtype_ == DataType::Float32,
                       std::format("bmm requires Float32 inputs "
                                   "(left_dtype={}({}), right_dtype={}({}), "
                                   "left_shape={}, right_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(other.dtype_), static_cast<int>(other.dtype_),
                                   shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(shape_.rank() == 3 && other.shape_.rank() == 3,
                       std::format("bmm requires rank-3 inputs "
                                   "(left_rank={}, right_rank={}, left_shape={}, right_shape={})",
                                   shape_.rank(), other.shape_.rank(),
                                   shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(shape_[0] == other.shape_[0],
                       std::format("bmm batch dimensions must match "
                                   "(left_batch={}, right_batch={}, "
                                   "left_shape={}, right_shape={})",
                                   shape_[0], other.shape_[0], shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(shape_[2] == other.shape_[1],
                       std::format("bmm dimension mismatch: {}x{} @ {}x{}",
                                   shape_[1], shape_[2], other.shape_[1], other.shape_[2]));
        LFS_ASSERT_MSG(device_ == other.device_,
                       std::format("bmm inputs must be on the same device "
                                   "(left_device={}, right_device={})",
                                   device_name(device_), device_name(other.device_)));

        const size_t batch_size = shape_[0];
        const size_t m = shape_[1];
        const size_t k = shape_[2];
        const size_t n = other.shape_[2];

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        // GPU: use tiled CUDA batched sgemm kernel
        if (device_ == Device::CUDA) {
            auto result = empty({batch_size, m, n}, Device::CUDA, dtype_);
            tensor_ops::launch_sgemm_batched(a.ptr<float>(), b.ptr<float>(), result.ptr<float>(),
                                             batch_size, m, n, k, result.stream());
            return result;
        }

        auto result = empty({batch_size, m, n}, Device::CPU, dtype_);

        const float* a_data = a.ptr<float>();
        const float* b_data = b.ptr<float>();
        float* c_data = result.ptr<float>();

        const size_t a_stride = m * k;
        const size_t b_stride = k * n;
        const size_t c_stride = m * n;

        for (size_t batch = 0; batch < batch_size; ++batch) {
            cpu_matmul(a_data + batch * a_stride,
                       b_data + batch * b_stride,
                       c_data + batch * c_stride,
                       m, k, n);
        }

        return result;
    }

    Tensor Tensor::matmul(const Tensor& other) const {
        debug::OpTraceGuard trace("matmul", *this, other);

        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       std::format("matmul requires valid input tensors "
                                   "(left={}, right={})",
                                   str(), other.str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 && other.dtype_ == DataType::Float32,
                       std::format("matmul requires Float32 inputs "
                                   "(left_dtype={}({}), right_dtype={}({}), "
                                   "left_shape={}, right_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(other.dtype_), static_cast<int>(other.dtype_),
                                   shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(device_ == other.device_,
                       std::format("matmul inputs must be on the same device "
                                   "(left_device={}, right_device={})",
                                   device_name(device_), device_name(other.device_)));

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        // Vector dot product
        if (a.shape_.rank() == 1 && b.shape_.rank() == 1) {
            LFS_ASSERT_MSG(a.shape_[0] == b.shape_[0],
                           std::format("matmul vector dimensions must match "
                                       "(left_size={}, right_size={}, left_shape={}, right_shape={})",
                                       a.shape_[0], b.shape_[0], a.shape_.str(), b.shape_.str()));
            return a.dot(b);
        }

        // Vector-matrix: [k] @ [k, n] -> [n]
        if (a.shape_.rank() == 1 && b.shape_.rank() == 2) {
            LFS_ASSERT_MSG(a.shape_[0] == b.shape_[0],
                           std::format("matmul vector-matrix inner dimensions must match "
                                       "(vector_size={}, matrix_rows={}, "
                                       "left_shape={}, right_shape={})",
                                       a.shape_[0], b.shape_[0], a.shape_.str(), b.shape_.str()));
            return a.unsqueeze(0).mm(b).squeeze(0);
        }

        // Matrix-vector: [m, k] @ [k] -> [m]
        if (a.shape_.rank() == 2 && b.shape_.rank() == 1) {
            LFS_ASSERT_MSG(a.shape_[1] == b.shape_[0],
                           std::format("matmul matrix-vector inner dimensions must match "
                                       "(matrix_columns={}, vector_size={}, "
                                       "left_shape={}, right_shape={})",
                                       a.shape_[1], b.shape_[0], a.shape_.str(), b.shape_.str()));
            return a.mm(b.unsqueeze(1)).squeeze(1);
        }

        // Matrix-matrix: [m, k] @ [k, n] -> [m, n]
        if (a.shape_.rank() == 2 && b.shape_.rank() == 2) {
            return a.mm(b);
        }

        // Batch matrix multiply: [B, m, k] @ [B, k, n] -> [B, m, n]
        if (a.shape_.rank() == 3 && b.shape_.rank() == 3) {
            return a.bmm(b);
        }

        // 2D @ 3D: broadcast [m, k] @ [B, k, n] -> [B, m, n]
        if (a.shape_.rank() == 2 && b.shape_.rank() == 3) {
            LFS_ASSERT_MSG(a.shape_[1] == b.shape_[1],
                           std::format("matmul 2D @ 3D inner dimensions must match "
                                       "(left_columns={}, right_inner={}, "
                                       "left_shape={}, right_shape={})",
                                       a.shape_[1], b.shape_[1], a.shape_.str(), b.shape_.str()));
            const size_t batch = b.shape_[0];
            const size_t m = a.shape_[0];
            const size_t k = a.shape_[1];
            auto expanded = a.unsqueeze(0).expand({static_cast<int>(batch),
                                                   static_cast<int>(m),
                                                   static_cast<int>(k)});
            return expanded.bmm(b);
        }

        // 3D @ 2D: broadcast [B, m, k] @ [k, n] -> [B, m, n]
        if (a.shape_.rank() == 3 && b.shape_.rank() == 2) {
            LFS_ASSERT_MSG(a.shape_[2] == b.shape_[0],
                           std::format("matmul 3D @ 2D inner dimensions must match "
                                       "(left_inner={}, right_rows={}, "
                                       "left_shape={}, right_shape={})",
                                       a.shape_[2], b.shape_[0], a.shape_.str(), b.shape_.str()));
            const size_t batch = a.shape_[0];
            const size_t k = b.shape_[0];
            const size_t n = b.shape_[1];
            auto expanded = b.unsqueeze(0).expand({static_cast<int>(batch),
                                                   static_cast<int>(k),
                                                   static_cast<int>(n)});
            return a.bmm(expanded);
        }

        LFS_ASSERT_MSG(false,
                       std::format("matmul is unsupported for rank {} @ rank {}",
                                   a.shape_.rank(), b.shape_.rank()));
    }

    Tensor Tensor::dot(const Tensor& other) const {
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       std::format("dot requires valid input tensors "
                                   "(left={}, right={})",
                                   str(), other.str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 && other.dtype_ == DataType::Float32,
                       std::format("dot requires Float32 inputs "
                                   "(left_dtype={}({}), right_dtype={}({}), "
                                   "left_shape={}, right_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(other.dtype_), static_cast<int>(other.dtype_),
                                   shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(shape_.rank() == 1 && other.shape_.rank() == 1,
                       std::format("dot requires rank-1 inputs "
                                   "(left_rank={}, right_rank={}, left_shape={}, right_shape={})",
                                   shape_.rank(), other.shape_.rank(),
                                   shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(shape_[0] == other.shape_[0],
                       std::format("dot vector dimensions must match "
                                   "(left_size={}, right_size={}, left_shape={}, right_shape={})",
                                   shape_[0], other.shape_[0], shape_.str(), other.shape_.str()));
        LFS_ASSERT_MSG(device_ == other.device_,
                       std::format("dot inputs must be on the same device "
                                   "(left_device={}, right_device={})",
                                   device_name(device_), device_name(other.device_)));

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();
        const size_t n = a.shape_[0];

        // GPU: Use optimized CUDA kernel
        if (device_ == Device::CUDA) {
            auto result = empty({}, Device::CUDA, dtype_);
            tensor_ops::launch_dot_product(
                a.ptr<float>(),
                b.ptr<float>(),
                result.ptr<float>(),
                n,
                result.stream());
            return result;
        }

        // CPU: Simple loop
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += a.ptr<float>()[i] * b.ptr<float>()[i];
        }

        auto result = empty({1}, Device::CPU, dtype_);
        *result.ptr<float>() = sum;

        // Return as scalar (rank-0 view)
        Tensor scalar(result.data_ptr(), TensorShape(std::vector<size_t>{}), Device::CPU, dtype_);
        scalar.data_owner_ = result.data_owner_;
        scalar.is_view_ = true;
        return scalar;
    }

} // namespace lfs::core
