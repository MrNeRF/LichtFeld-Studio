/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/tensor_trace.hpp"
#include "internal/tensor_impl.hpp"

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
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for matrix multiplication");
            return Tensor();
        }

        if (shape_.rank() != 2 || other.shape_.rank() != 2) {
            LOG_ERROR("Matrix multiplication requires 2D tensors");
            return Tensor();
        }

        if (shape_[1] != other.shape_[0]) {
            LOG_ERROR("Matrix dimensions don't match: {}x{} @ {}x{}",
                      shape_[0], shape_[1], other.shape_[0], other.shape_[1]);
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Matrix multiplication requires tensors on same device");
            return Tensor();
        }

        const size_t m = shape_[0];
        const size_t k = shape_[1];
        const size_t n = other.shape_[1];

        // GPU: transfer to CPU, compute, transfer back
        if (device_ == Device::CUDA) {
            return cpu().contiguous().mm(other.cpu().contiguous()).cuda();
        }

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        auto result = empty({m, n}, Device::CPU, dtype_);
        cpu_matmul(a.ptr<float>(), b.ptr<float>(), result.ptr<float>(), m, k, n);
        return result;
    }

    Tensor Tensor::bmm(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for bmm");
            return Tensor();
        }

        if (shape_.rank() != 3 || other.shape_.rank() != 3) {
            LOG_ERROR("BMM requires 3D tensors");
            return Tensor();
        }

        if (shape_[0] != other.shape_[0]) {
            LOG_ERROR("Batch dimensions must match for bmm");
            return Tensor();
        }

        if (shape_[2] != other.shape_[1]) {
            LOG_ERROR("Matrix dimensions incompatible for bmm: {}x{} @ {}x{}",
                      shape_[1], shape_[2], other.shape_[1], other.shape_[2]);
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("BMM requires tensors on same device");
            return Tensor();
        }

        const size_t batch_size = shape_[0];
        const size_t m = shape_[1];
        const size_t k = shape_[2];
        const size_t n = other.shape_[2];

        // GPU: transfer to CPU, compute, transfer back
        if (device_ == Device::CUDA) {
            return cpu().contiguous().bmm(other.cpu().contiguous()).cuda();
        }

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

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

        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for matmul");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Matmul requires tensors on same device");
            return Tensor();
        }

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        // Vector dot product
        if (a.shape_.rank() == 1 && b.shape_.rank() == 1) {
            if (a.shape_[0] != b.shape_[0]) {
                LOG_ERROR("Vector dimensions don't match for dot product");
                return Tensor();
            }
            return a.dot(b);
        }

        // Vector-matrix: [k] @ [k, n] -> [n]
        if (a.shape_.rank() == 1 && b.shape_.rank() == 2) {
            if (a.shape_[0] != b.shape_[0]) {
                LOG_ERROR("Dimension mismatch for vector-matrix multiplication");
                return Tensor();
            }
            return a.unsqueeze(0).mm(b).squeeze(0);
        }

        // Matrix-vector: [m, k] @ [k] -> [m]
        if (a.shape_.rank() == 2 && b.shape_.rank() == 1) {
            if (a.shape_[1] != b.shape_[0]) {
                LOG_ERROR("Dimension mismatch for matrix-vector multiplication");
                return Tensor();
            }
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
            if (a.shape_[1] != b.shape_[1]) {
                LOG_ERROR("Dimension mismatch for 2D @ 3D matmul");
                return Tensor();
            }
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
            if (a.shape_[2] != b.shape_[0]) {
                LOG_ERROR("Dimension mismatch for 3D @ 2D matmul");
                return Tensor();
            }
            const size_t batch = a.shape_[0];
            const size_t k = b.shape_[0];
            const size_t n = b.shape_[1];
            auto expanded = b.unsqueeze(0).expand({static_cast<int>(batch),
                                                   static_cast<int>(k),
                                                   static_cast<int>(n)});
            return a.bmm(expanded);
        }

        LOG_ERROR("MatMul not implemented for {}D @ {}D", a.shape_.rank(), b.shape_.rank());
        return Tensor();
    }

    Tensor Tensor::dot(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for dot product");
            return Tensor();
        }

        if (shape_.rank() != 1 || other.shape_.rank() != 1) {
            LOG_ERROR("Dot product requires 1D tensors");
            return Tensor();
        }

        if (shape_[0] != other.shape_[0]) {
            LOG_ERROR("Vector dimensions don't match for dot product");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Dot product requires tensors on same device");
            return Tensor();
        }

        // GPU: transfer to CPU, compute, transfer back
        if (device_ == Device::CUDA) {
            return cpu().contiguous().dot(other.cpu().contiguous()).cuda();
        }

        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();
        const size_t n = a.shape_[0];

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
