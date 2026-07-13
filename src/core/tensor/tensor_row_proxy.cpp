/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <string_view>
#include <vector>

namespace lfs::core {
    namespace {
        void cuda_copy_async_sync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind,
                                  cudaStream_t stream, const char* context) {
            cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
            if (err == cudaSuccess) {
                err = cudaStreamSynchronize(stream);
            }
            LFS_ASSERT_MSG(err == cudaSuccess,
                           std::string("CUDA copy failed in ") + context + ": " +
                               cudaGetErrorString(err));
        }

        void assert_proxy_tensor(const Tensor* tensor,
                                 const size_t row_index,
                                 const std::string_view operation) {
            LFS_ASSERT_MSG(tensor != nullptr && tensor->is_valid(),
                           std::string(operation) + " requires a valid tensor");
            LFS_ASSERT_MSG(tensor->ndim() > 0,
                           std::string(operation) +
                               " requires a tensor with at least one dimension");
            LFS_ASSERT_MSG(row_index < tensor->shape()[0],
                           std::string(operation) + " row index is out of bounds");
        }
    } // namespace

    void TensorRowProxy::flush_cuda_staging() const {
        if (!cuda_staging_pending_write_) {
            return;
        }
        if (!tensor_ || !tensor_->is_valid() || tensor_->device() != Device::CUDA) {
            cuda_staging_pending_write_ = false;
            return;
        }
        if (tensor_->dtype() != DataType::Float32) {
            throw std::runtime_error("TensorRowProxy CUDA staging writeback only supports Float32 tensors");
        }

        cuda_copy_async_sync(
            tensor_->ptr<float>() + cuda_staging_linear_idx_,
            &cuda_staging_,
            sizeof(float),
            cudaMemcpyHostToDevice,
            tensor_->stream(),
            "TensorRowProxy::flush_cuda_staging");
        cuda_staging_pending_write_ = false;
    }

    TensorRowProxy::~TensorRowProxy() {
        try {
            flush_cuda_staging();
        } catch (...) {
            // Destructors must not throw.
        }
    }

    // ============= TensorRowProxy 2D Access =============

    float& TensorRowProxy::operator[](size_t col_index) {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy::operator[]");
        LFS_ASSERT_MSG(tensor_->shape().rank() >= 2,
                       "TensorRowProxy::operator[] requires a tensor with rank at least 2");
        LFS_ASSERT_MSG(col_index < tensor_->shape()[1],
                       "TensorRowProxy column index is out of bounds");
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       "TensorRowProxy mutable element access requires Float32");

        // Use actual strides for proper indexing on non-contiguous tensors
        size_t linear_idx = row_index_ * tensor_->stride(0) + col_index * tensor_->stride(1);

        if (tensor_->device() != Device::CPU) {
            // Commit any previously staged element before staging another one.
            flush_cuda_staging();
            cuda_copy_async_sync(
                &cuda_staging_,
                tensor_->ptr<float>() + linear_idx,
                sizeof(float),
                cudaMemcpyDeviceToHost,
                tensor_->stream(),
                "TensorRowProxy::operator[]");
            cuda_staging_linear_idx_ = linear_idx;
            cuda_staging_pending_write_ = true;
            return cuda_staging_;
        }

        return tensor_->ptr<float>()[linear_idx];
    }

    float TensorRowProxy::operator[](size_t col_index) const {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy::operator[] const");
        flush_cuda_staging();
        LFS_ASSERT_MSG(tensor_->shape().rank() >= 2,
                       "TensorRowProxy::operator[] requires a tensor with rank at least 2");
        LFS_ASSERT_MSG(col_index < tensor_->shape()[1],
                       "TensorRowProxy column index is out of bounds");
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       "TensorRowProxy const element access requires Float32");

        // Use actual strides for proper indexing on non-contiguous tensors
        size_t linear_idx = row_index_ * tensor_->stride(0) + col_index * tensor_->stride(1);

        if (tensor_->device() == Device::CUDA) {
            float value = 0.0f;
            cuda_copy_async_sync(
                &value,
                tensor_->ptr<float>() + linear_idx,
                sizeof(float),
                cudaMemcpyDeviceToHost,
                tensor_->stream(),
                "TensorRowProxy::operator[] const");
            return value;
        } else {
            return tensor_->ptr<float>()[linear_idx];
        }
    }

    // ============= TensorRowProxy 1D Access =============

    float TensorRowProxy::item() const {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy::item()");
        flush_cuda_staging();

        // Handle 2D tensors with shape [N, 1] (like nonzero() output)
        if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
            Tensor row_tensor = static_cast<Tensor>(*this);
            return row_tensor.item();
        }

        // Standard 1D case
        LFS_ASSERT_MSG(tensor_->shape().rank() == 1,
                       "TensorRowProxy::item() requires a 1D or [N,1] tensor");
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       "TensorRowProxy::item() currently supports only Float32");

        // Use stride for proper indexing on non-contiguous 1D tensors
        size_t linear_idx = row_index_ * tensor_->stride(0);

        if (tensor_->device() == Device::CUDA) {
            float value = 0.0f;
            cuda_copy_async_sync(
                &value,
                tensor_->ptr<float>() + linear_idx,
                sizeof(float),
                cudaMemcpyDeviceToHost,
                tensor_->stream(),
                "TensorRowProxy::item()");
            return value;
        } else {
            return tensor_->ptr<float>()[linear_idx];
        }
    }

    TensorRowProxy::operator float() const {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy float conversion");
        flush_cuda_staging();

        if (tensor_->shape().rank() == 1) {
            return item();
        } else if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
            return item();
        }
        LFS_ASSERT_MSG(false,
                       "TensorRowProxy float conversion requires a 1D or [N,1] tensor");
    }

    // ============= TensorRowProxy Conversion to Tensor =============

    TensorRowProxy::operator Tensor() const {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy tensor conversion");
        flush_cuda_staging();

        if (tensor_->shape().rank() > 1) {
            // Build a proper row view so storage offsets/strides are respected for non-contiguous tensors.
            Tensor row_view = tensor_->slice(0, row_index_, row_index_ + 1).squeeze(0);
            LFS_ASSERT_MSG(row_view.is_valid(),
                           "TensorRowProxy failed to create a row view");
            return row_view.clone();
        }

        // For 1D tensors, return a scalar tensor
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       "TensorRowProxy scalar tensor conversion currently supports only Float32");
        float val = item();

        auto result = Tensor::empty({1}, tensor_->device(), tensor_->dtype());

        if (tensor_->device() == Device::CUDA) {
            cuda_copy_async_sync(
                result.data_ptr(),
                &val,
                sizeof(float),
                cudaMemcpyHostToDevice,
                tensor_->stream(),
                "TensorRowProxy scalar tensor conversion");
        } else {
            *result.ptr<float>() = val;
        }

        return result.squeeze();
    }

    // ============= TensorRowProxy Assignment Operators =============

    TensorRowProxy& TensorRowProxy::operator=(const TensorRowProxy& other) {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy assignment");
        assert_proxy_tensor(other.tensor_, other.row_index_,
                            "TensorRowProxy source assignment");
        if (this == &other) {
            return *this;
        }
        flush_cuda_staging();
        other.flush_cuda_staging();
        Tensor other_copy = other;
        return operator=(other_copy);
    }

    TensorRowProxy& TensorRowProxy::operator=(const Tensor& other) {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy tensor assignment");
        LFS_ASSERT_MSG(other.is_valid(),
                       "TensorRowProxy assignment requires a valid source tensor");
        LFS_ASSERT_MSG(other.dtype() == tensor_->dtype(),
                       "TensorRowProxy assignment requires matching dtypes");
        flush_cuda_staging();

        if (tensor_->shape().rank() > 1) {
            // Multi-dimensional: assign entire row slice while preserving view aliasing semantics.
            Tensor row_slice = tensor_->slice(0, row_index_, row_index_ + 1);
            LFS_ASSERT_MSG(row_slice.is_valid(),
                           "TensorRowProxy failed to create a row slice for assignment");

            std::vector<size_t> expected_dims;
            const auto& row_shape_dims = row_slice.shape().dims();
            expected_dims.reserve(row_shape_dims.size() - 1);
            for (size_t d = 1; d < row_shape_dims.size(); ++d) {
                expected_dims.push_back(row_shape_dims[d]);
            }
            TensorShape expected_shape(expected_dims);

            LFS_ASSERT_MSG(other.shape() == expected_shape ||
                               other.shape() == row_slice.shape(),
                           "TensorRowProxy assignment source shape does not match the row");

            auto other_copy = (other.device() == tensor_->device())
                                  ? other.clone()
                                  : other.to(tensor_->device());
            LFS_ASSERT_MSG(other_copy.is_valid(),
                           "TensorRowProxy failed to convert its assignment source");

            Tensor source_for_copy = other_copy;
            if (source_for_copy.shape() == expected_shape) {
                source_for_copy = source_for_copy.unsqueeze(0);
            }
            LFS_ASSERT_MSG(source_for_copy.shape() == row_slice.shape(),
                           "TensorRowProxy failed to align the assignment source shape");

            if (tensor_->device() == Device::CPU) {
                if (!source_for_copy.is_contiguous()) {
                    source_for_copy = source_for_copy.contiguous();
                }

                const size_t elem_size = dtype_size(tensor_->dtype());
                const char* src_base = static_cast<const char*>(source_for_copy.data_ptr());
                char* dst_base = static_cast<char*>(row_slice.data_ptr());
                std::vector<size_t> indices(row_slice.shape().rank(), 0);

                for (size_t i = 0; i < row_slice.numel(); ++i) {
                    size_t dst_offset = 0;
                    for (size_t d = 0; d < indices.size(); ++d) {
                        dst_offset += indices[d] * row_slice.stride(d);
                    }

                    std::memcpy(dst_base + dst_offset * elem_size,
                                src_base + i * elem_size,
                                elem_size);

                    if (!indices.empty()) {
                        for (int d = static_cast<int>(indices.size()) - 1; d >= 0; --d) {
                            indices[d]++;
                            if (indices[d] < row_slice.shape()[d]) {
                                break;
                            }
                            indices[d] = 0;
                        }
                    }
                }
            } else {
                row_slice.copy_from(source_for_copy);
            }
        } else {
            // 1D: assign single element
            LFS_ASSERT_MSG(other.numel() == 1,
                           "TensorRowProxy scalar assignment requires a one-element source");
            LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                           "TensorRowProxy scalar assignment currently supports only Float32");

            float val = other.item();

            // Use stride for proper indexing on non-contiguous 1D tensors
            size_t linear_idx = row_index_ * tensor_->stride(0);

            if (tensor_->device() == Device::CUDA) {
                cuda_copy_async_sync(
                    tensor_->ptr<float>() + linear_idx,
                    &val,
                    sizeof(float),
                    cudaMemcpyHostToDevice,
                    tensor_->stream(),
                    "TensorRowProxy scalar assignment from tensor");
            } else {
                tensor_->ptr<float>()[linear_idx] = val;
            }
        }
        return *this;
    }

    TensorRowProxy& TensorRowProxy::operator=(float value) {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy float assignment");
        flush_cuda_staging();
        LFS_ASSERT_MSG(tensor_->shape().rank() == 1,
                       "TensorRowProxy float assignment requires a 1D tensor");
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       "TensorRowProxy float assignment requires Float32");
        LFS_ASSERT_MSG(std::isfinite(value),
                       "TensorRowProxy float assignment requires a finite value");

        // Use stride for proper indexing on non-contiguous 1D tensors
        size_t linear_idx = row_index_ * tensor_->stride(0);

        if (tensor_->device() == Device::CUDA) {
            cuda_copy_async_sync(
                tensor_->ptr<float>() + linear_idx,
                &value,
                sizeof(float),
                cudaMemcpyHostToDevice,
                tensor_->stream(),
                "TensorRowProxy scalar assignment");
        } else {
            tensor_->ptr<float>()[linear_idx] = value;
        }
        return *this;
    }

    // ============= TensorRowProxy Arithmetic Operations =============

    Tensor TensorRowProxy::operator-(const TensorRowProxy& other) const {
        return Tensor(*this).sub(Tensor(other));
    }

    Tensor TensorRowProxy::operator+(const TensorRowProxy& other) const {
        return Tensor(*this).add(Tensor(other));
    }

    Tensor TensorRowProxy::operator*(const TensorRowProxy& other) const {
        return Tensor(*this).mul(Tensor(other));
    }

    Tensor TensorRowProxy::operator/(const TensorRowProxy& other) const {
        return Tensor(*this).div(Tensor(other));
    }

    Tensor TensorRowProxy::operator-(float scalar) const {
        return Tensor(*this).sub(scalar);
    }

    Tensor TensorRowProxy::operator+(float scalar) const {
        return Tensor(*this).add(scalar);
    }

    Tensor TensorRowProxy::operator*(float scalar) const {
        return Tensor(*this).mul(scalar);
    }

    Tensor TensorRowProxy::operator/(float scalar) const {
        return Tensor(*this).div(scalar);
    }

    // ============= TensorRowProxy Unary Operations =============

    Tensor TensorRowProxy::operator-() const {
        return Tensor(*this).neg();
    }

    Tensor TensorRowProxy::pow(float exponent) const {
        return Tensor(*this).pow(exponent);
    }

    Tensor TensorRowProxy::sqrt() const {
        return Tensor(*this).sqrt();
    }

    Tensor TensorRowProxy::abs() const {
        return Tensor(*this).abs();
    }

    Tensor TensorRowProxy::neg() const {
        return Tensor(*this).neg();
    }

    Tensor TensorRowProxy::sum() const {
        return Tensor(*this).sum();
    }

    Tensor TensorRowProxy::mean() const {
        return Tensor(*this).mean();
    }

    Tensor TensorRowProxy::square() const {
        return Tensor(*this).square();
    }

} // namespace lfs::core
