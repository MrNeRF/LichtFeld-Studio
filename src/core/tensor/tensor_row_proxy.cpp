/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
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
                           std::format("{} CUDA copy/synchronization failed "
                                       "(cuda_error={}({}), bytes={}, copy_kind={}, "
                                       "source_pointer={}, destination_pointer={}, stream={})",
                                       context, cudaGetErrorString(err), static_cast<int>(err), bytes,
                                       static_cast<int>(kind), src, dst,
                                       static_cast<const void*>(stream)));
        }

        void assert_proxy_tensor(const Tensor* tensor,
                                 const size_t row_index,
                                 const std::string_view operation) {
            LFS_ASSERT_MSG(tensor != nullptr && tensor->is_valid(),
                           std::format("{} requires a non-null valid tensor "
                                       "(tensor_pointer={}, tensor_state={}, row_index={})",
                                       operation, static_cast<const void*>(tensor),
                                       tensor != nullptr ? tensor->str() : "null", row_index));
            LFS_ASSERT_MSG(tensor->ndim() > 0,
                           std::format("{} requires a tensor with at least one dimension "
                                       "(tensor_rank={}, tensor_shape={}, row_index={})",
                                       operation, tensor->ndim(), tensor->shape().str(), row_index));
            LFS_ASSERT_MSG(row_index < tensor->shape()[0],
                           std::format("{} row index must be in range "
                                       "(row_index={}, row_count={}, tensor_shape={})",
                                       operation, row_index, tensor->shape()[0], tensor->shape().str()));
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
                       std::format("TensorRowProxy::operator[] requires rank at least 2 "
                                   "(tensor_rank={}, tensor_shape={}, row_index={}, column_index={})",
                                   tensor_->shape().rank(), tensor_->shape().str(),
                                   row_index_, col_index));
        LFS_ASSERT_MSG(col_index < tensor_->shape()[1],
                       std::format("TensorRowProxy column index must be in range "
                                   "(column_index={}, column_count={}, row_index={}, tensor_shape={})",
                                   col_index, tensor_->shape()[1], row_index_, tensor_->shape().str()));
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       std::format("TensorRowProxy mutable element access requires Float32 "
                                   "(tensor_dtype={}({}), tensor_shape={}, row_index={}, "
                                   "column_index={})",
                                   dtype_name(tensor_->dtype()), static_cast<int>(tensor_->dtype()),
                                   tensor_->shape().str(), row_index_, col_index));

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
                       std::format("TensorRowProxy::operator[] const requires rank at least 2 "
                                   "(tensor_rank={}, tensor_shape={}, row_index={}, column_index={})",
                                   tensor_->shape().rank(), tensor_->shape().str(),
                                   row_index_, col_index));
        LFS_ASSERT_MSG(col_index < tensor_->shape()[1],
                       std::format("TensorRowProxy const column index must be in range "
                                   "(column_index={}, column_count={}, row_index={}, tensor_shape={})",
                                   col_index, tensor_->shape()[1], row_index_, tensor_->shape().str()));
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       std::format("TensorRowProxy const element access requires Float32 "
                                   "(tensor_dtype={}({}), tensor_shape={}, row_index={}, "
                                   "column_index={})",
                                   dtype_name(tensor_->dtype()), static_cast<int>(tensor_->dtype()),
                                   tensor_->shape().str(), row_index_, col_index));

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
                       std::format("TensorRowProxy::item() requires a 1D or [N,1] tensor "
                                   "(tensor_rank={}, tensor_shape={}, row_index={})",
                                   tensor_->shape().rank(), tensor_->shape().str(), row_index_));
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       std::format("TensorRowProxy::item() requires Float32 input "
                                   "(tensor_dtype={}({}), tensor_shape={}, row_index={})",
                                   dtype_name(tensor_->dtype()), static_cast<int>(tensor_->dtype()),
                                   tensor_->shape().str(), row_index_));

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
                       std::format("TensorRowProxy float conversion requires a 1D or [N,1] tensor "
                                   "(tensor_rank={}, tensor_shape={}, row_index={})",
                                   tensor_->shape().rank(), tensor_->shape().str(), row_index_));
    }

    // ============= TensorRowProxy Conversion to Tensor =============

    TensorRowProxy::operator Tensor() const {
        assert_proxy_tensor(tensor_, row_index_, "TensorRowProxy tensor conversion");
        flush_cuda_staging();

        if (tensor_->shape().rank() > 1) {
            // Build a proper row view so storage offsets/strides are respected for non-contiguous tensors.
            Tensor row_view = tensor_->slice(0, row_index_, row_index_ + 1).squeeze(0);
            LFS_ASSERT_MSG(row_view.is_valid(),
                           std::format("TensorRowProxy row-view construction must produce a valid tensor "
                                       "(row_view={}, source_shape={}, row_index={})",
                                       row_view.str(), tensor_->shape().str(), row_index_));
            return row_view.clone();
        }

        // For 1D tensors, return a scalar tensor
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       std::format("TensorRowProxy scalar conversion requires Float32 input "
                                   "(tensor_dtype={}({}), tensor_shape={}, row_index={})",
                                   dtype_name(tensor_->dtype()), static_cast<int>(tensor_->dtype()),
                                   tensor_->shape().str(), row_index_));
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
                       std::format("TensorRowProxy assignment requires a valid source tensor "
                                   "(source_valid=false, destination_row={}, destination_shape={})",
                                   row_index_, tensor_->shape().str()));
        LFS_ASSERT_MSG(other.dtype() == tensor_->dtype(),
                       std::format("TensorRowProxy assignment requires matching dtypes "
                                   "(destination_dtype={}({}), source_dtype={}({}), row={})",
                                   dtype_name(tensor_->dtype()), static_cast<int>(tensor_->dtype()),
                                   dtype_name(other.dtype()), static_cast<int>(other.dtype()), row_index_));
        flush_cuda_staging();

        if (tensor_->shape().rank() > 1) {
            // Multi-dimensional: assign entire row slice while preserving view aliasing semantics.
            Tensor row_slice = tensor_->slice(0, row_index_, row_index_ + 1);
            LFS_ASSERT_MSG(row_slice.is_valid(),
                           std::format("TensorRowProxy assignment row-slice construction must "
                                       "produce a valid tensor "
                                       "(row_slice={}, destination_shape={}, row_index={})",
                                       row_slice.str(), tensor_->shape().str(), row_index_));

            std::vector<size_t> expected_dims;
            const auto& row_shape_dims = row_slice.shape().dims();
            expected_dims.reserve(row_shape_dims.size() - 1);
            for (size_t d = 1; d < row_shape_dims.size(); ++d) {
                expected_dims.push_back(row_shape_dims[d]);
            }
            TensorShape expected_shape(expected_dims);

            LFS_ASSERT_MSG(other.shape() == expected_shape ||
                               other.shape() == row_slice.shape(),
                           std::format("TensorRowProxy assignment source shape does not match the row "
                                       "(source_shape={}, expected_shape={} or {}, row={})",
                                       other.shape().str(), expected_shape.str(), row_slice.shape().str(),
                                       row_index_));

            auto other_copy = (other.device() == tensor_->device())
                                  ? other.clone()
                                  : other.to(tensor_->device());
            LFS_ASSERT_MSG(other_copy.is_valid(),
                           std::format("TensorRowProxy assignment source conversion must produce "
                                       "a valid tensor (converted_source={}, original_source={}, "
                                       "destination_device={}, row_index={})",
                                       other_copy.str(), other.str(),
                                       device_name(tensor_->device()), row_index_));

            Tensor source_for_copy = other_copy;
            if (source_for_copy.shape() == expected_shape) {
                source_for_copy = source_for_copy.unsqueeze(0);
            }
            LFS_ASSERT_MSG(source_for_copy.shape() == row_slice.shape(),
                           std::format("TensorRowProxy aligned assignment source shape must match "
                                       "the destination row slice "
                                       "(aligned_source_shape={}, row_slice_shape={}, row_index={})",
                                       source_for_copy.shape().str(), row_slice.shape().str(), row_index_));

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
                           std::format("TensorRowProxy scalar assignment requires a one-element source "
                                       "(source_numel={}, source_shape={}, destination_shape={}, "
                                       "row_index={})",
                                       other.numel(), other.shape().str(),
                                       tensor_->shape().str(), row_index_));
            LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                           std::format("TensorRowProxy scalar assignment requires Float32 destination "
                                       "(destination_dtype={}({}), destination_shape={}, row_index={})",
                                       dtype_name(tensor_->dtype()),
                                       static_cast<int>(tensor_->dtype()),
                                       tensor_->shape().str(), row_index_));

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
                       std::format("TensorRowProxy float assignment requires a 1D destination "
                                   "(destination_rank={}, destination_shape={}, row_index={}, value={})",
                                   tensor_->shape().rank(), tensor_->shape().str(), row_index_, value));
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32,
                       std::format("TensorRowProxy float assignment requires Float32 destination "
                                   "(destination_dtype={}({}), destination_shape={}, row_index={}, "
                                   "value={})",
                                   dtype_name(tensor_->dtype()), static_cast<int>(tensor_->dtype()),
                                   tensor_->shape().str(), row_index_, value));
        LFS_ASSERT_MSG(std::isfinite(value),
                       std::format("TensorRowProxy float assignment requires a finite value "
                                   "(value={}, value_finite={}, destination_shape={}, row_index={})",
                                   value, std::isfinite(value), tensor_->shape().str(), row_index_));

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
