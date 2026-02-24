/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

namespace lfs::core {
    namespace {
        void cuda_copy_async_sync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind,
                                  cudaStream_t stream, const char* context) {
            cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
            if (err == cudaSuccess) {
                err = cudaStreamSynchronize(stream);
            }
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memcpy failed in ") + context + ": " + cudaGetErrorString(err));
            }
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
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy: invalid tensor pointer");
        }

        if (tensor_->shape().rank() < 2) {
            throw std::runtime_error(
                "TensorRowProxy: Cannot use tensor[i][j] on " + std::to_string(tensor_->shape().rank()) +
                "-D tensor. Use tensor[i] for 1-D access, or tensor.unsqueeze() to add dimensions.");
        }

        if (col_index >= tensor_->shape()[1]) {
            throw std::out_of_range(
                "Column index " + std::to_string(col_index) + " out of bounds for dimension 1 with size " +
                std::to_string(tensor_->shape()[1]));
        }

        // Use actual strides for proper indexing on non-contiguous tensors
        size_t linear_idx = row_index_ * tensor_->stride(0) + col_index * tensor_->stride(1);

        if (tensor_->device() != Device::CPU) {
            if (tensor_->dtype() != DataType::Float32) {
                throw std::runtime_error("TensorRowProxy::operator[] mutable access requires Float32 tensor");
            }
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

        if (tensor_->dtype() != DataType::Float32) {
            throw std::runtime_error("TensorRowProxy::operator[] mutable access requires Float32 tensor");
        }
        return tensor_->ptr<float>()[linear_idx];
    }

    float TensorRowProxy::operator[](size_t col_index) const {
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy: invalid tensor pointer");
        }
        flush_cuda_staging();

        if (tensor_->shape().rank() < 2) {
            throw std::runtime_error(
                "TensorRowProxy: tensor rank " + std::to_string(tensor_->shape().rank()) + " < 2");
        }

        if (col_index >= tensor_->shape()[1]) {
            throw std::out_of_range(
                "Column index " + std::to_string(col_index) + " out of bounds for dimension 1 with size " +
                std::to_string(tensor_->shape()[1]));
        }

        // Use actual strides for proper indexing on non-contiguous tensors
        size_t linear_idx = row_index_ * tensor_->stride(0) + col_index * tensor_->stride(1);

        if (tensor_->device() == Device::CUDA) {
            if (tensor_->dtype() != DataType::Float32) {
                throw std::runtime_error("TensorRowProxy::operator[] const access requires Float32 tensor");
            }
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
            if (tensor_->dtype() != DataType::Float32) {
                throw std::runtime_error("TensorRowProxy::operator[] const access requires Float32 tensor");
            }
            return tensor_->ptr<float>()[linear_idx];
        }
    }

    // ============= TensorRowProxy 1D Access =============

    float TensorRowProxy::item() const {
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy::item(): invalid tensor pointer");
        }
        flush_cuda_staging();

        // Handle 2D tensors with shape [N, 1] (like nonzero() output)
        if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
            Tensor row_tensor = static_cast<Tensor>(*this);
            return row_tensor.item();
        }

        // Standard 1D case
        if (tensor_->shape().rank() != 1) {
            throw std::runtime_error(
                "TensorRowProxy::item(): only valid for 1D tensors, got rank " +
                std::to_string(tensor_->shape().rank()));
        }

        if (row_index_ >= tensor_->numel()) {
            throw std::out_of_range(
                "TensorRowProxy::item(): index " + std::to_string(row_index_) +
                " out of bounds for size " + std::to_string(tensor_->numel()));
        }

        // Use stride for proper indexing on non-contiguous 1D tensors
        size_t linear_idx = row_index_ * tensor_->stride(0);

        if (tensor_->device() == Device::CUDA) {
            if (tensor_->dtype() != DataType::Float32) {
                throw std::runtime_error("TensorRowProxy::item() currently supports Float32 tensors on CUDA");
            }
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
            if (tensor_->dtype() != DataType::Float32) {
                throw std::runtime_error("TensorRowProxy::item() currently supports Float32 tensors on CPU");
            }
            return tensor_->ptr<float>()[linear_idx];
        }
    }

    TensorRowProxy::operator float() const {
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy: invalid tensor pointer in float conversion");
        }
        flush_cuda_staging();

        if (tensor_->shape().rank() == 1) {
            return item();
        } else if (tensor_->shape().rank() == 2 && tensor_->shape()[1] == 1) {
            return item();
        } else {
            throw std::runtime_error(
                "Implicit float conversion only valid for 1D or [N,1] tensors, got rank " +
                std::to_string(tensor_->shape().rank()) + " with shape " + tensor_->shape().str() +
                ". Use .item() or convert to Tensor first.");
        }
    }

    // ============= TensorRowProxy Conversion to Tensor =============

    TensorRowProxy::operator Tensor() const {
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy: invalid tensor pointer");
        }
        flush_cuda_staging();

        if (tensor_->shape().rank() > 1) {
            // Build a proper row view so storage offsets/strides are respected for non-contiguous tensors.
            Tensor row_view = tensor_->slice(0, row_index_, row_index_ + 1).squeeze(0);
            if (!row_view.is_valid()) {
                throw std::runtime_error("TensorRowProxy: failed to create row view");
            }
            return row_view.clone();
        }

        // For 1D tensors, return a scalar tensor
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
        if (this == &other) {
            return *this;
        }
        flush_cuda_staging();
        other.flush_cuda_staging();
        Tensor other_copy = other;
        return operator=(other_copy);
    }

    TensorRowProxy& TensorRowProxy::operator=(const Tensor& other) {
        if (!tensor_ || !tensor_->is_valid()) {
            return *this;
        }
        flush_cuda_staging();

        if (tensor_->shape().rank() > 1) {
            // Multi-dimensional: assign entire row slice while preserving view aliasing semantics.
            Tensor row_slice = tensor_->slice(0, row_index_, row_index_ + 1);
            if (!row_slice.is_valid()) {
                throw std::runtime_error("TensorRowProxy: failed to create row slice for assignment");
            }

            std::vector<size_t> expected_dims;
            const auto& row_shape_dims = row_slice.shape().dims();
            expected_dims.reserve(row_shape_dims.size() - 1);
            for (size_t d = 1; d < row_shape_dims.size(); ++d) {
                expected_dims.push_back(row_shape_dims[d]);
            }
            TensorShape expected_shape(expected_dims);

            if (other.shape() != expected_shape && other.shape() != row_slice.shape()) {
                throw std::runtime_error(
                    "Shape mismatch in row assignment: expected " + expected_shape.str() +
                    " (or " + row_slice.shape().str() + "), got " + other.shape().str());
            }

            auto other_copy = (other.device() == tensor_->device())
                                  ? other.clone()
                                  : other.to(tensor_->device());
            if (!other_copy.is_valid()) {
                throw std::runtime_error("TensorRowProxy: failed to convert source row for assignment");
            }

            Tensor source_for_copy = other_copy;
            if (source_for_copy.shape() == expected_shape) {
                source_for_copy = source_for_copy.unsqueeze(0);
            }
            if (source_for_copy.shape() != row_slice.shape()) {
                throw std::runtime_error("TensorRowProxy: failed to align source shape for row assignment");
            }

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
            if (other.numel() != 1) {
                throw std::runtime_error(
                    "Cannot assign tensor with " + std::to_string(other.numel()) + " elements to single position");
            }

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
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy: invalid tensor pointer in float assignment");
        }
        flush_cuda_staging();

        if (tensor_->shape().rank() != 1) {
            throw std::runtime_error(
                "Float assignment only valid for 1D tensors, got rank " +
                std::to_string(tensor_->shape().rank()));
        }

        if (row_index_ >= tensor_->numel()) {
            throw std::out_of_range(
                "Index " + std::to_string(row_index_) + " out of bounds for size " +
                std::to_string(tensor_->numel()));
        }

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
