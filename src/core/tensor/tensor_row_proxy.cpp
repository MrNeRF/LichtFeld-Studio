/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include <cstring>
#include <cuda_runtime.h>

namespace lfs::core {

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
            thread_local static float cuda_read_value = 0.0f;

            cudaError_t err = cudaMemcpy(
                &cuda_read_value,
                tensor_->ptr<float>() + linear_idx,
                sizeof(float),
                cudaMemcpyDeviceToHost);

            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memcpy failed in TensorRowProxy::operator[]: ") + cudaGetErrorString(err));
            }

            return cuda_read_value;
        }

        return tensor_->ptr<float>()[linear_idx];
    }

    float TensorRowProxy::operator[](size_t col_index) const {
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy: invalid tensor pointer");
        }

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
            float value = 0.0f;
            cudaError_t err = cudaMemcpy(
                &value,
                tensor_->ptr<float>() + linear_idx,
                sizeof(float),
                cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memcpy failed in TensorRowProxy::operator[]: ") + cudaGetErrorString(err));
            }
            return value;
        } else {
            return tensor_->ptr<float>()[linear_idx];
        }
    }

    // ============= TensorRowProxy 1D Access =============

    float TensorRowProxy::item() const {
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy::item(): invalid tensor pointer");
        }

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
            float value = 0.0f;
            cudaError_t err = cudaMemcpy(
                &value,
                tensor_->ptr<float>() + linear_idx,
                sizeof(float),
                cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memcpy failed in TensorRowProxy::item(): ") + cudaGetErrorString(err));
            }
            return value;
        } else {
            return tensor_->ptr<float>()[linear_idx];
        }
    }

    TensorRowProxy::operator float() const {
        if (!tensor_ || !tensor_->is_valid()) {
            throw std::runtime_error("TensorRowProxy: invalid tensor pointer in float conversion");
        }

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

        if (tensor_->shape().rank() > 1) {
            // For multi-dimensional tensors, manually copy the row data
            std::vector<size_t> row_shape;
            for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                row_shape.push_back(tensor_->shape()[i]);
            }

            auto result = Tensor::empty(TensorShape(row_shape),
                                        tensor_->device(),
                                        tensor_->dtype());

            size_t row_elements = 1;
            for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                row_elements *= tensor_->shape()[i];
            }

            size_t byte_offset = row_index_ * row_elements * dtype_size(tensor_->dtype());
            size_t copy_bytes = row_elements * dtype_size(tensor_->dtype());

            if (tensor_->device() == Device::CUDA) {
                cudaError_t err = cudaMemcpy(
                    result.data_ptr(),
                    static_cast<const char*>(tensor_->data_ptr()) + byte_offset,
                    copy_bytes,
                    cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("CUDA memcpy failed in TensorRowProxy tensor conversion: ") +
                        cudaGetErrorString(err));
                }
            } else {
                std::memcpy(
                    result.data_ptr(),
                    static_cast<const char*>(tensor_->data_ptr()) + byte_offset,
                    copy_bytes);
            }

            return result;
        }

        // For 1D tensors, return a scalar tensor
        float val = item();

        auto result = Tensor::empty({1}, tensor_->device(), tensor_->dtype());

        if (tensor_->device() == Device::CUDA) {
            cudaMemcpy(result.data_ptr(), &val, sizeof(float), cudaMemcpyHostToDevice);
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
        Tensor other_copy = other;
        return operator=(other_copy);
    }

    TensorRowProxy& TensorRowProxy::operator=(const Tensor& other) {
        if (!tensor_ || !tensor_->is_valid()) {
            return *this;
        }

        if (tensor_->shape().rank() > 1) {
            // Multi-dimensional: assign entire row slice
            std::vector<size_t> slice_shape;
            for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                slice_shape.push_back(tensor_->shape()[i]);
            }
            TensorShape expected_shape(slice_shape);

            if (other.shape() != expected_shape) {
                throw std::runtime_error(
                    "Shape mismatch in row assignment: expected " + expected_shape.str() +
                    ", got " + other.shape().str());
            }

            size_t row_elements = 1;
            for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                row_elements *= tensor_->shape()[i];
            }

            size_t byte_offset = row_index_ * row_elements * dtype_size(tensor_->dtype());
            size_t copy_bytes = row_elements * dtype_size(tensor_->dtype());

            auto other_copy = (other.device() == tensor_->device())
                                  ? other.clone()
                                  : other.to(tensor_->device());

            if (tensor_->device() == Device::CUDA) {
                cudaError_t err = cudaMemcpy(
                    static_cast<char*>(tensor_->data_ptr()) + byte_offset,
                    other_copy.data_ptr(),
                    copy_bytes,
                    cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("CUDA memcpy failed in row assignment: ") + cudaGetErrorString(err));
                }
            } else {
                std::memcpy(
                    static_cast<char*>(tensor_->data_ptr()) + byte_offset,
                    other_copy.data_ptr(),
                    copy_bytes);
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
                cudaMemcpy(
                    tensor_->ptr<float>() + linear_idx,
                    &val,
                    sizeof(float),
                    cudaMemcpyHostToDevice);
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
            cudaMemcpy(
                tensor_->ptr<float>() + linear_idx,
                &value,
                sizeof(float),
                cudaMemcpyHostToDevice);
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
