/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <execution>
#include <format>
#include <limits>
#include <numeric>
#include <ranges>

#define CHECK_CUDA(call)                                                                 \
    do {                                                                                 \
        const cudaError_t error = (call);                                                \
        LFS_ASSERT_MSG(error == cudaSuccess,                                             \
                       std::format("{} failed (cuda_error={}({}))", #call,               \
                                   cudaGetErrorString(error), static_cast<int>(error))); \
    } while (0)

namespace lfs::core {

    namespace {
        template <typename T>
        T masked_fill_cast(float value) {
            return static_cast<T>(value);
        }

        template <>
        __half masked_fill_cast<__half>(float value) {
            return __float2half(value);
        }

        template <typename T>
        void masked_fill_cpu(T* data, const unsigned char* mask_data, size_t n, float value) {
            const T cast_value = masked_fill_cast<T>(value);
            for (size_t i = 0; i < n; ++i) {
                if (mask_data[i]) {
                    data[i] = cast_value;
                }
            }
        }
        template <typename T>
        void masked_select_cpu(const T* input, const unsigned char* mask, T* output, size_t n) {
            size_t write_idx = 0;
            for (size_t i = 0; i < n; ++i) {
                if (mask[i]) {
                    output[write_idx++] = input[i];
                }
            }
        }

        void assert_masked_fill_value_representable(const DataType dtype, const float value) {
            switch (dtype) {
            case DataType::Float32:
                return;
            case DataType::Float16:
                LFS_ASSERT_MSG(std::abs(value) <= 65504.0f,
                               std::format("masked_fill_ value is outside the Float16 finite range "
                                           "(value={}, allowed=[-65504,65504])",
                                           value));
                return;
            case DataType::Int32:
                LFS_ASSERT_MSG(value >= -std::ldexp(1.0f, 31) &&
                                   value < std::ldexp(1.0f, 31),
                               std::format("masked_fill_ value is outside the Int32 range "
                                           "(value={}, allowed=[-2147483648,2147483648))",
                                           value));
                return;
            case DataType::Int64:
                LFS_ASSERT_MSG(value >= -std::ldexp(1.0f, 63) &&
                                   value < std::ldexp(1.0f, 63),
                               std::format("masked_fill_ value is outside the Int64 range "
                                           "(value={}, allowed=[-2^63,2^63))",
                                           value));
                return;
            case DataType::UInt8:
                LFS_ASSERT_MSG(value >= 0.0f && value <= 255.0f,
                               std::format("masked_fill_ value is outside the UInt8 range "
                                           "(value={}, allowed=[0,255])",
                                           value));
                return;
            case DataType::Bool:
                LFS_ASSERT_MSG(value == 0.0f || value == 1.0f,
                               std::format("masked_fill_ Bool value must be zero or one "
                                           "(value={})",
                                           value));
                return;
            }
            LFS_ASSERT_MSG(false,
                           std::format("masked_fill_ encountered an unsupported dtype "
                                       "(dtype={}({}))",
                                       dtype_name(dtype), static_cast<int>(dtype)));
        }

        [[nodiscard]] bool is_integer_index_dtype(const DataType dtype) {
            return dtype == DataType::Int32 || dtype == DataType::Int64;
        }

        void assert_index_tensor(const Tensor& indices,
                                 const size_t upper_bound,
                                 const std::string_view operation,
                                 const bool check_bounds,
                                 const bool allow_negative = false) {
            LFS_ASSERT_MSG(indices.is_valid(),
                           std::format("{} received an invalid index tensor "
                                       "(valid=false, upper_bound={}, check_bounds={})",
                                       operation, upper_bound, check_bounds));
            LFS_ASSERT_MSG(is_integer_index_dtype(indices.dtype()),
                           std::format("{} indices must be Int32 or Int64 "
                                       "(dtype={}({}), shape={}, device={})",
                                       operation, dtype_name(indices.dtype()),
                                       static_cast<int>(indices.dtype()), indices.shape().str(),
                                       device_name(indices.device())));
            if (indices.numel() == 0) {
                return;
            }
            LFS_ASSERT_MSG(upper_bound > 0,
                           std::format("{} cannot index an empty dimension "
                                       "(upper_bound=0, index_count={})",
                                       operation, indices.numel()));
            LFS_ASSERT_MSG(upper_bound <= static_cast<size_t>(std::numeric_limits<int>::max()),
                           std::format("{} indexed dimension exceeds the Int32 kernel range "
                                       "(upper_bound={}, max={})",
                                       operation, upper_bound, std::numeric_limits<int>::max()));

            const Tensor cpu_indices = indices.device() == Device::CPU
                                           ? indices.contiguous()
                                           : indices.cpu().contiguous();
            const auto assert_value = [&](const int64_t value, const size_t position) {
                LFS_ASSERT_MSG(value >= std::numeric_limits<int>::min() &&
                                   value <= std::numeric_limits<int>::max(),
                               std::format("{}: index {} at position {} cannot be represented by the Int32 kernel",
                                           operation, value, position));
                if (!check_bounds) {
                    return;
                }
                const int64_t lower_bound = allow_negative ? -static_cast<int64_t>(upper_bound) : 0;
                LFS_ASSERT_MSG(value >= lower_bound && value < static_cast<int64_t>(upper_bound),
                               std::format("{}: index {} at position {} is out of bounds for size {}",
                                           operation, value, position, upper_bound));
            };

            if (cpu_indices.dtype() == DataType::Int64) {
                const auto* values = cpu_indices.ptr<int64_t>();
                for (size_t i = 0; i < cpu_indices.numel(); ++i) {
                    assert_value(values[i], i);
                }
            } else {
                const auto* values = cpu_indices.ptr<int32_t>();
                for (size_t i = 0; i < cpu_indices.numel(); ++i) {
                    assert_value(values[i], i);
                }
            }
        }
    } // namespace

    // ============= Masking Operations =============
    Tensor Tensor::masked_select(const Tensor& mask) const {
        LFS_ASSERT_MSG(is_valid() && mask.is_valid(),
                       std::format("masked_select requires valid input and mask tensors "
                                   "(input_valid={}, mask_valid={})",
                                   is_valid(), mask.is_valid()));
        LFS_ASSERT_MSG(is_bool_like(mask.dtype()),
                       std::format("masked_select mask must be Bool or UInt8 "
                                   "(mask_dtype={}({}))",
                                   dtype_name(mask.dtype()), static_cast<int>(mask.dtype())));
        LFS_ASSERT_MSG(mask.shape() == shape_,
                       std::format("masked_select mask shape must match the input "
                                   "(input_shape={}, mask_shape={})",
                                   shape_.str(), mask.shape().str()));
        LFS_ASSERT_MSG(mask.device() == device_,
                       std::format("masked_select mask must be on the input device "
                                   "(input_device={}, mask_device={})",
                                   device_name(device_), device_name(mask.device())));

        // Count TRUE values in mask
        size_t output_size = mask.count_nonzero();

        LOG_DEBUG("masked_select: input size={}, mask trues={}, output size={}",
                  numel(), output_size, output_size);

        if (output_size == 0) {
            return empty({0}, device_, dtype_);
        }

        auto result = empty({output_size}, device_, dtype_);

        if (device_ == Device::CUDA) {
            result.set_stream(stream());
            switch (dtype_) {
            case DataType::Float32:
                tensor_ops::launch_masked_select(ptr<float>(), mask.ptr<unsigned char>(),
                                                 result.ptr<float>(), numel(), output_size, stream());
                break;
            case DataType::Float16:
                tensor_ops::launch_masked_select(ptr<__half>(), mask.ptr<unsigned char>(),
                                                 result.ptr<__half>(), numel(), output_size, stream());
                break;
            case DataType::Int32:
                tensor_ops::launch_masked_select(ptr<int32_t>(), mask.ptr<unsigned char>(),
                                                 result.ptr<int32_t>(), numel(), output_size, stream());
                break;
            case DataType::Int64:
                tensor_ops::launch_masked_select(ptr<int64_t>(), mask.ptr<unsigned char>(),
                                                 result.ptr<int64_t>(), numel(), output_size, stream());
                break;
            case DataType::UInt8:
            case DataType::Bool:
                tensor_ops::launch_masked_select(ptr<uint8_t>(), mask.ptr<unsigned char>(),
                                                 result.ptr<uint8_t>(), numel(), output_size, stream());
                break;
            }
            const cudaError_t launch_error = cudaGetLastError();
            LFS_ASSERT_MSG(launch_error == cudaSuccess,
                           std::format("masked_select CUDA kernel launch failed "
                                       "(cuda_error={}({}), input_shape={}, input_dtype={}({}), "
                                       "mask_shape={}, selected_count={}, stream={})",
                                       cudaGetErrorString(launch_error),
                                       static_cast<int>(launch_error), shape_.str(),
                                       dtype_name(dtype_), static_cast<int>(dtype_),
                                       mask.shape().str(), output_size,
                                       static_cast<const void*>(stream())));
            // No sync - tensor operation
        } else {
            switch (dtype_) {
            case DataType::Float32:
                masked_select_cpu(ptr<float>(), mask.ptr<unsigned char>(), result.ptr<float>(), numel());
                break;
            case DataType::Float16:
                masked_select_cpu(ptr<__half>(), mask.ptr<unsigned char>(), result.ptr<__half>(), numel());
                break;
            case DataType::Int32:
                masked_select_cpu(ptr<int32_t>(), mask.ptr<unsigned char>(), result.ptr<int32_t>(), numel());
                break;
            case DataType::Int64:
                masked_select_cpu(ptr<int64_t>(), mask.ptr<unsigned char>(), result.ptr<int64_t>(), numel());
                break;
            case DataType::UInt8:
            case DataType::Bool:
                masked_select_cpu(ptr<uint8_t>(), mask.ptr<unsigned char>(), result.ptr<uint8_t>(), numel());
                break;
            }
        }

        return result;
    }

    Tensor& Tensor::masked_fill_(const Tensor& mask, float value) {
        LFS_ASSERT_MSG(is_valid() && mask.is_valid(),
                       std::format("masked_fill_ requires valid destination and mask tensors "
                                   "(destination_valid={}, mask_valid={})",
                                   is_valid(), mask.is_valid()));
        LFS_ASSERT_MSG(is_bool_like(mask.dtype()),
                       std::format("masked_fill_ mask must be Bool or UInt8 "
                                   "(mask_dtype={}({}))",
                                   dtype_name(mask.dtype()), static_cast<int>(mask.dtype())));
        LFS_ASSERT_MSG(mask.shape() == shape_,
                       std::format("masked_fill_ mask shape must match the destination "
                                   "(destination_shape={}, mask_shape={})",
                                   shape_.str(), mask.shape().str()));
        LFS_ASSERT_MSG(mask.device() == device_,
                       std::format("masked_fill_ mask must be on the destination device "
                                   "(destination_device={}, mask_device={})",
                                   device_name(device_), device_name(mask.device())));
        LFS_ASSERT_MSG(std::isfinite(value),
                       std::format("masked_fill_ value must be finite (value={})", value));
        assert_masked_fill_value_representable(dtype_, value);

        if (device_ == Device::CUDA) {
            switch (dtype_) {
            case DataType::Float32:
                tensor_ops::launch_masked_fill(ptr<float>(), mask.ptr<unsigned char>(),
                                               value, numel(), stream());
                break;
            case DataType::Float16:
                tensor_ops::launch_masked_fill(ptr<__half>(), mask.ptr<unsigned char>(),
                                               __float2half(value), numel(), stream());
                break;
            case DataType::Int32:
                tensor_ops::launch_masked_fill(ptr<int32_t>(), mask.ptr<unsigned char>(),
                                               static_cast<int32_t>(value), numel(), stream());
                break;
            case DataType::Int64:
                tensor_ops::launch_masked_fill(ptr<int64_t>(), mask.ptr<unsigned char>(),
                                               static_cast<int64_t>(value), numel(), stream());
                break;
            case DataType::UInt8:
            case DataType::Bool:
                tensor_ops::launch_masked_fill(ptr<uint8_t>(), mask.ptr<unsigned char>(),
                                               static_cast<uint8_t>(value), numel(), stream());
                break;
            default:
                throw std::runtime_error("masked_fill_: unsupported dtype");
            }
            // No sync - tensor operation
        } else {
            const unsigned char* mask_data = mask.ptr<unsigned char>();

            switch (dtype_) {
            case DataType::Float32:
                masked_fill_cpu(ptr<float>(), mask_data, numel(), value);
                break;
            case DataType::Float16:
                masked_fill_cpu(ptr<__half>(), mask_data, numel(), value);
                break;
            case DataType::Int32:
                masked_fill_cpu(ptr<int32_t>(), mask_data, numel(), value);
                break;
            case DataType::Int64:
                masked_fill_cpu(ptr<int64_t>(), mask_data, numel(), value);
                break;
            case DataType::UInt8:
            case DataType::Bool:
                masked_fill_cpu(ptr<unsigned char>(), mask_data, numel(), value);
                break;
            default:
                throw std::runtime_error("masked_fill_: unsupported dtype");
            }
        }

        return *this;
    }

    Tensor Tensor::masked_fill(const Tensor& mask, float value) const {
        auto result = clone();
        result.masked_fill_(mask, value);
        return result;
    }

    // ============= Indexing Operations =============
    Tensor Tensor::index_select(int dim, const Tensor& indices) const {
        return index_select(dim, indices, BoundaryMode::Assert);
    }

    Tensor Tensor::index_select(int dim, const Tensor& indices, BoundaryMode mode) const {
        const_cast<Tensor*>(this)->materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && indices.is_valid(),
                       std::format("index_select requires valid input and index tensors "
                                   "(input={}, indices={}, requested_dimension={}, boundary_mode={})",
                                   str(), indices.str(), dim, static_cast<int>(mode)));
        LFS_ASSERT_MSG(indices.ndim() == 1,
                       std::format("index_select requires rank-1 indices "
                                   "(index_rank={}, index_shape={}, requested_dimension={})",
                                   indices.ndim(), indices.shape().str(), dim));
        LFS_ASSERT_MSG(indices.device() == device_,
                       std::format("index_select indices must be on the input device "
                                   "(input_device={}, index_device={}, input_shape={}, index_shape={})",
                                   device_name(device_), device_name(indices.device()),
                                   shape_.str(), indices.shape().str()));

        const int requested_dim = dim;
        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(shape_.rank()),
                       std::format("index_select dimension must be in range "
                                   "(requested_dimension={}, resolved_dimension={}, "
                                   "valid_range=[0,{}), input_shape={})",
                                   requested_dim, dim, shape_.rank(), shape_.str()));

        if (is_bool_like(indices.dtype())) {
            LFS_ASSERT_MSG(indices.numel() == shape_[dim],
                           std::format("index_select boolean mask length must match the indexed "
                                       "dimension (mask_length={}, dimension={}, dimension_size={}, "
                                       "input_shape={}, mask_shape={})",
                                       indices.numel(), dim, shape_[dim],
                                       shape_.str(), indices.shape().str()));
            const auto idx = indices.nonzero().squeeze(1);
            if (idx.numel() == 0) {
                auto dims = shape_.dims();
                dims[dim] = 0;
                return empty(TensorShape(dims), device_, dtype_);
            }
            return index_select(dim, idx, mode);
        }

        auto dims = shape_.dims();
        dims[dim] = indices.numel();
        auto result = zeros(TensorShape(dims), device_, dtype_);
        index_select_into(result, dim, indices, mode);
        return result;
    }

    void Tensor::index_select_into(Tensor& out, int dim, const Tensor& indices, BoundaryMode mode) const {
        const_cast<Tensor*>(this)->materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && out.is_valid() && indices.is_valid(),
                       std::format("index_select_into requires valid input, index, and output tensors "
                                   "(input={}, indices={}, output={}, requested_dimension={}, "
                                   "boundary_mode={})",
                                   str(), indices.str(), out.str(), dim, static_cast<int>(mode)));
        LFS_ASSERT_MSG(indices.ndim() == 1,
                       std::format("index_select_into requires rank-1 indices "
                                   "(index_rank={}, index_shape={}, requested_dimension={})",
                                   indices.ndim(), indices.shape().str(), dim));

        const int requested_dim = dim;
        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(shape_.rank()),
                       std::format("index_select_into dimension must be in range "
                                   "(requested_dimension={}, resolved_dimension={}, "
                                   "valid_range=[0,{}), input_shape={})",
                                   requested_dim, dim, shape_.rank(), shape_.str()));
        LFS_ASSERT_MSG(out.device() == device_ && indices.device() == device_,
                       std::format("index_select_into tensors must be on the same device "
                                   "(input_device={}, index_device={}, output_device={})",
                                   device_name(device_), device_name(indices.device()),
                                   device_name(out.device())));
        LFS_ASSERT_MSG(out.dtype() == dtype_,
                       std::format("index_select_into output dtype must match the input "
                                   "(input_dtype={}({}), output_dtype={}({}), "
                                   "input_shape={}, output_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(out.dtype()), static_cast<int>(out.dtype()),
                                   shape_.str(), out.shape().str()));
        auto expected_shape = shape_.dims();
        expected_shape[dim] = indices.numel();
        LFS_ASSERT_MSG(out.shape() == TensorShape(expected_shape),
                       std::format("index_select_into output shape must match the requested gather "
                                   "(output_shape={}, expected_shape={}, dimension={}, "
                                   "index_count={})",
                                   out.shape().str(), TensorShape(expected_shape).str(),
                                   dim, indices.numel()));
        assert_index_tensor(indices, shape_[dim], "index_select_into",
                            mode == BoundaryMode::Assert);

        auto indices_same_device = ensure_same_device(indices);

        // Keep Int64 indices, don't convert to Int32 (causes corruption)
        bool is_int64 = indices_same_device.dtype() == DataType::Int64;
        Tensor indices_int32;
        if (is_int64) {
            // Only convert for the kernel call, not in-place
            indices_int32 = indices_same_device.to(DataType::Int32);
        }

        if (device_ == Device::CUDA) {
            const int* idx_ptr = is_int64 ? indices_int32.ptr<int>() : indices_same_device.ptr<int>();

            // Dispatch based on source tensor dtype
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_index_select(ptr<float>(), idx_ptr,
                                                out.ptr<float>(), shape_.dims().data(),
                                                shape_.rank(), dim, indices.numel(),
                                                static_cast<int>(mode), stream());
            } else if (dtype_ == DataType::Int64) {
                tensor_ops::launch_index_select(ptr<int64_t>(), idx_ptr,
                                                out.ptr<int64_t>(), shape_.dims().data(),
                                                shape_.rank(), dim, indices.numel(),
                                                static_cast<int>(mode), stream());
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_index_select(ptr<int32_t>(), idx_ptr,
                                                out.ptr<int32_t>(), shape_.dims().data(),
                                                shape_.rank(), dim, indices.numel(),
                                                static_cast<int>(mode), stream());
            } else if (dtype_ == DataType::UInt8 || dtype_ == DataType::Bool) {
                tensor_ops::launch_index_select(ptr<uint8_t>(), idx_ptr,
                                                out.ptr<uint8_t>(), shape_.dims().data(),
                                                shape_.rank(), dim, indices.numel(),
                                                static_cast<int>(mode), stream());
            } else {
                throw std::runtime_error("index_select: unsupported dtype for CUDA");
            }
            const cudaError_t launch_error = cudaGetLastError();
            LFS_ASSERT_MSG(launch_error == cudaSuccess,
                           std::format(
                               "index_select CUDA kernel launch failed for input {}, output {}, {} indices on dimension {}: {}",
                               shape_.str(), out.shape().str(), indices.numel(), dim,
                               cudaGetErrorString(launch_error)));
            // No sync - tensor operation
        } else {
            // CPU implementation
            size_t outer = 1, inner = 1;
            for (int i = 0; i < dim; ++i)
                outer *= shape_[i];
            for (size_t i = dim + 1; i < shape_.rank(); ++i)
                inner *= shape_[i];

            const int* const idx = is_int64 ? indices_int32.ptr<int>() : indices_same_device.ptr<int>();
            const size_t n_indices = indices.numel();
            const size_t dim_size = shape_[dim];

            const auto process_idx = [dim_size, mode](int sel) -> int {
                if (mode == BoundaryMode::Clamp) {
                    return std::clamp(sel, 0, static_cast<int>(dim_size) - 1);
                } else if (mode == BoundaryMode::Wrap) {
                    return ((sel % static_cast<int>(dim_size)) + dim_size) % dim_size;
                }
                if (sel < 0)
                    sel += dim_size;
                return sel;
            };

            // Templated copy for all dtypes
            const auto copy_selected = [&]<typename T>(const T* src, T* dst) {
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t i = 0; i < n_indices; ++i) {
                        const int sel = process_idx(idx[i]);
                        if (sel >= 0 && sel < static_cast<int>(dim_size)) {
                            std::copy_n(src + (o * dim_size + sel) * inner,
                                        inner,
                                        dst + (o * n_indices + i) * inner);
                        }
                    }
                }
            };

            if (dtype_ == DataType::Float32) {
                copy_selected(ptr<float>(), out.ptr<float>());
            } else if (dtype_ == DataType::Int64) {
                copy_selected(ptr<int64_t>(), out.ptr<int64_t>());
            } else if (dtype_ == DataType::Int32) {
                copy_selected(ptr<int32_t>(), out.ptr<int32_t>());
            } else if (dtype_ == DataType::Bool || dtype_ == DataType::UInt8) {
                copy_selected(ptr<unsigned char>(), out.ptr<unsigned char>());
            } else {
                throw std::runtime_error("index_select: unsupported dtype for CPU");
            }
        }
    }

    Tensor Tensor::gather(int dim, const Tensor& indices) const {
        return gather(dim, indices, BoundaryMode::Assert);
    }

    Tensor Tensor::gather(int dim, const Tensor& indices, BoundaryMode mode) const {
        const_cast<Tensor*>(this)->materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && indices.is_valid(),
                       std::format("gather requires valid input and index tensors "
                                   "(input={}, indices={}, requested_dimension={}, boundary_mode={})",
                                   str(), indices.str(), dim, static_cast<int>(mode)));
        LFS_ASSERT_MSG(indices.device() == device_,
                       std::format("gather indices must be on the input device "
                                   "(input_device={}, index_device={}, input_shape={}, index_shape={})",
                                   device_name(device_), device_name(indices.device()),
                                   shape_.str(), indices.shape().str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int64,
                       std::format("gather currently supports only Float32 and Int64 inputs "
                                   "(input_dtype={}({}), shape={}, device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_), shape_.str(),
                                   device_name(device_)));

        // Ensure we have contiguous data for correct memory access
        if (!is_contiguous()) {
            return contiguous().gather(dim, indices, mode);
        }

        const int requested_dim = dim;
        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(shape_.rank()),
                       std::format("gather dimension must be in range "
                                   "(requested_dimension={}, resolved_dimension={}, "
                                   "valid_range=[0,{}), input_shape={})",
                                   requested_dim, dim, shape_.rank(), shape_.str()));
        assert_index_tensor(indices, shape_[dim], "gather", mode == BoundaryMode::Assert);

        if (indices.ndim() == 1) {
            std::vector<size_t> out_dims = shape_.dims();
            out_dims[dim] = indices.numel();
            auto result = zeros(TensorShape(out_dims), device_, dtype_);

            auto indices_same_device = ensure_same_device(indices);

            // Handle Int64 indices properly
            bool is_int64 = indices_same_device.dtype() == DataType::Int64;
            Tensor indices_int32;
            if (is_int64) {
                indices_int32 = indices_same_device.to(DataType::Int32);
            }

            if (device_ == Device::CUDA) {
                const int* idx_ptr = is_int64 ? indices_int32.ptr<int>() : indices_same_device.ptr<int>();

                // Dispatch based on source tensor dtype
                if (dtype_ == DataType::Float32) {
                    tensor_ops::launch_gather(ptr<float>(), idx_ptr,
                                              result.ptr<float>(), shape_.dims().data(),
                                              indices.shape().dims().data(), shape_.rank(), dim,
                                              result.numel(), static_cast<int>(mode), stream());
                } else if (dtype_ == DataType::Int64) {
                    tensor_ops::launch_gather(ptr<int64_t>(), idx_ptr,
                                              result.ptr<int64_t>(), shape_.dims().data(),
                                              indices.shape().dims().data(), shape_.rank(), dim,
                                              result.numel(), static_cast<int>(mode), stream());
                } else {
                    throw std::runtime_error("gather: unsupported dtype for CUDA");
                }
                const cudaError_t launch_error = cudaGetLastError();
                LFS_ASSERT_MSG(launch_error == cudaSuccess,
                               std::format(
                                   "gather CUDA kernel launch failed "
                                   "(cuda_error={}({}), input_shape={}, output_shape={}, "
                                   "index_shape={}, dimension={}, boundary_mode={}, stream={})",
                                   cudaGetErrorString(launch_error),
                                   static_cast<int>(launch_error), shape_.str(),
                                   result.shape().str(), indices.shape().str(), dim,
                                   static_cast<int>(mode), static_cast<const void*>(stream())));
                // No sync - tensor operation
            } else {
                const int* idx_data = is_int64 ? indices_int32.ptr<int>() : indices_same_device.ptr<int>();

                size_t outer = 1;
                for (int i = 0; i < dim; ++i) {
                    outer *= shape_[i];
                }

                size_t inner = 1;
                for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                    inner *= shape_[i];
                }

                auto process_gather = [&](auto* src, auto* dst) {
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < indices.numel(); ++i) {
                            int idx = idx_data[i];

                            if (mode == BoundaryMode::Clamp) {
                                idx = std::clamp(idx, 0, static_cast<int>(shape_[dim]) - 1);
                            } else if (mode == BoundaryMode::Wrap) {
                                idx = ((idx % static_cast<int>(shape_[dim])) + static_cast<int>(shape_[dim])) % static_cast<int>(shape_[dim]);
                            } else {
                                if (idx < 0)
                                    idx += shape_[dim];
                                if (idx < 0 || idx >= static_cast<int>(shape_[dim])) {
                                    continue;
                                }
                            }

                            size_t src_base = o * shape_[dim] * inner + idx * inner;
                            size_t dst_base = o * indices.numel() * inner + i * inner;
                            for (size_t j = 0; j < inner; ++j) {
                                dst[dst_base + j] = src[src_base + j];
                            }
                        }
                    }
                };

                // Dispatch based on dtype
                if (dtype_ == DataType::Float32) {
                    process_gather(ptr<float>(), result.ptr<float>());
                } else if (dtype_ == DataType::Int64) {
                    process_gather(ptr<int64_t>(), result.ptr<int64_t>());
                } else {
                    throw std::runtime_error("gather: unsupported dtype for CPU");
                }
            }

            return result;
        }

        LFS_ASSERT_MSG(indices.ndim() == shape_.rank(),
                       std::format("multi-dimensional gather indices must have the input rank "
                                   "(index_rank={}, input_rank={}, index_shape={}, input_shape={})",
                                   indices.ndim(), shape_.rank(),
                                   indices.shape().str(), shape_.str()));
        for (size_t d = 0; d < shape_.rank(); ++d) {
            if (d != static_cast<size_t>(dim)) {
                LFS_ASSERT_MSG(indices.shape()[d] <= shape_[d],
                               std::format("gather index shape must not exceed the input outside "
                                           "the gather dimension "
                                           "(dimension={}, gather_dimension={}, index_size={}, "
                                           "input_size={}, index_shape={}, input_shape={})",
                                           d, dim, indices.shape()[d], shape_[d],
                                           indices.shape().str(), shape_.str()));
            }
        }

        auto result = zeros(indices.shape(), device_, dtype_);
        auto indices_same_device = ensure_same_device(indices);
        const bool is_int64 = indices_same_device.dtype() == DataType::Int64;
        Tensor indices_int32;
        if (is_int64) {
            indices_int32 = indices_same_device.to(DataType::Int32);
        }
        const int* idx_ptr = is_int64 ? indices_int32.ptr<int>() : indices_same_device.ptr<int>();

        if (device_ == Device::CUDA) {
            result.set_stream(stream());
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_gather(ptr<float>(), idx_ptr,
                                          result.ptr<float>(), shape_.dims().data(),
                                          indices.shape().dims().data(), shape_.rank(), dim,
                                          result.numel(), static_cast<int>(mode), stream());
            } else {
                tensor_ops::launch_gather(ptr<int64_t>(), idx_ptr,
                                          result.ptr<int64_t>(), shape_.dims().data(),
                                          indices.shape().dims().data(), shape_.rank(), dim,
                                          result.numel(), static_cast<int>(mode), stream());
            }
            const cudaError_t launch_error = cudaGetLastError();
            LFS_ASSERT_MSG(launch_error == cudaSuccess,
                           std::format(
                               "multi-dimensional gather CUDA kernel launch failed "
                               "(cuda_error={}({}), input_shape={}, output_shape={}, "
                               "index_shape={}, dimension={}, boundary_mode={}, stream={})",
                               cudaGetErrorString(launch_error),
                               static_cast<int>(launch_error), shape_.str(),
                               result.shape().str(), indices.shape().str(), dim,
                               static_cast<int>(mode), static_cast<const void*>(stream())));
            // No sync - tensor operation
        } else {
            const int* idx_data = idx_ptr;

            size_t total_elements = indices.numel();

            auto input_strides = shape_.strides();
            auto output_strides = indices.shape().strides();

            const auto gather_values = [&](const auto* src, auto* dst) {
                for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
                    std::vector<size_t> coords(indices.shape().rank());
                    size_t temp = linear_idx;
                    for (size_t d = 0; d < indices.shape().rank(); ++d) {
                        coords[d] = temp / output_strides[d];
                        temp %= output_strides[d];
                    }

                    int idx = idx_data[linear_idx];

                    if (mode == BoundaryMode::Clamp) {
                        idx = std::clamp(idx, 0, static_cast<int>(shape_[dim]) - 1);
                    } else if (mode == BoundaryMode::Wrap) {
                        idx = ((idx % static_cast<int>(shape_[dim])) + static_cast<int>(shape_[dim])) % static_cast<int>(shape_[dim]);
                    } else {
                        if (idx < 0)
                            idx += shape_[dim];
                        if (idx < 0 || idx >= static_cast<int>(shape_[dim])) {
                            continue;
                        }
                    }

                    size_t input_linear_idx = 0;
                    for (size_t d = 0; d < shape_.rank(); ++d) {
                        size_t coord = (d == static_cast<size_t>(dim)) ? idx : coords[d];
                        input_linear_idx += coord * input_strides[d];
                    }

                    dst[linear_idx] = src[input_linear_idx];
                }
            };
            if (dtype_ == DataType::Float32) {
                gather_values(ptr<float>(), result.ptr<float>());
            } else {
                gather_values(ptr<int64_t>(), result.ptr<int64_t>());
            }
        }

        return result;
    }

    Tensor Tensor::take(const Tensor& indices) const {
        LFS_ASSERT_MSG(is_valid() && indices.is_valid(),
                       std::format("take requires valid input and index tensors "
                                   "(input={}, indices={})",
                                   str(), indices.str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       std::format("take requires Float32 input "
                                   "(input_dtype={}({}), input_shape={}, input_device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));
        LFS_ASSERT_MSG(indices.device() == device_,
                       std::format("take indices must be on the input device "
                                   "(input_device={}, index_device={}, input_shape={}, index_shape={})",
                                   device_name(device_), device_name(indices.device()),
                                   shape_.str(), indices.shape().str()));
        assert_index_tensor(indices, numel(), "take", true, true);

        auto indices_same_device = ensure_same_device(indices);
        Tensor indices_int32 = indices_same_device.dtype() == DataType::Int64
                                   ? indices_same_device.to(DataType::Int32)
                                   : indices_same_device;
        auto flat = flatten();
        Tensor result;

        // DEBUG: Log device and CUDA state
        if (device_ == Device::CUDA) {
            const cudaStream_t execution_stream =
                getCurrentCUDAStream() ? getCurrentCUDAStream() : stream();
            sync_to_stream(execution_stream);
            indices_int32.sync_to_stream(execution_stream);
            CUDAStreamGuard guard(execution_stream);
            result = empty(indices.shape(), device_, dtype_);
            tensor_ops::launch_take(flat.ptr<float>(), indices_int32.ptr<int>(),
                                    result.ptr<float>(), flat.numel(), indices_int32.numel(), result.stream());
            // No sync - tensor operation
        } else {
            result = empty(indices.shape(), device_, dtype_);
            const float* src = flat.ptr<float>();
            float* dst = result.ptr<float>();
            const int* idx = indices_int32.ptr<int>();
            size_t total = flat.numel();

            // IMPORTANT: Use sequential execution to avoid TBB threading issues with CUDA
            // TBB worker threads don't have CUDA device context, causing cudaErrorInvalidDevice
            std::transform(std::execution::seq,
                           idx, idx + indices_int32.numel(), dst,
                           [src, total](int pos) {
                               if (pos < 0)
                                   pos += total;
                               return (pos >= 0 && pos < static_cast<int>(total)) ? src[pos] : 0.0f;
                           });
        }
        return result;
    }

    // Scatter Operations
    Tensor& Tensor::scatter_(int dim, const Tensor& idx, const Tensor& src, ScatterMode mode) {
        materialize_if_deferred();
        if (mode == ScatterMode::Add) {
            return index_add_(dim, idx, src);
        }

        LFS_ASSERT_MSG(is_valid() && idx.is_valid() && src.is_valid(),
                       std::format("scatter_ requires valid destination, index, and source tensors "
                                   "(destination={}, indices={}, source={}, requested_dimension={}, "
                                   "mode={})",
                                   str(), idx.str(), src.str(), dim, static_cast<int>(mode)));
        LFS_ASSERT_MSG(idx.ndim() == 1,
                       std::format("scatter_ currently requires rank-1 indices "
                                   "(index_rank={}, index_shape={}, requested_dimension={})",
                                   idx.ndim(), idx.shape().str(), dim));
        LFS_ASSERT_MSG(idx.device() == device_ && src.device() == device_,
                       std::format("scatter_ tensors must be on the same device "
                                   "(destination_device={}, index_device={}, source_device={})",
                                   device_name(device_), device_name(idx.device()),
                                   device_name(src.device())));
        LFS_ASSERT_MSG(src.dtype() == dtype_,
                       std::format("scatter_ source dtype must match the destination "
                                   "(destination_dtype={}({}), source_dtype={}({}), "
                                   "destination_shape={}, source_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(src.dtype()), static_cast<int>(src.dtype()),
                                   shape_.str(), src.shape().str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32 ||
                           dtype_ == DataType::Bool || dtype_ == DataType::UInt8,
                       std::format("scatter_ destination dtype must be supported "
                                   "(destination_dtype={}({}), valid_dtypes=[float32,int32,bool,uint8], "
                                   "destination_shape={}, destination_device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));
        LFS_ASSERT_MSG(device_ != Device::CUDA || mode == ScatterMode::None,
                       std::format("CUDA scatter_ supports assignment mode only; use index_add_ "
                                   "for addition (destination_device={}, mode={}({}), "
                                   "destination_shape={})",
                                   device_name(device_), static_cast<int>(mode),
                                   static_cast<int>(mode), shape_.str()));

        const int requested_dim = dim;
        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(shape_.rank()),
                       std::format("scatter_ dimension must be in range "
                                   "(requested_dimension={}, resolved_dimension={}, "
                                   "valid_range=[0,{}), destination_shape={})",
                                   requested_dim, dim, shape_.rank(), shape_.str()));
        assert_index_tensor(idx, shape_[dim], "scatter_", true, true);

        if (shape_.rank() == 1 && dim == 0) {
            LFS_ASSERT_MSG(src.ndim() == 1,
                           std::format("rank-1 scatter_ requires a rank-1 source "
                                       "(source_rank={}, source_shape={}, destination_shape={})",
                                       src.ndim(), src.shape().str(), shape_.str()));
            LFS_ASSERT_MSG(idx.numel() == src.numel(),
                           std::format("rank-1 scatter_ index and source lengths must match "
                                       "(index_count={}, source_numel={}, index_shape={}, "
                                       "source_shape={})",
                                       idx.numel(), src.numel(),
                                       idx.shape().str(), src.shape().str()));

            auto indices_same_device = ensure_same_device(idx);
            auto src_same_device = ensure_same_device(src);
            const bool is_int64 = indices_same_device.dtype() == DataType::Int64;
            Tensor indices_int32;
            if (is_int64) {
                indices_int32 = indices_same_device.to(DataType::Int32);
            }

            const int* indices = is_int64 ? indices_int32.ptr<int>() : indices_same_device.ptr<int>();

            if (device_ == Device::CUDA) {
                if (dtype_ == DataType::Float32) {
                    tensor_ops::launch_scatter(ptr<float>(), indices, src_same_device.ptr<float>(),
                                               shape_.dims().data(), src.shape().dims().data(),
                                               shape_.rank(), dim, src.numel(),
                                               static_cast<int>(mode), stream());
                } else if (dtype_ == DataType::Int32) {
                    tensor_ops::launch_scatter(ptr<int>(), indices, src_same_device.ptr<int>(),
                                               shape_.dims().data(), src.shape().dims().data(),
                                               shape_.rank(), dim, src.numel(),
                                               static_cast<int>(mode), stream());
                } else if (dtype_ == DataType::Bool || dtype_ == DataType::UInt8) {
                    tensor_ops::launch_scatter(ptr<uint8_t>(), indices, src_same_device.ptr<uint8_t>(),
                                               shape_.dims().data(), src.shape().dims().data(),
                                               shape_.rank(), dim, src.numel(),
                                               static_cast<int>(mode), stream());
                } else {
                    LFS_ASSERT_MSG(false,
                                   std::format("scatter_ CUDA dispatch reached an unsupported dtype "
                                               "(dtype={}({}), destination_shape={}, source_shape={}, "
                                               "dimension={}, mode={})",
                                               dtype_name(dtype_), static_cast<int>(dtype_),
                                               shape_.str(), src.shape().str(), dim,
                                               static_cast<int>(mode)));
                }
            } else {
                const auto scatter_1d = [&](auto* dst, const auto* src_data) {
                    for (size_t i = 0; i < idx.numel(); ++i) {
                        int pos = indices[i];
                        if (pos < 0)
                            pos += static_cast<int>(shape_[0]);
                        if (pos >= 0 && pos < static_cast<int>(shape_[0])) {
                            switch (mode) {
                            case ScatterMode::Multiply:
                                dst[pos] *= src_data[i];
                                break;
                            case ScatterMode::Max:
                                dst[pos] = std::max(dst[pos], src_data[i]);
                                break;
                            case ScatterMode::Min:
                                dst[pos] = std::min(dst[pos], src_data[i]);
                                break;
                            default:
                                dst[pos] = src_data[i];
                                break;
                            }
                        }
                    }
                };

                if (dtype_ == DataType::Float32) {
                    scatter_1d(ptr<float>(), src_same_device.ptr<float>());
                } else if (dtype_ == DataType::Int32) {
                    scatter_1d(ptr<int>(), src_same_device.ptr<int>());
                } else if (dtype_ == DataType::Bool || dtype_ == DataType::UInt8) {
                    scatter_1d(ptr<unsigned char>(), src_same_device.ptr<unsigned char>());
                } else {
                    LFS_ASSERT_MSG(false,
                                   std::format("scatter_ CPU dispatch reached an unsupported dtype "
                                               "(dtype={}({}), destination_shape={}, source_shape={}, "
                                               "dimension={}, mode={})",
                                               dtype_name(dtype_), static_cast<int>(dtype_),
                                               shape_.str(), src.shape().str(), dim,
                                               static_cast<int>(mode)));
                }
            }

            return *this;
        }

        std::vector<size_t> expected_shape = shape_.dims();
        expected_shape[dim] = idx.numel();

        LFS_ASSERT_MSG(src.shape() == TensorShape(expected_shape),
                       std::format("scatter_ source shape mismatch: expected {}, got {}",
                                   TensorShape(expected_shape).str(), src.shape().str()));

        auto idx_same_device = ensure_same_device(idx);
        auto src_same_device = ensure_same_device(src);

        const bool is_int64 = idx_same_device.dtype() == DataType::Int64;
        Tensor idx_int32;
        if (is_int64) {
            idx_int32 = idx_same_device.to(DataType::Int32);
        }
        const int* idx_ptr = is_int64 ? idx_int32.ptr<int>() : idx_same_device.ptr<int>();

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_scatter(ptr<float>(), idx_ptr,
                                           src_same_device.ptr<float>(), shape_.dims().data(),
                                           src.shape().dims().data(),
                                           shape_.rank(), dim, src.numel(),
                                           static_cast<int>(mode), stream());
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_scatter(ptr<int>(), idx_ptr,
                                           src_same_device.ptr<int>(), shape_.dims().data(),
                                           src.shape().dims().data(),
                                           shape_.rank(), dim, src.numel(),
                                           static_cast<int>(mode), stream());
            } else if (dtype_ == DataType::Bool || dtype_ == DataType::UInt8) {
                tensor_ops::launch_scatter(ptr<uint8_t>(), idx_ptr,
                                           src_same_device.ptr<uint8_t>(), shape_.dims().data(),
                                           src.shape().dims().data(),
                                           shape_.rank(), dim, src.numel(),
                                           static_cast<int>(mode), stream());
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("scatter_ CUDA dispatch reached an unsupported dtype "
                                           "(dtype={}({}), destination_shape={}, source_shape={}, "
                                           "dimension={}, mode={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), src.shape().str(), dim,
                                           static_cast<int>(mode)));
            }
        } else {
            size_t outer = 1;
            for (int i = 0; i < dim; ++i) {
                outer *= shape_[i];
            }

            size_t inner = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner *= shape_[i];
            }

            const int* indices = idx_ptr;

            const auto scatter_nd = [&](auto* dst, const auto* src_data) {
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t i = 0; i < idx.numel(); ++i) {
                        int pos = indices[i];

                        if (pos < 0)
                            pos += static_cast<int>(shape_[dim]);

                        if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                            continue;
                        }

                        size_t src_base = o * idx.numel() * inner + i * inner;
                        size_t dst_base = o * shape_[dim] * inner + pos * inner;

                        for (size_t j = 0; j < inner; ++j) {
                            size_t src_idx = src_base + j;
                            size_t dst_idx = dst_base + j;

                            if (src_idx >= src.numel() || dst_idx >= numel()) {
                                LFS_ASSERT_MSG(false,
                                               std::format("scatter_ computed source or destination "
                                                           "offset outside its allocation "
                                                           "(source_offset={}, source_numel={}, "
                                                           "destination_offset={}, destination_numel={}, "
                                                           "outer_index={}, index_position={}, "
                                                           "inner_index={}, dimension={}, mode={})",
                                                           src_idx, src.numel(), dst_idx, numel(),
                                                           o, i, j, dim, static_cast<int>(mode)));
                            }

                            switch (mode) {
                            case ScatterMode::Add:
                                dst[dst_idx] += src_data[src_idx];
                                break;
                            case ScatterMode::Multiply:
                                dst[dst_idx] *= src_data[src_idx];
                                break;
                            case ScatterMode::Max:
                                dst[dst_idx] = std::max(dst[dst_idx], src_data[src_idx]);
                                break;
                            case ScatterMode::Min:
                                dst[dst_idx] = std::min(dst[dst_idx], src_data[src_idx]);
                                break;
                            default:
                                dst[dst_idx] = src_data[src_idx];
                                break;
                            }
                        }
                    }
                }
                return true;
            };

            if (dtype_ == DataType::Float32) {
                if (!scatter_nd(ptr<float>(), src_same_device.ptr<float>()))
                    return *this;
            } else if (dtype_ == DataType::Int32) {
                if (!scatter_nd(ptr<int>(), src_same_device.ptr<int>()))
                    return *this;
            } else if (dtype_ == DataType::Bool || dtype_ == DataType::UInt8) {
                if (!scatter_nd(ptr<unsigned char>(), src_same_device.ptr<unsigned char>()))
                    return *this;
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("scatter_ CPU dispatch reached an unsupported dtype "
                                           "(dtype={}({}), destination_shape={}, source_shape={}, "
                                           "dimension={}, mode={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), src.shape().str(), dim,
                                           static_cast<int>(mode)));
            }
        }

        return *this;
    }

    Tensor& Tensor::scatter_(int dim, const Tensor& idx, float val, ScatterMode mode) {
        LFS_ASSERT_MSG(is_valid() && idx.is_valid(),
                       std::format("scalar scatter_ requires valid destination and index tensors "
                                   "(destination={}, indices={}, requested_dimension={}, "
                                   "value={}, mode={})",
                                   str(), idx.str(), dim, val, static_cast<int>(mode)));
        const int resolved_dim = resolve_dim(dim);
        LFS_ASSERT_MSG(resolved_dim >= 0 && resolved_dim < static_cast<int>(shape_.rank()),
                       std::format("scalar scatter_ dimension must be in range "
                                   "(requested_dimension={}, resolved_dimension={}, "
                                   "valid_range=[0,{}), destination_shape={}, value={}, mode={})",
                                   dim, resolved_dim, shape_.rank(), shape_.str(),
                                   val, static_cast<int>(mode)));
        std::vector<size_t> src_shape = shape_.dims();
        src_shape[resolved_dim] = idx.numel();
        auto src = full(TensorShape(src_shape), val, device_, dtype_);
        return scatter_(dim, idx, src, mode);
    }

    Tensor& Tensor::index_fill_(int dim, const Tensor& idx, float val) {
        return scatter_(dim, idx, val, ScatterMode::None);
    }

    Tensor& Tensor::index_copy_(int dim, const Tensor& idx, const Tensor& src) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && idx.is_valid() && src.is_valid(),
                       std::format("index_copy_ requires valid destination, index, and source tensors "
                                   "(destination={}, indices={}, source={}, requested_dimension={})",
                                   str(), idx.str(), src.str(), dim));
        LFS_ASSERT_MSG(idx.ndim() == 1,
                       std::format("index_copy_ requires rank-1 indices "
                                   "(index_rank={}, index_shape={}, requested_dimension={})",
                                   idx.ndim(), idx.shape().str(), dim));
        LFS_ASSERT_MSG(idx.device() == device_ && src.device() == device_,
                       std::format("index_copy_ tensors must be on the same device "
                                   "(destination_device={}, index_device={}, source_device={})",
                                   device_name(device_), device_name(idx.device()),
                                   device_name(src.device())));
        LFS_ASSERT_MSG(src.dtype() == dtype_,
                       std::format("index_copy_ source dtype must match the destination "
                                   "(destination_dtype={}({}), source_dtype={}({}), "
                                   "destination_shape={}, source_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(src.dtype()), static_cast<int>(src.dtype()),
                                   shape_.str(), src.shape().str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32 ||
                           dtype_ == DataType::Bool || dtype_ == DataType::UInt8,
                       std::format("index_copy_ destination dtype must be supported "
                                   "(destination_dtype={}({}), valid_dtypes=[float32,int32,bool,uint8], "
                                   "destination_shape={}, destination_device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));

        const int requested_dim = dim;
        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(shape_.rank()),
                       std::format("index_copy_ dimension must be in range "
                                   "(requested_dimension={}, resolved_dimension={}, "
                                   "valid_range=[0,{}), destination_shape={})",
                                   requested_dim, dim, shape_.rank(), shape_.str()));
        assert_index_tensor(idx, shape_[dim], "index_copy_", true);

        std::vector<size_t> expected_src_shape = shape_.dims();
        expected_src_shape[dim] = idx.numel();

        LFS_ASSERT_MSG(src.shape() == TensorShape(expected_src_shape),
                       std::format("index_copy_ source shape must match indices and destination "
                                   "(source_shape={}, expected_shape={}, index_count={}, "
                                   "dimension={}, destination_shape={})",
                                   src.shape().str(), TensorShape(expected_src_shape).str(),
                                   idx.numel(), dim, shape_.str()));

        auto idx_same_device = ensure_same_device(idx);
        auto src_same_device = ensure_same_device(src);

        const bool is_int64 = idx_same_device.dtype() == DataType::Int64;
        Tensor idx_int32;
        if (is_int64) {
            idx_int32 = idx_same_device.to(DataType::Int32);
        }
        const int* idx_ptr = is_int64 ? idx_int32.ptr<int>() : idx_same_device.ptr<int>();

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_index_copy(ptr<float>(), idx_ptr,
                                              src_same_device.ptr<float>(), shape_.dims().data(),
                                              shape_.rank(), dim, idx.numel(), stream());
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_index_copy(ptr<int>(), idx_ptr,
                                              src_same_device.ptr<int>(), shape_.dims().data(),
                                              shape_.rank(), dim, idx.numel(), stream());
            } else if (dtype_ == DataType::Bool || dtype_ == DataType::UInt8) {
                tensor_ops::launch_index_copy(ptr<uint8_t>(), idx_ptr,
                                              src_same_device.ptr<uint8_t>(), shape_.dims().data(),
                                              shape_.rank(), dim, idx.numel(), stream());
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("index_copy_ CUDA dispatch reached an unsupported dtype "
                                           "(dtype={}({}), destination_shape={}, source_shape={}, "
                                           "dimension={}, index_count={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), src.shape().str(), dim, idx.numel()));
            }
        } else {
            size_t outer = 1, inner = 1;
            for (int i = 0; i < dim; ++i)
                outer *= shape_[i];
            for (size_t i = dim + 1; i < shape_.rank(); ++i)
                inner *= shape_[i];

            const int* indices = idx_ptr;
            const auto index_copy = [&](auto* dst, const auto* src_data) {
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t i = 0; i < idx.numel(); ++i) {
                        int pos = indices[i];
                        if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                            LFS_ASSERT_MSG(false,
                                           std::format("index_copy_ index {} is out of bounds for dimension {} of size {}",
                                                       pos, dim, shape_[dim]));
                        }

                        for (size_t j = 0; j < inner; ++j) {
                            size_t src_idx = o * idx.numel() * inner + i * inner + j;
                            size_t dst_idx = o * shape_[dim] * inner + pos * inner + j;

                            if (src_idx < src.numel() && dst_idx < numel()) {
                                dst[dst_idx] = src_data[src_idx];
                            }
                        }
                    }
                }
            };

            if (dtype_ == DataType::Float32) {
                index_copy(ptr<float>(), src_same_device.ptr<float>());
            } else if (dtype_ == DataType::Int32) {
                index_copy(ptr<int>(), src_same_device.ptr<int>());
            } else if (dtype_ == DataType::Bool || dtype_ == DataType::UInt8) {
                index_copy(ptr<unsigned char>(), src_same_device.ptr<unsigned char>());
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("index_copy_ CPU dispatch reached an unsupported dtype "
                                           "(dtype={}({}), destination_shape={}, source_shape={}, "
                                           "dimension={}, index_count={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), src.shape().str(), dim, idx.numel()));
            }
        }

        return *this;
    }

    Tensor& Tensor::index_add_(int dim, const Tensor& idx, const Tensor& src) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && idx.is_valid() && src.is_valid(),
                       std::format("index_add_ requires valid destination, index, and source tensors "
                                   "(destination={}, indices={}, source={}, requested_dimension={})",
                                   str(), idx.str(), src.str(), dim));
        LFS_ASSERT_MSG(idx.ndim() == 1,
                       std::format("index_add_ requires rank-1 indices "
                                   "(index_rank={}, index_shape={}, requested_dimension={})",
                                   idx.ndim(), idx.shape().str(), dim));
        LFS_ASSERT_MSG(idx.device() == device_ && src.device() == device_,
                       std::format("index_add_ tensors must be on the same device "
                                   "(destination_device={}, index_device={}, source_device={})",
                                   device_name(device_), device_name(idx.device()),
                                   device_name(src.device())));
        LFS_ASSERT_MSG(src.dtype() == dtype_,
                       std::format("index_add_ source dtype must match the destination "
                                   "(destination_dtype={}({}), source_dtype={}({}), "
                                   "destination_shape={}, source_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(src.dtype()), static_cast<int>(src.dtype()),
                                   shape_.str(), src.shape().str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Int32,
                       std::format("index_add_ requires Float32 or Int32 destination "
                                   "(destination_dtype={}({}), destination_shape={}, "
                                   "destination_device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));

        const int requested_dim = dim;
        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(shape_.rank()),
                       std::format("index_add_ dimension must be in range "
                                   "(requested_dimension={}, resolved_dimension={}, "
                                   "valid_range=[0,{}), destination_shape={})",
                                   requested_dim, dim, shape_.rank(), shape_.str()));
        assert_index_tensor(idx, shape_[dim], "index_add_", true);

        if (shape_.rank() == 1 && dim == 0) {
            LFS_ASSERT_MSG(src.ndim() == 1 && src.numel() == idx.numel(),
                           std::format("rank-1 index_add_ source must be rank 1 and match the "
                                       "index length (source_rank={}, source_numel={}, "
                                       "index_count={}, source_shape={}, index_shape={})",
                                       src.ndim(), src.numel(), idx.numel(),
                                       src.shape().str(), idx.shape().str()));

            auto idx_same_device = ensure_same_device(idx);
            auto src_same_device = ensure_same_device(src);

            if (device_ == Device::CUDA) {
                // Convert int64 indices to int32 for kernel (kernel expects int* not int64_t*)
                auto idx_int32 = (idx_same_device.dtype() == DataType::Int64)
                                     ? idx_same_device.to(DataType::Int32)
                                     : idx_same_device;

                // Dispatch based on data type
                if (dtype_ == DataType::Float32) {
                    tensor_ops::launch_index_add<float>(ptr<float>(), idx_int32.ptr<int>(),
                                                        src_same_device.ptr<float>(), shape_.dims().data(),
                                                        shape_.rank(), dim, idx.numel(), stream());
                } else if (dtype_ == DataType::Int32) {
                    tensor_ops::launch_index_add<int>(ptr<int>(), idx_int32.ptr<int>(),
                                                      src_same_device.ptr<int>(), shape_.dims().data(),
                                                      shape_.rank(), dim, idx.numel(), stream());
                } else {
                    LFS_ASSERT_MSG(false,
                                   std::format("rank-1 index_add_ CUDA dispatch reached an "
                                               "unsupported dtype "
                                               "(dtype={}({}), destination_shape={}, source_shape={}, "
                                               "index_count={})",
                                               dtype_name(dtype_), static_cast<int>(dtype_),
                                               shape_.str(), src.shape().str(), idx.numel()));
                }
                // No sync - tensor operation
            } else {
                // CPU path - dispatch based on data type
                if (dtype_ == DataType::Float32) {
                    float* data = ptr<float>();
                    const float* src_data = src_same_device.ptr<float>();

                    // Handle int64 indices correctly
                    if (idx_same_device.dtype() == DataType::Int64) {
                        const int64_t* indices = idx_same_device.ptr<int64_t>();
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int64_t pos = indices[i];
                            if (pos < 0)
                                pos += shape_[0];
                            if (pos >= 0 && pos < static_cast<int64_t>(shape_[0])) {
                                data[pos] += src_data[i];
                            }
                        }
                    } else {
                        const int* indices = idx_same_device.ptr<int>();
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int pos = indices[i];
                            if (pos < 0)
                                pos += shape_[0];
                            if (pos >= 0 && pos < static_cast<int>(shape_[0])) {
                                data[pos] += src_data[i];
                            }
                        }
                    }
                } else if (dtype_ == DataType::Int32) {
                    int* data = ptr<int>();
                    const int* src_data = src_same_device.ptr<int>();

                    // Handle int64 indices correctly
                    if (idx_same_device.dtype() == DataType::Int64) {
                        const int64_t* indices = idx_same_device.ptr<int64_t>();
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int64_t pos = indices[i];
                            if (pos < 0)
                                pos += shape_[0];
                            if (pos >= 0 && pos < static_cast<int64_t>(shape_[0])) {
                                data[pos] += src_data[i];
                            }
                        }
                    } else {
                        const int* indices = idx_same_device.ptr<int>();
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int pos = indices[i];
                            if (pos < 0)
                                pos += shape_[0];
                            if (pos >= 0 && pos < static_cast<int>(shape_[0])) {
                                data[pos] += src_data[i];
                            }
                        }
                    }
                } else {
                    LFS_ASSERT_MSG(false,
                                   std::format("rank-1 index_add_ CPU dispatch reached an "
                                               "unsupported dtype "
                                               "(dtype={}({}), destination_shape={}, source_shape={}, "
                                               "index_count={})",
                                               dtype_name(dtype_), static_cast<int>(dtype_),
                                               shape_.str(), src.shape().str(), idx.numel()));
                }
            }
            return *this;
        }

        std::vector<size_t> expected_shape = shape_.dims();
        expected_shape[dim] = idx.numel();

        LFS_ASSERT_MSG(src.shape() == TensorShape(expected_shape),
                       std::format("index_add_ source shape mismatch: expected {}, got {}",
                                   TensorShape(expected_shape).str(), src.shape().str()));

        auto idx_same_device = ensure_same_device(idx);
        auto src_same_device = ensure_same_device(src);

        if (device_ == Device::CUDA) {
            // Convert int64 indices to int32 for kernel (kernel expects int* not int64_t*)
            auto idx_int32 = (idx_same_device.dtype() == DataType::Int64)
                                 ? idx_same_device.to(DataType::Int32)
                                 : idx_same_device;

            // Dispatch based on data type
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_index_add<float>(ptr<float>(), idx_int32.ptr<int>(),
                                                    src_same_device.ptr<float>(), shape_.dims().data(),
                                                    shape_.rank(), dim, idx.numel(), stream());
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_index_add<int>(ptr<int>(), idx_int32.ptr<int>(),
                                                  src_same_device.ptr<int>(), shape_.dims().data(),
                                                  shape_.rank(), dim, idx.numel(), stream());
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("index_add_ CUDA dispatch reached an unsupported dtype "
                                           "(dtype={}({}), destination_shape={}, source_shape={}, "
                                           "dimension={}, index_count={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), src.shape().str(), dim, idx.numel()));
            }
            // No sync - tensor operation
        } else {
            size_t outer = 1;
            for (int i = 0; i < dim; ++i) {
                outer *= shape_[i];
            }

            size_t inner = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner *= shape_[i];
            }

            // CPU path - dispatch based on data type
            if (dtype_ == DataType::Float32) {
                float* data = ptr<float>();
                const float* src_data = src_same_device.ptr<float>();

                // Handle int64 indices correctly
                if (idx_same_device.dtype() == DataType::Int64) {
                    const int64_t* indices = idx_same_device.ptr<int64_t>();
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int64_t pos = indices[i];

                            if (pos < 0)
                                pos += static_cast<int64_t>(shape_[dim]);

                            if (pos < 0 || pos >= static_cast<int64_t>(shape_[dim])) {
                                continue;
                            }

                            size_t src_base = o * idx.numel() * inner + i * inner;
                            size_t dst_base = o * shape_[dim] * inner + pos * inner;

                            for (size_t j = 0; j < inner; ++j) {
                                data[dst_base + j] += src_data[src_base + j];
                            }
                        }
                    }
                } else {
                    const int* indices = idx_same_device.ptr<int>();
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int pos = indices[i];

                            if (pos < 0)
                                pos += static_cast<int>(shape_[dim]);

                            if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                                continue;
                            }

                            size_t src_base = o * idx.numel() * inner + i * inner;
                            size_t dst_base = o * shape_[dim] * inner + pos * inner;

                            for (size_t j = 0; j < inner; ++j) {
                                data[dst_base + j] += src_data[src_base + j];
                            }
                        }
                    }
                }
            } else if (dtype_ == DataType::Int32) {
                int* data = ptr<int>();
                const int* src_data = src_same_device.ptr<int>();

                // Handle int64 indices correctly
                if (idx_same_device.dtype() == DataType::Int64) {
                    const int64_t* indices = idx_same_device.ptr<int64_t>();
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int64_t pos = indices[i];

                            if (pos < 0)
                                pos += static_cast<int64_t>(shape_[dim]);

                            if (pos < 0 || pos >= static_cast<int64_t>(shape_[dim])) {
                                continue;
                            }

                            size_t src_base = o * idx.numel() * inner + i * inner;
                            size_t dst_base = o * shape_[dim] * inner + pos * inner;

                            for (size_t j = 0; j < inner; ++j) {
                                data[dst_base + j] += src_data[src_base + j];
                            }
                        }
                    }
                } else {
                    const int* indices = idx_same_device.ptr<int>();
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < idx.numel(); ++i) {
                            int pos = indices[i];

                            if (pos < 0)
                                pos += static_cast<int>(shape_[dim]);

                            if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                                continue;
                            }

                            size_t src_base = o * idx.numel() * inner + i * inner;
                            size_t dst_base = o * shape_[dim] * inner + pos * inner;

                            for (size_t j = 0; j < inner; ++j) {
                                data[dst_base + j] += src_data[src_base + j];
                            }
                        }
                    }
                }
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("index_add_ CPU dispatch reached an unsupported dtype "
                                           "(dtype={}({}), destination_shape={}, source_shape={}, "
                                           "dimension={}, index_count={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), src.shape().str(), dim, idx.numel()));
            }
        }

        return *this;
    }

    Tensor& Tensor::index_put_(const Tensor& idx, const Tensor& vals) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && idx.is_valid() && vals.is_valid(),
                       std::format("index_put_ requires valid destination, index, and value tensors "
                                   "(destination={}, indices={}, values={})",
                                   str(), idx.str(), vals.str()));
        LFS_ASSERT_MSG(idx.ndim() == 1,
                       std::format("index_put_ requires rank-1 indices "
                                   "(index_rank={}, index_shape={}, destination_shape={})",
                                   idx.ndim(), idx.shape().str(), shape_.str()));
        LFS_ASSERT_MSG(idx.device() == device_ && vals.device() == device_,
                       std::format("index_put_ tensors must be on the same device "
                                   "(destination_device={}, index_device={}, value_device={})",
                                   device_name(device_), device_name(idx.device()),
                                   device_name(vals.device())));
        LFS_ASSERT_MSG(vals.dtype() == dtype_,
                       std::format("index_put_ value dtype must match the destination "
                                   "(destination_dtype={}({}), value_dtype={}({}), "
                                   "destination_shape={}, value_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   dtype_name(vals.dtype()), static_cast<int>(vals.dtype()),
                                   shape_.str(), vals.shape().str()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::Bool ||
                           dtype_ == DataType::Int32 || dtype_ == DataType::Int64,
                       std::format("index_put_ destination dtype must be supported "
                                   "(destination_dtype={}({}), valid_dtypes=[float32,bool,int32,int64], "
                                   "destination_shape={}, destination_device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));
        LFS_ASSERT_MSG(is_integer_index_dtype(idx.dtype()),
                       std::format("index_put_ indices must be Int32 or Int64 "
                                   "(index_dtype={}({}), index_shape={}, index_device={})",
                                   dtype_name(idx.dtype()), static_cast<int>(idx.dtype()),
                                   idx.shape().str(), device_name(idx.device())));

        // No-op for zero-element tensors
        if (idx.numel() == 0 || vals.numel() == 0)
            return *this;

        auto idx_same_device = ensure_same_device(idx);
        auto vals_same_device = ensure_same_device(vals);

        // Check if this is row-wise assignment (idx is 1D, vals is multi-dimensional)
        // Example: tensor[indices] = values where tensor:[N,M], indices:[K], values:[K,M]
        const bool is_row_assignment = (idx_same_device.ndim() == 1 && vals_same_device.ndim() >= 2 && ndim() >= 2);
        if (is_row_assignment) {
            std::vector<size_t> expected_shape = shape_.dims();
            expected_shape[0] = idx.numel();
            LFS_ASSERT_MSG(vals.shape() == TensorShape(expected_shape),
                           std::format("index_put_ row values must match the indexed destination rows "
                                       "(value_shape={}, expected_shape={}, index_count={}, "
                                       "destination_shape={})",
                                       vals.shape().str(), TensorShape(expected_shape).str(),
                                       idx.numel(), shape_.str()));
            assert_index_tensor(idx, shape_[0], "index_put_", true);
        } else {
            LFS_ASSERT_MSG(vals.numel() == idx.numel(),
                           std::format("index_put_ flat assignment requires one value per index "
                                       "(value_count={}, index_count={}, value_shape={}, index_shape={})",
                                       vals.numel(), idx.numel(),
                                       vals.shape().str(), idx.shape().str()));
            assert_index_tensor(idx, numel(), "index_put_", true, true);
        }

        // Fast path: use GPU kernel for row assignment on CUDA (avoids CPU roundtrip)
        if (device_ == Device::CUDA && is_row_assignment && dtype_ == DataType::Float32) {
            // Verify shape compatibility: vals should be [K, d1, d2, ...]
            std::vector<size_t> expected_shape = shape_.dims();
            expected_shape[0] = idx_same_device.numel();
            if (vals_same_device.shape() == TensorShape(expected_shape)) {
                // Convert indices to Int32 if needed (index_copy_ requires Int32)
                Tensor idx_int32 = (idx_same_device.dtype() == DataType::Int32)
                                       ? idx_same_device
                                       : idx_same_device.to(DataType::Int32);
                tensor_ops::launch_index_copy(ptr<float>(), idx_int32.ptr<int>(),
                                              vals_same_device.ptr<float>(), shape_.dims().data(),
                                              shape_.rank(), 0, idx_int32.numel(), stream());
                return *this;
            }
        }

        // Helper lambda for index_put_ implementation (fallback path)
        auto index_put_impl = [&]<typename DataT, typename IndexT>() {
            if (device_ == Device::CUDA) {
                // Fallback: CPU roundtrip for complex cases
                auto cpu_tensor = to(Device::CPU);
                auto cpu_idx = idx_same_device.to(Device::CPU);
                auto cpu_vals = vals_same_device.to(Device::CPU);

                DataT* data = cpu_tensor.ptr<DataT>();
                const IndexT* indices = cpu_idx.ptr<IndexT>();
                const DataT* values = cpu_vals.ptr<DataT>();

                if (is_row_assignment) {
                    size_t row_size = 1;
                    for (size_t i = 1; i < cpu_tensor.ndim(); ++i) {
                        row_size *= cpu_tensor.shape()[i];
                    }
                    const size_t num_rows = cpu_tensor.shape()[0];

                    for (size_t i = 0; i < cpu_idx.numel(); ++i) {
                        IndexT row_idx = indices[i];
                        if (row_idx < 0)
                            row_idx += num_rows;
                        if (row_idx >= 0 && row_idx < static_cast<IndexT>(num_rows)) {
                            std::memcpy(data + row_idx * row_size,
                                        values + i * row_size,
                                        row_size * sizeof(DataT));
                        }
                    }
                } else {
                    const size_t num_elements = cpu_tensor.numel();
                    for (size_t i = 0; i < cpu_idx.numel(); ++i) {
                        IndexT pos = indices[i];
                        if (pos < 0)
                            pos += num_elements;
                        if (pos >= 0 && pos < static_cast<IndexT>(num_elements)) {
                            data[pos] = values[i];
                        }
                    }
                }

                // Copy back preserving capacity
                auto result = cpu_tensor.to(device_);
                const size_t bytes = numel() * dtype_size(dtype_);
                CHECK_CUDA(cudaMemcpyAsync(data_, result.ptr<void>(), bytes, cudaMemcpyDeviceToDevice, stream()));
                CHECK_CUDA(cudaStreamSynchronize(stream()));
            } else {
                // CPU implementation
                DataT* data = ptr<DataT>();
                const IndexT* indices = idx_same_device.ptr<IndexT>();
                const DataT* values = vals_same_device.ptr<DataT>();

                if (is_row_assignment) {
                    // Row-wise assignment
                    size_t row_size = 1;
                    for (size_t i = 1; i < ndim(); ++i) {
                        row_size *= shape()[i];
                    }
                    size_t num_rows = shape()[0];

                    for (size_t i = 0; i < idx_same_device.numel(); ++i) {
                        IndexT row_idx = indices[i];
                        if (row_idx < 0)
                            row_idx += num_rows;
                        if (row_idx >= 0 && row_idx < static_cast<IndexT>(num_rows)) {
                            // Copy entire row
                            std::memcpy(data + row_idx * row_size,
                                        values + i * row_size,
                                        row_size * sizeof(DataT));
                        }
                    }
                } else {
                    // Element-wise assignment
                    size_t num_elements = numel();
                    std::for_each(std::execution::seq,
                                  std::views::iota(size_t(0), idx.numel()).begin(),
                                  std::views::iota(size_t(0), idx.numel()).end(),
                                  [data, indices, values, num_elements](size_t i) {
                                      IndexT pos = indices[i];
                                      if (pos < 0)
                                          pos += num_elements;
                                      if (pos >= 0 && pos < static_cast<IndexT>(num_elements)) {
                                          data[pos] = values[i];
                                      }
                                  });
                }
            }
        };

        // Dispatch based on data dtype and index dtype
        if (idx_same_device.dtype() == DataType::Int32) {
            if (dtype_ == DataType::Float32) {
                index_put_impl.template operator()<float, int>();
            } else if (dtype_ == DataType::Bool) {
                index_put_impl.template operator()<unsigned char, int>();
            } else if (dtype_ == DataType::Int32) {
                index_put_impl.template operator()<int, int>();
            } else if (dtype_ == DataType::Int64) {
                index_put_impl.template operator()<int64_t, int>();
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("index_put_ Int32-index dispatch reached an unsupported "
                                           "data dtype (data_dtype={}({}), index_dtype={}({}), "
                                           "destination_shape={}, value_shape={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           dtype_name(idx_same_device.dtype()),
                                           static_cast<int>(idx_same_device.dtype()),
                                           shape_.str(), vals.shape().str()));
            }
        } else if (idx_same_device.dtype() == DataType::Int64) {
            if (dtype_ == DataType::Float32) {
                index_put_impl.template operator()<float, int64_t>();
            } else if (dtype_ == DataType::Bool) {
                index_put_impl.template operator()<unsigned char, int64_t>();
            } else if (dtype_ == DataType::Int32) {
                index_put_impl.template operator()<int, int64_t>();
            } else if (dtype_ == DataType::Int64) {
                index_put_impl.template operator()<int64_t, int64_t>();
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("index_put_ Int64-index dispatch reached an unsupported "
                                           "data dtype (data_dtype={}({}), index_dtype={}({}), "
                                           "destination_shape={}, value_shape={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           dtype_name(idx_same_device.dtype()),
                                           static_cast<int>(idx_same_device.dtype()),
                                           shape_.str(), vals.shape().str()));
            }
        } else {
            LFS_ASSERT_MSG(false,
                           std::format("index_put_ dispatch requires Int32 or Int64 indices "
                                       "(index_dtype={}({}), index_shape={}, destination_shape={})",
                                       dtype_name(idx_same_device.dtype()),
                                       static_cast<int>(idx_same_device.dtype()),
                                       idx_same_device.shape().str(), shape_.str()));
        }

        return *this;
    }

    Tensor& Tensor::index_put_(const std::vector<Tensor>& indices, const Tensor& vals) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && vals.is_valid(),
                       std::format("multi-index index_put_ requires valid destination and value tensors "
                                   "(destination={}, values={}, index_tensor_count={})",
                                   str(), vals.str(), indices.size()));
        LFS_ASSERT_MSG(!indices.empty(),
                       std::format("multi-index index_put_ requires at least one index tensor "
                                   "(index_tensor_count={}, destination_shape={}, value_shape={})",
                                   indices.size(), shape_.str(), vals.shape().str()));
        LFS_ASSERT_MSG(vals.device() == device_,
                       std::format("multi-index index_put_ values must be on the destination device "
                                   "(destination_device={}, value_device={}, "
                                   "destination_shape={}, value_shape={})",
                                   device_name(device_), device_name(vals.device()),
                                   shape_.str(), vals.shape().str()));

        // No-op for zero-element tensors
        if (vals.numel() == 0)
            return *this;

        if (indices.size() == 1) {
            return index_put_(indices[0], vals);
        }

        if (indices.size() == 2 && shape_.rank() == 2) {
            LFS_ASSERT_MSG(indices[0].is_valid() && indices[1].is_valid(),
                           std::format("multi-index index_put_ requires valid row and column indices "
                                       "(row_indices={}, column_indices={})",
                                       indices[0].str(), indices[1].str()));
            LFS_ASSERT_MSG(indices[0].device() == device_ && indices[1].device() == device_,
                           std::format("multi-index index_put_ tensors must be on the same device "
                                       "(destination_device={}, row_index_device={}, "
                                       "column_index_device={}, value_device={})",
                                       device_name(device_), device_name(indices[0].device()),
                                       device_name(indices[1].device()), device_name(vals.device())));
            LFS_ASSERT_MSG(dtype_ == DataType::Float32 && vals.dtype() == DataType::Float32,
                           std::format("multi-index index_put_ currently requires Float32 values "
                                       "(destination_dtype={}({}), value_dtype={}({}), "
                                       "destination_shape={}, value_shape={})",
                                       dtype_name(dtype_), static_cast<int>(dtype_),
                                       dtype_name(vals.dtype()), static_cast<int>(vals.dtype()),
                                       shape_.str(), vals.shape().str()));
            assert_index_tensor(indices[0], shape_[0], "index_put_ row index", true, true);
            assert_index_tensor(indices[1], shape_[1], "index_put_ column index", true, true);
            auto row_idx = ensure_same_device(indices[0]);
            auto col_idx = ensure_same_device(indices[1]);
            auto vals_same_device = ensure_same_device(vals);

            LFS_ASSERT_MSG(row_idx.numel() == col_idx.numel() &&
                               row_idx.numel() == vals_same_device.numel(),
                           std::format("multi-index index_put_ row, column, and value counts must "
                                       "match (row_count={}, column_count={}, value_count={}, "
                                       "row_shape={}, column_shape={}, value_shape={})",
                                       row_idx.numel(), col_idx.numel(), vals_same_device.numel(),
                                       row_idx.shape().str(), col_idx.shape().str(),
                                       vals_same_device.shape().str()));

            auto normalize_index_to_int64 = [&](Tensor index, const char* label) -> Tensor {
                if (index.dtype() == DataType::Int64) {
                    return index;
                }
                if (index.dtype() == DataType::Int32) {
                    return index.to(DataType::Int64);
                }
                LFS_ASSERT_MSG(false,
                               std::format("index_put_ {} indices must be Int32 or Int64 "
                                           "(index_dtype={}({}), index_shape={}, index_device={})",
                                           label, dtype_name(index.dtype()),
                                           static_cast<int>(index.dtype()), index.shape().str(),
                                           device_name(index.device())));
            };

            row_idx = normalize_index_to_int64(std::move(row_idx), "row");
            col_idx = normalize_index_to_int64(std::move(col_idx), "col");
            LFS_DEBUG_ASSERT_MSG(row_idx.is_valid() && col_idx.is_valid(),
                                 std::format("normalized multi-index tensors must remain valid "
                                             "(row_index={}, column_index={})",
                                             row_idx.str(), col_idx.str()));
            if (!row_idx.is_contiguous()) {
                row_idx = row_idx.contiguous();
            }
            if (!col_idx.is_contiguous()) {
                col_idx = col_idx.contiguous();
            }
            if (!vals_same_device.is_contiguous()) {
                vals_same_device = vals_same_device.contiguous();
            }

            const int64_t row_bound = static_cast<int64_t>(shape_[0]);
            const int64_t col_bound = static_cast<int64_t>(shape_[1]);

            if (device_ == Device::CUDA) {
                Tensor row_idx_cpu = row_idx.to(Device::CPU);
                Tensor col_idx_cpu = col_idx.to(Device::CPU);
                const int64_t* row_ptr = row_idx_cpu.ptr<int64_t>();
                const int64_t* col_ptr = col_idx_cpu.ptr<int64_t>();
                const float* val_ptr = vals_same_device.ptr<float>();
                float* data_ptr = ptr<float>();

                for (size_t i = 0; i < row_idx.numel(); ++i) {
                    int64_t r = row_ptr[i];
                    int64_t c = col_ptr[i];

                    if (r < 0)
                        r += row_bound;
                    if (c < 0)
                        c += col_bound;

                    if (r >= 0 && r < row_bound &&
                        c >= 0 && c < col_bound) {
                        const size_t offset = static_cast<size_t>(r) * strides_[0] +
                                              static_cast<size_t>(c) * strides_[1];
                        CHECK_CUDA(cudaMemcpyAsync(
                            data_ptr + offset,
                            val_ptr + i,
                            sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream()));
                    }
                }
            } else {
                const int64_t* row_ptr = row_idx.ptr<int64_t>();
                const int64_t* col_ptr = col_idx.ptr<int64_t>();
                const float* val_ptr = vals_same_device.ptr<float>();
                float* data_ptr = ptr<float>();

                for (size_t i = 0; i < row_idx.numel(); ++i) {
                    int64_t r = row_ptr[i];
                    int64_t c = col_ptr[i];
                    if (r < 0)
                        r += row_bound;
                    if (c < 0)
                        c += col_bound;
                    if (r >= 0 && r < row_bound &&
                        c >= 0 && c < col_bound) {
                        const size_t offset = static_cast<size_t>(r) * strides_[0] +
                                              static_cast<size_t>(c) * strides_[1];
                        data_ptr[offset] = val_ptr[i];
                    }
                }
            }
            return *this;
        }

        LFS_ASSERT_MSG(false,
                       std::format("index_put_ index tensor count is unsupported for the "
                                   "destination rank (index_tensor_count={}, destination_rank={}, "
                                   "destination_shape={}, value_shape={})",
                                   indices.size(), shape_.rank(), shape_.str(), vals.shape().str()));
    }

    // Nonzero & Count
    size_t Tensor::count_nonzero() const {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("count_nonzero requires a valid input tensor "
                                   "(input={})",
                                   str()));
        LFS_ASSERT_MSG(is_bool_like(dtype_) || dtype_ == DataType::Float32 ||
                           (device_ == Device::CPU && dtype_ == DataType::Int32),
                       std::format("count_nonzero requires a supported dtype/device combination "
                                   "(input_dtype={}({}), input_device={}, input_shape={}, "
                                   "supported=[bool:any,uint8:any,float32:any,int32:cpu])",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   device_name(device_), shape_.str()));
        if (numel() == 0) {
            return 0;
        }

        // Ensure we have contiguous data for correct linear iteration
        if (!is_contiguous()) {
            return contiguous().count_nonzero();
        }

        if (device_ == Device::CUDA) {
            // Use CUDA kernel for counting
            size_t count = 0;
            size_t* d_count = nullptr;
            CHECK_CUDA(cudaMalloc(&d_count, sizeof(size_t)));
            CHECK_CUDA(cudaMemset(d_count, 0, sizeof(size_t)));

            if (is_bool_like(dtype_)) {
                tensor_ops::launch_count_nonzero_bool(ptr<unsigned char>(), d_count, numel(), stream());
            } else if (dtype_ == DataType::Float32) {
                tensor_ops::launch_count_nonzero_float(ptr<float>(), d_count, numel(), stream());
            }

            // API BOUNDARY: Sync before reading result from GPU
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(&count, d_count, sizeof(size_t), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaFree(d_count));

            return count;
        } else {
            // CPU implementation
            size_t count = 0;

            if (is_bool_like(dtype_)) {
                const unsigned char* data = ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i])
                        count++;
                }
            } else if (dtype_ == DataType::Float32) {
                const float* data = ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i] != 0.0f)
                        count++;
                }
            } else if (dtype_ == DataType::Int32) {
                const int* data = ptr<int>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i] != 0)
                        count++;
                }
            }

            return count;
        }
    }

    Tensor Tensor::nonzero() const {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("nonzero requires a valid input tensor "
                                   "(input={})",
                                   str()));
        LFS_ASSERT_MSG(is_bool_like(dtype_) || dtype_ == DataType::Float32 ||
                           (device_ == Device::CPU && dtype_ == DataType::Int32),
                       std::format("nonzero requires a supported dtype/device combination "
                                   "(input_dtype={}({}), input_device={}, input_shape={}, "
                                   "supported=[bool:any,uint8:any,float32:any,int32:cpu])",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   device_name(device_), shape_.str()));

        // Ensure we have contiguous data for correct linear iteration
        if (!is_contiguous()) {
            return contiguous().nonzero();
        }

        if (numel() == 0) {
            return empty({0, ndim()}, device_, DataType::Int64);
        }

        size_t count = count_nonzero();

        if (count == 0) {
            return empty({0, ndim()}, device_, DataType::Int64);
        }

        size_t n_dims = ndim();

        // Special case for 1D tensors
        if (n_dims == 1) {
            // Allocate MAXIMUM size to prevent buffer overflow from Thrust/CUB mismatch
            auto temp = empty({numel()}, device_, DataType::Int64);
            size_t actual_count = count; // Start with Thrust's count

            if (device_ == Device::CUDA) {
                // Get ACTUAL count from CUB (not Thrust which may differ!)
                if (is_bool_like(dtype_)) {
                    actual_count = tensor_ops::launch_nonzero_bool(ptr<unsigned char>(),
                                                                   reinterpret_cast<int64_t*>(temp.data_ptr()),
                                                                   numel(), numel(), stream());
                } else {
                    actual_count = tensor_ops::launch_nonzero(ptr<float>(),
                                                              reinterpret_cast<int64_t*>(temp.data_ptr()),
                                                              numel(), numel(), stream());
                }

                // DEBUG: Check count mismatch
                if (actual_count != count) {
                    LOG_DEBUG("nonzero() count mismatch: Thrust={}, CUB={}, numel={}", count, actual_count, numel());
                }

                // Slice to actual size - slice is [start, end) exclusive on end
                if (actual_count < numel()) {
                    if (actual_count > 0) {
                        temp = temp.slice(0, 0, actual_count);
                    } else {
                        temp = empty({0}, device_, DataType::Int64);
                    }
                }
                // No sync - tensor operation
            } else {
                int64_t* indices = reinterpret_cast<int64_t*>(temp.data_ptr());
                size_t write_idx = 0;

                if (is_bool_like(dtype_)) {
                    const unsigned char* data = ptr<unsigned char>();
                    for (size_t i = 0; i < numel(); ++i) {
                        if (data[i]) {
                            indices[write_idx++] = static_cast<int64_t>(i);
                        }
                    }
                } else if (dtype_ == DataType::Float32) {
                    const float* data = ptr<float>();
                    for (size_t i = 0; i < numel(); ++i) {
                        if (data[i] != 0.0f) {
                            indices[write_idx++] = static_cast<int64_t>(i);
                        }
                    }
                } else if (dtype_ == DataType::Int32) {
                    const int* data = ptr<int>();
                    for (size_t i = 0; i < numel(); ++i) {
                        if (data[i] != 0) {
                            indices[write_idx++] = static_cast<int64_t>(i);
                        }
                    }
                }

                // Update actual_count from write_idx (CPU path)
                actual_count = write_idx;
            }

            // Slice to actual size (same as CUDA path does at lines 948-954)
            if (actual_count < numel()) {
                if (actual_count > 0) {
                    temp = temp.slice(0, 0, actual_count);
                } else {
                    temp = empty({0}, device_, DataType::Int64);
                }
            }

            // Reshape to (actual_count, 1) to match PyTorch
            return temp.reshape({static_cast<int>(actual_count), 1});
        }

        // Multi-dimensional case
        auto result = empty({static_cast<size_t>(count), static_cast<size_t>(n_dims)}, device_, DataType::Int64);

        if (device_ == Device::CUDA) {
            auto cpu_tensor = to(Device::CPU);
            auto cpu_result = cpu_tensor.nonzero();
            result = cpu_result.to(Device::CUDA);
        } else {
            int64_t* indices = reinterpret_cast<int64_t*>(result.data_ptr());
            size_t write_idx = 0;

            auto strides = shape_.strides();

            if (is_bool_like(dtype_)) {
                const unsigned char* data = ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i]) {
                        size_t temp = i;
                        for (size_t dim = 0; dim < n_dims; ++dim) {
                            size_t coord = temp / strides[dim];
                            temp %= strides[dim];
                            indices[write_idx * n_dims + dim] = static_cast<int64_t>(coord);
                        }
                        write_idx++;
                    }
                }
            } else if (dtype_ == DataType::Float32) {
                const float* data = ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i] != 0.0f) {
                        size_t temp = i;
                        for (size_t dim = 0; dim < n_dims; ++dim) {
                            size_t coord = temp / strides[dim];
                            temp %= strides[dim];
                            indices[write_idx * n_dims + dim] = static_cast<int64_t>(coord);
                        }
                        write_idx++;
                    }
                }
            } else if (dtype_ == DataType::Int32) {
                const int* data = ptr<int>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i] != 0) {
                        size_t temp = i;
                        for (size_t dim = 0; dim < n_dims; ++dim) {
                            size_t coord = temp / strides[dim];
                            temp %= strides[dim];
                            indices[write_idx * n_dims + dim] = static_cast<int64_t>(coord);
                        }
                        write_idx++;
                    }
                }
            }
        }

        return result;
    }

    std::vector<Tensor> Tensor::nonzero_split() const {
        std::vector<Tensor> result;
        result.push_back(nonzero());
        return result;
    }

    // Pythonic Indexing
    TensorIndexer Tensor::operator[](const Tensor& idx) {
        LFS_ASSERT_MSG(is_valid() && idx.is_valid(),
                       std::format("tensor indexing requires valid source and index tensors "
                                   "(source={}, index={})",
                                   str(), idx.str()));
        LFS_ASSERT_MSG(idx.device() == device_,
                       std::format("tensor indices must be on the indexed tensor device "
                                   "(source_device={}, index_device={}, "
                                   "source_shape={}, index_shape={})",
                                   device_name(device_), device_name(idx.device()),
                                   shape_.str(), idx.shape().str()));
        LFS_ASSERT_MSG(is_bool_like(idx.dtype()) || is_integer_index_dtype(idx.dtype()),
                       std::format("tensor indices must be Bool, UInt8, Int32, or Int64 "
                                   "(index_dtype={}({}), index_shape={}, index_device={})",
                                   dtype_name(idx.dtype()), static_cast<int>(idx.dtype()),
                                   idx.shape().str(), device_name(idx.device())));
        std::vector<Tensor> indices;
        indices.reserve(1);
        indices.push_back(idx.clone());
        return TensorIndexer(this, std::move(indices));
    }

    TensorIndexer Tensor::operator[](const std::vector<Tensor>& idx) {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("tensor indexing requires a valid source tensor "
                                   "(source_valid=false, index_tensor_count={})",
                                   idx.size()));
        LFS_ASSERT_MSG(idx.size() == 1,
                       std::format("multi-tensor indexing currently supports exactly one index tensor "
                                   "(index_tensor_count={})",
                                   idx.size()));
        LFS_ASSERT_MSG(idx.front().is_valid(),
                       std::format("tensor indexing requires a valid index tensor "
                                   "(index={}, source_shape={})",
                                   idx.front().str(), shape_.str()));
        LFS_ASSERT_MSG(idx.front().device() == device_,
                       std::format("tensor indices must be on the indexed tensor device "
                                   "(source_device={}, index_device={}, "
                                   "source_shape={}, index_shape={})",
                                   device_name(device_), device_name(idx.front().device()),
                                   shape_.str(), idx.front().shape().str()));
        LFS_ASSERT_MSG(is_bool_like(idx.front().dtype()) ||
                           is_integer_index_dtype(idx.front().dtype()),
                       std::format("tensor indices must be Bool, UInt8, Int32, or Int64 "
                                   "(index_dtype={}({}), index_shape={}, index_device={})",
                                   dtype_name(idx.front().dtype()),
                                   static_cast<int>(idx.front().dtype()),
                                   idx.front().shape().str(), device_name(idx.front().device())));
        std::vector<Tensor> cloned;
        cloned.reserve(idx.size());
        std::ranges::transform(idx, std::back_inserter(cloned),
                               [](const auto& i) { return i.clone(); });
        return TensorIndexer(this, std::move(cloned));
    }

    MaskedTensorProxy Tensor::operator[](const Tensor& mask) const {
        LFS_ASSERT_MSG(is_valid() && mask.is_valid(),
                       std::format("masked indexing requires valid source and mask tensors "
                                   "(source={}, mask={})",
                                   str(), mask.str()));
        LFS_ASSERT_MSG(is_bool_like(mask.dtype()),
                       std::format("masked indexing requires a Bool or UInt8 mask "
                                   "(mask_dtype={}({}), mask_shape={}, mask_device={})",
                                   dtype_name(mask.dtype()), static_cast<int>(mask.dtype()),
                                   mask.shape().str(), device_name(mask.device())));
        LFS_ASSERT_MSG(mask.device() == device_,
                       std::format("masked indexing requires source and mask on the same device "
                                   "(source_device={}, mask_device={}, "
                                   "source_shape={}, mask_shape={})",
                                   device_name(device_), device_name(mask.device()),
                                   shape_.str(), mask.shape().str()));
        return MaskedTensorProxy(this, mask.clone());
    }

    // Element Access
    float& Tensor::at(std::initializer_list<size_t> indices) {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("mutable at() requires a valid tensor "
                                   "(tensor={}, index_count={})",
                                   str(), indices.size()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       std::format("mutable at() requires Float32 input "
                                   "(dtype={}({}), shape={}, device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));
        LFS_ASSERT_MSG(indices.size() == shape_.rank(),
                       std::format("mutable at() index rank must match tensor rank "
                                   "(index_count={}, tensor_rank={}, tensor_shape={})",
                                   indices.size(), shape_.rank(), shape_.str()));
        LFS_ASSERT_MSG(device_ == Device::CPU,
                       std::format("mutable at() cannot return a host reference to CUDA memory "
                                   "(device={}, shape={}, index_count={})",
                                   device_name(device_), shape_.str(), indices.size()));

        std::vector<size_t> idx_vec(indices);

        size_t linear_idx = 0;
        // Use actual strides_ member, not shape_.strides() which assumes contiguous layout
        // This is critical for non-contiguous tensors (e.g., sliced views)

        for (size_t i = 0; i < idx_vec.size(); ++i) {
            LFS_ASSERT_MSG(idx_vec[i] < shape_[i],
                           std::format("at() index {} is out of bounds for dimension {} of size {}",
                                       idx_vec[i], i, shape_[i]));
            linear_idx += idx_vec[i] * strides_[i];
        }

        return ptr<float>()[linear_idx];
    }

    float Tensor::at(std::initializer_list<size_t> indices) const {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("at() requires a valid tensor "
                                   "(tensor={}, index_count={})",
                                   str(), indices.size()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       std::format("at() requires Float32 input "
                                   "(dtype={}({}), shape={}, device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));
        LFS_ASSERT_MSG(indices.size() == shape_.rank(),
                       std::format("at() index rank must match tensor rank "
                                   "(index_count={}, tensor_rank={}, tensor_shape={})",
                                   indices.size(), shape_.rank(), shape_.str()));

        std::vector<size_t> idx_vec(indices);

        size_t linear_idx = 0;
        // Use actual strides_ member, not shape_.strides() which assumes contiguous layout
        // This is critical for non-contiguous tensors (e.g., sliced views)

        for (size_t i = 0; i < idx_vec.size(); ++i) {
            LFS_ASSERT_MSG(idx_vec[i] < shape_[i],
                           std::format("at() index {} is out of bounds for dimension {} of size {}",
                                       idx_vec[i], i, shape_[i]));
            linear_idx += idx_vec[i] * strides_[i];
        }

        if (device_ == Device::CUDA) {
            float value;
            cudaError_t err = cudaMemcpy(&value, ptr<float>() + linear_idx, sizeof(float), cudaMemcpyDeviceToHost);
            LFS_ASSERT_MSG(err == cudaSuccess,
                           std::format("at() CUDA device-to-host copy failed "
                                       "(cuda_error={}({}), bytes={}, linear_index={}, "
                                       "tensor_shape={}, source_pointer={})",
                                       cudaGetErrorString(err), static_cast<int>(err), sizeof(float),
                                       linear_idx, shape_.str(),
                                       static_cast<const void*>(ptr<float>() + linear_idx)));
            return value;
        }
        return ptr<float>()[linear_idx];
    }

    // From Vector
    template <typename T>
    static Tensor from_vector_impl(const std::vector<T>& data, TensorShape shape,
                                   Device device, DataType dtype) {
        LFS_ASSERT_MSG(shape.elements() == data.size(),
                       std::format("from_vector shape has {} elements but input has {}",
                                   shape.elements(), data.size()));
        auto t = Tensor::empty(shape, device, dtype);
        if (!t.is_valid() || t.numel() == 0)
            return t;

        if (t.numel() > 0 && data.data() != nullptr) {
            if (device == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(t.data_ptr(), data.data(), t.bytes(),
                                      cudaMemcpyHostToDevice));
            } else {
                std::memcpy(t.data_ptr(), data.data(), t.bytes());
            }
        }
        return t;
    }

    Tensor Tensor::from_vector(const std::vector<float>& data, TensorShape shape, Device device) {
        return from_vector_impl(data, shape, device, DataType::Float32);
    }

    Tensor Tensor::from_vector(const std::vector<int>& data, TensorShape shape, Device device) {
        return from_vector_impl(data, shape, device, DataType::Int32);
    }

    Tensor Tensor::from_vector(const std::vector<bool>& data, TensorShape shape, Device device) {
        LFS_ASSERT_MSG(shape.elements() == data.size(),
                       std::format("from_vector<bool> shape element count must match input length "
                                   "(shape={}, shape_elements={}, input_length={}, device={})",
                                   shape.str(), shape.elements(), data.size(), device_name(device)));

        std::vector<unsigned char> bytes(data.size());
        std::ranges::transform(data, bytes.begin(),
                               [](bool b) { return b ? 1 : 0; });

        return from_vector_impl(bytes, shape, device, DataType::Bool);
    }

    void Tensor::set_bool(std::initializer_list<size_t> indices, bool value) {
        set_bool(std::span<const size_t>(indices.begin(), indices.size()), value);
    }

    bool Tensor::get_bool(std::initializer_list<size_t> indices) const {
        return get_bool(std::span<const size_t>(indices.begin(), indices.size()));
    }

    // Location: After the existing get_bool/set_bool implementations (around line 800+)
    // grep -C 3 "bool Tensor::get_bool"

    void Tensor::set_bool(std::span<const size_t> indices, bool value) {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("set_bool requires a valid tensor "
                                   "(tensor={}, index_count={}, value={})",
                                   str(), indices.size(), value));
        LFS_ASSERT_MSG(dtype_ == DataType::Bool,
                       std::format("set_bool requires Bool tensor dtype "
                                   "(dtype={}({}), shape={}, device={}, value={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_), value));
        LFS_ASSERT_MSG(indices.size() == shape_.rank(),
                       std::format("set_bool index rank must match tensor rank "
                                   "(index_count={}, tensor_rank={}, tensor_shape={}, value={})",
                                   indices.size(), shape_.rank(), shape_.str(), value));

        // Use actual strides_ member, not shape_.strides() which assumes contiguous layout
        size_t linear_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            LFS_ASSERT_MSG(indices[i] < shape_[i],
                           std::format("set_bool index must be in range "
                                       "(dimension={}, index={}, dimension_size={}, "
                                       "tensor_shape={}, value={})",
                                       i, indices[i], shape_[i], shape_.str(), value));
            linear_idx += indices[i] * strides_[i];
        }

        unsigned char val = value ? 1 : 0;

        if (device_ == Device::CUDA) {
            cudaError_t err = cudaMemcpy(
                ptr<unsigned char>() + linear_idx,
                &val,
                1,
                cudaMemcpyHostToDevice);
            LFS_ASSERT_MSG(err == cudaSuccess,
                           std::format("set_bool CUDA host-to-device copy failed "
                                       "(cuda_error={}({}), bytes=1, linear_index={}, "
                                       "tensor_shape={}, value={})",
                                       cudaGetErrorString(err), static_cast<int>(err),
                                       linear_idx, shape_.str(), value));
        } else {
            ptr<unsigned char>()[linear_idx] = val;
        }
    }

    bool Tensor::get_bool(std::span<const size_t> indices) const {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("get_bool requires a valid tensor "
                                   "(tensor={}, index_count={})",
                                   str(), indices.size()));
        LFS_ASSERT_MSG(dtype_ == DataType::Bool,
                       std::format("get_bool requires Bool tensor dtype "
                                   "(dtype={}({}), shape={}, device={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   shape_.str(), device_name(device_)));
        LFS_ASSERT_MSG(indices.size() == shape_.rank(),
                       std::format("get_bool index rank must match tensor rank "
                                   "(index_count={}, tensor_rank={}, tensor_shape={})",
                                   indices.size(), shape_.rank(), shape_.str()));

        // Use actual strides_ member, not shape_.strides() which assumes contiguous layout
        size_t linear_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            LFS_ASSERT_MSG(indices[i] < shape_[i],
                           std::format("get_bool index must be in range "
                                       "(dimension={}, index={}, dimension_size={}, tensor_shape={})",
                                       i, indices[i], shape_[i], shape_.str()));
            linear_idx += indices[i] * strides_[i];
        }

        if (device_ == Device::CUDA) {
            unsigned char val;
            cudaError_t err = cudaMemcpy(
                &val,
                ptr<unsigned char>() + linear_idx,
                1,
                cudaMemcpyDeviceToHost);
            LFS_ASSERT_MSG(err == cudaSuccess,
                           std::format("get_bool CUDA device-to-host copy failed "
                                       "(cuda_error={}({}), bytes=1, linear_index={}, tensor_shape={})",
                                       cudaGetErrorString(err), static_cast<int>(err),
                                       linear_idx, shape_.str()));
            return val != 0;
        } else {
            return ptr<unsigned char>()[linear_idx] != 0;
        }
    }

    // Proxy Implementations
    void MaskedTensorProxy::operator=(float value) {
        LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid() && mask_.is_valid(),
                       std::format("masked scalar assignment requires valid destination and mask tensors "
                                   "(destination_pointer={}, destination={}, mask={}, value={})",
                                   static_cast<const void*>(tensor_),
                                   tensor_ != nullptr ? tensor_->str() : "null", mask_.str(), value));
        const_cast<Tensor*>(tensor_)->masked_fill_(mask_, value);
    }

    void MaskedTensorProxy::operator=(const Tensor& other) {
        LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid() && other.is_valid(),
                       std::format("masked assignment requires valid destination and source tensors "
                                   "(destination_pointer={}, destination={}, source={}, mask={})",
                                   static_cast<const void*>(tensor_),
                                   tensor_ != nullptr ? tensor_->str() : "null",
                                   other.str(), mask_.str()));
        LFS_ASSERT_MSG(tensor_->dtype() == DataType::Float32 && other.dtype() == DataType::Float32,
                       std::format("masked tensor assignment currently requires Float32 tensors "
                                   "(destination_dtype={}({}), source_dtype={}({}), "
                                   "destination_shape={}, source_shape={})",
                                   dtype_name(tensor_->dtype()), static_cast<int>(tensor_->dtype()),
                                   dtype_name(other.dtype()), static_cast<int>(other.dtype()),
                                   tensor_->shape().str(), other.shape().str()));
        LFS_ASSERT_MSG(tensor_->device() == other.device(),
                       std::format("masked assignment tensors must be on the same device "
                                   "(destination_device={}, source_device={}, mask_device={})",
                                   device_name(tensor_->device()), device_name(other.device()),
                                   device_name(mask_.device())));
        auto selected = tensor_->masked_select(mask_);
        LFS_ASSERT_MSG(selected.numel() == other.numel(),
                       std::format("masked assignment value count must equal the selected element "
                                   "count (selected_count={}, source_numel={}, "
                                   "mask_shape={}, source_shape={})",
                                   selected.numel(), other.numel(),
                                   mask_.shape().str(), other.shape().str()));

        if (tensor_->device() == Device::CUDA) {
            tensor_ops::launch_masked_scatter(const_cast<Tensor*>(tensor_)->ptr<float>(),
                                              mask_.ptr<unsigned char>(), other.ptr<float>(),
                                              tensor_->numel(), other.numel(), tensor_->stream());
            CHECK_CUDA(cudaGetLastError());
            // No sync - tensor operation
        } else {
            float* data = const_cast<Tensor*>(tensor_)->ptr<float>();
            const unsigned char* mask = mask_.ptr<unsigned char>();
            const float* src = other.ptr<float>();

            size_t src_idx = 0;
            for (size_t i = 0; i < tensor_->numel() && src_idx < other.numel(); ++i) {
                if (mask[i])
                    data[i] = src[src_idx++];
            }
        }
    }

    MaskedTensorProxy::operator Tensor() const {
        LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid() && mask_.is_valid(),
                       std::format("masked tensor conversion requires valid source and mask tensors "
                                   "(source_pointer={}, source={}, mask={})",
                                   static_cast<const void*>(tensor_),
                                   tensor_ != nullptr ? tensor_->str() : "null", mask_.str()));
        LFS_ASSERT_MSG(is_bool_like(mask_.dtype()),
                       std::format("masked tensor conversion requires a Bool or UInt8 mask "
                                   "(mask_dtype={}({}), mask_shape={}, mask_device={})",
                                   dtype_name(mask_.dtype()), static_cast<int>(mask_.dtype()),
                                   mask_.shape().str(), device_name(mask_.device())));
        LFS_ASSERT_MSG(mask_.device() == tensor_->device(),
                       std::format("masked tensor conversion requires source and mask on the same "
                                   "device (source_device={}, mask_device={}, "
                                   "source_shape={}, mask_shape={})",
                                   device_name(tensor_->device()), device_name(mask_.device()),
                                   tensor_->shape().str(), mask_.shape().str()));
        // For 1D mask on ND tensor, use row selection (PyTorch-style)
        // tensor[bool_mask] selects rows where mask is True
        return tensor_->index_select(0, mask_);
    }

    void TensorIndexer::operator=(float value) {
        LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid(),
                       std::format("TensorIndexer scalar assignment requires a valid destination "
                                   "(destination_pointer={}, destination_state={}, "
                                   "index_tensor_count={}, value={})",
                                   static_cast<void*>(tensor_),
                                   tensor_ != nullptr ? tensor_->str() : "null",
                                   indices_.size(), value));
        LFS_ASSERT_MSG(indices_.size() == 1 && indices_[0].is_valid(),
                       std::format("TensorIndexer scalar assignment requires exactly one valid "
                                   "index tensor (index_tensor_count={}, index_state={}, value={})",
                                   indices_.size(),
                                   indices_.empty() ? "missing" : indices_[0].str(), value));
        if (is_bool_like(indices_[0].dtype())) {
            tensor_->masked_fill_(indices_[0], value);
        } else {
            tensor_->scatter_(0, indices_[0], value);
        }
    }

    void TensorIndexer::operator=(const Tensor& other) {
        LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid() && other.is_valid(),
                       std::format("TensorIndexer assignment requires valid destination and source "
                                   "tensors (destination_pointer={}, destination_state={}, "
                                   "source={}, index_tensor_count={})",
                                   static_cast<void*>(tensor_),
                                   tensor_ != nullptr ? tensor_->str() : "null",
                                   other.str(), indices_.size()));
        LFS_ASSERT_MSG(indices_.size() == 1 && indices_[0].is_valid(),
                       std::format("TensorIndexer assignment requires exactly one valid index tensor "
                                   "(index_tensor_count={}, index_state={}, source_shape={})",
                                   indices_.size(),
                                   indices_.empty() ? "missing" : indices_[0].str(),
                                   other.shape().str()));
        if (is_bool_like(indices_[0].dtype())) {
            MaskedTensorProxy proxy(tensor_, std::move(indices_[0]));
            proxy = other;
        } else {
            tensor_->scatter_(0, indices_[0], other);
        }
    }

    TensorIndexer::operator Tensor() const {
        LFS_ASSERT_MSG(tensor_ != nullptr && tensor_->is_valid(),
                       std::format("TensorIndexer conversion requires a valid source tensor "
                                   "(source_pointer={}, source_state={}, index_tensor_count={})",
                                   static_cast<void*>(tensor_),
                                   tensor_ != nullptr ? tensor_->str() : "null", indices_.size()));
        LFS_ASSERT_MSG(indices_.size() == 1,
                       std::format("TensorIndexer conversion currently supports exactly one index "
                                   "tensor (index_tensor_count={}, source_shape={})",
                                   indices_.size(), tensor_->shape().str()));
        // For both bool and int indices, use index_select for row selection.
        return indices_[0].ndim() == 1 ? tensor_->index_select(0, indices_[0]) : tensor_->take(indices_[0]);
    }

    Tensor& Tensor::append_gather(const Tensor& indices) {
        materialize_if_deferred();
        LFS_ASSERT_MSG(is_valid() && indices.is_valid(),
                       std::format("append_gather requires valid destination and index tensors "
                                   "(destination={}, indices={})",
                                   str(), indices.str()));
        LFS_ASSERT_MSG(indices.ndim() == 1,
                       std::format("append_gather requires rank-1 indices "
                                   "(index_rank={}, index_shape={}, destination_shape={})",
                                   indices.ndim(), indices.shape().str(), shape_.str()));
        LFS_ASSERT_MSG(indices.device() == device_,
                       std::format("append_gather indices must be on the destination device "
                                   "(destination_device={}, index_device={}, "
                                   "destination_shape={}, index_shape={})",
                                   device_name(device_), device_name(indices.device()),
                                   shape_.str(), indices.shape().str()));
        LFS_ASSERT_MSG(state_->capacity > 0,
                       std::format("append_gather requires reserved row capacity "
                                   "(capacity={}, logical_size={}, destination_shape={}, "
                                   "index_count={})",
                                   state_->capacity, state_->logical_size,
                                   shape_.str(), indices.numel()));
        LFS_ASSERT_MSG(ndim() > 0,
                       std::format("append_gather requires at least one destination dimension "
                                   "(destination_rank={}, destination_shape={}, index_count={})",
                                   ndim(), shape_.str(), indices.numel()));
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 || dtype_ == DataType::UInt8 || dtype_ == DataType::Bool ||
                           device_ == Device::CPU,
                       std::format("CUDA append_gather requires Float32, UInt8, or Bool destination "
                                   "(destination_dtype={}({}), destination_device={}, "
                                   "destination_shape={})",
                                   dtype_name(dtype_), static_cast<int>(dtype_),
                                   device_name(device_), shape_.str()));
        assert_index_tensor(indices,
                            state_->logical_size > 0 ? state_->logical_size : shape_[0],
                            "append_gather",
                            true);

        size_t n_gather = indices.numel();

        // Use logical_size_ if set, otherwise use shape_[0] (for tensors not created with reserve())
        const size_t current_size = (state_->capacity > 0 && state_->logical_size > 0) ? state_->logical_size : shape_[0];
        LFS_ASSERT_MSG(n_gather <= std::numeric_limits<size_t>::max() - current_size,
                       std::format("append_gather row count must not overflow size_t "
                                   "(current_rows={}, appended_rows={}, size_t_max={}, capacity={})",
                                   current_size, n_gather, std::numeric_limits<size_t>::max(),
                                   state_->capacity));
        const size_t new_size = current_size + n_gather;

        LOG_DEBUG("append_gather: capacity_={}, logical_size_={}, shape_[0]={}, current_size={}, n_gather={}, new_size={}",
                  state_->capacity, state_->logical_size, shape_[0], current_size, n_gather, new_size);

        LFS_ASSERT_MSG(new_size <= state_->capacity,
                       std::format("append_gather needs capacity {}, but only {} is reserved",
                                   new_size, state_->capacity));

        // Calculate row size (all elements in dims 1+)
        size_t row_size = 1;
        for (size_t i = 1; i < shape_.rank(); i++) {
            LFS_ASSERT_MSG(shape_[i] == 0 ||
                               row_size <= std::numeric_limits<size_t>::max() / shape_[i],
                           std::format("append_gather row element count must not overflow size_t "
                                       "(dimension={}, dimension_size={}, product_before={}, "
                                       "size_t_max={}, destination_shape={})",
                                       i, shape_[i], row_size,
                                       std::numeric_limits<size_t>::max(), shape_.str()));
            row_size *= shape_[i];
        }

        // Calculate write offset (in elements, not rows)
        LFS_ASSERT_MSG(row_size == 0 ||
                           current_size <= std::numeric_limits<size_t>::max() / row_size,
                       std::format("append_gather write offset must not overflow size_t "
                                   "(current_rows={}, row_size={}, size_t_max={}, "
                                   "destination_shape={})",
                                   current_size, row_size,
                                   std::numeric_limits<size_t>::max(), shape_.str()));
        const size_t write_offset_elements = current_size * row_size;

        // Convert Int64 indices to Int32 for kernel
        auto indices_same_device = ensure_same_device(indices);
        bool is_int64 = indices_same_device.dtype() == DataType::Int64;
        Tensor indices_int32;
        if (is_int64) {
            indices_int32 = indices_same_device.to(DataType::Int32);
        }
        const int* idx_ptr = is_int64 ? indices_int32.ptr<int>() : indices_same_device.ptr<int>();

        // Launch kernel to append gathered rows directly to the end
        if (device_ == Device::CUDA) {
            LOG_DEBUG("  Launching index_select kernel: write_offset_elements={}, output_offset_bytes={}, n_gather={}",
                      write_offset_elements, write_offset_elements * dtype_size(dtype_), n_gather);

            // IMPORTANT: Pass the INPUT shape to the kernel, not the output shape!
            // The kernel needs to know the source tensor dimensions to validate indices
            const size_t* input_shape = shape_.dims().data();

            // Use index_select kernel to gather into the output location
            if (dtype_ == DataType::Float32) {
                float* output_ptr = ptr<float>() + write_offset_elements;
                tensor_ops::launch_index_select(ptr<float>(), idx_ptr,
                                                output_ptr, input_shape,
                                                shape_.rank(), 0, n_gather,
                                                0 /*BoundaryMode::Assert*/, stream());
                CHECK_CUDA(cudaStreamSynchronize(stream()));
            } else if (dtype_ == DataType::UInt8 || dtype_ == DataType::Bool) {
                uint8_t* output_ptr = ptr<uint8_t>() + write_offset_elements;
                tensor_ops::launch_index_select(ptr<uint8_t>(), idx_ptr,
                                                output_ptr, input_shape,
                                                shape_.rank(), 0, n_gather,
                                                0 /*BoundaryMode::Assert*/, stream());
                CHECK_CUDA(cudaStreamSynchronize(stream()));
            } else {
                LFS_ASSERT_MSG(false,
                               std::format("append_gather CUDA dispatch reached an unsupported dtype "
                                           "(dtype={}({}), destination_shape={}, index_shape={}, "
                                           "current_rows={}, appended_rows={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), indices.shape().str(),
                                           current_size, n_gather));
            }
        } else {
            // CPU implementation (byte-wise; works for any dtype)
            const size_t elem = dtype_size(dtype_);
            const uint8_t* src = static_cast<const uint8_t*>(data_);
            uint8_t* dst = static_cast<uint8_t*>(data_) + write_offset_elements * elem;

            for (size_t i = 0; i < n_gather; i++) {
                int sel = idx_ptr[i];
                if (sel < 0) {
                    sel += static_cast<int>(state_->logical_size);
                }

                if (sel < 0 || sel >= static_cast<int>(state_->logical_size)) {
                    LFS_ASSERT_MSG(false,
                                   std::format("append_gather index {} is out of range [0, {})",
                                               sel, state_->logical_size));
                }

                std::memcpy(dst + i * row_size * elem,
                            src + static_cast<size_t>(sel) * row_size * elem,
                            row_size * elem);
            }
        }

        // Update logical size and shape
        state_->logical_size = new_size;
        auto new_shape = shape_.dims();
        new_shape[0] = new_size;
        shape_ = TensorShape(new_shape);

        LOG_DEBUG("append_gather: grew tensor from {} to {} rows (capacity: {})",
                  state_->logical_size - n_gather, state_->logical_size, state_->capacity);

        return *this;
    }

    Tensor& Tensor::append_zeros(size_t n_rows) {
        materialize_if_deferred();
        LOG_DEBUG("append_zeros: n_rows={}", n_rows);
        LFS_ASSERT_MSG(is_valid(),
                       std::format("append_zeros requires a valid destination tensor "
                                   "(destination={}, appended_rows={})",
                                   str(), n_rows));

        if (n_rows == 0) {
            return *this;
        }

        // Validation: check capacity
        if (state_->capacity == 0) {
            LOG_ERROR("append_zeros({}) failed on tensor '{}' (id={}): capacity_=0, shape={}, is_view_={}",
                      n_rows, name().empty() ? "<unnamed>" : name(), id_, shape_.str(), is_view_);
            throw std::runtime_error("append_zeros() requires tensor with capacity > 0 (use reserve() first)");
        }

        if (shape_.rank() == 0) {
            throw std::runtime_error(std::format(
                "append_zeros cannot append rows to a scalar tensor "
                "(destination_shape={}, appended_rows={}, capacity={})",
                shape_.str(), n_rows, state_->capacity));
        }

        // Calculate sizes
        const size_t current_size = (state_->logical_size > 0) ? state_->logical_size : shape_[0];
        LFS_ASSERT_MSG(n_rows <= std::numeric_limits<size_t>::max() - current_size,
                       std::format("append_zeros row count must not overflow size_t "
                                   "(current_rows={}, appended_rows={}, size_t_max={}, capacity={})",
                                   current_size, n_rows, std::numeric_limits<size_t>::max(),
                                   state_->capacity));
        const size_t new_size = current_size + n_rows;

        if (new_size > state_->capacity) {
            throw std::runtime_error(std::format(
                "append_zeros() requires sufficient capacity: current={}, n_rows={}, new_size={}, capacity={}",
                current_size, n_rows, new_size, state_->capacity));
        }

        // Calculate row size in elements
        size_t row_size = 1;
        for (size_t i = 1; i < shape_.rank(); i++) {
            LFS_ASSERT_MSG(shape_[i] == 0 ||
                               row_size <= std::numeric_limits<size_t>::max() / shape_[i],
                           std::format("append_zeros row element count must not overflow size_t "
                                       "(dimension={}, dimension_size={}, product_before={}, "
                                       "size_t_max={}, destination_shape={})",
                                       i, shape_[i], row_size,
                                       std::numeric_limits<size_t>::max(), shape_.str()));
            row_size *= shape_[i];
        }

        // Calculate write offset in elements
        LFS_ASSERT_MSG(row_size == 0 ||
                           current_size <= std::numeric_limits<size_t>::max() / row_size,
                       std::format("append_zeros write offset must not overflow size_t "
                                   "(current_rows={}, row_size={}, size_t_max={}, "
                                   "destination_shape={})",
                                   current_size, row_size,
                                   std::numeric_limits<size_t>::max(), shape_.str()));
        const size_t write_offset_elements = current_size * row_size;
        LFS_ASSERT_MSG(row_size == 0 ||
                           n_rows <= std::numeric_limits<size_t>::max() / row_size,
                       std::format("append_zeros zeroed element count must not overflow size_t "
                                   "(appended_rows={}, row_size={}, size_t_max={}, "
                                   "destination_shape={})",
                                   n_rows, row_size,
                                   std::numeric_limits<size_t>::max(), shape_.str()));
        const size_t zero_elements = n_rows * row_size;
        const size_t element_bytes = dtype_size(dtype_);
        LFS_ASSERT_MSG(element_bytes == 0 ||
                           zero_elements <= std::numeric_limits<size_t>::max() / element_bytes,
                       std::format("append_zeros zeroed byte count must not overflow size_t "
                                   "(zero_elements={}, element_bytes={}, size_t_max={}, "
                                   "destination_dtype={}({}))",
                                   zero_elements, element_bytes,
                                   std::numeric_limits<size_t>::max(), dtype_name(dtype_),
                                   static_cast<int>(dtype_)));
        LFS_ASSERT_MSG(element_bytes == 0 ||
                           write_offset_elements <= std::numeric_limits<size_t>::max() / element_bytes,
                       std::format("append_zeros byte offset must not overflow size_t "
                                   "(write_offset_elements={}, element_bytes={}, size_t_max={}, "
                                   "destination_shape={})",
                                   write_offset_elements, element_bytes,
                                   std::numeric_limits<size_t>::max(), shape_.str()));
        const size_t zero_bytes = zero_elements * element_bytes;
        const size_t write_offset_bytes = write_offset_elements * element_bytes;

        // Zero out the appended region
        if (device_ == Device::CUDA) {
            void* write_ptr = static_cast<uint8_t*>(data_) + write_offset_bytes;
            CHECK_CUDA(cudaMemsetAsync(write_ptr, 0, zero_bytes, stream()));
            CHECK_CUDA(cudaStreamSynchronize(stream()));
        } else {
            void* write_ptr = static_cast<uint8_t*>(data_) + write_offset_bytes;
            std::memset(write_ptr, 0, zero_bytes);
        }

        // Update logical size and shape
        state_->logical_size = new_size;
        auto new_shape = shape_.dims();
        new_shape[0] = new_size;
        shape_ = TensorShape(new_shape);

        LOG_DEBUG("append_zeros: grew tensor from {} to {} rows (capacity: {})",
                  state_->logical_size - n_rows, state_->logical_size, state_->capacity);

        return *this;
    }

#undef CHECK_CUDA

} // namespace lfs::core
