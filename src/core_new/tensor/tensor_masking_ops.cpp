/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cstring>
#include <execution>
#include <numeric>
#include <ranges>

#define CHECK_CUDA(call)                                        \
    do {                                                        \
        if (auto e = call; e != cudaSuccess) {                  \
            LOG_ERROR("CUDA error: {}", cudaGetErrorString(e)); \
        }                                                       \
    } while (0)

namespace lfs::core {

    // ============= Masking Operations =============
    Tensor Tensor::masked_select(const Tensor& mask) const {
        if (!is_valid() || !mask.is_valid()) {
            LOG_ERROR("masked_select on invalid tensor");
            return Tensor();
        }

        if (mask.dtype() != DataType::Bool) {
            LOG_ERROR("masked_select requires boolean mask");
            return Tensor();
        }

        if (mask.shape() != shape_) {
            LOG_ERROR("Mask shape {} doesn't match tensor shape {}",
                      mask.shape().str(), shape_.str());
            return Tensor();
        }

        // CRITICAL: Check device compatibility BEFORE any operations
        if (mask.device() != device_) {
            LOG_ERROR("masked_select: mask device ({}) doesn't match tensor device ({})",
                      device_name(mask.device()), device_name(device_));
            return Tensor();
        }

        // CRITICAL FIX: Count TRUE values in mask to determine output size
        size_t output_size = mask.count_nonzero();

        LOG_DEBUG("masked_select: input size={}, mask trues={}, output size={}",
                  numel(), output_size, output_size);

        if (output_size == 0) {
            return empty({0}, device_, dtype_);
        }

        auto result = empty({output_size}, device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use CUDA kernel
            tensor_ops::launch_masked_select(ptr<float>(), mask.ptr<unsigned char>(),
                                             result.ptr<float>(), numel(), output_size, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation - FIXED to respect mask
            const float* src = ptr<float>();
            const unsigned char* mask_data = mask.ptr<unsigned char>();
            float* dst = result.ptr<float>();

            size_t write_idx = 0;
            for (size_t i = 0; i < numel(); ++i) {
                // CRITICAL FIX: Only copy when mask is TRUE
                if (mask_data[i]) {
                    dst[write_idx++] = src[i];
                }
            }

            LOG_DEBUG("masked_select CPU: wrote {} elements", write_idx);
        }

        return result;
    }

    Tensor& Tensor::masked_fill_(const Tensor& mask, float value) {
        if (!is_valid() || !mask.is_valid()) {
            LOG_ERROR("masked_fill_ on invalid tensor");
            return *this;
        }

        if (mask.dtype() != DataType::Bool) {
            LOG_ERROR("masked_fill_ requires boolean mask");
            return *this;
        }

        if (mask.shape() != shape_) {
            LOG_ERROR("Mask shape doesn't match tensor shape");
            return *this;
        }

        // CRITICAL: Check device compatibility
        if (mask.device() != device_) {
            LOG_ERROR("masked_fill_: mask device ({}) doesn't match tensor device ({})",
                      device_name(mask.device()), device_name(device_));
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_masked_fill(ptr<float>(), mask.ptr<unsigned char>(),
                                           value, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = ptr<float>();
            const unsigned char* mask_data = mask.ptr<unsigned char>();

            for (size_t i = 0; i < numel(); ++i) {
                if (mask_data[i]) {
                    data[i] = value;
                }
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
        if (!is_valid() || !indices.is_valid())
            return {};

        if (indices.ndim() != 1)
            return {};

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(shape_.rank()))
            return {};

        auto dims = shape_.dims();
        dims[dim] = indices.numel();
        auto result = zeros(TensorShape(dims), device_, dtype_);

        auto indices_same_device = ensure_same_device(indices);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_select(ptr<float>(), indices_same_device.ptr<int>(),
                                            result.ptr<float>(), shape_.dims().data(),
                                            shape_.rank(), dim, indices.numel(),
                                            static_cast<int>(mode), 0);
            cudaDeviceSynchronize();
        } else {
            size_t outer = 1, inner = 1;
            for (int i = 0; i < dim; ++i)
                outer *= shape_[i];
            for (size_t i = dim + 1; i < shape_.rank(); ++i)
                inner *= shape_[i];

            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            const int* idx = indices_same_device.ptr<int>();

            auto process_idx = [&](int sel) -> int {
                if (mode == BoundaryMode::Clamp) {
                    return std::clamp(sel, 0, static_cast<int>(shape_[dim]) - 1);
                } else if (mode == BoundaryMode::Wrap) {
                    return ((sel % static_cast<int>(shape_[dim])) + shape_[dim]) % shape_[dim];
                }
                if (sel < 0)
                    sel += shape_[dim];
                return sel;
            };

            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < indices.numel(); ++i) {
                    int sel = process_idx(idx[i]);
                    if (sel >= 0 && sel < static_cast<int>(shape_[dim])) {
                        std::copy_n(src + (o * shape_[dim] + sel) * inner,
                                    inner,
                                    dst + (o * indices.numel() + i) * inner);
                    }
                }
            }
        }
        return result;
    }

    Tensor Tensor::gather(int dim, const Tensor& indices) const {
        return gather(dim, indices, BoundaryMode::Assert);
    }

    Tensor Tensor::gather(int dim, const Tensor& indices, BoundaryMode mode) const {
        if (!is_valid() || !indices.is_valid())
            return {};

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(shape_.rank()))
            return {};

        if (indices.ndim() == 1) {
            std::vector<size_t> out_dims = shape_.dims();
            out_dims[dim] = indices.numel();
            auto result = zeros(TensorShape(out_dims), device_, dtype_);

            auto indices_same_device = ensure_same_device(indices);

            if (device_ == Device::CUDA) {
                tensor_ops::launch_gather(ptr<float>(), indices_same_device.ptr<int>(),
                                          result.ptr<float>(), shape_.dims().data(),
                                          indices.shape().dims().data(), shape_.rank(), dim,
                                          result.numel(), static_cast<int>(mode), 0);
                cudaDeviceSynchronize();
            } else {
                const float* src = ptr<float>();
                float* dst = result.ptr<float>();
                const int* idx_data = indices_same_device.ptr<int>();

                size_t outer = 1;
                for (int i = 0; i < dim; ++i) {
                    outer *= shape_[i];
                }

                size_t inner = 1;
                for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                    inner *= shape_[i];
                }

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
            }

            return result;
        }

        if (indices.ndim() != shape_.rank()) {
            LOG_ERROR("For multi-dimensional gather, indices must have same rank as input");
            return {};
        }

        auto result = zeros(indices.shape(), device_, dtype_);
        auto indices_same_device = ensure_same_device(indices);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_gather(ptr<float>(), indices_same_device.ptr<int>(),
                                      result.ptr<float>(), shape_.dims().data(),
                                      indices.shape().dims().data(), shape_.rank(), dim,
                                      result.numel(), static_cast<int>(mode), 0);
            cudaDeviceSynchronize();
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            const int* idx_data = indices_same_device.ptr<int>();

            size_t total_elements = indices.numel();

            auto input_strides = shape_.strides();
            auto output_strides = indices.shape().strides();

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
        }

        return result;
    }

    Tensor Tensor::take(const Tensor& indices) const {
        if (!is_valid() || !indices.is_valid())
            return {};

        auto indices_same_device = ensure_same_device(indices);
        auto flat = flatten();
        auto result = empty(indices_same_device.shape(), device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_take(flat.ptr<float>(), indices_same_device.ptr<int>(),
                                    result.ptr<float>(), flat.numel(), indices_same_device.numel(), 0);
            cudaDeviceSynchronize();
        } else {
            const float* src = flat.ptr<float>();
            float* dst = result.ptr<float>();
            const int* idx = indices_same_device.ptr<int>();
            size_t total = flat.numel();

            std::transform(std::execution::par_unseq,
                           idx, idx + indices_same_device.numel(), dst,
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
        if (mode == ScatterMode::Add) {
            return index_add_(dim, idx, src);
        }

        if (!is_valid() || !idx.is_valid() || !src.is_valid())
            return *this;

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(shape_.rank()))
            return *this;

        if (shape_.rank() == 1 && dim == 0) {
            if (idx.ndim() != 1 || src.ndim() != 1) {
                LOG_ERROR("1D scatter requires 1D indices and source");
                return *this;
            }

            if (idx.numel() != src.numel()) {
                LOG_ERROR("Index and source must have same number of elements");
                return *this;
            }

            float* dst = ptr<float>();

            auto indices_same_device = ensure_same_device(idx);
            auto src_same_device = ensure_same_device(src);

            const int* indices = indices_same_device.ptr<int>();
            const float* src_data = src_same_device.ptr<float>();

            if (device_ == Device::CUDA) {
                tensor_ops::launch_scatter(dst, indices, src_data,
                                           shape_.dims().data(), src.shape().dims().data(),
                                           shape_.rank(), dim, src.numel(),
                                           static_cast<int>(mode), 0);
                cudaDeviceSynchronize();
            } else {
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
            }

            return *this;
        }

        if (idx.ndim() != 1) {
            LOG_ERROR("scatter_ currently only supports 1D index tensors");
            return *this;
        }

        std::vector<size_t> expected_shape = shape_.dims();
        expected_shape[dim] = idx.numel();

        if (src.shape() != TensorShape(expected_shape)) {
            LOG_ERROR("Source shape mismatch: expected {}, got {}",
                      TensorShape(expected_shape).str(), src.shape().str());
            return *this;
        }

        auto idx_same_device = ensure_same_device(idx);
        auto src_same_device = ensure_same_device(src);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scatter(ptr<float>(), idx_same_device.ptr<int>(),
                                       src_same_device.ptr<float>(), shape_.dims().data(),
                                       src.shape().dims().data(),
                                       shape_.rank(), dim, src.numel(),
                                       static_cast<int>(mode), 0);
            cudaDeviceSynchronize();
        } else {
            size_t outer = 1;
            for (int i = 0; i < dim; ++i) {
                outer *= shape_[i];
            }

            size_t inner = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner *= shape_[i];
            }

            float* dst = ptr<float>();
            const int* indices = idx_same_device.ptr<int>();
            const float* src_data = src_same_device.ptr<float>();

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
                            LOG_ERROR("Index out of bounds in scatter_");
                            return *this;
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
        }

        return *this;
    }

    Tensor& Tensor::scatter_(int dim, const Tensor& idx, float val, ScatterMode mode) {
        auto src = full(idx.shape(), val, device_, dtype_);
        return scatter_(dim, idx, src, mode);
    }

    Tensor& Tensor::index_fill_(int dim, const Tensor& idx, float val) {
        return scatter_(dim, idx, val, ScatterMode::None);
    }

    Tensor& Tensor::index_copy_(int dim, const Tensor& idx, const Tensor& src) {
        if (!is_valid() || !idx.is_valid() || !src.is_valid())
            return *this;

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(shape_.rank()))
            return *this;

        if (idx.ndim() != 1) {
            LOG_ERROR("index_copy_ requires 1D index tensor");
            return *this;
        }

        std::vector<size_t> expected_src_shape = shape_.dims();
        expected_src_shape[dim] = idx.numel();

        if (src.shape() != TensorShape(expected_src_shape)) {
            LOG_ERROR("Source tensor has wrong shape for index_copy_");
            return *this;
        }

        auto idx_same_device = ensure_same_device(idx);
        auto src_same_device = ensure_same_device(src);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_copy(ptr<float>(), idx_same_device.ptr<int>(),
                                          src_same_device.ptr<float>(), shape_.dims().data(),
                                          shape_.rank(), dim, idx.numel(), 0);
            cudaDeviceSynchronize();
        } else {
            size_t outer = 1, inner = 1;
            for (int i = 0; i < dim; ++i)
                outer *= shape_[i];
            for (size_t i = dim + 1; i < shape_.rank(); ++i)
                inner *= shape_[i];

            float* dst = ptr<float>();
            const int* indices = idx_same_device.ptr<int>();
            const float* src_data = src_same_device.ptr<float>();

            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < idx.numel(); ++i) {
                    int pos = indices[i];
                    if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                        LOG_ERROR("Index {} out of bounds for dimension {} of size {}",
                                  pos, dim, shape_[dim]);
                        continue;
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
        }

        return *this;
    }

    Tensor& Tensor::index_add_(int dim, const Tensor& idx, const Tensor& src) {
        if (!is_valid() || !idx.is_valid() || !src.is_valid())
            return *this;

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(shape_.rank()))
            return *this;

        if (idx.ndim() != 1) {
            LOG_ERROR("index_add_ requires 1D index tensor");
            return *this;
        }

        if (shape_.rank() == 1 && dim == 0) {
            if (src.ndim() != 1 || src.numel() != idx.numel()) {
                LOG_ERROR("Source must be 1D with same size as indices for 1D tensor");
                return *this;
            }

            auto idx_same_device = ensure_same_device(idx);
            auto src_same_device = ensure_same_device(src);

            if (device_ == Device::CUDA) {
                tensor_ops::launch_index_add(ptr<float>(), idx_same_device.ptr<int>(),
                                             src_same_device.ptr<float>(), shape_.dims().data(),
                                             shape_.rank(), dim, idx.numel(), 0);
                cudaDeviceSynchronize();
            } else {
                float* data = ptr<float>();
                const int* indices = idx_same_device.ptr<int>();
                const float* src_data = src_same_device.ptr<float>();

                for (size_t i = 0; i < idx.numel(); ++i) {
                    int pos = indices[i];
                    if (pos < 0)
                        pos += shape_[0];
                    if (pos >= 0 && pos < static_cast<int>(shape_[0])) {
                        data[pos] += src_data[i];
                    }
                }
            }
            return *this;
        }

        std::vector<size_t> expected_shape = shape_.dims();
        expected_shape[dim] = idx.numel();

        if (src.shape() != TensorShape(expected_shape)) {
            LOG_ERROR("Source shape mismatch in index_add_: expected {}, got {}",
                      TensorShape(expected_shape).str(), src.shape().str());
            return *this;
        }

        auto idx_same_device = ensure_same_device(idx);
        auto src_same_device = ensure_same_device(src);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_add(ptr<float>(), idx_same_device.ptr<int>(),
                                         src_same_device.ptr<float>(), shape_.dims().data(),
                                         shape_.rank(), dim, idx.numel(), 0);
            cudaDeviceSynchronize();
        } else {
            size_t outer = 1;
            for (int i = 0; i < dim; ++i) {
                outer *= shape_[i];
            }

            size_t inner = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner *= shape_[i];
            }

            float* data = ptr<float>();
            const int* indices = idx_same_device.ptr<int>();
            const float* src_data = src_same_device.ptr<float>();

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

        return *this;
    }

    Tensor& Tensor::index_put_(const Tensor& idx, const Tensor& vals) {
        if (!is_valid() || !idx.is_valid() || !vals.is_valid())
            return *this;

        auto idx_same_device = ensure_same_device(idx);
        auto vals_same_device = ensure_same_device(vals);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_put(ptr<float>(), idx_same_device.ptr<int>(),
                                         vals_same_device.ptr<float>(), numel(), idx.numel(), 0);
            cudaDeviceSynchronize();
        } else {
            float* data = ptr<float>();
            const int* indices = idx_same_device.ptr<int>();
            const float* values = vals_same_device.ptr<float>();
            size_t num_elements = numel();

            std::for_each(std::execution::par_unseq,
                          std::views::iota(0uz, idx.numel()).begin(),
                          std::views::iota(0uz, idx.numel()).end(),
                          [data, indices, values, num_elements](size_t i) {
                              int pos = indices[i];
                              if (pos < 0)
                                  pos += num_elements;
                              if (pos >= 0 && pos < static_cast<int>(num_elements)) {
                                  data[pos] = values[i];
                              }
                          });
        }
        return *this;
    }

    Tensor& Tensor::index_put_(const std::vector<Tensor>& indices, const Tensor& vals) {
        if (!is_valid() || !vals.is_valid())
            return *this;

        if (indices.empty())
            return *this;

        if (indices.size() == 1) {
            return index_put_(indices[0], vals);
        }

        if (indices.size() == 2 && shape_.rank() == 2) {
            auto row_idx = ensure_same_device(indices[0]);
            auto col_idx = ensure_same_device(indices[1]);
            auto vals_same_device = ensure_same_device(vals);

            if (row_idx.numel() != col_idx.numel() || row_idx.numel() != vals_same_device.numel()) {
                LOG_ERROR("Index tensors and values must have same number of elements");
                return *this;
            }

            if (device_ == Device::CUDA) {
                auto cpu_tensor = to(Device::CPU);
                auto cpu_row = row_idx.to(Device::CPU);
                auto cpu_col = col_idx.to(Device::CPU);
                auto cpu_vals = vals_same_device.to(Device::CPU);

                const int* row_ptr = cpu_row.ptr<int>();
                const int* col_ptr = cpu_col.ptr<int>();
                const float* val_ptr = cpu_vals.ptr<float>();
                float* data_ptr = cpu_tensor.ptr<float>();

                for (size_t i = 0; i < cpu_row.numel(); ++i) {
                    int r = row_ptr[i];
                    int c = col_ptr[i];

                    if (r < 0)
                        r += shape_[0];
                    if (c < 0)
                        c += shape_[1];

                    if (r >= 0 && r < static_cast<int>(shape_[0]) &&
                        c >= 0 && c < static_cast<int>(shape_[1])) {
                        data_ptr[r * shape_[1] + c] = val_ptr[i];
                    }
                }

                *this = cpu_tensor.to(device_);
            } else {
                const int* row_ptr = row_idx.ptr<int>();
                const int* col_ptr = col_idx.ptr<int>();
                const float* val_ptr = vals_same_device.ptr<float>();
                float* data_ptr = ptr<float>();

                for (size_t i = 0; i < row_idx.numel(); ++i) {
                    int r = row_ptr[i];
                    int c = col_ptr[i];

                    if (r < 0)
                        r += shape_[0];
                    if (c < 0)
                        c += shape_[1];

                    if (r >= 0 && r < static_cast<int>(shape_[0]) &&
                        c >= 0 && c < static_cast<int>(shape_[1])) {
                        data_ptr[r * shape_[1] + c] = val_ptr[i];
                    }
                }
            }
            return *this;
        }

        LOG_WARN("Multi-dimensional index_put_ not fully implemented for {} dimensions", indices.size());
        return *this;
    }

    // Nonzero & Count
    size_t Tensor::count_nonzero() const {
        if (!is_valid() || numel() == 0) {
            return 0;
        }

        if (device_ == Device::CUDA) {
            // Use CUDA kernel for counting
            size_t count = 0;
            size_t* d_count = nullptr;
            CHECK_CUDA(cudaMalloc(&d_count, sizeof(size_t)));
            CHECK_CUDA(cudaMemset(d_count, 0, sizeof(size_t)));

            if (dtype_ == DataType::Bool) {
                tensor_ops::launch_count_nonzero_bool(ptr<unsigned char>(), d_count, numel(), 0);
            } else if (dtype_ == DataType::Float32) {
                tensor_ops::launch_count_nonzero_float(ptr<float>(), d_count, numel(), 0);
            }

            CHECK_CUDA(cudaMemcpy(&count, d_count, sizeof(size_t), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaFree(d_count));

            return count;
        } else {
            // CPU implementation
            size_t count = 0;

            if (dtype_ == DataType::Bool) {
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
        if (!is_valid()) {
            LOG_ERROR("nonzero() on invalid tensor");
            return {};
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
            // Create a flat tensor first
            auto temp = empty({count}, device_, DataType::Int64);

            if (device_ == Device::CUDA) {
                if (dtype_ == DataType::Bool) {
                    tensor_ops::launch_nonzero_bool(ptr<unsigned char>(),
                                                    reinterpret_cast<int64_t*>(temp.raw_ptr()),
                                                    numel(), count, 0);
                } else {
                    tensor_ops::launch_nonzero(ptr<float>(),
                                               reinterpret_cast<int64_t*>(temp.raw_ptr()),
                                               numel(), count, 0);
                }
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                int64_t* indices = reinterpret_cast<int64_t*>(temp.raw_ptr());
                size_t write_idx = 0;

                if (dtype_ == DataType::Bool) {
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
            }

            // Reshape to (count, 1) to match PyTorch
            return temp.reshape({static_cast<int>(count), 1});
        }

        // Multi-dimensional case
        auto result = empty({static_cast<size_t>(count), static_cast<size_t>(n_dims)}, device_, DataType::Int64);

        if (device_ == Device::CUDA) {
            auto cpu_tensor = to(Device::CPU);
            auto cpu_result = cpu_tensor.nonzero();
            result = cpu_result.to(Device::CUDA);
        } else {
            int64_t* indices = reinterpret_cast<int64_t*>(result.raw_ptr());
            size_t write_idx = 0;

            auto strides = shape_.strides();

            if (dtype_ == DataType::Bool) {
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
        std::vector<Tensor> indices;
        indices.reserve(1);
        indices.push_back(idx.clone());
        return TensorIndexer(this, std::move(indices));
    }

    TensorIndexer Tensor::operator[](const std::vector<Tensor>& idx) {
        std::vector<Tensor> cloned;
        cloned.reserve(idx.size());
        std::ranges::transform(idx, std::back_inserter(cloned),
                               [](const auto& i) { return i.clone(); });
        return TensorIndexer(this, std::move(cloned));
    }

    MaskedTensorProxy Tensor::operator[](const Tensor& mask) const {
        return MaskedTensorProxy(this, mask.clone());
    }

    // Element Access
    float& Tensor::at(std::initializer_list<size_t> indices) {
        static float dummy = 0;
        if (dtype_ != DataType::Float32 || indices.size() != shape_.rank()) {
            LOG_ERROR("at() type or dimension mismatch");
            return dummy;
        }

        std::vector<size_t> idx_vec(indices);

        size_t linear_idx = 0;
        auto strides = shape_.strides();

        for (size_t i = 0; i < idx_vec.size(); ++i) {
            if (idx_vec[i] >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {} with size {}",
                          idx_vec[i], i, shape_[i]);
                return dummy;
            }
            linear_idx += idx_vec[i] * strides[i];
        }

        if (device_ == Device::CUDA) {
            LOG_ERROR("Cannot get mutable reference to CUDA tensor element");
            return dummy;
        }
        return ptr<float>()[linear_idx];
    }

    float Tensor::at(std::initializer_list<size_t> indices) const {
        if (dtype_ != DataType::Float32 || indices.size() != shape_.rank()) {
            LOG_ERROR("at() type or dimension mismatch");
            return 0;
        }

        std::vector<size_t> idx_vec(indices);

        size_t linear_idx = 0;
        auto strides = shape_.strides();

        for (size_t i = 0; i < idx_vec.size(); ++i) {
            if (idx_vec[i] >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {} with size {}",
                          idx_vec[i], i, shape_[i]);
                return 0;
            }
            linear_idx += idx_vec[i] * strides[i];
        }

        if (device_ == Device::CUDA) {
            float value;
            cudaError_t err = cudaMemcpy(&value, ptr<float>() + linear_idx, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA memcpy failed in at() const: {}", cudaGetErrorString(err));
                return 0;
            }
            return value;
        }
        return ptr<float>()[linear_idx];
    }

    // From Vector
    template <typename T>
    static Tensor from_vector_impl(const std::vector<T>& data, TensorShape shape,
                                   Device device, DataType dtype) {
        if (shape.elements() != data.size())
            return {};
        auto t = Tensor::empty(shape, device, dtype);
        if (!t.is_valid() || t.numel() == 0)
            return t;

        if (t.numel() > 0 && data.data() != nullptr) {
            if (device == Device::CUDA) {
                cudaMemcpy(t.raw_ptr(), data.data(), t.bytes(), cudaMemcpyHostToDevice);
            } else {
                std::memcpy(t.raw_ptr(), data.data(), t.bytes());
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
        if (shape.elements() != data.size())
            return {};

        std::vector<unsigned char> bytes(data.size());
        std::ranges::transform(data, bytes.begin(),
                               [](bool b) { return b ? 1 : 0; });

        return from_vector_impl(bytes, shape, device, DataType::Bool);
    }

    void Tensor::set_bool(std::initializer_list<size_t> indices, bool value) {
        if (dtype_ != DataType::Bool)
            return;

        auto strides = shape_.strides();
        std::vector<size_t> idx_vec(indices);

        size_t idx = 0;
        for (size_t i = 0; i < idx_vec.size(); ++i) {
            idx += idx_vec[i] * strides[i];
        }

        unsigned char val = value ? 1 : 0;
        if (device_ == Device::CUDA) {
            cudaMemcpy(ptr<unsigned char>() + idx, &val, 1, cudaMemcpyHostToDevice);
        } else {
            ptr<unsigned char>()[idx] = val;
        }
    }

    bool Tensor::get_bool(std::initializer_list<size_t> indices) const {
        if (dtype_ != DataType::Bool)
            return false;

        auto strides = shape_.strides();
        std::vector<size_t> idx_vec(indices);

        size_t idx = 0;
        for (size_t i = 0; i < idx_vec.size(); ++i) {
            idx += idx_vec[i] * strides[i];
        }

        if (device_ == Device::CUDA) {
            unsigned char val;
            cudaMemcpy(&val, ptr<unsigned char>() + idx, 1, cudaMemcpyDeviceToHost);
            return val != 0;
        }
        return ptr<unsigned char>()[idx] != 0;
    }

    // Location: After the existing get_bool/set_bool implementations (around line 800+)
    // grep -C 3 "bool Tensor::get_bool"

    void Tensor::set_bool(std::span<const size_t> indices, bool value) {
        if (dtype_ != DataType::Bool) {
            LOG_ERROR("set_bool() only works on boolean tensors, got {}", dtype_name(dtype_));
            return;
        }

        if (indices.size() != shape_.rank()) {
            LOG_ERROR("set_bool() requires {} indices, got {}", shape_.rank(), indices.size());
            return;
        }

        // Calculate linear index from multi-dimensional indices
        auto strides = shape_.strides();

        size_t linear_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {} with size {}",
                          indices[i], i, shape_[i]);
                return;
            }
            linear_idx += indices[i] * strides[i];
        }

        unsigned char val = value ? 1 : 0;

        if (device_ == Device::CUDA) {
            cudaError_t err = cudaMemcpy(
                ptr<unsigned char>() + linear_idx,
                &val,
                1,
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA memcpy failed in set_bool: {}", cudaGetErrorString(err));
            }
        } else {
            ptr<unsigned char>()[linear_idx] = val;
        }
    }

    bool Tensor::get_bool(std::span<const size_t> indices) const {
        if (dtype_ != DataType::Bool) {
            LOG_ERROR("get_bool() only works on boolean tensors, got {}", dtype_name(dtype_));
            return false;
        }

        if (indices.size() != shape_.rank()) {
            LOG_ERROR("get_bool() requires {} indices, got {}", shape_.rank(), indices.size());
            return false;
        }

        // Calculate linear index from multi-dimensional indices
        auto strides = shape_.strides();

        size_t linear_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {} with size {}",
                          indices[i], i, shape_[i]);
                return false;
            }
            linear_idx += indices[i] * strides[i];
        }

        if (device_ == Device::CUDA) {
            unsigned char val;
            cudaError_t err = cudaMemcpy(
                &val,
                ptr<unsigned char>() + linear_idx,
                1,
                cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA memcpy failed in get_bool: {}", cudaGetErrorString(err));
                return false;
            }
            return val != 0;
        } else {
            return ptr<unsigned char>()[linear_idx] != 0;
        }
    }

    // Proxy Implementations
    void MaskedTensorProxy::operator=(float value) {
        const_cast<Tensor*>(tensor_)->masked_fill_(mask_, value);
    }

    void MaskedTensorProxy::operator=(const Tensor& other) {
        auto selected = tensor_->masked_select(mask_);
        if (selected.numel() != other.numel())
            return;

        if (tensor_->device() == Device::CUDA) {
            tensor_ops::launch_masked_scatter(const_cast<Tensor*>(tensor_)->ptr<float>(),
                                              mask_.ptr<unsigned char>(), other.ptr<float>(),
                                              tensor_->numel(), other.numel(), 0);
            cudaDeviceSynchronize();
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
        return tensor_->masked_select(mask_);
    }

    void TensorIndexer::operator=(float value) {
        if (indices_.size() == 1) {
            if (indices_[0].dtype() == DataType::Bool) {
                tensor_->masked_fill_(indices_[0], value);
            } else {
                tensor_->scatter_(0, indices_[0], value);
            }
        }
    }

    void TensorIndexer::operator=(const Tensor& other) {
        if (indices_.size() == 1) {
            if (indices_[0].dtype() == DataType::Bool) {
                MaskedTensorProxy proxy(tensor_, std::move(indices_[0]));
                proxy = other;
            } else {
                tensor_->scatter_(0, indices_[0], other);
            }
        }
    }

    TensorIndexer::operator Tensor() const {
        if (indices_.size() == 1) {
            if (indices_[0].dtype() == DataType::Bool) {
                return tensor_->masked_select(indices_[0]);
            } else {
                return indices_[0].ndim() == 1 ? tensor_->index_select(0, indices_[0]) : tensor_->take(indices_[0]);
            }
        }
        return Tensor();
    }

#undef CHECK_CUDA

} // namespace lfs::core