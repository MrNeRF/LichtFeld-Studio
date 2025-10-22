/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

namespace lfs::core {

    // ============= PAIRWISE DISTANCE (CDIST) =============
    Tensor Tensor::cdist(const Tensor& other, float p) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for cdist");
            return Tensor();
        }

        if (ndim() != 2 || other.ndim() != 2) {
            LOG_ERROR("cdist requires 2D tensors, got {}D and {}D", ndim(), other.ndim());
            return Tensor();
        }

        if (size(1) != other.size(1)) {
            LOG_ERROR("Feature dimensions must match: {} vs {}", size(1), other.size(1));
            return Tensor();
        }

        size_t N = size(0);
        size_t M = other.size(0);
        size_t D = size(1);

        auto other_same_device = (other.device() == device_) ? other.clone() : other.to(device_);
        auto result = empty({N, M}, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_cdist(ptr<float>(), other_same_device.ptr<float>(),
                                     result.ptr<float>(), N, M, D, p, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* a_data = ptr<float>();
            const float* b_data = other_same_device.ptr<float>();
            float* out_data = result.ptr<float>();

            if (p == 2.0f) {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < M; ++j) {
                        float dist = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            float diff = a_data[i * D + d] - b_data[j * D + d];
                            dist += diff * diff;
                        }
                        out_data[i * M + j] = std::sqrt(dist);
                    }
                }
            } else if (p == 1.0f) {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < M; ++j) {
                        float dist = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            dist += std::abs(a_data[i * D + d] - b_data[j * D + d]);
                        }
                        out_data[i * M + j] = dist;
                    }
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < M; ++j) {
                        float dist = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            float diff = std::abs(a_data[i * D + d] - b_data[j * D + d]);
                            dist += std::pow(diff, p);
                        }
                        out_data[i * M + j] = std::pow(dist, 1.0f / p);
                    }
                }
            }
        }

        return result;
    }

    // ============= MIN/MAX WITH INDICES =============
    std::pair<Tensor, Tensor> Tensor::min_with_indices(int dim, bool keepdim) const {
        LOG_DEBUG("=== min_with_indices START ===");
        LOG_DEBUG("  Input shape: {}", shape_.str());
        LOG_DEBUG("  Input device: {}", device_name(device_));
        LOG_DEBUG("  Input dtype: {}", dtype_name(dtype_));
        LOG_DEBUG("  dim: {}, keepdim: {}", dim, keepdim);
        LOG_DEBUG("  numel: {}", numel());

        if (!is_valid() || numel() == 0) {
            LOG_ERROR("Invalid tensor or empty tensor");
            return {Tensor(), Tensor()};
        }

        // Move to CPU for computation
        LOG_DEBUG("  Moving to CPU if needed...");
        auto cpu_tensor = (device_ == Device::CPU) ? clone() : to(Device::CPU);
        LOG_DEBUG("  CPU tensor created, shape: {}", cpu_tensor.shape().str());

        // Resolve dimension
        dim = cpu_tensor.resolve_dim(dim);
        LOG_DEBUG("  Resolved dim: {}", dim);

        if (dim < 0 || dim >= static_cast<int>(cpu_tensor.ndim())) {
            LOG_ERROR("Invalid dimension: {} for rank {}", dim, cpu_tensor.ndim());
            return {Tensor(), Tensor()};
        }

        // Handle 1D scalar reduction
        if (cpu_tensor.ndim() == 1 && dim == 0 && !keepdim) {
            LOG_DEBUG("  1D scalar reduction path");
            auto values = cpu_tensor.to_vector();
            LOG_DEBUG("  Got {} values from vector", values.size());

            auto min_it = std::min_element(values.begin(), values.end());
            size_t min_idx = std::distance(values.begin(), min_it);
            LOG_DEBUG("  Min value: {}, index: {}", *min_it, min_idx);

            // Create scalar value tensor
            LOG_DEBUG("  Creating scalar value tensor...");
            auto val = Tensor::empty({1}, Device::CPU, dtype_);
            LOG_DEBUG("  Val tensor created, setting value...");
            *val.ptr<float>() = *min_it;
            LOG_DEBUG("  Value set, squeezing...");
            val = val.squeeze();
            LOG_DEBUG("  Val squeezed, shape: {}", val.shape().str());

            // Create scalar index tensor - CRITICAL: proper Int64 handling
            LOG_DEBUG("  Creating scalar index tensor with Int64...");
            auto idx = Tensor::empty({1}, Device::CPU, DataType::Int64);
            LOG_DEBUG("  Idx tensor created, bytes: {}", idx.bytes());
            LOG_DEBUG("  Idx tensor raw_ptr: {}", static_cast<void*>(idx.raw_ptr()));

            int64_t* idx_ptr = reinterpret_cast<int64_t*>(idx.raw_ptr());
            LOG_DEBUG("  Idx pointer obtained: {}", static_cast<void*>(idx_ptr));
            LOG_DEBUG("  Setting index value to: {}", static_cast<int64_t>(min_idx));
            *idx_ptr = static_cast<int64_t>(min_idx);
            LOG_DEBUG("  Index value set, squeezing...");
            idx = idx.squeeze();
            LOG_DEBUG("  Idx squeezed, shape: {}", idx.shape().str());

            // Move back to original device if needed
            if (device_ == Device::CUDA) {
                LOG_DEBUG("  Moving results to CUDA...");
                auto val_cuda = val.to(Device::CUDA);
                LOG_DEBUG("  Val moved to CUDA");
                auto idx_cuda = idx.to(Device::CUDA);
                LOG_DEBUG("  Idx moved to CUDA");
                LOG_DEBUG("=== min_with_indices END (1D scalar) ===");
                return {val_cuda, idx_cuda};
            }
            LOG_DEBUG("=== min_with_indices END (1D scalar CPU) ===");
            return {val, idx};
        }

        // Calculate output shape
        LOG_DEBUG("  Calculating output shape...");
        std::vector<size_t> out_shape;
        for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
            if (static_cast<int>(i) != dim) {
                out_shape.push_back(cpu_tensor.size(i));
            } else if (keepdim) {
                out_shape.push_back(1);
            }
        }
        if (out_shape.empty())
            out_shape.push_back(1);

        LOG_DEBUG("  Output shape: [{}]", [&]() {
            std::string s;
            for (size_t i = 0; i < out_shape.size(); ++i) {
                if (i > 0)
                    s += ", ";
                s += std::to_string(out_shape[i]);
            }
            return s;
        }());

        // Create output tensors
        LOG_DEBUG("  Creating output tensors...");
        auto values = Tensor::empty(TensorShape(out_shape), Device::CPU, dtype_);
        LOG_DEBUG("  Values tensor created, numel: {}, bytes: {}", values.numel(), values.bytes());

        auto indices = Tensor::empty(TensorShape(out_shape), Device::CPU, DataType::Int64);
        LOG_DEBUG("  Indices tensor created, numel: {}, bytes: {}", indices.numel(), indices.bytes());

        const float* src = cpu_tensor.ptr<float>();
        LOG_DEBUG("  Source pointer: {}", static_cast<const void*>(src));

        float* vals = values.ptr<float>();
        LOG_DEBUG("  Values pointer: {}", static_cast<void*>(vals));

        int64_t* idxs = reinterpret_cast<int64_t*>(indices.raw_ptr());
        LOG_DEBUG("  Indices pointer: {}", static_cast<void*>(idxs));

        // 2D optimized path
        if (cpu_tensor.ndim() == 2) {
            LOG_DEBUG("  Using 2D optimized path");
            size_t rows = cpu_tensor.size(0);
            size_t cols = cpu_tensor.size(1);
            LOG_DEBUG("  Rows: {}, Cols: {}", rows, cols);

            if (dim == 0) {
                LOG_DEBUG("  Reducing along dim 0 (rows)");
                // Reduce along rows -> output is (cols,) or (1, cols)
                for (size_t c = 0; c < cols; ++c) {
                    if (c % 100 == 0) {
                        LOG_DEBUG("    Processing column {}/{}", c, cols);
                    }
                    float min_val = src[c];
                    int64_t min_row = 0;
                    for (size_t r = 1; r < rows; ++r) {
                        float v = src[r * cols + c];
                        if (v < min_val) {
                            min_val = v;
                            min_row = static_cast<int64_t>(r);
                        }
                    }
                    vals[c] = min_val;
                    idxs[c] = min_row;
                }
                LOG_DEBUG("  Finished reducing along dim 0");
            } else {
                LOG_DEBUG("  Reducing along dim 1 (cols)");
                // Reduce along cols -> output is (rows,) or (rows, 1)
                for (size_t r = 0; r < rows; ++r) {
                    if (r % 100 == 0) {
                        LOG_DEBUG("    Processing row {}/{}", r, rows);
                    }
                    float min_val = src[r * cols];
                    int64_t min_col = 0;
                    for (size_t c = 1; c < cols; ++c) {
                        float v = src[r * cols + c];
                        if (v < min_val) {
                            min_val = v;
                            min_col = static_cast<int64_t>(c);
                        }
                    }
                    vals[r] = min_val;
                    idxs[r] = min_col;
                }
                LOG_DEBUG("  Finished reducing along dim 1");
            }
        } else {
            // N-dimensional case - general implementation
            LOG_DEBUG("  Using N-D general path for rank {}", cpu_tensor.ndim());
            size_t dim_size = cpu_tensor.size(dim);
            size_t outer_size = 1;
            size_t inner_size = 1;

            for (int i = 0; i < dim; ++i) {
                outer_size *= cpu_tensor.size(i);
            }
            for (size_t i = dim + 1; i < cpu_tensor.ndim(); ++i) {
                inner_size *= cpu_tensor.size(i);
            }

            LOG_DEBUG("  dim_size: {}, outer_size: {}, inner_size: {}", dim_size, outer_size, inner_size);

            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    size_t base_idx = outer * dim_size * inner_size + inner;

                    float min_val = src[base_idx];
                    int64_t min_idx = 0;

                    for (size_t d = 1; d < dim_size; ++d) {
                        size_t idx = base_idx + d * inner_size;
                        if (src[idx] < min_val) {
                            min_val = src[idx];
                            min_idx = static_cast<int64_t>(d);
                        }
                    }

                    size_t out_idx = outer * inner_size + inner;
                    vals[out_idx] = min_val;
                    idxs[out_idx] = min_idx;
                }
            }
            LOG_DEBUG("  Finished N-D reduction");
        }

        LOG_DEBUG("  Reduction complete, preparing return values");
        LOG_DEBUG("  Values tensor valid: {}, numel: {}", values.is_valid(), values.numel());
        LOG_DEBUG("  Indices tensor valid: {}, numel: {}", indices.is_valid(), indices.numel());

        // Move back to original device if needed
        if (device_ == Device::CUDA) {
            LOG_DEBUG("  Moving results to CUDA...");
            auto values_cuda = values.to(Device::CUDA);
            LOG_DEBUG("  Values moved to CUDA");
            auto indices_cuda = indices.to(Device::CUDA);
            LOG_DEBUG("  Indices moved to CUDA");
            LOG_DEBUG("=== min_with_indices END (2D CUDA) ===");
            return {values_cuda, indices_cuda};
        }

        LOG_DEBUG("=== min_with_indices END (2D CPU) ===");
        return {values, indices};
    }

    std::pair<Tensor, Tensor> Tensor::max_with_indices(int dim, bool keepdim) const {
        LOG_DEBUG("=== max_with_indices START ===");
        LOG_DEBUG("  Input shape: {}", shape_.str());
        LOG_DEBUG("  Input device: {}", device_name(device_));
        LOG_DEBUG("  dim: {}, keepdim: {}", dim, keepdim);

        if (!is_valid()) {
            LOG_ERROR("max_with_indices on invalid tensor");
            return {Tensor(), Tensor()};
        }

        if (numel() == 0) {
            LOG_ERROR("max_with_indices on empty tensor");
            return {Tensor(), Tensor()};
        }

        // Move to CPU for computation
        LOG_DEBUG("  Moving to CPU if needed...");
        auto cpu_tensor = (device_ == Device::CPU) ? clone() : to(Device::CPU);
        LOG_DEBUG("  CPU tensor shape: {}", cpu_tensor.shape().str());

        // Resolve dimension
        dim = cpu_tensor.resolve_dim(dim);
        LOG_DEBUG("  Resolved dim: {}", dim);

        if (dim < 0 || dim >= static_cast<int>(cpu_tensor.ndim())) {
            LOG_ERROR("Invalid dimension for max_with_indices: {}", dim);
            return {Tensor(), Tensor()};
        }

        // For 1D tensors with dim=0 (returns scalar)
        if (cpu_tensor.ndim() == 1 && dim == 0 && !keepdim) {
            LOG_DEBUG("  1D scalar reduction path");
            auto values = cpu_tensor.to_vector();
            auto max_it = std::max_element(values.begin(), values.end());
            size_t max_idx = std::distance(values.begin(), max_it);

            auto val = Tensor::empty({1}, Device::CPU, dtype_);
            *val.ptr<float>() = *max_it;
            val = val.squeeze();

            auto idx = Tensor::empty({1}, Device::CPU, DataType::Int64);
            int64_t* idx_ptr = reinterpret_cast<int64_t*>(idx.raw_ptr());
            *idx_ptr = static_cast<int64_t>(max_idx);
            idx = idx.squeeze();

            if (device_ == Device::CUDA) {
                LOG_DEBUG("  Moving to CUDA...");
                return {val.to(Device::CUDA), idx.to(Device::CUDA)};
            }
            LOG_DEBUG("=== max_with_indices END (1D) ===");
            return {val, idx};
        }

        // Calculate output shape
        LOG_DEBUG("  Calculating output shape...");
        std::vector<size_t> out_shape;
        for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
            if (static_cast<int>(i) != dim || keepdim) {
                out_shape.push_back((static_cast<int>(i) == dim) ? 1 : cpu_tensor.size(i));
            }
        }
        if (out_shape.empty()) {
            out_shape.push_back(1);
        }

        LOG_DEBUG("  Creating output tensors...");
        auto values = Tensor::empty(TensorShape(out_shape), Device::CPU, dtype_);
        auto indices = Tensor::empty(TensorShape(out_shape), Device::CPU, DataType::Int64);

        const float* data = cpu_tensor.ptr<float>();
        float* val_data = values.ptr<float>();
        int64_t* idx_data = reinterpret_cast<int64_t*>(indices.raw_ptr());

        LOG_DEBUG("  Pointers - data: {}, val_data: {}, idx_data: {}",
                  static_cast<const void*>(data),
                  static_cast<void*>(val_data),
                  static_cast<void*>(idx_data));

        // Special case for 2D tensors
        if (cpu_tensor.ndim() == 2) {
            LOG_DEBUG("  Using 2D optimized path");
            size_t rows = cpu_tensor.size(0);
            size_t cols = cpu_tensor.size(1);
            LOG_DEBUG("  Rows: {}, Cols: {}", rows, cols);

            if (dim == 0) {
                LOG_DEBUG("  Reducing along dim 0 (rows)");
                // Max along rows (across columns)
                for (size_t col = 0; col < cols; ++col) {
                    if (col % 100 == 0) {
                        LOG_DEBUG("    Processing column {}/{}", col, cols);
                    }
                    float max_val = data[col];
                    int64_t max_idx = 0;

                    for (size_t row = 1; row < rows; ++row) {
                        float val = data[row * cols + col];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = static_cast<int64_t>(row);
                        }
                    }

                    val_data[col] = max_val;
                    idx_data[col] = max_idx;
                }
            } else if (dim == 1) {
                LOG_DEBUG("  Reducing along dim 1 (cols)");
                // Max along columns (across rows)
                for (size_t row = 0; row < rows; ++row) {
                    if (row % 100 == 0) {
                        LOG_DEBUG("    Processing row {}/{}", row, rows);
                    }
                    float max_val = data[row * cols];
                    int64_t max_idx = 0;

                    for (size_t col = 1; col < cols; ++col) {
                        float val = data[row * cols + col];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = static_cast<int64_t>(col);
                        }
                    }

                    val_data[row] = max_val;
                    idx_data[row] = max_idx;
                }
            }
            LOG_DEBUG("  Reduction complete");
        } else {
            // General N-dimensional case
            LOG_DEBUG("  Using N-D general path");
            size_t dim_size = cpu_tensor.size(dim);
            size_t outer_size = 1;
            size_t inner_size = 1;

            for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
                outer_size *= cpu_tensor.size(i);
            }
            for (size_t i = dim + 1; i < cpu_tensor.ndim(); ++i) {
                inner_size *= cpu_tensor.size(i);
            }

            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    size_t base_idx = outer * dim_size * inner_size + inner;

                    float max_val = data[base_idx];
                    int64_t max_idx = 0;

                    for (size_t d = 1; d < dim_size; ++d) {
                        size_t idx = base_idx + d * inner_size;
                        if (data[idx] > max_val) {
                            max_val = data[idx];
                            max_idx = static_cast<int64_t>(d);
                        }
                    }

                    size_t out_idx = outer * inner_size + inner;
                    val_data[out_idx] = max_val;
                    idx_data[out_idx] = max_idx;
                }
            }
        }

        LOG_DEBUG("  Moving to original device if needed...");
        if (device_ == Device::CUDA) {
            LOG_DEBUG("  Moving to CUDA...");
            auto values_cuda = values.to(Device::CUDA);
            LOG_DEBUG("  Values moved");
            auto indices_cuda = indices.to(Device::CUDA);
            LOG_DEBUG("  Indices moved");
            LOG_DEBUG("=== max_with_indices END (CUDA) ===");
            return {values_cuda, indices_cuda};
        }
        LOG_DEBUG("=== max_with_indices END (CPU) ===");
        return {values, indices};
    }

    // ============= SORTING =============
    std::pair<Tensor, Tensor> Tensor::sort(int dim, bool descending) const {
        if (!is_valid()) {
            LOG_ERROR("sort on invalid tensor");
            return {Tensor(), Tensor()};
        }

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(ndim())) {
            LOG_ERROR("Invalid dimension for sort: {}", dim);
            return {Tensor(), Tensor()};
        }

        // Create output tensors on same device
        auto sorted = clone();
        auto indices = Tensor::empty(shape_, device_, DataType::Int64);

        // 1D case - optimized path
        if (ndim() == 1 && dim == 0) {
            if (device_ == Device::CUDA) {
                tensor_ops::launch_sort_1d(sorted.ptr<float>(),
                                           reinterpret_cast<int64_t*>(indices.raw_ptr()),
                                           numel(), descending, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                // CPU fallback
                auto values_vec = to_vector();
                std::vector<size_t> idx_vec(values_vec.size());
                std::iota(idx_vec.begin(), idx_vec.end(), 0);

                if (descending) {
                    std::sort(idx_vec.begin(), idx_vec.end(),
                              [&](size_t a, size_t b) { return values_vec[a] > values_vec[b]; });
                } else {
                    std::sort(idx_vec.begin(), idx_vec.end(),
                              [&](size_t a, size_t b) { return values_vec[a] < values_vec[b]; });
                }

                float* sorted_data = sorted.ptr<float>();
                int64_t* idx_data = reinterpret_cast<int64_t*>(indices.raw_ptr());

                for (size_t i = 0; i < idx_vec.size(); ++i) {
                    sorted_data[i] = values_vec[idx_vec[i]];
                    idx_data[i] = static_cast<int64_t>(idx_vec[i]);
                }
            }

            return {sorted, indices};
        }

        // Multi-dimensional sort
        size_t dim_size = size(dim);
        size_t outer_size = 1;
        size_t inner_size = 1;

        for (int i = 0; i < dim; ++i) {
            outer_size *= size(i);
        }
        for (size_t i = dim + 1; i < ndim(); ++i) {
            inner_size *= size(i);
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_sort_2d(sorted.ptr<float>(),
                                       reinterpret_cast<int64_t*>(indices.raw_ptr()),
                                       outer_size, dim_size, inner_size,
                                       dim, descending, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            const float* src_data = ptr<float>();
            float* sorted_data = sorted.ptr<float>();
            int64_t* idx_data = reinterpret_cast<int64_t*>(indices.raw_ptr());

            // Sort each slice independently
            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    // Extract values for this slice
                    std::vector<std::pair<float, size_t>> slice_data(dim_size);
                    for (size_t d = 0; d < dim_size; ++d) {
                        size_t src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                        slice_data[d] = {src_data[src_idx], d};
                    }

                    // Sort the slice
                    if (descending) {
                        std::sort(slice_data.begin(), slice_data.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                    } else {
                        std::sort(slice_data.begin(), slice_data.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
                    }

                    // Write back sorted values and indices
                    for (size_t d = 0; d < dim_size; ++d) {
                        size_t dst_idx = outer * dim_size * inner_size + d * inner_size + inner;
                        sorted_data[dst_idx] = slice_data[d].first;
                        idx_data[dst_idx] = static_cast<int64_t>(slice_data[d].second);
                    }
                }
            }
        }

        return {sorted, indices};
    }

    // ============= SCALAR BOOLEAN REDUCTIONS =============
    bool Tensor::any_scalar() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }
        return count_nonzero() > 0;
    }

    bool Tensor::all_scalar() const {
        if (!is_valid() || numel() == 0) {
            return true;
        }
        return count_nonzero() == numel();
    }

#undef CHECK_CUDA

} // namespace lfs::core