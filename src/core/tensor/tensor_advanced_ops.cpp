/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace lfs::core {

    // ============= PAIRWISE DISTANCE (CDIST) =============
    Tensor Tensor::cdist(const Tensor& other, float p) const {
        LFS_ASSERT_MSG(is_valid() && other.is_valid(),
                       "cdist requires valid tensors");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32 && other.dtype() == DataType::Float32,
                       "cdist currently supports only Float32 tensors");
        LFS_ASSERT_MSG(device_ == other.device(),
                       "cdist requires tensors on the same device");
        LFS_ASSERT_MSG(ndim() == 2 && other.ndim() == 2,
                       "cdist requires rank-2 tensors");
        LFS_ASSERT_MSG(size(1) == other.size(1),
                       "cdist feature dimensions must match");
        LFS_ASSERT_MSG(!std::isnan(p) && p >= 0.0f,
                       "cdist p must be non-negative");

        Tensor lhs_materialized;
        Tensor rhs_materialized;
        const Tensor& lhs = contiguous_read(lhs_materialized);
        const Tensor& rhs = other.contiguous_read(rhs_materialized);

        size_t N = size(0);
        size_t M = other.size(0);
        size_t D = size(1);

        auto result = empty({N, M}, device_, dtype_);

        if (device_ == Device::CUDA) {
            pin_operands({&lhs, &rhs});
            const cudaStream_t execution_stream =
                prepare_inputs_for_stream({&lhs, &rhs}, result.stream());
            tensor_ops::launch_cdist(lhs.ptr<float>(), rhs.ptr<float>(),
                                     result.ptr<float>(), N, M, D, p, execution_stream);
            // No sync - returns tensor
        } else {
            pin_operands({&lhs, &rhs});
            const float* a_data = lhs.ptr<float>();
            const float* b_data = rhs.ptr<float>();
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
            } else if (p == 0.0f) {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < M; ++j) {
                        float dist = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            dist += a_data[i * D + d] != b_data[j * D + d] ? 1.0f : 0.0f;
                        }
                        out_data[i * M + j] = dist;
                    }
                }
            } else if (std::isinf(p)) {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < M; ++j) {
                        float dist = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            const float diff = std::abs(a_data[i * D + d] - b_data[j * D + d]);
                            dist = ops::maximum_op{}(dist, diff);
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

    namespace {
        std::pair<Tensor, Tensor> indexed_extreme_with_indices(
            const Tensor& input, int dim, const bool keepdim, const bool find_maximum) {
            LFS_ASSERT_MSG(input.is_valid(),
                           "indexed extrema require a valid tensor");
            LFS_ASSERT_MSG(input.dtype() == DataType::Float32,
                           "indexed extrema currently support only Float32");

            const size_t rank = input.ndim();
            if (rank == 0) {
                LFS_ASSERT_MSG(dim == 0 || dim == -1,
                               "scalar indexed-extrema dimension is out of range");
                dim = 0;
            } else {
                if (dim < 0)
                    dim += static_cast<int>(rank);
                LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(rank),
                               "indexed-extrema dimension is out of range");
            }

            Tensor cpu = input.device() == Device::CPU
                             ? input.contiguous()
                             : input.to(Device::CPU).contiguous();
            const size_t dim_size = rank == 0 ? 1 : cpu.size(static_cast<size_t>(dim));

            std::vector<size_t> output_shape;
            if (rank > 0) {
                output_shape.reserve(rank - (keepdim ? 0 : 1));
                for (size_t axis = 0; axis < rank; ++axis) {
                    if (static_cast<int>(axis) == dim) {
                        if (keepdim)
                            output_shape.push_back(1);
                    } else {
                        output_shape.push_back(cpu.size(axis));
                    }
                }
            }

            Tensor values = Tensor::empty(
                TensorShape(output_shape), Device::CPU, DataType::Float32);
            Tensor indices = Tensor::empty(
                TensorShape(output_shape), Device::CPU, DataType::Int64);
            if (values.numel() > 0) {
                LFS_ASSERT_MSG(dim_size > 0,
                               "indexed extrema cannot reduce an empty dimension");

                size_t outer_size = 1;
                size_t inner_size = 1;
                if (rank > 0) {
                    for (int axis = 0; axis < dim; ++axis) {
                        outer_size *= cpu.size(static_cast<size_t>(axis));
                    }
                    for (size_t axis = static_cast<size_t>(dim) + 1;
                         axis < rank; ++axis) {
                        inner_size *= cpu.size(axis);
                    }
                }

                const float* src = cpu.ptr<float>();
                float* dst_values = values.ptr<float>();
                int64_t* dst_indices = indices.ptr<int64_t>();
                for (size_t outer = 0; outer < outer_size; ++outer) {
                    for (size_t inner = 0; inner < inner_size; ++inner) {
                        const size_t base = outer * dim_size * inner_size + inner;
                        float selected = src[base];
                        int64_t selected_index = 0;
                        for (size_t index = 1; index < dim_size; ++index) {
                            const float candidate = src[base + index * inner_size];
                            if (std::isnan(candidate)) {
                                selected = candidate;
                                selected_index = static_cast<int64_t>(index);
                                break;
                            }
                            if (!std::isnan(selected) &&
                                (find_maximum ? candidate > selected
                                              : candidate < selected)) {
                                selected = candidate;
                                selected_index = static_cast<int64_t>(index);
                            }
                        }
                        const size_t output_index = outer * inner_size + inner;
                        dst_values[output_index] = selected;
                        dst_indices[output_index] = selected_index;
                    }
                }
            }

            if (input.device() == Device::CUDA) {
                return {values.to(Device::CUDA), indices.to(Device::CUDA)};
            }
            return {values, indices};
        }
    } // namespace

    // ============= MIN/MAX WITH INDICES =============
    std::pair<Tensor, Tensor> Tensor::min_with_indices(int dim, bool keepdim) const {
        return indexed_extreme_with_indices(*this, dim, keepdim, false);
    }

    std::pair<Tensor, Tensor> Tensor::max_with_indices(int dim, bool keepdim) const {
        return indexed_extreme_with_indices(*this, dim, keepdim, true);
    }
    // ============= SORTING =============
    std::pair<Tensor, Tensor> Tensor::sort(int dim, bool descending) const {
        LFS_ASSERT_MSG(is_valid(),
                       "sort requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "sort currently supports only Float32");
        if (ndim() == 0) {
            LFS_ASSERT_MSG(dim == 0 || dim == -1,
                           "scalar sort dimension is out of range");
            return {clone(), Tensor::zeros(TensorShape{}, device_, DataType::Int64)};
        }

        dim = resolve_dim(dim);
        LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(ndim()),
                       "sort dimension is out of range");

        Tensor materialized;
        const Tensor& source = contiguous_read(materialized);

        // Create output tensors on same device
        auto sorted = source.clone();
        auto indices = Tensor::empty(shape_, device_, DataType::Int64);
        if (numel() == 0)
            return {sorted, indices};

        // 1D case - optimized path
        if (ndim() == 1 && dim == 0) {
            if (device_ == Device::CUDA) {
                tensor_ops::launch_sort_1d(sorted.ptr<float>(),
                                           reinterpret_cast<int64_t*>(indices.data_ptr()),
                                           numel(), descending, 0);
                // No sync - returns tensors
            } else {
                // CPU fallback
                auto values_vec = to_vector();
                std::vector<size_t> idx_vec(values_vec.size());
                std::iota(idx_vec.begin(), idx_vec.end(), 0);

                if (descending) {
                    std::stable_sort(idx_vec.begin(), idx_vec.end(),
                                     [&](size_t a, size_t b) {
                                         return ops::sort_greater_op{}(values_vec[a], values_vec[b]);
                                     });
                } else {
                    std::stable_sort(idx_vec.begin(), idx_vec.end(),
                                     [&](size_t a, size_t b) {
                                         return ops::sort_less_op{}(values_vec[a], values_vec[b]);
                                     });
                }

                float* sorted_data = sorted.ptr<float>();
                int64_t* idx_data = reinterpret_cast<int64_t*>(indices.data_ptr());

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
                                       reinterpret_cast<int64_t*>(indices.data_ptr()),
                                       outer_size, dim_size, inner_size,
                                       dim, descending, 0);
            // No sync - returns tensors
        } else {
            // CPU implementation
            const float* src_data = source.ptr<float>();
            float* sorted_data = sorted.ptr<float>();
            int64_t* idx_data = reinterpret_cast<int64_t*>(indices.data_ptr());

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
                        std::stable_sort(slice_data.begin(), slice_data.end(),
                                         [](const auto& a, const auto& b) {
                                             return ops::sort_greater_op{}(a.first, b.first);
                                         });
                    } else {
                        std::stable_sort(slice_data.begin(), slice_data.end(),
                                         [](const auto& a, const auto& b) {
                                             return ops::sort_less_op{}(a.first, b.first);
                                         });
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
        LFS_ASSERT_MSG(is_valid(),
                       "any_scalar requires a valid tensor");
        if (numel() == 0) {
            return false;
        }
        return count_nonzero() > 0;
    }

    bool Tensor::all_scalar() const {
        LFS_ASSERT_MSG(is_valid(),
                       "all_scalar requires a valid tensor");
        if (numel() == 0) {
            return true;
        }
        return count_nonzero() == numel();
    }

} // namespace lfs::core
