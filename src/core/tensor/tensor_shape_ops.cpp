/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cstdint>
#include <execution>
#include <numeric>

#define CHECK_CUDA(call)                                        \
    do {                                                        \
        const cudaError_t error = (call);                       \
        LFS_ASSERT_MSG(error == cudaSuccess,                    \
                       std::string("CUDA operation failed: ") + \
                           cudaGetErrorString(error));          \
    } while (0)

namespace lfs::core {

    Tensor Tensor::reshape(TensorShape new_shape) const {
        LFS_ASSERT_MSG(is_valid(), "reshape requires a valid tensor");

        if (new_shape.rank() == 0 && numel() == 1) {
            return create_view(new_shape);
        }

        LFS_ASSERT_MSG(new_shape.elements() == numel(),
                       std::format("reshape element mismatch: requested {} elements for tensor with {}",
                                   new_shape.elements(), numel()));

        return create_view(new_shape);
    }

    Tensor Tensor::t() const {
        LFS_ASSERT_MSG(is_valid(), "t() requires a valid tensor");

        if (shape_.rank() <= 1) {
            return clone();
        }

        return transpose(-2, -1);
    }

    Tensor Tensor::permute(std::span<const int> axes) const {
        LFS_ASSERT_MSG(is_valid(), "permute requires a valid tensor");

        const size_t rank = shape_.rank();
        LFS_ASSERT_MSG(axes.size() == rank,
                       std::format("permute requires {} axes, got {}", rank, axes.size()));

        // Fast path: use stack allocation for common small ranks (up to 8D)
        constexpr size_t STACK_SIZE = 8;
        int resolved_axes_buf[STACK_SIZE];
        uint8_t used_buf[STACK_SIZE] = {};

        std::vector<int> resolved_axes_heap;
        std::vector<uint8_t> used_heap;

        int* resolved_axes;
        uint8_t* used;

        if (rank <= STACK_SIZE) {
            resolved_axes = resolved_axes_buf;
            used = used_buf;
        } else {
            resolved_axes_heap.resize(rank);
            used_heap.resize(rank, 0);
            resolved_axes = resolved_axes_heap.data();
            used = used_heap.data();
        }

        // Validate and resolve axes
        for (size_t i = 0; i < rank; ++i) {
            int resolved = resolve_dim(axes[i]);
            LFS_ASSERT_MSG(resolved >= 0 && resolved < static_cast<int>(rank),
                           std::format("permute axis {} is out of range for rank {}", axes[i], rank));
            LFS_ASSERT_MSG(!used[resolved],
                           std::format("permute axis {} is duplicated", axes[i]));
            used[resolved] = true;
            resolved_axes[i] = resolved;
        }

        std::vector<size_t> new_dims(rank);
        std::vector<size_t> new_strides(rank);

        for (size_t i = 0; i < rank; ++i) {
            new_dims[i] = shape_[resolved_axes[i]];
            new_strides[i] = strides_[resolved_axes[i]];
        }

        if (state_ && state_->has_deferred_expr) {
            std::vector<int> deferred_axes(rank);
            for (size_t i = 0; i < rank; ++i) {
                deferred_axes[i] = resolved_axes[i];
            }
            const uint64_t source_id = lazy_expr_id();
            Tensor source = *this;
            const cudaStream_t source_stream = source.stream();
            TensorShape deferred_shape(new_dims);
            std::vector<uint64_t> deferred_inputs;
            if (source_id != 0) {
                deferred_inputs.push_back(source_id);
            }
            Tensor deferred = make_deferred_expr_tensor(
                deferred_shape, device_, dtype_,
                [source = std::move(source), deferred_axes = std::move(deferred_axes)]() mutable {
                    Tensor materialized = source;
                    materialized.materialize_if_deferred();
                    return materialized.permute(std::span<const int>(deferred_axes));
                },
                std::move(deferred_inputs));
            deferred.set_stream(source_stream);
            return deferred;
        }

        // ZERO-COPY PERMUTE: Create a view with permuted dimensions and strides
        Tensor view;
        view.data_ = data_;
        view.data_owner_ = data_owner_; // Share ownership
        view.device_ = device_;
        view.dtype_ = dtype_;
        view.is_view_ = true;
        view.id_ = profiling_enabled_ ? next_id_++ : 0; // Only increment ID when profiling
        view.storage_offset_ = storage_offset_;

        view.shape_ = TensorShape(new_dims);
        view.strides_ = std::move(new_strides);

        // Check if the result is contiguous
        size_t expected_stride = 1;
        bool is_contiguous_result = true;
        for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
            if (view.strides_[i] != expected_stride) {
                is_contiguous_result = false;
                break;
            }
            expected_stride *= new_dims[i];
        }
        view.is_contiguous_ = is_contiguous_result;
        propagate_view_meta(view);

        return view;
    }

    Tensor Tensor::expand(const TensorShape& target_shape) const {
        LFS_ASSERT_MSG(is_valid(), "expand requires a valid tensor");

        LFS_ASSERT_MSG(target_shape.rank() >= shape_.rank(),
                       "expand cannot reduce tensor rank");

        std::vector<size_t> padded_shape = shape_.dims();
        while (padded_shape.size() < target_shape.rank()) {
            padded_shape.insert(padded_shape.begin(), 1);
        }

        std::vector<size_t> final_shape(target_shape.rank());
        for (size_t i = 0; i < target_shape.rank(); ++i) {
            size_t target_dim = target_shape[i];

            if (target_dim == static_cast<size_t>(-1)) {
                LFS_ASSERT_MSG(i < padded_shape.size(), "expand cannot use -1 for a new dimension");
                final_shape[i] = padded_shape[i];
            } else {
                LFS_ASSERT_MSG(padded_shape[i] == 1 || padded_shape[i] == target_dim,
                               std::format("expand cannot change dimension {} from {} to {}",
                                           i, padded_shape[i], target_dim));
                final_shape[i] = target_dim;
            }
        }

        auto reshaped = reshape(TensorShape(padded_shape));
        return reshaped.broadcast_to(TensorShape(final_shape));
    }

    Tensor Tensor::slice(std::span<const std::pair<int, int>> ranges) const {
        LFS_ASSERT_MSG(is_valid(), "slice requires a valid tensor");

        LFS_ASSERT_MSG(ranges.size() <= shape_.rank(), "slice has more ranges than tensor dimensions");

        std::vector<size_t> starts(shape_.rank());
        std::vector<size_t> ends(shape_.rank());

        for (size_t i = 0; i < shape_.rank(); ++i) {
            if (i < ranges.size()) {
                int start = ranges[i].first;
                int end = ranges[i].second;

                if (start < 0)
                    start = shape_[i] + start;
                if (end < 0)
                    end = shape_[i] + end;

                start = std::max(0, std::min(start, static_cast<int>(shape_[i])));
                end = std::max(start, std::min(end, static_cast<int>(shape_[i])));

                starts[i] = start;
                ends[i] = end;
            } else {
                starts[i] = 0;
                ends[i] = shape_[i];
            }
        }

        std::vector<size_t> new_shape;
        for (size_t i = 0; i < shape_.rank(); ++i) {
            new_shape.push_back(ends[i] - starts[i]);
        }

        if (state_ && state_->has_deferred_expr) {
            std::vector<std::pair<int, int>> deferred_ranges(ranges.begin(), ranges.end());
            const uint64_t source_id = lazy_expr_id();
            Tensor source = *this;
            const cudaStream_t source_stream = source.stream();
            TensorShape deferred_shape(new_shape);
            std::vector<uint64_t> deferred_inputs;
            if (source_id != 0) {
                deferred_inputs.push_back(source_id);
            }
            Tensor deferred = make_deferred_expr_tensor(
                deferred_shape, device_, dtype_,
                [source = std::move(source), deferred_ranges = std::move(deferred_ranges)]() mutable {
                    Tensor materialized = source;
                    materialized.materialize_if_deferred();
                    return materialized.slice(std::span<const std::pair<int, int>>(deferred_ranges));
                },
                std::move(deferred_inputs));
            deferred.set_stream(source_stream);
            return deferred;
        }

        bool is_contiguous = is_contiguous_slice(starts, ends);

        if (is_contiguous) {
            size_t offset = calculate_offset(starts);
            void* new_data = static_cast<char*>(data_) + offset * dtype_size(dtype_);

            Tensor view(new_data, TensorShape(new_shape), device_, dtype_);
            view.data_owner_ = data_owner_;
            view.is_view_ = true;
            propagate_view_meta(view);
            return view;
        } else {
            return copy_slice(starts, ends, new_shape);
        }
    }

    Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
        LFS_ASSERT_MSG(is_valid(), "slice requires a valid tensor");

        LFS_ASSERT_MSG(dim < shape_.rank(),
                       std::format("slice dimension {} is out of range for rank {}", dim, shape_.rank()));
        LFS_ASSERT_MSG(start < end && end <= shape_[dim],
                       std::format("slice range [{}, {}) is invalid for dimension {} of size {}",
                                   start, end, dim, shape_[dim]));

        std::vector<size_t> new_dims = shape_.dims();
        new_dims[dim] = end - start;

        if (state_ && state_->has_deferred_expr) {
            const uint64_t source_id = lazy_expr_id();
            Tensor source = *this;
            const cudaStream_t source_stream = source.stream();
            TensorShape deferred_shape(new_dims);
            std::vector<uint64_t> deferred_inputs;
            if (source_id != 0) {
                deferred_inputs.push_back(source_id);
            }
            Tensor deferred = make_deferred_expr_tensor(
                deferred_shape, device_, dtype_,
                [source = std::move(source), dim, start, end]() mutable {
                    Tensor materialized = source;
                    materialized.materialize_if_deferred();
                    return materialized.slice(dim, start, end);
                },
                std::move(deferred_inputs));
            deferred.set_stream(source_stream);
            return deferred;
        }

        // ZERO-COPY SLICE: Adjust offset and shape - NO DATA COPYING!
        Tensor view;
        view.data_ = data_;
        view.data_owner_ = data_owner_; // Share ownership
        view.strides_ = strides_;       // Keep same strides
        view.device_ = device_;
        view.dtype_ = dtype_;
        view.is_view_ = true;
        view.id_ = profiling_enabled_ ? next_id_++ : 0; // Only increment ID when profiling

        // Adjust offset to point to slice start (in elements)
        view.storage_offset_ = storage_offset_ + start * strides_[dim];

        // Adjust shape for the sliced dimension
        view.shape_ = TensorShape(new_dims);

        // Check if strides match the new shape's expected contiguous layout
        // A sliced tensor is contiguous if its strides match row-major order for its shape
        size_t expected_stride = 1;
        bool still_contiguous = true;
        for (int i = static_cast<int>(view.shape_.rank()) - 1; i >= 0; --i) {
            if (view.strides_[i] != expected_stride) {
                still_contiguous = false;
                break;
            }
            expected_stride *= view.shape_[i];
        }
        view.is_contiguous_ = still_contiguous;
        propagate_view_meta(view);

        return view;
    }

    bool Tensor::is_contiguous_slice(const std::vector<size_t>& starts,
                                     const std::vector<size_t>& ends) const {
        LFS_DEBUG_ASSERT(starts.size() == shape_.rank());
        LFS_DEBUG_ASSERT(ends.size() == shape_.rank());
        for (size_t i = 1; i < shape_.rank(); ++i) {
            if (starts[i] != 0 || ends[i] != shape_[i]) {
                return false;
            }
        }

        return true;
    }

    size_t Tensor::calculate_offset(const std::vector<size_t>& indices) const {
        LFS_DEBUG_ASSERT(indices.size() <= strides_.size());
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            LFS_DEBUG_ASSERT(indices[i] < shape_[i]);
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

    Tensor Tensor::copy_slice(const std::vector<size_t>& starts,
                              const std::vector<size_t>& ends,
                              const std::vector<size_t>& new_shape) const {
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "non-contiguous slice copying currently supports only Float32");
        auto result = empty(TensorShape(new_shape), device_, dtype_);

        if (device_ == Device::CUDA) {
            auto cpu_copy = to(Device::CPU);
            auto cpu_result = cpu_copy.copy_slice(starts, ends, new_shape);
            return cpu_result.to(Device::CUDA);
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();

            size_t total = 1;
            for (size_t s : new_shape) {
                total *= s;
            }

            std::vector<size_t> indices(shape_.rank());
            for (size_t i = 0; i < shape_.rank(); ++i) {
                indices[i] = starts[i];
            }

            for (size_t dst_idx = 0; dst_idx < total; ++dst_idx) {
                size_t src_idx = calculate_offset(indices);
                dst[dst_idx] = src[src_idx];

                for (int d = static_cast<int>(shape_.rank()) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < ends[d]) {
                        break;
                    }
                    indices[d] = starts[d];
                }
            }
        }

        return result;
    }

    std::vector<size_t> Tensor::resolve_dims(std::span<const int> dims) const {
        std::vector<size_t> resolved;
        resolved.reserve(dims.size());

        for (int dim : dims) {
            int r = resolve_dim(dim);
            LFS_ASSERT_MSG(r >= 0 && r < static_cast<int>(shape_.rank()),
                           std::format("dimension {} is out of range for rank {}", dim, shape_.rank()));
            resolved.push_back(static_cast<size_t>(r));
        }

        return resolved;
    }

#undef CHECK_CUDA

} // namespace lfs::core
