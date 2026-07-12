/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <string>

namespace lfs::core {

    // ============= Helper: Infer dimension size =============
    static std::vector<size_t> infer_shape(const std::vector<int>& shape, size_t total_elements) {
        std::vector<size_t> result;
        int infer_dim = -1;
        size_t known_size = 1;

        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                LFS_ASSERT_MSG(infer_dim == -1,
                               std::format("reshape can infer only one dimension "
                                           "(second_infer_index={}, first_infer_index={}, "
                                           "dimension_count={}, total_elements={})",
                                           i, infer_dim, shape.size(), total_elements));
                infer_dim = i;
                result.push_back(1); // Placeholder
            } else {
                LFS_ASSERT_MSG(shape[i] >= 0,
                               std::format("reshape dimensions must be non-negative or -1 "
                                           "(dimension={}, value={}, dimension_count={})",
                                           i, shape[i], shape.size()));
                LFS_ASSERT_MSG(shape[i] == 0 ||
                                   known_size <= std::numeric_limits<size_t>::max() /
                                                     static_cast<size_t>(shape[i]),
                               std::format("reshape dimension product must not overflow size_t "
                                           "(dimension={}, value={}, product_before={}, "
                                           "size_t_max={})",
                                           i, shape[i], known_size,
                                           std::numeric_limits<size_t>::max()));
                result.push_back(shape[i]);
                known_size *= shape[i];
            }
        }

        if (infer_dim != -1) {
            LFS_ASSERT_MSG(known_size != 0 && total_elements % known_size == 0,
                           std::format("reshape inferred dimension must divide the element count "
                                       "exactly with a non-zero known product "
                                       "(infer_dimension={}, known_product={}, total_elements={}, "
                                       "remainder={})",
                                       infer_dim, known_size, total_elements,
                                       known_size == 0 ? total_elements : total_elements % known_size));
            result[infer_dim] = total_elements / known_size;
        }

        return result;
    }

    // ============= Helper: Check contiguity =============
    static bool check_contiguous(const TensorShape& shape, const std::vector<size_t>& strides) {
        if (strides.empty())
            return true;
        if (strides.size() != shape.rank())
            return false;

        // Check if strides match row-major contiguous layout
        size_t expected_stride = 1;
        for (int i = static_cast<int>(shape.rank()) - 1; i >= 0; --i) {
            if (strides[i] != expected_stride)
                return false;
            expected_stride *= shape[i];
        }
        return true;
    }

    // ============= Unified Movement Operation =============
    Tensor Tensor::movement(MovementOp op, const MovementArgs& args) const {
        LFS_ASSERT_MSG(is_valid(),
                       std::format("movement operation requires a valid input tensor "
                                   "(input={}, operation={}, argument_variant={})",
                                   str(), static_cast<int>(op), args.args.index()));

        switch (op) {
        case MovementOp::Reshape: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                auto new_shape = infer_shape(*vec, numel());
                LFS_ASSERT_MSG(!new_shape.empty(),
                               std::format("reshape requires at least one output dimension "
                                           "(requested_dimension_count={}, input_shape={}, "
                                           "input_numel={})",
                                           vec->size(), shape_.str(), numel()));

                size_t total = 1;
                for (auto d : new_shape)
                    total *= d;

                LFS_ASSERT_MSG(total == numel(),
                               std::format("reshape must preserve the element count "
                                           "(requested_elements={}, input_numel={}, "
                                           "requested_shape={}, input_shape={})",
                                           total, numel(), TensorShape(new_shape).str(), shape_.str()));

                return create_view(TensorShape(new_shape));
            }
            LFS_ASSERT_MSG(false,
                           std::format("reshape movement requires vector<int> arguments "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Permute: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                return permute(std::span<const int>(*vec));
            }
            LFS_ASSERT_MSG(false,
                           std::format("permute movement requires vector<int> arguments "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Expand: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                std::vector<size_t> target_shape;
                for (size_t i = 0; i < vec->size(); ++i) {
                    const int dim = (*vec)[i];
                    LFS_ASSERT_MSG(dim >= -1,
                                   std::format("expand dimensions must be non-negative or -1 "
                                               "(dimension={}, value={}, target_rank={}, "
                                               "input_shape={})",
                                               i, dim, vec->size(), shape_.str()));
                    target_shape.push_back(static_cast<size_t>(dim));
                }
                return expand(TensorShape(target_shape));
            }
            LFS_ASSERT_MSG(false,
                           std::format("expand movement requires vector<int> arguments "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Transpose: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                int dim1 = resolve_dim(pair->first);
                int dim2 = resolve_dim(pair->second);

                LFS_ASSERT_MSG(dim1 >= 0 && dim1 < static_cast<int>(shape_.rank()) &&
                                   dim2 >= 0 && dim2 < static_cast<int>(shape_.rank()),
                               std::format("transpose dimensions must be in range "
                                           "(requested_dimensions=[{},{}], "
                                           "resolved_dimensions=[{},{}], valid_range=[0,{}), "
                                           "input_shape={})",
                                           pair->first, pair->second, dim1, dim2,
                                           shape_.rank(), shape_.str()));

                if (state_ && state_->has_deferred_expr) {
                    std::vector<int> axes(shape_.rank());
                    std::iota(axes.begin(), axes.end(), 0);
                    std::swap(axes[dim1], axes[dim2]);
                    return permute(std::span<const int>(axes));
                }

                // ZERO-COPY TRANSPOSE: Just swap stride metadata!
                Tensor view;
                view.data_ = data_;
                view.data_owner_ = data_owner_; // Share ownership
                view.device_ = device_;
                view.dtype_ = dtype_;
                view.is_view_ = true;
                view.id_ = next_id_++;

                // Create new shape with swapped dimensions
                std::vector<size_t> new_dims = shape_.dims();
                std::swap(new_dims[dim1], new_dims[dim2]);
                view.shape_ = TensorShape(new_dims);

                // Swap strides (metadata-only operation!)
                view.strides_ = strides_;
                std::swap(view.strides_[dim1], view.strides_[dim2]);

                // Copy storage offset
                view.storage_offset_ = storage_offset_;

                view.is_contiguous_ = check_contiguous(view.shape_, view.strides_);
                propagate_view_meta(view);

                return view;
            }
            if (shape_.rank() < 2)
                return clone();
            return transpose(-2, -1);
        }

        case MovementOp::Squeeze: {
            if (auto* dim_ptr = std::get_if<int>(&args.args)) {
                int dim = *dim_ptr;
                std::vector<size_t> new_shape;

                // Check if this is "squeeze all" (using sentinel value)
                bool squeeze_all = (dim == std::numeric_limits<int>::min());

                if (squeeze_all) {
                    // Remove ALL dimensions of size 1
                    for (size_t i = 0; i < shape_.rank(); ++i) {
                        if (shape_[i] != 1) {
                            new_shape.push_back(shape_[i]);
                        }
                    }

                    // If all dims were 1, keep at least one dimension
                    if (new_shape.empty()) {
                        new_shape.push_back(1);
                    }
                } else {
                    // Squeeze specific dimension
                    int resolved = resolve_dim(dim);

                    LFS_ASSERT_MSG(resolved >= 0 && resolved < static_cast<int>(shape_.rank()),
                                   std::format("squeeze dimension must be in range "
                                               "(requested_dimension={}, resolved_dimension={}, "
                                               "valid_range=[0,{}), input_shape={})",
                                               dim, resolved, shape_.rank(), shape_.str()));

                    // Check if the dimension has size 1
                    if (shape_[resolved] != 1) {
                        LOG_WARN("Squeeze dimension {} has size {}, not 1. Returning clone.",
                                 dim, shape_[resolved]);
                        return clone();
                    }

                    // Build new shape without this dimension
                    for (size_t i = 0; i < shape_.rank(); ++i) {
                        if (i != static_cast<size_t>(resolved)) {
                            new_shape.push_back(shape_[i]);
                        }
                    }

                    // Ensure we have at least one dimension
                    if (new_shape.empty()) {
                        new_shape.push_back(1);
                    }
                }

                return create_view(TensorShape(new_shape));
            }

            LFS_ASSERT_MSG(false,
                           std::format("squeeze movement requires an integer dimension "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Unsqueeze: {
            if (auto* dim = std::get_if<int>(&args.args)) {
                int resolved = *dim;
                // For unsqueeze, negative dims are relative to NEW rank (after adding dimension)
                if (resolved < 0) {
                    resolved = static_cast<int>(shape_.rank()) + resolved + 1;
                }
                LFS_ASSERT_MSG(resolved >= 0 && resolved <= static_cast<int>(shape_.rank()),
                               std::format("unsqueeze dimension must be in the new-rank range "
                                           "(requested_dimension={}, resolved_dimension={}, "
                                           "valid_range=[0,{}], input_shape={})",
                                           *dim, resolved, shape_.rank(), shape_.str()));

                std::vector<size_t> new_shape;
                for (int i = 0; i < resolved; ++i) {
                    new_shape.push_back(shape_[i]);
                }
                new_shape.push_back(1);
                for (size_t i = resolved; i < shape_.rank(); ++i) {
                    new_shape.push_back(shape_[i]);
                }

                return create_view(TensorShape(new_shape));
            }
            LFS_ASSERT_MSG(false,
                           std::format("unsqueeze movement requires an integer dimension "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Flatten: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                int start = resolve_dim(pair->first);
                int end = resolve_dim(pair->second);

                LFS_ASSERT_MSG(start >= 0 && start < static_cast<int>(shape_.rank()) &&
                                   end >= 0 && end < static_cast<int>(shape_.rank()) && start <= end,
                               std::format("flatten dimensions must be ordered and in range "
                                           "(requested_start={}, requested_end={}, "
                                           "resolved_start={}, resolved_end={}, "
                                           "valid_range=[0,{}), input_shape={})",
                                           pair->first, pair->second, start, end,
                                           shape_.rank(), shape_.str()));

                std::vector<size_t> new_shape;
                for (int i = 0; i < start; ++i) {
                    new_shape.push_back(shape_[i]);
                }

                size_t flattened_size = 1;
                for (int i = start; i <= end; ++i) {
                    flattened_size *= shape_[i];
                }
                new_shape.push_back(flattened_size);

                for (size_t i = end + 1; i < shape_.rank(); ++i) {
                    new_shape.push_back(shape_[i]);
                }

                return create_view(TensorShape(new_shape));
            }
            return create_view(TensorShape({numel()}));
        }

        case MovementOp::Slice: {
            if (auto* ranges = std::get_if<std::vector<std::pair<int, int>>>(&args.args)) {
                return slice(std::span<const std::pair<int, int>>(*ranges));
            }
            LFS_ASSERT_MSG(false,
                           std::format("slice movement requires range arguments "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Cat: {
            if (auto* cat_args = std::get_if<std::pair<void*, int>>(&args.args)) {
                LFS_ASSERT_MSG(cat_args->first != nullptr,
                               std::format("cat movement requires a non-null source tensor pointer "
                                           "(source_pointer={}, requested_dimension={}, "
                                           "input_shape={})",
                                           cat_args->first, cat_args->second, shape_.str()));
                const Tensor& other = *static_cast<const Tensor*>(cat_args->first);
                int dim = resolve_dim(cat_args->second);
                LFS_ASSERT_MSG(other.is_valid(),
                               std::format("cat movement requires a valid source tensor "
                                           "(source={}, destination={})",
                                           other.str(), str()));
                LFS_ASSERT_MSG(shape_.rank() > 0,
                               std::format("cat movement cannot concatenate rank-0 tensors "
                                           "(destination_rank={}, destination_shape={}, "
                                           "source_shape={})",
                                           shape_.rank(), shape_.str(), other.shape().str()));
                LFS_ASSERT_MSG(dim == 0,
                               std::format("cat movement currently supports only dimension 0 "
                                           "(requested_dimension={}, resolved_dimension={}, "
                                           "destination_shape={}, source_shape={})",
                                           cat_args->second, dim, shape_.str(), other.shape().str()));
                LFS_ASSERT_MSG(shape_.rank() == other.shape().rank(),
                               std::format("cat movement tensor ranks must match "
                                           "(destination_rank={}, source_rank={}, "
                                           "destination_shape={}, source_shape={})",
                                           shape_.rank(), other.shape().rank(),
                                           shape_.str(), other.shape().str()));
                LFS_ASSERT_MSG(device_ == other.device(),
                               std::format("cat movement tensors must share a device "
                                           "(destination_device={}, source_device={})",
                                           device_name(device_), device_name(other.device())));
                LFS_ASSERT_MSG(dtype_ == other.dtype(),
                               std::format("cat movement tensor dtypes must match "
                                           "(destination_dtype={}({}), source_dtype={}({}))",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           dtype_name(other.dtype()), static_cast<int>(other.dtype())));

                for (size_t i = 0; i < shape_.rank(); ++i) {
                    LFS_ASSERT_MSG(i == static_cast<size_t>(dim) ||
                                       shape_[i] == other.shape()[i],
                                   std::format("cat movement non-concatenated dimensions must match "
                                               "(dimension={}, concatenate_dimension={}, "
                                               "destination_size={}, source_size={}, "
                                               "destination_shape={}, source_shape={})",
                                               i, dim, shape_[i], other.shape()[i],
                                               shape_.str(), other.shape().str()));
                }

                LFS_ASSERT_MSG(shape_[0] <=
                                   std::numeric_limits<size_t>::max() - other.shape()[0],
                               std::format("cat movement concatenated dimension must not overflow size_t "
                                           "(destination_size={}, source_size={}, size_t_max={})",
                                           shape_[0], other.shape()[0],
                                           std::numeric_limits<size_t>::max()));

                std::vector<size_t> result_dims = shape_.dims();
                result_dims[dim] = shape_[dim] + other.shape()[dim];

                auto result = empty(TensorShape(result_dims), device_, dtype_);

                size_t self_bytes = bytes();
                size_t other_bytes = other.bytes();

                if (device_ == Device::CUDA) {
                    if (self_bytes > 0) {
                        const cudaError_t self_status =
                            cudaMemcpy(result.data_ptr(), data_ptr(), self_bytes,
                                       cudaMemcpyDeviceToDevice);
                        ensure_cuda_success(
                            self_status, "cudaMemcpy(cat movement destination)",
                            std::format("bytes={}, destination_shape={}, result_shape={}",
                                        self_bytes, shape_.str(), result.shape().str()));
                    }
                    if (other_bytes > 0) {
                        const cudaError_t other_status =
                            cudaMemcpy(static_cast<char*>(result.data_ptr()) + self_bytes,
                                       other.data_ptr(), other_bytes,
                                       cudaMemcpyDeviceToDevice);
                        ensure_cuda_success(
                            other_status, "cudaMemcpy(cat movement source)",
                            std::format("bytes={}, source_shape={}, result_shape={}, "
                                        "destination_offset={}",
                                        other_bytes, other.shape().str(), result.shape().str(),
                                        self_bytes));
                    }
                } else {
                    if (self_bytes > 0) {
                        std::memcpy(result.data_ptr(), data_ptr(), self_bytes);
                    }
                    if (other_bytes > 0) {
                        std::memcpy(static_cast<char*>(result.data_ptr()) + self_bytes,
                                    other.data_ptr(), other_bytes);
                    }
                }

                return result;
            }
            LFS_ASSERT_MSG(false,
                           std::format("cat movement requires tensor-pointer and dimension arguments "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Pad: {
            if (auto* padding = std::get_if<std::vector<std::pair<int, int>>>(&args.args)) {
                LFS_ASSERT_MSG(padding->size() <= shape_.rank(),
                               std::format("pad entry count must not exceed tensor rank "
                                           "(padding_count={}, tensor_rank={}, input_shape={})",
                                           padding->size(), shape_.rank(), shape_.str()));
                LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                               std::format("pad currently requires Float32 input "
                                           "(input_dtype={}({}), input_shape={}, input_device={})",
                                           dtype_name(dtype_), static_cast<int>(dtype_),
                                           shape_.str(), device_name(device_)));
                std::vector<size_t> new_shape = shape_.dims();
                std::vector<size_t> pad_before(shape_.rank(), 0);
                std::vector<size_t> pad_after(shape_.rank(), 0);

                for (size_t i = 0; i < padding->size() && i < shape_.rank(); ++i) {
                    LFS_ASSERT_MSG((*padding)[i].first >= 0 && (*padding)[i].second >= 0,
                                   std::format("pad widths must be non-negative "
                                               "(dimension={}, before={}, after={}, input_shape={})",
                                               i, (*padding)[i].first, (*padding)[i].second,
                                               shape_.str()));
                    pad_before[i] = (*padding)[i].first;
                    pad_after[i] = (*padding)[i].second;
                    new_shape[i] += pad_before[i] + pad_after[i];
                }

                auto result = zeros(TensorShape(new_shape), device_, dtype_);

                if (device_ == Device::CUDA && dtype_ == DataType::Float32) {
                    tensor_ops::launch_pad(
                        ptr<float>(), result.ptr<float>(),
                        shape_.dims().data(), strides_.data(),
                        new_shape.data(), pad_before.data(),
                        shape_.rank(), numel(), result.stream());
                } else if (device_ == Device::CPU && dtype_ == DataType::Float32) {
                    if (!is_contiguous())
                        return contiguous().movement(op, args);

                    const float* src = ptr<float>();
                    float* dst = result.ptr<float>();
                    const auto src_strides = shape_.strides();
                    const auto dst_strides = result.shape().strides();

                    for (size_t i = 0; i < numel(); ++i) {
                        std::vector<size_t> coords(shape_.rank());
                        size_t temp = i;
                        for (size_t d = 0; d < shape_.rank(); ++d) {
                            coords[d] = temp / src_strides[d];
                            temp %= src_strides[d];
                        }
                        size_t dst_idx = 0;
                        for (size_t d = 0; d < shape_.rank(); ++d) {
                            dst_idx += (coords[d] + pad_before[d]) * dst_strides[d];
                        }
                        dst[dst_idx] = src[i];
                    }
                } else {
                    LOG_WARN("Pad: unsupported dtype/device");
                }

                return result;
            }
            LFS_ASSERT_MSG(false,
                           std::format("pad movement requires width-pair arguments "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        case MovementOp::Flip: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                LFS_ASSERT_MSG(device_ == Device::CPU && dtype_ == DataType::Float32,
                               std::format("flip currently requires a CPU Float32 input "
                                           "(input_device={}, input_dtype={}({}), input_shape={})",
                                           device_name(device_), dtype_name(dtype_),
                                           static_cast<int>(dtype_), shape_.str()));
                auto result = clone();

                if (device_ == Device::CPU && dtype_ == DataType::Float32) {
                    float* data = result.ptr<float>();

                    for (int axis : *vec) {
                        const int requested_axis = axis;
                        axis = resolve_dim(axis);
                        LFS_ASSERT_MSG(axis >= 0 && axis < static_cast<int>(shape_.rank()),
                                       std::format("flip axis must be in range "
                                                   "(requested_axis={}, resolved_axis={}, "
                                                   "valid_range=[0,{}), input_shape={})",
                                                   requested_axis, axis, shape_.rank(), shape_.str()));

                        size_t stride = 1;
                        for (size_t i = axis + 1; i < shape_.rank(); ++i) {
                            stride *= shape_[i];
                        }

                        size_t outer_size = 1;
                        for (int i = 0; i < axis; ++i) {
                            outer_size *= shape_[i];
                        }

                        for (size_t o = 0; o < outer_size; ++o) {
                            for (size_t i = 0; i < shape_[axis] / 2; ++i) {
                                size_t j = shape_[axis] - 1 - i;

                                for (size_t inner = 0; inner < stride; ++inner) {
                                    size_t idx1 = o * shape_[axis] * stride + i * stride + inner;
                                    size_t idx2 = o * shape_[axis] * stride + j * stride + inner;
                                    std::swap(data[idx1], data[idx2]);
                                }
                            }
                        }
                    }
                } else {
                    LOG_WARN("Flip not fully implemented for CUDA");
                }

                return result;
            }
            LFS_ASSERT_MSG(false,
                           std::format("flip movement requires axis-vector arguments "
                                       "(argument_variant={}, operation={})",
                                       args.args.index(), static_cast<int>(op)));
        }

        default:
            LFS_ASSERT_MSG(false,
                           std::format("tensor movement received an unknown operation enum "
                                       "(operation={}, argument_variant={}, input_shape={})",
                                       static_cast<int>(op), args.args.index(), shape_.str()));
        }
    }

} // namespace lfs::core
