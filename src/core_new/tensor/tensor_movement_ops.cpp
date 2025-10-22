/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <numeric>

namespace lfs::core {

    // ============= Helper: Infer dimension size =============
    static std::vector<size_t> infer_shape(const std::vector<int>& shape, size_t total_elements) {
        std::vector<size_t> result;
        int infer_dim = -1;
        size_t known_size = 1;

        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                if (infer_dim != -1) {
                    LOG_ERROR("Only one dimension can be inferred");
                    return {};
                }
                infer_dim = i;
                result.push_back(1); // Placeholder
            } else if (shape[i] < 0) {
                LOG_ERROR("Invalid reshape dimension: {}", shape[i]);
                return {};
            } else {
                result.push_back(shape[i]);
                known_size *= shape[i];
            }
        }

        if (infer_dim != -1) {
            if (total_elements % known_size != 0) {
                LOG_ERROR("Cannot infer dimension for reshape");
                return {};
            }
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
        if (!is_valid())
            return {};

        switch (op) {
        case MovementOp::Reshape: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                auto new_shape = infer_shape(*vec, numel());
                if (new_shape.empty())
                    return {};

                size_t total = 1;
                for (auto d : new_shape)
                    total *= d;

                if (total != numel()) {
                    LOG_ERROR("View shape {} has {} elements, but tensor has {} elements",
                              TensorShape(new_shape).str(), total, numel());
                    return {};
                }

                return create_view(TensorShape(new_shape));
            }
            LOG_ERROR("Reshape requires vector<int> args");
            return {};
        }

        case MovementOp::Permute: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                return permute(std::span<const int>(*vec));
            }
            LOG_ERROR("Permute requires vector<int> args");
            return {};
        }

        case MovementOp::Expand: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                std::vector<size_t> target_shape;
                for (int dim : *vec) {
                    target_shape.push_back(static_cast<size_t>(dim));
                }
                return expand(TensorShape(target_shape));
            }
            LOG_ERROR("Expand requires vector<int> args");
            return {};
        }

        case MovementOp::Transpose: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                int dim1 = resolve_dim(pair->first);
                int dim2 = resolve_dim(pair->second);

                if (dim1 < 0 || dim1 >= static_cast<int>(shape_.rank()) ||
                    dim2 < 0 || dim2 >= static_cast<int>(shape_.rank())) {
                    LOG_ERROR("Invalid transpose dimensions");
                    return {};
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

                // After transpose, usually non-contiguous
                view.is_contiguous_ = check_contiguous(view.shape_, view.strides_);

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

                    if (resolved < 0 || resolved >= static_cast<int>(shape_.rank())) {
                        LOG_ERROR("Squeeze dimension {} out of range for tensor with {} dimensions",
                                  dim, shape_.rank());
                        return {};
                    }

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

            LOG_ERROR("Squeeze requires int dim argument");
            return {};
        }

        case MovementOp::Unsqueeze: {
            if (auto* dim = std::get_if<int>(&args.args)) {
                int resolved = *dim;
                // For unsqueeze, negative dims are relative to NEW rank (after adding dimension)
                if (resolved < 0) {
                    resolved = static_cast<int>(shape_.rank()) + resolved + 1;
                }
                if (resolved < 0 || resolved > static_cast<int>(shape_.rank())) {
                    LOG_ERROR("Invalid unsqueeze dimension: {} for rank {}", *dim, shape_.rank());
                    return {};
                }

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
            LOG_ERROR("Unsqueeze requires int dim arg");
            return {};
        }

        case MovementOp::Flatten: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                int start = resolve_dim(pair->first);
                int end = resolve_dim(pair->second);

                if (start < 0 || start >= static_cast<int>(shape_.rank()) ||
                    end < 0 || end >= static_cast<int>(shape_.rank()) ||
                    start > end) {
                    LOG_ERROR("Invalid flatten dimensions");
                    return {};
                }

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
            LOG_ERROR("Slice requires vector<pair<int,int>> args");
            return {};
        }

        case MovementOp::Cat: {
            if (auto* cat_args = std::get_if<std::pair<void*, int>>(&args.args)) {
                const Tensor& other = *static_cast<const Tensor*>(cat_args->first);
                int dim = resolve_dim(cat_args->second);

                if (!other.is_valid() || shape_.rank() != other.shape().rank()) {
                    LOG_ERROR("Cannot concatenate tensors with different ranks");
                    return {};
                }

                for (size_t i = 0; i < shape_.rank(); ++i) {
                    if (i != static_cast<size_t>(dim) && shape_[i] != other.shape()[i]) {
                        LOG_ERROR("Dimension {} size mismatch for concatenation", i);
                        return {};
                    }
                }

                if (dim != 0) {
                    LOG_ERROR("Concatenation only implemented for dim=0");
                    return {};
                }

                std::vector<size_t> result_dims = shape_.dims();
                result_dims[dim] = shape_[dim] + other.shape()[dim];

                auto result = empty(TensorShape(result_dims), device_, dtype_);

                size_t self_bytes = bytes();
                size_t other_bytes = other.bytes();

                if (device_ == Device::CUDA) {
                    cudaMemcpy(result.raw_ptr(), raw_ptr(), self_bytes, cudaMemcpyDeviceToDevice);
                    cudaMemcpy(static_cast<char*>(result.raw_ptr()) + self_bytes,
                               other.raw_ptr(), other_bytes, cudaMemcpyDeviceToDevice);
                } else {
                    std::memcpy(result.raw_ptr(), raw_ptr(), self_bytes);
                    std::memcpy(static_cast<char*>(result.raw_ptr()) + self_bytes,
                                other.raw_ptr(), other_bytes);
                }

                return result;
            }
            LOG_ERROR("Cat requires (Tensor*, dim) pair");
            return {};
        }

        case MovementOp::Pad: {
            if (auto* padding = std::get_if<std::vector<std::pair<int, int>>>(&args.args)) {
                std::vector<size_t> new_shape = shape_.dims();
                std::vector<size_t> pad_before(shape_.rank(), 0);
                std::vector<size_t> pad_after(shape_.rank(), 0);

                for (size_t i = 0; i < padding->size() && i < shape_.rank(); ++i) {
                    pad_before[i] = (*padding)[i].first;
                    pad_after[i] = (*padding)[i].second;
                    new_shape[i] += pad_before[i] + pad_after[i];
                }

                auto result = zeros(TensorShape(new_shape), device_, dtype_);

                if (device_ == Device::CPU && dtype_ == DataType::Float32) {
                    const float* src = ptr<float>();
                    float* dst = result.ptr<float>();

                    auto src_strides = shape_.strides();
                    auto dst_strides = result.shape().strides();

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
                    LOG_WARN("Pad not fully implemented for CUDA");
                }

                return result;
            }
            LOG_ERROR("Pad requires vector<pair<int,int>> args");
            return {};
        }

        case MovementOp::Flip: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                auto result = clone();

                if (device_ == Device::CPU && dtype_ == DataType::Float32) {
                    float* data = result.ptr<float>();

                    for (int axis : *vec) {
                        axis = resolve_dim(axis);
                        if (axis < 0 || axis >= static_cast<int>(shape_.rank()))
                            continue;

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
            LOG_ERROR("Flip requires vector<int> axes");
            return {};
        }

        default:
            LOG_ERROR("Unknown movement operation");
            return {};
        }
    }

} // namespace lfs::core