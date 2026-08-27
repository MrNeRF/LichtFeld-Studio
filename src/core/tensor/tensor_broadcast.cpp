/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_broadcast.hpp"
#include "core/logger.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"

namespace lfs::core {

    namespace {

        /// Build expand/broadcast strides aligned from the right (NumPy / PyTorch).
        /// Broadcast dims (src size 1 → target > 1, or missing leading dims) get stride 0.
        std::vector<size_t> expand_strides(const Tensor& src, const TensorShape& target) {
            const auto src_dims = src.shape().dims();
            const auto& src_strides = src.strides();
            const auto target_dims = target.dims();
            const size_t src_rank = src_dims.size();
            const size_t tgt_rank = target_dims.size();

            std::vector<size_t> out(tgt_rank, 0);
            // Align from the right
            for (size_t i = 0; i < tgt_rank; ++i) {
                const size_t t_i = tgt_rank - 1 - i;
                if (i < src_rank) {
                    const size_t s_i = src_rank - 1 - i;
                    const size_t sd = src_dims[s_i];
                    const size_t td = target_dims[t_i];
                    if (sd == td) {
                        out[t_i] = src_strides[s_i];
                    } else if (sd == 1) {
                        out[t_i] = 0; // broadcast
                    } else {
                        // Should have been rejected by can_broadcast
                        out[t_i] = 0;
                    }
                } else {
                    // New leading dimension: always broadcast
                    out[t_i] = 0;
                }
            }
            return out;
        }

    } // namespace

    Tensor broadcast_to(const Tensor& src, const TensorShape& target) {
        LFS_ASSERT_MSG(src.is_valid(),
                       "Cannot broadcast an invalid tensor");

        // An empty dimension vector represents a scalar, not an incompatibility.
        if (src.shape() == target) {
            // Same shape: return a shallow handle (no clone).
            return src;
        }

        // Check if shapes are compatible for broadcasting
        auto src_dims = src.shape().dims();
        auto target_dims = target.dims();

        // Validate broadcasting rules
        auto broadcast_shape = broadcast::shape(src_dims, target_dims);
        LFS_ASSERT_MSG(!broadcast_shape.empty() && broadcast_shape == target_dims,
                       std::format("Cannot broadcast shape {} to {}", src.shape().str(), target.str()));

        if (src.numel() == 0 || target.elements() == 0) {
            return Tensor::empty(target, src.device(), src.dtype());
        }

        // zero-stride expand view — no device allocation.
        // Consumers that are not on the zero_stride allowlist materialize via
        // contiguous_read / contiguous() / dense-for-kernel at their boundary.
        auto new_strides = expand_strides(src, target);
        return src.create_broadcast_view(target, std::move(new_strides));
    }

} // namespace lfs::core
