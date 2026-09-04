/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_backend_ops.hpp"

#include "core/assert.hpp"

#include <format>

namespace lfs::core::internal {
    namespace {
        [[noreturn]] void not_implemented(const char* const entry) {
            LFS_ASSERT_MSG(false,
                           std::format("Vulkan backend: {} is not implemented yet", entry));
            std::unreachable();
        }

    } // namespace

#define LFS_VK_NOTIMPL_VOID(Name, Parameters) \
    void VulkanBackendOps::Name Parameters { not_implemented(#Name); }
#define LFS_VK_NOTIMPL_FLOAT(Name, Parameters) \
    float VulkanBackendOps::Name Parameters { not_implemented(#Name); }
#define LFS_VK_NOTIMPL_SIZE(Name, Parameters) \
    size_t VulkanBackendOps::Name Parameters { not_implemented(#Name); }
#define LFS_VK_NOTIMPL_BOOL(Name, Parameters) \
    bool VulkanBackendOps::Name Parameters { not_implemented(#Name); }

    LFS_VK_NOTIMPL_FLOAT(sum_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_FLOAT(mean_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_FLOAT(max_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_FLOAT(min_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(reduce, (StorageRef, StorageRef, const StridedLayout&, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(column_reduce, (StorageRef, StorageRef, size_t, size_t, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(strided_reduce, (StorageRef, StorageRef, size_t, size_t, size_t, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(fused_transform_reduce, (StorageRef, StorageRef, size_t, const tensor_ops::FusedPointwiseOpChain&, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(fused_segmented_transform_reduce, (StorageRef, StorageRef, size_t, size_t, const tensor_ops::FusedPointwiseOpChain&, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(count_nonzero_bool, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_SIZE(count_nonzero_float, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_BOOL(has_nan, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_BOOL(has_inf, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(cumsum, (StorageRef, const StridedLayout&, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(sort_1d, (StorageRef, StorageRef, size_t, const SortProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sort_2d, (StorageRef, StorageRef, const SortProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_tn, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_batched, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_bias_relu, (StorageRef, StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(dot_product, (StorageRef, StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(diag, (StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(eye, (StorageRef, size_t, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(cdist, (StorageRef, StorageRef, StorageRef, size_t, size_t, size_t, float, ExecContext))
    LFS_VK_NOTIMPL_VOID(max_pool2d, (StorageRef, StorageRef, const PoolProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(adaptive_avg_pool2d, (StorageRef, StorageRef, const PoolProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(bias_add, (StorageRef, StorageRef, StorageRef, int, int, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(bias_relu, (StorageRef, StorageRef, StorageRef, int, int, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(relu, (StorageRef, StorageRef, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(uniform, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(bernoulli, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(randint, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(multinomial, (StorageRef, StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(normal, (StorageRef, StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(gather, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(gather_fused_unary, (StorageRef, StorageRef, StorageRef, PointwiseOp, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(take, (StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_select, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(scatter, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_copy, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_add, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_fill, (StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ScalarOperand, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_put, (StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(masked_fill, (StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(masked_select, (StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(masked_scatter, (StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(and_live, (StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(where, (StorageRef, StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const StridedLayout&, const StridedLayout&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(nonzero, (StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(nonzero_bool, (StorageRef, StorageRef, const MaskProgram&, ExecContext))

#undef LFS_VK_NOTIMPL_VOID
#undef LFS_VK_NOTIMPL_FLOAT
#undef LFS_VK_NOTIMPL_SIZE
#undef LFS_VK_NOTIMPL_BOOL

} // namespace lfs::core::internal
