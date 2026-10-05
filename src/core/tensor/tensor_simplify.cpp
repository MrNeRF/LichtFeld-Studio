/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_simplify.hpp"
#include "internal/tensor_impl.hpp"

#include <array>
#include <format>

namespace lfs::core {
    using namespace internal;

    std::optional<SimplifyRows> simplify_merge_groups(const SimplifyRows& rows, const Tensor& offsets,
                                                      const Tensor& members) {
        const size_t count = rows.means.is_valid() ? rows.means.size(0) : 0;
        LFS_ASSERT_MSG(rows.means.is_valid() && rows.means.device() == Device::GPU,
                       "simplify_merge_groups requires rows on a GPU backend");
        const auto valid = [&](const Tensor& tensor, const size_t width) {
            return tensor.is_valid() && tensor.dtype() == DataType::Float32 && tensor.device() == Device::GPU &&
                   tensor.size(0) == count && (width == 0 ? tensor.ndim() == 1 : tensor.ndim() == 2 && tensor.size(1) == width);
        };
        LFS_ASSERT_MSG(valid(rows.means, 3) && valid(rows.scales, 3) && valid(rows.rotation, 4) && valid(rows.opacity, 0) &&
                           rows.appearance.is_valid() && rows.appearance.ndim() == 2 && valid(rows.appearance, rows.appearance.size(1)),
                       "simplify_merge_groups requires Float32 rows [N,3], [N,3], [N,4], [N] and [N,A]");
        LFS_ASSERT_MSG(offsets.is_valid() && offsets.ndim() == 1 && offsets.dtype() == DataType::Int32 && offsets.numel() >= 1 &&
                           members.is_valid() && members.ndim() == 1 && members.dtype() == DataType::Int32,
                       "simplify_merge_groups requires Int32 [G+1] offsets and Int32 members");
        for (const auto* tensor : {&rows.scales, &rows.rotation, &rows.opacity, &rows.appearance, &offsets, &members})
            internal::require_same_gpu_backend(rows.means, *tensor, "simplify_merge_groups");
        const size_t groups = offsets.numel() - 1;
        const size_t app_dim = rows.appearance.size(1);
        SimplifyRows merged{
            internal::allocate_like(rows.means, TensorShape{groups, 3}, DataType::Float32),
            internal::allocate_like(rows.means, TensorShape{groups, 3}, DataType::Float32),
            internal::allocate_like(rows.means, TensorShape{groups, 4}, DataType::Float32),
            internal::allocate_like(rows.means, TensorShape{groups}, DataType::Float32),
            internal::allocate_like(rows.means, TensorShape{groups, app_dim}, DataType::Float32)};
        // An empty grouping still asks the backend, so callers can learn whether it has the kernel.
        const std::array inputs{rows.means.contiguous(), rows.scales.contiguous(), rows.rotation.contiguous(),
                                rows.opacity.contiguous(), rows.appearance.contiguous()};
        const auto group_offsets = offsets.contiguous();
        const auto group_members = members.contiguous();
        auto stream = merged.means.stream();
        if (groups != 0) {
            pin_operands({&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &group_offsets, &group_members,
                          &merged.means, &merged.scales, &merged.rotation, &merged.opacity, &merged.appearance});
            stream = prepare_inputs_for_stream({&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &group_offsets,
                                                &group_members, &merged.scales, &merged.rotation, &merged.opacity,
                                                &merged.appearance},
                                               merged.means.stream());
        }
        const std::array<StorageRef, 5> read{storage_ref(inputs[0]), storage_ref(inputs[1]), storage_ref(inputs[2]),
                                             storage_ref(inputs[3]), storage_ref(inputs[4])};
        const std::array<StorageRef, 5> write{storage_ref(merged.means), storage_ref(merged.scales),
                                              storage_ref(merged.rotation), storage_ref(merged.opacity),
                                              storage_ref(merged.appearance)};
        if (!backend_ops_for(rows.means)
                 .simplify_merge(read, storage_ref(group_offsets), storage_ref(group_members), write,
                                 SimplifyMergeProgram{static_cast<uint32_t>(groups), static_cast<uint32_t>(app_dim)},
                                 ExecContext{stream}))
            return std::nullopt;
        return merged;
    }
} // namespace lfs::core
