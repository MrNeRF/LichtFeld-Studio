/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_labels.hpp"
#include "internal/point_labels.hpp"
#include "internal/tensor_impl.hpp"
#include <array>
#include <cstring>
#include <limits>

namespace lfs::core {
    void update_labels(Tensor& output, const Tensor& selected, const LabelUpdate& update) {
        const auto byte_vector = [](const Tensor& value) {
            return value.is_valid() && value.ndim() == 1 &&
                   (value.dtype() == DataType::Bool || value.dtype() == DataType::UInt8);
        };
        LFS_ASSERT_MSG(byte_vector(output) && output.dtype() == DataType::UInt8 && !output.has_zero_stride(),
                       "update_labels requires a writable UInt8 [M] output");
        LFS_ASSERT_MSG(byte_vector(selected), "update_labels requires Bool/UInt8 [N] selection flags");
        const auto n = selected.numel(), m = output.numel();
        constexpr auto max_count = static_cast<size_t>(std::numeric_limits<int32_t>::max());
        LFS_ASSERT_MSG(n <= max_count && m <= max_count, "update_labels count exceeds int32");
        LFS_ASSERT_MSG(update.mode == LabelUpdateMode::Add || update.mode == LabelUpdateMode::Remove ||
                           update.mode == LabelUpdateMode::Replace,
                       "update_labels received an unknown mode");
        LFS_ASSERT_MSG(update.indices || n == m, "update_labels requires equal sizes without indices");
        LFS_ASSERT_MSG(bool(update.categories) == bool(update.allowed),
                       "update_labels requires both categories and allowed flags");
        const std::array<const Tensor*, 6> sources{&selected, update.existing, update.locked,
                                                   update.indices, update.categories, update.allowed};
        for (const auto* input : sources) {
            if (!input)
                continue;
            LFS_ASSERT_MSG(input->is_valid() && input->device() == output.device(),
                           "update_labels requires valid tensors on the same device");
            internal::require_same_gpu_backend(output, *input, "update_labels");
        }
        if (update.existing)
            LFS_ASSERT_MSG(byte_vector(*update.existing) && update.existing->dtype() == DataType::UInt8 &&
                               update.existing->numel() == m,
                           "update_labels existing labels must be UInt8 [M]");
        if (update.locked)
            LFS_ASSERT_MSG(byte_vector(*update.locked) && update.locked->numel() == 256,
                           "update_labels lock flags must be Bool/UInt8 [256]");
        for (const auto* ids : {update.indices, update.categories}) {
            if (ids)
                LFS_ASSERT_MSG(ids->dtype() == DataType::Int32 && ids->ndim() == 1 && ids->numel() == n,
                               "update_labels indices and categories must be Int32 [N]");
        }
        if (update.allowed)
            LFS_ASSERT_MSG(byte_vector(*update.allowed) && update.allowed->numel() <= max_count,
                           "update_labels allowed flags must be a Bool/UInt8 vector");

        internal::preserve_lazy_snapshots_before_write(output);
        std::array<Tensor, 6> inputs;
        for (size_t i = 0; i < sources.size(); ++i) {
            if (sources[i])
                inputs[i] =
                    internal::shares_storage(output, *sources[i]) ? sources[i]->clone() : sources[i]->contiguous();
        }
        auto result = output.contiguous();
        if (output.device() == Device::GPU) {
            pin_operands({&result, &inputs[0], update.existing ? &inputs[1] : nullptr,
                          update.locked ? &inputs[2] : nullptr, update.indices ? &inputs[3] : nullptr,
                          update.categories ? &inputs[4] : nullptr, update.allowed ? &inputs[5] : nullptr});
            const auto stream = prepare_inputs_for_stream({&result, &inputs[0]}, result.stream());
            result.set_stream(stream);
            if (!output.is_contiguous())
                output.set_stream(stream);
            internal::LabelUpdateProgram program{.count = n,
                                                 .output_count = m,
                                                 .allowed_count = update.allowed ? update.allowed->numel() : 0,
                                                 .label = update.label,
                                                 .mode = static_cast<uint32_t>(update.mode)};
            const std::array slots{&program.existing, &program.locked, &program.indices, &program.categories,
                                   &program.allowed};
            for (size_t i = 1; i < inputs.size(); ++i) {
                if (sources[i] && inputs[i].numel() != 0) {
                    (void)prepare_inputs_for_stream({&inputs[i]}, stream);
                    *slots[i - 1] = internal::storage_ref(inputs[i]);
                }
            }
            if (update.indices && n == 0) {
                if (update.existing)
                    result.copy_from(inputs[1]);
                else
                    result.zero_();
            }
            if (n != 0 && m != 0)
                internal::backend_ops_for(result).update_labels(internal::storage_ref(result),
                                                                internal::storage_ref(inputs[0]), program,
                                                                internal::ExecContext{stream});
        } else {
            auto* dest = result.ptr<uint8_t>();
            const auto* selected_bytes = static_cast<const uint8_t*>(inputs[0].data_ptr());
            const auto* existing = update.existing ? inputs[1].ptr<uint8_t>() : nullptr;
            const auto* locked = update.locked ? static_cast<const uint8_t*>(inputs[2].data_ptr()) : nullptr;
            const auto* indices = update.indices ? inputs[3].ptr<int32_t>() : nullptr;
            const auto* categories = update.categories ? inputs[4].ptr<int32_t>() : nullptr;
            const auto* allowed = update.allowed ? static_cast<const uint8_t*>(inputs[5].data_ptr()) : nullptr;
            if (m) {
                if (existing)
                    std::memcpy(dest, existing, m);
                else
                    std::memset(dest, 0, m);
            }
            const internal::LabelUpdateParams p{.output = dest,
                                                .selected = selected_bytes,
                                                .existing = existing,
                                                .locked = locked,
                                                .allowed = allowed,
                                                .indices = indices,
                                                .categories = categories,
                                                .count = uint32_t(n),
                                                .output_count = uint32_t(m),
                                                .allowed_count = update.allowed ? uint32_t(update.allowed->numel()) : 0,
                                                .label = update.label,
                                                .mode = uint32_t(update.mode)};
            if (indices && update.mode == LabelUpdateMode::Replace)
                for (uint32_t row = 0; row < n; ++row)
                    internal::updateLabel<1>(p, row);
            for (uint32_t row = 0; row < n; ++row) {
                if (indices)
                    internal::updateLabel<2>(p, row);
                else
                    internal::updateLabel<0>(p, row);
            }
        }
        if (!output.is_contiguous())
            output.copy_from(result);
    }
} // namespace lfs::core
