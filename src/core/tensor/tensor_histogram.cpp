/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_histogram.hpp"
#include "internal/tensor_impl.hpp"

#include <limits>

namespace lfs::core {
    void histogram_u8(const Tensor& values, Tensor& counts) {
        LFS_ASSERT_MSG(values.is_valid() && counts.is_valid(), "histogram_u8 requires valid tensors");
        LFS_ASSERT_MSG(values.dtype() == DataType::UInt8 || values.dtype() == DataType::Bool,
                       "histogram_u8 requires UInt8 or Bool input");
        LFS_ASSERT_MSG(counts.dtype() == DataType::Int32 && counts.ndim() == 1 &&
                           counts.numel() >= 256 && counts.is_contiguous(),
                       "histogram_u8 requires contiguous Int32 counts with at least 256 bins");
        LFS_ASSERT_MSG(values.device() == counts.device(), "histogram_u8 requires the same device");
        internal::require_same_gpu_backend(values, counts, "histogram_u8");
        LFS_ASSERT_MSG(!internal::shares_storage(values, counts), "histogram_u8 input and counts must not alias");
        LFS_ASSERT_MSG(values.numel() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "histogram_u8 input size exceeds int32");
        const auto input = values.contiguous();
        if (values.device() == Device::GPU) {
            pin_operands({&input, &counts});
            const auto stream = prepare_inputs_for_stream({&input, &counts}, counts.stream());
            counts.fill_(0.f, stream);
            if (input.numel() != 0) {
                internal::backend_ops_for(counts).histogram_u8(
                    internal::storage_ref(input), internal::storage_ref(counts), input.numel(),
                    internal::ExecContext{stream});
            }
            return;
        }
        counts.zero_();
        const auto* data = static_cast<const uint8_t*>(input.data_ptr());
        auto* bins = counts.ptr<int32_t>();
        for (size_t i = 0; i < input.numel(); ++i) {
            if (data[i] != 0)
                ++bins[data[i]];
        }
    }
} // namespace lfs::core
