/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include "core/tensor_backend.hpp"
#include "core/tensor_histogram.hpp"

#include <stdexcept>

namespace lfs::rendering {
    namespace {
        constexpr std::size_t kSelectionGroupCountBins = 256;
        constexpr std::size_t kSelectionGroupScratchWords = kSelectionGroupCountBins;

        void prepareSelectionGroupCountsScratch(Tensor& counts_scratch) {
            if (!counts_scratch.is_valid() ||
                counts_scratch.device() != lfs::core::Device::GPU ||
                lfs::core::gpu_backend_of(counts_scratch) != lfs::core::default_gpu_backend() ||
                counts_scratch.dtype() != lfs::core::DataType::Int32 ||
                counts_scratch.ndim() != 1 || !counts_scratch.is_contiguous() ||
                counts_scratch.numel() != kSelectionGroupScratchWords) {
                counts_scratch = Tensor::empty(
                    {kSelectionGroupScratchWords}, lfs::core::Device::GPU, lfs::core::DataType::Int32);
            }
        }
    } // namespace

    void count_selection_groups_async(const Tensor& selection_mask, Tensor& counts_scratch) {
        const auto backend = lfs::core::gpu_backend_of(selection_mask);
        if (selection_mask.is_valid() && selection_mask.numel() && !backend)
            throw std::runtime_error("count_selection_groups_async requires a GPU mask");
        const lfs::core::GpuBackendScope scope(backend.value_or(
            lfs::core::gpu_backend_of(counts_scratch).value_or(lfs::core::default_gpu_backend())));
        prepareSelectionGroupCountsScratch(counts_scratch);
        if (!selection_mask.is_valid() || selection_mask.numel() == 0) {
            counts_scratch.zero_();
            return;
        }
        lfs::core::histogram_u8(selection_mask, counts_scratch);
    }

} // namespace lfs::rendering
