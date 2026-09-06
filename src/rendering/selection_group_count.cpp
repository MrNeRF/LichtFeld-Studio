/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include "core/tensor_backend.hpp"

#include <stdexcept>

namespace lfs::rendering {
    namespace {
        constexpr std::size_t kSelectionGroupCountBins = 256;
        constexpr std::size_t kSelectionGroupScratchWords = kSelectionGroupCountBins + 1;

        void prepareSelectionGroupCountsScratch(Tensor& counts_scratch) {
            if (!counts_scratch.is_valid() ||
                counts_scratch.device() != lfs::core::Device::CUDA ||
                counts_scratch.dtype() != lfs::core::DataType::Int32 ||
                counts_scratch.numel() != kSelectionGroupScratchWords) {
                counts_scratch = Tensor::zeros(
                    {kSelectionGroupScratchWords}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
            } else {
                counts_scratch.zero_();
            }
        }
    } // namespace

    void prepare_cuda_selection_group_counts_scratch(Tensor& counts_scratch) {
        lfs::core::GpuBackendScope cuda_scope(lfs::core::GpuBackend::CUDA);
        prepareSelectionGroupCountsScratch(counts_scratch);
    }

    void count_selection_groups_tensor_program(const Tensor& selection_mask,
                                               Tensor& counts_scratch) {
        const auto backend = lfs::core::gpu_backend_of(selection_mask);
        if (!backend) {
            throw std::runtime_error("count_selection_groups_async requires a GPU mask");
        }
        lfs::core::GpuBackendScope scope(*backend);
        prepareSelectionGroupCountsScratch(counts_scratch);
        const Tensor indices = selection_mask.to(lfs::core::DataType::Int32);
        const Tensor ones = Tensor::ones(
            {static_cast<size_t>(selection_mask.numel())},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Int32);
        counts_scratch.index_add_(0, indices, ones);
        counts_scratch.slice(0, 0, 1).fill_(0.0f);
    }

} // namespace lfs::rendering
