/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../internal/tensor_impl.hpp"
namespace lfs::core::internal {
    // The public indexing contract accepts Int32/Int64. Native kernels that
    // require Int32 retain their conversion scratch until submission completes.
    class NativeIndexScratch {
        Tensor owner_;
        StorageRef storage_;

    public:
        template <class Normalize>
        NativeIndexScratch(StorageRef indices, size_t count, ExecContext context, Normalize&& normalize)
            : storage_(indices) {
            if (indices.dtype != DataType::Int64)
                return;
            const GpuBackendScope backend(indices.backend);
            const CUDAStreamGuard stream(context.cuda_stream);
            owner_ = Tensor::empty({count}, Device::GPU, DataType::Int32);
            storage_ = storage_ref(owner_);
            normalize(indices, storage_);
        }
        StorageRef storage() const { return storage_; }
    };
} // namespace lfs::core::internal
