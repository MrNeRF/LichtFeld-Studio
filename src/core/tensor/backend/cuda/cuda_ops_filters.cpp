/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "kernels/tensor_filters.hpp"
namespace lfs::core::internal {
    void CudaBackendOps::filter_points(StorageRef mask, const PointFilterProgram& p, ExecContext context) {
        LFS_FACADE_TRACE(filter_points);
        auto a = pointFilterArgs(p.window);
        const auto data = [&]<class T>(size_t i) -> const T* {
            const auto& s = p.inputs[i];
            return s ? reinterpret_cast<const T*>(static_cast<const uint8_t*>(s->data) + s->byte_offset) : nullptr;
        };
        loadPointFilterInputs(a, static_cast<uint8_t*>(mask.data) + mask.byte_offset, data, p.count, p.transform_count,
                              p.allowed_count, p.flags);
        tensor_ops::launch_filter_points(a, context.cuda_stream);
    }
} // namespace lfs::core::internal
