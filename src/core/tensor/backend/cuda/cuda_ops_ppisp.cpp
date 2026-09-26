/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "kernels/tensor_ppisp.hpp"
namespace lfs::core::internal {
    void CudaBackendOps::ppisp_apply(StorageRef input, StorageRef output, int width, int height, const PpispParams& p, ExecContext context) {
        LFS_FACADE_TRACE(ppisp_apply);
        tensor_ops::launch_ppisp_apply(reinterpret_cast<const float*>(static_cast<unsigned char*>(input.data) + input.byte_offset),
                                       reinterpret_cast<float*>(static_cast<unsigned char*>(output.data) + output.byte_offset), width, height, p, context.cuda_stream);
    }
} // namespace lfs::core::internal
