/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "kernels/tensor_environment.hpp"
namespace lfs::core::internal {
    void CudaBackendOps::environment_composite(StorageRef rgb, StorageRef alpha, StorageRef environment,
                                               StorageRef output, const EnvironmentCompositeParams& p, ExecContext context) {
        LFS_FACADE_TRACE(environment_composite);
        const auto data = [](StorageRef s) { return static_cast<unsigned char*>(s.data) + s.byte_offset; };
        tensor_ops::launch_environment_composite(p, reinterpret_cast<const float*>(data(environment)),
                                                 reinterpret_cast<const float*>(data(rgb)), reinterpret_cast<const float*>(data(alpha)), data(output), context.cuda_stream);
    }
} // namespace lfs::core::internal
