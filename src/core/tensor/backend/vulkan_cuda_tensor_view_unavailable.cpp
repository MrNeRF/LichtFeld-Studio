/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor_backend_vulkan.hpp"
namespace lfs::core {
    bool vulkan_backend_exports_memory() {

        return false;
    }
    lfs::Result<Tensor> cuda_view_of_vulkan_tensor(const Tensor& tensor,
                                                   const cudaStream_t stream) {

        (void)tensor;
        (void)stream;
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "this build has no Vulkan tensor backend",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }
} // namespace lfs::core
