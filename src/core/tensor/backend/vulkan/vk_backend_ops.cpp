/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_backend_ops.hpp"

#include "../facade_trace.hpp"

#include "core/assert.hpp"

#include <format>

namespace lfs::core::internal {
    namespace {
        [[noreturn]] void not_implemented(const char* const entry) {
            LFS_ASSERT_MSG(false,
                           std::format("Vulkan backend: {} is not implemented yet", entry));
            std::unreachable();
        }

    } // namespace

#define LFS_VK_NOTIMPL_VOID(Name, Parameters) \
    void VulkanBackendOps::Name Parameters {  \
        LFS_FACADE_TRACE(Name);               \
        not_implemented(#Name);               \
    }
#define LFS_VK_NOTIMPL_FLOAT(Name, Parameters) \
    float VulkanBackendOps::Name Parameters {  \
        LFS_FACADE_TRACE(Name);                \
        not_implemented(#Name);                \
    }
#define LFS_VK_NOTIMPL_SIZE(Name, Parameters)  \
    size_t VulkanBackendOps::Name Parameters { \
        LFS_FACADE_TRACE(Name);                \
        not_implemented(#Name);                \
    }
#define LFS_VK_NOTIMPL_BOOL(Name, Parameters) \
    bool VulkanBackendOps::Name Parameters {  \
        LFS_FACADE_TRACE(Name);               \
        not_implemented(#Name);               \
    }

    LFS_VK_NOTIMPL_VOID(sgemm, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_tn, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_batched, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_bias_relu, (StorageRef, StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(dot_product, (StorageRef, StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(diag, (StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(eye, (StorageRef, size_t, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(cdist, (StorageRef, StorageRef, StorageRef, size_t, size_t, size_t, float, ExecContext))
    LFS_VK_NOTIMPL_VOID(max_pool2d, (StorageRef, StorageRef, const PoolProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(adaptive_avg_pool2d, (StorageRef, StorageRef, const PoolProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(bias_add, (StorageRef, StorageRef, StorageRef, int, int, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(bias_relu, (StorageRef, StorageRef, StorageRef, int, int, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(relu, (StorageRef, StorageRef, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(uniform, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(bernoulli, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(randint, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(multinomial, (StorageRef, StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(normal, (StorageRef, StorageRef, const RandomProgram&, ExecContext))

#undef LFS_VK_NOTIMPL_VOID
#undef LFS_VK_NOTIMPL_FLOAT
#undef LFS_VK_NOTIMPL_SIZE
#undef LFS_VK_NOTIMPL_BOOL

} // namespace lfs::core::internal
