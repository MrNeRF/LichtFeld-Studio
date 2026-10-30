/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "backend_driver.hpp"
#if LFS_HAS_CUDA
#include "cuda/deferred_target.hpp"
#endif

namespace lfs::core::internal {
    // Construct only after the implicit-home early-outs. CUDA's binding is the
    // original native stream guard; other backends retain their driver policy.
    class DeferredTargetBinding {
        BackendDriver* driver_ = nullptr;
        std::optional<std::array<uintptr_t, 2>> token_;

    public:
        DeferredTargetBinding(GpuBackend backend, const Tensor& tensor) {
#if LFS_HAS_CUDA
            if (backend == GpuBackend::CUDA) {
                token_ = bind_cuda_deferred_home(tensor.stream());
                return;
            }
#endif
            driver_ = &backend_driver(backend);
            token_ = driver_->bind_tensor_target(tensor, true);
        }
        ~DeferredTargetBinding() {
            if (!token_)
                return;
            if (driver_) {
                driver_->unbind_target(*token_);
            }
#if LFS_HAS_CUDA
            else {
                unbind_cuda_deferred_home(*token_);
            }
#endif
        }
        DeferredTargetBinding(const DeferredTargetBinding&) = delete;
        DeferredTargetBinding& operator=(const DeferredTargetBinding&) = delete;
    };

    // Keep the compatibility fast paths in the backend layer. The generic
    // materializer must not look up a driver for a no-op CUDA target policy.
    inline bool deferred_home_needs_binding(GpuBackend backend, void* home) {
        if (!home)
            return false;
#if LFS_HAS_CUDA
        if (backend == GpuBackend::CUDA)
            return cuda_deferred_home_needs_binding(home);
#endif
        return true;
    }

    inline void* canonical_deferred_home(GpuBackend backend, void* home) {
        if (!home)
            return nullptr;
#if LFS_HAS_CUDA
        // CUDA retains its home unchanged, including retired handles. The
        // producer/retirement checks decide whether to preserve that home.
        if (backend == GpuBackend::CUDA)
            return home;
#endif
        return backend_driver(backend).canonical_home_target(home);
    }
} // namespace lfs::core::internal
