/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include <compare>

namespace lfs::core {
    class TensorWorkQueue;

    // Non-owning, allocation-free queue identity. The queue must outlive every
    // use. Vulkan and Metal select their existing implicit storage timelines.
    class LFS_CORE_API TensorExecutionTarget {
    public:
        TensorExecutionTarget(const TensorWorkQueue& queue);
        static TensorExecutionTarget current();
        static TensorExecutionTarget default_queue(GpuBackend backend);
        [[nodiscard]] GpuBackend backend() const { return backend_; }
        [[nodiscard]] auto operator<=>(const TensorExecutionTarget&) const = default;
        [[nodiscard]] bool is_default_queue() const { return target_ == nullptr; }
        void wait() const;
        // Enqueues the existing queue bridge without waiting on the host.
        // Explicit queue bridges are currently supported only by CUDA.
        void wait_for(TensorExecutionTarget producer) const;
        void set_name(const char* name) const;

        class LFS_CORE_API Scope {
        public:
            explicit Scope(TensorExecutionTarget target);
            ~Scope();
            Scope(const Scope&) = delete;
            Scope& operator=(const Scope&) = delete;

        private:
            GpuBackendScope backend_scope_;
            void* previous_target_;
            bool rebound_vulkan_ = false;
        };

        // Native interoperability belongs to backend implementations.
        [[nodiscard]] void* native_handle() const { return target_; }

    private:
        friend class Tensor;
        TensorExecutionTarget(GpuBackend backend, void* target)
            : backend_(backend), target_(target) {}
        GpuBackend backend_;
        void* target_;
    };

    LFS_CORE_API void push_gpu_range(const char* name);
    LFS_CORE_API void pop_gpu_range();
} // namespace lfs::core
