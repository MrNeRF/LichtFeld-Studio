/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_elapsed.hpp"

#include "core/cuda_types.hpp"

#include <vector>

#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace lfs::core {

    struct GpuElapsed::Impl {
        GpuBackend backend;
        bool ready = false;
#if LFS_HAS_CUDA
        std::vector<cudaEvent_t> events;
#endif
    };

    GpuElapsed::GpuElapsed(const GpuBackend backend, const std::size_t event_count)
        : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
#if LFS_HAS_CUDA
        if (backend != GpuBackend::CUDA || event_count == 0)
            return;
        impl_->events.resize(event_count, nullptr);
        for (auto& event : impl_->events) {
            if (cudaEventCreate(&event) != cudaSuccess) {
                for (auto& created : impl_->events) {
                    if (created != nullptr)
                        (void)cudaEventDestroy(created);
                    created = nullptr;
                }
                impl_->events.clear();
                (void)cudaGetLastError();
                return;
            }
        }
        impl_->ready = true;
#else
        (void)event_count;
#endif
    }

    GpuElapsed::~GpuElapsed() {
#if LFS_HAS_CUDA
        if (impl_) {
            for (const auto event : impl_->events) {
                if (event != nullptr)
                    (void)cudaEventDestroy(event);
            }
        }
#endif
    }

    bool GpuElapsed::ready() const noexcept {
        return impl_ && impl_->ready;
    }

    bool GpuElapsed::mark(const std::size_t index, void* const execution_target) {
#if LFS_HAS_CUDA
        if (!ready() || index >= impl_->events.size() ||
            (impl_->backend != GpuBackend::CUDA && execution_target != nullptr))
            return false;
        const auto stream = reinterpret_cast<cudaStream_t>(execution_target);
        if (cudaEventRecord(impl_->events[index], stream) == cudaSuccess)
            return true;
        (void)cudaGetLastError();
#else
        (void)index;
        (void)execution_target;
#endif
        return false;
    }

    bool GpuElapsed::wait_queue(void* const execution_target) {
#if LFS_HAS_CUDA
        if (!ready() ||
            (impl_->backend != GpuBackend::CUDA && execution_target != nullptr))
            return false;
        if (cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(execution_target)) == cudaSuccess)
            return true;
        (void)cudaGetLastError();
#else
        (void)execution_target;
#endif
        return false;
    }

    bool GpuElapsed::wait_event(const std::size_t index) {
#if LFS_HAS_CUDA
        if (!ready() || index >= impl_->events.size())
            return false;
        if (cudaEventSynchronize(impl_->events[index]) == cudaSuccess)
            return true;
        (void)cudaGetLastError();
#else
        (void)index;
#endif
        return false;
    }

    std::optional<float> GpuElapsed::milliseconds(const std::size_t begin,
                                                  const std::size_t end) const {
#if LFS_HAS_CUDA
        if (!ready() || begin >= impl_->events.size() || end >= impl_->events.size())
            return std::nullopt;
        float elapsed = 0.0f;
        if (cudaEventElapsedTime(&elapsed, impl_->events[begin], impl_->events[end]) == cudaSuccess)
            return elapsed;
        (void)cudaGetLastError();
#else
        (void)begin;
        (void)end;
#endif
        return std::nullopt;
    }

} // namespace lfs::core
