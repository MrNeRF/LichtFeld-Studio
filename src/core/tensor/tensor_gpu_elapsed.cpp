/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "backend/gpu_backend_ops.hpp"
#include "core/gpu_elapsed.hpp"
#include <stdexcept>

namespace lfs::core {
    struct GpuElapsed::Impl {
        GpuBackend backend;
        std::unique_ptr<internal::BackendTimestamps> timestamps;
    };
    GpuElapsed::GpuElapsed(GpuBackend backend, size_t count)
        : impl_(std::make_unique<Impl>(Impl{backend, {}})) {
        if (count && gpu_backend_compiled(backend))
            impl_->timestamps = internal::backend_ops(backend).create_timestamps(count);
    }
    GpuElapsed::~GpuElapsed() = default;
    bool GpuElapsed::ready() const noexcept { return impl_ && impl_->timestamps != nullptr; }
    bool GpuElapsed::mark(size_t index, TensorExecutionTarget target) {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("GpuElapsed queue backend mismatch");
        return mark(index, target.native_handle());
    }
    bool GpuElapsed::wait_queue(TensorExecutionTarget target) {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("GpuElapsed queue backend mismatch");
        return wait_queue(target.native_handle());
    }
    bool GpuElapsed::mark(size_t index, void* target) { return ready() && impl_->timestamps->mark(index, target); }
    bool GpuElapsed::wait_queue(void* target) { return ready() && impl_->timestamps->wait_queue(target); }
    bool GpuElapsed::wait_event(size_t index) { return ready() && impl_->timestamps->wait_event(index); }
    std::optional<float> GpuElapsed::milliseconds(size_t begin, size_t end) const {
        return ready() ? impl_->timestamps->milliseconds(begin, end) : std::nullopt;
    }
} // namespace lfs::core
