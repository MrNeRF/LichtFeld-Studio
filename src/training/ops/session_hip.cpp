/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/session_hip.hpp"
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include <core/gpu_device_runtime.hpp>
#include <core/tensor_backend.hpp>
#include <cstdio>
#include <stdexcept>
namespace lfs::training {
namespace {
thread_local uint32_t timeout_ms=0;
void require_target(core::TensorExecutionTarget target) {
    if(target.backend()!=core::GpuBackend::HIP || !target.is_default_queue())
        throw std::invalid_argument("HIP training session requires its native default queue");
}
std::optional<std::string> last_error() {
    // StreamQuery surfaces queued device errors without draining a healthy
    // training queue or initializing another backend.
    if(!core::gpu_backend_live(core::GpuBackend::HIP)) return std::nullopt;
    try {
        auto& ops=static_cast<core::internal::HipBackendOps&>(core::internal::backend_ops(core::GpuBackend::HIP));
        (void)ops.queue_ready({});
        return std::nullopt;
    } catch(const std::exception& error) { /* LFS-CENSUS-OK(empty-catch): Return failure text through the session-validation callback. */  return std::string(error.what()); }
}
}
const gpu_ops::SessionOps& hip_session_ops() {
    // Raster workspaces are per-FastSaved and separately bounded. There is no
    // global arena to lend: optional scratch callers keep their Tensor path,
    // as on Metal. Never return a pointer into a live raster frame.
    static const gpu_ops::SessionOps table{
        .profile=[](bool) {},
        .resize_arena=[](std::string_view,bool release_all) { if(release_all) core::gpu_trim_cached_memory(core::GpuBackend::HIP); },
        .reset_arena=[] {},
        .arena_memory_info=[]()->std::optional<gpu_ops::ArenaMemoryInfo> { return std::nullopt; },
        .log_arena_failure=[](const char* context) { std::fprintf(stderr,"[error] HIP training workspace allocation failed: %s\n",context ? context : "unknown"); },
        .set_arena_timeout=[](uint32_t value) { const auto previous=timeout_ms; timeout_ms=value; return previous; },
        .borrow_idle_arena=[](size_t,size_t,core::TensorExecutionTarget target) { require_target(target); return gpu_ops::IdleArenaBorrow{}; },
        .release_idle_arena=[](const gpu_ops::IdleArenaBorrow& borrow,core::TensorExecutionTarget target) {
            require_target(target);
            if(borrow.owner || borrow.data || borrow.bytes || borrow.frame) throw std::invalid_argument("HIP session received an unknown arena borrow");
        },
        .zero_idle_arena=[](const gpu_ops::IdleArenaBorrow& borrow,size_t,core::TensorExecutionTarget target)->void* {
            require_target(target);
            if(borrow.owner || borrow.data || borrow.bytes || borrow.frame) throw std::invalid_argument("HIP session received an unknown arena borrow");
            return nullptr;
        },
        .allocation_bytes=[](size_t bytes) { return core::gpu_allocation_bytes(core::GpuBackend::HIP,bytes); },
        .direct_storage_live_bytes=[] { return size_t{0}; },
        .last_error=last_error,
        .device_baseline_bytes=[] { return core::gpu_backend_memory_info(core::GpuBackend::HIP,true).allocated_bytes; },
        .sample_memory=[] { if(core::gpu_backend_live(core::GpuBackend::HIP)) (void)core::gpu_backend_memory_info(core::GpuBackend::HIP,true); },
        .release_workspaces_before_evaluation=true,
        .release_workspaces_before_reorder=true,
    };
    return table;
}
}
