/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <functional>

namespace lfs::core {

    // Installs process-wide last-resort diagnostics. Call only after the ABI
    // tripwire: a stale core must never execute current-core startup hooks.
    LFS_CORE_API void install_crash_handlers();

    // Flushes the logger and any other flushable diagnostic sink. Swallows
    // all exceptions; safe to call before the logger has been initialized.
    LFS_CORE_API void flush_diagnostics_noexcept() noexcept;

    // ---------------------------------------------------------------------------
    // ordered GPU release hooks (TLS / static CUDA tensor holders)
    //
    // Why hooks (not only pool-liveness-aware deleters):
    //   Static / thread_local Tensors that outlive CudaMemoryPool's function-
    //   local static call into a destroyed Meyers singleton → SIGSEGV after a
    //   green suite. Freeing them *before* pool/arena shutdown (while CUDA is
    //   still healthy) is the primary fix; pool-liveness-aware deleters are the
    //   belt-and-suspenders so late dtors become no-ops instead of crashes.
    //
    // Pattern matches training-thread TLS release (FastGS sort / rasterizer /
    // nan-check buffers): explicit release, not relying on destruction order.
    // Register from a TU static initializer; hooks must be noexcept and
    // idempotent.
    // ---------------------------------------------------------------------------
    using GpuPreShutdownHook = void (*)() noexcept;

    // Registers a hook to run once at the start of teardown_gpu_before_exit,
    // before device_fault / arena / pool / pinned shutdown. Capacity is fixed;
    // excess registrations are dropped (logged once).
    LFS_CORE_API void register_gpu_pre_shutdown_hook(GpuPreShutdownHook hook) noexcept;

    // True after pre-shutdown hooks have run and subsystem teardown has begun.
    // Deleters / cudaFree paths that may run during static/TLS destruction
    // should no-op when this is set (CUDA context may already be unusable).
    [[nodiscard]] LFS_CORE_API bool gpu_process_teardown_started() noexcept;

    // Explicit, idempotent GPU teardown while CUDA and diagnostics are still
    // alive. Preserve this order:
    // - registered hooks release TLS caches and static Tensor holders;
    // - device_fault_registry_teardown() releases dedicated cudaMalloc slots;
    // - GlobalArenaManager::instance().shutdown() releases arenas;
    // - Tensor::shutdown_memory_pool() releases the tensor pool;
    // - PinnedMemoryAllocator::instance().shutdown() releases pinned storage.
    // CPU-only processes pay only idempotent-guard checks — see
    // Tensor::shutdown_memory_pool's g_cuda_memory_pool_instance guard and
    // PinnedMemoryAllocator's constructor, neither of which touches CUDA unless
    // a prior allocation path already did. Call this before flush_and_exit, never
    // after: a teardown failure must have a chance to reach the flushed log.
    // Safe to call multiple times (all steps are idempotent no-ops on repeat).
    LFS_CORE_API void teardown_gpu_before_exit() noexcept;

    // Flushes diagnostics, then terminates the process without running
    // destructors. The single sanctioned replacement for std::_Exit/_exit.
    [[noreturn]] LFS_CORE_API void flush_and_exit(int code) noexcept;

    // Invokes fn, converting any exception that escapes it into a failure
    // report instead of letting it reach std::terminate. Returns fn()'s
    // result on success, or the frozen firewall exit code (70 / EX_SOFTWARE)
    // if fn threw. Use at a single outermost dispatch site.
    LFS_CORE_API int run_with_exception_firewall(const std::function<int()>& fn) noexcept;

} // namespace lfs::core
