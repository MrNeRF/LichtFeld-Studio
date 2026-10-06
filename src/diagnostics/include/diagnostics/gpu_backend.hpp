/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/vram_profiler.hpp"

namespace lfs::diagnostics {

    enum class GpuEventStatus { Ready,
                                Pending,
                                Failed };

    // One immutable backend with process lifetime. The CPU diagnostics library
    // owns all profiler state; Studio supplies the native GPU operations.
    struct GpuDiagnosticsBackend {
        bool (*context_live)() noexcept;
        bool (*sample_used_bytes)(std::size_t&, std::size_t*) noexcept;
        void (*sample_process)(VramProcessSnapshot&);
        std::optional<std::size_t> (*process_memory_bytes)();
        void* (*create_event)() noexcept;
        bool (*record_event)(void*, void*) noexcept;
        GpuEventStatus (*elapsed_time)(void*, void*, float&) noexcept;
    };

    // A second, different backend is rejected so existing event handles can
    // never be passed to a replacement implementation.
    LFS_DIAGNOSTICS_API bool register_gpu_diagnostics_backend(const GpuDiagnosticsBackend&) noexcept;

} // namespace lfs::diagnostics
