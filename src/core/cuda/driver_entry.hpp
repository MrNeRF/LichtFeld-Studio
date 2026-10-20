/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/cuda_types.hpp"

#if defined(__linux__) && LFS_HAS_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace lfs::core::cuda_driver {
    template <class Signature>
    struct Entry;

    template <class... Args>
    struct Entry<CUresult (*)(Args...)> {
        using Function = CUresult (*)(Args...);
        Function function = nullptr;

        CUresult operator()(Args... args) const {
            return function ? function(args...) : CUDA_ERROR_NOT_INITIALIZED;
        }
    };

    template <class Signature>
    Entry<Signature> resolve(const char* name) {
        void* entry = nullptr;
        if (cudaGetDriverEntryPointByVersion(name, &entry, CUDA_VERSION, cudaEnableDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            return {};
        }
        return {reinterpret_cast<Signature>(entry)};
    }
} // namespace lfs::core::cuda_driver

// Each call site caches its own correctly typed entry, including functions
// that share a signature. Keep the unexpanded API name for versioned lookup;
// decltype follows cuda.h's ABI aliases without referencing a driver symbol.
#define LFS_CUDA_DRIVER(name)                                            \
    ([] {                                                                \
        static const auto entry =                                        \
            ::lfs::core::cuda_driver::resolve<decltype(&::name)>(#name); \
        return entry;                                                    \
    }())
#else
// Preserve the SDK's existing direct calls and delay loading on Windows.
#define LFS_CUDA_DRIVER(name) name
#endif
