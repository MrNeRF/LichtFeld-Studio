/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_kernels.hpp"

namespace lfs::training::metal {

    extern const char* const kTrainingKernelSource;

    core::GpuKernelModule& kernels() {
        // The CUDA kernels build with -use_fast_math; so do these.
        static core::GpuKernelModule module(core::GpuBackend::Metal, kTrainingKernelSource, true);
        return module;
    }

} // namespace lfs::training::metal
