/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/gpu_slab_allocator.hpp"

namespace lfs::core {

    GPUSlabAllocator& GPUSlabAllocator::instance() {
        static GPUSlabAllocator allocator;
        return allocator;
    }

} // namespace lfs::core
