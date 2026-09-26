/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"

namespace lfs::core {
    uint64_t record_cuda_breadcrumb(const char*, const char*, uint32_t, cudaStream_t) noexcept {
        return 0;
    }

    uint64_t record_cuda_breadcrumb(const char*, const char*, uint32_t, cudaStream_t,
                                    uint64_t, uint64_t, uint64_t) noexcept {
        return 0;
    }

    void register_cuda_address_range(const void*, std::size_t, std::string) {}
    void unregister_cuda_address_range(const void*) {}
    bool cuda_is_unavailable() noexcept { return true; }
} // namespace lfs::core
