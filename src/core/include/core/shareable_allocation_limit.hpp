/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::core {

    inline constexpr const char* kShareableAllocLimitEnvName = "LFS_SHAREABLE_ALLOC_LIMIT_BYTES";
    inline constexpr const char* kShareableChunkBytesEnvName = "LFS_SHAREABLE_CHUNK_BYTES";

    class LFS_CORE_API ShareableAllocationLimitError : public TensorError {
    public:
        explicit ShareableAllocationLimitError(const std::string& msg, const Tensor* t = nullptr);
    };

    // Unrounded per-allocation ceiling: min(env LFS_SHAREABLE_CHUNK_BYTES or
    // LFS_SHAREABLE_ALLOC_LIMIT_BYTES, 4 GiB - 64 MiB, device limit if published).
    [[nodiscard]] LFS_CORE_API std::size_t max_shareable_allocation_bytes();

    // Same ceiling rounded down to the CUDA VMM granularity, never below one granule.
    [[nodiscard]] LFS_CORE_API std::size_t shareable_chunk_bytes(int device = 0);

    [[nodiscard]] LFS_CORE_API bool shareable_allocation_limited();

    [[nodiscard]] LFS_CORE_API bool shareable_allocation_limit_from_env();

    [[nodiscard]] LFS_CORE_API std::optional<std::string>
    shareable_allocation_violation(std::size_t bytes, std::string_view what);

    [[nodiscard]] inline bool is_shareable_allocation_limit_message(const std::string_view text) noexcept {
        return text.find("shareable GPU allocation") != std::string_view::npos;
    }

    LFS_CORE_API void set_shareable_device_allocation_limit(std::size_t bytes);

    LFS_CORE_API void reset_shareable_allocation_limit_for_tests();

} // namespace lfs::core
