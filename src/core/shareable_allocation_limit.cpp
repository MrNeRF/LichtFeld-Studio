/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/shareable_allocation_limit.hpp"

#include "core/environment.hpp"
#include "core/exportable_storage.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <mutex>
#include <optional>

namespace lfs::core {
    namespace {

        constexpr std::size_t kPlatformCeiling = (std::size_t{1} << 32) - (std::size_t{64} << 20);

        struct LimitState {
            std::size_t bytes = std::numeric_limits<std::size_t>::max();
            bool from_env = false;
        };

        std::mutex g_limit_mu;
        std::optional<LimitState> g_limit;
        std::optional<std::size_t> g_device_limit;

        LimitState compute_limit() {
            std::size_t env_bytes = std::numeric_limits<std::size_t>::max();
            bool from_env = false;
            if (const auto chunk_env = environment::unsigned_integer<std::size_t>(kShareableChunkBytesEnvName)) {
                env_bytes = *chunk_env;
                from_env = true;
            } else if (const auto alloc_env =
                           environment::unsigned_integer<std::size_t>(kShareableAllocLimitEnvName)) {
                env_bytes = *alloc_env;
                from_env = true;
            }
            const std::size_t device = g_device_limit.value_or(std::numeric_limits<std::size_t>::max());
            const std::size_t bytes = std::min({env_bytes, kPlatformCeiling, device});
            return LimitState{.bytes = bytes, .from_env = from_env};
        }

        const LimitState& cached_limit() {
            std::lock_guard lock(g_limit_mu);
            if (!g_limit) {
                g_limit = compute_limit();
            }
            return *g_limit;
        }

        std::size_t align_down(const std::size_t value, const std::size_t alignment) {
            if (alignment == 0) {
                return value;
            }
            return (value / alignment) * alignment;
        }

    } // namespace

    ShareableAllocationLimitError::ShareableAllocationLimitError(const std::string& msg, const Tensor* t)
        : TensorError(msg, t) {}

    std::size_t max_shareable_allocation_bytes() {
        return cached_limit().bytes;
    }

    std::size_t shareable_chunk_bytes(const int device) {
        const std::size_t gran = exportable_allocation_granularity(device);
        const std::size_t limit = max_shareable_allocation_bytes();
        const std::size_t rounded = align_down(limit, gran);
        return rounded < gran ? gran : rounded;
    }

    bool shareable_allocation_limited() {
        return cached_limit().bytes != std::numeric_limits<std::size_t>::max();
    }

    bool shareable_allocation_limit_from_env() {
        return cached_limit().from_env;
    }

    std::optional<std::string> shareable_allocation_violation(const std::size_t bytes,
                                                              const std::string_view what) {
        const std::size_t limit = max_shareable_allocation_bytes();
        if (bytes <= limit) {
            return std::nullopt;
        }
        return std::format(
            "'{}' needs {} bytes in one shareable GPU allocation; this platform caps "
            "exportable/imported allocations at {} bytes (Windows drivers truncate shared "
            "allocation sizes to 32 bits). Use a lower max-cap or SH degree, or split the model.",
            what,
            bytes,
            limit);
    }

    void set_shareable_device_allocation_limit(const std::size_t bytes) {
        std::lock_guard lock(g_limit_mu);
        g_device_limit = bytes;
        g_limit.reset();
    }

    void reset_shareable_allocation_limit_for_tests() {
        std::lock_guard lock(g_limit_mu);
        g_limit.reset();
        g_device_limit.reset();
    }

} // namespace lfs::core
