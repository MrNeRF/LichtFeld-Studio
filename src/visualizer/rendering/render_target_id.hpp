/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace lfs::vis {
    struct RenderTargetId {
        std::uint32_t value = 0;
        [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }
        auto operator<=>(const RenderTargetId&) const = default;
    };
    struct RenderTargetIdHash {
        std::size_t operator()(RenderTargetId id) const noexcept {
            return std::hash<std::uint32_t>{}(id.value);
        }
    };
    class RenderTargetRegistry {
    public:
        [[nodiscard]] RenderTargetId allocate() {
            if (next_ > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Render target IDs exhausted");
            const RenderTargetId id{static_cast<std::uint32_t>(next_++)};
            live_.insert(id);
            return id;
        }
        bool release(RenderTargetId id) { return live_.erase(id) != 0; }
        [[nodiscard]] bool contains(RenderTargetId id) const { return live_.contains(id); }

    private:
        std::uint64_t next_ = 1;
        std::unordered_set<RenderTargetId, RenderTargetIdHash> live_;
    };
} // namespace lfs::vis
