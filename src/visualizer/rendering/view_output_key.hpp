/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "workspace/view_id.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace lfs::vis {

    // The tag keeps a scene's stable ViewId distinct from renderer-owned
    // compatibility keys. The workspace allocator can therefore issue ViewId
    // 1 without knowing that legacy output columns exist.
    struct ViewOutputKey {
        enum class Kind : std::uint8_t { Invalid,
                                         Legacy,
                                         Scene,
                                         Preview };

        Kind kind = Kind::Invalid;
        ViewId view_id = 0;

        [[nodiscard]] constexpr bool valid() const noexcept {
            switch (kind) {
            case Kind::Legacy:
            case Kind::Scene:
            case Kind::Preview:
                return view_id != 0;
            case Kind::Invalid:
            default:
                return false;
            }
        }
        [[nodiscard]] constexpr bool scene() const noexcept {
            return kind == Kind::Scene && view_id != 0;
        }
        friend constexpr bool operator==(const ViewOutputKey&, const ViewOutputKey&) = default;
    };

    struct ViewOutputKeyHash {
        [[nodiscard]] std::size_t operator()(const ViewOutputKey key) const noexcept {
            const auto kind = static_cast<std::uint64_t>(key.kind);
            const auto value = static_cast<std::uint64_t>(key.view_id);
            return std::hash<std::uint64_t>{}(value ^ (kind * 0x9e3779b97f4a7c15ull));
        }
    };

    inline constexpr ViewOutputKey kInvalidViewOutputKey{};

    // Renderer-owned compatibility keys. Their numeric payload is local to
    // this tagged namespace and can never alias a scene ViewId.
    inline constexpr ViewOutputKey kLegacyMainOutputKey{ViewOutputKey::Kind::Legacy, 1};
    inline constexpr ViewOutputKey kLegacySplitLeftOutputKey{ViewOutputKey::Kind::Legacy, 2};
    inline constexpr ViewOutputKey kLegacySplitRightOutputKey{ViewOutputKey::Kind::Legacy, 3};
    inline constexpr ViewOutputKey kLegacyPreviewOutputKey{ViewOutputKey::Kind::Preview, 1};

    // The mapping preserves the workspace ViewId as the key payload.
    [[nodiscard]] constexpr ViewOutputKey sceneOutputKey(const ViewId id) noexcept {
        return {ViewOutputKey::Kind::Scene, id};
    }

    [[nodiscard]] constexpr bool isReservedLegacyOutputKey(const ViewOutputKey key) noexcept {
        return key.kind == ViewOutputKey::Kind::Legacy || key.kind == ViewOutputKey::Kind::Preview;
    }

} // namespace lfs::vis
