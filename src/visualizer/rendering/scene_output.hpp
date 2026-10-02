/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <type_traits>

namespace lfs::vis {
    // Opaque compositor resources. They do not select the scene rasterizer,
    // own a native resource, or extend its lifetime. The producing renderer
    // retains ownership until its normal target/consumer retirement boundary.
    template <class Tag>
    class SceneOutputHandle {
    public:
        constexpr SceneOutputHandle() = default;
        template <class Native>
        static SceneOutputHandle fromNative(Native value) {
            static_assert(std::is_pointer_v<Native> || std::is_integral_v<Native> || std::is_enum_v<Native>);
            static_assert(sizeof(Native) <= sizeof(uint64_t));
            SceneOutputHandle result;
            if constexpr (std::is_pointer_v<Native>)
                result.value_ = reinterpret_cast<uintptr_t>(value);
            else
                result.value_ = static_cast<uint64_t>(value);
            return result;
        }
        template <class Native>
        Native native() const {
            static_assert(std::is_pointer_v<Native> || std::is_integral_v<Native> || std::is_enum_v<Native>);
            static_assert(sizeof(Native) <= sizeof(uint64_t));
            if constexpr (std::is_pointer_v<Native>)
                return reinterpret_cast<Native>(static_cast<uintptr_t>(value_));
            else
                return static_cast<Native>(value_);
        }
        constexpr explicit operator bool() const { return value_ != 0; }
        friend constexpr bool operator==(SceneOutputHandle, SceneOutputHandle) = default;

    private:
        uint64_t value_ = 0;
    };
    struct SceneImageTag;
    struct SceneImageViewTag;
    struct SceneImageLayoutTag;
    struct SceneTimelineTag;
    using SceneImageHandle = SceneOutputHandle<SceneImageTag>;
    using SceneImageViewHandle = SceneOutputHandle<SceneImageViewTag>;
    using SceneImageLayout = SceneOutputHandle<SceneImageLayoutTag>;
    using SceneTimelineHandle = SceneOutputHandle<SceneTimelineTag>;
    static_assert(sizeof(SceneImageHandle) == sizeof(uint64_t));
    static_assert(std::is_trivially_copyable_v<SceneImageHandle>);
    static_assert(!std::is_convertible_v<SceneImageHandle, SceneImageViewHandle>);
    static_assert(!std::is_convertible_v<SceneImageHandle, SceneTimelineHandle>);
} // namespace lfs::vis
