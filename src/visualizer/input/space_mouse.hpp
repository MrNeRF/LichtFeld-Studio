/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <chrono>
#include <core/export.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <string>

namespace lfs::vis::input {

    struct SpaceMouseProfile {
        std::uint16_t vendor;
        std::uint16_t product;
        const char* name;
        float full_scale;
    };

    [[nodiscard]] LFS_VIS_API const SpaceMouseProfile* spaceMouseProfile(std::uint16_t vendor, std::uint16_t product);
    [[nodiscard]] LFS_VIS_API bool isSpaceMouse(std::uint16_t vendor, std::uint16_t product,
                                                std::uint16_t usage_page, std::uint16_t usage);
    [[nodiscard]] LFS_VIS_API float spaceMouseHidScale(std::span<const std::uint8_t> descriptor, float fallback);

    // Normalized camera frame: right, up, toward the viewer. Axes are velocities.
    struct SpaceMouseMotion {
        glm::vec3 translation{0.0f};
        glm::vec3 rotation{0.0f};
        bool operator==(const SpaceMouseMotion&) const = default;
    };

    struct SpaceMouseReport {
        std::optional<glm::vec3> translation;
        std::optional<glm::vec3> rotation;
        std::uint64_t buttons = 0;
        std::uint64_t button_mask = 0;
        bool long_press = false;
    };

    [[nodiscard]] LFS_VIS_API std::optional<SpaceMouseReport> decodeSpaceMouseHid(
        std::span<const std::uint8_t> bytes, float full_scale);
    [[nodiscard]] LFS_VIS_API std::optional<SpaceMouseReport> decodeSpaceMouseSpnav(
        std::span<const std::uint8_t> bytes);
    [[nodiscard]] LFS_VIS_API float spaceMouseResponse(float value);
    [[nodiscard]] LFS_VIS_API SpaceMouseMotion spaceMouseStep(
        const SpaceMouseMotion& motion, float translation_speed, float rotation_speed,
        float delta_time, bool object_mode, bool keep_horizon);

    struct SpaceMouseConnection {
        std::string name;
        std::string backend;
        bool operator==(const SpaceMouseConnection&) const = default;
    };

    struct SpaceMouseFrame {
        SpaceMouseMotion motion;
        std::uint64_t buttons = 0;
        std::uint64_t pressed = 0;
        std::optional<SpaceMouseConnection> connection;
    };

    enum class SpaceMouseRead { Report,
                                Timeout,
                                Disconnected };

    class LFS_VIS_API SpaceMouseSource {
    public:
        virtual ~SpaceMouseSource() = default;
        virtual std::optional<SpaceMouseConnection> open() = 0;
        virtual SpaceMouseRead read(SpaceMouseReport& report) = 0;
        virtual void close() = 0;
    };

    // Single-threaded IO state machine; the device publishes snapshots to the viewer.
    class LFS_VIS_API SpaceMouseSession {
    public:
        using Clock = std::chrono::steady_clock;
        explicit SpaceMouseSession(SpaceMouseSource& source);
        ~SpaceMouseSession();
        SpaceMouseSession(const SpaceMouseSession&) = delete;
        SpaceMouseSession& operator=(const SpaceMouseSession&) = delete;
        bool poll(Clock::time_point now);
        SpaceMouseFrame takeFrame();

    private:
        SpaceMouseSource& source_;
        SpaceMouseFrame frame_;
        std::array<std::uint64_t, 2> button_groups_{};
        Clock::time_point next_scan_{};
        Clock::time_point translation_time_{};
        Clock::time_point rotation_time_{};
    };

} // namespace lfs::vis::input
