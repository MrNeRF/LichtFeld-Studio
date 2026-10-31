/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "space_mouse.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace lfs::vis::input {
    namespace {
        // IDs: spacenavd/src/dev.c. Nominal USB range: Blender GHOST_NDOFManager.
        // Vintage models with a different report layout use the daemon backend.
        constexpr SpaceMouseProfile kProfiles[] = {
            {0x046d, 0xc605, "CADman", 350.0f},
            {0x046d, 0xc621, "Spaceball 5000", 350.0f},
            {0x046d, 0xc623, "SpaceTraveler", 350.0f},
            {0x046d, 0xc625, "SpacePilot", 350.0f},
            {0x046d, 0xc626, "SpaceNavigator", 350.0f},
            {0x046d, 0xc627, "SpaceExplorer", 350.0f},
            {0x046d, 0xc628, "SpaceNavigator for Notebooks", 350.0f},
            {0x046d, 0xc629, "SpacePilot Pro", 350.0f},
            {0x046d, 0xc62b, "SpaceMouse Pro", 350.0f},
            {0x256f, 0xc62e, "SpaceMouse Wireless", 350.0f},
            {0x256f, 0xc62f, "SpaceMouse Wireless", 350.0f},
            {0x256f, 0xc631, "SpaceMouse Pro Wireless", 350.0f},
            {0x256f, 0xc632, "SpaceMouse Pro Wireless", 350.0f},
            {0x256f, 0xc633, "SpaceMouse Enterprise", 350.0f},
            {0x256f, 0xc635, "SpaceMouse Compact", 350.0f},
            {0x256f, 0xc636, "SpaceMouse Module", 350.0f},
            {0x256f, 0xc638, "SpaceMouse Pro Wireless", 350.0f},
            {0x256f, 0xc63a, "SpaceMouse Wireless", 350.0f},
        };

        std::uint16_t unsigned16(const std::uint8_t* bytes) {
            return std::uint16_t(bytes[0]) | (std::uint16_t(bytes[1]) << 8);
        }

        float signed16(const std::uint8_t* bytes) {
            const int value = unsigned16(bytes);
            return static_cast<float>(value >= 0x8000 ? value - 0x10000 : value);
        }

        glm::vec3 axes(const std::uint8_t* bytes, const float scale) {
            return glm::vec3(signed16(bytes), signed16(bytes + 2), signed16(bytes + 4)) / scale;
        }
    } // namespace

    const SpaceMouseProfile* spaceMouseProfile(const std::uint16_t vendor, const std::uint16_t product) {
        for (const auto& profile : kProfiles)
            if (profile.vendor == vendor && profile.product == product)
                return &profile;
        return nullptr;
    }

    bool isSpaceMouse(const std::uint16_t vendor, const std::uint16_t product,
                      const std::uint16_t usage_page, const std::uint16_t usage) {
        if (usage_page == 1 && usage == 8)
            return true;
        // Missing usage information on some hidraw backends; never open the
        // keyboard/consumer-control collections of a composite device.
        return usage_page == 0 && usage == 0 && spaceMouseProfile(vendor, product);
    }

    float spaceMouseHidScale(const std::span<const std::uint8_t> descriptor, const float fallback) {
        struct Globals {
            std::uint32_t page = 0, width = 0, report = 0;
            std::int64_t minimum = 0, maximum = 0;
        } globals;
        std::vector<Globals> stack;
        bool motion_usage = false;
        float scale = 0.0f;
        for (std::size_t offset = 0; offset < descriptor.size();) {
            const auto prefix = descriptor[offset++];
            if (prefix == 0xfe) {
                if (descriptor.size() - offset < 2)
                    return fallback;
                const std::size_t length = descriptor[offset];
                offset += 2;
                if (descriptor.size() - offset < length)
                    return fallback;
                offset += length;
                continue;
            }
            const std::size_t size = (prefix & 3) == 3 ? 4 : prefix & 3;
            if (descriptor.size() - offset < size)
                return fallback;
            std::uint32_t value = 0;
            for (std::size_t i = 0; i < size; ++i)
                value |= std::uint32_t(descriptor[offset++]) << (8 * i);
            const auto signed_value = [&] {
                const bool negative = size && (value & (std::uint32_t{1} << (size * 8 - 1)));
                return std::int64_t(value) - (negative ? (std::int64_t{1} << (size * 8)) : 0);
            };
            const int type = (prefix >> 2) & 3;
            const int tag = prefix >> 4;
            if (type == 1) {
                switch (tag) {
                case 0: globals.page = value; break;
                case 1: globals.minimum = signed_value(); break;
                case 2: globals.maximum = globals.minimum < 0 ? signed_value() : value; break;
                case 7: globals.width = value; break;
                case 8: globals.report = value; break;
                case 10: stack.push_back(globals); break;
                case 11:
                    if (stack.empty())
                        return fallback;
                    globals = stack.back();
                    stack.pop_back();
                    break;
                }
            } else if (type == 2 && (tag == 0 || tag == 1)) {
                motion_usage |= globals.page == 1 && value >= 0x30 && value <= 0x35;
            } else if (type == 0) {
                if (tag == 8 && !(value & 1) && motion_usage && globals.width == 16 &&
                    (globals.report == 1 || globals.report == 2) && globals.minimum < 0 && globals.maximum > 0)
                    scale = std::max(scale, float(std::max(-globals.minimum, globals.maximum)));
                motion_usage = false;
            }
        }
        return scale > 0.0f ? scale : fallback;
    }

    std::optional<SpaceMouseReport> decodeSpaceMouseHid(const std::span<const std::uint8_t> bytes,
                                                        const float full_scale) {
        if (bytes.empty() || !std::isfinite(full_scale) || full_scale <= 0.0f)
            return std::nullopt;
        SpaceMouseReport report;
        switch (bytes[0]) {
        case 1: {
            if (bytes.size() < 7)
                return std::nullopt;
            const auto t = axes(bytes.data() + 1, full_scale);
            report.translation = glm::vec3(t.x, -t.z, t.y);
            if (bytes.size() >= 13) {
                const auto r = axes(bytes.data() + 7, full_scale);
                report.rotation = glm::vec3(r.x, -r.z, r.y);
            }
            break;
        }
        case 2: {
            if (bytes.size() < 7)
                return std::nullopt;
            const auto r = axes(bytes.data() + 1, full_scale);
            report.rotation = glm::vec3(r.x, -r.z, r.y);
            break;
        }
        case 3:
            if (bytes.size() < 2)
                return std::nullopt;
            report.button_mask = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t i = 1; i < std::min(bytes.size(), std::size_t{9}); ++i)
                report.buttons |= std::uint64_t(bytes[i]) << ((i - 1) * 8);
            break;
        case 0x1c:
        case 0x1d:
            if (bytes.size() < 3 || bytes.size() % 2 != 1)
                return std::nullopt;
            report.long_press = bytes[0] == 0x1d;
            report.button_mask = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t i = 1; i + 1 < bytes.size(); i += 2) {
                const auto button = unsigned16(bytes.data() + i);
                if (button > 0 && button <= 64)
                    report.buttons |= std::uint64_t{1} << (button - 1);
            }
            break;
        default:
            return std::nullopt;
        }
        return report;
    }

    std::optional<SpaceMouseReport> decodeSpaceMouseSpnav(const std::span<const std::uint8_t> bytes) {
        std::array<std::int32_t, 8> packet{};
        static_assert(sizeof(packet) == 32);
        if (bytes.size() != sizeof(packet))
            return std::nullopt;
        std::memcpy(packet.data(), bytes.data(), sizeof(packet));
        SpaceMouseReport report;
        if (packet[0] == 0) {
            // spacenavd normalizes evdev axes to +/-500 and maps USB Y/Z to -Z/-Y.
            report.translation = glm::vec3(packet[1], packet[2], -float(packet[3])) / 500.0f;
            report.rotation = glm::vec3(packet[4], packet[5], -float(packet[6])) / 500.0f;
        } else if ((packet[0] == 1 || packet[0] == 2) && packet[1] >= 0 && packet[1] < 64) {
            report.button_mask = std::uint64_t{1} << packet[1];
            report.buttons = packet[0] == 1 ? report.button_mask : 0;
        } else {
            return std::nullopt;
        }
        return report;
    }

    float spaceMouseResponse(const float value) {
        constexpr float dead_zone = 0.05f;
        if (!std::isfinite(value))
            return 0.0f;
        const float magnitude = std::clamp((std::abs(value) - dead_zone) / (1.0f - dead_zone), 0.0f, 1.0f);
        return std::copysign(magnitude * magnitude, value);
    }

    SpaceMouseMotion spaceMouseStep(const SpaceMouseMotion& motion, const float translation_speed,
                                    const float rotation_speed, const float delta_time,
                                    const bool object_mode, const bool keep_horizon) {
        SpaceMouseMotion step;
        if (!std::isfinite(delta_time) || delta_time <= 0.0f)
            return step;
        const float sign = object_mode ? -1.0f : 1.0f;
        for (int i = 0; i < 3; ++i) {
            step.translation[i] = sign * spaceMouseResponse(motion.translation[i]) * translation_speed * delta_time;
            step.rotation[i] = sign * spaceMouseResponse(motion.rotation[i]) * rotation_speed * delta_time;
        }
        if (keep_horizon)
            step.rotation.z = 0.0f;
        return step;
    }

    SpaceMouseSession::SpaceMouseSession(SpaceMouseSource& source) : source_(source) {}
    SpaceMouseSession::~SpaceMouseSession() { source_.close(); }

    bool SpaceMouseSession::poll(const Clock::time_point now) {
        using namespace std::chrono_literals;
        const auto previous = frame_;
        if (!frame_.connection) {
            if (now < next_scan_)
                return false;
            next_scan_ = now + 2s;
            frame_.connection = source_.open();
        } else {
            SpaceMouseReport report;
            const auto result = source_.read(report);
            if (result == SpaceMouseRead::Disconnected) {
                source_.close();
                frame_ = {};
                button_groups_ = {};
                next_scan_ = now + 2s;
            } else if (result == SpaceMouseRead::Report) {
                if (report.translation) {
                    frame_.motion.translation = *report.translation;
                    translation_time_ = now;
                }
                if (report.rotation) {
                    frame_.motion.rotation = *report.rotation;
                    rotation_time_ = now;
                }
                auto& group = button_groups_[report.long_press ? 1 : 0];
                group = (group & ~report.button_mask) | (report.buttons & report.button_mask);
                const auto buttons = button_groups_[0] | button_groups_[1];
                frame_.pressed |= buttons & ~frame_.buttons;
                frame_.buttons = buttons;
            }
        }
        if (now - translation_time_ > 250ms)
            frame_.motion.translation = glm::vec3(0.0f);
        if (now - rotation_time_ > 250ms)
            frame_.motion.rotation = glm::vec3(0.0f);
        return frame_.motion != previous.motion || frame_.buttons != previous.buttons ||
               frame_.pressed != previous.pressed || frame_.connection != previous.connection;
    }

    SpaceMouseFrame SpaceMouseSession::takeFrame() {
        const auto result = frame_;
        frame_.pressed = 0;
        return result;
    }
} // namespace lfs::vis::input
