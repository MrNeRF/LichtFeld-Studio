/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/status_bar_pacman.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <utility>
#include <vector>

namespace lfs::vis::gui::pacman {
    namespace {
        struct Encounter {
            float power_step;
            float cherry_step;
            const char* ghost_color;
        };
        constexpr std::array kEncounters{Encounter{0.20f, 0.49f, "red"}, Encounter{0.65f, 0.94f, "pink"}};
        constexpr float kPowerDuration = 0.23f;
        constexpr float kGhostOffset = 0.15f;
    } // namespace

    Frame buildFrame(float progress, const float bar_dp, const int64_t animation_ms) {
        Frame frame;
        const float width = std::isfinite(bar_dp) ? std::max(0.0f, bar_dp) : 0.0f;
        const float travel = std::max(0.0f, width - 14.0f);
        progress = std::isfinite(progress) ? std::clamp(progress, 0.0f, 1.0f) : 0.0f;
        frame.left_dp = progress * travel;
        const int tick = static_cast<int>((std::max(int64_t{0}, animation_ms) / 100) % 4);
        const auto between = [&](float start, float end) { return progress >= start && progress < end; };
        frame.powered = std::ranges::any_of(kEncounters, [&](const Encounter& encounter) {
            return between(encounter.power_step, encounter.power_step + kPowerDuration);
        });
        constexpr std::array kMouthFrames{"closed", "half", "open", "half"};
        frame.sprite = std::format("../icon/pacman/{}{}.png", frame.powered ? "power-" : "", kMouthFrames[tick]);
        if (width < 28.0f)
            return frame;

        const auto image = [&](const char* kind, const std::string& sprite, const float x) {
            frame.scenery_rml += std::format(
                "<img class=\"pacman-item {}\" src=\"../icon/pacman/{}.png\" style=\"left:{:.2f}dp;\"/>",
                kind, sprite, std::clamp(x, 0.0f, travel));
        };
        std::vector<std::pair<float, float>> gaps;
        const auto reserve = [&](float step, float radius) { gaps.emplace_back(step * travel + 7.0f, radius); };
        for (const auto& [power_step, cherry_step, ghost_color] : kEncounters) {
            if (between(power_step - 0.16f, power_step)) {
                frame.scenery_rml += std::format(
                    "<div class=\"pacman-power-pellet\" style=\"left:{:.2f}dp; opacity:{:.2f};\"></div>",
                    power_step * travel + 3.0f, tick % 2 == 0 ? 1.0f : 0.65f);
                reserve(power_step, 12.0f);
            }
            if (between(cherry_step - 0.09f, cherry_step)) {
                image("pacman-cherries", "cherries", cherry_step * travel);
                reserve(cherry_step, 12.0f);
            }
            const float ghost_step = power_step + kGhostOffset;
            if (progress >= ghost_step) {
                const float retreat = (progress - ghost_step) / 0.035f;
                if (retreat < 1.0f)
                    image("pacman-eyes", "eyes", ghost_step * travel * (1.0f - retreat));
            } else if (between(power_step - 0.10f, ghost_step)) {
                const auto sprite = frame.powered && progress >= power_step
                                        ? std::string("frightened")
                                        : std::format("{}-{}", ghost_color, tick % 2 == 0 ? "a" : "b");
                image("pacman-ghost", sprite, ghost_step * travel);
                reserve(ghost_step, 14.0f);
            }
        }
        // Reserve space once per visible item instead of re-evaluating encounters for every dot.
        for (float x = 22.0f; x < width - 4.0f; x += 20.0f) {
            const bool occupied = std::ranges::any_of(gaps, [x](const auto& gap) {
                return std::abs(x - gap.first) < gap.second;
            });
            if (x > frame.left_dp + 12.0f && !occupied)
                frame.scenery_rml += std::format("<div class=\"pacman-pellet\" style=\"left:{:.2f}dp;\"></div>", x);
        }
        return frame;
    }
} // namespace lfs::vis::gui::pacman
