/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "visualizer/window/x11_input_modifier.hpp"

#include <array>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <string_view>

namespace {

    TEST(X11InputModifierTest, SelectsLocalXimAndRestoresAfterWindowCreationFailure) {
        using lfs::vis::window::selectX11InputModifier;
        using lfs::vis::window::withX11InputModifier;

        const auto ibus = selectX11InputModifier("@im=ibus", false);
        ASSERT_TRUE(ibus.override_value.has_value());
        ASSERT_TRUE(ibus.restore_value.has_value());
        EXPECT_EQ(*ibus.override_value, "@im=none");
        EXPECT_EQ(*ibus.restore_value, "@im=ibus");

        for (const auto value : {std::string_view("@im=fcitx"), std::string_view("@im=xim")}) {
            const auto plan = selectX11InputModifier(value, false);
            EXPECT_FALSE(plan.override_value.has_value());
            EXPECT_FALSE(plan.restore_value.has_value());
        }
        const auto unset = selectX11InputModifier(std::nullopt, false);
        EXPECT_FALSE(unset.override_value.has_value());
        EXPECT_FALSE(unset.restore_value.has_value());

        std::optional<std::string> environment = "@im=ibus";
        const void* const window = withX11InputModifier(
            [&] { return environment ? environment->c_str() : nullptr; },
            [&](const std::string& value) {
                environment = value;
                return 0;
            },
            false,
            [&] {
                EXPECT_EQ(environment, "@im=none");
                return static_cast<const void*>(nullptr); // SDL_CreateWindow failed after SDL_Init succeeded.
            });
        EXPECT_EQ(window, nullptr);
        EXPECT_EQ(environment, "@im=ibus");

        for (const auto& value : std::array<std::optional<std::string>, 3>{
                 std::string("@im=fcitx"), std::string("@im=xim"), std::nullopt}) {
            environment = value;
            const auto before = environment;
            withX11InputModifier(
                [&] { return environment ? environment->c_str() : nullptr; },
                [&](const std::string& replacement) {
                    environment = replacement;
                    return 0;
                },
                false,
                [&] { EXPECT_EQ(environment, before); });
            EXPECT_EQ(environment, before);
        }
    }

    TEST(X11InputModifierTest, LeavesUsableIbusXimSelected) {
        const auto plan = lfs::vis::window::selectX11InputModifier("@im=ibus", true);
        EXPECT_FALSE(plan.override_value.has_value());
        EXPECT_FALSE(plan.restore_value.has_value());

        std::optional<std::string> environment = "@im=ibus";
        int set_calls = 0;
        lfs::vis::window::withX11InputModifier(
            [&] { return environment->c_str(); },
            [&](const std::string& value) {
                ++set_calls;
                environment = value;
                return 0;
            },
            true,
            [&] {
                EXPECT_EQ(environment, "@im=ibus");
                return true;
            });
        EXPECT_EQ(set_calls, 0);
        EXPECT_EQ(environment, "@im=ibus");
    }

} // namespace
