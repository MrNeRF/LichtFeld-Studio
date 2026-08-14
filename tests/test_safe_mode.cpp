/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "python/runner.hpp"

#include <cstdlib>
#include <optional>
#include <string>

namespace {

    class ScopedSafeModeEnvironment {
    public:
        explicit ScopedSafeModeEnvironment(const std::optional<std::string>& value) {
            if (const char* previous = std::getenv("LFS_SAFE_MODE"))
                previous_ = previous;
            set(value);
        }

        ~ScopedSafeModeEnvironment() { set(previous_); }

    private:
        static void set(const std::optional<std::string>& value) {
#ifdef _WIN32
            (void)_putenv_s("LFS_SAFE_MODE", value ? value->c_str() : "");
#else
            if (value)
                (void)setenv("LFS_SAFE_MODE", value->c_str(), 1);
            else
                (void)unsetenv("LFS_SAFE_MODE");
#endif
        }

        std::optional<std::string> previous_;
    };

    TEST(SafeModeContract, NormalLaunchPreservesExternalSafeModeFlag) {
        const ScopedSafeModeEnvironment environment("1");
        lfs::python::set_user_plugin_loading_enabled(true);
        ASSERT_NE(std::getenv("LFS_SAFE_MODE"), nullptr);
        EXPECT_STREQ(std::getenv("LFS_SAFE_MODE"), "1");
    }

    TEST(SafeModeContract, EnablingSafeModeSetsProcessFlag) {
        const ScopedSafeModeEnvironment environment(std::nullopt);
        lfs::python::set_user_plugin_loading_enabled(false);
        ASSERT_NE(std::getenv("LFS_SAFE_MODE"), nullptr);
        EXPECT_STREQ(std::getenv("LFS_SAFE_MODE"), "1");
        lfs::python::set_user_plugin_loading_enabled(true);
    }

} // namespace
