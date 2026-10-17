/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/generated_keymap_defaults.hpp"
#include "input/input_bindings.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace lfs::vis::input::test {
    class KeymapTest : public ::testing::Test {
    protected:
        void SetUp() override {
            persistence_ = InputBindings::isPersistenceEnabled();
            InputBindings::setPersistenceEnabled(false);
        }
        void TearDown() override { InputBindings::setPersistenceEnabled(persistence_); }

    private:
        bool persistence_ = true;
    };

    class ScopedKeymapHome {
    public:
        explicit ScopedKeymapHome(std::filesystem::path root) : root_(std::move(root)), enabled_(InputBindings::isPersistenceEnabled()) {
            std::error_code ec;
            std::filesystem::remove_all(root_, ec);
            if (const char* old = std::getenv("LFS_HOME"))
                previous_ = old;
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", root_.string().c_str());
#else
            (void)setenv("LFS_HOME", root_.string().c_str(), 1);
#endif
            InputBindings::setPersistenceEnabled(true);
            std::filesystem::create_directories(root_ / "config" / "keymaps");
        }
        ~ScopedKeymapHome() {
            InputBindings::setPersistenceEnabled(enabled_);
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", previous_ ? previous_->c_str() : "");
#else
            if (previous_)
                (void)setenv("LFS_HOME", previous_->c_str(), 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
            std::error_code ec;
            std::filesystem::remove_all(root_, ec);
        }

    private:
        std::filesystem::path root_;
        std::optional<std::string> previous_;
        bool enabled_;
    };
    // The version this build saves: the newest recorded history is always the previous one.
    inline constexpr int currentProfileVersion() { return kLatestHistoricalProfileVersion + 1; }

    inline void writeProfile(const std::filesystem::path& path, const nlohmann::json& profile) {
        std::ofstream output(path, std::ios::binary);
        output << profile.dump(2);
    }

    inline nlohmann::json historicalSavedProfile(const std::string_view name) {
        const auto& history = historicalKeymapDefaults();
        for (const auto& snapshot : history.at("profiles")) {
            if (snapshot.at("name").get_ref<const std::string&>() != name)
                continue;
            const auto& data = history.at("data").at(snapshot.at("data").get<size_t>());
            auto bindings = data.at("bindings");
            for (auto& binding : bindings)
                binding.erase("action_symbol");
            return {{"name", "Default"},
                    {"version", snapshot.at("version")},
                    {"bindings", std::move(bindings)}};
        }
        throw std::runtime_error("Missing historical keymap profile");
    }

    inline const nlohmann::json& historicalProfiles() {
        return historicalKeymapDefaults().at("profiles");
    }
} // namespace lfs::vis::input::test
