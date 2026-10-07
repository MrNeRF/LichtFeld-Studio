/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "indicators.hpp"

#include <algorithm>
#include <format>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::app {

    class TerminalProgressBar {
    public:
        explicit TerminalProgressBar(std::string prefix = "Converting") {
            bar_.set_option(indicators::option::Start("["));
#ifdef _WIN32
            bar_.set_option(indicators::option::BarWidth(38));
            bar_.set_option(indicators::option::Fill("="));
            bar_.set_option(indicators::option::Lead(">"));
            bar_.set_option(indicators::option::Remainder(" "));
#else
            bar_.set_option(indicators::option::BarWidth(40));
            bar_.set_option(indicators::option::Fill("█"));
            bar_.set_option(indicators::option::Lead("▌"));
            bar_.set_option(indicators::option::Remainder("░"));
#endif
            bar_.set_option(indicators::option::End("]"));
            bar_.set_option(indicators::option::PrefixText(std::move(prefix) + " "));
            bar_.set_option(indicators::option::ShowPercentage(true));
            bar_.set_option(indicators::option::ShowElapsedTime(true));
            bar_.set_option(indicators::option::ShowRemainingTime(true));
            bar_.set_option(indicators::option::ForegroundColor(indicators::Color::cyan));
            bar_.set_option(indicators::option::FontStyles(
                std::vector<indicators::FontStyle>{indicators::FontStyle::bold}));
        }

        bool report(const float progress, const std::string& stage) {
            const int percent = static_cast<int>(std::clamp(progress, 0.0f, 1.0f) * 100.0f);
            std::lock_guard lock(mutex_);
            if (percent == last_percent_ && stage == last_stage_) {
                return true;
            }
            last_percent_ = percent;
            last_stage_ = stage;
            bar_.set_option(indicators::option::PostfixText(std::format("{:<40}", stage)));
            bar_.set_progress(static_cast<size_t>(percent));
            return true;
        }

        void complete() {
            std::lock_guard lock(mutex_);
            if (!bar_.is_completed()) {
                bar_.set_progress(100);
                bar_.mark_as_completed();
                std::cout << std::endl;
            }
        }

        void abort() {
            std::lock_guard lock(mutex_);
            if (!bar_.is_completed()) {
                bar_.mark_as_completed();
                std::cout << std::endl;
            }
        }

        ~TerminalProgressBar() { abort(); }

    private:
        indicators::ProgressBar bar_;
        std::mutex mutex_;
        int last_percent_ = -1;
        std::string last_stage_;
    };

} // namespace lfs::app
