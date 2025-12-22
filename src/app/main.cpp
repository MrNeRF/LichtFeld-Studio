/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "core/argument_parser.hpp"
#include "core/converter.hpp"
#include "core/logger.hpp"

#include <print>

int main(int argc, char* argv[]) {
    auto result = lfs::core::args::parse_args(argc, argv);
    if (!result) {
        std::println(stderr, "Error: {}", result.error());
        return 1;
    }

    return std::visit([](auto&& mode) -> int {
        using T = std::decay_t<decltype(mode)>;

        if constexpr (std::is_same_v<T, lfs::core::args::HelpMode>) {
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::WarmupMode>) {
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::ConvertMode>) {
            return lfs::core::run_converter(mode.params);
        } else if constexpr (std::is_same_v<T, lfs::core::args::TrainingMode>) {
            LOG_INFO("LichtFeld Studio");
            lfs::app::Application app;
            return app.run(std::move(mode.params));
        }
    },
                      std::move(*result));
}
