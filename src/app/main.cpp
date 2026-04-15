/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "app/converter.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "git_version.h"
#include "python/plugin_runner.hpp"
#include "python/runner.hpp"

#include <cstdlib>
#include <filesystem>
#include <print>

#include <pxr/base/plug/registry.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
    std::filesystem::path get_exe_directory() {
#ifdef _WIN32
        wchar_t buf[MAX_PATH]{};
        if (GetModuleFileNameW(nullptr, buf, MAX_PATH) > 0) {
            return std::filesystem::path(buf).parent_path();
        }
#else
        std::error_code ec;
        auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (!ec) return p.parent_path();
#endif
        return {};
    }

    // Register OpenUSD plugin resources deployed at <exe_dir>/../lib/usd/.
    // Must be called before any USD API usage (stage creation, schema lookup).
    // The env-var approach (PXR_PLUGINPATH_NAME) does not work reliably on
    // Windows because USD DLLs may initialise before main() runs.
    void configure_usd_plugins() {
        const auto exe_dir = get_exe_directory();
        if (exe_dir.empty()) return;

        auto usd_dir = exe_dir / ".." / "lib" / "usd";
        std::error_code ec;
        usd_dir = std::filesystem::canonical(usd_dir, ec);
        if (ec || !std::filesystem::is_directory(usd_dir, ec)) return;

        const std::string path_utf8 = lfs::core::path_to_utf8(usd_dir);

        // Also set the env var for any code that reads it directly.
#ifdef _WIN32
        _putenv_s("PXR_PLUGINPATH_NAME", path_utf8.c_str());
#else
        setenv("PXR_PLUGINPATH_NAME", path_utf8.c_str(), /*overwrite=*/0);
#endif

        // Programmatically register plugins so the Plug system finds them
        // regardless of compiled-in search paths or env var timing.
        pxr::PlugRegistry::GetInstance().RegisterPlugins(path_utf8);
    }
} // namespace

int main(int argc, char* argv[]) {
    configure_usd_plugins();

    auto result = lfs::core::args::parse_args(argc, argv);
    if (!result) {
        std::println(stderr, "Error: {}", result.error());
        return 1;
    }

    return std::visit([](auto&& mode) -> int {
        using T = std::decay_t<decltype(mode)>;

        if constexpr (std::is_same_v<T, lfs::core::args::HelpMode>) {
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::VersionMode>) {
            std::println("LichtFeld Studio {} ({})", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::WarmupMode>) {
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::ConvertMode>) {
            return lfs::app::run_converter(mode.params);
        } else if constexpr (std::is_same_v<T, lfs::core::args::PluginMode>) {
            return lfs::python::run_plugin_command(mode);
        } else if constexpr (std::is_same_v<T, lfs::core::args::TrainingMode>) {
            LOG_INFO("LichtFeld Studio");
            LOG_INFO("version {} | tag {}", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);

            if (mode.params->optimization.debug_python) {
                lfs::python::start_debugpy(mode.params->optimization.debug_python_port);
            }

            lfs::app::Application app;
            return app.run(std::move(mode.params));
        }
    },
                      std::move(*result));
}
