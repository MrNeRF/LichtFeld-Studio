/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Only standalone loader tests use these logger/user-storage boundary fixtures.
// Production always links the normal core implementations.
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/user_paths.hpp"
#include <cassert>
#include <iostream>

// The storage fixture always returns success and never constructs an Error payload.
namespace lfs {
    Error::~Error() { assert(payload_ == nullptr); }
} // namespace lfs

namespace lfs::core {
    struct Logger::Impl {};
    Logger::Logger() : impl_(std::make_unique<Impl>()) {}
    Logger::~Logger() = default;
    Logger& Logger::get() {
        static Logger logger;
        return logger;
    }
    void Logger::log(LogLevel, const SourceSite&, std::string_view message) { std::cout << message << '\n'; }

    UserPaths::UserPaths(std::filesystem::path config, std::filesystem::path data,
                         std::filesystem::path cache, std::filesystem::path logs,
                         std::filesystem::path plugins, std::filesystem::path venv, bool unified)
        : config_dir_(std::move(config)), data_dir_(std::move(data)), cache_dir_(std::move(cache)), log_dir_(std::move(logs)), plugin_dir_(std::move(plugins)), venv_dir_(std::move(venv)), unified_root_(unified) {}
    lfs::Result<UserPaths> UserPaths::resolve(const UserPathOptions&) {
        const auto root = getExecutableDir() / "loader-test-storage";
        return UserPaths(root / "config", root / "data", root / "cache", root / "logs", root / "plugins", root / "venv", true);
    }
} // namespace lfs::core
