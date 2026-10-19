/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lfs::test::long_paths {

    namespace fs = std::filesystem;

    // Path lengths on both sides of the 248-unit switch to extended-length paths, dense where
    // temporary-file suffixes of 30 to 50 units push a shorter final path across it.
    inline constexpr std::array kLengthsAroundExtendedPathSwitch{
        std::size_t{160}, std::size_t{196}, std::size_t{200}, std::size_t{202}, std::size_t{204},
        std::size_t{206}, std::size_t{208}, std::size_t{212}, std::size_t{220}, std::size_t{230},
        std::size_t{247}, std::size_t{248}, std::size_t{259}, std::size_t{260}, std::size_t{261},
        std::size_t{320}};

    inline const fs::path& unicode_folder() {
        static const auto folder = core::utf8_to_path("проект 保存 é 🙂");
        return folder;
    }

    class LongPathDirectory {
    public:
        explicit LongPathDirectory(const fs::path& root = fs::temp_directory_path()) {
            static std::atomic_uint64_t sequence{0};
            path = root /
                   std::format("lfs-long-path-{}-{}", std::chrono::steady_clock::now().time_since_epoch().count(),
                               sequence.fetch_add(1));
            fs::create_directories(path);
        }
        ~LongPathDirectory() {
            std::error_code ignored;
            core::native_fs::remove_all(path, ignored);
        }
        LongPathDirectory(const LongPathDirectory&) = delete;
        LongPathDirectory& operator=(const LongPathDirectory&) = delete;

        fs::path path;
    };

    // directory / filler components / leaf, exactly `length` native units long.
    inline fs::path with_native_length(const fs::path& directory, const std::size_t length, const fs::path& leaf) {
        const std::size_t fixed = directory.native().size() + 1 + leaf.native().size();
        if (length < fixed || length == fixed + 1) {
            throw std::runtime_error("requested path length does not fit its fixed components");
        }
        const std::size_t remaining = length - fixed;
        const std::size_t components = (remaining + 100) / 101;
        std::size_t characters = remaining - components;
        auto path = directory;
        for (std::size_t i = 0; i < components; ++i) {
            const std::size_t size = characters / (components - i);
            path /= std::string(size, static_cast<char>('a' + i % 26));
            characters -= size;
        }
        return path / leaf;
    }

#ifdef _WIN32
    // A share provisioned for the test run (the Windows CI job creates one): LFS_TEST_UNC_ROOT is its network path
    // and LFS_TEST_UNC_LOCAL the folder it exposes.
    struct NetworkShare {
        fs::path network;
        fs::path local;
    };

    inline std::optional<std::wstring> environment(const wchar_t* const name) {
        const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
        if (size == 0) {
            return std::nullopt;
        }
        std::wstring value(size, L'\0');
        value.resize(GetEnvironmentVariableW(name, value.data(), size));
        return value;
    }

    inline std::optional<NetworkShare> network_share() {
        const auto network = environment(L"LFS_TEST_UNC_ROOT");
        const auto local = environment(L"LFS_TEST_UNC_LOCAL");
        std::error_code error;
        if (!network || !local || !fs::is_directory(fs::path(*network), error)) {
            return std::nullopt;
        }
        return NetworkShare{fs::path(*network), fs::path(*local)};
    }

    inline bool network_share_required() { return environment(L"LFS_TEST_REQUIRE_UNC").has_value(); }
#endif

} // namespace lfs::test::long_paths
