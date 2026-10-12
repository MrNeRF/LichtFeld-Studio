/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::io {

    enum class AtomicOutputTempName {
        AppendSuffix,
        PreserveExtension
    };

    inline std::filesystem::path make_atomic_temp_output_path_for_suffix(
        const std::filesystem::path& output_path,
        const std::string& suffix,
        AtomicOutputTempName name_style = AtomicOutputTempName::AppendSuffix) {
        std::filesystem::path temporary;
        if (name_style == AtomicOutputTempName::PreserveExtension && output_path.has_extension()) {
            auto temp_name = output_path.stem();
            temp_name += suffix;
            temp_name += output_path.extension();
            temporary = output_path.parent_path() / temp_name;
        } else {
            temporary = output_path;
            temporary += suffix;
        }

        if (temporary.filename().native().size() <= 255)
            return temporary;

#ifdef _WIN32
        const auto process_id = GetCurrentProcessId();
#else
        const auto process_id = ::getpid();
#endif
        auto short_name = std::filesystem::path(std::format(".lfs.{}", process_id));
        short_name += suffix;
        if (name_style == AtomicOutputTempName::PreserveExtension && output_path.has_extension())
            short_name += output_path.extension();
        return output_path.parent_path() / short_name;
    }

    inline std::filesystem::path make_atomic_temp_output_path(
        const std::filesystem::path& output_path,
        AtomicOutputTempName name_style = AtomicOutputTempName::AppendSuffix) {
        static std::atomic_uint64_t counter{0};

        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
        const auto process_id = GetCurrentProcessId();
#else
        const auto process_id = ::getpid();
#endif
        const auto unique_suffix =
            std::format(".{}.{}.{}.tmp", ticks, process_id, counter.fetch_add(1, std::memory_order_relaxed));

        return make_atomic_temp_output_path_for_suffix(output_path, unique_suffix, name_style);
    }

} // namespace lfs::io
