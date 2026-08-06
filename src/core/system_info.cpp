/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/system_info.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#include <winternl.h>
#else
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#endif

namespace lfs::core::system_info {

    namespace {

        std::string trim(std::string value) {
            const auto not_space = [](const unsigned char ch) { return !std::isspace(ch); };
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
            value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
            return value;
        }

        std::string cuda_runtime_version() {
            int version = 0;
            if (cudaRuntimeGetVersion(&version) != cudaSuccess || version <= 0)
                return {};
            return std::to_string(version / 1000) + "." +
                   std::to_string((version % 1000) / 10);
        }

#ifdef _WIN32
        void collect_os(SystemInfo& info) {
            using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
            const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
            if (!ntdll)
                return;
            const auto rtl_get_version = reinterpret_cast<RtlGetVersionFn>(
                GetProcAddress(ntdll, "RtlGetVersion"));
            if (!rtl_get_version)
                return;

            RTL_OSVERSIONINFOW version{};
            version.dwOSVersionInfoSize = sizeof(version);
            if (rtl_get_version(&version) != 0)
                return;
            info.os = "Windows";
            info.os_build = std::to_string(version.dwMajorVersion) + "." +
                            std::to_string(version.dwMinorVersion) + "." +
                            std::to_string(version.dwBuildNumber);
        }

        std::string collect_cpu() {
            std::array<int, 4> registers{};
            __cpuid(registers.data(), 0x80000000);
            if (static_cast<unsigned int>(registers[0]) < 0x80000004)
                return {};

            std::array<char, 49> brand{};
            for (int leaf = 0; leaf < 3; ++leaf) {
                __cpuid(registers.data(), 0x80000002 + leaf);
                std::memcpy(brand.data() + leaf * 16, registers.data(), 16);
            }
            return trim(std::string(brand.data()));
        }

        std::uint64_t collect_ram_mb() {
            MEMORYSTATUSEX state{};
            state.dwLength = sizeof(state);
            if (!GlobalMemoryStatusEx(&state))
                return 0;
            return static_cast<std::uint64_t>(state.ullTotalPhys / (1024ULL * 1024ULL));
        }
#else
        std::string unquote_os_release(std::string value) {
            value = trim(std::move(value));
            if (value.size() < 2 ||
                (value.front() != '"' && value.front() != '\'') ||
                value.back() != value.front())
                return value;
            value = value.substr(1, value.size() - 2);
            std::string result;
            result.reserve(value.size());
            bool escaped = false;
            for (const char ch : value) {
                if (escaped) {
                    result.push_back(ch);
                    escaped = false;
                } else if (ch == '\\') {
                    escaped = true;
                } else {
                    result.push_back(ch);
                }
            }
            if (escaped)
                result.push_back('\\');
            return result;
        }

        void collect_os(SystemInfo& info) {
            std::ifstream release("/etc/os-release");
            std::string line;
            while (std::getline(release, line)) {
                constexpr std::string_view prefix = "PRETTY_NAME=";
                if (line.starts_with(prefix)) {
                    info.os = unquote_os_release(line.substr(prefix.size()));
                    break;
                }
            }

            utsname name{};
            if (uname(&name) == 0)
                info.os_build = name.release;
        }

        std::string collect_cpu() {
            std::ifstream cpuinfo("/proc/cpuinfo");
            std::string line;
            while (std::getline(cpuinfo, line)) {
                const auto separator = line.find(':');
                if (separator == std::string::npos)
                    continue;
                if (trim(line.substr(0, separator)) == "model name")
                    return trim(line.substr(separator + 1));
            }
            return {};
        }

        std::uint64_t collect_ram_mb() {
            struct sysinfo state {};
            if (::sysinfo(&state) != 0)
                return 0;
            const auto bytes = static_cast<std::uint64_t>(state.totalram) *
                               static_cast<std::uint64_t>(state.mem_unit);
            return bytes / (1024ULL * 1024ULL);
        }
#endif

    } // namespace

    SystemInfo collect() noexcept {
        SystemInfo info;
        try {
            collect_os(info);
        } catch (...) {
        }
        try {
            info.cpu = collect_cpu();
        } catch (...) {
        }
        try {
            info.ram_mb = collect_ram_mb();
        } catch (...) {
        }
        try {
            info.cuda_runtime = cuda_runtime_version();
        } catch (...) {
        }
        return info;
    }

} // namespace lfs::core::system_info
