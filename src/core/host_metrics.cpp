/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/host_metrics.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// clang-format off: windows.h must precede psapi.h.
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <unistd.h>
#endif

namespace lfs::core::host_metrics {
    namespace {

        struct CpuState {
            std::uint64_t idle = 0;
            std::uint64_t total = 0;
        };

        struct Previous {
            bool valid = false;
            std::uint64_t process_ticks = 0;
            std::uint64_t system_total_ticks = 0;
            std::uint64_t system_idle_ticks = 0;
            std::vector<CpuState> cores;
            std::chrono::steady_clock::time_point timestamp{};
        };

        thread_local Previous previous;

#ifndef _WIN32
        std::uint64_t parse_u64(const std::string& value) {
            try {
                return static_cast<std::uint64_t>(std::stoull(value));
            } catch (...) {
                return 0;
            }
        }

        bool read_cpu(std::uint64_t& process_ticks,
                      CpuState& sys_ticks,
                      std::vector<CpuState>& cores) {
            std::ifstream self("/proc/self/stat");
            std::string line;
            if (!self || !std::getline(self, line))
                return false;
            const auto close = line.rfind(')');
            if (close == std::string::npos || close + 2 >= line.size())
                return false;
            std::istringstream fields(line.substr(close + 2));
            std::string value;
            std::vector<std::string> values;
            while (fields >> value)
                values.push_back(value);
            // After the command name, field 3 is at index 0: utime is field 14.
            if (values.size() <= 12)
                return false;
            process_ticks = parse_u64(values[11]) + parse_u64(values[12]);

            std::ifstream stat("/proc/stat");
            if (!stat)
                return false;
            std::string name;
            std::uint64_t user = 0, nice = 0, system_ticks = 0, idle = 0, iowait = 0,
                          irq = 0, softirq = 0, steal = 0;
            cores.clear();
            while (stat >> name) {
                if (name == "cpu") {
                    stat >> user >> nice >> system_ticks >> idle >> iowait >> irq >> softirq >> steal;
                    const auto busy = user + nice + system_ticks + irq + softirq + steal;
                    sys_ticks.idle = idle + iowait;
                    sys_ticks.total = busy + sys_ticks.idle;
                } else if (name.size() > 3 && name.rfind("cpu", 0) == 0 &&
                           std::isdigit(static_cast<unsigned char>(name[3]))) {
                    stat >> user >> nice >> system_ticks >> idle >> iowait >> irq >> softirq >> steal;
                    const auto busy = user + nice + system_ticks + irq + softirq + steal;
                    cores.push_back({idle + iowait, busy + idle + iowait});
                } else {
                    std::getline(stat, line);
                }
            }
            return sys_ticks.total != 0;
        }

        void read_memory(Sample& result) {
            std::ifstream statm("/proc/self/statm");
            std::uint64_t pages = 0;
            if (statm)
                statm >> pages >> pages;
            if (pages != 0)
                result.process_rss_bytes = static_cast<std::size_t>(pages) *
                                           static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));

            std::ifstream meminfo("/proc/meminfo");
            std::string key;
            std::uint64_t value = 0;
            std::string unit;
            std::uint64_t total_kib = 0;
            std::uint64_t available_kib = 0;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemTotal:")
                    total_kib = value;
                else if (key == "MemAvailable:")
                    available_kib = value;
            }
            if (total_kib != 0 && available_kib <= total_kib) {
                result.system_total_bytes = static_cast<std::size_t>(total_kib) * 1024;
                result.system_used_bytes = static_cast<std::size_t>(total_kib - available_kib) * 1024;
                result.ram_valid = true;
            }
        }
#else
        void read_memory(Sample& result) {
            PROCESS_MEMORY_COUNTERS counters{};
            if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
                result.process_rss_bytes = static_cast<std::size_t>(counters.WorkingSetSize);
            MEMORYSTATUSEX memory{};
            memory.dwLength = sizeof(memory);
            if (GlobalMemoryStatusEx(&memory)) {
                result.system_total_bytes = static_cast<std::size_t>(memory.ullTotalPhys);
                result.system_used_bytes = result.system_total_bytes -
                                           static_cast<std::size_t>(memory.ullAvailPhys);
                result.ram_valid = true;
            }
        }
#endif

    } // namespace

    Sample sample() {
        Sample result;
        read_memory(result);
#ifndef _WIN32
        std::uint64_t process_ticks = 0;
        CpuState system;
        std::vector<CpuState> cores;
        if (read_cpu(process_ticks, system, cores)) {
            result.per_core_cpu_percent.resize(cores.size(), -1.f);
            const auto now = std::chrono::steady_clock::now();
            if (previous.valid) {
                const auto wall_seconds = std::chrono::duration<float>(now - previous.timestamp).count();
                const auto process_delta = process_ticks - previous.process_ticks;
                const auto system_delta = system.total - previous.system_total_ticks;
                const auto ticks_per_second = static_cast<float>(::sysconf(_SC_CLK_TCK));
                if (wall_seconds > 0.f && ticks_per_second > 0.f) {
                    // Normalize process usage to all cores: 100% means all cores busy.
                    const auto cores_count = std::max<std::size_t>(1, cores.size());
                    result.process_cpu_percent = std::clamp(
                        100.f * static_cast<float>(process_delta) /
                            (wall_seconds * ticks_per_second * static_cast<float>(cores_count)),
                        0.f, 100.f);
                    result.cpu_valid = system_delta != 0;
                    for (std::size_t i = 0; i < cores.size() && i < previous.cores.size(); ++i) {
                        const auto total_delta = cores[i].total - previous.cores[i].total;
                        const auto idle_delta = cores[i].idle - previous.cores[i].idle;
                        if (total_delta != 0)
                            result.per_core_cpu_percent[i] = std::clamp(
                                100.f * static_cast<float>(total_delta - idle_delta) /
                                    static_cast<float>(total_delta),
                                0.f, 100.f);
                    }
                }
            }
            previous.valid = true;
            previous.process_ticks = process_ticks;
            previous.system_total_ticks = system.total;
            previous.system_idle_ticks = system.idle;
            previous.cores = std::move(cores);
            previous.timestamp = now;
        }
#else
        result.per_core_cpu_percent.resize(std::thread::hardware_concurrency(), -1.f);
#endif
        return result;
    }

} // namespace lfs::core::host_metrics
