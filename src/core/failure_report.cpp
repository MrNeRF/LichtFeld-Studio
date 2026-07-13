/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/failure_report.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <format>
#include <list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<stacktrace>)
#include <stacktrace>
#endif

#if defined(__cpp_lib_stacktrace) && __cpp_lib_stacktrace >= 202011L
#define LFS_HAS_STD_STACKTRACE 1
#elif defined(__unix__) || defined(__APPLE__)
#include <execinfo.h>
#define LFS_HAS_POSIX_BACKTRACE 1
#endif

namespace lfs::core {
    namespace {

        struct DedupDecision {
            bool emit_full;
            uint64_t count;
        };

        struct FailureReportEntry {
            std::string family;
            long long code;
            std::string site;
            uint64_t count;
            std::chrono::steady_clock::time_point last_full_time;
            uint64_t count_at_last_full;
        };

        struct RegisteredSectionProvider {
            std::string family;
            FailureReportSectionProvider provider;
        };

        constexpr size_t FAILURE_REPORT_DEDUP_CAPACITY = 64;
        std::mutex g_dedup_mutex;
        std::list<FailureReportEntry> g_dedup_entries;
        std::mutex g_section_provider_mutex;
        std::vector<RegisteredSectionProvider> g_section_providers;

        [[nodiscard]] DedupDecision decide_failure_report(
            const std::string_view family,
            const long long code,
            const std::string_view site) {
            const auto now = std::chrono::steady_clock::now();
            std::lock_guard lock(g_dedup_mutex);
            const auto entry = std::find_if(
                g_dedup_entries.begin(), g_dedup_entries.end(),
                [family, code, site](const FailureReportEntry& candidate) {
                    return candidate.code == code && candidate.family == family && candidate.site == site;
                });
            if (entry == g_dedup_entries.end()) {
                if (g_dedup_entries.size() == FAILURE_REPORT_DEDUP_CAPACITY) {
                    g_dedup_entries.pop_back();
                }
                g_dedup_entries.push_front(FailureReportEntry{
                    .family = std::string(family),
                    .code = code,
                    .site = std::string(site),
                    .count = 1,
                    .last_full_time = now,
                    .count_at_last_full = 1,
                });
                return {.emit_full = true, .count = 1};
            }

            ++entry->count;
            const bool emit_full =
                now - entry->last_full_time >= std::chrono::seconds(30) ||
                entry->count - entry->count_at_last_full + 1 >= 100;
            if (emit_full) {
                entry->last_full_time = now;
                entry->count_at_last_full = entry->count;
            }
            const uint64_t count = entry->count;
            g_dedup_entries.splice(g_dedup_entries.begin(), g_dedup_entries, entry);
            return {.emit_full = emit_full, .count = count};
        }

        [[nodiscard]] std::string detection_site(const SourceSite& location) {
            return std::format("{}:{}", location.file_name(), location.line());
        }

        void emit_failure_repeat_notice(const DedupDecision& decision,
                                        const SourceSite& location,
                                        const LogLevel level) {
            Logger::get().log_internal(
                level, location,
                std::format("LFS failure repeated x{} (same as above)", decision.count));
        }

        [[nodiscard]] FailureReportSectionProvider section_provider_for(
            const std::string_view family) {
            std::lock_guard lock(g_section_provider_mutex);
            const auto entry = std::find_if(
                g_section_providers.begin(), g_section_providers.end(),
                [family](const RegisteredSectionProvider& candidate) {
                    return candidate.family == family;
                });
            return entry == g_section_providers.end() ? nullptr : entry->provider;
        }

        [[nodiscard]] LogLevel log_level(const FailureReportSeverity severity) {
            return severity == FailureReportSeverity::Critical
                       ? LogLevel::Critical
                       : LogLevel::Error;
        }

    } // namespace

    void register_failure_report_section_provider(
        const std::string_view family,
        const FailureReportSectionProvider provider) {
        std::lock_guard lock(g_section_provider_mutex);
        const auto entry = std::find_if(
            g_section_providers.begin(), g_section_providers.end(),
            [family](const RegisteredSectionProvider& candidate) {
                return candidate.family == family;
            });
        if (entry == g_section_providers.end()) {
            g_section_providers.push_back({std::string(family), provider});
        } else {
            entry->provider = provider;
        }
    }

    std::string capture_host_stacktrace(const size_t skip_frames) {
#if defined(LFS_HAS_STD_STACKTRACE)
        std::ostringstream out;
        const auto trace = std::stacktrace::current(skip_frames + 1);
        if (trace.empty()) {
            return "  <unavailable>\n";
        }
        size_t index = 0;
        for (const auto& entry : trace) {
            out << "  #" << index++ << ' ' << entry << '\n';
        }
        return out.str();
#elif defined(LFS_HAS_POSIX_BACKTRACE)
        std::array<void*, 128> frames{};
        const int count = ::backtrace(frames.data(), static_cast<int>(frames.size()));
        if (count <= 0) {
            return "  <unavailable>\n";
        }
        char** symbols = ::backtrace_symbols(frames.data(), count);
        if (!symbols) {
            return "  <unavailable>\n";
        }
        std::ostringstream out;
        for (int i = static_cast<int>(skip_frames + 1); i < count; ++i) {
            out << "  #" << (i - static_cast<int>(skip_frames + 1)) << ' ' << symbols[i] << '\n';
        }
        std::free(symbols);
        return out.str();
#else
        (void)skip_frames;
        return "  <unavailable on this platform>\n";
#endif
    }

    std::string format_failure_report(
        const FailureReport& report,
        const std::string_view stacktrace) {
        const FailureReportSectionProvider provider = section_provider_for(report.family);
        std::ostringstream out;
        out << "========== LFS FAILURE REPORT ==========\n";
        out << "Family: " << report.family << '\n';
        if (!report.error.empty()) {
            out << "Error: " << report.error << '\n';
        }
        if (!report.contract.empty()) {
            out << "Contract: " << report.contract << '\n';
        }
        out << "Failed expression: " << report.expression << '\n';
        out << std::format("Detection site: {}:{} ({})\n",
                           report.location.file_name(), report.location.line(), report.location.function_name());
        if (!report.message.empty()) {
            out << "Context: " << report.message << '\n';
        }
        out << report.detail_sections;
        if (provider) {
            provider(out, FailureReportSectionPosition::BeforeStackTrace, report);
        }
        out << "Host stack trace:\n"
            << stacktrace;
        if (provider) {
            provider(out, FailureReportSectionPosition::AfterStackTrace, report);
        }
        out << "========================================";
        return out.str();
    }

    std::string format_failure_report(
        const std::string_view family,
        const std::string_view contract,
        const std::string_view expression,
        const std::string_view message,
        const SourceSite& location,
        const std::string_view stacktrace) {
        return format_failure_report(
            FailureReport{
                .family = family,
                .contract = contract,
                .expression = expression,
                .message = message,
                .location = location,
            },
            stacktrace);
    }

    std::string format_contract_failure_report(
        const std::string_view contract,
        const std::string_view expression,
        const std::string_view message,
        const SourceSite& location,
        const std::string_view stacktrace) {
        return format_failure_report(
            "tensor contract violation", contract, expression, message, location, stacktrace);
    }

    void emit_failure_report(
        const FailureReport& report,
        const FailureReportSeverity severity) {
        const std::string_view deduplication_family = report.deduplication_family.empty()
                                                          ? report.family
                                                          : report.deduplication_family;
        const DedupDecision decision = decide_failure_report(
            deduplication_family, report.deduplication_code, detection_site(report.location));
        const LogLevel level = log_level(severity);
        if (decision.emit_full) {
            const std::string stacktrace = capture_host_stacktrace(report.stacktrace_skip_frames);
            Logger::get().log_internal(level, report.location, format_failure_report(report, stacktrace));
        } else {
            emit_failure_repeat_notice(decision, report.location, level);
        }
    }

    void reset_failure_report_dedup_for_testing() noexcept {
        std::lock_guard lock(g_dedup_mutex);
        g_dedup_entries.clear();
    }

    bool decide_failure_report_for_testing(const std::string_view family,
                                           const long long code,
                                           const std::string_view site,
                                           uint64_t& out_count) {
        const DedupDecision decision = decide_failure_report(family, code, site);
        out_count = decision.count;
        return decision.emit_full;
    }

    namespace detail {

        [[noreturn]] void assertion_failed(
            const std::string_view contract,
            const std::string_view expression,
            const std::string_view message,
            const SourceSite location) {
            emit_failure_report(FailureReport{
                .family = "tensor contract violation",
                .contract = contract,
                .expression = expression,
                .message = message,
                .location = location,
                .deduplication_family = contract,
                .stacktrace_skip_frames = 2,
            });

            std::string error = std::format("{} failed: {}", contract, expression);
            if (!message.empty()) {
                error += " — ";
                error += message;
            }
            error += std::format(" ({}:{})", location.file_name(), location.line());
            throw std::runtime_error(error);
        }

    } // namespace detail

} // namespace lfs::core
