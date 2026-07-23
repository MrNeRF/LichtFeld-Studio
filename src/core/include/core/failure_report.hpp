/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/source_site.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

namespace lfs::core {

    enum class FailureReportSectionPosition : uint8_t {
        BeforeStackTrace,
        AfterStackTrace,
    };

    struct FailureReport;
    using FailureReportSectionProvider = void (*)(
        std::ostream& out,
        FailureReportSectionPosition position,
        const FailureReport& report);

    enum class FailureReportSeverity : uint8_t {
        Error,
        Critical,
    };

    struct LFS_CORE_API FailureReport {
        std::string_view family;
        std::string_view error;
        std::string_view contract;
        std::string_view expression;
        std::string_view message;
        std::string_view detail_sections;
        SourceSite location;
        std::string_view deduplication_family;
        long long deduplication_code = 0;
        // Overrides the file:line key derived from `location` when non-empty.
        // lfs::Error reports fold native status and top context-frame
        // operation into the fingerprint (error-architecture-analysis.md
        // Section 5.2) without changing the human-readable "Detection site"
        // line, which stays keyed off `location`.
        std::string_view deduplication_site;
        size_t stacktrace_skip_frames = 0;
        // Assertions/contracts always want a trace. lfs::Error reports apply
        // a per-code policy (Section 5.4) and set this to false when a trace
        // is not warranted, so emit_failure_report skips capturing one.
        bool capture_stack = true;
    };

    LFS_CORE_API void register_failure_report_section_provider(
        std::string_view family,
        FailureReportSectionProvider provider);

    LFS_CORE_API std::string capture_host_stacktrace(size_t skip_frames = 0);
    LFS_CORE_API std::string format_failure_report(
        const FailureReport& report,
        std::string_view stacktrace);
    LFS_CORE_API std::string format_failure_report(
        std::string_view family,
        std::string_view contract,
        std::string_view expression,
        std::string_view message,
        const SourceSite& location,
        std::string_view stacktrace);
    LFS_CORE_API std::string format_contract_failure_report(
        std::string_view contract,
        std::string_view expression,
        std::string_view message,
        const SourceSite& location,
        std::string_view stacktrace);
    LFS_CORE_API void emit_failure_report(
        const FailureReport& report,
        FailureReportSeverity severity = FailureReportSeverity::Error);

    LFS_CORE_API void reset_failure_report_dedup_for_testing() noexcept;
    LFS_CORE_API bool decide_failure_report_for_testing(
        std::string_view family,
        long long code,
        std::string_view site,
        uint64_t& out_count);

    namespace detail {

        [[noreturn]] LFS_CORE_API void assertion_failed(
            std::string_view contract,
            std::string_view expression,
            std::string_view message,
            SourceSite location);

    } // namespace detail

} // namespace lfs::core
