/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_reporter.hpp"

#include "core/failure_report.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cstdio>
#include <format>
#include <functional>
#include <string>
#include <thread>

namespace lfs::core {

    namespace {

        // Set for the duration of one report() call on this thread. Detects
        // a failure occurring while reporting (e.g. a log handler that
        // itself calls report()) so the inner call takes the fallback path
        // instead of recursing into Logger/FailureReport again.
        thread_local bool g_reporting = false;

        class ReportingGuard {
        public:
            ReportingGuard() noexcept { g_reporting = true; }
            ~ReportingGuard() noexcept { g_reporting = false; }
            ReportingGuard(const ReportingGuard&) = delete;
            ReportingGuard& operator=(const ReportingGuard&) = delete;
        };

        [[nodiscard]] std::uint64_t current_thread_id() noexcept {
            static thread_local const std::uint64_t id =
                static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
            return id;
        }

        [[nodiscard]] const char* channel_name(const ReportChannel channel) noexcept {
            switch (channel) {
            case ReportChannel::OwnerLog: return "OwnerLog";
            case ReportChannel::ProcessBoundary: return "ProcessBoundary";
            }
            return "Unknown";
        }

        [[nodiscard]] FailureReportSeverity to_failure_report_severity(const Severity severity) noexcept {
            return severity == Severity::Fatal ? FailureReportSeverity::Critical
                                               : FailureReportSeverity::Error;
        }

        [[nodiscard]] LogLevel to_log_level(const FailureReportSeverity severity) noexcept {
            return severity == FailureReportSeverity::Critical ? LogLevel::Critical : LogLevel::Error;
        }

        [[nodiscard]] bool capture_stack_policy(const ErrorCode code) noexcept {
            switch (code) {
            case ErrorCode::ContractViolation:
            case ErrorCode::Internal:
            case ErrorCode::DeviceLost:
                return true;
            case ErrorCode::NotFound:
            case ErrorCode::Cancelled:
                return Logger::get().is_enabled(LogLevel::Debug);
            default:
                return false;
            }
        }

        // Section 5.2 fingerprint: code + native code + detection source +
        // top context-frame operation. `code` and `domain` already occupy
        // FailureReport's deduplication_code/deduplication_family slots; this
        // folds the remaining two dimensions into deduplication_site rather
        // than adding a second dedup engine. The two immortal OOM/unknown
        // seeds (make_error's allocation-failure fallback) carry zero frames
        // by construction, so this degrades to a fixed key instead of
        // indexing an empty span.
        [[nodiscard]] std::string fingerprint_site(const Error& error) {
            if (error.frames().empty()) {
                return "<immortal-seed, no frames>";
            }
            const ErrorFrame& detection_frame = error.frames().front();
            const ErrorFrame& top_frame = error.frames().back();
            const std::int64_t native_code = error.native().has_value() ? error.native()->code : 0;
            return std::format("{}:{}|native={}|op={}", detection_frame.source.file_name(),
                               detection_frame.source.line(), native_code, top_frame.operation);
        }

        // Same empty-frames degradation as fingerprint_site: falls back to
        // this call site so FailureReport::location always has a valid
        // SourceSite to render as "Detection site:".
        [[nodiscard]] core::SourceSite detection_location(const Error& error) noexcept {
            return error.frames().empty() ? LFS_SOURCE_SITE_CURRENT() : error.frames().front().source;
        }

        [[nodiscard]] std::string build_detail_sections(const Error& error, const ReportChannel channel) {
            return std::format("  thread_id={:#x}\n  channel={}\n{}", current_thread_id(),
                               channel_name(channel), format_for_developer(error));
        }

        // Bounded, allocation-averse last resort: used when Logger is not
        // ready, the reporting pipeline itself threw, or a call is a process
        // boundary that must guarantee stderr visibility. Mirrors
        // crash_handler.cpp's fprintf-into-fixed-buffer pattern rather than
        // routing back through Logger or std::format for the header line.
        void write_stderr_fallback(const Error& error, const char* reason) noexcept {
            static constexpr char kAbsoluteFallback[] =
                "lichtfeld: ErrorReporter fallback formatting failed; error suppressed\n";
            try {
                char header[192];
                const int header_length = std::snprintf(
                    header, sizeof(header), "lichtfeld: ErrorReporter stderr fallback (%s)\n", reason);
                if (header_length > 0) {
                    const auto length = std::min<std::size_t>(static_cast<std::size_t>(header_length),
                                                              sizeof(header) - 1);
                    std::fwrite(header, 1, length, stderr);
                }
                std::string formatted;
                try {
                    formatted = format_for_developer(error);
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): this is the fallback path itself; a static
                    // literal replaces the formatted body instead of retrying formatting.
                    formatted = "lfs::Error (formatting failed)";
                }
                std::fwrite(formatted.data(), 1, formatted.size(), stderr);
                std::fputc('\n', stderr);
                std::fflush(stderr);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): absolute last resort below the fallback path
                // itself; writes a fixed literal since nothing here may allocate or format.
                std::fwrite(kAbsoluteFallback, 1, sizeof(kAbsoluteFallback) - 1, stderr);
                std::fflush(stderr);
            }
        }

    } // namespace

    ErrorReporter& ErrorReporter::get() {
        static ErrorReporter instance;
        return instance;
    }

    void ErrorReporter::report(const Error& error, const ReportChannel channel) const noexcept {
        if (g_reporting) {
            write_stderr_fallback(error, "recursive report suppressed");
            return;
        }
        const ReportingGuard guard;

        try {
            const FailureReportSeverity severity = to_failure_report_severity(error.severity());
            const LogLevel level = to_log_level(severity);
            Logger& logger = Logger::get();
            const bool logger_unavailable = !logger.is_ready();

            if (channel == ReportChannel::ProcessBoundary || logger_unavailable) {
                write_stderr_fallback(
                    error, logger_unavailable ? "logger not initialized" : "process boundary");
            }
            if (logger_unavailable || !logger.is_enabled(level)) {
                return; // fast path: nothing further would be observable through Logger
            }

            const std::string detail_sections = build_detail_sections(error, channel);
            const std::string dedup_site = fingerprint_site(error);
            const FailureReport report{
                .family = lfs::to_string(error.domain()),
                .error = lfs::to_string(error.code()),
                .message = error.user_message().empty() ? error.detail() : error.user_message(),
                .detail_sections = detail_sections,
                .location = detection_location(error),
                .deduplication_family = lfs::to_string(error.domain()),
                .deduplication_code = static_cast<long long>(error.code()),
                .deduplication_site = dedup_site,
                .stacktrace_skip_frames = 1,
                .capture_stack = capture_stack_policy(error.code()),
            };
            emit_failure_report(report, severity);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): report() is the owner-level noexcept boundary;
            // anything the pipeline throws (Logger, FailureReport, a section provider)
            // degrades to the fixed-buffer stderr fallback instead of propagating.
            write_stderr_fallback(error, "reporting pipeline threw");
        }
    }

    bool should_capture_stack_for_testing(const ErrorCode code) noexcept {
        return capture_stack_policy(code);
    }

    std::uint64_t error_fingerprint(const Error& error) noexcept {
        try {
            // fingerprint_site already folds detection source + native code +
            // top-frame operation; combine with code + domain (the two
            // dimensions FailureReport keeps in its separate dedup slots) so the
            // key covers the full Section 5.2 fingerprint.
            std::uint64_t hash = std::hash<std::string>{}(fingerprint_site(error));
            const std::uint64_t code_domain =
                (static_cast<std::uint64_t>(error.code()) << 16) ^
                static_cast<std::uint64_t>(error.domain());
            hash ^= code_domain + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
            return hash;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): noexcept dedup helper; an allocation
            // failure while formatting the site degrades to a fixed key rather
            // than propagating into ErrorBus::publish.
            return 0;
        }
    }

} // namespace lfs::core
