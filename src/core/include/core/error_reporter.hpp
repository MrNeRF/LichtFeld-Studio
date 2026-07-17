/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/error_codes.hpp"
#include "core/export.hpp"

// lfs::core::ErrorReporter — the single owner-level path that renders and
// emits an lfs::Error. See .codex_tmp/error-architecture-analysis.md Section
// 5.4 ("Propagation and log once ownership") and Section 7.1 row 2 for the
// frozen contract. Summary of the load-bearing invariants:
//   * report() is noexcept and never throws past its own boundary: a
//     failure anywhere in the reporting pipeline (Logger not initialized, a
//     throwing FailureReport section provider, a formatting exception)
//     degrades to a fixed-buffer stderr line instead of propagating or
//     silently dropping the error.
//   * Reporting a failure that occurs while reporting does not recurse: a
//     thread-local guard detects re-entry on the same thread (e.g. a log
//     handler that itself calls report()) and takes the fallback path
//     immediately.
//   * FailureReport is the formatter/sink; ErrorReporter does not implement
//     a second dedup engine. It extends FailureReport's existing
//     family/code/site fingerprint (see failure_report.hpp's
//     deduplication_site) with the Section 5.2 fingerprint definition: code
//     + native code + detection source + top context-frame operation.
//   * No user callback (log handler, FailureReport section provider) runs
//     under a lock owned by ErrorReporter; it does not hold one.
namespace lfs::core {

    // Where a report() call sits in the Section 5.4 propagation chain.
    enum class ReportChannel : std::uint8_t {
        // The normal operation-owner reporting site: renders and dedupes
        // through the standard Logger-mediated FailureReport path, subject
        // to the usual level/module gating like any other log call.
        OwnerLog,
        // A process boundary (main/thread-entry/mode dispatch). In addition
        // to the OwnerLog path, this guarantees one direct stderr line so
        // the failure is visible even if Logger's console sink, level, or
        // per-module filters would otherwise have hidden it.
        ProcessBoundary,
    };

    class LFS_CORE_API ErrorReporter {
    public:
        static ErrorReporter& get();

        // Renders error's context chain and stable fields once, applies the
        // Section 5.4 stack-capture policy, deduplicates by fingerprint, and
        // emits through Logger/FailureReport. Never throws.
        void report(const Error& error, ReportChannel channel) const noexcept;

        ErrorReporter(const ErrorReporter&) = delete;
        ErrorReporter& operator=(const ErrorReporter&) = delete;

    private:
        ErrorReporter() = default;
    };

    // Test-only exposure of the Section 5.4 stack-capture policy matrix
    // (always for ContractViolation/Internal/DeviceLost; optional/debug for
    // NotFound/Cancelled, gated on Logger's active debug level; never for
    // every other anticipated code) so it can be asserted directly instead
    // of by scraping captured stack-trace text.
    [[nodiscard]] LFS_CORE_API bool should_capture_stack_for_testing(ErrorCode code) noexcept;

    // Stable 64-bit dedup key over the Section 5.2 fingerprint dimensions
    // (code + native code + detection source + top context-frame operation).
    // Reuses ErrorReporter's existing fingerprint_site() algorithm rather than
    // duplicating it; ErrorBus keys repeated-fault suppression off this. Equal
    // for two errors that share those dimensions, different otherwise.
    [[nodiscard]] LFS_CORE_API std::uint64_t error_fingerprint(const Error& error) noexcept;

} // namespace lfs::core
