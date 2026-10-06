// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/error_bus.hpp"
#include "core/failure_report.hpp"
#ifdef _WIN32
#define BRIDGE_API __declspec(dllexport)
#else
#define BRIDGE_API __attribute__((visibility("default")))
#endif
extern "C" BRIDGE_API void mediaTestRegisterWriter(lfs::core::FailureReportWriter writer) {
    lfs::core::register_failure_report_writer(writer);
}
extern "C" BRIDGE_API bool mediaTestDedup(std::uint64_t& count) {
    return lfs::core::decide_failure_report_for_testing("media-shared-state", 1, "bridge", count);
}
extern "C" BRIDGE_API const void* mediaTestErrorBus() {
    return &lfs::ErrorBus::instance();
}
