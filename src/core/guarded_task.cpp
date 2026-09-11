/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/guarded_task.hpp"

#include "core/error_reporter.hpp"

namespace lfs::core::detail {

    Error normalize_current_exception(const TaskContext& context) noexcept {
        try {
            throw;
        } catch (const lfs::Exception& e) {
            return lfs::Error(e.error()).with_context(context.name, context.site);
        } catch (const std::exception& e) {
            return lfs::make_error(lfs::ErrorInit{
                                       .code = ErrorCode::Internal,
                                       .domain = context.domain,
                                       .operation_id = context.operation_id,
                                       .detail = e.what(),
                                       .detection = context.site,
                                   })
                .with_context(context.name, context.site);
        } catch (...) {
            return lfs::make_error(lfs::ErrorInit{
                                       .code = ErrorCode::Internal,
                                       .domain = context.domain,
                                       .operation_id = context.operation_id,
                                       .detail = "unknown exception",
                                       .detection = context.site,
                                   })
                .with_context(context.name, context.site);
        }
    }

    void report_completion_violation(const TaskContext& context) noexcept {
        const Error violation = lfs::make_error(lfs::ErrorInit{
                                                    .code = ErrorCode::Internal,
                                                    .domain = context.domain,
                                                    .operation_id = context.operation_id,
                                                    .detail = "TaskCompletion sink threw during settlement",
                                                    .detection = context.site,
                                                })
                                    .with_context(context.name, context.site);
        ErrorReporter::get().report(violation, ReportChannel::ProcessBoundary);
    }

} // namespace lfs::core::detail
