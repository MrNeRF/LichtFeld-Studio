/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "app/application_startup.hpp"

#include "core/parameters.hpp"

namespace lfs::app {

    lfs::Result<void> validate_application_startup(
        const lfs::core::param::TrainingParameters& params) {
        if (!params.optimization.headless &&
            params.resume_project) {
            return lfs::Status::failure(
                lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Unsupported,
                    .domain = lfs::ErrorDomain::App,
                    .severity = lfs::Severity::Error,
                    .retryability =
                        lfs::Retryability::NotRetryable,
                    .user_message =
                        "This .licht project cannot be resumed in the GUI yet.",
                    .detail =
                        "Project resume in GUI arrives with the lifecycle phase; use --headless for .licht resume today.",
                    .detection =
                        LFS_SOURCE_SITE_CURRENT(),
                }));
        }
        return {};
    }

} // namespace lfs::app
