/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/logger.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace lfs::app::detail {

    // Owns the headless ProjectDocument together with the recovery attachment
    // that protects its lazy source file. The trainer receives only a shared
    // RecoverySession copy and cannot detach a document it does not own.
    class HeadlessRecoveryDocument {
    public:
        HeadlessRecoveryDocument(
            lfs::io::project::ProjectDocument document,
            std::optional<lfs::io::project::RecoverySession>
                recovery_session)
            : document_(std::make_unique<
                        lfs::io::project::ProjectDocument>(
                  std::move(document))),
              recovery_session_(
                  std::move(recovery_session)) {
            if (recovery_session_) {
                recovery_session_->attach_document();
            }
        }

        HeadlessRecoveryDocument(
            const HeadlessRecoveryDocument&) = delete;
        HeadlessRecoveryDocument& operator=(
            const HeadlessRecoveryDocument&) = delete;

        HeadlessRecoveryDocument(
            HeadlessRecoveryDocument&& other) noexcept
            : document_(std::move(other.document_)),
              recovery_session_(
                  std::move(other.recovery_session_)) {}

        HeadlessRecoveryDocument& operator=(
            HeadlessRecoveryDocument&& other) noexcept {
            if (this != &other) {
                teardown();
                document_ = std::move(other.document_);
                recovery_session_ =
                    std::move(other.recovery_session_);
            }
            return *this;
        }

        ~HeadlessRecoveryDocument() {
            teardown();
        }

        [[nodiscard]] lfs::io::project::ProjectDocument&
        document() noexcept {
            return *document_;
        }

        [[nodiscard]] const lfs::io::project::ProjectDocument&
        document() const noexcept {
            return *document_;
        }

        [[nodiscard]] const lfs::io::project::RecoverySession*
        recovery_session() const noexcept {
            return recovery_session_
                       ? &*recovery_session_
                       : nullptr;
        }

        // Call only after the trainer has durably published the recovered
        // generation. Reopening first transfers every lazy row to the master;
        // only then may the staging attachment be detached and released.
        [[nodiscard]] lfs::Result<void>
        rebind_after_durable_merge() {
            if (!recovery_session_) {
                return {};
            }
            auto rebound =
                lfs::io::project::ProjectDocument::open(
                    recovery_session_->master_path(),
                    lfs::io::project::
                        ProjectDocumentOpenOptions{
                            .reader = {},
                            .geometry = {},
                            .defer_geometry_payloads = true,
                        });
            if (!rebound) {
                return lfs::Status::failure(
                    std::move(rebound)
                        .error()
                        .with_context(
                            "rebind headless recovered document to durable master",
                            LFS_SOURCE_SITE_CURRENT()));
            }

            document_ = std::make_unique<
                lfs::io::project::ProjectDocument>(
                std::move(*rebound));
            recovery_session_->detach_document();
            auto released = recovery_session_->release();
            recovery_session_.reset();
            if (!released) {
                // The master is durable and the document is already rebound.
                // Retain cleanup failure as a warning, matching the explicit
                // save post-publish contract.
                LOG_WARN(
                    "Headless recovery merge is durable, but staging cleanup failed: {}",
                    lfs::format_for_developer(
                        released.error()));
            }
            return {};
        }

    private:
        void teardown() noexcept {
            // ProjectDocument destruction must precede detach: its deferred
            // readers may still refer to the recovery staging file.
            document_.reset();
            if (!recovery_session_) {
                return;
            }
            recovery_session_->detach_document();
            if (auto released =
                    recovery_session_->release();
                !released) {
                LOG_WARN(
                    "Could not clean up a headless recovered-session staging file after document teardown: {}",
                    lfs::format_for_developer(
                        released.error()));
            }
            recovery_session_.reset();
        }

        std::unique_ptr<
            lfs::io::project::ProjectDocument>
            document_;
        std::optional<
            lfs::io::project::RecoverySession>
            recovery_session_;
    };

} // namespace lfs::app::detail
