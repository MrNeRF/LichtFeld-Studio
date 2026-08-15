/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "io/project_container.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::io::project {

    enum class RecoveryDisposition {
        None,
        Offer,
        StaleDeleted,
        Invalid,
        Ambiguous,
    };

    struct RecoveryInspection {
        RecoveryDisposition disposition =
            RecoveryDisposition::None;
        std::optional<std::filesystem::path>
            selected_path;
        std::uint64_t autosave_sequence = 0;
        lfs::core::Uuid snapshot_uuid;
        std::vector<std::filesystem::path>
            deleted_paths;
        std::vector<std::string> diagnostics;
    };

    struct ProjectStorageStats {
        std::uint64_t physical_bytes = 0;
        std::uint64_t estimated_live_bytes = 0;
        std::uint64_t dead_bytes = 0;
        double dead_ratio = 0.0;
    };

    // Non-nil identities and a non-zero clock make recovery publication
    // reproducible for release-corpus generation. Production callers use the
    // defaults and receive generated identities/current time as before.
    struct RecoveryMaterializationOptions {
        lfs::core::Uuid file_uuid = {};
        lfs::core::Uuid commit_uuid = {};
        std::uint64_t wallclock_unix_ns = 0;
        IndexCompression index_compression =
            IndexCompression::Zstd;
        std::uint64_t disk_reserve_bytes =
            64ull * 1024 * 1024;
    };

    class LFS_IO_API RecoverySession {
    public:
        RecoverySession() noexcept;
        RecoverySession(const RecoverySession&) noexcept;
        RecoverySession& operator=(const RecoverySession&) noexcept;
        RecoverySession(RecoverySession&&) noexcept;
        RecoverySession& operator=(RecoverySession&&) noexcept;
        ~RecoverySession();

        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] WriterLockLease writer_lock() const noexcept;
        [[nodiscard]] const std::filesystem::path&
        master_path() const noexcept;
        // The recovered ProjectDocument lazy-reads from the staging file.
        // Detach it only after rebinding or replacing that document.
        void attach_document() noexcept;
        void detach_document() noexcept;
        [[nodiscard]] bool document_attached() const noexcept;
        // Deletes the staging file, then requests lease release. The flock
        // drops after remaining WriterLockLease copies are destroyed
        // (d1dac3cfa).
        [[nodiscard]] lfs::Result<void> release();

    private:
        struct State;
        explicit RecoverySession(
            std::shared_ptr<State> state) noexcept;
        void detach_temporary() noexcept;
        std::shared_ptr<State> state_;

        friend lfs::Result<RecoverySession>
        begin_recovery_session(
            const std::filesystem::path&,
            const std::filesystem::path&);
        friend lfs::Result<void>
        materialize_recovered_project(
            const std::filesystem::path&,
            const std::filesystem::path&,
            const std::filesystem::path&,
            CommitBoundaryObserver,
            const RecoveryMaterializationOptions&);
        friend lfs::Result<void>
        materialize_recovered_project(
            const std::filesystem::path&,
            const std::filesystem::path&,
            const std::filesystem::path&,
            const RecoverySession&,
            CommitBoundaryObserver,
            const RecoveryMaterializationOptions&);
    };

    [[nodiscard]] LFS_IO_API std::filesystem::path
    autosave_sidecar_path(
        const std::filesystem::path& master_path);

    [[nodiscard]] LFS_IO_API std::filesystem::path
    recovery_session_temp_path(
        const std::filesystem::path& master_path);

    // Renames `source` to `source.corrupt-<unix_seconds>` (with -N
    // disambiguation). Bytes are preserved. The caller must already hold
    // the master writer lock when `source` is a sidecar/temp of that master.
    [[nodiscard]] LFS_IO_API
        lfs::Result<std::filesystem::path>
        quarantine_project_artifact(
            const std::filesystem::path& source);

    // Caller must already hold the master writer lock. Flock-free
    // write/compact/replace-backup temps are deleted only when the
    // published master file exists at `master_path` (it supersedes
    // them). If that master is absent they are quarantined via
    // `quarantine_project_artifact` so a crashed first publish is not
    // `remove()`d. Prunes older `.corrupt-*` asides of this master
    // (newest 3 per stem), including asides created by this sweep.
    // Does not touch `.lock`, `.autosave`, or recovery-session temps.
    LFS_IO_API void sweep_orphan_project_artifacts(
        const std::filesystem::path& master_path,
        RecoveryInspection& into);

    // Acquires the master writer lock, validates every stable/temp/backup
    // sidecar candidate, deletes stale candidates, and applies the exact
    // §9 predicate. It also removes orphan compaction temps after the master
    // authority has opened successfully.
    [[nodiscard]] LFS_IO_API
        lfs::Result<RecoveryInspection>
        inspect_autosave_recovery(
            const std::filesystem::path& master_path,
            const ReaderOptions& master_reader_options = {});

    // Acquires and retains the original master's OS writer lock, then
    // revalidates the selected complete bound sidecar under that lock.
    [[nodiscard]] LFS_IO_API
        lfs::Result<RecoverySession>
        begin_recovery_session(
            const std::filesystem::path& master_path,
            const std::filesystem::path& sidecar_path);

    // Called only after a durable explicit master head. Stable and
    // replacement candidates are removed while holding the master lock.
    [[nodiscard]] LFS_IO_API lfs::Result<void>
    remove_autosave_artifacts(
        const std::filesystem::path& master_path);

    // Materializes one validated overlay into a separate complete master.
    // The caller keeps the original master untouched until the next explicit
    // save publishes the recovered document.
    [[nodiscard]] LFS_IO_API lfs::Result<void>
    materialize_recovered_project(
        const std::filesystem::path& master_path,
        const std::filesystem::path& sidecar_path,
        const std::filesystem::path& destination,
        CommitBoundaryObserver boundary_observer = {},
        const RecoveryMaterializationOptions& options = {});

    [[nodiscard]] LFS_IO_API lfs::Result<void>
    materialize_recovered_project(
        const std::filesystem::path& master_path,
        const std::filesystem::path& sidecar_path,
        const std::filesystem::path& destination,
        const RecoverySession& session,
        CommitBoundaryObserver boundary_observer = {},
        const RecoveryMaterializationOptions& options = {});

    [[nodiscard]] LFS_IO_API
        lfs::Result<ProjectStorageStats>
        project_storage_stats(const ProjectReader& reader);
    [[nodiscard]] LFS_IO_API
        lfs::Result<ProjectStorageStats>
        project_storage_stats(
            const std::filesystem::path& master_path);

} // namespace lfs::io::project
