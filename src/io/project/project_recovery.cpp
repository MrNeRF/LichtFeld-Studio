/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_recovery.hpp"

#include "project_container_internal.hpp"
#include "project_recovery_internal.hpp"

#include <algorithm>
#include <concepts>
#include <format>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <type_traits>

namespace lfs::io::project {

    namespace {

        [[nodiscard]] lfs::Error recovery_error(
            const lfs::ErrorCode code,
            const std::filesystem::path& path,
            std::string message, std::string detail,
            const std::string_view field) {
            lfs::SmallFields fields;
            fields.add(
                "path", path.string());
            fields.add("field", field);
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability =
                    lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection =
                    LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        [[nodiscard]] lfs::Result<T> fail(
            const lfs::ErrorCode code,
            const std::filesystem::path& path,
            std::string message, std::string detail,
            const std::string_view field) {
            if constexpr (std::same_as<T, void>) {
                return lfs::Result<void>::failure(
                    recovery_error(
                        code, path, std::move(message),
                        std::move(detail), field));
            } else {
                return recovery_error(
                    code, path, std::move(message),
                    std::move(detail), field);
            }
        }

        [[nodiscard]] bool
        base_echo_matches(
            const ChunkInfo& reference,
            const ChunkInfo& base) noexcept {
            return reference.key == base.key &&
                   reference.row_kind ==
                       RowKind::SidecarBaseReference &&
                   base.row_kind == RowKind::Live &&
                   reference.chunk_version ==
                       base.chunk_version &&
                   reference.compression ==
                       base.compression &&
                   reference.flags == base.flags &&
                   reference.header_offset == 0 &&
                   reference.payload_offset == 0 &&
                   reference.stored_bytes ==
                       base.stored_bytes &&
                   reference.uncompressed_bytes ==
                       base.uncompressed_bytes &&
                   reference.source_generation ==
                       base.source_generation &&
                   reference.payload_crc32c ==
                       base.payload_crc32c &&
                   reference.header_crc32c ==
                       base.header_crc32c;
        }

        [[nodiscard]] lfs::Result<void>
        validate_complete_overlay(
            const ProjectReader& master,
            const ProjectReader& sidecar) {
            if (sidecar.superblock().role !=
                ContainerRole::AutosaveSidecar) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    sidecar.path(),
                    "The recovery candidate is not an autosave sidecar.",
                    "container role must be AUTOSAVE_SIDECAR",
                    "superblock.container_role");
            }
            if (sidecar.superblock().project_uuid !=
                    master.superblock().project_uuid ||
                sidecar.superblock()
                        .base_explicit_commit_uuid !=
                    master.commit().commit_uuid) {
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    sidecar.path(),
                    "The autosave does not bind to the current master.",
                    "project UUID and base explicit commit UUID must "
                    "match the selected master head",
                    "autosave.binding");
            }
            if (auto verified = sidecar.verify_all();
                !verified) {
                return verified;
            }

            std::map<ChunkKey, const ChunkInfo*,
                     ChunkKeyLess>
                base_live;
            for (const auto& row : master.chunks()) {
                if (row.row_kind == RowKind::Live) {
                    base_live.emplace(row.key, &row);
                }
            }
            std::set<ChunkKey, ChunkKeyLess> covered;
            for (const auto& row : sidecar.chunks()) {
                const auto base =
                    base_live.find(row.key);
                if (row.row_kind ==
                    RowKind::SidecarBaseReference) {
                    if (base == base_live.end() ||
                        !base_echo_matches(
                            row, *base->second)) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            sidecar.path(),
                            "The autosave contains an invalid base reference.",
                            std::format(
                                "{} does not exactly echo its bound "
                                "master row",
                                row.key_string()),
                            "autosave.completeness");
                    }
                } else if (
                    row.row_kind ==
                        RowKind::Tombstone &&
                    base == base_live.end()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        sidecar.path(),
                        "The autosave tombstones a missing base key.",
                        row.key_string(),
                        "autosave.completeness");
                }
                if (base != base_live.end()) {
                    covered.insert(row.key);
                } else if (
                    row.row_kind != RowKind::Live) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        sidecar.path(),
                        "The autosave adds a non-live key.",
                        row.key_string(),
                        "autosave.completeness");
                }
            }
            if (covered.size() != base_live.size()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    sidecar.path(),
                    "The autosave overlay is incomplete.",
                    std::format(
                        "{} of {} bound master keys are represented",
                        covered.size(), base_live.size()),
                    "autosave.completeness");
            }
            return {};
        }

        [[nodiscard]] bool starts_and_ends(
            const std::string_view value,
            const std::string_view prefix,
            const std::string_view suffix) {
            return value.starts_with(prefix) &&
                   value.ends_with(suffix);
        }

        [[nodiscard]] std::vector<
            std::filesystem::path>
        sidecar_candidates(
            const std::filesystem::path& master_path) {
            const auto stable =
                autosave_sidecar_path(master_path);
            std::vector<std::filesystem::path> result;
            std::error_code error;
            if (std::filesystem::exists(stable, error) &&
                !error) {
                result.push_back(stable);
            }
            const auto directory =
                stable.parent_path().empty()
                    ? std::filesystem::path{"."}
                    : stable.parent_path();
            const auto stem = stable.stem().string();
            const auto extension =
                stable.extension().string();
            const auto write_prefix =
                stem + ".project-write.";
            const auto backup_prefix =
                stem + ".replace-backup.";
            for (std::filesystem::directory_iterator
                     iterator(directory, error),
                 end;
                 !error && iterator != end;
                 iterator.increment(error)) {
                const auto filename =
                    iterator->path()
                        .filename()
                        .string();
                const auto suffix =
                    ".tmp" + extension;
                if (starts_and_ends(
                        filename, write_prefix,
                        suffix) ||
                    starts_and_ends(
                        filename, backup_prefix,
                        suffix)) {
                    result.push_back(
                        iterator->path());
                }
            }
            std::ranges::sort(result);
            result.erase(
                std::unique(result.begin(),
                            result.end()),
                result.end());
            return result;
        }

        [[nodiscard]] std::vector<
            std::filesystem::path>
        compaction_temps(
            const std::filesystem::path& master_path) {
            std::vector<std::filesystem::path> result;
            const auto directory =
                master_path.parent_path().empty()
                    ? std::filesystem::path{"."}
                    : master_path.parent_path();
            const auto prefix =
                master_path.stem().string() +
                ".compact.";
            const auto backup_prefix =
                master_path.stem().string() +
                ".replace-backup.";
            const auto suffix =
                ".tmp" +
                master_path.extension().string();
            std::error_code error;
            for (std::filesystem::directory_iterator
                     iterator(directory, error),
                 end;
                 !error && iterator != end;
                 iterator.increment(error)) {
                const auto filename =
                    iterator->path()
                        .filename()
                        .string();
                if (starts_and_ends(
                        filename, prefix, suffix) ||
                    starts_and_ends(
                        filename, backup_prefix,
                        suffix)) {
                    result.push_back(
                        iterator->path());
                }
            }
            return result;
        }

        [[nodiscard]] std::vector<
            std::filesystem::path>
        recovery_session_temps(
            const std::filesystem::path& master_path) {
            std::vector<std::filesystem::path> result;
            const auto directory =
                master_path.parent_path().empty()
                    ? std::filesystem::path{"."}
                    : master_path.parent_path();
            const auto prefix =
                master_path.stem().string() +
                ".recovery-session.";
            const auto suffix =
                ".tmp" +
                master_path.extension().string();
            std::error_code error;
            for (std::filesystem::directory_iterator
                     iterator(directory, error),
                 end;
                 !error && iterator != end;
                 iterator.increment(error)) {
                const auto filename =
                    iterator->path()
                        .filename()
                        .string();
                if (starts_and_ends(
                        filename, prefix, suffix)) {
                    result.push_back(
                        iterator->path());
                }
            }
            return result;
        }

        lfs::Result<void> remove_path(
            const std::filesystem::path& path) {
            std::error_code error;
            const bool removed =
                std::filesystem::remove(path, error);
            if (error) {
                return fail<void>(
                    lfs::ErrorCode::PermissionDenied,
                    path,
                    "A stale project artifact could not be removed.",
                    error.message(),
                    "recovery.cleanup");
            }
            (void)removed;
            return {};
        }

    } // namespace

    struct RecoverySession::State {
        State(WriterLockLease lock_in,
              std::filesystem::path master_in,
              std::filesystem::path sidecar_in,
              const lfs::core::Uuid base_in)
            : lock(std::move(lock_in)),
              master(std::move(master_in)),
              sidecar(std::move(sidecar_in)),
              base_commit_uuid(base_in) {}

        ~State() {
            if (!temporary.empty() &&
                !document_attached) {
                std::error_code ignored;
                std::filesystem::remove(
                    temporary, ignored);
            }
        }

        WriterLockLease lock;
        std::filesystem::path master;
        std::filesystem::path sidecar;
        lfs::core::Uuid base_commit_uuid;
        std::filesystem::path temporary;
        bool document_attached = false;
    };

    RecoverySession::RecoverySession() noexcept = default;
    RecoverySession::RecoverySession(
        const RecoverySession&) noexcept = default;
    RecoverySession& RecoverySession::operator=(
        const RecoverySession&) noexcept = default;
    RecoverySession::RecoverySession(
        RecoverySession&&) noexcept = default;
    RecoverySession& RecoverySession::operator=(
        RecoverySession&&) noexcept = default;
    RecoverySession::~RecoverySession() = default;

    RecoverySession::RecoverySession(
        std::shared_ptr<State> state) noexcept
        : state_(std::move(state)) {}

    bool RecoverySession::valid() const noexcept {
        return state_ && state_->lock.valid();
    }

    WriterLockLease
    RecoverySession::writer_lock() const noexcept {
        return state_ ? state_->lock
                      : WriterLockLease{};
    }

    const std::filesystem::path&
    RecoverySession::master_path() const noexcept {
        static const std::filesystem::path empty;
        return state_ ? state_->master : empty;
    }

    void RecoverySession::attach_document() noexcept {
        if (state_) {
            state_->document_attached = true;
        }
    }

    void RecoverySession::detach_document() noexcept {
        if (state_) {
            state_->document_attached = false;
        }
    }

    bool RecoverySession::document_attached() const noexcept {
        return state_ && state_->document_attached;
    }

    lfs::Result<void> RecoverySession::release() {
        if (!state_) {
            return {};
        }
        if (state_->document_attached) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                state_->temporary,
                "The recovered project is still using its staging file.",
                "detach or rebind the live ProjectDocument before releasing recovery",
                "recovery.document");
        }
        lfs::Result<void> cleanup;
        if (!state_->temporary.empty()) {
            cleanup = remove_path(
                state_->temporary);
            if (cleanup) {
                cleanup =
                    detail::sync_parent_directory(
                        state_->temporary);
            }
            state_->temporary.clear();
        }
        state_->lock.release();
        state_.reset();
        return cleanup;
    }

    void RecoverySession::detach_temporary() noexcept {
        if (state_) {
            state_->temporary.clear();
        }
    }

    std::filesystem::path autosave_sidecar_path(
        const std::filesystem::path& master_path) {
        auto result = master_path;
        result += ".autosave";
        return result;
    }

    std::filesystem::path recovery_session_temp_path(
        const std::filesystem::path& master_path) {
        return master_path.parent_path() /
               std::format(
                   "{}.recovery-session.{}.tmp{}",
                   master_path.stem().string(),
                   lfs::core::generate_uuid_v4()
                       .to_string(),
                   master_path.extension().string());
    }

    namespace detail {

        lfs::Result<std::vector<ValidBoundAutosave>>
        valid_bound_autosaves_locked(
            const std::filesystem::path& master_path,
            const ProjectReader& master) {
            std::vector<ValidBoundAutosave> valid;
            ReaderOptions inspection_options;
            inspection_options
                .allow_unsupported_inspection = true;
            for (const auto& candidate :
                 sidecar_candidates(master_path)) {
                auto sidecar = ProjectReader::open(
                    candidate, inspection_options);
                if (!sidecar) {
                    continue;
                }
                if (auto complete =
                        validate_complete_overlay(
                            master, *sidecar);
                    !complete) {
                    continue;
                }
                valid.push_back({
                    .path = candidate,
                    .sequence =
                        sidecar->superblock()
                            .autosave_sequence,
                    .snapshot_uuid =
                        sidecar->superblock()
                            .sidecar_snapshot_uuid,
                });
            }
            return valid;
        }

    } // namespace detail

    lfs::Result<RecoveryInspection>
    inspect_autosave_recovery(
        const std::filesystem::path& master_path,
        const ReaderOptions& master_reader_options) {
        auto lock =
            detail::WriterLock::acquire(master_path);
        if (!lock) {
            return std::move(lock).error();
        }
        auto master = ProjectReader::open(
            master_path, master_reader_options);
        if (!master) {
            return std::move(master).error();
        }

        RecoveryInspection result;
        for (const auto& temp :
             compaction_temps(master_path)) {
            if (auto removed = remove_path(temp);
                !removed) {
                return std::move(removed).error();
            }
            result.deleted_paths.push_back(temp);
        }
        for (const auto& temp :
             recovery_session_temps(master_path)) {
            if (auto removed = remove_path(temp);
                !removed) {
                return std::move(removed).error();
            }
            result.deleted_paths.push_back(temp);
        }

        struct Valid {
            std::filesystem::path path;
            std::uint64_t sequence = 0;
            lfs::core::Uuid snapshot_uuid;
        };
        std::vector<Valid> valid;
        const auto stable =
            autosave_sidecar_path(master_path);
        ReaderOptions inspection_options;
        inspection_options
            .allow_unsupported_inspection = true;
        for (const auto& candidate :
             sidecar_candidates(master_path)) {
            auto sidecar =
                ProjectReader::open(
                    candidate,
                    inspection_options);
            if (!sidecar) {
                result.diagnostics.push_back(
                    std::format(
                        "{}: {}",
                        candidate.filename().string(),
                        lfs::format_for_developer(
                            sidecar.error())));
                if (candidate != stable) {
                    if (auto removed =
                            remove_path(candidate);
                        !removed) {
                        return std::move(removed)
                            .error();
                    }
                    result.deleted_paths.push_back(
                        candidate);
                }
                continue;
            }
            const bool same_project =
                sidecar->superblock().project_uuid ==
                master->superblock().project_uuid;
            const bool same_base =
                sidecar->superblock()
                    .base_explicit_commit_uuid ==
                master->commit().commit_uuid;
            if (same_project && !same_base) {
                if (auto removed =
                        remove_path(candidate);
                    !removed) {
                    return std::move(removed).error();
                }
                result.deleted_paths.push_back(
                    candidate);
                continue;
            }
            auto complete =
                validate_complete_overlay(
                    *master, *sidecar);
            if (!complete) {
                result.diagnostics.push_back(
                    std::format(
                        "{}: {}",
                        candidate.filename().string(),
                        lfs::format_for_developer(
                            complete.error())));
                if (candidate != stable) {
                    if (auto removed =
                            remove_path(candidate);
                        !removed) {
                        return std::move(removed)
                            .error();
                    }
                    result.deleted_paths.push_back(
                        candidate);
                }
                continue;
            }
            valid.push_back(Valid{
                .path = candidate,
                .sequence =
                    sidecar->superblock()
                        .autosave_sequence,
                .snapshot_uuid =
                    sidecar->superblock()
                        .sidecar_snapshot_uuid,
            });
        }

        if (!result.deleted_paths.empty()) {
            if (auto synced =
                    detail::sync_parent_directory(
                        master_path);
                !synced) {
                return std::move(synced).error();
            }
        }

        if (valid.empty()) {
            if (!result.deleted_paths.empty()) {
                result.disposition =
                    RecoveryDisposition::
                        StaleDeleted;
            } else if (
                !result.diagnostics.empty()) {
                result.disposition =
                    RecoveryDisposition::Invalid;
            }
            return result;
        }
        const auto highest =
            std::ranges::max_element(
                valid, {}, &Valid::sequence)
                ->sequence;
        std::vector<const Valid*> winners;
        for (const auto& candidate : valid) {
            if (candidate.sequence == highest) {
                winners.push_back(&candidate);
            }
        }
        if (winners.size() != 1) {
            result.disposition =
                RecoveryDisposition::Ambiguous;
            result.diagnostics.push_back(
                std::format(
                    "{} valid autosaves tie at sequence {}",
                    winners.size(), highest));
            return result;
        }
        result.disposition =
            RecoveryDisposition::Offer;
        result.selected_path = winners.front()->path;
        result.autosave_sequence = highest;
        result.snapshot_uuid =
            winners.front()->snapshot_uuid;
        return result;
    }

    lfs::Result<void> remove_autosave_artifacts(
        const std::filesystem::path& master_path) {
        auto lock =
            detail::WriterLock::acquire(master_path);
        if (!lock) {
            return lfs::Result<void>::failure(
                std::move(lock).error());
        }
        for (const auto& candidate :
             sidecar_candidates(master_path)) {
            if (auto removed =
                    remove_path(candidate);
                !removed) {
                return removed;
            }
        }
        return detail::sync_parent_directory(
            master_path);
    }

    lfs::Result<RecoverySession>
    begin_recovery_session(
        const std::filesystem::path& master_path,
        const std::filesystem::path& sidecar_path) {
        auto lock =
            WriterLockLease::acquire(master_path);
        if (!lock) {
            return std::move(lock).error();
        }
        auto master =
            ProjectReader::open(master_path);
        if (!master) {
            return std::move(master).error();
        }
        auto sidecar =
            ProjectReader::open(sidecar_path);
        if (!sidecar) {
            return std::move(sidecar).error();
        }
        if (auto valid = validate_complete_overlay(
                *master, *sidecar);
            !valid) {
            return std::move(valid).error();
        }
        return RecoverySession(
            std::make_shared<RecoverySession::State>(
                std::move(*lock),
                master_path.lexically_normal(),
                sidecar_path.lexically_normal(),
                master->commit().commit_uuid));
    }

    lfs::Result<void>
    materialize_recovered_project(
        const std::filesystem::path& master_path,
        const std::filesystem::path& sidecar_path,
        const std::filesystem::path& destination,
        CommitBoundaryObserver boundary_observer) {
        auto session = begin_recovery_session(
            master_path, sidecar_path);
        if (!session) {
            return lfs::Result<void>::failure(
                std::move(session).error());
        }
        auto materialized =
            materialize_recovered_project(
                master_path, sidecar_path,
                destination, *session,
                std::move(boundary_observer));
        if (materialized) {
            session->detach_temporary();
        }
        return materialized;
    }

    lfs::Result<void>
    materialize_recovered_project(
        const std::filesystem::path& master_path,
        const std::filesystem::path& sidecar_path,
        const std::filesystem::path& destination,
        const RecoverySession& session,
        CommitBoundaryObserver boundary_observer) {
        const auto normalized_master =
            master_path.lexically_normal();
        const auto normalized_sidecar =
            sidecar_path.lexically_normal();
        const auto normalized_destination =
            destination.lexically_normal();
        if (normalized_destination ==
                normalized_master ||
            normalized_destination ==
                normalized_sidecar) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                destination,
                "Recovery requires a separate staging destination.",
                "materialization must not overwrite the master or "
                "selected autosave sidecar",
                "recovery.destination");
        }
        if (!session.state_ ||
            !session.state_->lock.valid() ||
            session.state_->master !=
                normalized_master ||
            session.state_->sidecar !=
                normalized_sidecar) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                master_path,
                "The recovery session no longer holds the selected master.",
                "materialization requires the retained master writer lock and selected sidecar",
                "recovery.writer_lock");
        }
        if (!session.state_->temporary.empty()) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                destination,
                "This recovery session already owns a staging project.",
                session.state_->temporary.string(),
                "recovery.destination");
        }
        auto master =
            ProjectReader::open(master_path);
        if (!master) {
            return lfs::Result<void>::failure(
                std::move(master).error());
        }
        if (master->commit().commit_uuid !=
            session.state_->base_commit_uuid) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                master_path,
                "The recovery base changed while the session was held.",
                "the current master head no longer equals the recovery session base",
                "recovery.base_commit_uuid");
        }
        auto sidecar =
            ProjectReader::open(sidecar_path);
        if (!sidecar) {
            return lfs::Result<void>::failure(
                std::move(sidecar).error());
        }
        if (auto valid =
                validate_complete_overlay(
                    *master, *sidecar);
            !valid) {
            return valid;
        }
        session.state_->temporary = destination;
        auto writer = ProjectWriter::create(
            destination,
            CreateOptions{
                .project_uuid =
                    master->superblock()
                        .project_uuid,
                .file_uuid =
                    lfs::core::generate_uuid_v4(),
                .role = ContainerRole::Master,
                .base_explicit_commit_uuid = {},
                .autosave_sequence = 0,
                .sidecar_snapshot_uuid = {},
                .creation_time_unix_ns =
                    master->superblock()
                        .creation_time_unix_ns,
                .index_compression =
                    IndexCompression::Zstd,
                .disk_reserve_bytes =
                    64ull * 1024 * 1024,
                .boundary_observer =
                    std::move(boundary_observer),
                .writer_lock_anchor =
                    std::nullopt,
            });
        if (!writer) {
            return lfs::Result<void>::failure(
                std::move(writer).error());
        }
        if (auto planned = writer->plan_commit(
                CommitOptions{
                    .kind =
                        CommitKind::Recovered,
                    .commit_uuid =
                        lfs::core::
                            generate_uuid_v4(),
                    .snapshot_uuid =
                        sidecar->superblock()
                            .sidecar_snapshot_uuid,
                    .wallclock_unix_ns = 0,
                    .min_reader_version =
                        master->commit()
                            .min_reader_version,
                    .min_safe_writer_version =
                        master->commit()
                            .min_safe_writer_version,
                    .extra_reader_capabilities =
                        master->commit()
                            .required_reader_capabilities,
                    .extra_writer_capabilities =
                        master->commit()
                            .required_writer_capabilities,
                });
            !planned) {
            return planned;
        }
        std::uint64_t planned_bytes = 0;
        for (const auto& row :
             sidecar->chunks()) {
            const ChunkInfo* source = nullptr;
            if (row.row_kind == RowKind::Live) {
                source = &row;
            } else if (
                row.row_kind ==
                RowKind::SidecarBaseReference) {
                source = master->find(row.key);
            }
            if (source &&
                source->stored_bytes <=
                    std::numeric_limits<
                        std::uint64_t>::max() -
                        planned_bytes) {
                planned_bytes +=
                    source->stored_bytes;
            }
        }
        if (auto preflight =
                writer->preflight(planned_bytes);
            !preflight) {
            return preflight;
        }
        for (const auto& row :
             sidecar->chunks()) {
            if (row.row_kind == RowKind::Tombstone) {
                continue;
            }
            const ProjectReader* source_reader =
                &*sidecar;
            const ChunkInfo* source_row = &row;
            if (row.row_kind ==
                RowKind::SidecarBaseReference) {
                source_reader = &*master;
                source_row = master->find(row.key);
            }
            if (!source_row) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    sidecar_path,
                    "The recovery overlay lost a base row.",
                    row.key_string(),
                    "autosave.completeness");
            }
            if (auto copied =
                    writer->copy_chunk_verbatim(
                        *source_reader,
                        *source_row);
                !copied) {
                return copied;
            }
        }
        if (auto committed = writer->commit();
            !committed) {
            return committed;
        }
        auto recovered =
            ProjectReader::open(destination);
        if (!recovered) {
            return lfs::Result<void>::failure(
                std::move(recovered).error());
        }
        return recovered->verify_all();
    }

    lfs::Result<ProjectStorageStats>
    project_storage_stats(
        const std::filesystem::path& master_path) {
        auto reader =
            ProjectReader::open(master_path);
        if (!reader) {
            return std::move(reader).error();
        }
        ProjectStorageStats result;
        result.physical_bytes =
            reader->physical_file_size();
        std::uint64_t live =
            APPEND_REGION_OFFSET +
            COMMIT_RECORD_BYTES +
            reader->commit().index_stored_bytes;
        for (const auto& row : reader->chunks()) {
            if (row.row_kind != RowKind::Live) {
                continue;
            }
            const std::uint64_t occupied =
                (row.payload_offset -
                 row.header_offset) +
                row.stored_bytes;
            std::uint64_t trailing = 0;
            if ((row.flags & TENSOR_PAYLOAD) != 0) {
                const auto remainder =
                    (row.payload_offset +
                     row.stored_bytes) %
                    TENSOR_PAYLOAD_ALIGNMENT;
                trailing =
                    remainder == 0
                        ? 0
                        : TENSOR_PAYLOAD_ALIGNMENT -
                              remainder;
            }
            std::uint64_t block_table_bytes = 0;
            if (row.block_crc_table) {
                const auto entry_count =
                    row.block_crc_table->entries.size();
                if (entry_count >
                    (std::numeric_limits<
                         std::uint64_t>::max() -
                     BLOCK_CRC_HEADER_BYTES) /
                        sizeof(std::uint32_t)) {
                    return fail<ProjectStorageStats>(
                        lfs::ErrorCode::ResourceExhausted,
                        master_path,
                        "The project block-CRC estimate overflowed.",
                        row.key_string(),
                        "storage.block_crc_bytes");
                }
                block_table_bytes =
                    BLOCK_CRC_HEADER_BYTES +
                    static_cast<std::uint64_t>(
                        entry_count) *
                        sizeof(std::uint32_t);
            }
            if (occupied >
                    std::numeric_limits<
                        std::uint64_t>::max() -
                        trailing ||
                occupied + trailing >
                    std::numeric_limits<
                        std::uint64_t>::max() -
                        block_table_bytes ||
                occupied + trailing +
                        block_table_bytes >
                    std::numeric_limits<
                        std::uint64_t>::max() -
                        live) {
                return fail<ProjectStorageStats>(
                    lfs::ErrorCode::ResourceExhausted,
                    master_path,
                    "The project live-byte estimate overflowed.",
                    row.key_string(),
                    "storage.live_bytes");
            }
            live += occupied + trailing +
                    block_table_bytes;
        }
        result.estimated_live_bytes =
            std::min(
                live, result.physical_bytes);
        result.dead_bytes =
            result.physical_bytes -
            result.estimated_live_bytes;
        if (result.physical_bytes != 0) {
            result.dead_ratio =
                static_cast<double>(
                    result.dead_bytes) /
                static_cast<double>(
                    result.physical_bytes);
        }
        return result;
    }

} // namespace lfs::io::project
