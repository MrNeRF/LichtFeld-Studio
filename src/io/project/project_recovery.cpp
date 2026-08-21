/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_recovery.hpp"

#include "core/logger.hpp"
#include "core/uuid.hpp"
#include "io/project_chapters.hpp"
#include "project_container_internal.hpp"
#include "project_recovery_internal.hpp"

#include <algorithm>
#include <chrono>
#include <concepts>
#include <format>
#include <limits>
#include <map>
#include <optional>
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

        [[nodiscard]] bool is_scratch_payload_fourcc(
            const Fourcc fourcc) {
            return fourcc == FOURCC_SPLT ||
                   fourcc == FOURCC_PCLD ||
                   fourcc == FOURCC_MESH ||
                   fourcc == FOURCC_CKPT ||
                   fourcc == FOURCC_PPIS;
        }

        // Same content bar as New Project's blank untitled
        // session: no scene nodes and no payload chapters.
        [[nodiscard]] bool
        scratch_has_recoverable_content(
            const ProjectReader& reader) {
            for (const auto& chunk : reader.chunks()) {
                if (!chunk.is_live()) {
                    continue;
                }
                if (is_scratch_payload_fourcc(
                        chunk.key.fourcc)) {
                    return true;
                }
                if (chunk.key.fourcc != FOURCC_SCNG) {
                    continue;
                }
                auto bytes = reader.read_chunk(chunk);
                if (!bytes) {
                    continue;
                }
                auto chapter =
                    SceneGraphChapter::from_bytes(
                        *bytes);
                if (!chapter) {
                    continue;
                }
                auto nodes = chapter->nodes();
                if (nodes && !nodes->empty()) {
                    return true;
                }
            }
            return false;
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

        void record_remove_failure(
            RecoveryInspection& into,
            const std::filesystem::path& path,
            const lfs::Error& error) {
            into.diagnostics.push_back(
                std::format(
                    "{}: {}",
                    path.filename().string(),
                    lfs::format_for_developer(error)));
        }

        void best_effort_remove_into(
            RecoveryInspection& into,
            const std::filesystem::path& path) {
            if (auto removed = remove_path(path);
                !removed) {
                record_remove_failure(
                    into, path, removed.error());
                return;
            }
            into.deleted_paths.push_back(path);
        }

        void quarantine_into(
            RecoveryInspection& into,
            const std::filesystem::path& source,
            const std::string& reason) {
            auto aside =
                quarantine_project_artifact(source);
            if (!aside) {
                into.diagnostics.push_back(
                    std::format(
                        "{}: {} (quarantine failed: {})",
                        source.filename().string(),
                        reason,
                        lfs::format_for_developer(
                            aside.error())));
                return;
            }
            into.deleted_paths.push_back(source);
            into.diagnostics.push_back(
                std::format(
                    "{}: {} (quarantined to {})",
                    source.filename().string(),
                    reason,
                    aside->filename().string()));
        }

        [[nodiscard]] bool is_write_temp_name(
            const std::string_view filename,
            const std::string_view stem,
            const std::string_view suffix) {
            return starts_and_ends(
                       filename,
                       std::string(stem) +
                           ".project-write.",
                       suffix) ||
                   starts_and_ends(
                       filename,
                       std::string(stem) + ".compact.",
                       suffix) ||
                   starts_and_ends(
                       filename,
                       std::string(stem) +
                           ".replace-backup.",
                       suffix);
        }

        [[nodiscard]] std::optional<std::string>
        referenced_saveas_master_name(
            const std::string_view filename) {
            if (!filename.starts_with('.')) {
                return std::nullopt;
            }
            constexpr std::string_view marker =
                ".saveas-";
            const auto marker_at =
                filename.rfind(marker);
            if (marker_at == std::string_view::npos ||
                marker_at < 2) {
                return std::nullopt;
            }
            return std::string(
                filename.substr(1, marker_at - 1));
        }

        [[nodiscard]] std::optional<std::string>
        referenced_write_temp_master_name(
            const std::string_view filename) {
            constexpr std::string_view tags[] = {
                ".project-write.",
                ".compact.",
                ".replace-backup.",
            };
            auto tag_at = std::string_view::npos;
            std::size_t tag_size = 0;
            for (const auto tag : tags) {
                const auto found = filename.find(tag);
                if (found != std::string_view::npos &&
                    found > 0 &&
                    found < tag_at) {
                    tag_at = found;
                    tag_size = tag.size();
                }
            }
            if (tag_at == std::string_view::npos) {
                return std::nullopt;
            }
            const auto after =
                filename.substr(tag_at + tag_size);
            constexpr std::string_view tmp_marker =
                ".tmp";
            const auto tmp_at =
                after.rfind(tmp_marker);
            if (tmp_at == std::string_view::npos) {
                return std::nullopt;
            }
            const auto extension = after.substr(
                tmp_at + tmp_marker.size());
            if (extension.empty() ||
                !extension.starts_with('.')) {
                return std::nullopt;
            }
            const auto stem =
                filename.substr(0, tag_at);
            const auto suffix =
                std::string(tmp_marker) +
                std::string(extension);
            if (!is_write_temp_name(
                    filename, stem, suffix)) {
                return std::nullopt;
            }
            return std::string(stem) +
                   std::string(extension);
        }

        [[nodiscard]] bool master_file_is_absent(
            const std::filesystem::path& directory,
            const std::string_view master_name) {
            if (master_name.empty() ||
                master_name == "." ||
                master_name == ".." ||
                master_name.find('/') !=
                    std::string_view::npos ||
                master_name.find('\\') !=
                    std::string_view::npos) {
                return false;
            }
            std::error_code error;
            const bool present =
                std::filesystem::exists(
                    directory /
                        std::string(master_name),
                    error);
            return !error && !present;
        }

        [[nodiscard]] bool is_master_corrupt_aside(
            const std::string_view filename,
            const std::filesystem::path& master_path) {
            if (filename.find(".corrupt-") ==
                std::string_view::npos) {
                return false;
            }
            const auto stem =
                master_path.stem().string();
            const auto published =
                master_path.filename().string();
            return filename.starts_with(published) ||
                   filename.starts_with(stem);
        }

        [[nodiscard]] std::uint64_t corrupt_stamp(
            const std::string_view filename) {
            const auto marker =
                filename.find(".corrupt-");
            if (marker == std::string_view::npos) {
                return 0;
            }
            const auto rest =
                filename.substr(marker + 9);
            std::uint64_t stamp = 0;
            std::size_t index = 0;
            while (index < rest.size() &&
                   rest[index] >= '0' &&
                   rest[index] <= '9') {
                const auto next =
                    stamp * 10ull +
                    static_cast<std::uint64_t>(
                        rest[index] - '0');
                if (next < stamp) {
                    return stamp;
                }
                stamp = next;
                ++index;
            }
            return stamp;
        }

        [[nodiscard]] std::string corrupt_group_key(
            const std::string_view filename) {
            const auto marker =
                filename.find(".corrupt-");
            if (marker == std::string_view::npos) {
                return std::string(filename);
            }
            return std::string(filename.substr(0, marker));
        }

        void try_reclaim_write_temp(
            RecoveryInspection& into,
            const std::filesystem::path& temp_path,
            const bool published_master_exists) {
            {
                auto lease =
                    WriterLockLease::acquire(
                        temp_path);
                if (!lease) {
                    if (lease.error().code() ==
                        lfs::ErrorCode::Unavailable) {
                        return;
                    }
                    record_remove_failure(
                        into, temp_path,
                        lease.error());
                    return;
                }
            }
            if (published_master_exists) {
                best_effort_remove_into(
                    into, temp_path);
                return;
            }
            quarantine_into(
                into,
                temp_path,
                "unpublished write temp preserved because the published master is absent");
        }

        void prune_corrupt_asides(
            RecoveryInspection& into,
            std::vector<std::filesystem::path> asides) {
            struct Ranked {
                std::filesystem::path path;
                std::uint64_t stamp = 0;
                std::filesystem::file_time_type mtime{};
            };
            std::map<std::string, std::vector<Ranked>>
                groups;
            for (auto& aside : asides) {
                const auto name =
                    aside.filename().string();
                std::error_code error;
                auto mtime =
                    std::filesystem::last_write_time(
                        aside, error);
                if (error) {
                    mtime = {};
                }
                groups[corrupt_group_key(name)].push_back(
                    Ranked{
                        .path = std::move(aside),
                        .stamp = corrupt_stamp(name),
                        .mtime = mtime,
                    });
            }
            for (auto& [_, group] : groups) {
                std::ranges::sort(
                    group,
                    [](const Ranked& lhs,
                       const Ranked& rhs) {
                        if (lhs.stamp != rhs.stamp) {
                            return lhs.stamp > rhs.stamp;
                        }
                        if (lhs.mtime != rhs.mtime) {
                            return lhs.mtime > rhs.mtime;
                        }
                        return lhs.path.generic_string() >
                               rhs.path.generic_string();
                    });
                for (std::size_t index = 3;
                     index < group.size();
                     ++index) {
                    best_effort_remove_into(
                        into, group[index].path);
                }
            }
        }

        [[nodiscard]] std::filesystem::path
        path_without_lock_suffix(
            const std::filesystem::path& path) {
            const auto name =
                path.filename().string();
            constexpr std::string_view suffix =
                ".lock";
            if (name.size() > suffix.size() &&
                name.ends_with(suffix)) {
                return path.parent_path() /
                       name.substr(
                           0,
                           name.size() -
                               suffix.size());
            }
            return path;
        }

        void try_remove_unheld_artifact(
            const std::filesystem::path& data_path,
            const bool remove_data) {
            auto lock_path = data_path;
            lock_path += ".lock";
            std::error_code exists_error;
            const bool lock_existed =
                std::filesystem::exists(
                    lock_path, exists_error) &&
                !exists_error;
            {
                auto lock =
                    detail::WriterLock::acquire(
                        data_path);
                if (!lock) {
                    return;
                }
                if (remove_data) {
                    std::error_code error;
                    if (std::filesystem::remove(
                            data_path, error) &&
                        !error) {
                        LOG_INFO(
                            "Removed stale project artifact {}",
                            data_path.string());
                    }
                }
            }
            if (!lock_existed) {
                return;
            }
            exists_error.clear();
            if (std::filesystem::exists(
                    lock_path, exists_error) &&
                !exists_error) {
                return;
            }
            LOG_INFO(
                "Removed stale project artifact {}",
                lock_path.string());
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

    std::filesystem::path scratch_autosave_path(
        const std::filesystem::path& recovery_directory,
        const lfs::core::Uuid& session_uuid) {
        return recovery_directory /
               (session_uuid.to_string() + ".licht");
    }

    bool is_scratch_autosave_path(
        const std::filesystem::path& path,
        const std::filesystem::path& recovery_directory) {
        if (path.empty() || recovery_directory.empty()) {
            return false;
        }
        const auto normalized_path =
            path.lexically_normal();
        const auto normalized_dir =
            recovery_directory.lexically_normal();
        if (normalized_path.parent_path() !=
            normalized_dir) {
            return false;
        }
        const auto name =
            normalized_path.filename().string();
        constexpr std::string_view extension = ".licht";
        if (name.size() <= extension.size() ||
            !name.ends_with(extension)) {
            return false;
        }
        const auto stem = name.substr(
            0, name.size() - extension.size());
        const auto uuid =
            lfs::core::Uuid::from_string(stem);
        return uuid && !uuid->is_nil();
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

    lfs::Result<std::filesystem::path>
    quarantine_project_artifact(
        const std::filesystem::path& source) {
        const auto unix_seconds =
            std::chrono::duration_cast<
                std::chrono::seconds>(
                std::chrono::system_clock::now()
                    .time_since_epoch())
                .count();
        auto aside = source;
        aside += std::format(
            ".corrupt-{}", unix_seconds);
        constexpr int kMaxDisambiguators = 32;
        for (int attempt = 0;
             attempt <= kMaxDisambiguators;
             ++attempt) {
            auto candidate = aside;
            if (attempt > 0) {
                candidate += std::format(
                    "-{}", attempt);
            }
            std::error_code exists_error;
            if (std::filesystem::exists(
                    candidate, exists_error) &&
                !exists_error) {
                continue;
            }
            std::error_code rename_error;
            std::filesystem::rename(
                source, candidate, rename_error);
            if (!rename_error) {
                return candidate;
            }
        }
        return fail<std::filesystem::path>(
            lfs::ErrorCode::Unavailable,
            source,
            "A corrupt project artifact could not be quarantined.",
            "no unused .corrupt-* aside name was available",
            "recovery.quarantine");
    }

    void sweep_orphan_project_artifacts(
        const std::filesystem::path& master_path,
        RecoveryInspection& into) {
        const auto directory =
            master_path.parent_path().empty()
                ? std::filesystem::path{"."}
                : master_path.parent_path();
        const auto stem = master_path.stem().string();
        const auto suffix =
            ".tmp" + master_path.extension().string();
        std::error_code master_stat_error;
        const bool published_master_exists =
            std::filesystem::is_regular_file(
                master_path, master_stat_error) &&
            !master_stat_error;
        std::vector<std::filesystem::path> write_temps;
        std::error_code error;
        for (std::filesystem::directory_iterator
                 iterator(directory, error),
             end;
             !error && iterator != end;
             iterator.increment(error)) {
            std::error_code type_error;
            const auto entry = iterator->path();
            const auto filename =
                entry.filename().string();
            if (filename.ends_with(".lock")) {
                continue;
            }
            if (is_write_temp_name(
                    filename, stem, suffix)) {
                if (iterator->is_regular_file(
                        type_error) &&
                    !type_error) {
                    write_temps.push_back(entry);
                } else if (published_master_exists) {
                    best_effort_remove_into(
                        into, entry);
                } else {
                    quarantine_into(
                        into,
                        entry,
                        "unpublished write temp preserved because the published master is absent");
                }
                continue;
            }
        }
        for (const auto& temp : write_temps) {
            try_reclaim_write_temp(
                into,
                temp,
                published_master_exists);
        }
        std::vector<std::filesystem::path> corrupt_asides;
        error.clear();
        for (std::filesystem::directory_iterator
                 iterator(directory, error),
             end;
             !error && iterator != end;
             iterator.increment(error)) {
            std::error_code type_error;
            const auto entry = iterator->path();
            const auto filename =
                entry.filename().string();
            if (filename.ends_with(".lock")) {
                continue;
            }
            if (iterator->is_regular_file(type_error) &&
                !type_error &&
                is_master_corrupt_aside(
                    filename, master_path)) {
                corrupt_asides.push_back(entry);
            }
        }
        prune_corrupt_asides(
            into, std::move(corrupt_asides));
    }

    void sweep_stale_licht_artifacts(
        const std::filesystem::path& master_path,
        const bool reclaim_master_lock) {
        if (master_path.empty()) {
            return;
        }
        const auto directory =
            master_path.parent_path().empty()
                ? std::filesystem::path{"."}
                : master_path.parent_path();
        const auto master_name =
            master_path.filename().string();
        const auto saveas_prefix =
            "." + master_name + ".saveas-";
        const auto stem =
            master_path.stem().string();
        const auto suffix =
            ".tmp" + master_path.extension().string();
        const auto master_lock_name =
            master_name + ".lock";

        std::vector<std::filesystem::path> saveas_data;
        std::vector<std::filesystem::path> write_temp_data;
        std::map<
            std::string,
            std::vector<std::filesystem::path>>
            unreferenced_write_temps;
        std::error_code error;
        for (std::filesystem::directory_iterator
                 iterator(directory, error),
             end;
             !error && iterator != end;
             iterator.increment(error)) {
            std::error_code type_error;
            if (!iterator->is_regular_file(
                    type_error) ||
                type_error) {
                continue;
            }
            const auto entry = iterator->path();
            const auto filename =
                entry.filename().string();
            if (filename == master_lock_name) {
                continue;
            }
            const auto data =
                path_without_lock_suffix(entry);
            const auto data_name =
                data.filename().string();
            if (data_name.starts_with(saveas_prefix)) {
                saveas_data.push_back(data);
                continue;
            }
            if (is_write_temp_name(
                    data_name, stem, suffix)) {
                write_temp_data.push_back(data);
                continue;
            }
            if (const auto referenced =
                    referenced_saveas_master_name(
                        data_name);
                referenced &&
                *referenced != master_name &&
                master_file_is_absent(
                    directory, *referenced)) {
                saveas_data.push_back(data);
                continue;
            }
            if (const auto referenced =
                    referenced_write_temp_master_name(
                        data_name);
                referenced &&
                *referenced != master_name &&
                master_file_is_absent(
                    directory, *referenced)) {
                unreferenced_write_temps[*referenced]
                    .push_back(data);
            }
        }
        const auto unique_sorted =
            [](std::vector<std::filesystem::path>&
                   paths) {
                std::ranges::sort(paths);
                paths.erase(
                    std::unique(
                        paths.begin(), paths.end()),
                    paths.end());
            };
        unique_sorted(saveas_data);
        unique_sorted(write_temp_data);
        for (auto& [_, temps] :
             unreferenced_write_temps) {
            unique_sorted(temps);
        }

        for (const auto& data : saveas_data) {
            try_remove_unheld_artifact(data, true);
        }
        for (const auto& [referenced_name, temps] :
             unreferenced_write_temps) {
            auto acquired =
                detail::WriterLock::acquire(
                    directory / referenced_name);
            if (!acquired) {
                continue;
            }
            for (const auto& data : temps) {
                try_remove_unheld_artifact(
                    data, true);
            }
        }

        std::error_code lock_exists_error;
        auto master_lock_path = master_path;
        master_lock_path += ".lock";
        const bool master_lock_existed =
            reclaim_master_lock &&
            std::filesystem::exists(
                master_lock_path,
                lock_exists_error) &&
            !lock_exists_error;

        std::optional<detail::WriterLock> master_guard;
        if (reclaim_master_lock ||
            !write_temp_data.empty()) {
            auto acquired =
                detail::WriterLock::acquire(
                    master_path);
            if (acquired) {
                master_guard.emplace(
                    std::move(*acquired));
            }
        }
        if (master_guard) {
            for (const auto& data : write_temp_data) {
                try_remove_unheld_artifact(
                    data, true);
            }
        }
        master_guard.reset();
        if (master_lock_existed) {
            lock_exists_error.clear();
            if (!std::filesystem::exists(
                    master_lock_path,
                    lock_exists_error) ||
                lock_exists_error) {
                LOG_INFO(
                    "Removed stale project artifact {}",
                    master_lock_path.string());
            }
        }
    }

    void sweep_stale_licht_artifacts_for_known_masters(
        const std::vector<std::filesystem::path>&
            master_paths) {
        for (const auto& master_path : master_paths) {
            std::error_code error;
            if (master_path.empty() ||
                !std::filesystem::is_regular_file(
                    master_path, error) ||
                error) {
                continue;
            }
            sweep_stale_licht_artifacts(
                master_path, true);
        }
    }

    lfs::Result<void> remove_scratch_autosave(
        const std::filesystem::path& scratch_path) {
        if (scratch_path.empty()) {
            return {};
        }
        {
            auto lock =
                detail::WriterLock::acquire(
                    scratch_path);
            if (!lock) {
                return lfs::Result<void>::failure(
                    std::move(lock).error());
            }
            if (auto removed = remove_path(scratch_path);
                !removed) {
                return removed;
            }
        }
        auto lock_path = scratch_path;
        lock_path += ".lock";
        std::error_code ignored;
        std::filesystem::remove(lock_path, ignored);
        return {};
    }

    lfs::Result<RecoveryInspection>
    inspect_scratch_autosave(
        const std::filesystem::path& scratch_path) {
        auto lock =
            detail::WriterLock::acquire(scratch_path);
        if (!lock) {
            return std::move(lock).error();
        }
        RecoveryInspection result;
        result.untitled_scratch = true;
        ReaderOptions inspection_options;
        inspection_options
            .allow_unsupported_inspection = true;
        auto reader = ProjectReader::open(
            scratch_path, inspection_options);
        if (!reader) {
            result.disposition =
                RecoveryDisposition::Invalid;
            result.diagnostics.push_back(
                std::format(
                    "{}: {}",
                    scratch_path.filename().string(),
                    lfs::format_for_developer(
                        reader.error())));
            return result;
        }
        if (reader->superblock().role !=
            ContainerRole::Master) {
            result.disposition =
                RecoveryDisposition::Invalid;
            result.diagnostics.push_back(
                std::format(
                    "{}: scratch recovery requires a complete master",
                    scratch_path.filename().string()));
            return result;
        }
        if (auto verified = reader->verify_all();
            !verified) {
            result.disposition =
                RecoveryDisposition::Invalid;
            result.diagnostics.push_back(
                std::format(
                    "{}: {}",
                    scratch_path.filename().string(),
                    lfs::format_for_developer(
                        verified.error())));
            return result;
        }
        if (!scratch_has_recoverable_content(*reader)) {
            result.disposition =
                RecoveryDisposition::Invalid;
            result.diagnostics.push_back(
                std::format(
                    "{}: empty untitled scratch has no recoverable content",
                    scratch_path.filename().string()));
            return result;
        }
        result.disposition = RecoveryDisposition::Offer;
        result.selected_path =
            scratch_path.lexically_normal();
        result.autosave_sequence =
            reader->superblock().autosave_sequence;
        result.snapshot_uuid =
            reader->commit().snapshot_uuid;
        result.commit_uuid =
            reader->commit().commit_uuid;
        result.wallclock_unix_ns =
            reader->commit().wallclock_unix_ns;
        return result;
    }

    void sweep_stale_scratch_autosaves(
        const std::filesystem::path& recovery_directory) {
        if (recovery_directory.empty()) {
            return;
        }
        std::error_code error;
        if (!std::filesystem::is_directory(
                recovery_directory, error) ||
            error) {
            return;
        }
        std::vector<std::filesystem::path> candidates;
        for (std::filesystem::directory_iterator
                 iterator(recovery_directory, error),
             end;
             !error && iterator != end;
             iterator.increment(error)) {
            std::error_code type_error;
            if (!iterator->is_regular_file(type_error) ||
                type_error) {
                continue;
            }
            const auto entry = iterator->path();
            const auto filename =
                entry.filename().string();
            if (filename.ends_with(".lock")) {
                continue;
            }
            candidates.push_back(entry);
        }
        for (const auto& entry : candidates) {
            if (!is_scratch_autosave_path(
                    entry, recovery_directory)) {
                try_remove_unheld_artifact(entry, true);
                continue;
            }
            auto inspection =
                inspect_scratch_autosave(entry);
            if (!inspection) {
                if (inspection.error().code() ==
                    lfs::ErrorCode::Unavailable) {
                    continue;
                }
                try_remove_unheld_artifact(entry, true);
                continue;
            }
            if (inspection->disposition ==
                RecoveryDisposition::Invalid) {
                try_remove_unheld_artifact(entry, true);
            }
        }
    }

    std::vector<RecoveryInspection>
    scan_scratch_autosaves(
        const std::filesystem::path& recovery_directory) {
        std::vector<RecoveryInspection> result;
        if (recovery_directory.empty()) {
            return result;
        }
        std::error_code error;
        if (!std::filesystem::is_directory(
                recovery_directory, error) ||
            error) {
            return result;
        }
        for (std::filesystem::directory_iterator
                 iterator(recovery_directory, error),
             end;
             !error && iterator != end;
             iterator.increment(error)) {
            std::error_code type_error;
            if (!iterator->is_regular_file(type_error) ||
                type_error) {
                continue;
            }
            const auto entry = iterator->path();
            if (!is_scratch_autosave_path(
                    entry, recovery_directory)) {
                continue;
            }
            auto inspection =
                inspect_scratch_autosave(entry);
            if (!inspection) {
                continue;
            }
            if (inspection->disposition ==
                RecoveryDisposition::Offer) {
                result.push_back(std::move(*inspection));
            } else if (
                inspection->disposition ==
                RecoveryDisposition::Invalid) {
                try_remove_unheld_artifact(entry, true);
            }
        }
        return result;
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
        sweep_orphan_project_artifacts(
            master_path, result);
        sweep_stale_licht_artifacts(
            master_path, false);
        for (const auto& temp :
             recovery_session_temps(master_path)) {
            best_effort_remove_into(result, temp);
        }

        struct Valid {
            std::filesystem::path path;
            std::uint64_t sequence = 0;
            std::uint64_t wallclock_unix_ns = 0;
            lfs::core::Uuid snapshot_uuid;
            lfs::core::Uuid commit_uuid;
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
                const auto reason = std::format(
                    "{}: {}",
                    candidate.filename().string(),
                    lfs::format_for_developer(
                        sidecar.error()));
                if (candidate == stable) {
                    quarantine_into(
                        result, candidate, reason);
                } else {
                    result.diagnostics.push_back(
                        reason);
                    best_effort_remove_into(
                        result, candidate);
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
                best_effort_remove_into(
                    result, candidate);
                continue;
            }
            if (!same_project) {
                const auto reason = std::format(
                    "{}: project UUID does not match the master",
                    candidate.filename().string());
                if (candidate == stable) {
                    quarantine_into(
                        result, candidate, reason);
                } else {
                    result.diagnostics.push_back(
                        reason);
                    best_effort_remove_into(
                        result, candidate);
                }
                continue;
            }
            auto complete =
                validate_complete_overlay(
                    *master, *sidecar);
            if (!complete) {
                const auto reason = std::format(
                    "{}: {}",
                    candidate.filename().string(),
                    lfs::format_for_developer(
                        complete.error()));
                if (candidate == stable) {
                    quarantine_into(
                        result, candidate, reason);
                } else {
                    result.diagnostics.push_back(
                        reason);
                    best_effort_remove_into(
                        result, candidate);
                }
                continue;
            }
            valid.push_back(Valid{
                .path = candidate,
                .sequence =
                    sidecar->superblock()
                        .autosave_sequence,
                .wallclock_unix_ns =
                    sidecar->commit()
                        .wallclock_unix_ns,
                .snapshot_uuid =
                    sidecar->superblock()
                        .sidecar_snapshot_uuid,
                .commit_uuid =
                    sidecar->commit().commit_uuid,
            });
        }

        if (!result.deleted_paths.empty()) {
            if (auto synced =
                    detail::sync_parent_directory(
                        master_path);
                !synced) {
                result.diagnostics.push_back(
                    lfs::format_for_developer(
                        synced.error()));
            }
        }

        if (valid.empty()) {
            if (!result.deleted_paths.empty()) {
                result.disposition =
                    RecoveryDisposition::
                        StaleDeleted;
            }
            return result;
        }
        const auto& winner = *std::ranges::max_element(
            valid, [](const Valid& lhs,
                      const Valid& rhs) {
                if (lhs.sequence != rhs.sequence) {
                    return lhs.sequence < rhs.sequence;
                }
                if (lhs.wallclock_unix_ns !=
                    rhs.wallclock_unix_ns) {
                    return lhs.wallclock_unix_ns <
                           rhs.wallclock_unix_ns;
                }
                return lhs.path.generic_string() <
                       rhs.path.generic_string();
            });
        std::string loser_list;
        for (const auto& candidate : valid) {
            if (candidate.path == winner.path) {
                continue;
            }
            if (!loser_list.empty()) {
                loser_list += ", ";
            }
            loser_list +=
                candidate.path.filename().string();
            quarantine_into(
                result,
                candidate.path,
                std::format(
                    "losing autosave tie (seq {} wallclock {})",
                    candidate.sequence,
                    candidate.wallclock_unix_ns));
        }
        if (!loser_list.empty()) {
            LOG_WARN(
                "Selected autosave {} (seq={}, wallclock={}) over {}",
                winner.path.filename().string(),
                winner.sequence,
                winner.wallclock_unix_ns,
                loser_list);
        }
        result.disposition =
            RecoveryDisposition::Offer;
        result.selected_path = winner.path;
        result.autosave_sequence = winner.sequence;
        result.snapshot_uuid = winner.snapshot_uuid;
        result.commit_uuid = winner.commit_uuid;
        result.wallclock_unix_ns =
            winner.wallclock_unix_ns;
        result.untitled_scratch = false;
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
        lfs::Result<void> cleanup_error;
        for (const auto& candidate :
             sidecar_candidates(master_path)) {
            if (auto removed =
                    remove_path(candidate);
                !removed) {
                LOG_WARN(
                    "Could not remove autosave artifact {}: {}",
                    candidate.filename().string(),
                    lfs::format_for_developer(
                        removed.error()));
                if (cleanup_error) {
                    cleanup_error = std::move(removed);
                }
            }
        }
        RecoveryInspection sweep;
        sweep_orphan_project_artifacts(
            master_path, sweep);
        for (const auto& diagnostic :
             sweep.diagnostics) {
            LOG_WARN(
                "Autosave artifact sweep: {}",
                diagnostic);
            if (cleanup_error) {
                cleanup_error = fail<void>(
                    lfs::ErrorCode::PermissionDenied,
                    master_path,
                    "A stale project artifact could not be removed.",
                    diagnostic,
                    "recovery.cleanup");
            }
        }
        if (auto synced =
                detail::sync_parent_directory(
                    master_path);
            !synced && cleanup_error) {
            cleanup_error = std::move(synced);
        }
        return cleanup_error;
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
        CommitBoundaryObserver boundary_observer,
        const RecoveryMaterializationOptions& options) {
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
                std::move(boundary_observer),
                options);
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
        CommitBoundaryObserver boundary_observer,
        const RecoveryMaterializationOptions& options) {
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
        CapabilitySet recovered_reader_capabilities =
            master->commit().required_reader_capabilities |
            sidecar->commit().required_reader_capabilities;
        CapabilitySet recovered_writer_capabilities =
            master->commit().required_writer_capabilities |
            sidecar->commit().required_writer_capabilities;
        // SIDECAR_OVERLAY_V1 describes the disposable overlay envelope, not
        // the materialized master. All content-level requirements survive.
        recovered_reader_capabilities.set(SIDECAR_OVERLAY_V1, false);
        recovered_writer_capabilities.set(SIDECAR_OVERLAY_V1, false);
        session.state_->temporary = destination;
        auto writer = ProjectWriter::create(
            destination,
            CreateOptions{
                .project_uuid =
                    master->superblock()
                        .project_uuid,
                .file_uuid =
                    options.file_uuid.is_nil()
                        ? lfs::core::generate_uuid_v4()
                        : options.file_uuid,
                .role = ContainerRole::Master,
                .base_explicit_commit_uuid = {},
                .autosave_sequence = 0,
                .sidecar_snapshot_uuid = {},
                .creation_time_unix_ns =
                    master->superblock()
                        .creation_time_unix_ns,
                .index_compression =
                    options.index_compression,
                .disk_reserve_bytes =
                    options.disk_reserve_bytes,
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
                        options.commit_uuid.is_nil()
                            ? lfs::core::generate_uuid_v4()
                            : options.commit_uuid,
                    .snapshot_uuid =
                        sidecar->superblock()
                            .sidecar_snapshot_uuid,
                    .wallclock_unix_ns =
                        options.wallclock_unix_ns == 0
                            ? detail::unix_time_ns()
                            : options.wallclock_unix_ns,
                    .min_reader_version =
                        std::max(
                            master->commit().min_reader_version,
                            sidecar->commit().min_reader_version),
                    .min_safe_writer_version =
                        std::max(
                            master->commit().min_safe_writer_version,
                            sidecar->commit().min_safe_writer_version),
                    .extra_reader_capabilities =
                        recovered_reader_capabilities,
                    .extra_writer_capabilities =
                        recovered_writer_capabilities,
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
            if (source) {
                if (source->stored_bytes >
                    std::numeric_limits<std::uint64_t>::max() -
                        planned_bytes) {
                    return fail<void>(
                        lfs::ErrorCode::ResourceExhausted,
                        master_path,
                        "The recovery overlay is too large to materialize.",
                        "stored chunk sizes overflow the recovery preflight total",
                        "recovery.preflight");
                }
                planned_bytes += source->stored_bytes;
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
    project_storage_stats(const ProjectReader& reader) {
        ProjectStorageStats result;
        result.physical_bytes =
            reader.physical_file_size();
        std::uint64_t live =
            APPEND_REGION_OFFSET +
            COMMIT_RECORD_BYTES +
            reader.commit().index_stored_bytes;
        for (const auto& row : reader.chunks()) {
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
                        reader.path(),
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
                    reader.path(),
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

    lfs::Result<ProjectStorageStats>
    project_storage_stats(
        const std::filesystem::path& master_path) {
        auto reader =
            ProjectReader::open(master_path);
        if (!reader) {
            return std::move(reader).error();
        }
        return project_storage_stats(*reader);
    }

} // namespace lfs::io::project
