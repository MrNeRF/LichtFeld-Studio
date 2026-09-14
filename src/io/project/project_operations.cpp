/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_operations.hpp"

#include "core/image_io.hpp"
#include "core/path_utils.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/project_document.hpp"

#include <stb_image_write.h>

#include <algorithm>
#include <concepts>
#include <cstring>
#include <format>
#include <istream>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::io::project {

    namespace {

        constexpr std::string_view WRITER_LOCK_MESSAGE =
            "The project is open for writing in another LichtFeld Studio";

        lfs::Error operation_error(const lfs::ErrorCode code,
                                   const std::filesystem::path& path,
                                   std::string message,
                                   std::string detail,
                                   const std::string_view field) {
            lfs::SmallFields fields;
            if (!path.empty()) {
                fields.add("path", lfs::core::path_to_utf8(path));
            }
            if (!field.empty()) {
                fields.add("field", field);
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        lfs::Result<T> fail(const lfs::ErrorCode code,
                            const std::filesystem::path& path,
                            std::string message,
                            std::string detail,
                            const std::string_view field) {
            auto error = operation_error(
                code, path, std::move(message), std::move(detail), field);
            if constexpr (std::same_as<T, void>) {
                return lfs::Result<void>::failure(std::move(error));
            } else {
                return error;
            }
        }

        lfs::Result<WriterLockLease>
        acquire_operation_lock(const std::filesystem::path& path) {
            auto lease = WriterLockLease::acquire(path);
            if (lease) {
                return std::move(*lease);
            }
            if (lease.error().code() == lfs::ErrorCode::Unavailable) {
                return fail<WriterLockLease>(
                    lfs::ErrorCode::Unavailable, path,
                    std::string(WRITER_LOCK_MESSAGE),
                    "the project writer lock is held by another process",
                    "writer_lock");
            }
            return std::move(lease).error();
        }

        bool is_singleton(const Fourcc fourcc) noexcept {
            return fourcc == FOURCC_PROJ || fourcc == FOURCC_PRMS ||
                   fourcc == FOURCC_SCNG || fourcc == FOURCC_SELM ||
                   fourcc == FOURCC_REFS || fourcc == FOURCC_GUIL ||
                   fourcc == FOURCC_VIEW || fourcc == FOURCC_EDTR ||
                   fourcc == FOURCC_SEQR || fourcc == FOURCC_METR;
        }

        bool same_path(const std::filesystem::path& lhs,
                       const std::filesystem::path& rhs) {
            std::error_code lhs_error;
            std::error_code rhs_error;
            const auto left = std::filesystem::absolute(lhs, lhs_error);
            const auto right = std::filesystem::absolute(rhs, rhs_error);
            return !lhs_error && !rhs_error &&
                   left.lexically_normal() == right.lexically_normal();
        }

        lfs::Result<void> refuse_existing_destination(
            const std::filesystem::path& path) {
            std::error_code error;
            if (std::filesystem::exists(path, error)) {
                return fail<void>(
                    lfs::ErrorCode::AlreadyExists, path,
                    "The restore destination already exists.",
                    "restore_save refuses to replace an existing file",
                    "destination");
            }
            if (error) {
                return fail<void>(
                    lfs::ErrorCode::PermissionDenied, path,
                    "The restore destination could not be inspected.",
                    std::format("filesystem::exists failed: {}", error.message()),
                    "destination");
            }
            return {};
        }

        lfs::Result<ProjectInspectorCard>
        inspect_after_save(const std::filesystem::path& path) {
            return inspect_project_card(path);
        }

        template <typename Mutator>
        lfs::Result<ProjectInspectorCard> mutate_document(
            const std::filesystem::path& path,
            Mutator&& mutator,
            const std::span<const std::byte> preview = {}) {
            auto lease = acquire_operation_lock(path);
            if (!lease) {
                return std::move(lease).error();
            }
            auto document = ProjectDocument::open(path);
            if (!document) {
                return std::move(document).error();
            }
            if (!document->source_reader() ||
                document->source_reader()->superblock().role != ContainerRole::Master) {
                return fail<ProjectInspectorCard>(
                    lfs::ErrorCode::FailedPrecondition, path,
                    "Autosave sidecars cannot be edited as projects.",
                    "closed project operations require a master container",
                    "superblock.container_role");
            }
            if (auto changed = std::forward<Mutator>(mutator)(*document);
                !changed) {
                return std::move(changed).error();
            }
            ProjectDocumentSaveOptions options;
            options.commit.kind = CommitKind::Explicit;
            options.writer_lock_lease = *lease;
            options.preview_png = preview;
            auto saved = document->save(path, options);
            if (!saved) {
                return std::move(saved).error();
            }
            return inspect_after_save(path);
        }

        lfs::Result<std::vector<std::byte>> encode_image_bytes(
            const std::span<const std::byte> input) {
            if (input.empty()) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::InvalidArgument, {},
                    "The embedded dataset image is empty.",
                    "image payload must contain encoded image bytes",
                    "preview.image");
            }
            auto [pixels, width, height, channels] =
                lfs::core::load_image_from_memory(
                    reinterpret_cast<const std::uint8_t*>(input.data()),
                    input.size());
            struct ImageGuard {
                unsigned char* pixels = nullptr;
                ~ImageGuard() {
                    if (pixels) {
                        lfs::core::free_image(pixels);
                    }
                }
            } guard{pixels};
            if (!pixels || width <= 0 || height <= 0 || channels < 1 || channels > 4) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::DataLoss, {},
                    "The embedded dataset image is invalid.",
                    "image payload could not be decoded into a supported layout",
                    "preview.image");
            }
            std::vector<std::byte> png;
            const auto callback = [](void* context, void* bytes, int size) {
                auto& target = *static_cast<std::vector<std::byte>*>(context);
                const auto* begin = static_cast<const std::byte*>(bytes);
                target.insert(target.end(), begin, begin + size);
            };
            if (!stbi_write_png_to_func(callback, &png, width, height, channels,
                                        pixels, width * channels) ||
                png.empty()) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::Unavailable, {},
                    "The embedded dataset image could not be encoded.",
                    "stbi_write_png_to_func failed", "preview.png");
            }
            return png;
        }

        lfs::Result<std::vector<std::byte>> read_lazy_payload(
            const LazyChunkValue& payload) {
            if (payload.size() > std::numeric_limits<std::size_t>::max()) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::ResourceExhausted, {},
                    "The embedded dataset image is too large.",
                    "image payload does not fit in this build", "preview.image");
            }
            std::vector<std::byte> result(static_cast<std::size_t>(payload.size()));
            std::uint64_t offset = 0;
            auto visited = payload.visit_stream(
                [&](std::istream& input, const std::uint64_t size) -> lfs::Result<void> {
                    if (size != result.size()) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss, {},
                            "The embedded dataset image size changed.",
                            "lazy payload size differs from its stream size",
                            "preview.image");
                    }
                    input.read(reinterpret_cast<char*>(result.data()),
                               static_cast<std::streamsize>(result.size()));
                    if (input.gcount() != static_cast<std::streamsize>(result.size())) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss, {},
                            "The embedded dataset image is incomplete.",
                            "lazy payload stream ended before the declared size",
                            "preview.image");
                    }
                    offset = size;
                    return {};
                });
            if (!visited) {
                return std::move(visited).error();
            }
            if (offset != result.size()) {
                return fail<std::vector<std::byte>>(
                    lfs::ErrorCode::DataLoss, {},
                    "The embedded dataset image is incomplete.",
                    "lazy payload stream did not provide its declared size",
                    "preview.image");
            }
            return result;
        }

    } // namespace

    lfs::Result<ProjectInspectorCard>
    restore_save(const std::filesystem::path& path,
                 const std::uint64_t generation,
                 const std::filesystem::path& destination) {
        if (path.empty() || destination.empty() || same_path(path, destination)) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::InvalidArgument, destination,
                "The restore paths are invalid.",
                "source and destination must be non-empty and different",
                "restore.path");
        }
        auto source_lock = acquire_operation_lock(path);
        if (!source_lock) {
            return std::move(source_lock).error();
        }
        auto destination_lock = acquire_operation_lock(destination);
        if (!destination_lock) {
            return std::move(destination_lock).error();
        }
        if (auto available = refuse_existing_destination(destination);
            !available) {
            return std::move(available).error();
        }
        auto reader = ProjectReader::open(path);
        if (!reader) {
            return std::move(reader).error();
        }
        if (reader->superblock().role != ContainerRole::Master) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::FailedPrecondition, path,
                "Autosave sidecars cannot be restored as projects.",
                "restore_save requires a master container", "superblock.container_role");
        }
        const auto lineage = reader->lineage();
        auto all_lineage_rows = reader->lineage_chunks();
        if (!all_lineage_rows) {
            return std::move(all_lineage_rows).error();
        }
        if (generation == 0 || generation > lineage.size()) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::InvalidArgument, path,
                "The requested save generation does not exist.",
                std::format("generation {} is outside the native lineage", generation),
                "restore.generation");
        }

        std::map<ChunkKey, ChunkInfo, ChunkKeyLess> selected_rows;
        for (std::size_t index = 0; index < generation; ++index) {
            for (const auto& row : (*all_lineage_rows)[index]) {
                if (row.row_kind == RowKind::Tombstone) {
                    selected_rows.erase(row.key);
                } else if (row.row_kind == RowKind::Live) {
                    selected_rows.insert_or_assign(row.key, row);
                } else {
                    return fail<ProjectInspectorCard>(
                        lfs::ErrorCode::DataLoss, path,
                        "The selected save contains an invalid overlay row.",
                        row.key_string(), "restore.lineage");
                }
            }
        }
        const auto old_project_uuid = reader->superblock().project_uuid;
        const auto new_project_uuid = lfs::core::generate_uuid_v4();
        const auto project_row = selected_rows.find(
            ChunkKey{FOURCC_PROJ, old_project_uuid});
        if (project_row == selected_rows.end()) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::DataLoss, path,
                "The selected save has no project chapter.",
                "PROJ is required for an openable restored project", "restore.PROJ");
        }
        auto selected_reader = ProjectReader::open_generation(path, generation,
                                                              reader->reader_options());
        if (!selected_reader) {
            return std::move(selected_reader).error();
        }
        auto project_bytes = selected_reader->read_chunk(project_row->second);
        if (!project_bytes) {
            return std::move(project_bytes).error();
        }
        auto project = ProjectChapter::from_bytes(*project_bytes);
        if (!project) {
            return std::move(project).error();
        }
        if (auto changed = project->set_project_uuid(new_project_uuid); !changed) {
            return std::move(changed).error();
        }
        auto project_lineage = project->project_lineage();
        if (!project_lineage) {
            return std::move(project_lineage).error();
        }
        if (std::ranges::find(*project_lineage, old_project_uuid) == project_lineage->end()) {
            project_lineage->push_back(old_project_uuid);
            if (auto changed = project->set_project_lineage(*project_lineage); !changed) {
                return std::move(changed).error();
            }
        }
        const auto rewritten_project = project->to_bytes();
        std::uint64_t planned_bytes = 0;
        for (const auto& [key, row] : selected_rows) {
            (void)key;
            const auto bytes = row.key == project_row->first
                                   ? rewritten_project.size()
                                   : row.stored_bytes;
            if (bytes > std::numeric_limits<std::uint64_t>::max() - planned_bytes) {
                return fail<ProjectInspectorCard>(
                    lfs::ErrorCode::ResourceExhausted, destination,
                    "The restored project is too large.",
                    "restore payload byte total overflowed", "restore.preflight");
            }
            planned_bytes += bytes;
        }
        auto created = ProjectWriter::create(
            destination,
            CreateOptions{
                .project_uuid = new_project_uuid,
                .file_uuid = lfs::core::generate_uuid_v4(),
                .role = ContainerRole::Master,
                .base_explicit_commit_uuid = {},
                .autosave_sequence = 0,
                .sidecar_snapshot_uuid = {},
                .creation_time_unix_ns = 0,
                .index_compression = IndexCompression::Zstd,
                .disk_reserve_bytes = 64ull * 1024 * 1024,
                .boundary_observer = {},
                .writer_lock_anchor_compatibility = {},
                .writer_lock_anchor = std::nullopt,
                .writer_lock_lease = *destination_lock,
            });
        if (!created) {
            return std::move(created).error();
        }
        ProjectWriter writer = std::move(*created);
        const auto& selected_commit = lineage[generation - 1];
        if (auto planned = writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = {},
                .snapshot_uuid = selected_commit.snapshot_uuid,
                .wallclock_unix_ns = selected_commit.wallclock_unix_ns,
                .min_reader_version = selected_commit.min_reader_version,
                .min_safe_writer_version = selected_commit.min_safe_writer_version,
                .extra_reader_capabilities = selected_commit.required_reader_capabilities,
                .extra_writer_capabilities = selected_commit.required_writer_capabilities,
            });
            !planned) {
            return std::move(planned).error();
        }
        if (auto preflight = writer.preflight(planned_bytes); !preflight) {
            return std::move(preflight).error();
        }
        for (const auto& [old_key, row] : selected_rows) {
            const ChunkKey new_key{
                .fourcc = old_key.fourcc,
                .instance_uuid = is_singleton(old_key.fourcc) &&
                                         old_key.instance_uuid == old_project_uuid
                                     ? new_project_uuid
                                     : old_key.instance_uuid,
            };
            if (old_key == project_row->first) {
                if (auto written = writer.write_chunk(
                        new_key, rewritten_project,
                        ChunkWriteOptions{
                            .chunk_version = row.chunk_version,
                            .compression = row.compression,
                            .tensor_payload = (row.flags & TENSOR_PAYLOAD) != 0,
                            .block_crcs = (row.flags & HAS_BLOCK_CRCS) != 0,
                            .expected_stream_bytes = row.uncompressed_bytes,
                        });
                    !written) {
                    return std::move(written).error();
                }
            } else if (old_key.fourcc == FOURCC_THMB) {
                auto preview = selected_reader->read_chunk(row);
                if (!preview) {
                    return std::move(preview).error();
                }
                if (auto written = writer.set_preview(*preview); !written) {
                    return std::move(written).error();
                }
            } else if (new_key != old_key) {
                auto bytes = selected_reader->read_chunk(row);
                if (!bytes) {
                    return std::move(bytes).error();
                }
                if (auto written = writer.write_chunk(
                        new_key, *bytes,
                        ChunkWriteOptions{
                            .chunk_version = row.chunk_version,
                            .compression = row.compression,
                            .tensor_payload = (row.flags & TENSOR_PAYLOAD) != 0,
                            .block_crcs = (row.flags & HAS_BLOCK_CRCS) != 0,
                            .expected_stream_bytes = row.uncompressed_bytes,
                        });
                    !written) {
                    return std::move(written).error();
                }
            } else if (auto copied = writer.copy_chunk_verbatim(*selected_reader, row);
                       !copied) {
                return std::move(copied).error();
            }
        }
        if (auto committed = writer.commit(); !committed) {
            return std::move(committed).error();
        }
        auto restored = ProjectReader::open(destination);
        if (!restored) {
            return std::move(restored).error();
        }
        if (auto verified = restored->verify_all(); !verified) {
            return std::move(verified).error();
        }
        return inspect_after_save(destination);
    }

    lfs::Result<ProjectInspectorCard>
    rebind_checkpoint(const std::filesystem::path& path,
                      const lfs::core::Uuid& checkpoint_instance_uuid) {
        if (checkpoint_instance_uuid.is_nil()) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::InvalidArgument, path,
                "The checkpoint UUID cannot be empty.",
                "resume requires a non-null retained checkpoint UUID",
                "checkpoint.instance_uuid");
        }
        auto lease = acquire_operation_lock(path);
        if (!lease) {
            return std::move(lease).error();
        }
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        if (!document->source_reader() ||
            document->source_reader()->superblock().role != ContainerRole::Master) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::FailedPrecondition, path,
                "Autosave sidecars cannot be resumed.",
                "resume requires a master project", "superblock.container_role");
        }
        if (!document->find_checkpoint(checkpoint_instance_uuid)) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::NotFound, path,
                "The retained checkpoint was not found.",
                checkpoint_instance_uuid.to_string(), "checkpoint.instance_uuid");
        }
        const auto recovery = path.parent_path() /
                              (path.filename().string() + ".before-rebind.licht");
        std::error_code copy_error;
        if (!std::filesystem::copy_file(path, recovery,
                                        std::filesystem::copy_options::none,
                                        copy_error)) {
            return fail<ProjectInspectorCard>(
                copy_error == std::errc::file_exists
                    ? lfs::ErrorCode::AlreadyExists
                    : lfs::ErrorCode::PermissionDenied,
                recovery,
                "The recovery copy could not be created.",
                copy_error ? copy_error.message() : "copy_file returned false",
                "recovery_copy");
        }
        auto training = document->scene_graph().training_model_uuid();
        if (!training) {
            return std::move(training).error();
        }
        if (!*training) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::FailedPrecondition, path,
                "The project has no training model to resume.",
                "SCNG.training_model_uuid is absent", "SCNG.training_model_uuid");
        }
        auto node = document->scene_graph().find(**training);
        if (!node) {
            return std::move(node).error();
        }
        if (!*node || !(*node)->payload || (*node)->payload->fourcc != "CKPT") {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::FailedPrecondition, path,
                "The training model is not checkpoint-backed.",
                "SCNG training node does not bind a CKPT payload",
                "SCNG.training_model_uuid");
        }
        (*node)->payload->instance_uuid = checkpoint_instance_uuid;
        if (auto updated = document->edit_scene_graph().upsert_node(**node);
            !updated) {
            return std::move(updated).error();
        }
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.writer_lock_lease = *lease;
        auto saved = document->save(path, options);
        if (!saved) {
            return std::move(saved).error();
        }
        return inspect_after_save(path);
    }

    lfs::Result<ProjectInspectorCard>
    compact_project_file(const std::filesystem::path& path,
                         ProjectOperationProgress progress,
                         ProjectOperationCancel cancel) {
        CompactionOptions options;
        options.progress = std::move(progress);
        options.cancel = std::move(cancel);
        auto compacted = ProjectWriter::compact(path, std::move(options));
        if (!compacted) {
            if (compacted.error().code() == lfs::ErrorCode::Unavailable) {
                return fail<ProjectInspectorCard>(
                    lfs::ErrorCode::Unavailable, path,
                    std::string(WRITER_LOCK_MESSAGE),
                    "the project writer lock is held by another process",
                    "writer_lock");
            }
            return std::move(compacted).error();
        }
        auto card = inspect_after_save(path);
        if (!card) {
            return card;
        }
        card->diagnostic = "Compact removed older save points; retained checkpoints were preserved.";
        return card;
    }

    lfs::Result<ProjectVerificationResult>
    verify_project_file(const std::filesystem::path& path,
                        ProjectOperationProgress progress,
                        ProjectOperationCancel cancel) {
        auto reader = ProjectReader::open(path);
        if (!reader) {
            return std::move(reader).error();
        }
        ProjectVerificationResult result;
        std::size_t index = 0;
        const auto total = reader->chunks().size();
        if (progress) {
            progress(0.0F, "Verifying project");
        }
        for (const auto& row : reader->chunks()) {
            if (row.row_kind != RowKind::Live) {
                ++index;
                continue;
            }
            if (cancel && cancel()) {
                result.status = ProjectVerificationStatus::Canceled;
                if (progress) {
                    progress(total == 0 ? 1.0F
                                        : static_cast<float>(index) /
                                              static_cast<float>(total),
                             "Verification canceled");
                }
                return result;
            }
            auto verified = reader->verify_chunk(row);
            if (!verified) {
                result.status = ProjectVerificationStatus::Failed;
                result.first_mismatch = lfs::format_for_developer(verified.error());
                return result;
            }
            ++result.verified_chunks;
            ++index;
            if (progress) {
                progress(total == 0 ? 1.0F
                                    : static_cast<float>(index) /
                                          static_cast<float>(total),
                         "Verifying project");
            }
        }
        result.status = ProjectVerificationStatus::Verified;
        return result;
    }

    lfs::Result<ProjectInspectorCard>
    set_project_preview(const std::filesystem::path& path,
                        const std::span<const std::byte> png_bytes) {
        return mutate_document(path, [](ProjectDocument&) -> lfs::Result<void> { return {}; }, png_bytes);
    }

    lfs::Result<ProjectInspectorCard>
    preview_from_first_dataset_image(const std::filesystem::path& path) {
        auto lease = acquire_operation_lock(path);
        if (!lease) {
            return std::move(lease).error();
        }
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        const auto first = first_dataset_image(
            document->project(), document->references(), document->parameters(),
            path.parent_path());
        if (!first) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::NotFound, path,
                "The project has no reachable dataset image.",
                "first_dataset_image returned no image", "preview.dataset");
        }
        auto png = dataset_preview_png(*first);
        if (!png) {
            return std::move(png).error();
        }
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.writer_lock_lease = *lease;
        options.preview_png = *png;
        auto saved = document->save(path, options);
        if (!saved) {
            return std::move(saved).error();
        }
        return inspect_after_save(path);
    }

    lfs::Result<ProjectInspectorCard>
    preview_from_first_embedded_image(const std::filesystem::path& path) {
        auto lease = acquire_operation_lock(path);
        if (!lease) {
            return std::move(lease).error();
        }
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        auto manifest = document->parameters().embedded_dataset();
        if (!manifest) {
            return std::move(manifest).error();
        }
        if (!*manifest) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::NotFound, path,
                "The project has no embedded dataset image.",
                "embedded dataset manifest is absent", "preview.dataset");
        }
        const auto entry = std::ranges::find_if(
            (**manifest).entries,
            [](const EmbeddedDatasetEntry& value) { return value.kind == "image"; });
        if (entry == (**manifest).entries.end()) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::NotFound, path,
                "The embedded dataset has no image payload.",
                "embedded dataset manifest contains no image entry",
                "preview.dataset");
        }
        const auto* payload = document->find_dataset_source(entry->chunk_uuid);
        if (!payload) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::DataLoss, path,
                "The embedded dataset image payload is missing.",
                entry->chunk_uuid.to_string(), "preview.dataset");
        }
        auto image = read_lazy_payload(*payload);
        if (!image) {
            return std::move(image).error();
        }
        auto png = encode_image_bytes(*image);
        if (!png) {
            return std::move(png).error();
        }
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.writer_lock_lease = *lease;
        options.preview_png = *png;
        auto saved = document->save(path, options);
        if (!saved) {
            return std::move(saved).error();
        }
        return inspect_after_save(path);
    }

    lfs::Result<ProjectInspectorCard>
    set_project_license(const std::filesystem::path& path,
                        const std::string& identifier,
                        const std::string& notice) {
        return mutate_document(path, [&](ProjectDocument& document) {
            return document.set_license(ProjectLicense{identifier, notice});
        });
    }

    lfs::Result<ProjectInspectorCard>
    clear_project_license(const std::filesystem::path& path) {
        return mutate_document(path, [](ProjectDocument& document) {
            return document.clear_license();
        });
    }

    lfs::Result<ProjectInspectorCard>
    set_project_title(const std::filesystem::path& path,
                      const std::string& title) {
        return mutate_document(path, [&](ProjectDocument& document) {
            auto& dom = document.edit_project().dom();
            if (title.empty()) {
                auto removed = dom.remove("title");
                if (!removed) {
                    return lfs::Result<void>::failure(std::move(removed).error());
                }
                return lfs::Result<void>{};
            }
            return dom.set("title", title);
        });
    }

} // namespace lfs::io::project
