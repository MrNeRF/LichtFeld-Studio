/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_operations.hpp"

#include "core/image_io.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "core/user_paths.hpp"
#include "crc32c.hpp"
#include "io/atomic_output.hpp"
#include "io/embedded_dataset.hpp"
#include "io/exporter.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loader.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "project_container_internal.hpp"
#include "span_streambuf.hpp"

#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstring>
#include <format>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <istream>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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

        thread_local const WriterLockLease* active_operation_lease = nullptr;

        lfs::Result<WriterLockLease>
        acquire_operation_lock(const std::filesystem::path& path) {
            if (const auto* planned = detail::active_operation_identity()) {
                if (auto identity = planned->validate(); !identity)
                    return std::move(identity).error();
                std::error_code error;
                if (active_operation_lease && !active_operation_lease->owns(path) &&
                    std::filesystem::exists(path, error))
                    return fail<WriterLockLease>(lfs::ErrorCode::FailedPrecondition, path,
                                                 "The project path changed before writing. Refresh Projects and try again.",
                                                 "the operation tried to edit a different existing file", "operation.path");
            }
            if (active_operation_lease && active_operation_lease->owns(path))
                return *active_operation_lease;
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
            const auto left = std::filesystem::weakly_canonical(lhs, lhs_error);
            const auto right = std::filesystem::weakly_canonical(rhs, rhs_error);
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

        JsonChapterDom::Json save_origin(const ProjectInspectorSave& head) {
            const bool derived = head.kind == CommitKind::Contents || head.kind == CommitKind::Compaction;
            JsonChapterDom::Json source{
                {"generation", derived ? head.source_save_generation : head.generation},
                {"saved_at_unix_ns", derived ? head.source_saved_at_unix_ns : head.saved_at_unix_ns},
                {"kind", static_cast<std::uint32_t>(derived ? head.source_save_kind : head.kind)},
                {"strategy", head.strategy},
            };
            if (head.checkpoint_iteration)
                source["checkpoint_iteration"] = *head.checkpoint_iteration;
            if (head.planned_iterations)
                source["planned_iterations"] = *head.planned_iterations;
            if (head.gaussians)
                source["gaussians"] = *head.gaussians;
            return source;
        }

        lfs::Result<void> mark_contents_edit(ProjectDocument& document,
                                             const std::string_view operation, const std::uint64_t generations = 1) {
            const auto* reader = document.source_reader();
            auto details = inspect_project_details(reader->path());
            if (!details)
                return lfs::Result<void>::failure(std::move(details).error());
            if (details->save_history.empty())
                return {};
            return document.edit_project().dom().set_json("contents_edit", {
                                                                               {"file_uuid", reader->superblock().file_uuid.to_string()},
                                                                               {"first_generation", reader->commit().generation + 1},
                                                                               {"last_generation", reader->commit().generation + generations},
                                                                               {"operation", operation},
                                                                               {"source_save", save_origin(details->save_history.back())},
                                                                           });
        }

        template <typename Mutator>
        lfs::Result<ProjectInspectorCard> mutate_document(
            const std::filesystem::path& path,
            const std::string_view operation,
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
            if (auto marked = mark_contents_edit(*document, operation); !marked)
                return std::move(marked).error();
            ProjectDocumentSaveOptions options;
            options.commit.kind = CommitKind::Explicit;
            options.regenerate_dataset_preview = false;
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

        std::uint64_t row_occupied_bytes(const ChunkInfo& row) {
            const auto payload_end = row.payload_offset + row.stored_bytes;
            const auto aligned_end =
                (payload_end + CHUNK_ALIGNMENT - 1) & ~(CHUNK_ALIGNMENT - 1);
            return (aligned_end - row.header_offset) +
                   (row.block_crc_table
                        ? BLOCK_CRC_HEADER_BYTES +
                              static_cast<std::uint64_t>(row.block_crc_table->entries.size()) *
                                  sizeof(std::uint32_t)
                        : 0);
        }

        JsonChapterDom::Json pending_contents(const ProjectDocument& document) {
            auto pending = document.project().dom().get_json("contents_removals");
            const auto file_id = document.source_reader()->superblock().file_uuid.to_string();
            if (!pending || !pending->is_object() ||
                pending->value("file_uuid", std::string{}) != file_id ||
                !pending->contains("rows") || !(*pending)["rows"].is_array()) {
                return JsonChapterDom::Json{{"file_uuid", file_id}, {"rows", JsonChapterDom::Json::array()}};
            }
            return *pending;
        }

        lfs::Result<std::pair<std::int32_t, std::uint64_t>>
        checkpoint_facts(const LazyChunkValue& payload) {
            std::array<std::byte, sizeof(lfs::core::CheckpointHeader)> prefix{};
            if (auto read = payload.peek_prefix(prefix); !read) {
                return std::move(read).error();
            }
            SpanStreambuf buffer(std::span<const std::byte>(prefix.data(), prefix.size()));
            std::istream stream(&buffer);
            auto header = lfs::core::load_checkpoint_header(stream, payload.size());
            if (!header) {
                return fail<std::pair<std::int32_t, std::uint64_t>>(
                    lfs::ErrorCode::DataLoss, {},
                    "A retained checkpoint header could not be read.", header.error(),
                    "CKPT.header");
            }
            return std::pair{header->iteration, payload.size()};
        }

        struct ExternalDatasetStatus {
            bool valid = false;
            std::optional<ReferenceRecord> record;
            std::filesystem::path root;
        };

        lfs::Result<ExternalDatasetStatus> validate_external_dataset(
            const ProjectDocument& document,
            const EmbeddedDatasetManifest& manifest) {
            ExternalDatasetStatus status;
#ifdef LFS_FORMAT_TEST_TARGET
            (void)document;
            (void)manifest;
            return status;
#else
            auto dataset_uuid = document.project().dataset_reference();
            if (!dataset_uuid) {
                return std::move(dataset_uuid).error();
            }
            if (!*dataset_uuid) {
                return status;
            }
            auto record = document.references().find(**dataset_uuid);
            if (!record) {
                return std::move(record).error();
            }
            if (!*record || (*record)->kind != "dataset") {
                return status;
            }
            const auto root = resolve_path_reference(
                document.references(),
                document.source_path() ? document.source_path()->parent_path()
                                       : std::filesystem::path{},
                **dataset_uuid);
            if (!root || !std::filesystem::is_directory(*root)) {
                return status;
            }
            auto directory_check = check_fingerprint(*root, (*record)->fingerprint);
            if (!directory_check || !directory_check->matches()) {
                return status;
            }
            for (const auto& entry : manifest.entries) {
                const auto relative = lfs::core::utf8_to_path(entry.rel_path);
                const auto file = (*root / relative).lexically_normal();
                std::error_code error;
                if (relative.empty() || relative.is_absolute() ||
                    relative.lexically_normal() != relative ||
                    !std::filesystem::is_regular_file(file, error) || error ||
                    std::filesystem::file_size(file, error) != entry.bytes || error) {
                    return status;
                }
                auto hash = hash_dataset_file(file);
                if (!hash || *hash != entry.xxh3_128) {
                    return status;
                }
            }
            status.valid = true;
            status.record = **record;
            status.root = *root;
            return status;
#endif
        }

        lfs::Result<ProjectReducePlan>
        make_reduce_plan(const std::filesystem::path& path) {
            auto reader = ProjectReader::open(path);
            if (!reader) {
                return std::move(reader).error();
            }
            auto document = ProjectDocument::open(path);
            if (!document) {
                return std::move(document).error();
            }
            if (!document->source_reader() ||
                document->source_reader()->superblock().role != ContainerRole::Master) {
                return fail<ProjectReducePlan>(
                    lfs::ErrorCode::FailedPrecondition, path,
                    "Autosave sidecars cannot be reduced.",
                    "reduce_size requires a master container", "container.role");
            }
            ProjectReducePlan plan;
            plan.path = path;
            plan.input_commit_uuid = reader->commit().commit_uuid;
            plan.physical_size = reader->physical_file_size();
            auto storage = project_storage_stats(*reader);
            if (!storage) {
                return std::move(storage).error();
            }
            plan.tombstone_bytes = storage->dead_bytes;

            auto bound = document->bound_checkpoint_uuid();
            if (!bound) {
                return std::move(bound).error();
            }
            for (const auto& uuid : document->checkpoint_uuids()) {
                const auto* payload = document->find_checkpoint(uuid);
                if (!payload) {
                    continue;
                }
                auto facts = checkpoint_facts(*payload);
                if (!facts) {
                    return std::move(facts).error();
                }
                const auto* row = reader->find(FOURCC_CKPT, uuid);
                plan.retained_checkpoints.push_back(ProjectReduceCheckpoint{
                    .instance_uuid = uuid,
                    .iteration = facts->first,
                    .bytes = row ? row_occupied_bytes(*row) : facts->second,
                    .scng_bound = *bound && **bound == uuid,
                });
            }

            auto manifest = document->parameters().embedded_dataset();
            if (!manifest) {
                return std::move(manifest).error();
            }
            bool all_external = manifest->has_value();
            if (*manifest) {
                auto external = validate_external_dataset(*document, **manifest);
                if (!external) {
                    return std::move(external).error();
                }
                all_external = external->valid;
                for (const auto& entry : (**manifest).entries) {
                    const auto* row = reader->find(FOURCC_DSRC, entry.chunk_uuid);
                    plan.embedded_dataset.push_back(ProjectReduceDatasetPayload{
                        .chunk_uuid = entry.chunk_uuid,
                        .rel_path = entry.rel_path,
                        .kind = entry.kind,
                        .bytes = row ? row_occupied_bytes(*row) : entry.bytes,
                        .external_replacement_validated = external->valid,
                    });
                }
            } else {
                all_external = false;
            }

            std::set<ChunkKey, ChunkKeyLess> current;
            for (const auto& row : reader->chunks()) {
                if (row.is_live()) {
                    current.insert(row.key);
                }
            }
            auto lineage = reader->lineage_chunks();
            if (lineage) {
                for (const auto& generation : *lineage) {
                    for (const auto& row : generation) {
                        if (row.row_kind == RowKind::Tombstone) {
                            ++plan.superseded_rows;
                        } else if (row.row_kind == RowKind::Live &&
                                   row.source_generation < reader->commit().generation &&
                                   current.contains(row.key)) {
                            ++plan.superseded_rows;
                        }
                    }
                }
            }

            std::uint64_t checkpoint_reclaim = 0;
            for (const auto& checkpoint : plan.retained_checkpoints) {
                if (!checkpoint.scng_bound) {
                    checkpoint_reclaim += checkpoint.bytes;
                }
            }
            std::uint64_t dataset_reclaim = 0;
            for (const auto& payload : plan.embedded_dataset) {
                dataset_reclaim += payload.bytes;
            }
            const auto projected = [&](const std::uint64_t reclaim) {
                return reclaim >= plan.physical_size ? 0 : plan.physical_size - reclaim;
            };
            plan.drop_checkpoints = ProjectReduceProjection{
                .enabled = checkpoint_reclaim != 0,
                .allowed = true,
                .reclaimable_bytes = checkpoint_reclaim,
                .projected_size = projected(plan.tombstone_bytes + checkpoint_reclaim),
            };
            plan.drop_embedded_dataset = ProjectReduceProjection{
                .enabled = !plan.embedded_dataset.empty(),
                .allowed = !plan.embedded_dataset.empty() && all_external,
                .reclaimable_bytes = dataset_reclaim,
                .projected_size = projected(plan.tombstone_bytes + dataset_reclaim),
            };
            plan.compact = ProjectReduceProjection{
                .enabled = plan.tombstone_bytes != 0,
                .allowed = true,
                .reclaimable_bytes = plan.tombstone_bytes,
                .projected_size = projected(plan.tombstone_bytes),
            };
            return plan;
        }

#ifndef LFS_FORMAT_TEST_TARGET
        std::vector<std::pair<std::filesystem::path, std::string>>
        discover_dataset_files(const std::filesystem::path& root,
                               const std::string& configured_images,
                               std::string& images_folder) {
            auto info = lfs::io::detect_dataset_info(root);
            if (!configured_images.empty() && configured_images != ".") {
                const auto configured = root / lfs::core::utf8_to_path(configured_images);
                if (std::filesystem::is_directory(configured)) {
                    info.images_path = configured;
                }
            }
            images_folder = lfs::core::path_to_generic_utf8(
                info.images_path.lexically_relative(root));
            std::vector<std::pair<std::filesystem::path, std::string>> result;
            const auto append = [&](const std::filesystem::path& directory,
                                    const std::string_view kind) {
                if (!std::filesystem::is_directory(directory)) {
                    return;
                }
                std::error_code error;
                for (std::filesystem::recursive_directory_iterator it(directory, error), end;
                     !error && it != end; it.increment(error)) {
                    if (it->is_regular_file(error) && !error) {
                        result.emplace_back(it->path(), std::string(kind));
                    }
                }
            };
            append(info.images_path, "image");
            if (info.has_masks)
                append(info.masks_path, "mask");
            if (info.has_depths)
                append(info.depths_path, "depth");
            if (info.has_normals)
                append(info.normals_path, "normal");
            append(info.sparse_path, "sparse");
            for (const auto name : {"project.ini", "transforms.json",
                                    "transforms_train.json", "transforms_test.json",
                                    "transforms_val.json"}) {
                const auto file = root / name;
                if (std::filesystem::is_regular_file(file)) {
                    result.emplace_back(file, "meta");
                }
            }
            std::set<std::string> seen;
            std::erase_if(result, [&](const auto& item) {
                return !seen.insert(lfs::core::path_to_generic_utf8(
                                        item.first.lexically_relative(root)))
                            .second;
            });
            std::ranges::sort(result, {}, [&](const auto& item) {
                return lfs::core::path_to_generic_utf8(item.first.lexically_relative(root));
            });
            return result;
        }
#endif

#ifndef LFS_FORMAT_TEST_TARGET
        struct ExportSceneData {
            std::vector<std::shared_ptr<lfs::core::SplatData>> owners;
            std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> splats;
        };

        lfs::Result<std::shared_ptr<lfs::core::SplatData>> load_export_payload(
            const ProjectDocument& document, const SceneNodeRecord& node,
            const PayloadBinding& binding, const ProjectOperationCancel& cancel) {
            if (cancel && cancel()) {
                return fail<std::shared_ptr<lfs::core::SplatData>>(
                    lfs::ErrorCode::Cancelled, {}, "Project export was canceled.",
                    "the caller canceled while loading a visible payload", "export.cancel");
            }
            if (binding.fourcc == "SPLT") {
                const auto* chunk = document.source_reader()->find(
                    FOURCC_SPLT, binding.instance_uuid);
                if (!chunk) {
                    return fail<std::shared_ptr<lfs::core::SplatData>>(
                        lfs::ErrorCode::DataLoss, {}, "A visible splat payload is missing.",
                        binding.instance_uuid.to_string(), "export.payload");
                }
                auto bytes = document.source_reader()->read_chunk(*chunk);
                if (!bytes) {
                    return std::move(bytes).error();
                }
                auto payload = SplatChapterPayload::from_lfsp(std::move(*bytes));
                if (!payload) {
                    return std::move(payload).error();
                }
                auto data = payload->hydrate();
                if (!data) {
                    return std::move(data).error();
                }
                return std::shared_ptr<lfs::core::SplatData>(std::move(*data));
            }
            if (binding.fourcc == "CKPT") {
                const auto* payload = document.find_checkpoint(binding.instance_uuid);
                if (!payload) {
                    return fail<std::shared_ptr<lfs::core::SplatData>>(
                        lfs::ErrorCode::DataLoss, {}, "A visible checkpoint payload is missing.",
                        binding.instance_uuid.to_string(), "export.payload");
                }
                std::shared_ptr<lfs::core::SplatData> result;
                auto loaded = payload->visit_materialized(
                    [&](std::istream& stream, const std::uint64_t bytes) -> lfs::Result<void> {
                        auto data = lfs::core::load_checkpoint_splat_data(stream, bytes);
                        if (!data) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss, {},
                                "A visible checkpoint could not be decoded.", data.error(),
                                "export.payload");
                        }
                        result = std::make_shared<lfs::core::SplatData>(std::move(*data));
                        return {};
                    });
                if (!loaded) {
                    return std::move(loaded).error();
                }
                return result;
            }
            if (binding.fourcc != "DSRC" ||
                (binding.source_kind != "ply" && binding.source_kind != "sog" &&
                 binding.source_kind != "ssog" && binding.source_kind != "spz")) {
                return fail<std::shared_ptr<lfs::core::SplatData>>(
                    lfs::ErrorCode::Unsupported, {},
                    "The visible splat payload cannot be exported.",
                    "unsupported payload binding " + binding.fourcc,
                    "export.payload");
            }
            std::optional<std::filesystem::path> source_path;
            if (binding.reference_uuid) {
                source_path = resolve_path_reference(
                    document.references(),
                    document.source_path() ? document.source_path()->parent_path()
                                           : std::filesystem::path{},
                    *binding.reference_uuid);
            } else {
                auto materialized = document.materialize_embedded_asset(
                    binding.instance_uuid, binding.source_kind);
                if (!materialized) {
                    return std::move(materialized).error();
                }
                source_path = *materialized;
            }
            if (!source_path || !std::filesystem::is_regular_file(*source_path)) {
                return fail<std::shared_ptr<lfs::core::SplatData>>(
                    lfs::ErrorCode::NotFound, {},
                    "A visible splat source file could not be found.",
                    node.name, "export.payload");
            }
            auto loader = lfs::io::Loader::create();
            lfs::io::LoadOptions options;
            options.cancel_requested = cancel;
            auto loaded = loader->load(*source_path, options);
            if (!loaded) {
                return fail<std::shared_ptr<lfs::core::SplatData>>(
                    lfs::ErrorCode::DataLoss, *source_path,
                    "A visible splat source could not be decoded.", loaded.error().message,
                    "export.payload");
            }
            auto* data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&loaded->data);
            if (!data || !*data) {
                return fail<std::shared_ptr<lfs::core::SplatData>>(
                    lfs::ErrorCode::DataLoss, *source_path,
                    "The visible source is not a splat file.", source_path->string(),
                    "export.payload");
            }
            return *data;
        }

        lfs::Result<ExportSceneData> load_export_scene(
            const std::filesystem::path& path, const ProjectOperationCancel& cancel) {
            auto document = ProjectDocument::open(path, {.defer_geometry_payloads = true});
            if (!document) {
                return std::move(document).error();
            }
            auto records = document->scene_graph().nodes();
            if (!records) {
                return std::move(records).error();
            }
            std::unordered_map<lfs::core::Uuid, const SceneNodeRecord*> by_id;
            for (const auto& record : *records) {
                by_id.emplace(record.uuid, &record);
            }
            ExportSceneData result;
            for (const auto& record : *records) {
                if (record.type != "splat" || !record.visible || !record.payload) {
                    continue;
                }
                bool visible = true;
                glm::mat4 world = glm::make_mat4(record.local_transform.data());
                auto parent = record.parent_uuid;
                std::size_t depth = 0;
                while (parent) {
                    const auto found = by_id.find(*parent);
                    if (found == by_id.end() || ++depth > records->size()) {
                        return fail<ExportSceneData>(
                            lfs::ErrorCode::DataLoss, path,
                            "The project scene hierarchy is invalid.", record.name,
                            "export.scene_graph");
                    }
                    visible = visible && found->second->visible;
                    world = glm::make_mat4(found->second->local_transform.data()) * world;
                    parent = found->second->parent_uuid;
                }
                if (!visible || record.payload->reference_uuid && record.payload->fourcc != "DSRC") {
                    continue;
                }
                auto data = load_export_payload(
                    *document, record, *record.payload, cancel);
                if (!data) {
                    return std::move(data).error();
                }
                result.owners.push_back(*data);
                result.splats.emplace_back(result.owners.back().get(), world);
            }
            if (result.splats.empty()) {
                return fail<ExportSceneData>(
                    lfs::ErrorCode::NotFound, path,
                    "The project has no visible splats to export.",
                    "SCNG contains no visible splat payload", "export.scene_graph");
            }
            return result;
        }
#endif

        std::uint64_t read_u64_bytes(const std::span<const std::byte> bytes,
                                     const std::size_t offset) {
            std::uint64_t value = 0;
            for (std::size_t index = 0; index < 8; ++index) {
                value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(
                             bytes[offset + index]))
                         << (index * 8);
            }
            return value;
        }

        std::uint32_t read_u32_bytes(const std::span<const std::byte> bytes,
                                     const std::size_t offset) {
            std::uint32_t value = 0;
            for (std::size_t index = 0; index < 4; ++index) {
                value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(
                             bytes[offset + index]))
                         << (index * 8);
            }
            return value;
        }

        void write_u64_bytes(const std::span<std::byte> bytes,
                             const std::size_t offset, const std::uint64_t value) {
            for (std::size_t index = 0; index < 8; ++index) {
                bytes[offset + index] = static_cast<std::byte>((value >> (index * 8)) & 0xff);
            }
        }

        void write_u32_bytes(const std::span<std::byte> bytes,
                             const std::size_t offset, const std::uint32_t value) {
            for (std::size_t index = 0; index < 4; ++index) {
                bytes[offset + index] = static_cast<std::byte>((value >> (index * 8)) & 0xff);
            }
        }

        lfs::core::Uuid read_uuid_bytes(const std::span<const std::byte> bytes,
                                        const std::size_t offset) {
            lfs::core::Uuid value;
            for (std::size_t index = 0; index < value.bytes.size(); ++index) {
                value.bytes[index] = std::to_integer<std::uint8_t>(bytes[offset + index]);
            }
            return value;
        }

        struct RepairCandidate {
            std::filesystem::path path;
            std::uint64_t generation = 0;
            bool preview_selection_recovered = false;
        };

        lfs::Result<RepairCandidate>
        find_repair_candidate(const std::filesystem::path& path) {
            constexpr std::uint64_t kScanLimit = 256ull * 1024 * 1024;
            std::error_code size_error;
            const auto size = std::filesystem::file_size(path, size_error);
            if (size_error) {
                return fail<RepairCandidate>(
                    lfs::ErrorCode::NotFound, path,
                    "The repair source could not be inspected.", size_error.message(),
                    "repair.scan");
            }
            if (size < APPEND_REGION_OFFSET + COMMIT_RECORD_BYTES) {
                return fail<RepairCandidate>(
                    lfs::ErrorCode::DataLoss, path,
                    "No valid project save was found for repair.",
                    "the file is shorter than one complete commit", "repair.candidate");
            }
            std::ifstream input(path, std::ios::binary);
            if (!input) {
                return fail<RepairCandidate>(
                    lfs::ErrorCode::PermissionDenied, path,
                    "The repair source could not be opened.", "open failed", "repair.scan");
            }
            std::array<std::byte, SUPERBLOCK_BYTES> superblock{};
            input.read(reinterpret_cast<char*>(superblock.data()), superblock.size());
            if (input.gcount() != static_cast<std::streamsize>(superblock.size())) {
                return fail<RepairCandidate>(
                    lfs::ErrorCode::DataLoss, path,
                    "No valid project save was found for repair.",
                    "the superblock is incomplete", "repair.candidate");
            }
            const auto project_uuid = read_uuid_bytes(superblock, 24);
            const auto file_uuid = read_uuid_bytes(superblock, 40);
            std::array<std::array<std::byte, HEAD_SLOT_BYTES>, 2> original_heads{};
            for (std::uint32_t slot = 0; slot < original_heads.size(); ++slot) {
                input.clear();
                input.seekg(static_cast<std::streamoff>(HEAD_SLOT_OFFSETS[slot]));
                input.read(reinterpret_cast<char*>(original_heads[slot].data()),
                           original_heads[slot].size());
                if (input.gcount() !=
                    static_cast<std::streamsize>(original_heads[slot].size())) {
                    original_heads[slot].fill(std::byte{0});
                }
            }
            const auto start = size > kScanLimit ? size - kScanLimit : APPEND_REGION_OFFSET;
            const auto scan_start = std::max(start, APPEND_REGION_OFFSET);
            input.seekg(static_cast<std::streamoff>(scan_start));
            constexpr std::array<std::byte, 8> magic{
                std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'C'},
                std::byte{'O'}, std::byte{'M'}, std::byte{'I'}, std::byte{'T'}};
            std::vector<std::byte> bytes(static_cast<std::size_t>(size - scan_start));
            input.read(reinterpret_cast<char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()));
            const auto count = static_cast<std::size_t>(input.gcount());
            std::vector<std::pair<std::uint64_t, std::uint64_t>> candidates;
            for (std::size_t index = 0; index + COMMIT_RECORD_BYTES <= count; ++index) {
                if (std::memcmp(bytes.data() + index, magic.data(), magic.size()) != 0) {
                    continue;
                }
                const auto offset = scan_start + index;
                if (offset % CHUNK_ALIGNMENT != 0) {
                    continue;
                }
                const auto record = std::span<const std::byte>(
                    bytes.data() + index, COMMIT_RECORD_BYTES);
                if (crc32c(0, record.data(), 252) != read_u32_bytes(record, 252) ||
                    read_uuid_bytes(record, 16) != project_uuid ||
                    read_u64_bytes(record, 176) != offset + COMMIT_RECORD_BYTES ||
                    read_u64_bytes(record, 136) < APPEND_REGION_OFFSET ||
                    read_u64_bytes(record, 136) + read_u64_bytes(record, 144) > offset ||
                    read_u64_bytes(record, 176) > size) {
                    continue;
                }
                const auto generation = read_u64_bytes(record, 64);
                candidates.emplace_back(offset, generation);
            }
            if (candidates.empty()) {
                return fail<RepairCandidate>(
                    lfs::ErrorCode::DataLoss, path,
                    "No valid project save was found for repair.",
                    "the bounded commit scan found no complete commit record",
                    "repair.candidate");
            }
            std::ranges::sort(candidates, [](const auto& lhs, const auto& rhs) {
                return lhs.second > rhs.second;
            });
            for (const auto& [candidate_offset, generation] : candidates) {
                auto temporary = path;
                temporary += ".repair-" + lfs::core::generate_uuid_v4().to_string() + ".tmp";
                std::error_code copy_error;
                if (!std::filesystem::copy_file(path, temporary,
                                                std::filesystem::copy_options::none,
                                                copy_error)) {
                    return fail<RepairCandidate>(
                        lfs::ErrorCode::PermissionDenied, temporary,
                        "The repair staging file could not be created.", copy_error.message(),
                        "repair.staging");
                }
                std::array<std::byte, COMMIT_RECORD_BYTES> commit{};
                input.clear();
                input.seekg(static_cast<std::streamoff>(candidate_offset));
                input.read(reinterpret_cast<char*>(commit.data()), commit.size());
                if (input.gcount() != static_cast<std::streamsize>(commit.size())) {
                    std::filesystem::remove(temporary);
                    continue;
                }
                std::array<std::byte, HEAD_SLOT_BYTES> head{};
                std::optional<std::optional<PreviewLocator>> recovered_preview;
                for (const auto& original_head : original_heads) {
                    const auto original = std::span<const std::byte>(original_head);
                    auto sanitized_head = original_head;
                    std::fill(sanitized_head.begin() + 128,
                              sanitized_head.begin() + 4092, std::byte{0});
                    constexpr std::array<std::byte, 8> head_magic{
                        std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'H'},
                        std::byte{'E'}, std::byte{'A'}, std::byte{'D'}, std::byte{0}};
                    if (std::memcmp(original.data(), head_magic.data(), head_magic.size()) != 0 ||
                        read_u32_bytes(original, 12) != HEAD_SLOT_BYTES ||
                        read_u64_bytes(original, 24) != generation ||
                        read_uuid_bytes(original, 32) != project_uuid ||
                        read_uuid_bytes(original, 48) != file_uuid ||
                        read_uuid_bytes(original, 64) != read_uuid_bytes(commit, 48) ||
                        read_u64_bytes(original, 80) != candidate_offset ||
                        read_u64_bytes(original, 88) != COMMIT_RECORD_BYTES ||
                        read_u64_bytes(original, 96) != read_u64_bytes(commit, 176) ||
                        read_u32_bytes(original, 104) != read_u32_bytes(commit, 252) ||
                        crc32c(0, sanitized_head.data(), 4092) !=
                            read_u32_bytes(original, 4092)) {
                        continue;
                    }
                    const auto preview_offset = read_u64_bytes(original, 112);
                    const auto preview_bytes = read_u32_bytes(original, 120);
                    const auto preview_format = read_u32_bytes(original, 124);
                    std::optional<PreviewLocator> selection;
                    if (preview_offset == 0 && preview_bytes == 0 && preview_format == 0) {
                        selection.reset();
                    } else if (preview_offset % CHUNK_ALIGNMENT == 0 && preview_bytes >= 1 &&
                               preview_bytes <= MAX_PREVIEW_BYTES &&
                               preview_format == static_cast<std::uint32_t>(PreviewFormat::Png) &&
                               preview_bytes <= read_u64_bytes(commit, 176) &&
                               preview_offset <= read_u64_bytes(commit, 176) - preview_bytes) {
                        selection = PreviewLocator{
                            .offset = preview_offset,
                            .bytes = preview_bytes,
                            .format = PreviewFormat::Png,
                        };
                    } else {
                        continue;
                    }
                    if (recovered_preview.has_value() && *recovered_preview != selection) {
                        recovered_preview.reset();
                        break;
                    }
                    recovered_preview.emplace(selection);
                }
                const auto fill_head = [&](const std::uint32_t slot,
                                           const std::uint64_t sequence) {
                    head.fill(std::byte{0});
                    constexpr std::array<std::byte, 8> head_magic{
                        std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'H'},
                        std::byte{'E'}, std::byte{'A'}, std::byte{'D'}, std::byte{0}};
                    std::memcpy(head.data(), head_magic.data(), head_magic.size());
                    write_u32_bytes(head, 8, slot);
                    write_u32_bytes(head, 12, HEAD_SLOT_BYTES);
                    write_u64_bytes(head, 16, sequence);
                    write_u64_bytes(head, 24, read_u64_bytes(commit, 64));
                    std::memcpy(head.data() + 32, superblock.data() + 24, 16);
                    std::memcpy(head.data() + 48, superblock.data() + 40, 16);
                    std::memcpy(head.data() + 64, commit.data() + 48, 16);
                    write_u64_bytes(head, 80, candidate_offset);
                    write_u64_bytes(head, 88, COMMIT_RECORD_BYTES);
                    write_u64_bytes(head, 96, read_u64_bytes(commit, 176));
                    write_u32_bytes(head, 104, read_u32_bytes(commit, 252));
                    if (recovered_preview.has_value() && recovered_preview->has_value()) {
                        write_u64_bytes(head, 112, (*recovered_preview)->offset);
                        write_u32_bytes(head, 120, (*recovered_preview)->bytes);
                        write_u32_bytes(
                            head, 124,
                            static_cast<std::uint32_t>((*recovered_preview)->format));
                    }
                    write_u32_bytes(head, 4092, crc32c(0, head.data(), 4092));
                };
                {
                    std::fstream output(temporary,
                                        std::ios::binary | std::ios::in | std::ios::out);
                    if (!output) {
                        std::filesystem::remove(temporary);
                        return fail<RepairCandidate>(
                            lfs::ErrorCode::PermissionDenied, temporary,
                            "The repair staging file could not be opened.", "open failed",
                            "repair.staging");
                    }
                    for (std::uint32_t slot = 0; slot < 2; ++slot) {
                        fill_head(slot, slot + 1);
                        output.seekp(static_cast<std::streamoff>(HEAD_SLOT_OFFSETS[slot]));
                        output.write(reinterpret_cast<const char*>(head.data()), head.size());
                    }
                    output.flush();
                    if (!output) {
                        std::filesystem::remove(temporary);
                        return fail<RepairCandidate>(
                            lfs::ErrorCode::PermissionDenied, temporary,
                            "The repair staging head could not be written.", "write failed",
                            "repair.staging");
                    }
                }
                auto reader = ProjectReader::open(temporary);
                if (reader && reader->verify_all()) {
                    return RepairCandidate{
                        .path = temporary,
                        .generation = generation,
                        .preview_selection_recovered = recovered_preview.has_value(),
                    };
                }
                std::filesystem::remove(temporary);
            }
            return fail<RepairCandidate>(
                lfs::ErrorCode::DataLoss, path,
                "No valid project save was found for repair.",
                "the bounded scan found commits but none passed index and live-header validation",
                "repair.candidate");
        }

    } // namespace

    lfs::Result<void> run_project_operation(
        const std::filesystem::path& path, const lfs::core::Uuid& expected_project,
        const lfs::core::Uuid& expected_commit, const std::function<void()>& operation) {
        auto lease = acquire_operation_lock(path);
        if (!lease)
            return lfs::Result<void>::failure(std::move(lease).error());
        auto identity = detail::ProjectPathIdentity::capture(path);
        if (!identity)
            return lfs::Result<void>::failure(std::move(identity).error());
        {
            auto reader = ProjectReader::open(path);
            if (!reader)
                return lfs::Result<void>::failure(std::move(reader).error());
            if (expected_project.is_nil() || reader->superblock().project_uuid != expected_project ||
                (!expected_commit.is_nil() && reader->commit().commit_uuid != expected_commit))
                return fail<void>(lfs::ErrorCode::FailedPrecondition, path,
                                  "The project changed before the operation started. Refresh Projects and try again.",
                                  "selected project or commit identity does not match the locked file", "operation.identity");
        }
        if (auto checked = identity->validate(); !checked)
            return checked;
        struct RestoreLease {
            const WriterLockLease* previous;
            const detail::ProjectPathIdentity* previous_identity;
            ~RestoreLease() {
                active_operation_lease = previous;
                detail::set_active_operation_identity(previous_identity);
            }
        } restore{active_operation_lease, detail::active_operation_identity()};
        active_operation_lease = &*lease;
        detail::set_active_operation_identity(&*identity);
        operation();
        return {};
    }

    static lfs::Result<std::filesystem::path> backup_locked_project_file(const std::filesystem::path& path) {
        if (const auto* planned = detail::active_operation_identity()) {
            if (auto checked = planned->validate(); !checked)
                return std::move(checked).error();
        }
        auto source_identity = detail::ProjectPathIdentity::capture(path);
        if (!source_identity)
            return std::move(source_identity).error();
        auto reader = ProjectReader::open(path);
        if (!reader)
            return std::move(reader).error();
        auto paths = lfs::core::UserPaths::resolve();
        if (!paths)
            return std::move(paths).error();
        const auto recovery = paths->backupDir() / "contents" /
                              reader->superblock().project_uuid.to_string() /
                              (reader->commit().commit_uuid.to_string() + ".licht.bak");
        auto recovery_identity = detail::ProjectPathIdentity::capture(recovery);
        if (!recovery_identity)
            return std::move(recovery_identity).error();
        std::error_code error;
        if (std::filesystem::exists(recovery, error)) {
            auto backup = ProjectReader::open(recovery);
            if (backup && backup->superblock().project_uuid == reader->superblock().project_uuid &&
                backup->commit().commit_uuid == reader->commit().commit_uuid &&
                backup->superblock().file_uuid == reader->superblock().file_uuid)
                return recovery;
            return fail<std::filesystem::path>(lfs::ErrorCode::DataLoss, recovery,
                                               "The existing recovery copy is damaged or belongs to another file.",
                                               "backup identity does not match its source", "recovery_copy");
        }
        std::filesystem::create_directories(recovery.parent_path(), error);
        if (error)
            return fail<std::filesystem::path>(lfs::ErrorCode::PermissionDenied, recovery,
                                               "The recovery folder could not be created.", error.message(), "recovery_copy");
        const auto temporary = lfs::io::make_atomic_temp_output_path(recovery);
        if (!std::filesystem::copy_file(path, temporary, std::filesystem::copy_options::none, error)) {
            const auto reason = error.message();
            std::filesystem::remove(temporary, error);
            return fail<std::filesystem::path>(lfs::ErrorCode::PermissionDenied, temporary,
                                               "The recovery copy could not be created.", reason, "recovery_copy");
        }
        for (const auto* identity : {&*source_identity, &*recovery_identity}) {
            if (auto checked = identity->validate(); !checked) {
                std::filesystem::remove(temporary, error);
                return std::move(checked).error();
            }
        }
        auto copied = lfs::io::replace_atomic_output_file(temporary, recovery_identity->canonical_path, lfs::io::AtomicOutputDurability::Durable);
        if (!copied) {
            std::filesystem::remove(temporary, error);
            return fail<std::filesystem::path>(lfs::ErrorCode::PermissionDenied, recovery,
                                               "The recovery copy could not be saved.", copied.error().message, "recovery_copy");
        }
        return recovery;
    }

    lfs::Result<std::filesystem::path> backup_project_file(const std::filesystem::path& path) {
        auto lease = acquire_operation_lock(path);
        if (!lease)
            return std::move(lease).error();
        return backup_locked_project_file(path);
    }

    lfs::Result<void> restore_project_backup(const std::filesystem::path& path,
                                             const std::filesystem::path& backup,
                                             const lfs::core::Uuid& expected_project,
                                             const lfs::core::Uuid& expected_commit) {
        auto lease = acquire_operation_lock(path);
        if (!lease)
            return lfs::Result<void>::failure(std::move(lease).error());
        auto identity = detail::ProjectPathIdentity::capture(path);
        if (!identity)
            return lfs::Result<void>::failure(std::move(identity).error());
        auto backup_identity = detail::ProjectPathIdentity::capture(backup);
        if (!backup_identity)
            return lfs::Result<void>::failure(std::move(backup_identity).error());
        {
            auto current = ProjectReader::open(path);
            auto recovery = ProjectReader::open(backup);
            if (!current)
                return lfs::Result<void>::failure(std::move(current).error());
            if (!recovery)
                return lfs::Result<void>::failure(std::move(recovery).error());
            if (current->superblock().project_uuid != expected_project ||
                current->commit().commit_uuid != expected_commit ||
                recovery->superblock().project_uuid != expected_project)
                return fail<void>(lfs::ErrorCode::FailedPrecondition, path,
                                  "The project identity changed. The recovery copy was kept.",
                                  "recovery refused to overwrite a different project or commit", "recovery.identity");
            if (auto verified = recovery->verify_all(); !verified)
                return lfs::Result<void>::failure(std::move(verified).error());
        }
        const auto temporary = lfs::io::make_atomic_temp_output_path(identity->canonical_path);
        std::error_code error;
        if (!std::filesystem::copy_file(backup, temporary, std::filesystem::copy_options::none, error)) {
            const auto reason = error.message();
            std::filesystem::remove(temporary, error);
            return fail<void>(lfs::ErrorCode::PermissionDenied, path,
                              "The recovery copy could not be restored.", reason, "recovery.copy");
        }
        if (auto checked = identity->validate(); !checked) {
            std::filesystem::remove(temporary, error);
            return checked;
        }
        if (auto checked = backup_identity->validate(); !checked) {
            std::filesystem::remove(temporary, error);
            return checked;
        }
        auto restored = lfs::io::replace_atomic_output_file(temporary, identity->canonical_path, lfs::io::AtomicOutputDurability::Durable);
        if (!restored) {
            std::filesystem::remove(temporary, error);
            return fail<void>(lfs::ErrorCode::PermissionDenied, path,
                              "The recovery copy could not replace the project.", restored.error().message, "recovery.replace");
        }
        return {};
    }

    lfs::Result<ProjectInspectorCard>
    restore_save(const std::filesystem::path& path,
                 const std::uint64_t generation,
                 const std::filesystem::path& requested_destination) {
        const bool replace_source = same_path(path, requested_destination);
        auto destination = requested_destination;
        if (replace_source) {
            destination = path;
            destination += ".restore-" + lfs::core::generate_uuid_v4().to_string() + ".tmp";
        }
        auto identity = detail::ProjectPathIdentity::capture(path);
        if (!identity)
            return std::move(identity).error();
        if (path.empty() || requested_destination.empty()) {
            return fail<ProjectInspectorCard>(
                lfs::ErrorCode::InvalidArgument, destination,
                "The restore paths are invalid.",
                "source and destination must be non-empty",
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
        auto destination_identity = detail::ProjectPathIdentity::capture(destination);
        if (!destination_identity)
            return std::move(destination_identity).error();
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
        if (const auto* row = reader->find(FOURCC_PROJ, reader->superblock().project_uuid)) {
            auto bytes = reader->read_chunk(*row);
            if (!bytes)
                return std::move(bytes).error();
            auto project = ProjectChapter::from_bytes(*bytes);
            if (!project)
                return std::move(project).error();
            auto pending = project->dom().get_json("contents_removals");
            if (pending && pending->is_object() && pending->value("file_uuid", std::string{}) == reader->superblock().file_uuid.to_string() &&
                pending->contains("rows") && (*pending)["rows"].is_array()) {
                for (const auto& removed : (*pending)["rows"]) {
                    if (removed.is_object() && removed.value("id", std::string{}) == "save:" + std::to_string(generation))
                        return fail<ProjectInspectorCard>(lfs::ErrorCode::FailedPrecondition, path,
                                                          "This save has been removed.", "the removal is pending compaction", "restore.save");
                }
            }
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
        const ChunkInfo* selected_preview_row = nullptr;
        if (selected_reader->preview().has_value()) {
            const auto& locator = *selected_reader->preview();
            const auto selected = std::ranges::find_if(
                selected_rows, [&locator](const auto& entry) {
                    const auto& row = entry.second;
                    return row.key.fourcc == FOURCC_THMB &&
                           row.payload_offset == locator.offset &&
                           row.stored_bytes == locator.bytes;
                });
            if (selected == selected_rows.end()) {
                return fail<ProjectInspectorCard>(
                    lfs::ErrorCode::DataLoss, path,
                    "The selected save has an invalid preview reference.",
                    "the published historical preview locator does not match a live THMB row",
                    "restore.preview");
            }
            selected_preview_row = &selected->second;
        } else {
            for (const auto& [key, row] : selected_rows) {
                if (key.fourcc != FOURCC_THMB)
                    continue;
                (void)row;
                return fail<ProjectInspectorCard>(
                    lfs::ErrorCode::FailedPrecondition, path,
                    "The selected save's preview is ambiguous.",
                    "its historical head locator is no longer published while THMB rows remain",
                    "restore.preview");
            }
        }
        const auto is_selected_preview = [selected_preview_row](const ChunkInfo& row) {
            return selected_preview_row != nullptr &&
                   row.key == selected_preview_row->key &&
                   row.payload_offset == selected_preview_row->payload_offset &&
                   row.stored_bytes == selected_preview_row->stored_bytes;
        };
        auto project_bytes = selected_reader->read_chunk(project_row->second);
        if (!project_bytes) {
            return std::move(project_bytes).error();
        }
        auto project = ProjectChapter::from_bytes(*project_bytes);
        if (!project) {
            return std::move(project).error();
        }
        if (replace_source) {
            // Restore in place appends the selected state. The catalog identity
            // and other saves stay available until the user removes/compacts them.
            auto details = inspect_project_details(path);
            if (!details)
                return std::move(details).error();
            auto current = ProjectDocument::open(path);
            if (!current)
                return std::move(current).error();
            if (auto changed = project->dom().set_json("contents_removals", pending_contents(*current)); !changed)
                return std::move(changed).error();
            if (auto changed = project->dom().set_json("contents_edit", {
                                                                            {"file_uuid", reader->superblock().file_uuid.to_string()},
                                                                            {"first_generation", reader->commit().generation + 1},
                                                                            {"last_generation", reader->commit().generation + 1},
                                                                            {"operation", "restored"},
                                                                            {"source_save", save_origin(details->save_history[generation - 1])},
                                                                        });
                !changed)
                return std::move(changed).error();
            const auto restored_project = project->to_bytes();
            std::uint64_t bytes = restored_project.size();
            for (const auto& [key, row] : selected_rows) {
                if (key != project_row->first) {
                    if (row.stored_bytes > std::numeric_limits<std::uint64_t>::max() - bytes)
                        return fail<ProjectInspectorCard>(lfs::ErrorCode::ResourceExhausted, path,
                                                          "The restored project is too large.", "restore payload byte total overflowed", "restore.preflight");
                    bytes += row.stored_bytes;
                }
            }
            auto recovery = backup_locked_project_file(path);
            if (!recovery)
                return std::move(recovery).error();
            auto appended = ProjectWriter::append(path, AppendOptions{.writer_lock_lease = *source_lock});
            if (!appended)
                return std::move(appended).error();
            auto writer = std::move(*appended);
            const auto& selected_commit = lineage[generation - 1];
            if (auto planned = writer.plan_commit(CommitOptions{
                    .kind = CommitKind::Explicit,
                    .snapshot_uuid = selected_commit.snapshot_uuid,
                    .min_reader_version = reader->commit().min_reader_version,
                    .min_safe_writer_version = reader->commit().min_safe_writer_version,
                    .extra_reader_capabilities = reader->commit().required_reader_capabilities,
                    .extra_writer_capabilities = reader->commit().required_writer_capabilities,
                });
                !planned)
                return std::move(planned).error();
            if (auto planned = writer.preflight(bytes); !planned)
                return std::move(planned).error();
            for (const auto& [key, row] : selected_rows) {
                if (key == project_row->first) {
                    if (auto written = writer.write_chunk(key, restored_project); !written)
                        return std::move(written).error();
                } else if (is_selected_preview(row)) {
                    if (auto copied = writer.copy_chunk_verbatim(*selected_reader, row); !copied)
                        return std::move(copied).error();
                } else if (auto copied = writer.copy_chunk_verbatim(*selected_reader, row); !copied) {
                    return std::move(copied).error();
                }
            }
            for (const auto& row : reader->chunks()) {
                if (row.is_live() && !selected_rows.contains(row.key)) {
                    if (auto erased = writer.erase(row.key); !erased)
                        return std::move(erased).error();
                }
            }
            if (auto committed = writer.commit(); !committed)
                return std::move(committed).error();
            return inspect_after_save(path);
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
        if (auto checked = identity->validate(); !checked)
            return std::move(checked).error();
        if (auto checked = destination_identity->validate(); !checked)
            return std::move(checked).error();
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
            } else if (is_selected_preview(row)) {
                if (new_key != old_key) {
                    auto preview = selected_reader->read_chunk(row);
                    if (!preview) {
                        return std::move(preview).error();
                    }
                    if (auto written = writer.set_preview(*preview); !written) {
                        return std::move(written).error();
                    }
                } else if (auto copied = writer.copy_chunk_verbatim(*selected_reader, row);
                           !copied) {
                    return std::move(copied).error();
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
        if (replace_source) {
            auto recovery = backup_locked_project_file(path);
            if (!recovery)
                return std::move(recovery).error();
            if (auto checked = identity->validate(); !checked)
                return std::move(checked).error();
            if (auto replaced = lfs::io::replace_atomic_output_file(destination, identity->canonical_path, lfs::io::AtomicOutputDurability::Durable); !replaced)
                return fail<ProjectInspectorCard>(lfs::ErrorCode::PermissionDenied, path,
                                                  "The restored save could not replace the project.", replaced.error().message, "restore.replace");
            return inspect_after_save(path);
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
        auto recovery = backup_locked_project_file(path);
        if (!recovery)
            return std::move(recovery).error();
        if (auto marked = mark_contents_edit(*document, "checkpoint_rebound"); !marked)
            return std::move(marked).error();
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.regenerate_dataset_preview = false;
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
        auto lease = acquire_operation_lock(path);
        if (!lease)
            return std::move(lease).error();
        auto document = ProjectDocument::open(path);
        if (!document)
            return std::move(document).error();
        if (auto marked = mark_contents_edit(*document, "compacted"); !marked)
            return std::move(marked).error();
        CompactionOptions options;
        options.writer_lock_lease = *lease;
        options.progress = std::move(progress);
        options.cancel = std::move(cancel);
        options.project_chapter_override = document->project().to_bytes();
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

    lfs::Result<ProjectReducePlan>
    plan_reduce_size(const std::filesystem::path& path) {
        if (path.empty()) {
            return fail<ProjectReducePlan>(
                lfs::ErrorCode::InvalidArgument, path,
                "The project path is empty.", "reduce_size requires a path", "path");
        }
        return make_reduce_plan(path);
    }

    lfs::Result<ProjectReduceResult>
    reduce_size(const std::filesystem::path& path,
                const bool drop_unbound_checkpoints,
                const bool drop_embedded_dataset,
                ProjectOperationProgress progress,
                ProjectOperationCancel cancel,
                const ProjectReduceSelection& selection) {
        auto identity = detail::ProjectPathIdentity::capture(path);
        if (!identity)
            return std::move(identity).error();
        auto plan = make_reduce_plan(path);
        if (!plan) {
            return std::move(plan).error();
        }
        if (drop_embedded_dataset && !plan->drop_embedded_dataset.allowed) {
            return fail<ProjectReduceResult>(
                lfs::ErrorCode::FailedPrecondition, path,
                "The embedded dataset cannot be removed safely.",
                "every required image, normal, sparse, mask, depth, and metadata input "
                "needs a matching external fingerprint",
                "reduce.drop_embedded_dataset");
        }
        if (cancel && cancel()) {
            return fail<ProjectReduceResult>(
                lfs::ErrorCode::Cancelled, path,
                "Project reduction was canceled.", "the caller canceled before the writer lock",
                "reduce.cancel");
        }
        auto lease = acquire_operation_lock(path);
        if (!lease) {
            return std::move(lease).error();
        }
        {
            auto current = ProjectReader::open(path);
            if (!current) {
                return std::move(current).error();
            }
            if (current->commit().commit_uuid != plan->input_commit_uuid) {
                return fail<ProjectReduceResult>(
                    lfs::ErrorCode::FailedPrecondition, path,
                    "The project changed before reduction started.",
                    "the planned commit is no longer the current head", "reduce.commit_uuid");
            }
        }
        auto recovery_result = backup_locked_project_file(path);
        if (!recovery_result)
            return std::move(recovery_result).error();
        const auto recovery = *recovery_result;
        if (progress) {
            progress(0.05F, "Preparing project reduction");
        }
        if (auto checked = identity->validate(); !checked)
            return std::move(checked).error();
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        auto bound = document->bound_checkpoint_uuid();
        if (!bound) {
            return std::move(bound).error();
        }
        ProjectReduceResult result;
        result.recovery_copy = recovery;
        auto pending = pending_contents(*document);
        auto remember = [&](const std::string& id, const std::string& kind, const std::uint64_t bytes,
                            JsonChapterDom::Json extra = JsonChapterDom::Json::object()) {
            extra["id"] = id;
            extra["kind"] = kind;
            extra["bytes"] = bytes;
            pending["rows"].push_back(std::move(extra));
        };
        if (selection.save_generation) {
            const auto& lineage = document->source_reader()->lineage();
            if (selection.save_generation >= lineage.size()) {
                return fail<ProjectReduceResult>(lfs::ErrorCode::InvalidArgument, path,
                                                 "The current save cannot be removed.", "only older saves can be removed", "reduce.save");
            }
            for (const auto& row : pending["rows"]) {
                if (row.value("id", std::string{}) == "save:" + std::to_string(selection.save_generation)) {
                    return fail<ProjectReduceResult>(lfs::ErrorCode::InvalidArgument, path,
                                                     "This save was already removed.", "the removal is pending compaction", "reduce.save");
                }
            }
            auto history = document->source_reader()->lineage_chunks();
            if (!history)
                return std::move(history).error();
            std::uint64_t bytes = 0;
            // Only count old payload bytes that are no longer part of the current save.
            for (const auto& row : (*history)[selection.save_generation - 1]) {
                const auto* live = document->source_reader()->find(row.key);
                if (row.is_live() && (!live || live->header_offset != row.header_offset))
                    bytes += row_occupied_bytes(row);
            }
            const auto& saved = lineage[selection.save_generation - 1];
            remember("save:" + std::to_string(selection.save_generation), "save", bytes,
                     {{"generation", selection.save_generation}, {"date", saved.wallclock_unix_ns}});
        }
        if (selection.checkpoint) {
            const auto checkpoint = std::ranges::find(plan->retained_checkpoints, *selection.checkpoint,
                                                      &ProjectReduceCheckpoint::instance_uuid);
            if (checkpoint == plan->retained_checkpoints.end()) {
                return fail<ProjectReduceResult>(lfs::ErrorCode::NotFound, path,
                                                 "The checkpoint was not found.", "checkpoint is no longer retained", "reduce.checkpoint");
            }
            if (*bound && **bound == *selection.checkpoint) {
#ifdef LFS_FORMAT_TEST_TARGET
                return fail<ProjectReduceResult>(lfs::ErrorCode::Unsupported, path,
                                                 "Removing this checkpoint requires the full application.", "model conversion is unavailable", "reduce.checkpoint");
#else
                auto training = document->scene_graph().training_model_uuid();
                if (!training)
                    return std::move(training).error();
                auto node = document->scene_graph().find(**training);
                if (!node)
                    return std::move(node).error();
                auto model = load_export_payload(*document, **node, *(*node)->payload, cancel);
                if (!model)
                    return std::move(model).error();
                auto splat = SplatChapterPayload::capture(**model, SplatSourceKind::Generated, false);
                if (!splat)
                    return std::move(splat).error();
                auto fingerprint = fingerprint_path(path);
                if (!fingerprint)
                    return std::move(fingerprint).error();
                if (auto updated = document->edit_project().upsert_embed_decision(EmbedDecision{
                        .uuid = lfs::core::generate_uuid_v4(),
                        .node_uuid = (*node)->uuid,
                        .payload_fourcc = "SPLT",
                        .decision = "embedded",
                        .reference_uuid = std::nullopt,
                        .reason = "Visible model preserved when removing its training checkpoint"});
                    !updated)
                    return std::move(updated).error();
                if (auto updated = document->edit_project().upsert_embedded_payload_provenance(EmbeddedPayloadProvenance{
                        .uuid = lfs::core::generate_uuid_v4(),
                        .node_uuid = (*node)->uuid,
                        .fourcc = "SPLT",
                        .import_locator = ReferenceLocator{.preferred = lfs::core::path_to_utf8(recovery), .base = LocatorBase::Absolute, .absolute_fallback = std::nullopt},
                        .import_fingerprint = *fingerprint,
                        .content_xxh3_128 = xxh3_128(splat->bytes())});
                    !updated)
                    return std::move(updated).error();
                (*node)->payload = PayloadBinding{"SPLT", (*node)->uuid, std::nullopt, "generated"};
                if (auto updated = document->edit_scene_graph().set_training_model_uuid(std::nullopt); !updated)
                    return std::move(updated).error();
                if (auto updated = document->edit_scene_graph().upsert_node(**node); !updated)
                    return std::move(updated).error();
                if (auto updated = document->set_splat((*node)->uuid, std::move(*splat)); !updated)
                    return std::move(updated).error();
#endif
            }
            if (document->remove_checkpoint(*selection.checkpoint))
                ++result.checkpoints_removed;
            remember("checkpoint:" + selection.checkpoint->to_string(), "checkpoint", checkpoint->bytes,
                     {{"iteration", checkpoint->iteration}});
        }
        if (selection.metrics) {
            std::uint64_t bytes = 0;
            for (const auto& row : document->source_reader()->chunks())
                if (row.is_live() && row.key.fourcc == FOURCC_METR)
                    bytes += row_occupied_bytes(row);
            const auto& metrics = document->metrics();
            const auto samples = metrics.loss_history.size() + metrics.psnr_history.size();
            document->edit_metrics() = MetricsChapter{};
            remember("metrics", "metrics", bytes, {{"samples", samples}});
        }
        if (selection.thumbnail) {
            std::uint64_t bytes = 0;
            for (const auto& row : document->source_reader()->chunks())
                if (row.is_live() && row.key.fourcc == FOURCC_THMB)
                    bytes += row_occupied_bytes(row);
            remember("thumbnail", "thumbnail", bytes);
        }
        if (drop_embedded_dataset) {
            const auto images = std::ranges::count(plan->embedded_dataset, "image", &ProjectReduceDatasetPayload::kind);
            remember("dataset:embedded", "dataset", plan->drop_embedded_dataset.reclaimable_bytes, {{"images", images}});
        }
        if (!selection.compact) {
            if (auto updated = document->edit_project().dom().set_json("contents_removals", pending); !updated)
                return std::move(updated).error();
        }
        if (drop_unbound_checkpoints) {
            for (const auto& uuid : document->checkpoint_uuids()) {
                if (*bound && **bound == uuid) {
                    continue;
                }
                if (document->remove_checkpoint(uuid)) {
                    ++result.checkpoints_removed;
                }
            }
        }
        const auto operation = drop_embedded_dataset       ? "dataset_removed"
                               : selection.save_generation ? "save_removed"
                               : selection.checkpoint      ? "checkpoint_removed"
                               : selection.thumbnail       ? "thumbnail_removed"
                               : selection.metrics         ? "metrics_removed"
                                                           : "changed";
        if (auto marked = mark_contents_edit(*document, operation, drop_embedded_dataset ? 3 : 1); !marked)
            return std::move(marked).error();
        ProjectDocumentSaveOptions save_options;
        save_options.commit.kind = CommitKind::Explicit;
        save_options.regenerate_dataset_preview = false;
        save_options.writer_lock_lease = *lease;
        save_options.remove_preview = selection.thumbnail;
        {
            auto saved = document->save(path, save_options);
            if (!saved) {
                return std::move(saved).error();
            }
        }
        if (drop_embedded_dataset) {
            EmbeddedDatasetManifest empty_manifest;
            empty_manifest.schema_version = 1;
            empty_manifest.complete = false;
            auto embedded = document->embed_dataset_batch(
                empty_manifest, {}, save_options);
            if (!embedded) {
                return std::move(embedded).error();
            }
            document = ProjectDocument::open(path);
            if (!document) {
                return std::move(document).error();
            }
            auto dataset_uuid = document->project().dataset_reference();
            if (!dataset_uuid) {
                return std::move(dataset_uuid).error();
            }
            if (!*dataset_uuid) {
                return fail<ProjectReduceResult>(
                    lfs::ErrorCode::DataLoss, path,
                    "The embedded dataset has no external reference.",
                    "validated external replacement disappeared before the reduction save",
                    "reduce.dataset_reference");
            }
            auto decisions = document->project().embed_decisions();
            if (!decisions) {
                return std::move(decisions).error();
            }
            for (auto& decision : *decisions) {
                if (decision.decision == "embedded" && decision.payload_fourcc == "DSRC") {
                    decision.decision = "external";
                    decision.reference_uuid = *dataset_uuid;
                    decision.reason = "external dataset fingerprint validated before reduction";
                    if (auto updated = document->edit_project().upsert_embed_decision(decision);
                        !updated) {
                        return std::move(updated).error();
                    }
                }
            }
            auto record = document->references().find(**dataset_uuid);
            if (!record) {
                return std::move(record).error();
            }
            if (*record) {
                auto refreshed = **record;
                refreshed.unresolved = false;
                if (auto updated = document->edit_references().upsert(refreshed);
                    !updated) {
                    return std::move(updated).error();
                }
            }
            document->edit_parameters().clear_embedded_dataset();
            auto saved = document->save(path, save_options);
            if (!saved) {
                return std::move(saved).error();
            }
        }
        if (!selection.compact) {
            auto details = inspect_project_details(path);
            if (!details)
                return std::move(details).error();
            result.card = details->card;
            for (const auto& payload : plan->embedded_dataset) {
                if (!drop_embedded_dataset)
                    break;
                if (payload.kind == "image")
                    ++result.dataset_images_removed;
                if (payload.kind == "normal")
                    ++result.dataset_normals_removed;
                if (payload.kind == "sparse")
                    ++result.dataset_sparse_removed;
            }
            if (progress)
                progress(1.0F, "Removal saved; compact to free space");
            return result;
        }
        save_options.writer_lock_lease.reset();
        if (progress) {
            progress(0.35F, "Compacting reduced project");
        }
        CompactionOptions compact_options;
        compact_options.writer_lock_lease = *lease;
        compact_options.progress = [progress](const float value, const std::string& stage) {
            if (progress) {
                progress(0.35F + value * 0.55F, stage);
            }
        };
        compact_options.cancel = cancel;
        auto compacted = ProjectWriter::compact(path, compact_options);
        if (!compacted) {
            if (compacted.error().code() == lfs::ErrorCode::Unavailable) {
                return fail<ProjectReduceResult>(
                    lfs::ErrorCode::Unavailable, path,
                    std::string(WRITER_LOCK_MESSAGE),
                    "the project writer lock is held by another process", "writer_lock");
            }
            return std::move(compacted).error();
        }
        auto details = inspect_project_details(path);
        if (!details) {
            return std::move(details).error();
        }
        auto after_bound = details->scene_graph.training_node_id;
        if (*bound && (!after_bound ||
                       !std::ranges::any_of(details->retained_checkpoints,
                                            [&](const auto& checkpoint) {
                                                return checkpoint.binds_scene_graph &&
                                                       checkpoint.instance_uuid == **bound;
                                            }))) {
            return fail<ProjectReduceResult>(
                lfs::ErrorCode::DataLoss, path,
                "The reduced project lost its bound checkpoint.",
                "post-reduction details validation did not find the SCNG-bound checkpoint",
                "reduce.post_check");
        }
        result.card = details->card;
        const auto after_size = details->card.physical_file_size;
        result.bytes_reclaimed = after_size < plan->physical_size
                                     ? plan->physical_size - after_size
                                     : 0;
        for (const auto& payload : plan->embedded_dataset) {
            if (drop_embedded_dataset) {
                if (payload.kind == "image")
                    ++result.dataset_images_removed;
                if (payload.kind == "normal")
                    ++result.dataset_normals_removed;
                if (payload.kind == "sparse")
                    ++result.dataset_sparse_removed;
            }
        }
        if (progress) {
            progress(1.0F, "Project reduction complete");
        }
        return result;
    }

#ifdef LFS_FORMAT_TEST_TARGET
    lfs::Result<DatasetEmbedResult>
    embed_dataset_file(const std::filesystem::path& path,
                       ProjectOperationProgress,
                       ProjectOperationCancel) {
        return fail<DatasetEmbedResult>(
            lfs::ErrorCode::Unsupported, path,
            "Dataset embedding is unavailable in the CPU format-test composition.",
            "the full I/O library is required for closed-file dataset embedding",
            "dataset_embed.composition");
    }
#else
    lfs::Result<DatasetEmbedResult>
    embed_dataset_file(const std::filesystem::path& path,
                       ProjectOperationProgress progress,
                       ProjectOperationCancel cancel) {
        auto lease = acquire_operation_lock(path);
        if (!lease) {
            return std::move(lease).error();
        }
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        auto dataset_uuid = document->project().dataset_reference();
        if (!dataset_uuid) {
            return std::move(dataset_uuid).error();
        }
        if (!*dataset_uuid) {
            return fail<DatasetEmbedResult>(
                lfs::ErrorCode::NotFound, path,
                "The project has no dataset reference.",
                "REFS has no dataset record selected by PROJ", "dataset.reference");
        }
        auto root = resolve_path_reference(
            document->references(), path.parent_path(), **dataset_uuid);
        if (!root || !std::filesystem::is_directory(*root)) {
            return fail<DatasetEmbedResult>(
                lfs::ErrorCode::NotFound, path,
                "The external dataset folder could not be found.",
                "the dataset reference is unreachable", "dataset.path");
        }
        auto snapshot = document->parameters().snapshot();
        if (!snapshot) {
            return std::move(snapshot).error();
        }
        std::string images_folder;
        auto files = discover_dataset_files(
            *root, snapshot->dataset.images, images_folder);
        EmbeddedDatasetManifest manifest{
            .schema_version = 1,
            .images_folder = images_folder,
            .complete = true,
            .entries = {},
        };
        std::vector<DatasetEmbedSource> sources;
        std::uint64_t total_bytes = 0;
        for (std::size_t index = 0; index < files.size(); ++index) {
            if (cancel && cancel()) {
                return fail<DatasetEmbedResult>(
                    lfs::ErrorCode::Cancelled, path,
                    "Dataset embedding was canceled.", "the caller canceled while scanning",
                    "dataset_embed.cancel");
            }
            const auto& [source_path, kind] = files[index];
            std::error_code error;
            const auto bytes = std::filesystem::file_size(source_path, error);
            if (error) {
                return fail<DatasetEmbedResult>(
                    lfs::ErrorCode::DataLoss, source_path,
                    "A dataset file could not be inspected.", error.message(), "dataset.file");
            }
            auto hash = hash_dataset_file(source_path);
            if (!hash) {
                return std::move(hash).error();
            }
            EmbeddedDatasetEntry entry{
                .rel_path = lfs::core::path_to_generic_utf8(source_path.lexically_relative(*root)),
                .kind = kind,
                .chunk_uuid = lfs::core::generate_uuid_v4(),
                .bytes = bytes,
                .xxh3_128 = *hash,
            };
            manifest.entries.push_back(entry);
            sources.push_back(DatasetEmbedSource{.entry = entry, .source_path = source_path});
            total_bytes += bytes;
            if (progress) {
                progress(files.empty() ? 0.2F
                                       : 0.2F + 0.3F * static_cast<float>(index + 1) /
                                                    static_cast<float>(files.size()),
                         "Hashing dataset files");
            }
        }
        std::error_code space_error;
        const auto available = std::filesystem::space(path.parent_path(), space_error).available;
        if (space_error || available < total_bytes + 64ull * 1024 * 1024) {
            return fail<DatasetEmbedResult>(
                lfs::ErrorCode::ResourceExhausted, path,
                "There is not enough disk space to embed the dataset.",
                space_error ? space_error.message() : "dataset bytes plus writer reserve do not fit",
                "dataset_embed.disk_space");
        }
        if (cancel && cancel()) {
            return fail<DatasetEmbedResult>(
                lfs::ErrorCode::Cancelled, path,
                "Dataset embedding was canceled.", "the caller canceled before writing",
                "dataset_embed.cancel");
        }
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.regenerate_dataset_preview = false;
        options.writer_lock_lease = *lease;
        options.disk_reserve_bytes = 64ull * 1024 * 1024;
        if (auto marked = mark_contents_edit(*document, "dataset_embedded", 2); !marked)
            return std::move(marked).error();
        if (auto marked = document->save(path, options); !marked)
            return std::move(marked).error();
        auto saved = document->embed_dataset_batch(manifest, sources, options);
        if (!saved) {
            return std::move(saved).error();
        }
        auto card = inspect_after_save(path);
        if (!card) {
            return std::move(card).error();
        }
        DatasetEmbedResult result{
            .card = *card,
            .bytes_embedded = total_bytes,
        };
        for (const auto& entry : manifest.entries) {
            if (entry.kind == "image")
                ++result.images_embedded;
            if (entry.kind == "normal")
                ++result.normals_embedded;
            if (entry.kind == "sparse")
                ++result.sparse_embedded;
        }
        if (progress) {
            progress(1.0F, "Dataset embedding complete");
        }
        return result;
    }
#endif

    lfs::Result<DatasetReferenceResult>
    set_dataset_reference(const std::filesystem::path& path,
                          const std::filesystem::path& dataset_dir,
                          const bool accept_content_change) {
        if (dataset_dir.empty() || !std::filesystem::is_directory(dataset_dir)) {
            return fail<DatasetReferenceResult>(
                lfs::ErrorCode::NotFound, dataset_dir,
                "The dataset folder could not be found.",
                "set_dataset_reference requires an existing directory", "dataset.path");
        }
        auto lease = acquire_operation_lock(path);
        if (!lease) {
            return std::move(lease).error();
        }
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        auto current = document->project().dataset_reference();
        if (!current) {
            return std::move(current).error();
        }
        bool content_replaced = false;
        if (*current) {
            auto existing = document->references().find(**current);
            if (!existing) {
                return std::move(existing).error();
            }
            if (*existing) {
                const auto resolved = resolve_path_reference(
                    document->references(), path.parent_path(), **current);
                if (resolved && same_path(*resolved, dataset_dir)) {
                    auto check = check_fingerprint(dataset_dir, (*existing)->fingerprint);
                    if (check && check->matches()) {
                        content_replaced = false;
                    }
                } else {
                    content_replaced = true;
                }
                if (content_replaced && !accept_content_change) {
                    return fail<DatasetReferenceResult>(
                        lfs::ErrorCode::FailedPrecondition, dataset_dir,
                        "The selected dataset has different content.",
                        "pass accept_content_change to deliberately replace the dataset reference",
                        "dataset.fingerprint");
                }
                ReferenceLocator locator{
                    .preferred = lfs::core::path_to_utf8(std::filesystem::absolute(dataset_dir)),
                    .base = LocatorBase::Absolute,
                    .absolute_fallback = lfs::core::path_to_utf8(std::filesystem::absolute(dataset_dir)),
                };
                auto relinked = document->edit_references().relink(
                    **current, locator, dataset_dir, accept_content_change);
                if (!relinked) {
                    return std::move(relinked).error();
                }
            } else {
                if (auto record = upsert_path_reference(
                        document->edit_references(), path.parent_path(), dataset_dir,
                        "dataset", "dataset", **current);
                    !record) {
                    return std::move(record).error();
                }
            }
        } else {
            auto record = upsert_path_reference(
                document->edit_references(), path.parent_path(), dataset_dir,
                "dataset", "dataset");
            if (!record) {
                return std::move(record).error();
            }
            current = *record;
        }
        if (!current || !*current) {
            return fail<DatasetReferenceResult>(
                lfs::ErrorCode::Internal, path,
                "The dataset reference could not be assigned.",
                "reference UUID was not produced", "dataset.reference_uuid");
        }
        if (auto selected = document->edit_project().set_dataset_reference(*current);
            !selected) {
            return std::move(selected).error();
        }
        if (auto marked = mark_contents_edit(*document, "dataset_located"); !marked)
            return std::move(marked).error();
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.regenerate_dataset_preview = false;
        options.writer_lock_lease = *lease;
        auto saved = document->save(path, options);
        if (!saved) {
            return std::move(saved).error();
        }
        auto card = inspect_after_save(path);
        if (!card) {
            return std::move(card).error();
        }
        return DatasetReferenceResult{
            .card = *card,
            .reference_uuid = **current,
            .content_replaced = content_replaced,
        };
    }

#ifdef LFS_FORMAT_TEST_TARGET
    lfs::Result<ProjectExportResult>
    export_project_as(const std::filesystem::path& path,
                      const ProjectExportFormat,
                      const std::filesystem::path&,
                      ProjectOperationProgress,
                      ProjectOperationCancel) {
        return fail<ProjectExportResult>(
            lfs::ErrorCode::Unsupported, path,
            "Project export is unavailable in the CPU format-test composition.",
            "the full I/O library is required for closed-file splat export",
            "export.composition");
    }
#else
    lfs::Result<ProjectExportResult>
    export_project_as(const std::filesystem::path& path,
                      const ProjectExportFormat format,
                      const std::filesystem::path& destination,
                      ProjectOperationProgress progress,
                      ProjectOperationCancel cancel) {
        if (path.empty() || destination.empty() || same_path(path, destination)) {
            return fail<ProjectExportResult>(
                lfs::ErrorCode::InvalidArgument, destination,
                "The export paths are invalid.",
                "source and destination must be non-empty and different", "export.path");
        }
        if (cancel && cancel()) {
            return fail<ProjectExportResult>(
                lfs::ErrorCode::Cancelled, path, "Project export was canceled.",
                "the caller canceled before loading", "export.cancel");
        }
        auto source_reader = ProjectReader::open(path);
        if (!source_reader) {
            return std::move(source_reader).error();
        }
        const auto source_commit = source_reader->commit().commit_uuid;
        std::error_code space_error;
        const auto available = std::filesystem::space(destination.parent_path(), space_error).available;
        if (space_error || available < source_reader->physical_file_size() / 4 + 64ull * 1024 * 1024) {
            return fail<ProjectExportResult>(
                lfs::ErrorCode::ResourceExhausted, destination,
                "There is not enough disk space for the export.",
                space_error ? space_error.message() : "memory and output reserve do not fit",
                "export.disk_space");
        }
        if (progress) {
            progress(0.05F, "Loading visible splats");
        }
        auto scene = load_export_scene(path, cancel);
        if (!scene) {
            return std::move(scene).error();
        }
        auto merged = lfs::core::Scene::mergeSplatsWithTransforms(scene->splats);
        if (!merged || merged->size() == 0) {
            return fail<ProjectExportResult>(
                lfs::ErrorCode::DataLoss, path,
                "The project did not produce any exportable splats.",
                "visible splat materialization returned no gaussians", "export.splats");
        }
        auto latest = ProjectReader::open(path);
        if (!latest) {
            return std::move(latest).error();
        }
        if (latest->commit().commit_uuid != source_commit) {
            return fail<ProjectExportResult>(
                lfs::ErrorCode::FailedPrecondition, path,
                "The project changed during export preparation.",
                "the export commit guard no longer matches", "export.commit_uuid");
        }
        const auto report = [progress](const float value, const std::string& stage) {
            if (progress) {
                progress(0.1F + value * 0.85F, stage);
            }
            return true;
        };
        switch (format) {
        case ProjectExportFormat::Ply:
            if (auto saved = save_ply(*merged, PlySaveOptions{
                                                   .output_path = destination,
                                                   .binary = true,
                                                   .async = false,
                                                   .progress_callback = report,
                                               });
                !saved) {
                return fail<ProjectExportResult>(
                    lfs::ErrorCode::Unavailable, destination,
                    "The PLY export failed.", saved.error().format(), "export.encoder");
            }
            break;
        case ProjectExportFormat::Sog:
            if (auto saved = save_sog(*merged, SogSaveOptions{
                                                   .output_path = destination,
                                                   .progress_callback = report,
                                               });
                !saved) {
                return fail<ProjectExportResult>(
                    lfs::ErrorCode::Unavailable, destination,
                    "The SOG export failed.", saved.error().format(), "export.encoder");
            }
            break;
        case ProjectExportFormat::Ssog:
            if (auto saved = save_ssog(*merged, SsogSaveOptions{
                                                    .output_path = destination,
                                                    .progress_callback = report,
                                                });
                !saved) {
                return fail<ProjectExportResult>(
                    lfs::ErrorCode::Unavailable, destination,
                    "The SSOG export failed.", saved.error().format(), "export.encoder");
            }
            break;
        case ProjectExportFormat::Spz:
            if (auto saved = save_spz(*merged, SpzSaveOptions{
                                                   .output_path = destination,
                                                   .progress_callback = report,
                                               });
                !saved) {
                return fail<ProjectExportResult>(
                    lfs::ErrorCode::Unavailable, destination,
                    "The SPZ export failed.", saved.error().format(), "export.encoder");
            }
            break;
        }
        std::error_code size_error;
        const auto bytes = std::filesystem::file_size(destination, size_error);
        if (size_error) {
            return fail<ProjectExportResult>(
                lfs::ErrorCode::DataLoss, destination,
                "The export output could not be measured.", size_error.message(),
                "export.output_size");
        }
        if (progress) {
            progress(1.0F, "Project export complete");
        }
        return ProjectExportResult{
            .destination = destination,
            .format = format,
            .gaussian_count = static_cast<std::uint64_t>(merged->size()),
            .bytes_written = bytes,
        };
    }
#endif

    lfs::Result<ProjectRepairResult>
    repair_project(const std::filesystem::path& path,
                   const std::filesystem::path& destination,
                   const lfs::core::Uuid& expected_project) {
        auto source_identity = detail::ProjectPathIdentity::capture(path);
        auto destination_identity = detail::ProjectPathIdentity::capture(destination);
        if (!source_identity)
            return std::move(source_identity).error();
        if (!destination_identity)
            return std::move(destination_identity).error();
        if (path.empty() || destination.empty() || same_path(path, destination)) {
            return fail<ProjectRepairResult>(
                lfs::ErrorCode::InvalidArgument, destination,
                "The repair paths are invalid.",
                "repair always writes a different destination", "repair.path");
        }
        auto source_lock = acquire_operation_lock(path);
        if (!source_lock) {
            return std::move(source_lock).error();
        }
        auto destination_lock = acquire_operation_lock(destination);
        if (!destination_lock) {
            return std::move(destination_lock).error();
        }
        if (auto available = refuse_existing_destination(destination); !available) {
            return std::move(available).error();
        }
        ReaderOptions classification_options;
        classification_options.allow_unsupported_inspection = true;
        auto classification = ProjectReader::classify(path, classification_options);
        if (classification.state != OpenState::RepairOnly) {
            return fail<ProjectRepairResult>(
                lfs::ErrorCode::FailedPrecondition, path,
                "The project is not marked RepairOnly.",
                "repair_project only recovers a damaged project head", "repair.open_state");
        }
        auto candidate = find_repair_candidate(path);
        if (!candidate) {
            return std::move(candidate).error();
        }
        const auto temporary = candidate->path;
        struct TemporaryCleanup {
            std::filesystem::path path;
            ~TemporaryCleanup() {
                if (!path.empty()) {
                    std::error_code ignored;
                    std::filesystem::remove(path, ignored);
                }
            }
        } cleanup{temporary};
        auto reader = ProjectReader::open(temporary);
        if (!reader) {
            return std::move(reader).error();
        }
        if (!expected_project.is_nil() && reader->superblock().project_uuid != expected_project)
            return fail<ProjectRepairResult>(lfs::ErrorCode::FailedPrecondition, path,
                                             "The project identity changed before repair started.",
                                             "recovered source does not match the selected project", "repair.identity");
        const auto& source_commit = reader->commit();
        if (!candidate->preview_selection_recovered) {
            const auto retained_thumbnail = std::ranges::find_if(
                reader->chunks(), [](const ChunkInfo& row) {
                    return row.is_live() && row.key.fourcc == FOURCC_THMB;
                });
            if (retained_thumbnail != reader->chunks().end()) {
                return fail<ProjectRepairResult>(
                    lfs::ErrorCode::FailedPrecondition, path,
                    "The recovered project preview is ambiguous.",
                    "the damaged heads do not provide a trustworthy preview selection",
                    "repair.preview");
            }
        }
        std::uint64_t planned_bytes = 0;
        for (const auto& row : reader->chunks()) {
            if (row.is_live()) {
                planned_bytes += row.stored_bytes;
            }
        }
        if (auto checked = source_identity->validate(); !checked)
            return std::move(checked).error();
        if (auto checked = destination_identity->validate(); !checked)
            return std::move(checked).error();
        auto writer = ProjectWriter::create(
            destination,
            CreateOptions{
                .project_uuid = reader->superblock().project_uuid,
                .file_uuid = lfs::core::generate_uuid_v4(),
                .role = ContainerRole::Master,
                .creation_time_unix_ns = reader->superblock().creation_time_unix_ns,
                .index_compression = IndexCompression::Zstd,
                .disk_reserve_bytes = 64ull * 1024 * 1024,
                .writer_lock_lease = *destination_lock,
            });
        if (!writer) {
            return std::move(writer).error();
        }
        auto planned = writer->plan_commit(CommitOptions{
            .kind = CommitKind::Recovered,
            .snapshot_uuid = source_commit.snapshot_uuid,
            .wallclock_unix_ns = source_commit.wallclock_unix_ns,
            .min_reader_version = source_commit.min_reader_version,
            .min_safe_writer_version = source_commit.min_safe_writer_version,
            .extra_reader_capabilities = source_commit.required_reader_capabilities,
            .extra_writer_capabilities = source_commit.required_writer_capabilities,
        });
        if (!planned) {
            return std::move(planned).error();
        }
        if (auto preflight = writer->preflight(planned_bytes); !preflight) {
            return std::move(preflight).error();
        }
        for (const auto& row : reader->chunks()) {
            if (!row.is_live()) {
                continue;
            }
            if (auto copied = writer->copy_chunk_verbatim(*reader, row); !copied) {
                return std::move(copied).error();
            }
        }
        if (auto committed = writer->commit(); !committed) {
            return std::move(committed).error();
        }
        auto repaired = ProjectReader::open(destination);
        if (!repaired) {
            return std::move(repaired).error();
        }
        if (auto verified = repaired->verify_all(); !verified) {
            return std::move(verified).error();
        }
        auto card = inspect_project_card(destination);
        if (!card) {
            return std::move(card).error();
        }
        return ProjectRepairResult{
            .card = *card,
            .saves_recovered = candidate->generation,
        };
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
        return mutate_document(path, "thumbnail_changed", [](ProjectDocument&) -> lfs::Result<void> { return {}; }, png_bytes);
    }

    lfs::Result<ProjectThumbnailSourceAvailability>
    inspect_project_thumbnail_sources(
        const std::filesystem::path& path) {
        auto opened = ProjectDocument::open(path);
        if (!opened) {
            return std::move(opened).error();
        }
        auto& document = *opened;
        ProjectThumbnailSourceAvailability availability;

        try {
            if (const auto first = first_dataset_image(
                    document.project(), document.references(),
                    document.parameters(), path.parent_path())) {
                availability.first_dataset_image =
                    static_cast<bool>(dataset_preview_png(*first));
            }
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): a bad external source only removes
            // this optional choice; keep the other thumbnail sources available.
            availability.first_dataset_image = false;
        }

        auto manifest = document.parameters().embedded_dataset();
        if (!manifest) {
            return availability;
        }
        if (*manifest) {
            const auto first_image = std::ranges::find_if(
                (**manifest).entries,
                [](const EmbeddedDatasetEntry& entry) {
                    return entry.kind == "image";
                });
            if (first_image != (**manifest).entries.end()) {
                const auto* payload =
                    document.find_dataset_source(first_image->chunk_uuid);
                if (payload) {
                    auto bytes = read_lazy_payload(*payload);
                    if (bytes) {
                        try {
                            availability.first_embedded_image =
                                static_cast<bool>(encode_image_bytes(*bytes));
                        } catch (...) {
                            // LFS-CENSUS-OK(empty-catch): a corrupt optional
                            // embedded source must not hide other choices.
                            availability.first_embedded_image = false;
                        }
                    }
                }
            }
        }
        return availability;
    }

    lfs::Result<std::vector<std::byte>>
    encode_preview_from_image_file(const std::filesystem::path& image_path) {
        return dataset_preview_png(image_path);
    }

    lfs::Result<std::vector<std::byte>>
    encode_preview_from_first_dataset_image(const std::filesystem::path& path) {
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        const auto first = first_dataset_image(
            document->project(), document->references(), document->parameters(),
            path.parent_path());
        if (!first) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::NotFound, path,
                "The project has no reachable dataset image.",
                "first_dataset_image returned no image", "preview.dataset");
        }
        return dataset_preview_png(*first);
    }

    lfs::Result<std::vector<std::byte>>
    encode_preview_from_first_embedded_image(const std::filesystem::path& path) {
        auto document = ProjectDocument::open(path);
        if (!document) {
            return std::move(document).error();
        }
        auto manifest = document->parameters().embedded_dataset();
        if (!manifest) {
            return std::move(manifest).error();
        }
        if (!*manifest) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::NotFound, path,
                "The project has no embedded dataset image.",
                "embedded dataset manifest is absent", "preview.dataset");
        }
        const auto entry = std::ranges::find_if(
            (**manifest).entries,
            [](const EmbeddedDatasetEntry& value) { return value.kind == "image"; });
        if (entry == (**manifest).entries.end()) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::NotFound, path,
                "The embedded dataset has no image payload.",
                "embedded dataset manifest contains no image entry",
                "preview.dataset");
        }
        const auto* payload = document->find_dataset_source(entry->chunk_uuid);
        if (!payload) {
            return fail<std::vector<std::byte>>(
                lfs::ErrorCode::DataLoss, path,
                "The embedded dataset image payload is missing.",
                entry->chunk_uuid.to_string(), "preview.dataset");
        }
        auto image = read_lazy_payload(*payload);
        if (!image) {
            return std::move(image).error();
        }
        return encode_image_bytes(*image);
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
        if (auto marked = mark_contents_edit(*document, "thumbnail_changed"); !marked)
            return std::move(marked).error();
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.regenerate_dataset_preview = false;
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
        if (auto marked = mark_contents_edit(*document, "thumbnail_changed"); !marked)
            return std::move(marked).error();
        ProjectDocumentSaveOptions options;
        options.commit.kind = CommitKind::Explicit;
        options.regenerate_dataset_preview = false;
        options.writer_lock_lease = *lease;
        options.preview_png = *png;
        auto saved = document->save(path, options);
        if (!saved) {
            return std::move(saved).error();
        }
        return inspect_after_save(path);
    }

    lfs::Result<ProjectInspectorCard>
    preview_from_image_file(const std::filesystem::path& project_path,
                            const std::filesystem::path& image_path) {
        auto png = dataset_preview_png(image_path);
        if (!png) {
            return std::move(png).error();
        }
        return set_project_preview(project_path, *png);
    }

    lfs::Result<ProjectInspectorCard>
    undo_contents_removal(const std::filesystem::path& path, const std::string& id) {
        return mutate_document(path, "removal_undone", [&](ProjectDocument& document) -> lfs::Result<void> {
            auto pending = pending_contents(document);
            auto& rows = pending["rows"];
            auto removed = std::find_if(rows.begin(), rows.end(), [&](const auto& row) {
                return row.value("id", std::string{}) == id && row.value("kind", std::string{}) == "save";
            });
            if (removed == rows.end())
                return fail<void>(lfs::ErrorCode::NotFound, path,
                                  "This removal can no longer be undone.", "the save was compacted or is not removed", "contents.undo");
            rows.erase(removed);
            return document.edit_project().dom().set_json("contents_removals", std::move(pending));
        });
    }

    lfs::Result<ProjectInspectorCard>
    set_project_license(const std::filesystem::path& path,
                        const std::string& identifier,
                        const std::string& notice) {
        return mutate_document(path, "license_changed", [&](ProjectDocument& document) {
            return document.set_license(ProjectLicense{identifier, notice});
        });
    }

    lfs::Result<ProjectInspectorCard>
    clear_project_license(const std::filesystem::path& path) {
        return mutate_document(path, "license_removed", [](ProjectDocument& document) -> lfs::Result<void> {
            auto license = document.project().license();
            if (!license)
                return lfs::Result<void>::failure(std::move(license).error());
            if (*license) {
                auto pending = pending_contents(document);
                pending["rows"].push_back({{"id", "license"}, {"kind", "license"}, {"bytes", (*license)->identifier.size() + (*license)->notice.size()}, {"identifier", (*license)->identifier}});
                if (auto updated = document.edit_project().dom().set_json("contents_removals", pending); !updated)
                    return updated;
            }
            return document.clear_license();
        });
    }

    lfs::Result<ProjectInspectorCard>
    set_project_title(const std::filesystem::path& path,
                      const std::string& title) {
        return mutate_document(path, "title_changed", [&](ProjectDocument& document) {
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
