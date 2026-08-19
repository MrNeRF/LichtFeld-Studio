/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_document.hpp"

#include "core/logger.hpp"
#include "io/loader.hpp"
#include "project_container_internal.hpp"
#include "project_framing.hpp"
#include "span_streambuf.hpp"

#include <zstd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <format>
#include <istream>
#include <limits>
#include <map>
#include <ostream>
#include <ranges>
#include <set>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lfs::io::project {

    namespace {

        constexpr std::uint16_t P3_CHUNK_VERSION = 1;
        constexpr std::uint64_t DOCUMENT_CLEAN_BASELINE = 0;
        constexpr std::uint32_t PPISP_FILE_MAGIC =
            0x50505349;
        constexpr std::uint32_t PPISP_FILE_VERSION = 2;
        constexpr std::uint32_t PPISP_FILE_KNOWN_FLAGS =
            (1u << 0) | (1u << 1);

        struct PpispFileHeader {
            std::uint32_t magic = 0;
            std::uint32_t version = 0;
            std::uint32_t num_cameras = 0;
            std::uint32_t num_frames = 0;
            std::uint32_t flags = 0;
            std::uint32_t reserved[3]{};
        };
        static_assert(sizeof(PpispFileHeader) == 32);

        lfs::Error document_error(const lfs::ErrorCode code,
                                  std::string message,
                                  std::string detail,
                                  const std::string_view field = {}) {
            lfs::SmallFields fields;
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
                            std::string message,
                            std::string detail,
                            const std::string_view field = {}) {
            if constexpr (std::same_as<T, void>) {
                return lfs::Result<void>::failure(document_error(
                    code, std::move(message), std::move(detail), field));
            } else {
                return document_error(
                    code, std::move(message), std::move(detail), field);
            }
        }

        std::uint64_t unix_time_ns() {
            const auto now = std::chrono::system_clock::now().time_since_epoch();
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        }

        lfs::Result<std::filesystem::path>
        normalized_absolute_path(const std::filesystem::path& path) {
            if (path.empty()) {
                return fail<std::filesystem::path>(
                    lfs::ErrorCode::InvalidArgument,
                    "The project path is empty.",
                    "A .licht document requires a destination path",
                    "project.path");
            }
            std::error_code error;
            auto absolute = std::filesystem::absolute(path, error);
            if (error) {
                return fail<std::filesystem::path>(
                    lfs::ErrorCode::InvalidArgument,
                    "The project path could not be resolved.",
                    std::format("filesystem::absolute failed: {}", error.message()),
                    "project.path");
            }
            return absolute.lexically_normal();
        }

        bool is_project_managed_fourcc(const Fourcc fourcc) noexcept {
            return fourcc == FOURCC_PROJ || fourcc == FOURCC_PRMS ||
                   fourcc == FOURCC_SCNG || fourcc == FOURCC_SELM ||
                   fourcc == FOURCC_REFS || fourcc == FOURCC_SPLT ||
                   fourcc == FOURCC_PCLD || fourcc == FOURCC_MESH ||
                   fourcc == FOURCC_GUIL || fourcc == FOURCC_VIEW ||
                   fourcc == FOURCC_EDTR || fourcc == FOURCC_SEQR ||
                   fourcc == FOURCC_METR;
        }

        bool is_lazy_binary_fourcc(const Fourcc fourcc) noexcept {
            return fourcc == FOURCC_CKPT || fourcc == FOURCC_PPIS;
        }

        bool is_singleton_fourcc(const Fourcc fourcc) noexcept {
            return fourcc == FOURCC_PROJ || fourcc == FOURCC_PRMS ||
                   fourcc == FOURCC_SCNG || fourcc == FOURCC_SELM ||
                   fourcc == FOURCC_REFS || fourcc == FOURCC_GUIL ||
                   fourcc == FOURCC_VIEW || fourcc == FOURCC_EDTR ||
                   fourcc == FOURCC_SEQR || fourcc == FOURCC_METR;
        }

        bool has_unknown_json_root(const Fourcc fourcc,
                                   const JsonChapterDom& dom) {
            using Json = JsonChapterDom::Json;
            const auto root = Json::parse(dom.dump());
            if (!root.is_object()) {
                return false;
            }
            std::set<std::string_view> known;
            if (fourcc == FOURCC_PROJ) {
                known = {"schema_version", "manifest", "project_uuid",
                         "created_at_unix_ns", "modified_at_unix_ns",
                         "dataset_reference_uuid", "project_lineage",
                         "georeference", "embed_decisions", "provenance",
                         "embedded_payloads"};
            } else if (fourcc == FOURCC_REFS) {
                known = {"schema_version", "references"};
            } else if (fourcc == FOURCC_SCNG) {
                known = {"schema_version", "nodes", "training_model_uuid"};
            } else if (fourcc == FOURCC_PRMS) {
                known = {"schema_version", "active_strategy", "presets",
                         "dataset"};
            } else if (fourcc == FOURCC_GUIL) {
                known = {"version", "layouts"};
            } else if (fourcc == FOURCC_VIEW) {
                known = {"version", "render_settings", "panel_cameras",
                         "navigation", "split", "camera_bookmarks", "tools",
                         "sequencer_view", "active_camera_uuid"};
            } else if (fourcc == FOURCC_EDTR) {
                known = {"version", "open_files", "active_file", "vim_mode",
                         "contains_embedded_secrets"};
            } else if (fourcc == FOURCC_SEQR) {
                known = {"version", "timeline", "ply_sequences", "playhead",
                         "loop_mode", "playback_speed", "preferences"};
            } else {
                return false;
            }
            return std::ranges::any_of(
                root.items(), [&](const auto& item) {
                    return !known.contains(item.key());
                });
        }

        ChunkKey singleton_key(const Fourcc fourcc,
                               const lfs::core::Uuid& project_uuid) {
            return ChunkKey{.fourcc = fourcc, .instance_uuid = project_uuid};
        }

        ParameterManagerSnapshot default_parameter_snapshot() {
            ParameterManagerSnapshot result;
            result.active_strategy =
                std::string(lfs::core::param::kStrategyMRNF);
            result.mcmc_session =
                lfs::core::param::OptimizationParameters::mcmc_defaults();
            result.mrnf_session =
                lfs::core::param::OptimizationParameters::mrnf_defaults();
            result.igs_session =
                lfs::core::param::OptimizationParameters::igs_plus_defaults();
            result.mcmc_current = result.mcmc_session;
            result.mrnf_current = result.mrnf_session;
            result.igs_current = result.igs_session;
            result.dataset.centralize_dataset = "off";
            result.dataset.loading_params = lfs::core::param::LoadingParams{};
            return result;
        }

        lfs::Result<std::vector<ReferenceOwnerBinding>>
        session_reference_bindings(
            const ViewSessionChapter& view,
            const SequencerSessionChapter& sequencer) {
            using Json = JsonChapterDom::Json;
            std::vector<ReferenceOwnerBinding> result;

            const auto parse_uuid =
                [](const Json& value,
                   const std::string_view field)
                -> lfs::Result<lfs::core::Uuid> {
                if (!value.is_string()) {
                    return fail<lfs::core::Uuid>(
                        lfs::ErrorCode::DataLoss,
                        "A session reference UUID has the wrong type.",
                        "Reference UUIDs must use canonical UUID strings",
                        field);
                }
                const auto parsed =
                    lfs::core::Uuid::from_string(
                        value.get<std::string>());
                if (!parsed || parsed->is_nil()) {
                    return fail<lfs::core::Uuid>(
                        lfs::ErrorCode::DataLoss,
                        "A session reference UUID is invalid.",
                        "Reference UUIDs must be canonical and non-nil",
                        field);
                }
                return *parsed;
            };

            const auto settings =
                view.dom().get_json("render_settings");
            if (!settings || !settings->is_object()) {
                return fail<
                    std::vector<ReferenceOwnerBinding>>(
                    lfs::ErrorCode::DataLoss,
                    "VIEW render settings are missing.",
                    "The environment reference owner cannot be indexed",
                    "VIEW.render_settings");
            }
            const auto environment =
                settings->find(
                    "environment_reference_uuid");
            if (environment != settings->end() &&
                !environment->is_null()) {
                auto uuid = parse_uuid(
                    *environment,
                    "VIEW.render_settings.environment_reference_uuid");
                if (!uuid) {
                    return std::move(uuid).error();
                }
                result.push_back({
                    .reference_uuid = *uuid,
                    .chapter = "VIEW",
                    .owner_uuid = std::nullopt,
                    .field =
                        "render_settings.environment_reference_uuid",
                });
            }

            const auto clips =
                sequencer.dom().get_json(
                    "ply_sequences");
            if (!clips || !clips->is_array()) {
                return fail<
                    std::vector<ReferenceOwnerBinding>>(
                    lfs::ErrorCode::DataLoss,
                    "SEQR PLY clips are missing.",
                    "The PLY-directory reference owners cannot be indexed",
                    "SEQR.ply_sequences");
            }
            for (const auto& clip : *clips) {
                if (!clip.is_object()) {
                    return fail<
                        std::vector<ReferenceOwnerBinding>>(
                        lfs::ErrorCode::DataLoss,
                        "A SEQR PLY clip is invalid.",
                        "PLY clips must be JSON objects",
                        "SEQR.ply_sequences");
                }
                const auto reference =
                    clip.find(
                        "directory_reference_uuid");
                if (reference == clip.end() ||
                    reference->is_null()) {
                    continue;
                }
                auto reference_uuid = parse_uuid(
                    *reference,
                    "SEQR.ply_sequences.directory_reference_uuid");
                if (!reference_uuid) {
                    return std::move(
                               reference_uuid)
                        .error();
                }
                const auto owner =
                    clip.find("node_uuid");
                if (owner == clip.end()) {
                    return fail<
                        std::vector<ReferenceOwnerBinding>>(
                        lfs::ErrorCode::DataLoss,
                        "A referenced SEQR PLY clip has no stable owner.",
                        "directory_reference_uuid requires node_uuid",
                        "SEQR.ply_sequences.node_uuid");
                }
                auto owner_uuid = parse_uuid(
                    *owner,
                    "SEQR.ply_sequences.node_uuid");
                if (!owner_uuid) {
                    return std::move(owner_uuid)
                        .error();
                }
                result.push_back({
                    .reference_uuid =
                        *reference_uuid,
                    .chapter = "SEQR",
                    .owner_uuid = *owner_uuid,
                    .field =
                        "ply_sequences.directory_reference_uuid",
                });
            }
            return result;
        }

        template <typename Map>
        std::vector<lfs::core::Uuid> sorted_uuids(const Map& values) {
            std::vector<lfs::core::Uuid> result;
            result.reserve(values.size());
            for (const auto& [uuid, ignored] : values) {
                (void)ignored;
                result.push_back(uuid);
            }
            std::ranges::sort(result, {}, [](const lfs::core::Uuid& uuid) {
                return uuid.bytes;
            });
            return result;
        }

        lfs::Result<std::uint64_t>
        checked_add(const std::uint64_t lhs, const std::uint64_t rhs,
                    const std::string_view field) {
            if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
                return fail<std::uint64_t>(
                    lfs::ErrorCode::ResourceExhausted,
                    "The project payload size exceeds the supported range.",
                    std::format("{} overflows uint64", field), field);
            }
            return lhs + rhs;
        }

        struct EncodedChunk {
            ChunkKey key;
            std::vector<std::byte> bytes;
            ChunkWriteOptions options;
        };

        lfs::Result<std::uint64_t>
        preflight_bytes(const std::map<ChunkKey, EncodedChunk, ChunkKeyLess>& chunks) {
            std::uint64_t total = 0;
            for (const auto& [key, chunk] : chunks) {
                (void)key;
                std::uint64_t estimate = chunk.bytes.size();
                if (chunk.options.compression == Compression::ZstdFramed ||
                    chunk.options.compression == Compression::ByteShuffleZstdFramed) {
                    const auto records = detail::framed_record_count(chunk.bytes.size());
                    estimate = detail::FRAMED_HEADER_BYTES + records * detail::FRAMED_RECORD_BYTES;
                    for (std::size_t index = 0; index < records; ++index) {
                        const auto offset = index * detail::FRAMED_RECORD_TARGET_BYTES;
                        estimate += ZSTD_compressBound(std::min(
                            detail::FRAMED_RECORD_TARGET_BYTES, chunk.bytes.size() - offset));
                    }
                }
                auto added = checked_add(total, estimate, "save.preflight_bytes");
                if (!added) {
                    return std::move(added).error();
                }
                total = *added;
            }
            return total;
        }

        ChunkWriteOptions json_options() {
            return ChunkWriteOptions{
                .chunk_version = P3_CHUNK_VERSION,
                .compression = Compression::ZstdFramed,
                .tensor_payload = false,
                .block_crcs = false,
                .expected_stream_bytes = std::nullopt,
            };
        }

        ChunkWriteOptions selection_options() {
            return ChunkWriteOptions{
                .chunk_version = SELM_CHAPTER_VERSION,
                .compression = Compression::ZstdFramed,
                .tensor_payload = false,
                .block_crcs = false,
                .expected_stream_bytes = std::nullopt,
            };
        }

        ChunkWriteOptions tensor_options(const std::size_t size) {
            return ChunkWriteOptions{
                .chunk_version = P3_CHUNK_VERSION,
                // Prefer framed ByteShuffleZstd (CHUNK_BYTESHUFFLE_ZSTD_V1) when
                // the payload is a multiple of 4; write_chunk falls back to
                // framed Zstd otherwise. Logical payload remains bit-exact after
                // unshuffle + inflate.
                .compression = Compression::ByteShuffleZstdFramed,
                .tensor_payload = true,
                .block_crcs =
                    size >= static_cast<std::size_t>(BLOCK_CRC_REQUIRED_AT),
                .expected_stream_bytes = std::nullopt,
            };
        }

        ChunkWriteOptions lazy_binary_options(
            const Fourcc fourcc,
            const std::uint64_t size) {
            return ChunkWriteOptions{
                .chunk_version = P3_CHUNK_VERSION,
                // Byte-plane + framed zstd of the CKPT/PPIS stream when size % 4
                // == 0; else framed zstd. Decompress (and unshuffle) before
                // LFKP/PPISP parse — byte-verbatim logical payload.
                .compression = Compression::ByteShuffleZstdFramed,
                .tensor_payload = fourcc == FOURCC_CKPT,
                .block_crcs = size >= BLOCK_CRC_REQUIRED_AT,
                .expected_stream_bytes = size,
            };
        }

        std::string payload_identity(const lfs::core::Uuid& node,
                                     const std::string_view fourcc) {
            return std::format("{}:{}", node.to_string(), fourcc);
        }

        WorldOriginProvenance project_provenance(
            const lfs::io::ImportWorldOriginProvenance value) {
            using Import =
                lfs::io::ImportWorldOriginProvenance;
            switch (value) {
            case Import::None:
                return WorldOriginProvenance::None;
            case Import::CentralizeByPointCloud:
                return WorldOriginProvenance::
                    CentralizeByPointCloud;
            case Import::CentralizeByCameras:
                return WorldOriginProvenance::
                    CentralizeByCameras;
            case Import::User:
                return WorldOriginProvenance::User;
            case Import::Import:
                return WorldOriginProvenance::Import;
            }
            assert(false &&
                   "unhandled import georeference provenance");
            return WorldOriginProvenance::None;
        }

    } // namespace

    struct LazyChunkValue::Impl {
        std::shared_ptr<ProjectReader> reader;
        std::optional<ChunkInfo> source;
        std::optional<CleanProof> proof;
        std::shared_ptr<const std::vector<std::byte>> owned;
        // Cache of container-decompressed logical bytes for Zstd /
        // ByteShuffleZstd sources. visit_stream prefers a bounded decode
        // stream and does not populate this. read_at still materializes
        // compressed sources once.
        mutable std::shared_ptr<const std::vector<std::byte>> inflated;
        lfs::core::Uuid snapshot_uuid;

        [[nodiscard]] std::uint64_t size() const noexcept {
            if (owned) {
                return owned->size();
            }
            if (inflated) {
                return inflated->size();
            }
            return source ? source->uncompressed_bytes : 0;
        }

        [[nodiscard]] lfs::Result<std::span<const std::byte>>
        logical_owned_or_inflated() const {
            if (owned) {
                return std::span<const std::byte>(owned->data(),
                                                  owned->size());
            }
            if (inflated) {
                return std::span<const std::byte>(inflated->data(),
                                                  inflated->size());
            }
            if (!reader || !source) {
                return fail<std::span<const std::byte>>(
                    lfs::ErrorCode::FailedPrecondition,
                    "The lazy chapter has no byte source.",
                    "Neither clean file range nor owned storage is available",
                    "lazy_chunk.source");
            }
            if (source->compression == Compression::Stored) {
                return fail<std::span<const std::byte>>(
                    lfs::ErrorCode::FailedPrecondition,
                    "Stored lazy chapters stream from the file.",
                    "logical_owned_or_inflated is for owned/compressed sources",
                    "lazy_chunk.compression");
            }
            auto decoded = reader->read_chunk(*source);
            if (!decoded) {
                return std::move(decoded).error();
            }
            inflated = std::make_shared<const std::vector<std::byte>>(
                std::move(*decoded));
            return std::span<const std::byte>(inflated->data(),
                                              inflated->size());
        }
    };

    LazyChunkValue::LazyChunkValue(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}
    LazyChunkValue::LazyChunkValue(LazyChunkValue&&) noexcept = default;
    LazyChunkValue&
    LazyChunkValue::operator=(LazyChunkValue&&) noexcept = default;
    LazyChunkValue::~LazyChunkValue() = default;

    lfs::Result<LazyChunkValue>
    LazyChunkValue::from_owned(
        std::shared_ptr<const std::vector<std::byte>> bytes,
        const lfs::core::Uuid& snapshot_uuid) {
        if (!bytes) {
            return fail<LazyChunkValue>(
                lfs::ErrorCode::InvalidArgument,
                "The staged chapter storage is missing.",
                "LazyChunkValue requires an immutable owned byte source",
                "lazy_chunk.bytes");
        }
        if (snapshot_uuid.is_nil()) {
            return fail<LazyChunkValue>(
                lfs::ErrorCode::InvalidArgument,
                "The snapshot UUID cannot be null.",
                "Every owned training snapshot piece must carry one UUID",
                "lazy_chunk.snapshot_uuid");
        }
        auto impl = std::make_unique<Impl>();
        impl->owned = std::move(bytes);
        impl->snapshot_uuid = snapshot_uuid;
        return LazyChunkValue(std::move(impl));
    }

    lfs::Result<LazyChunkValue>
    LazyChunkValue::from_owned(
        std::vector<std::byte> bytes,
        const lfs::core::Uuid& snapshot_uuid) {
        return from_owned(
            std::make_shared<const std::vector<std::byte>>(
                std::move(bytes)),
            snapshot_uuid);
    }

    std::uint64_t LazyChunkValue::size() const noexcept {
        return impl_->size();
    }

    const lfs::core::Uuid&
    LazyChunkValue::snapshot_uuid() const noexcept {
        return impl_->snapshot_uuid;
    }

    bool LazyChunkValue::is_clean_reference() const noexcept {
        return impl_->reader && impl_->source && impl_->proof &&
               !impl_->owned;
    }

    void LazyChunkValue::drop_clean_proof_for_testing() noexcept {
        impl_->proof.reset();
    }

    lfs::Result<void>
    LazyChunkValue::read_at(
        const std::uint64_t offset,
        const std::span<std::byte> destination) const {
        const auto total = size();
        if (offset > total ||
            destination.size() > total - offset) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The lazy chapter window is out of bounds.",
                std::format(
                    "offset {} + size {} exceeds chapter size {}",
                    offset, destination.size(), total),
                "lazy_chunk.window");
        }
        if (destination.empty()) {
            return {};
        }
        if (impl_->owned) {
            std::memcpy(
                destination.data(),
                impl_->owned->data() + offset,
                destination.size());
            return {};
        }
        if (!impl_->reader || !impl_->source) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The lazy chapter has no byte source.",
                "Neither clean file range nor owned storage is available",
                "lazy_chunk.source");
        }
        // Stored: random-access on file payload. Compressed: inflate once
        // (and unshuffle for ByteShuffleZstd) then slice.
        if (impl_->source->compression == Compression::Stored) {
            return impl_->reader->read_stored_at(
                *impl_->source, offset, destination);
        }
        auto logical = impl_->logical_owned_or_inflated();
        if (!logical) {
            return lfs::Result<void>::failure(std::move(logical).error());
        }
        std::memcpy(
            destination.data(), logical->data() + offset, destination.size());
        return {};
    }

    lfs::Result<void>
    LazyChunkValue::visit_stream(
        const StreamVisitor& visitor) const {
        if (!visitor) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The lazy chapter visitor is empty.",
                "visit_stream requires a callable",
                "lazy_chunk.visitor");
        }
        if (impl_->owned) {
            SpanStreambuf buffer(std::span<const std::byte>(
                impl_->owned->data(), impl_->owned->size()));
            std::istream stream(&buffer);
            return visitor(stream, impl_->owned->size());
        }
        if (!impl_->reader || !impl_->source) {
            return fail<void>(
                lfs::ErrorCode::FailedPrecondition,
                "The lazy chapter has no byte source.",
                "Neither clean file range nor owned storage is available",
                "lazy_chunk.source");
        }
        if (impl_->inflated) {
            SpanStreambuf buffer(std::span<const std::byte>(
                impl_->inflated->data(), impl_->inflated->size()));
            std::istream stream(&buffer);
            return visitor(stream, impl_->inflated->size());
        }
        if (impl_->source->compression == Compression::Stored ||
            impl_->source->compression == Compression::ZstdFramed ||
            impl_->source->compression == Compression::ByteShuffleZstdFramed) {
            auto bounded =
                impl_->reader->open_bounded_stream(*impl_->source);
            if (!bounded) {
                return lfs::Result<void>::failure(
                    std::move(bounded).error());
            }
            return visitor(bounded->stream(), bounded->size());
        }
        auto logical = impl_->logical_owned_or_inflated();
        if (!logical) {
            return lfs::Result<void>::failure(std::move(logical).error());
        }
        SpanStreambuf buffer(*logical);
        std::istream stream(&buffer);
        return visitor(stream, logical->size());
    }

    struct ProjectHydrationPlan::Impl {
        lfs::core::Scene* destination = nullptr;
        std::unique_ptr<lfs::core::Scene> staged_scene;
        ProjectDocumentHydrationReport report;
    };

    struct ProjectDocument::Impl {
        struct SourceRow {
            ChunkInfo info;
            CleanProof proof;
            bool opaque = false;

            [[nodiscard]] lfs::Result<void>
            reuse(ProjectWriter& writer) const {
                return writer.reuse_if_clean(
                    proof, proof.mutation_epoch());
            }

            [[nodiscard]] lfs::Result<void>
            carry_opaque(ProjectWriter& writer) const {
                return writer.carry_forward_opaque(
                    info, proof, proof.mutation_epoch());
            }
        };

        lfs::core::Uuid project_uuid;
        ProjectChapter project;
        ReferencesChapter references;
        SceneGraphChapter scene_graph;
        SelectionChapter selection;
        ParametersChapter parameters;
        GuiLayoutChapter gui_layout;
        ViewSessionChapter view;
        EditorSessionChapter editor;
        SequencerSessionChapter sequencer;
        MetricsChapter metrics;

        std::unordered_map<lfs::core::Uuid, SplatChapterPayload> splats;
        std::unordered_map<lfs::core::Uuid, PointCloudPayload> point_clouds;
        std::unordered_map<lfs::core::Uuid, MeshPayload> meshes;
        std::unordered_map<lfs::core::Uuid, LazyChunkValue> checkpoints;
        std::unordered_map<lfs::core::Uuid, LazyChunkValue> ppisp_payloads;

        std::optional<std::filesystem::path> source_path;
        std::shared_ptr<ProjectReader> source_reader;
        std::uint64_t generation = 0;
        std::map<ChunkKey, SourceRow, ChunkKeyLess> source_rows;
        std::set<ChunkKey, ChunkKeyLess> lazy_source_keys;
        std::set<ChunkKey, ChunkKeyLess> deferred_geometry_keys;
        std::map<ChunkKey, Hash128, ChunkKeyLess> content_hashes;
        std::set<ChunkKey, ChunkKeyLess> dirty;
        // Chapters normalized while loading must be materialized on the next
        // save instead of reusing their original clean byte spans. This is
        // intentionally separate from user-visible dirty state: accepting a
        // legacy document must not create a save prompt by itself.
        std::set<ChunkKey, ChunkKeyLess> normalized_source_keys;
        std::uint64_t dirty_epoch = 0;
        bool missing_opaque_preservation_capability = false;
        bool missing_retained_json_capability = false;
        mutable std::vector<ProjectDocumentDegradedState>
            degraded_states;

        [[nodiscard]] ChunkKey key(const Fourcc fourcc) const {
            return singleton_key(fourcc, project_uuid);
        }

        void mark(const Fourcc fourcc) {
            dirty.insert(key(fourcc));
            ++dirty_epoch;
        }

        void mark(const Fourcc fourcc, const lfs::core::Uuid& instance_uuid) {
            dirty.insert(ChunkKey{
                .fourcc = fourcc,
                .instance_uuid = instance_uuid,
            });
            ++dirty_epoch;
        }

        [[nodiscard]] bool dirty_or_new(const ChunkKey& chunk_key) const {
            return dirty.contains(chunk_key) ||
                   normalized_source_keys.contains(chunk_key) ||
                   !source_rows.contains(chunk_key);
        }

        [[nodiscard]] lfs::Result<void>
        validate(const ProjectChapter& candidate_project,
                 const std::map<ChunkKey, Hash128, ChunkKeyLess>& hashes) const {
            degraded_states.clear();
            auto manifest = candidate_project.manifest();
            auto manifest_uuid = candidate_project.project_uuid();
            auto created = candidate_project.created_at_unix_ns();
            auto modified = candidate_project.modified_at_unix_ns();
            auto dataset_reference = candidate_project.dataset_reference();
            auto lineage = candidate_project.project_lineage();
            auto georeference = candidate_project.georeference();
            auto decisions = candidate_project.embed_decisions();
            auto provenance = candidate_project.provenance();
            auto embedded = candidate_project.embedded_payload_provenance();
            if (!manifest) {
                return lfs::Result<void>::failure(std::move(manifest).error());
            }
            if (!manifest_uuid) {
                return lfs::Result<void>::failure(
                    std::move(manifest_uuid).error());
            }
            if (!created) {
                return lfs::Result<void>::failure(std::move(created).error());
            }
            if (!modified) {
                return lfs::Result<void>::failure(std::move(modified).error());
            }
            if (!dataset_reference) {
                return lfs::Result<void>::failure(
                    std::move(dataset_reference).error());
            }
            if (!lineage) {
                return lfs::Result<void>::failure(std::move(lineage).error());
            }
            if (!georeference) {
                return lfs::Result<void>::failure(
                    std::move(georeference).error());
            }
            if (!decisions) {
                return lfs::Result<void>::failure(std::move(decisions).error());
            }
            if (!provenance) {
                return lfs::Result<void>::failure(
                    std::move(provenance).error());
            }
            if (!embedded) {
                return lfs::Result<void>::failure(std::move(embedded).error());
            }
            if (*manifest_uuid != project_uuid) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The project UUID does not match the container.",
                    std::format("PROJ UUID {} differs from superblock UUID {}",
                                manifest_uuid->to_string(),
                                project_uuid.to_string()),
                    "PROJ.project_uuid");
            }
            if (*created == 0 || *modified == 0 || *modified < *created) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The project timestamps are invalid.",
                    "created_at and modified_at must be non-zero and monotonic",
                    "PROJ.timestamps");
            }

            auto reference_rows = references.records();
            if (!reference_rows) {
                return lfs::Result<void>::failure(
                    std::move(reference_rows).error());
            }
            std::unordered_map<lfs::core::Uuid, const ReferenceRecord*>
                references_by_uuid;
            for (const auto& record : *reference_rows) {
                references_by_uuid.emplace(record.uuid, &record);
            }
            if (*dataset_reference &&
                !references_by_uuid.contains(**dataset_reference)) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The project dataset reference is missing.",
                    std::format("PROJ dataset reference {} has no REFS row",
                                (**dataset_reference).to_string()),
                    "PROJ.dataset_reference");
            }

            if (auto hierarchy = scene_graph.validate_hierarchy(); !hierarchy) {
                return hierarchy;
            }
            auto nodes = scene_graph.nodes();
            auto training = scene_graph.training_model_uuid();
            if (!nodes) {
                return lfs::Result<void>::failure(std::move(nodes).error());
            }
            if (!training) {
                return lfs::Result<void>::failure(std::move(training).error());
            }
            auto pending = parameters.snapshot();
            if (!pending) {
                return lfs::Result<void>::failure(std::move(pending).error());
            }
            if (auto valid = gui_layout.validate(); !valid) {
                return valid;
            }
            if (auto valid = view.validate(); !valid) {
                return valid;
            }
            if (auto valid = editor.validate(); !valid) {
                return valid;
            }
            if (auto valid = sequencer.validate(); !valid) {
                return valid;
            }
            if (auto valid = metrics.validate(); !valid) {
                return valid;
            }
            auto session_bindings =
                session_reference_bindings(
                    view, sequencer);
            if (!session_bindings) {
                return lfs::Result<void>::failure(
                    std::move(session_bindings).error());
            }
            auto reverse = build_reverse_reference_index(
                references, candidate_project, scene_graph, parameters,
                *session_bindings);
            if (!reverse) {
                return lfs::Result<void>::failure(
                    std::move(reverse).error());
            }
            auto encoded_selection = encode_selection_chapter(selection);
            if (!encoded_selection) {
                return lfs::Result<void>::failure(
                    std::move(encoded_selection).error());
            }

            std::unordered_map<lfs::core::Uuid, const SceneNodeRecord*> nodes_by_uuid;
            nodes_by_uuid.reserve(nodes->size());
            for (const auto& node : *nodes) {
                nodes_by_uuid.emplace(node.uuid, &node);
            }
            const auto parse_session_uuid =
                [](const JsonChapterDom::Json& value,
                   const std::string_view field)
                -> lfs::Result<lfs::core::Uuid> {
                if (!value.is_string()) {
                    return fail<lfs::core::Uuid>(
                        lfs::ErrorCode::DataLoss,
                        "A session camera UUID has the wrong type.",
                        "Camera identities use canonical UUID strings",
                        field);
                }
                const auto parsed = lfs::core::Uuid::from_string(
                    value.get<std::string>());
                if (!parsed || parsed->is_nil()) {
                    return fail<lfs::core::Uuid>(
                        lfs::ErrorCode::DataLoss,
                        "A session camera UUID is invalid.",
                        "Camera identities must be canonical and non-null",
                        field);
                }
                return *parsed;
            };

            if (const auto active_camera =
                    view.dom().get_json("active_camera_uuid");
                active_camera && !active_camera->is_null()) {
                auto uuid = parse_session_uuid(
                    *active_camera, "VIEW.active_camera_uuid");
                if (!uuid) {
                    return lfs::Result<void>::failure(
                        std::move(uuid).error());
                }
                const auto found = nodes_by_uuid.find(*uuid);
                if (found == nodes_by_uuid.end() ||
                    found->second->type != "camera") {
                    degraded_states.push_back(
                        ProjectDocumentDegradedState::MissingActiveCamera);
                }
            }

            if (const auto timeline =
                    sequencer.dom().get_json("timeline");
                timeline && timeline->is_object()) {
                const auto keyframes = timeline->find("keyframes");
                if (keyframes != timeline->end() &&
                    keyframes->is_array()) {
                    for (const auto& keyframe : *keyframes) {
                        if (!keyframe.is_object()) {
                            continue;
                        }
                        const auto camera =
                            keyframe.find("camera_uuid");
                        if (camera == keyframe.end() ||
                            camera->is_null()) {
                            continue;
                        }
                        auto uuid = parse_session_uuid(
                            *camera,
                            "SEQR.timeline.keyframes.camera_uuid");
                        if (!uuid) {
                            return lfs::Result<void>::failure(
                                std::move(uuid).error());
                        }
                        const auto found = nodes_by_uuid.find(*uuid);
                        if (found == nodes_by_uuid.end() ||
                            found->second->type != "camera") {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "A sequencer keyframe references a missing camera.",
                                std::format(
                                    "SEQR camera {} is absent from SCNG",
                                    uuid->to_string()),
                                "SEQR.timeline.keyframes.camera_uuid");
                        }
                    }
                }
            }
            for (const auto& selected : selection.selected_node_uuids()) {
                if (!nodes_by_uuid.contains(selected)) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The saved node selection has a missing owner.",
                        std::format("SELM selected node {} is absent from SCNG",
                                    selected.to_string()),
                        "SELM.selected_node_uuids");
                }
            }
            for (const auto& slice : selection.slices()) {
                const auto found = nodes_by_uuid.find(slice.node_uuid);
                if (found == nodes_by_uuid.end()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A saved selection mask has a missing owner.",
                        std::format("SELM slice {} is absent from SCNG",
                                    slice.node_uuid.to_string()),
                        "SELM.slices.node_uuid");
                }
                const bool compatible =
                    (slice.domain == lfs::core::SelectionDomain::Splat &&
                     found->second->type == "splat") ||
                    (slice.domain ==
                         lfs::core::SelectionDomain::PointCloud &&
                     found->second->type == "pointcloud");
                if (!compatible) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A saved selection mask has the wrong geometry domain.",
                        std::format("SELM slice {} domain does not match SCNG type {}",
                                    slice.node_uuid.to_string(),
                                    found->second->type),
                        "SELM.slices.domain");
                }
            }

            std::unordered_map<std::string, const EmbeddedPayloadProvenance*>
                embedded_by_payload;
            embedded_by_payload.reserve(embedded->size());
            for (const auto& record : *embedded) {
                const auto identity =
                    payload_identity(record.node_uuid, record.fourcc);
                if (!embedded_by_payload.emplace(identity, &record).second) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "Embedded payload provenance is duplicated.",
                        std::format("PROJ has multiple provenance triples for {}",
                                    identity),
                        "PROJ.embedded_payloads");
                }
            }

            const auto require_decision =
                [&](const SceneNodeRecord& node,
                    const std::string_view expected_decision)
                -> lfs::Result<void> {
                if (!node.payload) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A geometry node is missing its payload binding.",
                        std::format("SCNG node {} has no payload binding",
                                    node.uuid.to_string()),
                        "SCNG.nodes.payload");
                }
                const auto matching = std::ranges::count_if(
                    *decisions, [&](const EmbedDecision& decision) {
                        return decision.node_uuid == node.uuid &&
                               decision.payload_fourcc ==
                                   node.payload->fourcc &&
                               decision.decision == expected_decision &&
                               decision.reference_uuid ==
                                   node.payload->reference_uuid;
                    });
                if (matching != 1) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The payload decision log disagrees with the scene.",
                        std::format(
                            "SCNG node {} requires exactly one matching '{}' "
                            "PROJ decision, found {}",
                            node.uuid.to_string(), expected_decision, matching),
                        "PROJ.embed_decisions");
                }
                return {};
            };

            std::set<ChunkKey, ChunkKeyLess> bound_embedded;
            std::optional<lfs::core::Uuid> bound_checkpoint;
            for (const auto& node : *nodes) {
                const bool geometry =
                    node.type == "splat" || node.type == "pointcloud" ||
                    node.type == "mesh";
                if (!geometry) {
                    continue;
                }
                if (!node.payload) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A geometry node is missing its payload binding.",
                        std::format("SCNG node {} type {} has no payload",
                                    node.uuid.to_string(), node.type),
                        "SCNG.nodes.payload");
                }
                const auto& binding = *node.payload;
                const bool is_training = *training && node.uuid == **training;
                if (is_training) {
                    if (node.type != "splat" || binding.fourcc != "CKPT") {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "The training model has the wrong payload authority.",
                            "Training splats bind to CKPT and never to SPLT",
                            "SCNG.training_model_uuid");
                    }
                    if (binding.instance_uuid.is_nil() ||
                        binding.reference_uuid ||
                        !checkpoints.contains(binding.instance_uuid)) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "The training checkpoint payload is missing.",
                            std::format(
                                "SCNG training node {} binds CKPT instance {}, "
                                "but that exact chapter is unavailable",
                                node.uuid.to_string(),
                                binding.instance_uuid.to_string()),
                            "CKPT.instance_uuid");
                    }
                    bound_checkpoint = binding.instance_uuid;
                    continue;
                }

                Fourcc fourcc{};
                if (node.type == "splat" && binding.fourcc == "SPLT") {
                    fourcc = FOURCC_SPLT;
                } else if (node.type == "pointcloud" &&
                           binding.fourcc == "PCLD") {
                    fourcc = FOURCC_PCLD;
                } else if (node.type == "mesh" &&
                           binding.fourcc == "MESH") {
                    fourcc = FOURCC_MESH;
                } else if (node.type == "splat" &&
                           binding.fourcc == "REFS" &&
                           binding.source_kind == "rad" &&
                           binding.reference_uuid &&
                           *binding.reference_uuid ==
                               binding.instance_uuid &&
                           references_by_uuid.contains(
                               *binding.reference_uuid) &&
                           !node.payload_diverged) {
                    if (auto decision = require_decision(node, "external");
                        !decision) {
                        return decision;
                    }
                    continue;
                } else {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A scene geometry payload binding is invalid.",
                        std::format("SCNG node {} type {} binds to {} ({})",
                                    node.uuid.to_string(), node.type,
                                    binding.fourcc, binding.source_kind),
                        "SCNG.nodes.payload");
                }
                if (binding.instance_uuid != node.uuid ||
                    binding.reference_uuid) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "An embedded geometry payload has the wrong identity.",
                        "SPLT/PCLD/MESH instance UUID must equal the node UUID "
                        "and carry no external reference",
                        "SCNG.nodes.payload.instance_uuid");
                }
                const ChunkKey chunk_key{
                    .fourcc = fourcc,
                    .instance_uuid = node.uuid,
                };
                const bool exists =
                    (fourcc == FOURCC_SPLT && splats.contains(node.uuid)) ||
                    (fourcc == FOURCC_PCLD &&
                     point_clouds.contains(node.uuid)) ||
                    (fourcc == FOURCC_MESH && meshes.contains(node.uuid)) ||
                    deferred_geometry_keys.contains(chunk_key);
                const auto hash = hashes.find(chunk_key);
                if (!exists || hash == hashes.end()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "An embedded geometry payload is missing.",
                        std::format("{} instance {} is not available",
                                    binding.fourcc, node.uuid.to_string()),
                        "SCNG.nodes.payload");
                }
                if (auto decision = require_decision(node, "embedded");
                    !decision) {
                    return decision;
                }
                const auto provenance_record =
                    embedded_by_payload.find(
                        payload_identity(node.uuid, binding.fourcc));
                if (provenance_record == embedded_by_payload.end() ||
                    provenance_record->second->content_xxh3_128 !=
                        hash->second) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "Embedded payload provenance is missing or stale.",
                        std::format(
                            "{} instance {} requires an import locator, import "
                            "fingerprint, and matching XXH3-128 content hash",
                            binding.fourcc, node.uuid.to_string()),
                        "PROJ.embedded_payloads");
                }
                bound_embedded.insert(chunk_key);
            }

            const auto ensure_all_bound =
                [&](const auto& payloads, const Fourcc fourcc,
                    const std::string_view name) -> lfs::Result<void> {
                for (const auto& [uuid, ignored] : payloads) {
                    (void)ignored;
                    if (!bound_embedded.contains(
                            ChunkKey{.fourcc = fourcc,
                                     .instance_uuid = uuid})) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "An embedded payload has no scene owner.",
                            std::format("{} instance {} has no matching SCNG node",
                                        name, uuid.to_string()),
                            "SCNG.nodes.payload");
                    }
                }
                return {};
            };
            if (auto result =
                    ensure_all_bound(splats, FOURCC_SPLT, "SPLT");
                !result) {
                return result;
            }
            if (auto result =
                    ensure_all_bound(point_clouds, FOURCC_PCLD, "PCLD");
                !result) {
                return result;
            }
            if (auto result =
                    ensure_all_bound(meshes, FOURCC_MESH, "MESH");
                !result) {
                return result;
            }
            for (const auto& key : deferred_geometry_keys) {
                if (!bound_embedded.contains(key)) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "An unloaded embedded payload has no scene owner.",
                        std::format("{} instance {} has no matching SCNG node",
                                    key.fourcc.to_string(),
                                    key.instance_uuid.to_string()),
                        "SCNG.nodes.payload");
                }
            }

            if (checkpoints.size() !=
                static_cast<std::size_t>(bound_checkpoint.has_value())) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The project contains an unbound training checkpoint.",
                    std::format(
                        "{} CKPT chapters exist but SCNG binds {}",
                        checkpoints.size(),
                        bound_checkpoint ? 1 : 0),
                    "CKPT");
            }
            if (!checkpoints.empty() && !ppisp_payloads.empty()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The project has conflicting PPISP authorities.",
                    "PPIS is valid only for a session without CKPT; training "
                    "PPISP and its controller live inside CKPT",
                    "PPIS");
            }
            if (ppisp_payloads.size() > 1) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The project contains multiple standalone PPISP chapters.",
                    "Only one authoritative PPIS payload is supported",
                    "PPIS");
            }
            for (const auto& [uuid, payload] :
                 ppisp_payloads) {
                if (payload.size() <
                    sizeof(PpispFileHeader)) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The standalone PPISP chapter is truncated.",
                        std::format(
                            "PPIS instance {} has {} bytes; at least {} are required",
                            uuid.to_string(),
                            payload.size(),
                            sizeof(PpispFileHeader)),
                        "PPIS.header");
                }
                PpispFileHeader header{};
                auto header_bytes = std::span<std::byte>(
                    reinterpret_cast<std::byte*>(&header),
                    sizeof(header));
                if (auto read =
                        payload.read_at(0, header_bytes);
                    !read) {
                    return read;
                }
                if (header.magic != PPISP_FILE_MAGIC ||
                    header.version == 0 ||
                    header.version >
                        PPISP_FILE_VERSION ||
                    header.num_cameras == 0 ||
                    header.num_frames == 0 ||
                    (header.flags &
                     ~PPISP_FILE_KNOWN_FLAGS) != 0 ||
                    std::ranges::any_of(
                        header.reserved,
                        [](const std::uint32_t value) {
                            return value != 0;
                        }) ||
                    (header.version < 2 &&
                     (header.flags & (1u << 1)) !=
                         0)) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The standalone PPISP header is invalid.",
                        std::format(
                            "PPIS instance {} header magic={:#x} version={} "
                            "cameras={} frames={} flags={:#x}",
                            uuid.to_string(), header.magic,
                            header.version,
                            header.num_cameras,
                            header.num_frames,
                            header.flags),
                        "PPIS.header");
                }
            }
            if (bound_checkpoint) {
                const auto found = checkpoints.find(*bound_checkpoint);
                assert(found != checkpoints.end());
                std::optional<lfs::core::CheckpointHeader> header;
                auto inspected = found->second.visit_stream(
                    [&](std::istream& stream,
                        const std::uint64_t bytes) -> lfs::Result<void> {
                        auto loaded =
                            lfs::core::load_checkpoint_header(
                                stream, bytes);
                        if (!loaded) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "The embedded checkpoint header is invalid.",
                                loaded.error(),
                                "CKPT.LFKP.header");
                        }
                        header = *loaded;
                        return {};
                    });
                if (!inspected) {
                    return inspected;
                }
                assert(header);
                if (header->num_gaussians == 0) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The training checkpoint has no Gaussian model.",
                        "CKPT header num_gaussians must be non-zero",
                        "CKPT.LFKP.num_gaussians");
                }
            }
            return {};
        }

        [[nodiscard]] lfs::Result<void>
        refresh_source_rows(const std::filesystem::path& path,
                            const ReaderOptions& options = {}) {
            auto reader = ProjectReader::open(path, options);
            if (!reader) {
                return lfs::Result<void>::failure(std::move(reader).error());
            }
            auto shared_reader = std::make_shared<ProjectReader>(
                std::move(*reader));
            std::map<ChunkKey, SourceRow, ChunkKeyLess> refreshed;
            std::unordered_map<lfs::core::Uuid, LazyChunkValue>
                refreshed_checkpoints;
            std::unordered_map<lfs::core::Uuid, LazyChunkValue>
                refreshed_ppisp;
            for (const auto& row : shared_reader->chunks()) {
                if (!row.is_live()) {
                    continue;
                }
                auto proof = shared_reader->make_clean_proof(
                    row, DOCUMENT_CLEAN_BASELINE);
                if (!proof) {
                    return lfs::Result<void>::failure(
                        std::move(proof).error());
                }
                if (is_lazy_binary_fourcc(row.key.fourcc)) {
                    auto lazy = std::make_unique<LazyChunkValue::Impl>();
                    lazy->reader = shared_reader;
                    lazy->source = row;
                    lazy->proof = std::move(*proof);
                    lazy->snapshot_uuid = row.key.instance_uuid;
                    auto& destination =
                        row.key.fourcc == FOURCC_CKPT
                            ? refreshed_checkpoints
                            : refreshed_ppisp;
                    destination.emplace(
                        row.key.instance_uuid,
                        LazyChunkValue(std::move(lazy)));
                } else {
                    refreshed.emplace(
                        row.key,
                        SourceRow{
                            .info = row,
                            .proof = std::move(*proof),
                        });
                }
            }
            source_rows = std::move(refreshed);
            normalized_source_keys.clear();
            lazy_source_keys.clear();
            for (const auto& [uuid, ignored] :
                 refreshed_checkpoints) {
                (void)ignored;
                lazy_source_keys.insert(ChunkKey{
                    .fourcc = FOURCC_CKPT,
                    .instance_uuid = uuid,
                });
            }
            for (const auto& [uuid, ignored] :
                 refreshed_ppisp) {
                (void)ignored;
                lazy_source_keys.insert(ChunkKey{
                    .fourcc = FOURCC_PPIS,
                    .instance_uuid = uuid,
                });
            }
            checkpoints = std::move(refreshed_checkpoints);
            ppisp_payloads = std::move(refreshed_ppisp);
            dirty.clear();
            source_path = path;
            source_reader = std::move(shared_reader);
            generation = source_reader->commit().generation;
            return {};
        }
    };

    ProjectDocument::ProjectDocument(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    ProjectHydrationPlan::ProjectHydrationPlan(
        std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    ProjectHydrationPlan::ProjectHydrationPlan(
        ProjectHydrationPlan&&) noexcept = default;
    ProjectHydrationPlan&
    ProjectHydrationPlan::operator=(
        ProjectHydrationPlan&&) noexcept = default;
    ProjectHydrationPlan::~ProjectHydrationPlan() = default;

    const ProjectDocumentHydrationReport&
    ProjectHydrationPlan::report() const noexcept {
        assert(impl_);
        return impl_->report;
    }

    ProjectDocument::ProjectDocument(ProjectDocument&&) noexcept = default;
    ProjectDocument&
    ProjectDocument::operator=(ProjectDocument&&) noexcept = default;
    ProjectDocument::~ProjectDocument() = default;

    lfs::Result<ProjectDocument>
    ProjectDocument::create(const lfs::core::Uuid& project_uuid,
                            std::uint64_t creation_time_unix_ns) {
        if (project_uuid.is_nil()) {
            return fail<ProjectDocument>(
                lfs::ErrorCode::InvalidArgument,
                "The project UUID cannot be null.",
                "ProjectDocument::create requires a non-null UUID",
                "PROJ.project_uuid");
        }
        if (creation_time_unix_ns == 0) {
            creation_time_unix_ns = unix_time_ns();
        }
        auto impl = std::make_unique<Impl>();
        impl->project_uuid = project_uuid;
        if (auto result = impl->project.set_manifest(ProjectManifest{});
            !result) {
            return std::move(result).error();
        }
        if (auto result = impl->project.set_project_uuid(project_uuid);
            !result) {
            return std::move(result).error();
        }
        if (auto result = impl->project.set_created_at_unix_ns(
                creation_time_unix_ns);
            !result) {
            return std::move(result).error();
        }
        if (auto result = impl->project.set_modified_at_unix_ns(
                creation_time_unix_ns);
            !result) {
            return std::move(result).error();
        }
        if (auto result = impl->project.set_dataset_reference(std::nullopt);
            !result) {
            return std::move(result).error();
        }
        if (auto result = impl->project.set_project_lineage(
                std::span<const lfs::core::Uuid>{});
            !result) {
            return std::move(result).error();
        }
        if (auto result =
                impl->project.set_georeference(ProjectGeoreference{});
            !result) {
            return std::move(result).error();
        }
        if (auto result = impl->parameters.set_snapshot(
                default_parameter_snapshot());
            !result) {
            return std::move(result).error();
        }
        return ProjectDocument(std::move(impl));
    }

    lfs::Result<ProjectDocument>
    ProjectDocument::open(const std::filesystem::path& path,
                          const ProjectDocumentOpenOptions& options) {
        const auto open_started =
            std::chrono::steady_clock::now();
        auto normalized = normalized_absolute_path(path);
        if (!normalized) {
            return std::move(normalized).error();
        }
        const auto normalized_at =
            std::chrono::steady_clock::now();
        auto reader = ProjectReader::open(*normalized, options.reader);
        if (!reader) {
            return std::move(reader).error();
        }
        const auto reader_opened_at =
            std::chrono::steady_clock::now();
        auto shared_reader = std::make_shared<ProjectReader>(
            std::move(*reader));
        auto impl = std::make_unique<Impl>();
        impl->project_uuid = shared_reader->superblock().project_uuid;
        impl->source_path = *normalized;
        impl->source_reader = shared_reader;
        impl->generation = shared_reader->commit().generation;

        bool have_project = false;
        bool have_references = false;
        bool have_scene = false;
        bool have_selection = false;
        bool have_parameters = false;
        bool have_gui_layout = false;
        bool have_view = false;
        bool have_editor = false;
        bool have_sequencer = false;
        bool have_metrics = false;
        double chapter_read_ms = 0.0;
        const auto chapter_scan_started =
            std::chrono::steady_clock::now();

        for (const auto& row : shared_reader->chunks()) {
            if (!row.is_live()) {
                continue;
            }
            auto proof = shared_reader->make_clean_proof(
                row, DOCUMENT_CLEAN_BASELINE);
            if (!proof) {
                return std::move(proof).error();
            }
            const bool lazy_binary =
                is_lazy_binary_fourcc(row.key.fourcc);
            const bool managed =
                is_project_managed_fourcc(row.key.fourcc);
            const bool thumbnail =
                row.key.fourcc == FOURCC_THMB;
            // Lazy binary (CKPT/PPIS): Stored, framed Zstd (CHUNK_ZSTD_V1), or
            // framed ByteShuffleZstd (CHUNK_BYTESHUFFLE_ZSTD_V1).
            // Any other encoding is opaque/unsupported.
            const bool lazy_binary_encoding_ok =
                !lazy_binary ||
                row.compression == Compression::Stored ||
                row.compression == Compression::ZstdFramed ||
                row.compression == Compression::ByteShuffleZstdFramed;
            const bool unsupported_known_encoding =
                (managed || lazy_binary || thumbnail) &&
                (row.chunk_version != P3_CHUNK_VERSION ||
                 !lazy_binary_encoding_ok);
            const bool opaque =
                unsupported_known_encoding ||
                (!managed && !lazy_binary && !thumbnail);
            if (opaque || thumbnail) {
                impl->source_rows.emplace(
                    row.key,
                    Impl::SourceRow{
                        .info = row,
                        .proof = std::move(*proof),
                        .opaque = opaque,
                    });
                if (opaque &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(OPAQUE_CHUNK_PRESERVATION)) {
                    impl->missing_opaque_preservation_capability = true;
                }
                continue;
            }
            if (lazy_binary) {
                auto lazy = std::make_unique<LazyChunkValue::Impl>();
                lazy->reader = shared_reader;
                lazy->source = row;
                lazy->proof = std::move(*proof);
                lazy->snapshot_uuid = row.key.instance_uuid;
                auto& destination =
                    row.key.fourcc == FOURCC_CKPT
                        ? impl->checkpoints
                        : impl->ppisp_payloads;
                if (!destination
                         .emplace(
                             row.key.instance_uuid,
                             LazyChunkValue(std::move(lazy)))
                         .second) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains a duplicate binary chapter.",
                        row.key_string(),
                        "chunk.instance_uuid");
                }
                impl->lazy_source_keys.insert(row.key);
                continue;
            }
            impl->source_rows.emplace(
                row.key,
                Impl::SourceRow{
                    .info = row,
                    .proof = std::move(*proof),
                    .opaque = false,
                });
            if (is_singleton_fourcc(row.key.fourcc) &&
                row.key.instance_uuid != impl->project_uuid) {
                return fail<ProjectDocument>(
                    lfs::ErrorCode::DataLoss,
                    "A singleton project chapter has the wrong identity.",
                    std::format("{} instance UUID {} differs from project UUID {}",
                                row.key.fourcc.to_string(),
                                row.key.instance_uuid.to_string(),
                                impl->project_uuid.to_string()),
                    "chunk.instance_uuid");
            }
            const bool geometry_payload =
                row.key.fourcc == FOURCC_SPLT ||
                row.key.fourcc == FOURCC_PCLD ||
                row.key.fourcc == FOURCC_MESH;
            const auto materialized_max =
                detail::max_materialized_bytes_for(row);
            const bool oversized_splat =
                row.key.fourcc == FOURCC_SPLT &&
                (row.stored_bytes > materialized_max ||
                 row.uncompressed_bytes > materialized_max);
            if ((options.defer_geometry_payloads && geometry_payload) ||
                oversized_splat) {
                impl->deferred_geometry_keys.insert(row.key);
                continue;
            }
            const auto chapter_read_started =
                std::chrono::steady_clock::now();
            auto bytes = shared_reader->read_chunk(row);
            chapter_read_ms +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    chapter_read_started)
                    .count();
            if (!bytes) {
                return std::move(bytes).error();
            }
            if (row.key.fourcc == FOURCC_PROJ) {
                if (have_project) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate PROJ chapters.",
                        "Only one PROJ instance is allowed", "PROJ");
                }
                auto chapter = ProjectChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->project = std::move(*chapter);
                if (has_unknown_json_root(
                        FOURCC_PROJ, impl->project.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_project = true;
            } else if (row.key.fourcc == FOURCC_REFS) {
                if (have_references) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate REFS chapters.",
                        "Only one REFS instance is allowed", "REFS");
                }
                auto chapter = ReferencesChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->references = std::move(*chapter);
                if (has_unknown_json_root(
                        FOURCC_REFS, impl->references.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_references = true;
            } else if (row.key.fourcc == FOURCC_SCNG) {
                if (have_scene) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate SCNG chapters.",
                        "Only one SCNG instance is allowed", "SCNG");
                }
                auto chapter = SceneGraphChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->scene_graph = std::move(*chapter);
                if (has_unknown_json_root(
                        FOURCC_SCNG, impl->scene_graph.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_scene = true;
            } else if (row.key.fourcc == FOURCC_SELM) {
                if (have_selection) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate SELM chapters.",
                        "Only one SELM instance is allowed", "SELM");
                }
                auto chapter = decode_selection_chapter(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->selection = std::move(*chapter);
                have_selection = true;
            } else if (row.key.fourcc == FOURCC_PRMS) {
                if (have_parameters) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate PRMS chapters.",
                        "Only one PRMS instance is allowed", "PRMS");
                }
                auto chapter = ParametersChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->parameters = std::move(*chapter);
                if (has_unknown_json_root(
                        FOURCC_PRMS, impl->parameters.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_parameters = true;
            } else if (row.key.fourcc == FOURCC_GUIL) {
                if (have_gui_layout) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate GUIL chapters.",
                        "Only one GUIL instance is allowed", "GUIL");
                }
                auto source_dom = JsonChapterDom::from_bytes(*bytes);
                if (!source_dom) {
                    return std::move(source_dom).error();
                }
                auto chapter = GuiLayoutChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->gui_layout = std::move(*chapter);
                if (source_dom->dump() != impl->gui_layout.dom().dump()) {
                    impl->normalized_source_keys.insert(row.key);
                }
                if (has_unknown_json_root(
                        FOURCC_GUIL, impl->gui_layout.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_gui_layout = true;
            } else if (row.key.fourcc == FOURCC_VIEW) {
                if (have_view) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate VIEW chapters.",
                        "Only one VIEW instance is allowed", "VIEW");
                }
                auto chapter = ViewSessionChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->view = std::move(*chapter);
                if (has_unknown_json_root(
                        FOURCC_VIEW, impl->view.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_view = true;
            } else if (row.key.fourcc == FOURCC_EDTR) {
                if (have_editor) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate EDTR chapters.",
                        "Only one EDTR instance is allowed", "EDTR");
                }
                auto chapter = EditorSessionChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->editor = std::move(*chapter);
                if (has_unknown_json_root(
                        FOURCC_EDTR, impl->editor.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_editor = true;
            } else if (row.key.fourcc == FOURCC_SEQR) {
                if (have_sequencer) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate SEQR chapters.",
                        "Only one SEQR instance is allowed", "SEQR");
                }
                auto chapter = SequencerSessionChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->sequencer = std::move(*chapter);
                if (has_unknown_json_root(
                        FOURCC_SEQR, impl->sequencer.dom()) &&
                    !shared_reader->commit()
                         .required_writer_capabilities
                         .contains(RETAINED_JSON_FIELDS)) {
                    impl->missing_retained_json_capability = true;
                }
                have_sequencer = true;
            } else if (row.key.fourcc == FOURCC_METR) {
                if (have_metrics) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::DataLoss,
                        "The project contains duplicate METR chapters.",
                        "Only one METR instance is allowed", "METR");
                }
                auto chapter = MetricsChapter::from_bytes(*bytes);
                if (!chapter) {
                    return std::move(chapter).error();
                }
                impl->metrics = std::move(*chapter);
                have_metrics = true;
            } else if (row.key.fourcc == FOURCC_SPLT) {
                const auto content_hash = xxh3_128(*bytes);
                auto payload = SplatChapterPayload::from_lfsp(std::move(*bytes));
                if (!payload) {
                    return std::move(payload).error();
                }
                impl->splats.emplace(row.key.instance_uuid,
                                     std::move(*payload));
                impl->content_hashes.emplace(row.key, content_hash);
            } else if (row.key.fourcc == FOURCC_PCLD) {
                auto payload =
                    decode_point_cloud_payload(*bytes, options.geometry);
                if (!payload) {
                    return std::move(payload).error();
                }
                impl->point_clouds.emplace(row.key.instance_uuid,
                                           std::move(*payload));
                impl->content_hashes.emplace(row.key, xxh3_128(*bytes));
            } else if (row.key.fourcc == FOURCC_MESH) {
                auto payload = decode_mesh_payload(*bytes, options.geometry);
                if (!payload) {
                    return std::move(payload).error();
                }
                impl->meshes.emplace(row.key.instance_uuid,
                                     std::move(*payload));
                impl->content_hashes.emplace(row.key, xxh3_128(*bytes));
            }
        }
        const auto chapter_scan_finished =
            std::chrono::steady_clock::now();

        if (!have_project || !have_references || !have_scene ||
            !have_selection || !have_parameters) {
            return fail<ProjectDocument>(
                lfs::ErrorCode::DataLoss,
                "The project is missing a required core chapter.",
                std::format(
                    "required P3 singleton presence: PROJ={}, REFS={}, SCNG={}, "
                    "SELM={}, PRMS={}",
                    have_project, have_references, have_scene, have_selection,
                    have_parameters),
                "index.core_chapters");
        }
        if (!impl->deferred_geometry_keys.empty()) {
            auto provenance = impl->project.embedded_payload_provenance();
            if (!provenance) {
                return std::move(provenance).error();
            }
            for (const auto& record : *provenance) {
                Fourcc fourcc{};
                if (record.fourcc == "SPLT") {
                    fourcc = FOURCC_SPLT;
                } else if (record.fourcc == "PCLD") {
                    fourcc = FOURCC_PCLD;
                } else if (record.fourcc == "MESH") {
                    fourcc = FOURCC_MESH;
                } else {
                    continue;
                }
                const ChunkKey key{
                    .fourcc = fourcc,
                    .instance_uuid = record.node_uuid,
                };
                if (impl->deferred_geometry_keys.contains(key)) {
                    impl->content_hashes.insert_or_assign(
                        key, record.content_xxh3_128);
                }
            }
        }
        if (auto valid = impl->validate(impl->project, impl->content_hashes);
            !valid) {
            return std::move(valid).error();
        }
        const auto open_finished =
            std::chrono::steady_clock::now();
        const auto milliseconds =
            [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(
                           end - begin)
                    .count();
            };
        const double chapter_scan_ms =
            milliseconds(
                chapter_scan_started,
                chapter_scan_finished);
        LOG_DEBUG(
            "Project document open stages: path={} deferred_geometry={} normalize={:.3f} ms reader_parse={:.3f} ms chapter_read={:.3f} ms chapter_decode_scan={:.3f} ms validate={:.3f} ms total={:.3f} ms",
            normalized->string(),
            options.defer_geometry_payloads,
            milliseconds(open_started, normalized_at),
            milliseconds(normalized_at, reader_opened_at),
            chapter_read_ms,
            std::max(
                0.0,
                chapter_scan_ms - chapter_read_ms),
            milliseconds(
                chapter_scan_finished,
                open_finished),
            milliseconds(open_started, open_finished));
        return ProjectDocument(std::move(impl));
    }

    const std::optional<std::filesystem::path>&
    ProjectDocument::source_path() const noexcept {
        return impl_->source_path;
    }

    const ProjectReader* ProjectDocument::source_reader() const noexcept {
        return impl_->source_reader.get();
    }

    std::optional<lfs::core::Uuid>
    ProjectDocument::source_commit_uuid() const noexcept {
        if (!impl_->source_reader) {
            return std::nullopt;
        }
        return impl_->source_reader->commit().commit_uuid;
    }

    std::span<const ProjectDocumentDegradedState>
    ProjectDocument::degraded_states() const noexcept {
        return impl_->degraded_states;
    }

    const lfs::core::Uuid&
    ProjectDocument::project_uuid() const noexcept {
        return impl_->project_uuid;
    }

    std::uint64_t ProjectDocument::generation() const noexcept {
        return impl_->generation;
    }

    std::uint64_t ProjectDocument::dirty_epoch() const noexcept {
        return impl_->dirty_epoch;
    }

    bool ProjectDocument::dirty() const noexcept {
        if (!impl_->source_path) {
            return true;
        }
        return !impl_->dirty.empty();
    }

    std::vector<std::string>
    ProjectDocument::dirty_chapters() const {
        std::set<std::string> names;
        if (!impl_->source_path) {
            names = {
                "PROJ",
                "REFS",
                "SCNG",
                "SELM",
                "PRMS",
                "GUIL",
                "VIEW",
                "EDTR",
                "SEQR",
                "METR",
            };
            for (const auto& [uuid, ignored] : impl_->splats) {
                (void)uuid;
                (void)ignored;
                names.insert("SPLT");
            }
            for (const auto& [uuid, ignored] : impl_->point_clouds) {
                (void)uuid;
                (void)ignored;
                names.insert("PCLD");
            }
            for (const auto& [uuid, ignored] : impl_->meshes) {
                (void)uuid;
                (void)ignored;
                names.insert("MESH");
            }
            for (const auto& [uuid, ignored] : impl_->checkpoints) {
                (void)uuid;
                (void)ignored;
                names.insert("CKPT");
            }
            for (const auto& [uuid, ignored] : impl_->ppisp_payloads) {
                (void)uuid;
                (void)ignored;
                names.insert("PPIS");
            }
        } else {
            for (const auto& key : impl_->dirty) {
                names.insert(key.fourcc.to_string());
            }
        }
        return {names.begin(), names.end()};
    }

    std::vector<ProjectDocumentPayloadState>
    ProjectDocument::payload_states() const {
        std::vector<ProjectDocumentPayloadState> result;
        const auto append_loaded =
            [&result](const auto& payloads, const Fourcc fourcc) {
                for (const auto& [uuid, ignored] : payloads) {
                    (void)ignored;
                    result.push_back({
                        .fourcc = fourcc,
                        .instance_uuid = uuid,
                        .loaded = true,
                    });
                }
            };
        append_loaded(impl_->splats, FOURCC_SPLT);
        append_loaded(impl_->point_clouds, FOURCC_PCLD);
        append_loaded(impl_->meshes, FOURCC_MESH);
        for (const auto& key : impl_->deferred_geometry_keys) {
            result.push_back({
                .fourcc = key.fourcc,
                .instance_uuid = key.instance_uuid,
                .loaded = false,
            });
        }
        std::ranges::sort(
            result, [](const auto& lhs, const auto& rhs) {
                const auto left =
                    std::pair{lhs.fourcc.bytes, lhs.instance_uuid.bytes};
                const auto right =
                    std::pair{rhs.fourcc.bytes, rhs.instance_uuid.bytes};
                return left < right;
            });
        return result;
    }

    const ProjectChapter& ProjectDocument::project() const noexcept {
        return impl_->project;
    }

    ProjectChapter& ProjectDocument::edit_project() noexcept {
        impl_->mark(FOURCC_PROJ);
        return impl_->project;
    }

    const ReferencesChapter& ProjectDocument::references() const noexcept {
        return impl_->references;
    }

    ReferencesChapter& ProjectDocument::edit_references() noexcept {
        impl_->mark(FOURCC_REFS);
        return impl_->references;
    }

    const SceneGraphChapter& ProjectDocument::scene_graph() const noexcept {
        return impl_->scene_graph;
    }

    SceneGraphChapter& ProjectDocument::edit_scene_graph() noexcept {
        impl_->mark(FOURCC_SCNG);
        return impl_->scene_graph;
    }

    const SelectionChapter& ProjectDocument::selection() const noexcept {
        return impl_->selection;
    }

    SelectionChapter& ProjectDocument::edit_selection() noexcept {
        impl_->mark(FOURCC_SELM);
        return impl_->selection;
    }

    const ParametersChapter& ProjectDocument::parameters() const noexcept {
        return impl_->parameters;
    }

    ParametersChapter& ProjectDocument::edit_parameters() noexcept {
        impl_->mark(FOURCC_PRMS);
        return impl_->parameters;
    }

    const GuiLayoutChapter& ProjectDocument::gui_layout() const noexcept {
        return impl_->gui_layout;
    }

    GuiLayoutChapter& ProjectDocument::edit_gui_layout() noexcept {
        impl_->mark(FOURCC_GUIL);
        return impl_->gui_layout;
    }

    const ViewSessionChapter& ProjectDocument::view() const noexcept {
        return impl_->view;
    }

    ViewSessionChapter& ProjectDocument::edit_view() noexcept {
        impl_->mark(FOURCC_VIEW);
        return impl_->view;
    }

    const EditorSessionChapter& ProjectDocument::editor() const noexcept {
        return impl_->editor;
    }

    EditorSessionChapter& ProjectDocument::edit_editor() noexcept {
        impl_->mark(FOURCC_EDTR);
        return impl_->editor;
    }

    const SequencerSessionChapter&
    ProjectDocument::sequencer() const noexcept {
        return impl_->sequencer;
    }

    SequencerSessionChapter&
    ProjectDocument::edit_sequencer() noexcept {
        impl_->mark(FOURCC_SEQR);
        return impl_->sequencer;
    }

    const MetricsChapter& ProjectDocument::metrics() const noexcept {
        return impl_->metrics;
    }

    MetricsChapter& ProjectDocument::edit_metrics() noexcept {
        impl_->mark(FOURCC_METR);
        return impl_->metrics;
    }

    const LazyChunkValue*
    ProjectDocument::find_checkpoint(
        const lfs::core::Uuid& instance_uuid) const noexcept {
        const auto found = impl_->checkpoints.find(instance_uuid);
        return found == impl_->checkpoints.end()
                   ? nullptr
                   : &found->second;
    }

    void ProjectDocument::drop_checkpoint_clean_proof_for_testing(
        const lfs::core::Uuid& instance_uuid) {
        const auto found = impl_->checkpoints.find(instance_uuid);
        if (found == impl_->checkpoints.end()) {
            return;
        }
        found->second.drop_clean_proof_for_testing();
    }

    lfs::Result<void>
    ProjectDocument::set_checkpoint(
        const lfs::core::Uuid& instance_uuid,
        LazyChunkValue payload) {
        if (instance_uuid.is_nil() ||
            payload.snapshot_uuid() != instance_uuid ||
            payload.size() == 0) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The checkpoint snapshot identity is invalid.",
                "CKPT instance UUID, snapshot UUID, and every staged piece "
                "must share one non-null identity; payload must be non-empty",
                "CKPT.instance_uuid");
        }
        impl_->checkpoints.insert_or_assign(
            instance_uuid, std::move(payload));
        impl_->mark(FOURCC_CKPT, instance_uuid);
        return {};
    }

    bool ProjectDocument::remove_checkpoint(
        const lfs::core::Uuid& instance_uuid) {
        const ChunkKey key{
            .fourcc = FOURCC_CKPT,
            .instance_uuid = instance_uuid,
        };
        const bool removed =
            impl_->checkpoints.erase(instance_uuid) != 0;
        if (removed) {
            impl_->mark(key.fourcc, key.instance_uuid);
        }
        return removed;
    }

    std::vector<lfs::core::Uuid>
    ProjectDocument::checkpoint_uuids() const {
        return sorted_uuids(impl_->checkpoints);
    }

    const LazyChunkValue*
    ProjectDocument::find_ppisp(
        const lfs::core::Uuid& instance_uuid) const noexcept {
        const auto found = impl_->ppisp_payloads.find(instance_uuid);
        return found == impl_->ppisp_payloads.end()
                   ? nullptr
                   : &found->second;
    }

    lfs::Result<void>
    ProjectDocument::set_ppisp(
        const lfs::core::Uuid& instance_uuid,
        LazyChunkValue payload) {
        if (instance_uuid.is_nil() ||
            payload.snapshot_uuid() != instance_uuid ||
            payload.size() == 0) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The PPISP chapter identity is invalid.",
                "PPIS instance UUID and staged payload UUID must match and "
                "the payload must be non-empty",
                "PPIS.instance_uuid");
        }
        impl_->ppisp_payloads.insert_or_assign(
            instance_uuid, std::move(payload));
        impl_->mark(FOURCC_PPIS, instance_uuid);
        return {};
    }

    lfs::Result<void> ProjectDocument::set_georeference(
        const ProjectGeoreference& value) {
        if (auto result =
                impl_->project.set_georeference(value);
            !result) {
            return result;
        }
        impl_->mark(FOURCC_PROJ);
        return {};
    }

    lfs::Result<void> ProjectDocument::capture_georeference(
        const lfs::io::LoadResult& load_result) {
        ProjectGeoreference value;
        if (load_result.georeference) {
            value = ProjectGeoreference{
                .crs = load_result.georeference->crs,
                .world_origin =
                    load_result.georeference->world_origin,
                .world_unit_scale =
                    load_result.georeference->world_unit_scale,
                .world_origin_provenance =
                    project_provenance(
                        load_result.georeference
                            ->world_origin_provenance),
            };
        }
        return set_georeference(value);
    }

    const SplatChapterPayload*
    ProjectDocument::find_splat(const lfs::core::Uuid& node_uuid) const noexcept {
        const auto found = impl_->splats.find(node_uuid);
        return found == impl_->splats.end() ? nullptr : &found->second;
    }

    SplatChapterPayload*
    ProjectDocument::edit_splat(const lfs::core::Uuid& node_uuid) noexcept {
        const auto found = impl_->splats.find(node_uuid);
        if (found == impl_->splats.end()) {
            return nullptr;
        }
        impl_->mark(FOURCC_SPLT, node_uuid);
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_SPLT, .instance_uuid = node_uuid});
        return &found->second;
    }

    lfs::Result<void>
    ProjectDocument::set_splat(const lfs::core::Uuid& node_uuid,
                               SplatChapterPayload payload) {
        if (node_uuid.is_nil()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The splat node UUID cannot be null.",
                "SPLT instance UUID must be non-null",
                "SPLT.instance_uuid");
        }
        impl_->splats.insert_or_assign(node_uuid, std::move(payload));
        impl_->mark(FOURCC_SPLT, node_uuid);
        const ChunkKey key{
            .fourcc = FOURCC_SPLT,
            .instance_uuid = node_uuid,
        };
        impl_->deferred_geometry_keys.erase(key);
        impl_->content_hashes.erase(key);
        return {};
    }

    bool ProjectDocument::remove_splat(const lfs::core::Uuid& node_uuid) {
        const ChunkKey key{
            .fourcc = FOURCC_SPLT,
            .instance_uuid = node_uuid,
        };
        impl_->content_hashes.erase(key);
        const bool removed_loaded = impl_->splats.erase(node_uuid) != 0;
        const bool removed_deferred =
            impl_->deferred_geometry_keys.erase(key) != 0;
        const bool removed = removed_loaded || removed_deferred;
        if (removed) {
            impl_->mark(key.fourcc, key.instance_uuid);
        }
        return removed;
    }

    const PointCloudPayload*
    ProjectDocument::find_point_cloud(
        const lfs::core::Uuid& node_uuid) const noexcept {
        const auto found = impl_->point_clouds.find(node_uuid);
        return found == impl_->point_clouds.end() ? nullptr : &found->second;
    }

    PointCloudPayload*
    ProjectDocument::edit_point_cloud(
        const lfs::core::Uuid& node_uuid) noexcept {
        const auto found = impl_->point_clouds.find(node_uuid);
        if (found == impl_->point_clouds.end()) {
            return nullptr;
        }
        impl_->mark(FOURCC_PCLD, node_uuid);
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_PCLD, .instance_uuid = node_uuid});
        return &found->second;
    }

    lfs::Result<void>
    ProjectDocument::set_point_cloud(const lfs::core::Uuid& node_uuid,
                                     PointCloudPayload payload) {
        if (node_uuid.is_nil()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The point-cloud node UUID cannot be null.",
                "PCLD instance UUID must be non-null",
                "PCLD.instance_uuid");
        }
        impl_->point_clouds.insert_or_assign(node_uuid, std::move(payload));
        impl_->mark(FOURCC_PCLD, node_uuid);
        const ChunkKey key{
            .fourcc = FOURCC_PCLD,
            .instance_uuid = node_uuid,
        };
        impl_->deferred_geometry_keys.erase(key);
        impl_->content_hashes.erase(key);
        return {};
    }

    bool ProjectDocument::remove_point_cloud(
        const lfs::core::Uuid& node_uuid) {
        const ChunkKey key{
            .fourcc = FOURCC_PCLD,
            .instance_uuid = node_uuid,
        };
        impl_->content_hashes.erase(key);
        const bool removed_loaded =
            impl_->point_clouds.erase(node_uuid) != 0;
        const bool removed_deferred =
            impl_->deferred_geometry_keys.erase(key) != 0;
        const bool removed = removed_loaded || removed_deferred;
        if (removed) {
            impl_->mark(key.fourcc, key.instance_uuid);
        }
        return removed;
    }

    const MeshPayload*
    ProjectDocument::find_mesh(
        const lfs::core::Uuid& node_uuid) const noexcept {
        const auto found = impl_->meshes.find(node_uuid);
        return found == impl_->meshes.end() ? nullptr : &found->second;
    }

    MeshPayload*
    ProjectDocument::edit_mesh(const lfs::core::Uuid& node_uuid) noexcept {
        const auto found = impl_->meshes.find(node_uuid);
        if (found == impl_->meshes.end()) {
            return nullptr;
        }
        impl_->mark(FOURCC_MESH, node_uuid);
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_MESH, .instance_uuid = node_uuid});
        return &found->second;
    }

    lfs::Result<void>
    ProjectDocument::set_mesh(const lfs::core::Uuid& node_uuid,
                              MeshPayload payload) {
        if (node_uuid.is_nil()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The mesh node UUID cannot be null.",
                "MESH instance UUID must be non-null",
                "MESH.instance_uuid");
        }
        impl_->meshes.insert_or_assign(node_uuid, std::move(payload));
        impl_->mark(FOURCC_MESH, node_uuid);
        const ChunkKey key{
            .fourcc = FOURCC_MESH,
            .instance_uuid = node_uuid,
        };
        impl_->deferred_geometry_keys.erase(key);
        impl_->content_hashes.erase(key);
        return {};
    }

    bool ProjectDocument::remove_mesh(const lfs::core::Uuid& node_uuid) {
        const ChunkKey key{
            .fourcc = FOURCC_MESH,
            .instance_uuid = node_uuid,
        };
        impl_->content_hashes.erase(key);
        const bool removed_loaded = impl_->meshes.erase(node_uuid) != 0;
        const bool removed_deferred =
            impl_->deferred_geometry_keys.erase(key) != 0;
        const bool removed = removed_loaded || removed_deferred;
        if (removed) {
            impl_->mark(key.fourcc, key.instance_uuid);
        }
        return removed;
    }

    void ProjectDocument::remove_geometry_payloads_not_bound_by_scene() {
        auto nodes = impl_->scene_graph.nodes();
        if (!nodes) {
            return;
        }

        std::set<ChunkKey, ChunkKeyLess> bound;
        for (const auto& node : *nodes) {
            if (!node.payload) {
                continue;
            }
            const auto& binding = *node.payload;
            Fourcc fourcc{};
            if (node.type == "splat" && binding.fourcc == "SPLT") {
                fourcc = FOURCC_SPLT;
            } else if (node.type == "pointcloud" &&
                       binding.fourcc == "PCLD") {
                fourcc = FOURCC_PCLD;
            } else if (node.type == "mesh" && binding.fourcc == "MESH") {
                fourcc = FOURCC_MESH;
            } else {
                continue;
            }
            bound.insert(ChunkKey{
                .fourcc = fourcc,
                .instance_uuid = node.uuid,
            });
        }

        std::set<ChunkKey, ChunkKeyLess> orphans;
        const auto consider = [&](const ChunkKey& key) {
            if ((key.fourcc == FOURCC_SPLT || key.fourcc == FOURCC_PCLD ||
                 key.fourcc == FOURCC_MESH) &&
                !bound.contains(key)) {
                orphans.insert(key);
            }
        };
        for (const auto& [uuid, ignored] : impl_->splats) {
            (void)ignored;
            consider(ChunkKey{
                .fourcc = FOURCC_SPLT,
                .instance_uuid = uuid,
            });
        }
        for (const auto& [uuid, ignored] : impl_->point_clouds) {
            (void)ignored;
            consider(ChunkKey{
                .fourcc = FOURCC_PCLD,
                .instance_uuid = uuid,
            });
        }
        for (const auto& [uuid, ignored] : impl_->meshes) {
            (void)ignored;
            consider(ChunkKey{
                .fourcc = FOURCC_MESH,
                .instance_uuid = uuid,
            });
        }
        for (const auto& key : impl_->deferred_geometry_keys) {
            consider(key);
        }
        for (const auto& [key, ignored] : impl_->content_hashes) {
            (void)ignored;
            consider(key);
        }

        for (const auto& key : orphans) {
            bool removed = false;
            if (key.fourcc == FOURCC_SPLT) {
                removed = remove_splat(key.instance_uuid);
            } else if (key.fourcc == FOURCC_PCLD) {
                removed = remove_point_cloud(key.instance_uuid);
            } else if (key.fourcc == FOURCC_MESH) {
                removed = remove_mesh(key.instance_uuid);
            }
            if (!removed) {
                impl_->content_hashes.erase(key);
                impl_->mark(key.fourcc, key.instance_uuid);
            }
        }
    }

    std::vector<lfs::core::Uuid> ProjectDocument::splat_uuids() const {
        auto result = sorted_uuids(impl_->splats);
        for (const auto& key : impl_->deferred_geometry_keys) {
            if (key.fourcc == FOURCC_SPLT) {
                result.push_back(key.instance_uuid);
            }
        }
        std::ranges::sort(result, {}, [](const lfs::core::Uuid& uuid) {
            return uuid.bytes;
        });
        return result;
    }

    std::vector<lfs::core::Uuid>
    ProjectDocument::point_cloud_uuids() const {
        auto result = sorted_uuids(impl_->point_clouds);
        for (const auto& key : impl_->deferred_geometry_keys) {
            if (key.fourcc == FOURCC_PCLD) {
                result.push_back(key.instance_uuid);
            }
        }
        std::ranges::sort(result, {}, [](const lfs::core::Uuid& uuid) {
            return uuid.bytes;
        });
        return result;
    }

    std::vector<lfs::core::Uuid> ProjectDocument::mesh_uuids() const {
        auto result = sorted_uuids(impl_->meshes);
        for (const auto& key : impl_->deferred_geometry_keys) {
            if (key.fourcc == FOURCC_MESH) {
                result.push_back(key.instance_uuid);
            }
        }
        std::ranges::sort(result, {}, [](const lfs::core::Uuid& uuid) {
            return uuid.bytes;
        });
        return result;
    }

    lfs::Result<ProjectDocumentSaveReport>
    ProjectDocument::save(const std::filesystem::path& path,
                          const ProjectDocumentSaveOptions& options) {
        return save_impl(path, options, nullptr);
    }

    lfs::Result<ProjectDocumentSaveReport>
    ProjectDocument::save_autosave(
        const std::filesystem::path& sidecar_path,
        const ProjectDocumentAutosaveOptions& options) {
        ProjectDocumentSaveOptions save_options;
        save_options.commit = CommitOptions{
            .kind = CommitKind::Autosave,
            .commit_uuid =
                options.commit_uuid.is_nil()
                    ? lfs::core::generate_uuid_v4()
                    : options.commit_uuid,
            .snapshot_uuid = options.snapshot_uuid,
            .wallclock_unix_ns =
                options.wallclock_unix_ns == 0
                    ? unix_time_ns()
                    : options.wallclock_unix_ns,
            .min_reader_version =
                impl_->source_reader
                    ? impl_->source_reader
                          ->commit()
                          .min_reader_version
                    : CURRENT_CONTAINER_VERSION,
            .min_safe_writer_version =
                impl_->source_reader
                    ? impl_->source_reader
                          ->commit()
                          .min_safe_writer_version
                    : CURRENT_CONTAINER_VERSION,
            .extra_reader_capabilities =
                impl_->source_reader
                    ? impl_->source_reader
                          ->commit()
                          .required_reader_capabilities
                    : CapabilitySet{},
            .extra_writer_capabilities =
                impl_->source_reader
                    ? impl_->source_reader
                          ->commit()
                          .required_writer_capabilities
                    : CapabilitySet{},
        };
        save_options.file_uuid = options.file_uuid;
        save_options.index_compression =
            options.index_compression;
        save_options.disk_reserve_bytes =
            options.disk_reserve_bytes;
        return save_impl(
            sidecar_path, save_options, &options);
    }

    lfs::Result<ProjectDocumentSaveReport>
    ProjectDocument::save_impl(
        const std::filesystem::path& path,
        const ProjectDocumentSaveOptions& options,
        const ProjectDocumentAutosaveOptions* autosave) {
        const bool is_autosave = autosave != nullptr;
        if (!options.preview_png.empty() &&
            (options.commit.kind != CommitKind::Explicit ||
             is_autosave)) {
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::FailedPrecondition,
                "Only an explicit project save may regenerate the preview.",
                "Autosave and recovered generations must carry THMB forward",
                "save.preview_png");
        }
        auto normalized = normalized_absolute_path(path);
        if (!normalized) {
            return std::move(normalized).error();
        }
        if (impl_->source_reader) {
            const auto compatibility =
                impl_->source_reader->write_compatibility();
            if (!compatibility.safe) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::Unsupported,
                    "This project is read-only in the current LichtFeld version.",
                    std::format(
                        "project save refused before writing bytes: {}",
                        compatibility.reasons.empty()
                            ? std::string{"unknown writer incompatibility"}
                            : compatibility.reasons.front()),
                    "commit.write_compatibility");
            }
        }
        if (impl_->missing_opaque_preservation_capability) {
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::Unsupported,
                "This project is read-only because it contains opaque chapters without a preservation declaration.",
                "safe append requires required_writer_capabilities bit 5 (OPAQUE_CHUNK_PRESERVATION)",
                "commit.required_writer_capabilities");
        }
        if (impl_->missing_retained_json_capability) {
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::Unsupported,
                "This project is read-only because it contains retained JSON fields without a preservation declaration.",
                "safe append requires required_writer_capabilities bit 6 (RETAINED_JSON_FIELDS)",
                "commit.required_writer_capabilities");
        }
        if (!is_autosave && impl_->source_path &&
            *impl_->source_path != *normalized) {
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::FailedPrecondition,
                "Saving an opened project to another path is not part of the "
                "core chapter layer.",
                "Save As and transactional project switching are P6; append "
                "to the opened source path",
                "project.path");
        }
        if (is_autosave) {
            if (!impl_->source_path ||
                !impl_->source_reader ||
                autosave->base_explicit_commit_uuid.is_nil() ||
                autosave->autosave_sequence == 0) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::FailedPrecondition,
                    "The autosave has no durable explicit base.",
                    "sidecar publication requires an opened master, "
                    "base commit UUID, and positive sequence",
                    "autosave.binding");
            }
            auto expected_sidecar = *impl_->source_path;
            expected_sidecar += ".autosave";
            auto expected_normalized =
                normalized_absolute_path(
                    expected_sidecar);
            if (!expected_normalized ||
                *expected_normalized != *normalized) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::InvalidArgument,
                    "Autosave uses one bounded project sidecar.",
                    std::format(
                        "expected '{}'",
                        expected_sidecar.string()),
                    "autosave.path");
            }
            if (impl_->source_reader->commit().commit_uuid !=
                autosave->base_explicit_commit_uuid) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::FailedPrecondition,
                    "The autosave base is no longer current.",
                    "the document source head differs from the requested "
                    "base explicit commit UUID",
                    "autosave.base_commit_uuid");
            }
        } else if (!impl_->source_path &&
                   !options.allow_existing_destination_replacement) {
            std::error_code error;
            if (std::filesystem::exists(*normalized, error)) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::AlreadyExists,
                    "The destination project already exists.",
                    "P3 first-save assembly refuses implicit replacement",
                    "project.path");
            }
            if (error) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::PermissionDenied,
                    "The destination project could not be inspected.",
                    std::format("filesystem::exists failed: {}", error.message()),
                    "project.path");
            }
        }

        CommitOptions commit = options.commit;
        const bool retains_unknown_json =
            has_unknown_json_root(FOURCC_PROJ, impl_->project.dom()) ||
            has_unknown_json_root(FOURCC_REFS, impl_->references.dom()) ||
            has_unknown_json_root(FOURCC_SCNG, impl_->scene_graph.dom()) ||
            has_unknown_json_root(FOURCC_PRMS, impl_->parameters.dom()) ||
            has_unknown_json_root(FOURCC_GUIL, impl_->gui_layout.dom()) ||
            has_unknown_json_root(FOURCC_VIEW, impl_->view.dom()) ||
            has_unknown_json_root(FOURCC_EDTR, impl_->editor.dom()) ||
            has_unknown_json_root(FOURCC_SEQR, impl_->sequencer.dom());
        if (retains_unknown_json) {
            commit.extra_writer_capabilities.set(
                RETAINED_JSON_FIELDS);
        }
        if (commit.wallclock_unix_ns == 0) {
            commit.wallclock_unix_ns = unix_time_ns();
        }
        if (impl_->checkpoints.size() == 1) {
            const auto& snapshot_uuid =
                impl_->checkpoints.begin()->first;
            if (commit.snapshot_uuid.is_nil()) {
                commit.snapshot_uuid = snapshot_uuid;
            } else if (commit.snapshot_uuid != snapshot_uuid) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::FailedPrecondition,
                    "The commit and checkpoint snapshot identities differ.",
                    std::format(
                        "commit snapshot {} must equal CKPT instance {}",
                        commit.snapshot_uuid.to_string(),
                        snapshot_uuid.to_string()),
                    "commit.snapshot_uuid");
            }
        }
        if (is_autosave && commit.snapshot_uuid.is_nil()) {
            commit.snapshot_uuid =
                lfs::core::generate_uuid_v4();
        }
        auto staged_project =
            ProjectChapter::from_bytes(impl_->project.to_bytes());
        if (!staged_project) {
            return std::move(staged_project).error();
        }
        if (auto modified = staged_project->set_modified_at_unix_ns(
                commit.wallclock_unix_ns);
            !modified) {
            return std::move(modified).error();
        }

        std::map<ChunkKey, EncodedChunk, ChunkKeyLess> encoded;
        std::map<ChunkKey, Hash128, ChunkKeyLess> hashes =
            impl_->content_hashes;

        const auto add_encoded =
            [&](const ChunkKey& key, std::vector<std::byte> bytes,
                ChunkWriteOptions write_options) {
                encoded.emplace(
                    key, EncodedChunk{
                             .key = key,
                             .bytes = std::move(bytes),
                             .options = write_options,
                         });
            };

        const ChunkKey project_key = impl_->key(FOURCC_PROJ);
        const ChunkKey references_key = impl_->key(FOURCC_REFS);
        const ChunkKey scene_key = impl_->key(FOURCC_SCNG);
        const ChunkKey selection_key = impl_->key(FOURCC_SELM);
        const ChunkKey parameters_key = impl_->key(FOURCC_PRMS);
        const ChunkKey gui_layout_key = impl_->key(FOURCC_GUIL);
        const ChunkKey view_key = impl_->key(FOURCC_VIEW);
        const ChunkKey editor_key = impl_->key(FOURCC_EDTR);
        const ChunkKey sequencer_key = impl_->key(FOURCC_SEQR);
        const ChunkKey metrics_key = impl_->key(FOURCC_METR);

        if (impl_->dirty_or_new(references_key)) {
            add_encoded(references_key, impl_->references.to_bytes(),
                        json_options());
        }
        if (impl_->dirty_or_new(scene_key)) {
            add_encoded(scene_key, impl_->scene_graph.to_bytes(),
                        json_options());
        }
        if (impl_->dirty_or_new(parameters_key)) {
            add_encoded(parameters_key, impl_->parameters.to_bytes(),
                        json_options());
        }
        if (impl_->dirty_or_new(selection_key)) {
            auto bytes = encode_selection_chapter(impl_->selection);
            if (!bytes) {
                return std::move(bytes).error();
            }
            add_encoded(selection_key, std::move(*bytes),
                        selection_options());
        }
        if (impl_->dirty_or_new(gui_layout_key)) {
            add_encoded(gui_layout_key, impl_->gui_layout.to_bytes(),
                        json_options());
        }
        if (impl_->dirty_or_new(view_key)) {
            add_encoded(view_key, impl_->view.to_bytes(),
                        json_options());
        }
        if (impl_->dirty_or_new(editor_key)) {
            add_encoded(editor_key, impl_->editor.to_bytes(),
                        json_options());
        }
        if (impl_->dirty_or_new(sequencer_key)) {
            add_encoded(sequencer_key, impl_->sequencer.to_bytes(),
                        json_options());
        }
        if (impl_->dirty_or_new(metrics_key)) {
            auto bytes = impl_->metrics.to_bytes();
            if (!bytes) {
                return std::move(bytes).error();
            }
            add_encoded(metrics_key, std::move(*bytes),
                        json_options());
        }

        for (const auto& [uuid, payload] : impl_->splats) {
            const ChunkKey key{
                .fourcc = FOURCC_SPLT,
                .instance_uuid = uuid,
            };
            if (!impl_->dirty_or_new(key)) {
                continue;
            }
            std::vector<std::byte> bytes(payload.bytes().begin(),
                                         payload.bytes().end());
            hashes.insert_or_assign(key, xxh3_128(bytes));
            add_encoded(key, std::move(bytes),
                        ChunkWriteOptions{.chunk_version = P3_CHUNK_VERSION,
                                          .compression = Compression::ZstdFramed,
                                          .tensor_payload = true,
                                          .block_crcs = payload.bytes().size() >= BLOCK_CRC_REQUIRED_AT,
                                          .expected_stream_bytes = std::nullopt});
        }
        for (const auto& [uuid, payload] : impl_->point_clouds) {
            const ChunkKey key{
                .fourcc = FOURCC_PCLD,
                .instance_uuid = uuid,
            };
            if (!impl_->dirty_or_new(key)) {
                continue;
            }
            auto bytes = encode_point_cloud_payload(payload);
            if (!bytes) {
                return std::move(bytes).error();
            }
            hashes.insert_or_assign(key, xxh3_128(*bytes));
            const auto size = bytes->size();
            add_encoded(key, std::move(*bytes), tensor_options(size));
        }
        for (const auto& [uuid, payload] : impl_->meshes) {
            const ChunkKey key{
                .fourcc = FOURCC_MESH,
                .instance_uuid = uuid,
            };
            if (!impl_->dirty_or_new(key)) {
                continue;
            }
            auto bytes = encode_mesh_payload(payload);
            if (!bytes) {
                return std::move(bytes).error();
            }
            hashes.insert_or_assign(key, xxh3_128(*bytes));
            const auto size = bytes->size();
            add_encoded(key, std::move(*bytes), tensor_options(size));
        }

        auto embedded = staged_project->embedded_payload_provenance();
        if (!embedded) {
            return std::move(embedded).error();
        }
        std::unordered_map<std::string, EmbeddedPayloadProvenance>
            provenance_by_payload;
        for (const auto& record : *embedded) {
            provenance_by_payload.emplace(
                payload_identity(record.node_uuid, record.fourcc), record);
        }
        for (const auto& [key, hash] : hashes) {
            if (key.fourcc != FOURCC_SPLT && key.fourcc != FOURCC_PCLD &&
                key.fourcc != FOURCC_MESH) {
                continue;
            }
            const std::string fourcc = key.fourcc.to_string();
            const auto found = provenance_by_payload.find(
                payload_identity(key.instance_uuid, fourcc));
            if (found == provenance_by_payload.end()) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::FailedPrecondition,
                    "An embedded payload is missing its import provenance.",
                    std::format(
                        "{} instance {} requires locator, import fingerprint, "
                        "and content hash before save",
                        fourcc, key.instance_uuid.to_string()),
                    "PROJ.embedded_payloads");
            }
            auto updated = found->second;
            updated.content_xxh3_128 = hash;
            if (auto result =
                    staged_project->upsert_embedded_payload_provenance(updated);
                !result) {
                return std::move(result).error();
            }
        }
        if (auto valid = impl_->validate(*staged_project, hashes); !valid) {
            return std::move(valid).error();
        }
        add_encoded(project_key, staged_project->to_bytes(),
                    json_options());

        std::set<ChunkKey, ChunkKeyLess> desired{
            project_key,
            references_key,
            scene_key,
            selection_key,
            parameters_key,
            gui_layout_key,
            view_key,
            editor_key,
            sequencer_key,
            metrics_key,
        };
        for (const auto& [uuid, ignored] : impl_->splats) {
            (void)ignored;
            desired.insert(
                ChunkKey{.fourcc = FOURCC_SPLT, .instance_uuid = uuid});
        }
        for (const auto& [uuid, ignored] : impl_->point_clouds) {
            (void)ignored;
            desired.insert(
                ChunkKey{.fourcc = FOURCC_PCLD, .instance_uuid = uuid});
        }
        for (const auto& [uuid, ignored] : impl_->meshes) {
            (void)ignored;
            desired.insert(
                ChunkKey{.fourcc = FOURCC_MESH, .instance_uuid = uuid});
        }
        desired.insert(
            impl_->deferred_geometry_keys.begin(),
            impl_->deferred_geometry_keys.end());
        for (const auto& [uuid, ignored] : impl_->checkpoints) {
            (void)ignored;
            desired.insert(
                ChunkKey{.fourcc = FOURCC_CKPT, .instance_uuid = uuid});
        }
        for (const auto& [uuid, ignored] : impl_->ppisp_payloads) {
            (void)ignored;
            desired.insert(
                ChunkKey{.fourcc = FOURCC_PPIS, .instance_uuid = uuid});
        }

        auto planned_bytes = preflight_bytes(encoded);
        if (!planned_bytes) {
            return std::move(planned_bytes).error();
        }
        const auto add_lazy_preflight =
            [&](const auto& payloads) -> lfs::Result<void> {
            for (const auto& [uuid, payload] : payloads) {
                (void)uuid;
                if (payload.is_clean_reference()) {
                    continue;
                }
                auto added = checked_add(
                    *planned_bytes, payload.size(),
                    "save.lazy_binary_bytes");
                if (!added) {
                    return lfs::Result<void>::failure(
                        std::move(added).error());
                }
                *planned_bytes = *added;
            }
            return {};
        };
        if (auto result = add_lazy_preflight(impl_->checkpoints);
            !result) {
            return std::move(result).error();
        }
        if (auto result = add_lazy_preflight(impl_->ppisp_payloads);
            !result) {
            return std::move(result).error();
        }
        if (!options.preview_png.empty()) {
            auto added = checked_add(
                *planned_bytes, options.preview_png.size(),
                "save.preview_bytes");
            if (!added) {
                return std::move(added).error();
            }
            *planned_bytes = *added;
        }

        std::optional<ProjectWriter> writer;
        if (is_autosave) {
            auto created =
                staged_project->created_at_unix_ns();
            if (!created) {
                return std::move(created).error();
            }
            auto result = ProjectWriter::create(
                *normalized,
                CreateOptions{
                    .project_uuid =
                        impl_->project_uuid,
                    .file_uuid =
                        autosave->file_uuid,
                    .role =
                        ContainerRole::
                            AutosaveSidecar,
                    .base_explicit_commit_uuid =
                        autosave
                            ->base_explicit_commit_uuid,
                    .autosave_sequence =
                        autosave->autosave_sequence,
                    .sidecar_snapshot_uuid =
                        commit.snapshot_uuid,
                    .creation_time_unix_ns =
                        *created,
                    .index_compression =
                        autosave->index_compression,
                    .disk_reserve_bytes =
                        autosave
                            ->disk_reserve_bytes,
                    .boundary_observer =
                        autosave
                            ->boundary_observer,
                    .writer_lock_anchor_compatibility =
                        impl_->source_reader
                            ->reader_options(),
                    .writer_lock_anchor =
                        *impl_->source_path,
                    .writer_lock_lease =
                        autosave->writer_lock_lease,
                });
            if (!result) {
                return std::move(result).error();
            }
            writer.emplace(std::move(*result));
        } else if (impl_->source_path) {
            auto result = ProjectWriter::append(
                *normalized,
                AppendOptions{
                    .compatibility =
                        impl_->source_reader->reader_options(),
                    .index_compression = options.index_compression,
                    .disk_reserve_bytes = options.disk_reserve_bytes,
                    .boundary_observer = {},
                    .writer_lock_lease =
                        options.writer_lock_lease,
                    .writer_lock_wait =
                        options.writer_lock_wait,
                });
            if (!result) {
                return std::move(result).error();
            }
            writer.emplace(std::move(*result));
        } else {
            auto created = staged_project->created_at_unix_ns();
            if (!created) {
                return std::move(created).error();
            }
            auto result = ProjectWriter::create(
                *normalized,
                CreateOptions{
                    .project_uuid = impl_->project_uuid,
                    .file_uuid = options.file_uuid,
                    .role = ContainerRole::Master,
                    .base_explicit_commit_uuid = {},
                    .autosave_sequence = 0,
                    .sidecar_snapshot_uuid = {},
                    .creation_time_unix_ns = *created,
                    .index_compression = options.index_compression,
                    .disk_reserve_bytes = options.disk_reserve_bytes,
                    .boundary_observer = {},
                    .writer_lock_anchor =
                        std::nullopt,
                    .writer_lock_lease =
                        options.writer_lock_lease,
                });
            if (!result) {
                return std::move(result).error();
            }
            writer.emplace(std::move(*result));
        }
        if (auto result = writer->plan_commit(commit); !result) {
            return std::move(result).error();
        }
        if (auto result = writer->preflight(*planned_bytes); !result) {
            return std::move(result).error();
        }

        ProjectDocumentSaveReport report;
        if (!options.preview_png.empty()) {
            if (auto result = writer->set_preview(options.preview_png);
                !result) {
                return std::move(result).error();
            }
            ++report.rewritten_chunks;
        }
        for (const auto& [key, source] : impl_->source_rows) {
            if (!options.preview_png.empty() &&
                key.fourcc == FOURCC_THMB) {
                continue;
            }
            if (desired.contains(key)) {
                if (encoded.contains(key)) {
                    continue;
                }
                auto result =
                    is_autosave
                        ? writer->add_sidecar_base_reference(
                              source.info)
                        : (source.opaque
                               ? source.carry_opaque(*writer)
                               : source.reuse(*writer));
                if (!result) {
                    return std::move(result).error();
                }
                if (source.opaque && !is_autosave) {
                    ++report.opaque_chunks_carried;
                } else {
                    ++report.reused_chunks;
                }
            } else if (
                is_autosave &&
                is_project_managed_fourcc(
                    key.fourcc)) {
                if (auto seeded =
                        writer
                            ->add_sidecar_base_reference(
                                source.info);
                    !seeded) {
                    return std::move(seeded).error();
                }
                if (auto result = writer->erase(key);
                    !result) {
                    return std::move(result).error();
                }
                ++report.erased_chunks;
            } else if (is_project_managed_fourcc(key.fourcc)) {
                if (auto result = writer->erase(key); !result) {
                    return std::move(result).error();
                }
                ++report.erased_chunks;
            } else if (is_autosave) {
                if (auto result =
                        writer
                            ->add_sidecar_base_reference(
                                source.info);
                    !result) {
                    return std::move(result).error();
                }
                ++report.opaque_chunks_carried;
            } else {
                if (auto result = source.carry_opaque(*writer);
                    !result) {
                    return std::move(result).error();
                }
                ++report.opaque_chunks_carried;
            }
        }
        for (const auto& key : impl_->lazy_source_keys) {
            if (desired.contains(key)) {
                continue;
            }
            if (is_autosave) {
                const auto* const base =
                    impl_->source_reader
                        ? impl_->source_reader->find(
                              key)
                        : nullptr;
                if (!base) {
                    return fail<
                        ProjectDocumentSaveReport>(
                        lfs::ErrorCode::DataLoss,
                        "The autosave base payload is missing.",
                        std::format(
                            "{} instance {} has no live base row",
                            key.fourcc.to_string(),
                            key.instance_uuid.to_string()),
                        "autosave.completeness");
                }
                if (auto seeded =
                        writer
                            ->add_sidecar_base_reference(
                                *base);
                    !seeded) {
                    return std::move(seeded).error();
                }
            }
            if (auto result = writer->erase(key); !result) {
                return std::move(result).error();
            }
            ++report.erased_chunks;
        }
        const auto write_lazy =
            [&](const Fourcc fourcc,
                const auto& payloads) -> lfs::Result<void> {
            for (const auto& [uuid, payload] : payloads) {
                if (payload.is_clean_reference()) {
                    assert(payload.impl_->proof);
                    auto result =
                        is_autosave
                            ? writer
                                  ->add_sidecar_base_reference(
                                      *payload.impl_
                                           ->source)
                            : writer->reuse_if_clean(
                                  *payload.impl_->proof,
                                  payload.impl_->proof
                                      ->mutation_epoch());
                    if (!result) {
                        return result;
                    }
                    ++report.reused_chunks;
                    continue;
                }
                const ChunkKey key{
                    .fourcc = fourcc,
                    .instance_uuid = uuid,
                };
                // Owned staged bytes go through write_chunk (container zstd;
                // begin_chunk is Stored-streaming only). File-backed sources
                // without a clean proof are copied stored-byte-verbatim so a
                // compressed CKPT/PPIS is never inflated on the save path.
                // The copy keeps the source compression, uncompressed_bytes,
                // and chunk_version; those are correct for an unchanged
                // payload. Counted as rewritten_chunks because stored bytes
                // are physically rewritten.
                if (payload.impl_->owned) {
                    const auto options =
                        lazy_binary_options(fourcc, payload.size());
                    if (auto written = writer->write_chunk(
                            key, *payload.impl_->owned, options);
                        !written) {
                        return written;
                    }
                } else if (payload.impl_->reader && payload.impl_->source) {
                    const ChunkInfo& source = *payload.impl_->source;
                    if (source.key != key) {
                        return fail<void>(
                            lfs::ErrorCode::Internal,
                            "The lazy chapter source key does not match the "
                            "chapter being saved.",
                            std::format(
                                "expected {} instance {}, source is {} "
                                "instance {}",
                                key.fourcc.to_string(),
                                key.instance_uuid.to_string(),
                                source.key.fourcc.to_string(),
                                source.key.instance_uuid.to_string()),
                            "lazy_chunk.source_key");
                    }
                    if (auto copied = writer->copy_chunk_verbatim(
                            *payload.impl_->reader, source);
                        !copied) {
                        return copied;
                    }
                } else {
                    return fail<void>(
                        lfs::ErrorCode::FailedPrecondition,
                        "The lazy chapter has no byte source.",
                        "Neither clean file range nor owned storage is "
                        "available",
                        "lazy_chunk.source");
                }
                ++report.rewritten_chunks;
            }
            return {};
        };
        if (auto result =
                write_lazy(FOURCC_CKPT, impl_->checkpoints);
            !result) {
            return std::move(result).error();
        }
        if (auto result =
                write_lazy(FOURCC_PPIS, impl_->ppisp_payloads);
            !result) {
            return std::move(result).error();
        }
        for (const auto& [key, chunk] : encoded) {
            if (auto result =
                    writer->write_chunk(key, chunk.bytes, chunk.options);
                !result) {
                return std::move(result).error();
            }
            ++report.rewritten_chunks;
        }
        if (auto result = writer->commit(); !result) {
            return std::move(result).error();
        }
        writer.reset();

        if (is_autosave) {
            auto reader =
                ProjectReader::open(
                    *normalized,
                    impl_->source_reader
                        ->reader_options());
            if (!reader) {
                return std::move(reader).error();
            }
            if (auto verified = reader->verify_all();
                !verified) {
                return std::move(verified).error();
            }
            report.generation =
                reader->commit().generation;
            report.commit_uuid =
                reader->commit().commit_uuid;
            report.snapshot_uuid =
                reader->commit().snapshot_uuid;
            return report;
        }

        impl_->project = std::move(*staged_project);
        impl_->content_hashes = std::move(hashes);
        if (auto refreshed = impl_->refresh_source_rows(*normalized);
            !refreshed) {
            return std::move(refreshed).error();
        }
        report.generation = impl_->source_reader->commit().generation;
        report.commit_uuid = impl_->source_reader->commit().commit_uuid;
        report.snapshot_uuid = impl_->source_reader->commit().snapshot_uuid;
        return report;
    }

    lfs::Result<ProjectDocumentSaveReport>
    ProjectDocument::save_as(
        const std::filesystem::path& path,
        const ProjectDocumentSaveOptions& options) {
        if (impl_->source_reader) {
            const auto compatibility =
                impl_->source_reader->write_compatibility();
            if (!compatibility.safe) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::Unsupported,
                    "This project is read-only in the current LichtFeld version.",
                    std::format(
                        "Save As refused before staging destination bytes: {}",
                        compatibility.reasons.empty()
                            ? std::string{"unknown writer incompatibility"}
                            : compatibility.reasons.front()),
                    "commit.write_compatibility");
            }
        }
        if (impl_->missing_opaque_preservation_capability ||
            impl_->missing_retained_json_capability) {
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::Unsupported,
                "This project cannot be saved under another name by the current writer.",
                impl_->missing_opaque_preservation_capability
                    ? "opaque rows lack required writer capability bit 5"
                    : "retained JSON fields lack required writer capability bit 6",
                "commit.required_writer_capabilities");
        }
        auto normalized = normalized_absolute_path(path);
        if (!normalized) {
            return std::move(normalized).error();
        }
        if (!impl_->source_path) {
            return save(*normalized, options);
        }
        if (*impl_->source_path == *normalized) {
            return save(*normalized, options);
        }

        std::optional<detail::WriterLock>
            destination_lock;
        if (options.writer_lock_lease) {
            if (!options.writer_lock_lease->owns(
                    *normalized)) {
                return fail<ProjectDocumentSaveReport>(
                    lfs::ErrorCode::FailedPrecondition,
                    "The retained recovery lock does not own the Save As destination.",
                    normalized->string(),
                    "writer_lock");
            }
        } else {
            auto acquired =
                detail::WriterLock::acquire(
                    *normalized);
            if (!acquired) {
                return std::move(acquired).error();
            }
            destination_lock.emplace(
                std::move(*acquired));
        }

        std::error_code error;
        static_cast<void>(
            std::filesystem::exists(
                *normalized, error));
        if (error) {
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::PermissionDenied,
                "The Save As destination could not be inspected.",
                std::format("filesystem::exists failed: {}", error.message()),
                "project.path");
        }

        const auto original_path = *impl_->source_path;
        const auto original_dirty = impl_->dirty;
        const auto original_normalized_source_keys =
            impl_->normalized_source_keys;
        const auto temporary =
            normalized->parent_path() /
            std::format(".{}.saveas-{}.tmp",
                        normalized->filename().string(),
                        lfs::core::generate_uuid_v4().to_string());

        const auto remove_temporary = [&temporary] {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
        };
        if (!std::filesystem::copy_file(
                original_path, temporary,
                std::filesystem::copy_options::none, error)) {
            remove_temporary();
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::Unavailable,
                "The project could not be staged for Save As.",
                std::format("copy_file failed: {}", error.message()),
                "project.save_as.copy");
        }

        const auto file_uuid =
            options.file_uuid.is_nil()
                ? lfs::core::generate_uuid_v4()
                : options.file_uuid;
        auto compacted = ProjectWriter::compact(
            temporary,
            CompactionOptions{
                .compatibility =
                    impl_->source_reader
                        ? impl_->source_reader
                              ->reader_options()
                        : ReaderOptions{},
                .new_file_uuid = file_uuid,
                .commit_uuid =
                    options.save_as_compaction_commit_uuid.is_nil()
                        ? lfs::core::generate_uuid_v4()
                        : options.save_as_compaction_commit_uuid,
                .snapshot_uuid = options.commit.snapshot_uuid,
                .creation_time_unix_ns =
                    options.save_as_creation_time_unix_ns,
                .wallclock_unix_ns = options.commit.wallclock_unix_ns,
                .disk_reserve_bytes = options.disk_reserve_bytes,
                .boundary_observer = {},
            });
        if (!compacted) {
            remove_temporary();
            return std::move(compacted).error();
        }

        const auto rebind_preserving_dirty_lazy =
            [this, &original_dirty,
             &original_normalized_source_keys](
                const std::filesystem::path& source)
            -> lfs::Result<void> {
            std::unordered_map<lfs::core::Uuid, LazyChunkValue>
                dirty_checkpoints;
            std::unordered_map<lfs::core::Uuid, LazyChunkValue>
                dirty_ppisp;
            const auto extract_dirty =
                [&original_dirty](auto& from, auto& to,
                                  const Fourcc fourcc) {
                    for (auto iterator = from.begin();
                         iterator != from.end();) {
                        const ChunkKey key{
                            .fourcc = fourcc,
                            .instance_uuid = iterator->first,
                        };
                        if (!original_dirty.contains(key)) {
                            ++iterator;
                            continue;
                        }
                        auto node = from.extract(iterator++);
                        to.insert(std::move(node));
                    }
                };
            extract_dirty(
                impl_->checkpoints, dirty_checkpoints,
                FOURCC_CKPT);
            extract_dirty(
                impl_->ppisp_payloads, dirty_ppisp,
                FOURCC_PPIS);
            auto refreshed = impl_->refresh_source_rows(source);
            if (!refreshed) {
                for (auto& [uuid, payload] : dirty_checkpoints) {
                    impl_->checkpoints.insert_or_assign(
                        uuid, std::move(payload));
                }
                for (auto& [uuid, payload] : dirty_ppisp) {
                    impl_->ppisp_payloads.insert_or_assign(
                        uuid, std::move(payload));
                }
                impl_->dirty = original_dirty;
                impl_->normalized_source_keys =
                    original_normalized_source_keys;
                return refreshed;
            }
            for (auto& [uuid, payload] : dirty_checkpoints) {
                impl_->checkpoints.insert_or_assign(
                    uuid, std::move(payload));
            }
            for (auto& [uuid, payload] : dirty_ppisp) {
                impl_->ppisp_payloads.insert_or_assign(
                    uuid, std::move(payload));
            }
            const auto erase_recorded_removals =
                [&original_dirty](auto& payloads,
                                  const auto& preserved,
                                  const Fourcc fourcc) {
                    for (const auto& key : original_dirty) {
                        if (key.fourcc == fourcc &&
                            !preserved.contains(
                                key.instance_uuid)) {
                            payloads.erase(key.instance_uuid);
                        }
                    }
                };
            erase_recorded_removals(
                impl_->checkpoints, dirty_checkpoints,
                FOURCC_CKPT);
            erase_recorded_removals(
                impl_->ppisp_payloads, dirty_ppisp,
                FOURCC_PPIS);
            impl_->dirty = original_dirty;
            impl_->normalized_source_keys =
                original_normalized_source_keys;
            return {};
        };

        if (auto rebound =
                rebind_preserving_dirty_lazy(temporary);
            !rebound) {
            remove_temporary();
            return std::move(rebound).error();
        }

        auto staging_options = options;
        staging_options.writer_lock_lease.reset();
        auto saved = save(temporary, staging_options);
        if (!saved) {
            auto save_error = std::move(saved).error();
            auto restored =
                rebind_preserving_dirty_lazy(original_path);
            remove_temporary();
            if (!restored) {
                return std::move(save_error).with_suppressed(std::move(restored).error());
            }
            return save_error;
        }

        auto staged_reader =
            ProjectReader::open(temporary);
        if (!staged_reader) {
            auto cause =
                std::move(staged_reader).error();
            auto restored =
                rebind_preserving_dirty_lazy(
                    original_path);
            remove_temporary();
            if (!restored) {
                return std::move(cause)
                    .with_suppressed(
                        std::move(restored).error());
            }
            return cause;
        }
        if (auto verified =
                staged_reader->verify_all();
            !verified) {
            auto cause =
                std::move(verified).error();
            auto restored =
                rebind_preserving_dirty_lazy(
                    original_path);
            remove_temporary();
            if (!restored) {
                return std::move(cause)
                    .with_suppressed(
                        std::move(restored).error());
            }
            return cause;
        }
        const auto expected_commit_uuid =
            staged_reader->commit().commit_uuid;

        auto replacement =
            detail::atomic_replace(
                temporary, *normalized);
        if (!replacement) {
            auto cause =
                std::move(replacement).error();
            auto restored =
                rebind_preserving_dirty_lazy(
                    original_path);
            if (!restored) {
                return std::move(cause)
                    .with_suppressed(
                        std::move(restored).error());
            }
            return cause;
        }
        auto published =
            ProjectReader::open(*normalized);
        if (!published ||
            published->commit().commit_uuid !=
                expected_commit_uuid) {
            auto cause =
                published
                    ? document_error(
                          lfs::ErrorCode::DataLoss,
                          "The Save As destination failed post-publication validation.",
                          "The published commit UUID does not match the independently validated staged project.",
                          "project.save_as.publish")
                    : std::move(published).error();
            auto rollback =
                detail::rollback_atomic_replace(
                    *replacement, *normalized);
            auto restored =
                rebind_preserving_dirty_lazy(
                    original_path);
            if (!rollback) {
                cause = std::move(cause)
                            .with_suppressed(
                                std::move(rollback)
                                    .error());
            }
            if (!restored) {
                cause = std::move(cause)
                            .with_suppressed(
                                std::move(restored)
                                    .error());
            }
            return cause;
        }
        if (auto verified = published->verify_all();
            !verified) {
            auto cause =
                std::move(verified).error();
            auto rollback =
                detail::rollback_atomic_replace(
                    *replacement, *normalized);
            auto restored =
                rebind_preserving_dirty_lazy(
                    original_path);
            if (!rollback) {
                cause = std::move(cause)
                            .with_suppressed(
                                std::move(rollback)
                                    .error());
            }
            if (!restored) {
                cause = std::move(cause)
                            .with_suppressed(
                                std::move(restored)
                                    .error());
            }
            return cause;
        }
        if (auto refreshed =
                impl_->refresh_source_rows(*normalized);
            !refreshed) {
            return std::move(refreshed).error();
        }
        if (auto finished =
                detail::finish_atomic_replace(
                    *replacement, *normalized);
            !finished) {
            return std::move(finished).error();
        }
        saved->generation = impl_->generation;
        return saved;
    }

    lfs::Result<std::unique_ptr<lfs::core::Scene>>
    ProjectDocument::stage_shell(
        lfs::core::Scene& destination) const {
        auto shell =
            stage_scene_shell(
                impl_->scene_graph, destination);
        if (!shell) {
            return std::move(shell).error();
        }
        (*shell)->installRestoreSelectionState(
            {
                .splat_mask = nullptr,
                .point_cloud_mask = nullptr,
                .groups = impl_->selection.groups(),
                .active_group_id =
                    impl_->selection.active_group_id(),
                .next_group_id =
                    impl_->selection.next_group_id(),
                .has_splat_selection = false,
                .has_point_cloud_selection = false,
            });
        return shell;
    }

    lfs::Result<ProjectHydrationPlan>
    ProjectDocument::stage_hydration(
        lfs::core::Scene& destination,
        const ScenePayloadResolver& external_payloads,
        lfs::core::SplatTensorAllocator splat_allocator,
        std::function<void(std::size_t, std::size_t)> payload_progress) const {
        const auto hydration_started =
            std::chrono::steady_clock::now();
        double splat_read_ms = 0.0;
        double splat_hash_ms = 0.0;
        double splat_copy_ms = 0.0;
        double splat_materialize_ms = 0.0;
        const auto milliseconds =
            [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(
                           end - begin)
                    .count();
            };
        try {
            std::map<ChunkKey, Hash128, ChunkKeyLess>
                hashes;
            std::unordered_map<
                lfs::core::Uuid,
                std::unique_ptr<lfs::core::SplatData>>
                staged_splats;
            std::unordered_map<
                lfs::core::Uuid,
                std::unique_ptr<lfs::core::SplatData>>
                staged_checkpoint_splats;
            std::unordered_map<
                lfs::core::Uuid,
                std::shared_ptr<lfs::core::PointCloud>>
                staged_point_clouds;
            std::unordered_map<
                lfs::core::Uuid,
                std::shared_ptr<lfs::core::MeshData>>
                staged_meshes;

            const auto deferred_count =
                [this](const Fourcc fourcc) {
                    return static_cast<std::size_t>(
                        std::ranges::count_if(
                            impl_->deferred_geometry_keys,
                            [fourcc](const ChunkKey& key) {
                                return key.fourcc == fourcc;
                            }));
                };
            const auto read_deferred =
                [this, &payload_progress](const ChunkKey& key)
                -> lfs::Result<std::vector<std::byte>> {
                if (!impl_->source_reader) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::FailedPrecondition,
                        "The unloaded project payload has no source file.",
                        std::format("{} instance {} lost its clean source handle",
                                    key.fourcc.to_string(),
                                    key.instance_uuid.to_string()),
                        "hydrate.deferred_source");
                }
                const auto found = impl_->source_rows.find(key);
                if (found == impl_->source_rows.end()) {
                    return fail<std::vector<std::byte>>(
                        lfs::ErrorCode::DataLoss,
                        "The unloaded project payload is missing.",
                        std::format("{} instance {} has no live source row",
                                    key.fourcc.to_string(),
                                    key.instance_uuid.to_string()),
                        "hydrate.deferred_source");
                }
                return impl_->source_reader->read_chunk(found->second.info,
                                                        payload_progress);
            };

            staged_splats.reserve(
                impl_->splats.size() + deferred_count(FOURCC_SPLT));
            const std::size_t splat_count =
                impl_->splats.size() + deferred_count(FOURCC_SPLT);
            for (const auto& [uuid, payload] :
                 impl_->splats) {
                const ChunkKey key{
                    .fourcc = FOURCC_SPLT,
                    .instance_uuid = uuid,
                };
                const auto hash_started =
                    std::chrono::steady_clock::now();
                hashes.emplace(key, xxh3_128(payload.bytes()));
                splat_hash_ms += milliseconds(
                    hash_started, std::chrono::steady_clock::now());
                const auto materialize_started =
                    std::chrono::steady_clock::now();
                auto materialized =
                    payload.hydrate(splat_allocator);
                splat_materialize_ms += milliseconds(
                    materialize_started,
                    std::chrono::steady_clock::now());
                if (!materialized) {
                    return std::move(materialized).error();
                }
                staged_splats.emplace(
                    uuid, std::move(*materialized));
            }
            for (const auto& key : impl_->deferred_geometry_keys) {
                if (key.fourcc != FOURCC_SPLT) {
                    continue;
                }
                if (!impl_->source_reader) {
                    return fail<ProjectHydrationPlan>(
                        lfs::ErrorCode::FailedPrecondition,
                        "The unloaded project payload has no source file.",
                        std::format("{} instance {} lost its clean source handle",
                                    key.fourcc.to_string(),
                                    key.instance_uuid.to_string()),
                        "hydrate.deferred_source");
                }
                const auto found = impl_->source_rows.find(key);
                if (found == impl_->source_rows.end()) {
                    return fail<ProjectHydrationPlan>(
                        lfs::ErrorCode::DataLoss,
                        "The unloaded project payload is missing.",
                        std::format("{} instance {} has no live source row",
                                    key.fourcc.to_string(),
                                    key.instance_uuid.to_string()),
                        "hydrate.deferred_source");
                }
                const auto read_started =
                    std::chrono::steady_clock::now();
                auto bounded = impl_->source_reader->open_bounded_stream(
                    found->second.info);
                splat_read_ms += milliseconds(
                    read_started, std::chrono::steady_clock::now());
                if (!bounded) {
                    return std::move(bounded).error();
                }
                const auto materialize_started =
                    std::chrono::steady_clock::now();
                auto hydrated = SplatChapterPayload::hydrate_lfsp_stream(
                    bounded->stream(), bounded->size(), splat_allocator,
                    payload_progress);
                splat_materialize_ms += milliseconds(
                    materialize_started,
                    std::chrono::steady_clock::now());
                if (!hydrated) {
                    return std::move(hydrated).error();
                }
                hashes.insert_or_assign(key, hydrated->content_xxh3_128);
                staged_splats.emplace(
                    key.instance_uuid, std::move(hydrated->splat));
            }

            std::optional<lfs::core::Uuid>
                checkpoint_uuid;
            std::optional<lfs::core::CheckpointHeader>
                checkpoint_header;
            staged_checkpoint_splats.reserve(
                impl_->checkpoints.size());
            for (const auto& [uuid, payload] :
                 impl_->checkpoints) {
                std::optional<lfs::core::SplatData>
                    materialized;
                auto decoded = payload.visit_stream(
                    [&](std::istream& stream,
                        const std::uint64_t bytes)
                        -> lfs::Result<void> {
                        auto header =
                            lfs::core::load_checkpoint_header(
                                stream, bytes);
                        if (!header) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "The embedded checkpoint header is invalid.",
                                header.error(),
                                "CKPT.LFKP.header");
                        }
                        checkpoint_header = *header;
                        stream.clear();
                        stream.seekg(0);
                        if (!stream) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "The embedded checkpoint cannot be rewound.",
                                "The bounded CKPT stream must support seek to byte zero",
                                "CKPT.LFKP.stream");
                        }
                        auto model =
                            lfs::core::load_checkpoint_splat_data(
                                stream, bytes,
                                splat_allocator);
                        if (!model) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "The checkpoint display model is invalid.",
                                model.error(),
                                "CKPT.LFKP.model");
                        }
                        materialized.emplace(
                            std::move(*model));
                        return {};
                    });
                if (!decoded) {
                    return std::move(decoded).error();
                }
                assert(materialized);
                checkpoint_uuid = uuid;
                staged_checkpoint_splats.emplace(
                    uuid,
                    std::make_unique<lfs::core::SplatData>(
                        std::move(*materialized)));
            }

            staged_point_clouds.reserve(
                impl_->point_clouds.size() +
                deferred_count(FOURCC_PCLD));
            for (const auto& [uuid, payload] :
                 impl_->point_clouds) {
                auto bytes =
                    encode_point_cloud_payload(payload);
                if (!bytes) {
                    return std::move(bytes).error();
                }
                const ChunkKey key{
                    .fourcc = FOURCC_PCLD,
                    .instance_uuid = uuid,
                };
                hashes.emplace(key, xxh3_128(*bytes));
                auto materialized =
                    decode_point_cloud_payload(*bytes);
                if (!materialized) {
                    return std::move(materialized).error();
                }
                staged_point_clouds.emplace(
                    uuid, materialized->point_cloud());
            }
            for (const auto& key : impl_->deferred_geometry_keys) {
                if (key.fourcc != FOURCC_PCLD) {
                    continue;
                }
                auto bytes = read_deferred(key);
                if (!bytes) {
                    return std::move(bytes).error();
                }
                hashes.insert_or_assign(key, xxh3_128(*bytes));
                auto payload =
                    decode_point_cloud_payload(*bytes);
                if (!payload) {
                    return std::move(payload).error();
                }
                staged_point_clouds.emplace(
                    key.instance_uuid, payload->point_cloud());
            }

            staged_meshes.reserve(
                impl_->meshes.size() +
                deferred_count(FOURCC_MESH));
            for (const auto& [uuid, payload] :
                 impl_->meshes) {
                auto bytes = encode_mesh_payload(payload);
                if (!bytes) {
                    return std::move(bytes).error();
                }
                const ChunkKey key{
                    .fourcc = FOURCC_MESH,
                    .instance_uuid = uuid,
                };
                hashes.emplace(key, xxh3_128(*bytes));
                auto materialized =
                    decode_mesh_payload(*bytes);
                if (!materialized) {
                    return std::move(materialized).error();
                }
                staged_meshes.emplace(
                    uuid, materialized->mesh());
            }
            for (const auto& key : impl_->deferred_geometry_keys) {
                if (key.fourcc != FOURCC_MESH) {
                    continue;
                }
                auto bytes = read_deferred(key);
                if (!bytes) {
                    return std::move(bytes).error();
                }
                hashes.insert_or_assign(key, xxh3_128(*bytes));
                auto payload = decode_mesh_payload(*bytes);
                if (!payload) {
                    return std::move(payload).error();
                }
                staged_meshes.emplace(
                    key.instance_uuid, payload->mesh());
            }

            if (auto valid =
                    impl_->validate(impl_->project, hashes);
                !valid) {
                return std::move(valid).error();
            }

            auto pending = impl_->parameters.snapshot();
            if (!pending) {
                return std::move(pending).error();
            }
            auto reverse = reverse_reference_index();
            if (!reverse) {
                return std::move(reverse).error();
            }
            ProjectSessionChapters pending_session{
                .gui_layout = impl_->gui_layout,
                .editor = impl_->editor,
                .view = impl_->view,
                .sequencer = impl_->sequencer,
                .metrics = impl_->metrics,
            };

            ScenePayloadResolver resolver;
            resolver.splat =
                [&staged_splats,
                 &staged_checkpoint_splats,
                 external_payloads](
                    const PayloadBinding& binding)
                -> lfs::Result<std::unique_ptr<
                    lfs::core::SplatData>> {
                if (binding.fourcc == "SPLT") {
                    const auto found = staged_splats.find(
                        binding.instance_uuid);
                    if (found == staged_splats.end() ||
                        !found->second) {
                        return fail<std::unique_ptr<
                            lfs::core::SplatData>>(
                            lfs::ErrorCode::NotFound,
                            "An embedded splat payload is missing.",
                            std::format(
                                "SPLT instance {} is unavailable or multiply bound",
                                binding.instance_uuid
                                    .to_string()),
                            "SPLT.instance_uuid");
                    }
                    return std::move(found->second);
                }
                if (binding.fourcc == "CKPT") {
                    const auto found =
                        staged_checkpoint_splats.find(
                            binding.instance_uuid);
                    if (found ==
                            staged_checkpoint_splats.end() ||
                        !found->second) {
                        return fail<std::unique_ptr<
                            lfs::core::SplatData>>(
                            lfs::ErrorCode::NotFound,
                            "The checkpoint display model is missing.",
                            std::format(
                                "CKPT instance {} is unavailable or multiply bound",
                                binding.instance_uuid
                                    .to_string()),
                            "CKPT.instance_uuid");
                    }
                    return std::move(found->second);
                }
                if (!external_payloads.splat) {
                    return fail<std::unique_ptr<
                        lfs::core::SplatData>>(
                        lfs::ErrorCode::FailedPrecondition,
                        "An external splat payload resolver is missing.",
                        std::format(
                            "{} instance {} cannot be hydrated",
                            binding.fourcc,
                            binding.instance_uuid.to_string()),
                        "SCNG.nodes.payload");
                }
                return external_payloads.splat(binding);
            };
            resolver.point_cloud =
                [&staged_point_clouds, external_payloads](
                    const PayloadBinding& binding)
                -> lfs::Result<std::shared_ptr<
                    lfs::core::PointCloud>> {
                if (binding.fourcc == "PCLD") {
                    const auto found =
                        staged_point_clouds.find(
                            binding.instance_uuid);
                    if (found ==
                            staged_point_clouds.end() ||
                        !found->second) {
                        return fail<std::shared_ptr<
                            lfs::core::PointCloud>>(
                            lfs::ErrorCode::NotFound,
                            "An embedded point-cloud payload is missing.",
                            std::format(
                                "PCLD instance {} is unavailable",
                                binding.instance_uuid
                                    .to_string()),
                            "PCLD.instance_uuid");
                    }
                    return found->second;
                }
                if (!external_payloads.point_cloud) {
                    return fail<std::shared_ptr<
                        lfs::core::PointCloud>>(
                        lfs::ErrorCode::FailedPrecondition,
                        "An external point-cloud payload resolver is missing.",
                        std::format(
                            "{} instance {} cannot be hydrated",
                            binding.fourcc,
                            binding.instance_uuid.to_string()),
                        "SCNG.nodes.payload");
                }
                return external_payloads.point_cloud(binding);
            };
            resolver.mesh =
                [&staged_meshes, external_payloads](
                    const PayloadBinding& binding)
                -> lfs::Result<std::shared_ptr<
                    lfs::core::MeshData>> {
                if (binding.fourcc == "MESH") {
                    const auto found = staged_meshes.find(
                        binding.instance_uuid);
                    if (found == staged_meshes.end() ||
                        !found->second) {
                        return fail<std::shared_ptr<
                            lfs::core::MeshData>>(
                            lfs::ErrorCode::NotFound,
                            "An embedded mesh payload is missing.",
                            std::format(
                                "MESH instance {} is unavailable",
                                binding.instance_uuid
                                    .to_string()),
                            "MESH.instance_uuid");
                    }
                    return found->second;
                }
                if (!external_payloads.mesh) {
                    return fail<std::shared_ptr<
                        lfs::core::MeshData>>(
                        lfs::ErrorCode::FailedPrecondition,
                        "An external mesh payload resolver is missing.",
                        std::format(
                            "{} instance {} cannot be hydrated",
                            binding.fourcc,
                            binding.instance_uuid.to_string()),
                        "SCNG.nodes.payload");
                }
                return external_payloads.mesh(binding);
            };

            auto staged_scene = stage_scene_graph(
                impl_->scene_graph, destination, resolver);
            if (!staged_scene) {
                return std::move(staged_scene).error();
            }
            auto staged_selection =
                stage_selection_chapter(
                    impl_->selection, **staged_scene);
            if (!staged_selection) {
                return std::move(staged_selection).error();
            }
            auto selection_report =
                std::move(staged_selection->report);
            (*staged_scene)
                ->installRestoreSelectionState(
                    std::move(staged_selection->state));

            auto plan =
                std::make_unique<ProjectHydrationPlan::Impl>();
            plan->destination = &destination;
            plan->staged_scene =
                std::move(*staged_scene);
            plan->report =
                ProjectDocumentHydrationReport{
                    .selection =
                        std::move(selection_report),
                    .pending_parameters =
                        std::move(*pending),
                    .reverse_reference_index =
                        std::move(*reverse),
                    .checkpoint_uuid =
                        checkpoint_uuid,
                    .checkpoint_header =
                        checkpoint_header,
                    .trainer_state_pending =
                        checkpoint_uuid.has_value(),
                    .pending_session =
                        std::move(pending_session),
                };
            const double total_ms = milliseconds(
                hydration_started,
                std::chrono::steady_clock::now());
            LOG_DEBUG(
                "Project SPLT hydration stages: splats={} chunk_read={:.3f} ms content_hash={:.3f} ms payload_copy={:.3f} ms materialize={:.3f} ms remaining={:.3f} ms total={:.3f} ms",
                splat_count, splat_read_ms, splat_hash_ms,
                splat_copy_ms, splat_materialize_ms,
                std::max(0.0,
                         total_ms - splat_read_ms - splat_hash_ms -
                             splat_copy_ms - splat_materialize_ms),
                total_ms);
            return ProjectHydrationPlan(std::move(plan));
        } catch (const std::bad_alloc& error) {
            // LFS-CENSUS-OK(empty-catch): Phase A converts allocation failure into the Result contract.
            return fail<ProjectHydrationPlan>(
                lfs::ErrorCode::ResourceExhausted,
                "The project could not be staged in memory.",
                error.what(), "hydrate.phase_a");
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): legacy tensor and payload APIs throw; Phase A normalizes them.
            return fail<ProjectHydrationPlan>(
                lfs::ErrorCode::DataLoss,
                "The project could not be staged.",
                error.what(), "hydrate.phase_a");
        }
    }

    ProjectDocumentHydrationReport
    ProjectDocument::commit_hydration(
        lfs::core::Scene& destination,
        ProjectHydrationPlan&& staged) noexcept {
        assert(staged.impl_);
        assert(staged.impl_->destination == &destination);
        assert(staged.impl_->staged_scene);
        auto report = std::move(staged.impl_->report);
        destination.commitRestoreStage(
            std::move(staged.impl_->staged_scene));
        report.selection_installed = true;
        staged.impl_.reset();
        return report;
    }

    ProjectDocumentHydrationReport
    ProjectDocument::commit_partial_hydration(
        lfs::core::Scene& destination,
        ProjectHydrationPlan&& staged,
        const bool install_selection) noexcept {
        assert(staged.impl_);
        assert(staged.impl_->destination == &destination);
        assert(staged.impl_->staged_scene);
        auto report = std::move(staged.impl_->report);
        const auto committed =
            destination.commitPayloadHydrationStage(
                std::move(staged.impl_->staged_scene),
                install_selection);
        report.hydrated_payload_units =
            committed.hydrated_units;
        report.invalidated_payload_units =
            committed.invalidated_units;
        report.selection_installed =
            committed.selection_installed;
        staged.impl_.reset();
        return report;
    }

    lfs::Result<ProjectDocumentHydrationReport>
    ProjectDocument::hydrate(
        lfs::core::Scene& scene,
        const ScenePayloadResolver& external_payloads,
        lfs::core::SplatTensorAllocator splat_allocator) const {
        auto staged = stage_hydration(
            scene, external_payloads,
            std::move(splat_allocator));
        if (!staged) {
            return std::move(staged).error();
        }
        return commit_hydration(
            scene, std::move(*staged));
    }

    lfs::Result<ReverseReferenceIndex>
    ProjectDocument::reverse_reference_index(
        const std::span<const ReferenceOwnerBinding> additional_bindings) const {
        auto session_bindings =
            session_reference_bindings(
                impl_->view, impl_->sequencer);
        if (!session_bindings) {
            return std::move(session_bindings).error();
        }
        session_bindings->insert(
            session_bindings->end(),
            additional_bindings.begin(),
            additional_bindings.end());
        return build_reverse_reference_index(
            impl_->references, impl_->project, impl_->scene_graph,
            impl_->parameters,
            *session_bindings);
    }

} // namespace lfs::io::project
