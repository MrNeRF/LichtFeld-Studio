/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_document.hpp"

#include "io/loader.hpp"

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
#include <streambuf>
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

        bool is_p3_fourcc(const Fourcc fourcc) noexcept {
            return fourcc == FOURCC_PROJ || fourcc == FOURCC_PRMS ||
                   fourcc == FOURCC_SCNG || fourcc == FOURCC_SELM ||
                   fourcc == FOURCC_REFS || fourcc == FOURCC_SPLT ||
                   fourcc == FOURCC_PCLD || fourcc == FOURCC_MESH;
        }

        bool is_lazy_binary_fourcc(const Fourcc fourcc) noexcept {
            return fourcc == FOURCC_CKPT || fourcc == FOURCC_PPIS;
        }

        bool is_singleton_fourcc(const Fourcc fourcc) noexcept {
            return fourcc == FOURCC_PROJ || fourcc == FOURCC_PRMS ||
                   fourcc == FOURCC_SCNG || fourcc == FOURCC_SELM ||
                   fourcc == FOURCC_REFS;
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
                if (chunk.options.compression == Compression::Zstd) {
                    estimate = ZSTD_compressBound(chunk.bytes.size());
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
                .compression = Compression::Zstd,
                .tensor_payload = false,
                .block_crcs = false,
                .expected_stream_bytes = std::nullopt,
            };
        }

        ChunkWriteOptions selection_options() {
            return ChunkWriteOptions{
                .chunk_version = SELM_CHAPTER_VERSION,
                .compression = Compression::Zstd,
                .tensor_payload = false,
                .block_crcs = false,
                .expected_stream_bytes = std::nullopt,
            };
        }

        ChunkWriteOptions tensor_options(const std::size_t size) {
            return ChunkWriteOptions{
                .chunk_version = P3_CHUNK_VERSION,
                .compression = Compression::Stored,
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
                .compression = Compression::Stored,
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

    namespace {

        class ReadOnlyMemoryBuffer final : public std::streambuf {
        public:
            explicit ReadOnlyMemoryBuffer(
                const std::span<const std::byte> bytes) {
                auto* begin = const_cast<char*>(
                    reinterpret_cast<const char*>(bytes.data()));
                setg(begin, begin, begin + bytes.size());
            }

        protected:
            pos_type seekoff(const off_type offset,
                             const std::ios_base::seekdir direction,
                             const std::ios_base::openmode mode) override {
                if ((mode & std::ios_base::in) == 0) {
                    return pos_type(off_type(-1));
                }
                char* base = eback();
                char* target = nullptr;
                if (direction == std::ios_base::beg) {
                    target = base + offset;
                } else if (direction == std::ios_base::cur) {
                    target = gptr() + offset;
                } else if (direction == std::ios_base::end) {
                    target = egptr() + offset;
                }
                if (!target || target < base || target > egptr()) {
                    return pos_type(off_type(-1));
                }
                setg(base, target, egptr());
                return pos_type(target - base);
            }

            pos_type seekpos(const pos_type position,
                             const std::ios_base::openmode mode) override {
                return seekoff(
                    static_cast<off_type>(position),
                    std::ios_base::beg, mode);
            }
        };

    } // namespace

    struct LazyChunkValue::Impl {
        std::shared_ptr<ProjectReader> reader;
        std::optional<ChunkInfo> source;
        std::optional<CleanProof> proof;
        std::shared_ptr<const std::vector<std::byte>> owned;
        lfs::core::Uuid snapshot_uuid;

        [[nodiscard]] std::uint64_t size() const noexcept {
            if (owned) {
                return owned->size();
            }
            return source ? source->uncompressed_bytes : 0;
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

    bool LazyChunkValue::owns_staged_bytes() const noexcept {
        return static_cast<bool>(impl_->owned);
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
        return impl_->reader->read_stored_at(
            *impl_->source, offset, destination);
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
            ReadOnlyMemoryBuffer buffer(std::span<const std::byte>(
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
        auto bounded =
            impl_->reader->open_bounded_stream(*impl_->source);
        if (!bounded) {
            return lfs::Result<void>::failure(
                std::move(bounded).error());
        }
        return visitor(bounded->stream(), bounded->size());
    }

    lfs::Result<void>
    LazyChunkValue::copy_to(
        std::ostream& destination,
        const std::size_t window_bytes) const {
        if (window_bytes == 0) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "The lazy chapter copy window cannot be zero.",
                "copy_to requires a bounded non-zero window",
                "lazy_chunk.window_bytes");
        }
        const std::size_t bounded_window = std::min<std::size_t>(
            window_bytes,
            static_cast<std::size_t>(
                std::numeric_limits<std::streamsize>::max()));
        std::vector<std::byte> window(
            static_cast<std::size_t>(
                std::min<std::uint64_t>(size(), bounded_window)));
        std::uint64_t offset = 0;
        while (offset < size()) {
            const auto count = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    window.size(), size() - offset));
            auto destination_window =
                std::span<std::byte>(window).first(count);
            if (auto read = read_at(offset, destination_window);
                !read) {
                return read;
            }
            destination.write(
                reinterpret_cast<const char*>(window.data()),
                static_cast<std::streamsize>(count));
            if (!destination) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The lazy chapter could not be streamed.",
                    std::format(
                        "destination failed after {} of {} bytes",
                        offset, size()),
                    "lazy_chunk.destination");
            }
            offset += count;
        }
        return {};
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

        std::unordered_map<lfs::core::Uuid, SplatChapterPayload> splats;
        std::unordered_map<lfs::core::Uuid, PointCloudPayload> point_clouds;
        std::unordered_map<lfs::core::Uuid, MeshPayload> meshes;
        std::unordered_map<lfs::core::Uuid, LazyChunkValue> checkpoints;
        std::unordered_map<lfs::core::Uuid, LazyChunkValue> ppisp_payloads;

        std::optional<std::filesystem::path> source_path;
        std::map<ChunkKey, SourceRow, ChunkKeyLess> source_rows;
        std::set<ChunkKey, ChunkKeyLess> lazy_source_keys;
        std::map<ChunkKey, Hash128, ChunkKeyLess> content_hashes;
        std::set<ChunkKey, ChunkKeyLess> dirty;

        [[nodiscard]] ChunkKey key(const Fourcc fourcc) const {
            return singleton_key(fourcc, project_uuid);
        }

        void mark(const Fourcc fourcc) {
            dirty.insert(key(fourcc));
        }

        void mark(const Fourcc fourcc, const lfs::core::Uuid& instance_uuid) {
            dirty.insert(ChunkKey{
                .fourcc = fourcc,
                .instance_uuid = instance_uuid,
            });
        }

        [[nodiscard]] bool dirty_or_new(const ChunkKey& chunk_key) const {
            return dirty.contains(chunk_key) ||
                   !source_rows.contains(chunk_key);
        }

        [[nodiscard]] lfs::Result<void>
        validate(const ProjectChapter& candidate_project,
                 const std::map<ChunkKey, Hash128, ChunkKeyLess>& hashes) const {
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
            auto reverse = build_reverse_reference_index(
                references, candidate_project, scene_graph, parameters);
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
                    (fourcc == FOURCC_MESH && meshes.contains(node.uuid));
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
        auto normalized = normalized_absolute_path(path);
        if (!normalized) {
            return std::move(normalized).error();
        }
        auto reader = ProjectReader::open(*normalized, options.reader);
        if (!reader) {
            return std::move(reader).error();
        }
        auto shared_reader = std::make_shared<ProjectReader>(
            std::move(*reader));
        auto impl = std::make_unique<Impl>();
        impl->project_uuid = shared_reader->superblock().project_uuid;
        impl->source_path = *normalized;

        bool have_project = false;
        bool have_references = false;
        bool have_scene = false;
        bool have_selection = false;
        bool have_parameters = false;

        for (const auto& row : shared_reader->chunks()) {
            if (!row.is_live()) {
                continue;
            }
            auto proof = shared_reader->make_clean_proof(
                row, DOCUMENT_CLEAN_BASELINE);
            if (!proof) {
                return std::move(proof).error();
            }
            if (is_lazy_binary_fourcc(row.key.fourcc)) {
                if (row.chunk_version != P3_CHUNK_VERSION ||
                    row.compression != Compression::Stored) {
                    return fail<ProjectDocument>(
                        lfs::ErrorCode::Unsupported,
                        "This project uses an unsupported binary chapter encoding.",
                        std::format(
                            "{} chunk version {} compression {} must be "
                            "version 1 stored bytes",
                            row.key.fourcc.to_string(),
                            row.chunk_version,
                            static_cast<unsigned>(row.compression)),
                        "chunk.binary_encoding");
                }
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
                });
            if (!is_p3_fourcc(row.key.fourcc)) {
                continue;
            }
            if (row.chunk_version != P3_CHUNK_VERSION) {
                return fail<ProjectDocument>(
                    lfs::ErrorCode::Unsupported,
                    "This project uses a newer core chapter version.",
                    std::format("{} chunk version {} is unsupported",
                                row.key.fourcc.to_string(),
                                row.chunk_version),
                    "chunk.chunk_version");
            }
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
            auto bytes = shared_reader->read_chunk(row);
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
                have_parameters = true;
            } else if (row.key.fourcc == FOURCC_SPLT) {
                auto payload = SplatChapterPayload::from_lfsp(*bytes);
                if (!payload) {
                    return std::move(payload).error();
                }
                impl->splats.emplace(row.key.instance_uuid,
                                     std::move(*payload));
                impl->content_hashes.emplace(row.key, xxh3_128(*bytes));
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
        if (auto valid = impl->validate(impl->project, impl->content_hashes);
            !valid) {
            return std::move(valid).error();
        }
        return ProjectDocument(std::move(impl));
    }

    const std::optional<std::filesystem::path>&
    ProjectDocument::source_path() const noexcept {
        return impl_->source_path;
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

    const LazyChunkValue*
    ProjectDocument::find_checkpoint(
        const lfs::core::Uuid& instance_uuid) const noexcept {
        const auto found = impl_->checkpoints.find(instance_uuid);
        return found == impl_->checkpoints.end()
                   ? nullptr
                   : &found->second;
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
        impl_->dirty.erase(ChunkKey{
            .fourcc = FOURCC_CKPT,
            .instance_uuid = instance_uuid,
        });
        return impl_->checkpoints.erase(instance_uuid) != 0;
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

    bool ProjectDocument::remove_ppisp(
        const lfs::core::Uuid& instance_uuid) {
        impl_->dirty.erase(ChunkKey{
            .fourcc = FOURCC_PPIS,
            .instance_uuid = instance_uuid,
        });
        return impl_->ppisp_payloads.erase(instance_uuid) != 0;
    }

    std::vector<lfs::core::Uuid>
    ProjectDocument::ppisp_uuids() const {
        return sorted_uuids(impl_->ppisp_payloads);
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
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_SPLT, .instance_uuid = node_uuid});
        return {};
    }

    bool ProjectDocument::remove_splat(const lfs::core::Uuid& node_uuid) {
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_SPLT, .instance_uuid = node_uuid});
        impl_->dirty.erase(
            ChunkKey{.fourcc = FOURCC_SPLT, .instance_uuid = node_uuid});
        return impl_->splats.erase(node_uuid) != 0;
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
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_PCLD, .instance_uuid = node_uuid});
        return {};
    }

    bool ProjectDocument::remove_point_cloud(
        const lfs::core::Uuid& node_uuid) {
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_PCLD, .instance_uuid = node_uuid});
        impl_->dirty.erase(
            ChunkKey{.fourcc = FOURCC_PCLD, .instance_uuid = node_uuid});
        return impl_->point_clouds.erase(node_uuid) != 0;
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
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_MESH, .instance_uuid = node_uuid});
        return {};
    }

    bool ProjectDocument::remove_mesh(const lfs::core::Uuid& node_uuid) {
        impl_->content_hashes.erase(
            ChunkKey{.fourcc = FOURCC_MESH, .instance_uuid = node_uuid});
        impl_->dirty.erase(
            ChunkKey{.fourcc = FOURCC_MESH, .instance_uuid = node_uuid});
        return impl_->meshes.erase(node_uuid) != 0;
    }

    std::vector<lfs::core::Uuid> ProjectDocument::splat_uuids() const {
        return sorted_uuids(impl_->splats);
    }

    std::vector<lfs::core::Uuid>
    ProjectDocument::point_cloud_uuids() const {
        return sorted_uuids(impl_->point_clouds);
    }

    std::vector<lfs::core::Uuid> ProjectDocument::mesh_uuids() const {
        return sorted_uuids(impl_->meshes);
    }

    lfs::Result<ProjectDocumentSaveReport>
    ProjectDocument::save(const std::filesystem::path& path,
                          const ProjectDocumentSaveOptions& options) {
        auto normalized = normalized_absolute_path(path);
        if (!normalized) {
            return std::move(normalized).error();
        }
        if (impl_->source_path && *impl_->source_path != *normalized) {
            return fail<ProjectDocumentSaveReport>(
                lfs::ErrorCode::FailedPrecondition,
                "Saving an opened project to another path is not part of the "
                "core chapter layer.",
                "Save As and transactional project switching are P6; append "
                "to the opened source path",
                "project.path");
        }
        if (!impl_->source_path) {
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
                        tensor_options(payload.bytes().size()));
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

        std::optional<ProjectWriter> writer;
        if (impl_->source_path) {
            auto result = ProjectWriter::append(
                *normalized,
                AppendOptions{
                    .compatibility = {},
                    .index_compression = options.index_compression,
                    .disk_reserve_bytes = options.disk_reserve_bytes,
                    .boundary_observer = {},
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
        for (const auto& [key, source] : impl_->source_rows) {
            if (desired.contains(key)) {
                if (encoded.contains(key)) {
                    continue;
                }
                if (auto result = source.reuse(*writer);
                    !result) {
                    return std::move(result).error();
                }
                ++report.reused_chunks;
            } else if (is_p3_fourcc(key.fourcc)) {
                if (auto result = writer->erase(key); !result) {
                    return std::move(result).error();
                }
                ++report.erased_chunks;
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
                    if (auto result = writer->reuse_if_clean(
                            *payload.impl_->proof,
                            payload.impl_->proof->mutation_epoch());
                        !result) {
                        return result;
                    }
                    ++report.reused_chunks;
                    continue;
                }
                const ChunkKey key{
                    .fourcc = fourcc,
                    .instance_uuid = uuid,
                };
                auto stream = writer->begin_chunk(
                    key, lazy_binary_options(fourcc, payload.size()));
                if (!stream) {
                    return lfs::Result<void>::failure(
                        std::move(stream).error());
                }
                if (auto copied = payload.copy_to(**stream);
                    !copied) {
                    return copied;
                }
                if (auto ended = writer->end_chunk(); !ended) {
                    return ended;
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

        impl_->project = std::move(*staged_project);
        impl_->content_hashes = std::move(hashes);
        if (auto refreshed = impl_->refresh_source_rows(*normalized);
            !refreshed) {
            return std::move(refreshed).error();
        }
        auto reader = ProjectReader::open(*normalized);
        if (!reader) {
            return std::move(reader).error();
        }
        report.generation = reader->commit().generation;
        report.commit_uuid = reader->commit().commit_uuid;
        report.snapshot_uuid = reader->commit().snapshot_uuid;
        return report;
    }

    lfs::Result<ProjectHydrationPlan>
    ProjectDocument::stage_hydration(
        lfs::core::Scene& destination,
        const ScenePayloadResolver& external_payloads,
        lfs::core::SplatTensorAllocator splat_allocator) const {
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

            staged_splats.reserve(impl_->splats.size());
            for (const auto& [uuid, payload] :
                 impl_->splats) {
                const ChunkKey key{
                    .fourcc = FOURCC_SPLT,
                    .instance_uuid = uuid,
                };
                hashes.emplace(
                    key, xxh3_128(payload.bytes()));
                auto materialized =
                    payload.hydrate(splat_allocator);
                if (!materialized) {
                    return std::move(materialized).error();
                }
                staged_splats.emplace(
                    uuid, std::move(*materialized));
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
                impl_->point_clouds.size());
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

            staged_meshes.reserve(impl_->meshes.size());
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
                };
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
        return build_reverse_reference_index(
            impl_->references, impl_->project, impl_->scene_graph,
            impl_->parameters,
            additional_bindings);
    }

} // namespace lfs::io::project
