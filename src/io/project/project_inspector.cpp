/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_inspector.hpp"

#include "core/checkpoint_format.hpp"
#include "core/path_utils.hpp"
#include "crc32c.hpp"
#include "project_container_internal.hpp"
#include "span_streambuf.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>

namespace lfs::io::project {

    std::string project_content_stamp(const std::filesystem::path& path) {
        auto opened = ProjectReader::open(path);
        if (!opened || opened->open_state() != OpenState::Open) {
            return {};
        }

        using Evidence = std::array<std::byte, 56>;
        std::vector<Evidence> content;
        std::vector<Evidence> view;
        bool has_scene = false;
        for (const auto& row : opened->chunks()) {
            const auto kind = row.key.fourcc;
            const bool is_content = kind == FOURCC_SCNG || kind == FOURCC_REFS ||
                                    kind == FOURCC_DSRC || kind == FOURCC_SPLT ||
                                    kind == FOURCC_CKPT || kind == FOURCC_SELM;
            if (!is_content && kind != FOURCC_VIEW && kind != FOURCC_SEQR) {
                continue;
            }
            Evidence bytes{};
            const auto put = [&bytes](const std::uint64_t value,
                                      const std::size_t offset,
                                      const std::size_t width) {
                for (std::size_t index = 0; index < width; ++index) {
                    bytes[offset + index] =
                        std::byte((value >> (index * 8)) & 0xff);
                }
            };
            for (std::size_t index = 0; index < 4; ++index) {
                bytes[index] = std::byte(kind.bytes[index]);
            }
            put(row.chunk_version, 4, 2);
            put(static_cast<std::uint8_t>(row.row_kind), 6, 1);
            put(static_cast<std::uint8_t>(row.compression), 7, 1);
            put(row.flags, 8, 4);
            for (std::size_t index = 0; index < row.key.instance_uuid.bytes.size(); ++index) {
                bytes[16 + index] = std::byte(row.key.instance_uuid.bytes[index]);
            }
            put(row.stored_bytes, 32, 8);
            put(row.uncompressed_bytes, 40, 8);
            put(row.payload_crc32c, 48, 4);
            put(row.header_crc32c, 52, 4);
            (is_content ? content : view).push_back(bytes);
            has_scene |= kind == FOURCC_SCNG && row.is_live();
        }
        if (!has_scene) {
            return {};
        }

        const auto digest = [](std::vector<Evidence>& rows) -> std::string {
            std::ranges::sort(rows);
            std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
            unsigned int size = 0;
            const void* data = rows.empty()
                                   ? static_cast<const void*>("")
                                   : static_cast<const void*>(rows.data());
            if (EVP_Digest(data, rows.size() * sizeof(Evidence), hash.data(), &size,
                           EVP_sha256(), nullptr) != 1 ||
                size != 32) {
                return {};
            }
            constexpr char hex[] = "0123456789abcdef";
            std::string result;
            result.reserve(size * 2);
            for (unsigned int index = 0; index < size; ++index) {
                result += hex[hash[index] >> 4];
                result += hex[hash[index] & 15];
            }
            return result;
        };
        const auto content_digest = digest(content);
        const auto view_digest = digest(view);
        return content_digest.empty() || view_digest.empty()
                   ? std::string{}
                   : content_digest + ':' + view_digest;
    }

    namespace {

        constexpr std::array<std::byte, 8> SUPERBLOCK_MAGIC = {
            std::byte{0x89}, std::byte{'L'}, std::byte{'F'}, std::byte{'S'},
            std::byte{'\r'}, std::byte{'\n'}, std::byte{0x1a}, std::byte{'\n'}};
        constexpr std::array<std::byte, 8> HEAD_MAGIC = {
            std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'H'},
            std::byte{'E'}, std::byte{'A'}, std::byte{'D'}, std::byte{0}};
        constexpr std::array<std::byte, 8> COMMIT_MAGIC = {
            std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'C'},
            std::byte{'O'}, std::byte{'M'}, std::byte{'I'}, std::byte{'T'}};
        constexpr std::array<std::byte, 8> PNG_MAGIC = {
            std::byte{0x89}, std::byte{'P'}, std::byte{'N'}, std::byte{'G'},
            std::byte{'\r'}, std::byte{'\n'}, std::byte{0x1a}, std::byte{'\n'}};

        template <typename T>
        T inspect_read_u(const std::span<const std::byte> bytes,
                         const std::size_t offset) noexcept {
            T result = 0;
            for (std::size_t index = 0; index < sizeof(T); ++index) {
                result |= static_cast<T>(std::to_integer<std::uint8_t>(
                              bytes[offset + index]))
                          << (index * 8);
            }
            return result;
        }

        lfs::core::Uuid inspect_read_uuid(
            const std::span<const std::byte> bytes,
            const std::size_t offset) noexcept {
            lfs::core::Uuid result;
            for (std::size_t index = 0; index < result.bytes.size(); ++index) {
                result.bytes[index] = std::to_integer<std::uint8_t>(
                    bytes[offset + index]);
            }
            return result;
        }

        CapabilitySet inspect_read_capabilities(
            const std::span<const std::byte> bytes,
            const std::size_t offset) noexcept {
            std::array<std::uint8_t, 16> result{};
            for (std::size_t index = 0; index < result.size(); ++index) {
                result[index] = std::to_integer<std::uint8_t>(
                    bytes[offset + index]);
            }
            return CapabilitySet(result);
        }

        template <std::size_t Size>
        bool inspect_read_fixed(std::ifstream& stream,
                                const std::uint64_t offset,
                                std::array<std::byte, Size>& bytes) {
            stream.clear();
            stream.seekg(static_cast<std::streamoff>(offset));
            stream.read(reinterpret_cast<char*>(bytes.data()),
                        static_cast<std::streamsize>(bytes.size()));
            return stream.good() &&
                   stream.gcount() == static_cast<std::streamsize>(bytes.size());
        }

        std::pair<std::uint32_t, std::uint32_t>
        inspect_preview_dimensions(std::ifstream& stream,
                                   const std::optional<PreviewLocator>& preview) {
            if (!preview || preview->bytes < 24) {
                return {};
            }
            std::array<std::byte, 24> header{};
            if (!inspect_read_fixed(stream, preview->offset, header) ||
                !std::equal(PNG_MAGIC.begin(), PNG_MAGIC.end(), header.begin()) ||
                header[12] != std::byte{'I'} || header[13] != std::byte{'H'} ||
                header[14] != std::byte{'D'} || header[15] != std::byte{'R'}) {
                return {};
            }
            const auto read_be_u32 = [&header](const std::size_t offset) {
                std::uint32_t value = 0;
                for (std::size_t index = 0; index < 4; ++index) {
                    value = (value << 8) |
                            std::to_integer<std::uint8_t>(header[offset + index]);
                }
                return value;
            };
            if (read_be_u32(8) != 13) {
                return {};
            }
            const std::uint32_t width = read_be_u32(16);
            const std::uint32_t height = read_be_u32(20);
            constexpr auto max_dimension =
                static_cast<std::uint32_t>(std::numeric_limits<int>::max());
            return width > 0 && height > 0 && width <= max_dimension &&
                           height <= max_dimension
                       ? std::pair{width, height}
                       : std::pair<std::uint32_t, std::uint32_t>{};
        }

        std::pair<std::uint32_t, std::uint32_t>
        inspect_preview_dimensions(const std::filesystem::path& path,
                                   const std::optional<PreviewLocator>& preview) {
            std::ifstream stream(path, std::ios::binary);
            return stream ? inspect_preview_dimensions(stream, preview)
                          : std::pair<std::uint32_t, std::uint32_t>{};
        }

        struct HeaderCardCandidate {
            std::uint64_t sequence = 0;
            std::uint64_t generation = 0;
            lfs::core::Uuid commit_uuid;
            std::uint64_t committed_file_end = 0;
            std::uint64_t commit_offset = 0;
            std::uint64_t parent_commit_offset = 0;
            lfs::core::Uuid parent_commit_uuid;
            std::optional<PreviewLocator> preview;
            CommitInfo commit;
            bool unsupported = false;
        };

        std::optional<ProjectInspectorCard>
        inspect_project_card_headers(const std::filesystem::path& path) {
            std::error_code size_error;
            const std::uint64_t physical_size = std::filesystem::file_size(
                path, size_error);
            if (size_error || physical_size < SUPERBLOCK_BYTES) {
                return std::nullopt;
            }
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                return std::nullopt;
            }

            std::array<std::byte, SUPERBLOCK_BYTES> superblock_raw{};
            if (!inspect_read_fixed(stream, 0, superblock_raw)) {
                return std::nullopt;
            }
            const auto superblock = std::span<const std::byte>(superblock_raw);
            if (!std::equal(SUPERBLOCK_MAGIC.begin(), SUPERBLOCK_MAGIC.end(),
                            superblock.begin()) ||
                inspect_read_u<std::uint32_t>(superblock, 252) !=
                    crc32c(0, superblock_raw.data(), 252) ||
                inspect_read_u<std::uint16_t>(superblock, 8) != 1 ||
                inspect_read_u<std::uint32_t>(superblock, 12) != 0x01020304u) {
                return std::nullopt;
            }
            const auto project_uuid = inspect_read_uuid(superblock, 24);
            const auto file_uuid = inspect_read_uuid(superblock, 40);
            if (project_uuid.is_nil() || file_uuid.is_nil()) {
                return std::nullopt;
            }

            std::array<std::optional<HeaderCardCandidate>, 2> candidates;
            for (std::size_t slot = 0; slot < candidates.size(); ++slot) {
                std::array<std::byte, HEAD_SLOT_BYTES> raw{};
                if (physical_size < HEAD_SLOT_OFFSETS[slot] + HEAD_SLOT_BYTES ||
                    !inspect_read_fixed(stream, HEAD_SLOT_OFFSETS[slot], raw)) {
                    continue;
                }
                const auto bytes = std::span<const std::byte>(raw);
                if (std::all_of(bytes.begin(), bytes.end(), [](const std::byte byte) {
                        return byte == std::byte{0};
                    }) ||
                    !std::equal(HEAD_MAGIC.begin(), HEAD_MAGIC.end(), bytes.begin()) || inspect_read_u<std::uint32_t>(bytes, 4092) != crc32c(0, raw.data(), 4092) || inspect_read_u<std::uint32_t>(bytes, 8) != slot || inspect_read_u<std::uint32_t>(bytes, 12) != HEAD_SLOT_BYTES) {
                    continue;
                }
                const auto candidate_project = inspect_read_uuid(bytes, 32);
                const auto candidate_file = inspect_read_uuid(bytes, 48);
                const auto commit_uuid = inspect_read_uuid(bytes, 64);
                const std::uint64_t commit_offset =
                    inspect_read_u<std::uint64_t>(bytes, 80);
                const std::uint64_t commit_bytes =
                    inspect_read_u<std::uint64_t>(bytes, 88);
                const std::uint64_t committed_end =
                    inspect_read_u<std::uint64_t>(bytes, 96);
                const std::uint32_t commit_crc_echo =
                    inspect_read_u<std::uint32_t>(bytes, 104);
                if (candidate_project != project_uuid || candidate_file != file_uuid ||
                    commit_uuid.is_nil() || commit_bytes != COMMIT_RECORD_BYTES ||
                    commit_offset < APPEND_REGION_OFFSET ||
                    commit_offset % CHUNK_ALIGNMENT != 0 ||
                    committed_end != commit_offset + COMMIT_RECORD_BYTES ||
                    committed_end > physical_size ||
                    inspect_read_u<std::uint32_t>(bytes, 108) != 0) {
                    continue;
                }

                std::array<std::byte, COMMIT_RECORD_BYTES> commit_raw{};
                if (!inspect_read_fixed(stream, commit_offset, commit_raw)) {
                    continue;
                }
                const auto commit_bytes_view =
                    std::span<const std::byte>(commit_raw);
                if (!std::equal(COMMIT_MAGIC.begin(), COMMIT_MAGIC.end(),
                                commit_bytes_view.begin()) ||
                    inspect_read_u<std::uint32_t>(commit_bytes_view, 252) !=
                        crc32c(0, commit_raw.data(), 252) ||
                    inspect_read_uuid(commit_bytes_view, 48) != commit_uuid ||
                    inspect_read_u<std::uint64_t>(commit_bytes_view, 64) !=
                        inspect_read_u<std::uint64_t>(bytes, 24) ||
                    inspect_read_u<std::uint64_t>(commit_bytes_view, 176) !=
                        committed_end ||
                    inspect_read_u<std::uint32_t>(commit_bytes_view, 252) !=
                        commit_crc_echo) {
                    continue;
                }
                const std::uint32_t kind_value =
                    inspect_read_u<std::uint32_t>(commit_bytes_view, 12);
                if (kind_value > static_cast<std::uint32_t>(CommitKind::Compaction)) {
                    continue;
                }
                const auto required_reader =
                    inspect_read_capabilities(commit_bytes_view, 192);
                const auto min_reader = Version{
                    inspect_read_u<std::uint16_t>(commit_bytes_view, 184),
                    inspect_read_u<std::uint16_t>(commit_bytes_view, 186)};
                const bool unsupported =
                    min_reader > CURRENT_CONTAINER_VERSION ||
                    !supported_reader_capabilities().contains_all(required_reader);
                std::optional<PreviewLocator> preview;
                const std::uint64_t preview_offset =
                    inspect_read_u<std::uint64_t>(bytes, 112);
                const std::uint32_t preview_bytes =
                    inspect_read_u<std::uint32_t>(bytes, 120);
                if (preview_offset != 0 && preview_bytes != 0) {
                    if (preview_offset % CHUNK_ALIGNMENT != 0 ||
                        preview_offset + preview_bytes > committed_end ||
                        preview_bytes > MAX_PREVIEW_BYTES) {
                        continue;
                    }
                    preview = PreviewLocator{
                        .offset = preview_offset,
                        .bytes = preview_bytes,
                        .format = PreviewFormat::Png,
                    };
                }
                HeaderCardCandidate candidate;
                candidate.sequence = inspect_read_u<std::uint64_t>(bytes, 16);
                candidate.generation = inspect_read_u<std::uint64_t>(bytes, 24);
                candidate.commit_uuid = commit_uuid;
                candidate.committed_file_end = committed_end;
                candidate.commit_offset = commit_offset;
                candidate.parent_commit_offset =
                    inspect_read_u<std::uint64_t>(commit_bytes_view, 88);
                candidate.parent_commit_uuid = inspect_read_uuid(commit_bytes_view, 72);
                candidate.preview = preview;
                candidate.commit.offset = commit_offset;
                candidate.commit.kind = static_cast<CommitKind>(kind_value);
                candidate.commit.commit_uuid = commit_uuid;
                candidate.commit.generation =
                    inspect_read_u<std::uint64_t>(commit_bytes_view, 64);
                candidate.commit.wallclock_unix_ns =
                    inspect_read_u<std::uint64_t>(commit_bytes_view, 128);
                candidate.commit.committed_file_end = committed_end;
                candidate.commit.min_reader_version = min_reader;
                candidate.commit.min_safe_writer_version = Version{
                    inspect_read_u<std::uint16_t>(commit_bytes_view, 188),
                    inspect_read_u<std::uint16_t>(commit_bytes_view, 190)};
                candidates[slot] = std::move(candidate);
                candidates[slot]->commit.required_reader_capabilities = required_reader;
                candidates[slot]->commit.crc32c = inspect_read_u<std::uint32_t>(
                    commit_bytes_view, 252);
                candidates[slot]->unsupported = unsupported;
            }
            if (!candidates[0] && !candidates[1]) {
                return std::nullopt;
            }
            const auto* selected = candidates[0] && candidates[1]
                                       ? (candidates[0]->sequence >= candidates[1]->sequence
                                              ? &*candidates[0]
                                              : &*candidates[1])
                                   : candidates[0] ? &*candidates[0]
                                                   : &*candidates[1];
            std::uint64_t parent_offset = selected->parent_commit_offset;
            auto parent_uuid = selected->parent_commit_uuid;
            std::uint64_t child_generation = selected->generation;
            while (child_generation > 1) {
                if (parent_offset < APPEND_REGION_OFFSET ||
                    parent_offset % CHUNK_ALIGNMENT != 0 ||
                    parent_offset + COMMIT_RECORD_BYTES > physical_size) {
                    return std::nullopt;
                }
                std::array<std::byte, COMMIT_RECORD_BYTES> parent_raw{};
                if (!inspect_read_fixed(stream, parent_offset, parent_raw)) {
                    return std::nullopt;
                }
                const auto parent_bytes = std::span<const std::byte>(parent_raw);
                if (!std::equal(COMMIT_MAGIC.begin(), COMMIT_MAGIC.end(),
                                parent_bytes.begin()) ||
                    inspect_read_u<std::uint32_t>(parent_bytes, 252) !=
                        crc32c(0, parent_raw.data(), 252) ||
                    inspect_read_uuid(parent_bytes, 48) != parent_uuid ||
                    inspect_read_u<std::uint64_t>(parent_bytes, 64) + 1 !=
                        child_generation) {
                    return std::nullopt;
                }
                parent_uuid = inspect_read_uuid(parent_bytes, 72);
                parent_offset = inspect_read_u<std::uint64_t>(parent_bytes, 88);
                child_generation = inspect_read_u<std::uint64_t>(parent_bytes, 64);
            }
            ProjectInspectorCard result;
            result.path = path;
            result.project_uuid = project_uuid;
            result.file_uuid = file_uuid;
            result.commit_uuid = selected->commit_uuid;
            result.generation = selected->generation;
            result.created_at_unix_ns = inspect_read_u<std::uint64_t>(superblock, 88);
            result.saved_at_unix_ns = selected->commit.wallclock_unix_ns;
            result.physical_file_size = physical_size;
            result.role = static_cast<ContainerRole>(
                inspect_read_u<std::uint32_t>(superblock, 20));
            result.open_state = selected->unsupported ? OpenState::UnsupportedNewer
                                                      : OpenState::Open;
            result.validation_scope = "head";
            result.has_preview = selected->preview.has_value();
            result.preview_bytes = selected->preview ? selected->preview->bytes : 0;
            std::tie(result.preview_width, result.preview_height) =
                inspect_preview_dimensions(stream, selected->preview);
            result.min_reader_version = selected->commit.min_reader_version;
            result.min_safe_writer_version = selected->commit.min_safe_writer_version;
            result.commit_kind = selected->commit.kind;
            if (!candidates[0] || !candidates[1]) {
                result.diagnostic = "One head slot was rejected during head inspection.";
            }
            return result;
        }

        template <typename T>
        lfs::Result<T> inspector_error(const lfs::ErrorCode code,
                                       std::string message,
                                       std::string detail,
                                       const std::filesystem::path& path = {}) {
            return detail::project_error(code, std::move(message), std::move(detail), path);
        }

        ProjectInspectorCard card_from_reader(const ProjectReader& reader) {
            const auto preview = reader.preview();
            const auto [preview_width, preview_height] =
                project_preview_dimensions(reader);
            std::optional<std::string> title;
            if (const auto* row = reader.find(FOURCC_PROJ,
                                              reader.superblock().project_uuid);
                row != nullptr && row->row_kind == RowKind::Live) {
                if (auto bytes = reader.read_chunk(*row); bytes) {
                    if (auto project = ProjectChapter::from_bytes(*bytes); project) {
                        const auto value = project->dom().get_json("title");
                        if (value && value->is_string() && !value->get<std::string>().empty()) {
                            title = value->get<std::string>();
                        }
                    }
                }
            }
            ProjectInspectorCard result{
                .path = reader.path(),
                .project_uuid = reader.superblock().project_uuid,
                .file_uuid = reader.superblock().file_uuid,
                .commit_uuid = reader.commit().commit_uuid,
                .generation = reader.commit().generation,
                .created_at_unix_ns = reader.superblock().creation_time_unix_ns,
                .saved_at_unix_ns = reader.commit().wallclock_unix_ns,
                .physical_file_size = reader.physical_file_size(),
                .role = reader.superblock().role,
                .open_state = reader.open_state(),
                .validation_scope = "head",
                .has_preview = preview.has_value(),
                .preview_bytes = preview ? preview->bytes : 0,
                .preview_width = preview_width,
                .preview_height = preview_height,
                .title = std::move(title),
                .min_reader_version = reader.commit().min_reader_version,
                .min_safe_writer_version = reader.commit().min_safe_writer_version,
                .commit_kind = reader.commit().kind,
                .diagnostic = {},
            };
            for (const auto& warning : reader.warnings()) {
                if (!result.diagnostic.empty()) {
                    result.diagnostic += "; ";
                }
                result.diagnostic += warning;
            }
            return result;
        }

        std::string classification_diagnostic(const OpenClassification& classification) {
            if (!classification.diagnostic.empty()) {
                return classification.diagnostic;
            }
            switch (classification.state) {
            case OpenState::RepairOnly:
                return "This project requires explicit repair.";
            case OpenState::HardFail:
                return "The project could not be classified.";
            case OpenState::UnsupportedNewer:
                return "This project requires a newer LichtFeld version.";
            case OpenState::Open:
                return {};
            }
            return "The project has an unknown open state.";
        }

        ProjectInspectorChapter chapter_from_row(const ChunkInfo& row) {
            return ProjectInspectorChapter{
                .fourcc = row.key.fourcc,
                .instance_uuid = row.key.instance_uuid,
                .row_kind = row.row_kind,
                .compression = row.compression,
                .stored_bytes = row.stored_bytes,
                .uncompressed_bytes = row.uncompressed_bytes,
                .source_generation = row.source_generation,
            };
        }

        bool needs_full_read(const ChunkInfo& row) noexcept {
            return (row.flags & TENSOR_PAYLOAD) != 0 ||
                   row.key.fourcc == FOURCC_CKPT ||
                   row.key.fourcc == FOURCC_DSRC ||
                   row.key.fourcc == FOURCC_SPLT ||
                   row.key.fourcc == FOURCC_PCLD ||
                   row.key.fourcc == FOURCC_MESH;
        }

        std::optional<std::filesystem::path>
        reference_path(const ReferenceRecord& record,
                       const std::filesystem::path& project_path) {
            std::filesystem::path preferred(record.locator.preferred);
            if (preferred.empty()) {
                if (!record.locator.absolute_fallback) {
                    return std::nullopt;
                }
                preferred = *record.locator.absolute_fallback;
            } else if (record.locator.base == LocatorBase::Project ||
                       record.locator.base == LocatorBase::Dataset) {
                preferred = project_path.parent_path() / preferred;
            }
            return preferred;
        }

        std::optional<core::CheckpointHeader>
        read_checkpoint_header(const ProjectReader& reader, const ChunkInfo& row,
                               const std::uint64_t byte_budget) {
            if (byte_budget < sizeof(core::CheckpointHeader) ||
                row.uncompressed_bytes < sizeof(core::CheckpointHeader)) {
                return std::nullopt;
            }

            std::array<std::byte, sizeof(core::CheckpointHeader)> prefix{};
            bool read = false;
            if (const auto* selected = reader.find(row.key);
                selected != nullptr && selected->header_offset == row.header_offset &&
                selected->payload_offset == row.payload_offset &&
                selected->stored_bytes == row.stored_bytes) {
                read = static_cast<bool>(reader.read_logical_prefix(row, prefix));
            } else if (row.compression == Compression::Stored) {
                std::ifstream stream(reader.path(), std::ios::binary);
                if (stream) {
                    stream.seekg(static_cast<std::streamoff>(row.payload_offset));
                    stream.read(reinterpret_cast<char*>(prefix.data()),
                                static_cast<std::streamsize>(prefix.size()));
                    read = stream.gcount() == static_cast<std::streamsize>(prefix.size());
                }
            }
            if (!read) {
                return std::nullopt;
            }
            SpanStreambuf buffer{std::span<const std::byte>(prefix)};
            std::istream input(&buffer);
            auto header = core::load_checkpoint_header(input, row.uncompressed_bytes);
            if (!header) {
                return std::nullopt;
            }
            return *header;
        }

        lfs::Result<void> read_current_json(const ProjectReader& reader, const Fourcc fourcc,
                                            std::vector<std::byte>& bytes) {
            bytes.clear();
            const auto* row = reader.find(fourcc, reader.superblock().project_uuid);
            if (row == nullptr || row->row_kind != RowKind::Live) {
                return {};
            }
            auto read = reader.read_chunk(*row);
            if (!read) {
                return lfs::Result<void>::failure(std::move(read).error());
            }
            bytes = std::move(*read);
            return {};
        }

    } // namespace

    ProjectFilterFacts inspect_project_filter_facts(const ProjectReader& reader) {
        ProjectFilterFacts facts;
        if (reader.open_state() != OpenState::Open)
            return facts;
        std::vector<std::byte> bytes;
        if (read_current_json(reader, FOURCC_SCNG, bytes) && !bytes.empty()) {
            if (auto scene = SceneGraphChapter::from_bytes(bytes)) {
                if (auto nodes = scene->nodes()) {
                    facts.has_dataset = std::ranges::any_of(*nodes, [](const auto& node) {
                        return node.type == "dataset" && !node.name.empty();
                    });
                }
                if (auto training = scene->training_model_uuid(); training && *training) {
                    facts.has_checkpoint = std::ranges::any_of(reader.chunks(), [&](const auto& row) {
                        return row.row_kind == RowKind::Live && row.key.fourcc == FOURCC_CKPT &&
                               read_checkpoint_header(reader, row, sizeof(core::CheckpointHeader)).has_value();
                    });
                }
            }
        }
        if (!facts.has_dataset && read_current_json(reader, FOURCC_REFS, bytes) && !bytes.empty()) {
            if (auto refs = ReferencesChapter::from_bytes(bytes)) {
                if (auto records = refs->records()) {
                    facts.has_dataset = std::ranges::any_of(*records, [](const auto& ref) {
                        std::string kind = ref.kind;
                        std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c) {
                            return static_cast<char>(std::tolower(c));
                        });
                        return kind == "dataset" || kind == "images" || kind == "data";
                    });
                }
            }
        }
        if (!facts.has_dataset && read_current_json(reader, FOURCC_PRMS, bytes) && !bytes.empty()) {
            if (auto params = ParametersChapter::from_bytes(bytes)) {
                const auto embedded = params->embedded_dataset();
                const auto snapshot = params->snapshot();
                facts.has_dataset = (embedded && embedded->has_value()) ||
                                    (snapshot && !snapshot->dataset.data_path.empty());
            }
        }
        return facts;
    }

    std::pair<std::uint32_t, std::uint32_t>
    project_preview_dimensions(const ProjectReader& reader) {
        return inspect_preview_dimensions(reader.path(), reader.preview());
    }

    lfs::Result<ProjectInspectorCard>
    inspect_project_card(const std::filesystem::path& path) {
        if (auto card = inspect_project_card_headers(path)) {
            if (card->open_state == OpenState::Open) {
                auto reader = ProjectReader::open(path);
                if (reader) {
                    auto full_card = card_from_reader(*reader);
                    full_card.validation_scope = card->validation_scope;
                    if (!card->diagnostic.empty()) {
                        full_card.diagnostic = card->diagnostic;
                    }
                    return full_card;
                }
            }
            return *card;
        }
        ReaderOptions options;
        options.allow_unsupported_inspection = true;
        auto opened = ProjectReader::open(path, options);
        if (opened) {
            return card_from_reader(*opened);
        }

        const auto classification = ProjectReader::classify(path, options);
        ProjectInspectorCard result;
        result.path = path;
        result.generation = classification.generation;
        result.open_state = classification.state;
        result.validation_scope = "head";
        result.diagnostic = classification_diagnostic(classification);
        return result;
    }

    lfs::Result<ProjectInspectorDetails>
    inspect_project_details(const std::filesystem::path& path,
                            const std::uint64_t checkpoint_byte_budget) {
        ReaderOptions options;
        options.allow_unsupported_inspection = true;
        auto opened = ProjectReader::open(path, options);
        if (!opened) {
            return std::move(opened).error();
        }
        ProjectReader reader = std::move(*opened);
        if (reader.open_state() != OpenState::Open) {
            return inspector_error<ProjectInspectorDetails>(
                lfs::ErrorCode::Unsupported,
                "Structural project details are unavailable for this project.",
                "details requires a fully supported project", path);
        }

        auto storage = project_storage_stats(reader);
        if (!storage) {
            return std::move(storage).error();
        }
        auto lineage_chunks = reader.lineage_chunks();
        if (!lineage_chunks) {
            return std::move(lineage_chunks).error();
        }

        ProjectInspectorDetails result;
        result.card = card_from_reader(reader);
        result.storage = *storage;
        result.card.validation_scope = "structure";
        const auto lineage = reader.lineage();
        std::map<ChunkKey, ChunkInfo, ChunkKeyLess> live;
        std::map<std::string, std::size_t> checkpoint_indices;
        std::set<std::string> tombstoned_checkpoints;
        const auto oldest_first = [&]() {
            std::vector<std::size_t> indices(lineage.size());
            for (std::size_t index = 0; index < indices.size(); ++index) {
                indices[index] = index;
            }
            return indices;
        }();
        std::uint64_t previous_end = APPEND_REGION_OFFSET;
        std::set<std::string> previous_removals;
        for (const std::size_t lineage_index : oldest_first) {
            const auto& commit = lineage[lineage_index];
            const auto& rows = (*lineage_chunks)[lineage_index];
            for (const auto& row : rows) {
                if (row.row_kind == RowKind::Tombstone) {
                    live.erase(row.key);
                    if (row.key.fourcc == FOURCC_CKPT) {
                        tombstoned_checkpoints.insert(row.key.instance_uuid.to_string());
                        const auto found = checkpoint_indices.find(
                            row.key.instance_uuid.to_string());
                        if (found != checkpoint_indices.end()) {
                            result.retained_checkpoints[found->second].retained = false;
                        }
                    }
                    continue;
                }
                if (row.row_kind != RowKind::Live) {
                    continue;
                }
                live[row.key] = row;
                if (row.key.fourcc == FOURCC_CKPT &&
                    !checkpoint_indices.contains(row.key.instance_uuid.to_string())) {
                    const auto header = read_checkpoint_header(reader, row, checkpoint_byte_budget);
                    ProjectInspectorCheckpoint checkpoint{
                        .instance_uuid = row.key.instance_uuid,
                        .source_generation = row.source_generation,
                        .iteration = header ? header->iteration : 0,
                        .gaussians = header ? header->num_gaussians : 0,
                        .sh_degree = header ? header->sh_degree : 0,
                        .binds_scene_graph = false,
                        .header_reachable = header.has_value(),
                        .retained = !tombstoned_checkpoints.contains(
                            row.key.instance_uuid.to_string()),
                    };
                    checkpoint_indices.emplace(
                        row.key.instance_uuid.to_string(),
                        result.retained_checkpoints.size());
                    result.retained_checkpoints.push_back(std::move(checkpoint));
                }
            }
            result.save_history.push_back(ProjectInspectorSave{
                .sequence = commit.generation,
                .generation = commit.generation,
                .kind = commit.kind,
                .saved_at_unix_ns = commit.wallclock_unix_ns,
                .bytes_added = commit.committed_file_end >= previous_end
                                   ? commit.committed_file_end - previous_end
                                   : 0,
                .holds_checkpoint = std::ranges::any_of(
                    live, [](const auto& item) { return item.first.fourcc == FOURCC_CKPT; }),
            });
            auto& save = result.save_history.back();
            // Read the small metadata chapters from this generation. The bound
            // checkpoint, not the first UUID in the index, owns its training facts.
            auto historical = ProjectReader::open_generation(path, commit.generation, reader.reader_options());
            if (!historical)
                return std::move(historical).error();
            std::vector<std::byte> metadata;
            if (auto read = read_current_json(*historical, FOURCC_SCNG, metadata); !read)
                return std::move(read).error();
            if (!metadata.empty()) {
                auto scene = SceneGraphChapter::from_bytes(metadata);
                if (!scene)
                    return std::move(scene).error();
                auto training = scene->training_model_uuid();
                if (!training)
                    return std::move(training).error();
                if (*training) {
                    auto node = scene->find(**training);
                    if (!node)
                        return std::move(node).error();
                    if (*node && (*node)->payload && (*node)->payload->fourcc == "CKPT") {
                        const auto* checkpoint = historical->find(FOURCC_CKPT, (*node)->payload->instance_uuid);
                        if (checkpoint) {
                            if (auto header = read_checkpoint_header(*historical, *checkpoint, checkpoint_byte_budget)) {
                                save.checkpoint_iteration = header->iteration;
                                save.gaussians = header->num_gaussians;
                            }
                        }
                    }
                }
            }
            if (auto read = read_current_json(*historical, FOURCC_PRMS, metadata); !read)
                return std::move(read).error();
            if (!metadata.empty()) {
                auto parameters = ParametersChapter::from_bytes(metadata);
                if (!parameters)
                    return std::move(parameters).error();
                auto snapshot = parameters->snapshot();
                if (!snapshot)
                    return std::move(snapshot).error();
                save.strategy = snapshot->active_strategy;
                const auto iterations = parameters->dom().get<std::uint64_t>(
                    "presets." + save.strategy + ".current.iterations");
                if (iterations && *iterations > 0)
                    save.planned_iterations = *iterations;
            }
            {
                if (auto read = read_current_json(*historical, FOURCC_PROJ, metadata); !read)
                    return std::move(read).error();
                if (!metadata.empty()) {
                    auto project = ProjectChapter::from_bytes(metadata);
                    if (!project)
                        return std::move(project).error();
                    const auto& dom = project->dom();
                    std::set<std::string> removals;
                    std::string legacy_removal;
                    if (const auto pending = dom.get_json("contents_removals");
                        pending && pending->is_object() &&
                        pending->value("file_uuid", std::string{}) == historical->superblock().file_uuid.to_string() &&
                        pending->contains("rows") && (*pending)["rows"].is_array()) {
                        for (const auto& part : (*pending)["rows"]) {
                            if (!part.is_object())
                                continue;
                            const auto id = part.value("id", std::string{});
                            removals.insert(id);
                            if (!previous_removals.contains(id))
                                legacy_removal = part.value("kind", std::string{}) + "_removed";
                        }
                    }
                    previous_removals = std::move(removals);
                    if (save.kind == CommitKind::Explicit &&
                        dom.get<std::string>("contents_edit.file_uuid").value_or("") == historical->superblock().file_uuid.to_string() &&
                        dom.get<std::uint64_t>("contents_edit.first_generation").value_or(0) <= commit.generation &&
                        dom.get<std::uint64_t>("contents_edit.last_generation").value_or(0) >= commit.generation) {
                        save.kind = CommitKind::Contents;
                    }
                    // Older Contents removals used EXPLICIT but wrote a durable
                    // removal record. A newly added record proves that edit.
                    if (save.kind == CommitKind::Explicit && !legacy_removal.empty() && result.save_history.size() > 1) {
                        const auto& source = result.save_history[result.save_history.size() - 2];
                        const bool derived = source.kind == CommitKind::Contents || source.kind == CommitKind::Compaction;
                        save.kind = CommitKind::Contents;
                        save.operation = legacy_removal;
                        save.source_save_generation = derived ? source.source_save_generation : source.generation;
                        save.source_saved_at_unix_ns = derived ? source.source_saved_at_unix_ns : source.saved_at_unix_ns;
                        save.source_save_kind = derived ? source.source_save_kind : source.kind;
                        previous_end = commit.committed_file_end;
                        continue;
                    }
                    if (save.kind != CommitKind::Contents && save.kind != CommitKind::Compaction) {
                        previous_end = commit.committed_file_end;
                        continue;
                    }
                    save.operation = save.kind == CommitKind::Compaction ? "compacted"
                                                                         : dom.get<std::string>("contents_edit.operation").value_or("changed");
                    save.source_save_generation = dom.get<std::uint64_t>("contents_edit.source_save.generation").value_or(0);
                    save.source_saved_at_unix_ns = dom.get<std::uint64_t>("contents_edit.source_save.saved_at_unix_ns").value_or(0);
                    const auto kind = dom.get<std::uint32_t>("contents_edit.source_save.kind").value_or(1);
                    if (kind >= 1 && kind <= 3)
                        save.source_save_kind = static_cast<CommitKind>(kind);
                    if (save.source_saved_at_unix_ns != 0) {
                        save.checkpoint_iteration = dom.get<std::int32_t>("contents_edit.source_save.checkpoint_iteration");
                        save.planned_iterations = dom.get<std::uint64_t>("contents_edit.source_save.planned_iterations");
                        save.gaussians = dom.get<std::uint32_t>("contents_edit.source_save.gaussians");
                        save.strategy = dom.get<std::string>("contents_edit.source_save.strategy").value_or("");
                    }
                    if (save.kind == CommitKind::Compaction)
                        save.source_save_generation = save.generation;
                }
            }
            previous_end = commit.committed_file_end;
        }

        for (const auto& checkpoint : result.retained_checkpoints) {
            const auto* row = reader.find(FOURCC_CKPT, checkpoint.instance_uuid);
            if (row != nullptr && row->row_kind == RowKind::Live) {
                const auto header = read_checkpoint_header(reader, *row, checkpoint_byte_budget);
                if (header) {
                    const auto index = checkpoint_indices.at(
                        checkpoint.instance_uuid.to_string());
                    result.retained_checkpoints[index].iteration = header->iteration;
                    result.retained_checkpoints[index].gaussians = header->num_gaussians;
                    result.retained_checkpoints[index].sh_degree = header->sh_degree;
                    result.retained_checkpoints[index].header_reachable = true;
                }
            }
        }

        for (const auto& row : reader.chunks()) {
            if (row.row_kind != RowKind::Live) {
                continue;
            }
            result.chapters.push_back(chapter_from_row(row));
            if (needs_full_read(row)) {
                result.chapters_requiring_full_read.push_back(row.key_string());
            }
        }

        std::vector<std::byte> bytes;
        if (auto status = read_current_json(reader, FOURCC_PROJ, bytes); !status) {
            return std::move(status).error();
        } else if (!bytes.empty()) {
            auto parsed = ProjectChapter::from_bytes(bytes);
            if (!parsed) {
                return std::move(parsed).error();
            }
            const auto manifest = parsed->manifest();
            if (!manifest) {
                return std::move(manifest).error();
            }
            result.manifest["application_name"] = manifest->application_name;
            result.manifest["schema_version"] = std::format(
                "{}.{}.{}", manifest->schema_version.major,
                manifest->schema_version.minor, manifest->schema_version.patch);
            result.manifest["minimum_reader_version"] = std::format(
                "{}.{}.{}", manifest->minimum_reader_version.major,
                manifest->minimum_reader_version.minor,
                manifest->minimum_reader_version.patch);
            result.manifest["minimum_safe_writer_version"] = std::format(
                "{}.{}.{}", manifest->minimum_safe_writer_version.major,
                manifest->minimum_safe_writer_version.minor,
                manifest->minimum_safe_writer_version.patch);
            auto license = parsed->license();
            if (!license) {
                return std::move(license).error();
            }
            result.license = std::move(*license);
            const auto removals = parsed->dom().get_json("contents_removals");
            if (removals && removals->is_object() &&
                removals->value("file_uuid", std::string{}) == reader.superblock().file_uuid.to_string()) {
                result.manifest["contents_removals"] = removals->dump();
            }
        }

        if (auto status = read_current_json(reader, FOURCC_SCNG, bytes); !status) {
            return std::move(status).error();
        } else if (!bytes.empty()) {
            auto parsed = SceneGraphChapter::from_bytes(bytes);
            if (!parsed) {
                return std::move(parsed).error();
            }
            auto nodes = parsed->nodes();
            if (!nodes) {
                return std::move(nodes).error();
            }
            for (const auto& node : *nodes) {
                ++result.scene_graph.node_counts_by_type[node.type];
                if (node.type == "dataset" && result.scene_graph.dataset_node_name.empty()) {
                    result.scene_graph.dataset_node_name = node.name;
                }
            }
            auto training = parsed->training_model_uuid();
            if (!training) {
                return std::move(training).error();
            }
            result.scene_graph.training_node_id = *training;
            std::optional<lfs::core::Uuid> bound_checkpoint;
            if (*training) {
                const auto training_node = std::ranges::find_if(
                    *nodes, [&](const auto& node) {
                        return node.uuid == **training;
                    });
                if (training_node != nodes->end() && training_node->payload &&
                    training_node->payload->fourcc == "CKPT") {
                    bound_checkpoint = training_node->payload->instance_uuid;
                }
            }
            for (auto& checkpoint : result.retained_checkpoints) {
                checkpoint.binds_scene_graph =
                    bound_checkpoint.has_value() &&
                    checkpoint.instance_uuid == *bound_checkpoint;
            }
        }

        std::filesystem::path parameter_dataset_path;
        if (auto status = read_current_json(reader, FOURCC_PRMS, bytes); !status) {
            return std::move(status).error();
        } else if (!bytes.empty()) {
            auto parsed = ParametersChapter::from_bytes(bytes);
            if (!parsed) {
                return std::move(parsed).error();
            }
            auto snapshot = parsed->snapshot();
            if (!snapshot) {
                return std::move(snapshot).error();
            }
            result.parameters.active_strategy = snapshot->active_strategy;
            const auto iterations = parsed->dom().get<std::uint64_t>(
                "presets." + snapshot->active_strategy + ".current.iterations");
            if (iterations && *iterations > 0)
                result.parameters.planned_iterations = *iterations;
            parameter_dataset_path = snapshot->dataset.data_path;
            auto embedded = parsed->embedded_dataset();
            if (!embedded) {
                return std::move(embedded).error();
            }
            if (*embedded) {
                result.parameters.embedded_dataset_present = true;
                result.parameters.embedded_dataset_complete = (**embedded).complete;
                for (const auto& entry : (**embedded).entries) {
                    if (entry.kind == "image") {
                        ++result.parameters.embedded_images;
                    } else if (entry.kind == "normal") {
                        ++result.parameters.embedded_normals;
                    } else if (entry.kind == "sparse") {
                        ++result.parameters.embedded_sparse;
                    }
                }
            }
        }

        if (auto status = read_current_json(reader, FOURCC_REFS, bytes); !status) {
            return std::move(status).error();
        } else if (!bytes.empty()) {
            auto parsed = ReferencesChapter::from_bytes(bytes);
            if (!parsed) {
                return std::move(parsed).error();
            }
            auto records = parsed->records();
            if (!records) {
                return std::move(records).error();
            }
            for (const auto& record : *records) {
                const auto path_value = reference_path(record, reader.path());
                bool reachable = false;
                if (path_value) {
                    std::error_code error;
                    reachable = std::filesystem::exists(*path_value, error) && !error;
                }
                result.references.push_back(ProjectInspectorReference{
                    .key = record.key,
                    .kind = record.kind,
                    .path = path_value.value_or(std::filesystem::path{}),
                    .reachable = reachable,
                });
            }
        }

        if (!parameter_dataset_path.empty() &&
            !std::ranges::any_of(result.references, [](const auto& ref) { return ref.kind == "dataset"; })) {
            if (parameter_dataset_path.is_relative())
                parameter_dataset_path = reader.path().parent_path() / parameter_dataset_path;
            std::error_code error;
            const bool reachable = std::filesystem::is_directory(parameter_dataset_path, error) && !error;
            result.references.push_back({"training_parameters", "dataset", parameter_dataset_path, reachable});
        }

        if (auto status = read_current_json(reader, FOURCC_METR, bytes); !status) {
            return std::move(status).error();
        } else if (!bytes.empty()) {
            auto parsed = MetricsChapter::from_bytes(bytes);
            if (!parsed) {
                return std::move(parsed).error();
            }
            result.metrics.loss_samples = parsed->loss_history.size();
            result.metrics.psnr_samples = parsed->psnr_history.size();
            if (!parsed->loss_history.empty()) {
                result.metrics.last_loss = parsed->loss_history.back();
            }
            if (!parsed->psnr_history.empty()) {
                result.metrics.last_psnr = parsed->psnr_history.back();
            }
            result.metrics.last_evaluation = parsed->last_evaluation;
        }

        std::error_code sidecar_error;
        result.autosave_sidecar_present = std::filesystem::exists(
                                              autosave_sidecar_path(path), sidecar_error) &&
                                          !sidecar_error;
        return result;
    }

    lfs::Result<std::vector<std::byte>>
    read_project_preview(const std::filesystem::path& path) {
        auto reader = ProjectReader::open(path);
        if (!reader) {
            return std::move(reader).error();
        }
        return reader->read_preview();
    }

} // namespace lfs::io::project
