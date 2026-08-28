/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_container.hpp"

#include "core/logger.hpp"
#include "crc32c.hpp"
#include "project_container_internal.hpp"
#include "project_framing.hpp"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <format>
#include <istream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <memoryapi.h>
#include <sysinfoapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace lfs::io::project {

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
        constexpr std::array<std::byte, 8> INDEX_MAGIC = {
            std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'I'},
            std::byte{'N'}, std::byte{'D'}, std::byte{'E'}, std::byte{'X'}};
        constexpr std::array<std::byte, 8> BLOCK_CRC_MAGIC = {
            std::byte{'L'}, std::byte{'F'}, std::byte{'S'}, std::byte{'B'},
            std::byte{'C'}, std::byte{'R'}, std::byte{'C'}, std::byte{0}};
        constexpr std::uint32_t BYTE_ORDER_TAG = 0x01020304u;
        constexpr std::uint32_t KNOWN_CHUNK_FLAGS = TENSOR_PAYLOAD | HAS_BLOCK_CRCS;
        constexpr std::uint64_t MAX_INDEX_UNCOMPRESSED_BYTES = 512ull * 1024 * 1024;
        constexpr std::uint64_t MAX_BLOCK_CRC_TABLE_BYTES = 512ull * 1024 * 1024;
        // Anti-zstd-bomb for JSON/index-era small chunk classes (and any
        // non-payload-class materialize). Declared logical size is still
        // cross-checked against the zstd frame content size before allocation.
        constexpr std::uint64_t MAX_MATERIALIZED_CHUNK_BYTES = 512ull * 1024 * 1024;
        // TENSOR_PAYLOAD / lazy-binary (CKPT, PPIS, ByteShuffleZstdFramed): decoded
        // size is declared in the chunk header and CRC-validated, so the
        // bomb-guard property does not need a small blanket cap. 16 GiB is
        // still size_t-bounded via the framed decoder / read_vector.
        constexpr std::uint64_t MAX_PAYLOAD_MATERIALIZED_BYTES =
            16ull * 1024 * 1024 * 1024;

        bool is_payload_class_chunk(const ChunkInfo& row) noexcept {
            return (row.flags & TENSOR_PAYLOAD) != 0 ||
                   row.key.fourcc == FOURCC_CKPT ||
                   row.key.fourcc == FOURCC_PPIS;
        }

        std::optional<std::uint64_t> payload_materialized_bytes_override;

        lfs::Result<void> status_failure(lfs::Error error) {
            return lfs::Result<void>::failure(std::move(error));
        }

        template <std::size_t Size>
        std::span<const std::byte> byte_span(const std::array<std::byte, Size>& bytes) {
            return std::span<const std::byte>(bytes.data(), bytes.size());
        }

        template <std::size_t Size>
        std::span<std::byte> byte_span(std::array<std::byte, Size>& bytes) {
            return std::span<std::byte>(bytes.data(), bytes.size());
        }

        std::uint8_t read_u8(const std::span<const std::byte> bytes,
                             const std::size_t offset) noexcept {
            return std::to_integer<std::uint8_t>(bytes[offset]);
        }

        std::uint16_t read_u16(const std::span<const std::byte> bytes,
                               const std::size_t offset) noexcept {
            return static_cast<std::uint16_t>(read_u8(bytes, offset)) |
                   (static_cast<std::uint16_t>(read_u8(bytes, offset + 1)) << 8);
        }

        std::uint32_t read_u32(const std::span<const std::byte> bytes,
                               const std::size_t offset) noexcept {
            std::uint32_t result = 0;
            for (std::size_t index = 0; index < 4; ++index) {
                result |= static_cast<std::uint32_t>(read_u8(bytes, offset + index))
                          << (index * 8);
            }
            return result;
        }

        std::uint64_t read_u64(const std::span<const std::byte> bytes,
                               const std::size_t offset) noexcept {
            std::uint64_t result = 0;
            for (std::size_t index = 0; index < 8; ++index) {
                result |= static_cast<std::uint64_t>(read_u8(bytes, offset + index))
                          << (index * 8);
            }
            return result;
        }

        lfs::core::Uuid read_uuid(const std::span<const std::byte> bytes,
                                  const std::size_t offset) noexcept {
            lfs::core::Uuid uuid;
            for (std::size_t index = 0; index < uuid.bytes.size(); ++index) {
                uuid.bytes[index] = read_u8(bytes, offset + index);
            }
            return uuid;
        }

        Fourcc read_fourcc(const std::span<const std::byte> bytes,
                           const std::size_t offset) noexcept {
            Fourcc fourcc;
            for (std::size_t index = 0; index < fourcc.bytes.size(); ++index) {
                fourcc.bytes[index] = read_u8(bytes, offset + index);
            }
            return fourcc;
        }

        CapabilitySet read_capabilities(const std::span<const std::byte> bytes,
                                        const std::size_t offset) noexcept {
            std::array<std::uint8_t, 16> result{};
            for (std::size_t index = 0; index < result.size(); ++index) {
                result[index] = read_u8(bytes, offset + index);
            }
            return CapabilitySet(result);
        }

        bool bytes_equal(const std::span<const std::byte> bytes, const std::size_t offset,
                         const std::span<const std::byte> expected) noexcept {
            return bytes.size() >= offset + expected.size() &&
                   std::equal(expected.begin(), expected.end(), bytes.begin() + offset);
        }

        std::optional<std::size_t>
        first_nonzero(const std::span<const std::byte> bytes, const std::size_t begin,
                      const std::size_t end) noexcept {
            for (std::size_t offset = begin; offset < end; ++offset) {
                if (bytes[offset] != std::byte{0}) {
                    return offset;
                }
            }
            return std::nullopt;
        }

        bool all_zero(const std::span<const std::byte> bytes) noexcept {
            return std::all_of(bytes.begin(), bytes.end(),
                               [](const std::byte value) { return value == std::byte{0}; });
        }

        lfs::Error format_error(const std::filesystem::path& path, const std::uint64_t offset,
                                const std::string_view field, const std::string_view expected,
                                const std::string& got,
                                const lfs::ErrorCode code = lfs::ErrorCode::DataLoss) {
            return detail::project_error(
                code,
                code == lfs::ErrorCode::Unsupported
                    ? "This project requires a newer LichtFeld version."
                    : "The project container is corrupt.",
                std::format("expected {}, got {}", expected, got), path, offset, field);
        }

        lfs::Result<void> require(const bool condition, const std::filesystem::path& path,
                                  const std::uint64_t offset, const std::string_view field,
                                  const std::string_view expected, const std::string& got,
                                  const lfs::ErrorCode code = lfs::ErrorCode::DataLoss) {
            if (!condition) {
                return status_failure(format_error(path, offset, field, expected, got, code));
            }
            return {};
        }

        lfs::Result<void> require_zero(const std::span<const std::byte> bytes,
                                       const std::size_t begin, const std::size_t end,
                                       const std::filesystem::path& path,
                                       const std::uint64_t base_offset,
                                       const std::string_view field) {
            if (const auto bad = first_nonzero(bytes, begin, end)) {
                return status_failure(format_error(
                    path, base_offset + *bad, field, "zero",
                    std::format("0x{:02x}", read_u8(bytes, *bad))));
            }
            return {};
        }

        template <std::size_t Size>
        lfs::Result<std::array<std::byte, Size>>
        read_fixed(const detail::NativeFile& file, const std::uint64_t offset,
                   const std::uint64_t physical_size, const std::uint64_t authority_end,
                   const std::string_view field) {
            auto end = detail::checked_add(offset, Size, file.path(), offset, field);
            if (!end) {
                return std::move(end).error();
            }
            const std::uint64_t limit = std::min(physical_size, authority_end);
            if (*end > limit) {
                return format_error(file.path(), offset, field,
                                    std::format("{} bytes within authority end 0x{:x}", Size,
                                                limit),
                                    std::format("range [0x{:x},0x{:x})", offset, *end));
            }
            std::array<std::byte, Size> result{};
            if (auto read = file.read_exact(offset, byte_span(result)); !read) {
                return std::move(read).error();
            }
            return result;
        }

        lfs::Result<std::uint64_t>
        checked_align_up(const std::uint64_t value, const std::uint64_t alignment,
                         const std::filesystem::path& path,
                         const std::uint64_t field_offset,
                         const std::string_view field) {
            assert(alignment != 0 && (alignment & (alignment - 1)) == 0);
            auto sum = detail::checked_add(value, alignment - 1, path, field_offset, field);
            if (!sum) {
                return std::move(sum).error();
            }
            return *sum & ~(alignment - 1);
        }

        std::uint64_t maximum_stored_index_bytes(const Compression compression) noexcept {
            if (compression == Compression::Stored) {
                return MAX_INDEX_UNCOMPRESSED_BYTES;
            }
            return static_cast<std::uint64_t>(
                ZSTD_compressBound(static_cast<std::size_t>(MAX_INDEX_UNCOMPRESSED_BYTES)));
        }

        lfs::Result<std::vector<std::byte>>
        read_vector(const detail::NativeFile& file, const std::uint64_t offset,
                    const std::uint64_t size, const std::uint64_t physical_size,
                    const std::uint64_t authority_end, const std::string_view field,
                    const std::uint64_t maximum_size,
                    const lfs::ErrorCode oversize_code = lfs::ErrorCode::DataLoss) {
            auto end = detail::checked_add(offset, size, file.path(), offset, field);
            if (!end) {
                return std::move(end).error();
            }
            const std::uint64_t limit = std::min(physical_size, authority_end);
            if (*end > limit || size > maximum_size ||
                size > std::numeric_limits<std::size_t>::max()) {
                return format_error(
                    file.path(), offset, field,
                    std::format("range within authority end 0x{:x}, implementation maximum "
                                "{}, and addressable memory",
                                limit, maximum_size),
                    std::format("[0x{:x},0x{:x}) size {}", offset, *end, size),
                    oversize_code);
            }
            try {
                std::vector<std::byte> result(static_cast<std::size_t>(size));
                if (auto read = file.read_exact(offset, result); !read) {
                    return std::move(read).error();
                }
                return result;
            } catch (const std::bad_alloc&) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to read this project region.",
                    std::format("allocation of {} bytes failed", size), file.path(), offset,
                    field);
            } catch (const std::length_error& error) {
                return detail::project_error(
                    lfs::ErrorCode::BoundsViolation,
                    "The project region cannot be represented by this build.",
                    std::format("allocation of {} bytes failed: {}", size, error.what()),
                    file.path(), offset, field);
            }
        }

        lfs::Result<void>
        validate_zero_range(const detail::NativeFile& file, const std::uint64_t offset,
                            const std::uint64_t size, const std::uint64_t physical_size,
                            const std::uint64_t authority_end,
                            const std::string_view field,
                            const std::uint64_t maximum_size) {
            auto end = detail::checked_add(offset, size, file.path(), offset, field);
            if (!end) {
                return status_failure(std::move(end).error());
            }
            const std::uint64_t limit = std::min(physical_size, authority_end);
            if (*end > limit || size > maximum_size) {
                return status_failure(format_error(
                    file.path(), offset, field,
                    std::format("zero range within authority end 0x{:x} and no larger than {}",
                                limit, maximum_size),
                    std::format("[0x{:x},0x{:x}) size {}", offset, *end, size)));
            }
            std::array<std::byte, 64 * 1024> buffer{};
            std::uint64_t cursor = offset;
            while (cursor < *end) {
                const std::size_t count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(*end - cursor, buffer.size()));
                auto bytes = std::span<std::byte>(buffer.data(), count);
                if (auto read = file.read_exact(cursor, bytes); !read) {
                    return read;
                }
                if (const auto bad = first_nonzero(bytes, 0, bytes.size())) {
                    return status_failure(format_error(
                        file.path(), cursor + *bad, field, "zero",
                        std::format("0x{:02x}", read_u8(bytes, *bad))));
                }
                cursor += count;
            }
            return {};
        }

        std::string uuid_string(const lfs::core::Uuid& uuid) {
            return uuid.to_string();
        }

        struct ParsedCommit {
            CommitInfo info;
            lfs::core::Uuid project_uuid;
            lfs::core::Uuid file_uuid;
        };

        using CommitRecordCache = std::map<std::uint64_t, ParsedCommit>;

        struct ParsedHead {
            HeadInfo info;
            ParsedCommit commit;
            std::vector<ParsedCommit> lineage;
            std::vector<ChunkInfo> chunks;
            std::uint32_t index_flags = 0;
            std::optional<lfs::Error> compatibility_error;
        };

        struct HeadAttempt {
            std::uint32_t slot_id = 0;
            bool blank = false;
            std::optional<std::uint64_t> sequence_hint;
            std::optional<ParsedHead> head;
            std::optional<lfs::Error> error;
        };

        struct ReaderState {
            std::filesystem::path path;
            std::shared_ptr<detail::NativeFile> file;
            std::uint64_t physical_size = 0;
            SuperblockInfo superblock;
            ParsedHead selected;
            std::vector<std::string> warnings;
            OpenState open_state = OpenState::Open;
            ReaderOptions options;
        };

        struct ParseOutcome {
            OpenState state = OpenState::HardFail;
            std::shared_ptr<ReaderState> reader;
            std::optional<lfs::Error> error;
        };

        lfs::Result<SuperblockInfo>
        parse_superblock(const detail::NativeFile& file, const std::uint64_t physical_size) {
            auto raw_result =
                read_fixed<SUPERBLOCK_BYTES>(file, 0, physical_size, physical_size, "superblock");
            if (!raw_result) {
                return std::move(raw_result).error();
            }
            const auto& raw = *raw_result;
            const auto bytes = byte_span(raw);
            if (auto valid = require(bytes_equal(bytes, 0, SUPERBLOCK_MAGIC), file.path(), 0,
                                     "superblock.magic", "89 4c 46 53 0d 0a 1a 0a",
                                     "different bytes");
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t expected_crc = crc32c(0, raw.data(), 252);
            if (auto valid =
                    require(read_u32(bytes, 252) == expected_crc, file.path(), 252,
                            "superblock.crc32c", std::format("0x{:08x}", expected_crc),
                            std::format("0x{:08x}", read_u32(bytes, 252)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u16(bytes, 8) == 1, file.path(), 8,
                                     "superblock.format_major", "1",
                                     std::to_string(read_u16(bytes, 8)),
                                     lfs::ErrorCode::Unsupported);
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u32(bytes, 12) == BYTE_ORDER_TAG, file.path(), 12,
                                     "superblock.byte_order_tag", "0x01020304",
                                     std::format("0x{:08x}", read_u32(bytes, 12)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u32(bytes, 16) == SUPERBLOCK_BYTES, file.path(), 16,
                                     "superblock.superblock_bytes", "256",
                                     std::to_string(read_u32(bytes, 16)));
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t role_value = read_u32(bytes, 20);
            if (auto valid = require(role_value <= 1, file.path(), 20,
                                     "superblock.container_role", "0 or 1",
                                     std::to_string(role_value));
                !valid) {
                return std::move(valid).error();
            }
            const lfs::core::Uuid project_uuid = read_uuid(bytes, 24);
            const lfs::core::Uuid file_uuid = read_uuid(bytes, 40);
            if (auto valid = require(!project_uuid.is_nil(), file.path(), 24,
                                     "superblock.project_uuid", "non-null UUID",
                                     uuid_string(project_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(!file_uuid.is_nil(), file.path(), 40,
                                     "superblock.file_uuid", "non-null UUID",
                                     uuid_string(file_uuid));
                !valid) {
                return std::move(valid).error();
            }
            const bool geometry_valid =
                read_u64(bytes, 56) == HEAD_SLOT_OFFSETS[0] &&
                read_u64(bytes, 64) == HEAD_SLOT_OFFSETS[1] &&
                read_u32(bytes, 72) == HEAD_SLOT_BYTES && read_u32(bytes, 76) == 2 &&
                read_u64(bytes, 80) == APPEND_REGION_OFFSET;
            if (auto valid = require(
                    geometry_valid, file.path(), 56, "superblock.head_geometry",
                    "(4096,8192,4096,2,65536)",
                    std::format("({},{},{},{},{})", read_u64(bytes, 56),
                                read_u64(bytes, 64), read_u32(bytes, 72),
                                read_u32(bytes, 76), read_u64(bytes, 80)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require_zero(bytes, 120, 128, file.path(), 0,
                                          "superblock.reserved_0");
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require_zero(bytes, 144, 252, file.path(), 0,
                                          "superblock.reserved_1");
                !valid) {
                return std::move(valid).error();
            }

            const auto role = static_cast<ContainerRole>(role_value);
            const lfs::core::Uuid base_uuid = read_uuid(bytes, 96);
            const std::uint64_t autosave_sequence = read_u64(bytes, 112);
            const lfs::core::Uuid sidecar_snapshot_uuid = read_uuid(bytes, 128);
            if (role == ContainerRole::Master) {
                if (auto valid = require_zero(bytes, 96, 144, file.path(), 0,
                                              "superblock.master_sidecar_binding");
                    !valid) {
                    return std::move(valid).error();
                }
            } else {
                if (auto valid = require(!base_uuid.is_nil(), file.path(), 96,
                                         "superblock.base_explicit_commit_uuid",
                                         "non-null UUID", uuid_string(base_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(autosave_sequence >= 1, file.path(), 112,
                                         "superblock.autosave_sequence", ">= 1",
                                         std::to_string(autosave_sequence));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(!sidecar_snapshot_uuid.is_nil(), file.path(), 128,
                                         "superblock.sidecar_snapshot_uuid",
                                         "non-null UUID", uuid_string(sidecar_snapshot_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
            }

            return SuperblockInfo{
                .format = Version{read_u16(bytes, 8), read_u16(bytes, 10)},
                .role = role,
                .project_uuid = project_uuid,
                .file_uuid = file_uuid,
                .creation_time_unix_ns = read_u64(bytes, 88),
                .base_explicit_commit_uuid = base_uuid,
                .autosave_sequence = autosave_sequence,
                .sidecar_snapshot_uuid = sidecar_snapshot_uuid,
                .crc32c = read_u32(bytes, 252),
            };
        }

        lfs::Result<ParsedCommit>
        parse_commit(const detail::NativeFile& file, const std::uint64_t physical_size,
                     const SuperblockInfo& superblock, const std::uint64_t offset,
                     const std::uint64_t authority_end,
                     CommitRecordCache* cache = nullptr) {
            if (cache != nullptr) {
                if (const auto found = cache->find(offset); found != cache->end()) {
                    auto end = detail::checked_add(
                        offset, COMMIT_RECORD_BYTES, file.path(), offset,
                        "commit_record");
                    if (!end) {
                        return std::move(end).error();
                    }
                    const std::uint64_t limit =
                        std::min(physical_size, authority_end);
                    if (*end > limit) {
                        return format_error(
                            file.path(), offset, "commit_record",
                            std::format("{} bytes within authority end 0x{:x}",
                                        COMMIT_RECORD_BYTES, limit),
                            std::format("range [0x{:x},0x{:x})", offset, *end));
                    }
                    return found->second;
                }
            }
            auto raw_result = read_fixed<COMMIT_RECORD_BYTES>(
                file, offset, physical_size, authority_end, "commit_record");
            if (!raw_result) {
                return std::move(raw_result).error();
            }
            const auto& raw = *raw_result;
            const auto bytes = byte_span(raw);
            if (auto valid = require(bytes_equal(bytes, 0, COMMIT_MAGIC), file.path(), offset,
                                     "commit.magic", "LFSCOMIT", "different bytes");
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t expected_crc = crc32c(0, raw.data(), 252);
            if (auto valid =
                    require(read_u32(bytes, 252) == expected_crc, file.path(), offset + 252,
                            "commit.crc32c", std::format("0x{:08x}", expected_crc),
                            std::format("0x{:08x}", read_u32(bytes, 252)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u16(bytes, 8) == COMMIT_RECORD_BYTES, file.path(),
                                     offset + 8, "commit.record_bytes", "256",
                                     std::to_string(read_u16(bytes, 8)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u16(bytes, 10) == 1, file.path(), offset + 10,
                                     "commit.record_version", "1",
                                     std::to_string(read_u16(bytes, 10)),
                                     lfs::ErrorCode::Unsupported);
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t kind_value = read_u32(bytes, 12);
            if (auto valid = require(kind_value >= 1 && kind_value <= 4, file.path(),
                                     offset + 12, "commit.kind", "1, 2, 3, or 4",
                                     std::to_string(kind_value));
                !valid) {
                return std::move(valid).error();
            }
            const lfs::core::Uuid project_uuid = read_uuid(bytes, 16);
            const lfs::core::Uuid file_uuid = read_uuid(bytes, 32);
            const lfs::core::Uuid commit_uuid = read_uuid(bytes, 48);
            const std::uint64_t generation = read_u64(bytes, 64);
            const lfs::core::Uuid parent_uuid = read_uuid(bytes, 72);
            const std::uint64_t parent_offset = read_u64(bytes, 88);
            const lfs::core::Uuid explicit_uuid = read_uuid(bytes, 96);
            const lfs::core::Uuid snapshot_uuid = read_uuid(bytes, 112);
            const std::uint64_t index_offset = read_u64(bytes, 136);
            const std::uint64_t index_stored_bytes = read_u64(bytes, 144);
            const std::uint64_t index_uncompressed_bytes = read_u64(bytes, 152);
            const std::uint32_t index_compression_value = read_u32(bytes, 168);
            const std::uint64_t committed_end = read_u64(bytes, 176);

            if (auto valid = require(project_uuid == superblock.project_uuid, file.path(),
                                     offset + 16, "commit.project_uuid",
                                     uuid_string(superblock.project_uuid),
                                     uuid_string(project_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(file_uuid == superblock.file_uuid, file.path(), offset + 32,
                                     "commit.file_uuid", uuid_string(superblock.file_uuid),
                                     uuid_string(file_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(!commit_uuid.is_nil(), file.path(), offset + 48,
                                     "commit.commit_uuid", "non-null UUID",
                                     uuid_string(commit_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(generation >= 1, file.path(), offset + 64,
                                     "commit.generation", ">= 1",
                                     std::to_string(generation));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(!snapshot_uuid.is_nil(), file.path(), offset + 112,
                                     "commit.snapshot_uuid", "non-null UUID",
                                     uuid_string(snapshot_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(index_offset >= APPEND_REGION_OFFSET &&
                                         index_offset % CHUNK_ALIGNMENT == 0,
                                     file.path(), offset + 136, "commit.index_offset",
                                     "64-byte-aligned offset >= 65536",
                                     std::format("0x{:x}", index_offset));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(index_stored_bytes > 0, file.path(), offset + 144,
                                     "commit.index_stored_bytes", "> 0",
                                     std::to_string(index_stored_bytes));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(index_uncompressed_bytes >= INDEX_HEADER_BYTES &&
                                index_uncompressed_bytes <= MAX_INDEX_UNCOMPRESSED_BYTES,
                            file.path(), offset + 152, "commit.index_uncompressed_bytes",
                            std::format("[{},{}]", INDEX_HEADER_BYTES,
                                        MAX_INDEX_UNCOMPRESSED_BYTES),
                            std::to_string(index_uncompressed_bytes));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(index_compression_value <= 1, file.path(), offset + 168,
                                     "commit.index_compression", "0 or 1",
                                     std::to_string(index_compression_value));
                !valid) {
                return std::move(valid).error();
            }
            const auto index_compression =
                static_cast<Compression>(index_compression_value);
            const std::uint64_t maximum_stored_bytes =
                maximum_stored_index_bytes(index_compression);
            if (auto valid =
                    require(index_stored_bytes <= maximum_stored_bytes, file.path(),
                            offset + 144, "commit.index_stored_bytes",
                            std::format("<= implementation structural maximum {}",
                                        maximum_stored_bytes),
                            std::to_string(index_stored_bytes));
                !valid) {
                return std::move(valid).error();
            }
            if (index_compression == Compression::Stored) {
                if (auto valid =
                        require(index_stored_bytes == index_uncompressed_bytes,
                                file.path(), offset + 144, "commit.index_sizes",
                                "equal in stored mode",
                                std::format("({}, {})", index_stored_bytes,
                                            index_uncompressed_bytes));
                    !valid) {
                    return std::move(valid).error();
                }
            }
            if (auto valid = require(read_u32(bytes, 172) == 0, file.path(), offset + 172,
                                     "commit.commit_flags", "0",
                                     std::to_string(read_u32(bytes, 172)));
                !valid) {
                return std::move(valid).error();
            }
            auto expected_end =
                detail::checked_add(offset, COMMIT_RECORD_BYTES, file.path(), offset + 176,
                                    "commit.committed_file_end");
            if (!expected_end) {
                return std::move(expected_end).error();
            }
            if (auto valid = require(committed_end == *expected_end, file.path(), offset + 176,
                                     "commit.committed_file_end",
                                     std::format("0x{:x}", *expected_end),
                                     std::format("0x{:x}", committed_end));
                !valid) {
                return std::move(valid).error();
            }
            auto index_end = detail::checked_add(index_offset, index_stored_bytes, file.path(),
                                                 offset + 136, "commit.index_range");
            if (!index_end) {
                return std::move(index_end).error();
            }
            if (auto valid = require(*index_end <= offset, file.path(), offset + 136,
                                     "commit.index_range",
                                     std::format("end <= commit offset 0x{:x}", offset),
                                     std::format("[0x{:x},0x{:x})", index_offset, *index_end));
                !valid) {
                return std::move(valid).error();
            }
            auto expected_commit_offset =
                checked_align_up(*index_end, CHUNK_ALIGNMENT, file.path(),
                                 offset + 136, "commit.index_to_commit_padding");
            if (!expected_commit_offset) {
                return std::move(expected_commit_offset).error();
            }
            if (auto valid =
                    require(offset == *expected_commit_offset, file.path(), offset + 136,
                            "commit.index_to_commit_padding",
                            std::format("commit at first 64-byte boundary 0x{:x}",
                                        *expected_commit_offset),
                            std::format("0x{:x}", offset));
                !valid) {
                return std::move(valid).error();
            }
            if (*index_end < offset) {
                if (auto padding = validate_zero_range(
                        file, *index_end, offset - *index_end, physical_size,
                        authority_end, "commit.index_to_commit_padding",
                        CHUNK_ALIGNMENT - 1);
                    !padding) {
                    return std::move(padding).error();
                }
            }
            if (auto valid = require_zero(bytes, 224, 252, file.path(), offset,
                                          "commit.reserved");
                !valid) {
                return std::move(valid).error();
            }

            const auto kind = static_cast<CommitKind>(kind_value);
            if (superblock.role == ContainerRole::Master) {
                if (auto valid = require(kind != CommitKind::Autosave, file.path(), offset + 12,
                                         "commit.kind",
                                         "EXPLICIT, RECOVERED, or COMPACTION in master",
                                         std::to_string(kind_value));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(explicit_uuid == commit_uuid, file.path(), offset + 96,
                                         "commit.explicit_ancestor_commit_uuid",
                                         uuid_string(commit_uuid), uuid_string(explicit_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                if (kind == CommitKind::Compaction) {
                    if (auto valid = require(generation == 1, file.path(), offset + 64,
                                             "commit.compaction_generation", "1",
                                             std::to_string(generation));
                        !valid) {
                        return std::move(valid).error();
                    }
                }
            } else {
                if (auto valid = require(kind == CommitKind::Autosave, file.path(), offset + 12,
                                         "commit.kind", "AUTOSAVE in sidecar",
                                         std::to_string(kind_value));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(generation == 1, file.path(), offset + 64,
                                         "commit.sidecar_generation", "1",
                                         std::to_string(generation));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(explicit_uuid == superblock.base_explicit_commit_uuid,
                                file.path(), offset + 96,
                                "commit.explicit_ancestor_commit_uuid",
                                uuid_string(superblock.base_explicit_commit_uuid),
                                uuid_string(explicit_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(snapshot_uuid == superblock.sidecar_snapshot_uuid,
                                         file.path(), offset + 112, "commit.snapshot_uuid",
                                         uuid_string(superblock.sidecar_snapshot_uuid),
                                         uuid_string(snapshot_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
            }

            if (generation == 1) {
                if (auto valid = require(parent_uuid.is_nil(), file.path(), offset + 72,
                                         "commit.parent_commit_uuid", "null UUID",
                                         uuid_string(parent_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(parent_offset == 0, file.path(), offset + 88,
                                         "commit.parent_commit_offset", "0",
                                         std::to_string(parent_offset));
                    !valid) {
                    return std::move(valid).error();
                }
            } else {
                if (auto valid = require(!parent_uuid.is_nil(), file.path(), offset + 72,
                                         "commit.parent_commit_uuid", "non-null UUID",
                                         uuid_string(parent_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(parent_offset >= APPEND_REGION_OFFSET &&
                                    parent_offset % CHUNK_ALIGNMENT == 0 &&
                                    parent_offset < offset,
                                file.path(), offset + 88, "commit.parent_commit_offset",
                                "earlier 64-byte-aligned append offset",
                                std::format("0x{:x}", parent_offset));
                    !valid) {
                    return std::move(valid).error();
                }
            }

            ParsedCommit parsed{
                .info =
                    CommitInfo{
                        .offset = offset,
                        .kind = kind,
                        .commit_uuid = commit_uuid,
                        .generation = generation,
                        .parent_commit_uuid = parent_uuid,
                        .parent_commit_offset = parent_offset,
                        .explicit_ancestor_commit_uuid = explicit_uuid,
                        .snapshot_uuid = snapshot_uuid,
                        .wallclock_unix_ns = read_u64(bytes, 128),
                        .index_offset = index_offset,
                        .index_stored_bytes = index_stored_bytes,
                        .index_uncompressed_bytes = index_uncompressed_bytes,
                        .index_stored_crc32c = read_u32(bytes, 160),
                        .index_uncompressed_crc32c = read_u32(bytes, 164),
                        .index_compression = index_compression,
                        .committed_file_end = committed_end,
                        .min_reader_version =
                            Version{read_u16(bytes, 184), read_u16(bytes, 186)},
                        .min_safe_writer_version =
                            Version{read_u16(bytes, 188), read_u16(bytes, 190)},
                        .required_reader_capabilities = read_capabilities(bytes, 192),
                        .required_writer_capabilities = read_capabilities(bytes, 208),
                        .crc32c = read_u32(bytes, 252),
                    },
                .project_uuid = project_uuid,
                .file_uuid = file_uuid,
            };
            if (cache != nullptr) {
                cache->emplace(offset, parsed);
            }
            return parsed;
        }

        lfs::Result<std::vector<ParsedCommit>>
        validate_lineage(const detail::NativeFile& file, const std::uint64_t physical_size,
                         const SuperblockInfo& superblock, const ParsedCommit& selected,
                         CommitRecordCache* cache = nullptr) {
            std::set<std::uint64_t> seen_offsets{selected.info.offset};
            std::set<std::array<std::uint8_t, 16>> seen_uuids{
                selected.info.commit_uuid.bytes};
            std::vector<ParsedCommit> newest_to_oldest{selected};
            ParsedCommit child = selected;
            while (child.info.generation > 1) {
                const std::uint64_t parent_offset = child.info.parent_commit_offset;
                if (auto valid = require(!seen_offsets.contains(parent_offset), file.path(),
                                         child.info.offset + 88,
                                         "commit.parent_commit_offset", "acyclic lineage",
                                         std::format("cycle to 0x{:x}", parent_offset));
                    !valid) {
                    return std::move(valid).error();
                }
                seen_offsets.insert(parent_offset);
                auto parent = parse_commit(file, physical_size, superblock, parent_offset,
                                           selected.info.committed_file_end, cache);
                if (!parent) {
                    return std::move(parent).error();
                }
                if (auto valid = require(!seen_uuids.contains(parent->info.commit_uuid.bytes),
                                         file.path(), parent->info.offset + 48,
                                         "commit.commit_uuid",
                                         "unique within same-file lineage",
                                         uuid_string(parent->info.commit_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                seen_uuids.insert(parent->info.commit_uuid.bytes);
                if (auto valid = require(parent->info.commit_uuid ==
                                             child.info.parent_commit_uuid,
                                         file.path(), child.info.offset + 72,
                                         "commit.parent_commit_uuid",
                                         uuid_string(parent->info.commit_uuid),
                                         uuid_string(child.info.parent_commit_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(parent->info.generation + 1 == child.info.generation,
                                file.path(), child.info.offset + 64,
                                "commit.generation_link",
                                std::to_string(parent->info.generation + 1),
                                std::to_string(child.info.generation));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(
                        child.info.index_offset >= parent->info.committed_file_end,
                        file.path(), child.info.offset + 136,
                        "commit.generation_append_start",
                        std::format("index offset >= parent end 0x{:x}",
                                    parent->info.committed_file_end),
                        std::format("0x{:x}", child.info.index_offset));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(child.info.offset > parent->info.committed_file_end,
                                file.path(), child.info.offset,
                                "commit.generation_commit_order",
                                std::format("> parent end 0x{:x}",
                                            parent->info.committed_file_end),
                                std::format("0x{:x}", child.info.offset));
                    !valid) {
                    return std::move(valid).error();
                }
                newest_to_oldest.push_back(*parent);
                child = std::move(*parent);
            }
            std::reverse(newest_to_oldest.begin(), newest_to_oldest.end());
            return newest_to_oldest;
        }

        struct UninitializedBuffer {
            std::unique_ptr<std::byte[]> storage;
            std::size_t size = 0;

            [[nodiscard]] std::span<std::byte> bytes() noexcept {
                return {storage.get(), size};
            }
            [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
                return {storage.get(), size};
            }
        };

        lfs::Result<UninitializedBuffer>
        allocate_uninitialized_buffer(const std::size_t bytes,
                                      const std::filesystem::path& path,
                                      const std::uint64_t offset,
                                      const std::string_view field) {
            UninitializedBuffer result;
            result.size = bytes;
            if (bytes == 0) {
                return result;
            }
            try {
                result.storage = std::make_unique_for_overwrite<std::byte[]>(bytes);
            } catch (const std::bad_alloc&) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to read this project chunk.",
                    std::format("allocation of {} bytes failed", bytes), path, offset,
                    field);
            } catch (const std::length_error& error) {
                return detail::project_error(
                    lfs::ErrorCode::BoundsViolation,
                    "The project chunk cannot be represented by this build.",
                    std::format("allocation of {} bytes failed: {}", bytes,
                                error.what()),
                    path, offset, field);
            }
            return result;
        }

        lfs::Result<std::vector<std::byte>>
        allocate_zeroed_vector(const std::size_t bytes,
                               const std::filesystem::path& path,
                               const std::uint64_t offset,
                               const std::string_view field) {
            std::vector<std::byte> result;
            if (bytes == 0) {
                return result;
            }
            try {
                result.resize(bytes);
            } catch (const std::bad_alloc&) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to read this project chunk.",
                    std::format("allocation of {} bytes failed", bytes), path, offset,
                    field);
            } catch (const std::length_error& error) {
                return detail::project_error(
                    lfs::ErrorCode::BoundsViolation,
                    "The project chunk cannot be represented by this build.",
                    std::format("allocation of {} bytes failed: {}", bytes,
                                error.what()),
                    path, offset, field);
            }
            return result;
        }

        lfs::Result<UninitializedBuffer>
        allocate_uninitialized_decoded(const std::size_t bytes,
                                       const std::filesystem::path& path,
                                       const std::uint64_t offset) {
            UninitializedBuffer result;
            result.size = bytes;
            if (bytes == 0) {
                return result;
            }
            try {
                result.storage = std::make_unique_for_overwrite<std::byte[]>(bytes);
            } catch (const std::bad_alloc&) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to decode this project region.",
                    std::format("allocation of {} decoded bytes failed", bytes), path,
                    offset, "payload.framed");
            } catch (const std::length_error& error) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "The framed project region cannot be represented by this build.",
                    std::format("allocation of {} decoded bytes failed: {}", bytes,
                                error.what()),
                    path, offset, "payload.framed");
            }
            return result;
        }

        // Inverse of writer byte-plane: planes of f32-word significance bytes
        // back to interleaved little-endian words. Requires size % 4 == 0.
        void unbyte_plane_f32_words(const std::span<const std::byte> planes,
                                    const std::span<std::byte> out) {
            const std::size_t n_words = planes.size() / 4;
            const auto* src = reinterpret_cast<const std::uint8_t*>(planes.data());
            auto* dst = reinterpret_cast<std::uint8_t*>(out.data());
            for (std::size_t w = 0; w < n_words; ++w) {
                dst[w * 4 + 0] = src[0 * n_words + w];
                dst[w * 4 + 1] = src[1 * n_words + w];
                dst[w * 4 + 2] = src[2 * n_words + w];
                dst[w * 4 + 3] = src[3 * n_words + w];
            }
        }

        lfs::Result<std::vector<std::byte>>
        decompress_index_zstd(const std::filesystem::path& path, const std::uint64_t offset,
                              const std::string_view field,
                              const std::span<const std::byte> stored,
                              const std::uint64_t expected_size,
                              const std::uint64_t maximum_decoded_size) {
            const unsigned long long frame_size =
                ZSTD_getFrameContentSize(stored.data(), stored.size());
            if (frame_size == ZSTD_CONTENTSIZE_ERROR || frame_size == ZSTD_CONTENTSIZE_UNKNOWN ||
                frame_size != expected_size) {
                return format_error(
                    path, offset, std::format("{}.declared_content_size", field),
                    std::to_string(expected_size),
                    frame_size == ZSTD_CONTENTSIZE_ERROR
                        ? "invalid zstd frame"
                    : frame_size == ZSTD_CONTENTSIZE_UNKNOWN
                        ? "unknown"
                        : std::to_string(frame_size));
            }
            const std::size_t frame_bytes =
                ZSTD_findFrameCompressedSize(stored.data(), stored.size());
            if (ZSTD_isError(frame_bytes) || frame_bytes != stored.size()) {
                return format_error(
                    path, offset, std::format("{}.frame_size", field),
                    "one zstd frame with no trailing bytes",
                    ZSTD_isError(frame_bytes) ? ZSTD_getErrorName(frame_bytes)
                                              : std::to_string(frame_bytes));
            }
            if (expected_size > maximum_decoded_size ||
                expected_size > std::numeric_limits<std::size_t>::max()) {
                return format_error(
                    path, offset, field,
                    std::format("decoded size <= implementation maximum {} and addressable "
                                "by this build",
                                maximum_decoded_size),
                    std::to_string(expected_size), lfs::ErrorCode::ResourceExhausted);
            }
            std::vector<std::byte> decoded;
            try {
                decoded.resize(static_cast<std::size_t>(expected_size));
            } catch (const std::bad_alloc&) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to decode this project region.",
                    std::format("allocation of {} decoded bytes failed", expected_size),
                    path, offset, field);
            } catch (const std::length_error& error) {
                return detail::project_error(
                    lfs::ErrorCode::BoundsViolation,
                    "The decoded project region cannot be represented by this build.",
                    std::format("allocation of {} bytes failed: {}", expected_size,
                                error.what()),
                    path, offset, field);
            }
            const std::size_t result =
                ZSTD_decompress(decoded.data(), decoded.size(), stored.data(), stored.size());
            if (ZSTD_isError(result) || result != decoded.size()) {
                return format_error(
                    path, offset, field, std::format("decode to {} bytes", expected_size),
                    ZSTD_isError(result) ? ZSTD_getErrorName(result)
                                         : std::to_string(result));
            }
            return decoded;
        }

        lfs::Result<BlockCrcTable>
        parse_block_table(const detail::NativeFile& file, const std::uint64_t physical_size,
                          const CommitInfo& commit, const ChunkInfo& row,
                          const std::uint64_t table_offset) {
            auto raw_result = read_fixed<BLOCK_CRC_HEADER_BYTES>(
                file, table_offset, physical_size, commit.committed_file_end,
                "block_crc_table");
            if (!raw_result) {
                return std::move(raw_result).error();
            }
            const auto& raw = *raw_result;
            const auto bytes = byte_span(raw);
            if (auto valid = require(bytes_equal(bytes, 0, BLOCK_CRC_MAGIC), file.path(),
                                     table_offset, "block_table.magic", "LFSBCRC\\0",
                                     "different bytes");
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t expected_header_crc = crc32c(0, raw.data(), 60);
            if (auto valid = require(read_u32(bytes, 60) == expected_header_crc, file.path(),
                                     table_offset + 60, "block_table.header_crc32c",
                                     std::format("0x{:08x}", expected_header_crc),
                                     std::format("0x{:08x}", read_u32(bytes, 60)));
                !valid) {
                return std::move(valid).error();
            }
            const std::array<std::tuple<bool, std::uint64_t, std::string_view, std::string,
                                        std::string>,
                             7>
                fixed_checks = {{
                    {read_u16(bytes, 8) == 1, table_offset + 8,
                     "block_table.table_version", "1", std::to_string(read_u16(bytes, 8))},
                    {read_u16(bytes, 10) == BLOCK_CRC_HEADER_BYTES, table_offset + 10,
                     "block_table.header_bytes", "64", std::to_string(read_u16(bytes, 10))},
                    {read_u16(bytes, 12) == 4, table_offset + 12,
                     "block_table.entry_bytes", "4", std::to_string(read_u16(bytes, 12))},
                    {read_u16(bytes, 14) == 1, table_offset + 14,
                     "block_table.crc_algorithm", "1", std::to_string(read_u16(bytes, 14))},
                    {read_u32(bytes, 16) == BLOCK_CRC_BYTES, table_offset + 16,
                     "block_table.block_size", std::to_string(BLOCK_CRC_BYTES),
                     std::to_string(read_u32(bytes, 16))},
                    {read_u64(bytes, 24) == row.payload_offset, table_offset + 24,
                     "block_table.payload_offset", std::format("0x{:x}", row.payload_offset),
                     std::format("0x{:x}", read_u64(bytes, 24))},
                    {read_u64(bytes, 32) == row.stored_bytes, table_offset + 32,
                     "block_table.stored_bytes", std::to_string(row.stored_bytes),
                     std::to_string(read_u64(bytes, 32))},
                }};
            for (const auto& [condition, check_offset, field, expected, got] : fixed_checks) {
                if (auto valid =
                        require(condition, file.path(), check_offset, field, expected, got);
                    !valid) {
                    return std::move(valid).error();
                }
            }
            if (auto valid = require(read_u32(bytes, 20) == 0, file.path(),
                                     table_offset + 20, "block_table.reserved_0", "0",
                                     std::to_string(read_u32(bytes, 20)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require_zero(bytes, 52, 60, file.path(), table_offset,
                                          "block_table.reserved_1");
                !valid) {
                return std::move(valid).error();
            }
            const std::uint64_t expected_count =
                row.stored_bytes / BLOCK_CRC_BYTES +
                (row.stored_bytes % BLOCK_CRC_BYTES != 0 ? 1 : 0);
            const std::uint64_t count = read_u64(bytes, 40);
            if (auto valid = require(count == expected_count && count >= 1, file.path(),
                                     table_offset + 40, "block_table.block_count",
                                     std::to_string(expected_count), std::to_string(count));
                !valid) {
                return std::move(valid).error();
            }
            auto entry_bytes =
                detail::checked_multiply(count, 4, file.path(), table_offset + 40,
                                         "block_table.entries_size");
            if (!entry_bytes) {
                return std::move(entry_bytes).error();
            }
            if (auto valid =
                    require(*entry_bytes <= MAX_BLOCK_CRC_TABLE_BYTES, file.path(),
                            table_offset + 40, "block_table.entries_size",
                            std::format("<= implementation structural maximum {}",
                                        MAX_BLOCK_CRC_TABLE_BYTES),
                            std::to_string(*entry_bytes));
                !valid) {
                return std::move(valid).error();
            }
            auto table_end = detail::checked_add(
                table_offset + BLOCK_CRC_HEADER_BYTES, *entry_bytes, file.path(),
                table_offset + 40, "block_table.entries_range");
            if (!table_end) {
                return std::move(table_end).error();
            }
            if (auto valid =
                    require(*table_end <= row.payload_offset, file.path(),
                            table_offset + 40, "block_table.entries_range",
                            std::format("end <= payload offset 0x{:x}",
                                        row.payload_offset),
                            std::format("[0x{:x},0x{:x})",
                                        table_offset + BLOCK_CRC_HEADER_BYTES,
                                        *table_end));
                !valid) {
                return std::move(valid).error();
            }
            auto entries_raw = read_vector(file, table_offset + BLOCK_CRC_HEADER_BYTES,
                                           *entry_bytes, physical_size,
                                           commit.committed_file_end,
                                           "block_table.entries", *entry_bytes);
            if (!entries_raw) {
                return std::move(entries_raw).error();
            }
            const std::uint32_t expected_entries_crc =
                crc32c(0, entries_raw->data(), entries_raw->size());
            if (auto valid = require(read_u32(bytes, 48) == expected_entries_crc, file.path(),
                                     table_offset + 48, "block_table.entries_crc32c",
                                     std::format("0x{:08x}", expected_entries_crc),
                                     std::format("0x{:08x}", read_u32(bytes, 48)));
                !valid) {
                return std::move(valid).error();
            }
            std::vector<std::uint32_t> entries;
            if (count > entries.max_size()) {
                return format_error(
                    file.path(), table_offset + 40, "block_table.block_count",
                    std::format("<= vector maximum {}", entries.max_size()),
                    std::to_string(count), lfs::ErrorCode::BoundsViolation);
            }
            try {
                entries.reserve(static_cast<std::size_t>(count));
            } catch (const std::bad_alloc&) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to index this block CRC table.",
                    std::format("allocation of {} CRC entries failed", count),
                    file.path(), table_offset + 40, "block_table.block_count");
            } catch (const std::length_error& error) {
                return detail::project_error(
                    lfs::ErrorCode::BoundsViolation,
                    "The block CRC table cannot be represented by this build.",
                    std::format("allocation of {} entries failed: {}", count,
                                error.what()),
                    file.path(), table_offset + 40, "block_table.block_count");
            }
            for (std::uint64_t index = 0; index < count; ++index) {
                entries.push_back(read_u32(*entries_raw, static_cast<std::size_t>(index * 4)));
            }
            return BlockCrcTable{
                .offset = table_offset,
                .payload_offset = row.payload_offset,
                .stored_bytes = row.stored_bytes,
                .block_size = static_cast<std::uint32_t>(BLOCK_CRC_BYTES),
                .entries = std::move(entries),
                .entries_crc32c = read_u32(bytes, 48),
                .header_crc32c = read_u32(bytes, 60),
            };
        }

        std::array<std::uint8_t, 20> key_bytes(const ChunkKey& key) noexcept {
            std::array<std::uint8_t, 20> result{};
            std::copy(key.fourcc.bytes.begin(), key.fourcc.bytes.end(), result.begin());
            std::copy(key.instance_uuid.bytes.begin(), key.instance_uuid.bytes.end(),
                      result.begin() + 4);
            return result;
        }

        lfs::Result<ChunkInfo>
        validate_live_row(const detail::NativeFile& file, const std::uint64_t physical_size,
                          const CommitInfo& commit, ChunkInfo row,
                          const std::uint64_t row_offset) {
            const std::string prefix = std::format("index.row[{}]", row.key_string());
            if (auto valid = require(row.chunk_version >= 1, file.path(), row_offset + 4,
                                     prefix + ".chunk_version", ">= 1",
                                     std::to_string(row.chunk_version));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(row.compression == Compression::Stored ||
                                row.compression == Compression::ZstdFramed ||
                                row.compression == Compression::ByteShuffleZstdFramed,
                            file.path(), row_offset + 7, prefix + ".compression",
                            "0, 1, or 2",
                            std::to_string(static_cast<std::uint16_t>(row.compression)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require((row.flags & ~KNOWN_CHUNK_FLAGS) == 0, file.path(),
                                     row_offset + 8, prefix + ".chunk_flags",
                                     std::format("no bits outside 0x{:x}", KNOWN_CHUNK_FLAGS),
                                     std::format("0x{:x}", row.flags));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(row.source_generation >= 1 &&
                                         row.source_generation <= commit.generation,
                                     file.path(), row_offset + 64,
                                     prefix + ".source_generation",
                                     std::format("[1,{}]", commit.generation),
                                     std::to_string(row.source_generation));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(row.header_offset >= APPEND_REGION_OFFSET &&
                                         row.header_offset % CHUNK_ALIGNMENT == 0,
                                     file.path(), row_offset + 32,
                                     prefix + ".header_offset",
                                     "64-byte-aligned offset >= 65536",
                                     std::format("0x{:x}", row.header_offset));
                !valid) {
                return std::move(valid).error();
            }
            auto minimum_payload = detail::checked_add(
                row.header_offset, CHUNK_HEADER_BYTES, file.path(), row_offset + 40,
                prefix + ".payload_offset");
            if (!minimum_payload) {
                return std::move(minimum_payload).error();
            }
            if (auto valid = require(row.payload_offset >= *minimum_payload, file.path(),
                                     row_offset + 40, prefix + ".payload_offset",
                                     std::format(">= 0x{:x}", *minimum_payload),
                                     std::format("0x{:x}", row.payload_offset));
                !valid) {
                return std::move(valid).error();
            }
            const std::uint64_t alignment =
                (row.flags & TENSOR_PAYLOAD) != 0 ? TENSOR_PAYLOAD_ALIGNMENT : CHUNK_ALIGNMENT;
            if (auto valid = require(row.payload_offset % alignment == 0, file.path(),
                                     row_offset + 40, prefix + ".payload_offset",
                                     std::format("{}-byte aligned", alignment),
                                     std::format("0x{:x}", row.payload_offset));
                !valid) {
                return std::move(valid).error();
            }
            auto payload_end =
                detail::checked_add(row.payload_offset, row.stored_bytes, file.path(),
                                    row_offset + 40, prefix + ".payload_range");
            if (!payload_end) {
                return std::move(payload_end).error();
            }
            if (auto valid = require(*payload_end <= commit.index_offset, file.path(),
                                     row_offset + 40, prefix + ".payload_range",
                                     std::format("end <= index offset 0x{:x}",
                                                 commit.index_offset),
                                     std::format("[0x{:x},0x{:x})", row.payload_offset,
                                                 *payload_end));
                !valid) {
                return std::move(valid).error();
            }
            if (row.compression == Compression::Stored) {
                if (auto valid = require(row.stored_bytes == row.uncompressed_bytes,
                                         file.path(), row_offset + 48,
                                         prefix + ".stored_uncompressed_sizes",
                                         "equal in stored mode",
                                         std::format("({}, {})", row.stored_bytes,
                                                     row.uncompressed_bytes));
                    !valid) {
                    return std::move(valid).error();
                }
            }

            auto header_result =
                read_fixed<CHUNK_HEADER_BYTES>(file, row.header_offset, physical_size,
                                               commit.committed_file_end,
                                               prefix + ".chunk_header");
            if (!header_result) {
                return std::move(header_result).error();
            }
            const auto& header = *header_result;
            const auto bytes = byte_span(header);
            const std::uint32_t expected_header_crc = crc32c(0, header.data(), 60);
            if (auto valid = require(read_u32(bytes, 60) == expected_header_crc, file.path(),
                                     row.header_offset + 60,
                                     prefix + ".chunk_header.crc32c",
                                     std::format("0x{:08x}", expected_header_crc),
                                     std::format("0x{:08x}", read_u32(bytes, 60)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u32(bytes, 60) == row.header_crc32c, file.path(),
                                     row_offset + 76, prefix + ".header_crc32c_echo",
                                     std::format("0x{:08x}", read_u32(bytes, 60)),
                                     std::format("0x{:08x}", row.header_crc32c));
                !valid) {
                return std::move(valid).error();
            }

            if (auto valid = require(read_fourcc(bytes, 0) == row.key.fourcc, file.path(),
                                     row.header_offset, prefix + ".chunk_header.fourcc",
                                     row.key.fourcc.to_string(),
                                     read_fourcc(bytes, 0).to_string());
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u16(bytes, 4) == row.chunk_version, file.path(),
                                     row.header_offset + 4,
                                     prefix + ".chunk_header.chunk_version",
                                     std::to_string(row.chunk_version),
                                     std::to_string(read_u16(bytes, 4)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u16(bytes, 6) == CHUNK_HEADER_BYTES, file.path(),
                                     row.header_offset + 6,
                                     prefix + ".chunk_header.header_bytes", "64",
                                     std::to_string(read_u16(bytes, 6)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_uuid(bytes, 8) == row.key.instance_uuid,
                                     file.path(), row.header_offset + 8,
                                     prefix + ".chunk_header.instance_uuid",
                                     uuid_string(row.key.instance_uuid),
                                     uuid_string(read_uuid(bytes, 8)));
                !valid) {
                return std::move(valid).error();
            }
            const std::array<std::tuple<bool, std::uint64_t, std::string, std::string,
                                        std::string>,
                             5>
                echoes = {{
                    {read_u32(bytes, 24) == row.flags, row.header_offset + 24,
                     prefix + ".chunk_header.chunk_flags", std::to_string(row.flags),
                     std::to_string(read_u32(bytes, 24))},
                    {read_u16(bytes, 28) == static_cast<std::uint16_t>(row.compression),
                     row.header_offset + 28, prefix + ".chunk_header.compression",
                     std::to_string(static_cast<std::uint16_t>(row.compression)),
                     std::to_string(read_u16(bytes, 28))},
                    {read_u64(bytes, 32) == row.stored_bytes, row.header_offset + 32,
                     prefix + ".chunk_header.stored_bytes",
                     std::to_string(row.stored_bytes), std::to_string(read_u64(bytes, 32))},
                    {read_u64(bytes, 40) == row.uncompressed_bytes, row.header_offset + 40,
                     prefix + ".chunk_header.uncompressed_bytes",
                     std::to_string(row.uncompressed_bytes),
                     std::to_string(read_u64(bytes, 40))},
                    {read_u32(bytes, 56) == row.payload_crc32c, row.header_offset + 56,
                     prefix + ".chunk_header.payload_crc32c",
                     std::format("0x{:08x}", row.payload_crc32c),
                     std::format("0x{:08x}", read_u32(bytes, 56))},
                }};
            for (const auto& [condition, check_offset, field, expected, got] : echoes) {
                if (auto valid =
                        require(condition, file.path(), check_offset, field, expected, got);
                    !valid) {
                    return std::move(valid).error();
                }
            }

            const std::uint16_t block_kind = read_u16(bytes, 30);
            const std::uint64_t table_offset = read_u64(bytes, 48);
            std::uint64_t padding_start = row.header_offset + CHUNK_HEADER_BYTES;
            if ((row.flags & HAS_BLOCK_CRCS) != 0) {
                if (auto valid = require(block_kind == 1, file.path(),
                                         row.header_offset + 30,
                                         prefix + ".chunk_header.block_crc_kind", "1",
                                         std::to_string(block_kind));
                    !valid) {
                    return std::move(valid).error();
                }
                const std::uint64_t expected_table_offset =
                    row.header_offset + CHUNK_HEADER_BYTES;
                if (auto valid =
                        require(table_offset == expected_table_offset, file.path(),
                                row.header_offset + 48,
                                prefix + ".chunk_header.block_crc_table_offset",
                                std::format("0x{:x}", expected_table_offset),
                                std::format("0x{:x}", table_offset));
                    !valid) {
                    return std::move(valid).error();
                }
                auto table =
                    parse_block_table(file, physical_size, commit, row, table_offset);
                if (!table) {
                    return std::move(table).error();
                }
                auto entry_bytes = detail::checked_multiply(
                    table->entries.size(), 4, file.path(), table->offset + 40,
                    prefix + ".block_table.range");
                if (!entry_bytes) {
                    return std::move(entry_bytes).error();
                }
                auto table_end = detail::checked_add(
                    table->offset + BLOCK_CRC_HEADER_BYTES, *entry_bytes, file.path(),
                    table->offset, prefix + ".block_table.range");
                if (!table_end) {
                    return std::move(table_end).error();
                }
                if (auto valid = require(*table_end <= row.payload_offset, file.path(),
                                         table->offset, prefix + ".block_table.range",
                                         std::format("end <= payload offset 0x{:x}",
                                                     row.payload_offset),
                                         std::format("[0x{:x},0x{:x})", table->offset,
                                                     *table_end));
                    !valid) {
                    return std::move(valid).error();
                }
                padding_start = *table_end;
                row.block_crc_table = std::move(*table);
            } else {
                if (auto valid = require(block_kind == 0, file.path(),
                                         row.header_offset + 30,
                                         prefix + ".chunk_header.block_crc_kind", "0",
                                         std::to_string(block_kind));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid = require(table_offset == 0, file.path(),
                                         row.header_offset + 48,
                                         prefix + ".chunk_header.block_crc_table_offset",
                                         "0", std::format("0x{:x}", table_offset));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(row.stored_bytes < BLOCK_CRC_REQUIRED_AT, file.path(),
                                row_offset + 48, prefix + ".block_crc_requirement",
                                std::format("stored bytes < {}", BLOCK_CRC_REQUIRED_AT),
                                std::to_string(row.stored_bytes));
                    !valid) {
                    return std::move(valid).error();
                }
            }
            auto expected_payload_offset =
                checked_align_up(padding_start, alignment, file.path(),
                                 row_offset + 40,
                                 prefix + ".payload_alignment_padding");
            if (!expected_payload_offset) {
                return std::move(expected_payload_offset).error();
            }
            if (auto valid =
                    require(row.payload_offset == *expected_payload_offset,
                            file.path(), row_offset + 40,
                            prefix + ".payload_offset",
                            std::format("first {}-byte boundary 0x{:x}", alignment,
                                        *expected_payload_offset),
                            std::format("0x{:x}", row.payload_offset));
                !valid) {
                return std::move(valid).error();
            }
            if (padding_start < row.payload_offset) {
                if (auto padding = validate_zero_range(
                        file, padding_start, row.payload_offset - padding_start,
                        physical_size, commit.committed_file_end,
                        prefix + ".payload_alignment_padding", alignment - 1);
                    !padding) {
                    return std::move(padding).error();
                }
            }
            return row;
        }

        struct ParsedIndex {
            std::vector<ChunkInfo> chunks;
            std::uint32_t flags = 0;
        };

        lfs::Result<ParsedIndex>
        parse_index(const detail::NativeFile& file, const std::uint64_t physical_size,
                    const SuperblockInfo& superblock, const ParsedCommit& parsed_commit,
                    const std::vector<ParsedCommit>& lineage) {
            const CommitInfo& commit = parsed_commit.info;
            auto stored =
                read_vector(file, commit.index_offset, commit.index_stored_bytes,
                            physical_size, commit.committed_file_end, "index.stored",
                            maximum_stored_index_bytes(commit.index_compression));
            if (!stored) {
                return std::move(stored).error();
            }
            const std::uint32_t stored_crc = crc32c(0, stored->data(), stored->size());
            if (auto valid =
                    require(stored_crc == commit.index_stored_crc32c, file.path(),
                            commit.offset + 160, "commit.index_stored_crc32c",
                            std::format("0x{:08x}", stored_crc),
                            std::format("0x{:08x}", commit.index_stored_crc32c));
                !valid) {
                return std::move(valid).error();
            }

            std::vector<std::byte> decoded;
            if (commit.index_compression == Compression::Stored) {
                if (auto valid =
                        require(commit.index_stored_bytes ==
                                    commit.index_uncompressed_bytes,
                                file.path(), commit.offset + 144,
                                "commit.index_sizes", "equal in stored mode",
                                std::format("({}, {})", commit.index_stored_bytes,
                                            commit.index_uncompressed_bytes));
                    !valid) {
                    return std::move(valid).error();
                }
                decoded = *stored;
            } else {
                auto decompressed =
                    decompress_index_zstd(file.path(), commit.index_offset, "index.zstd",
                                          *stored, commit.index_uncompressed_bytes,
                                          MAX_INDEX_UNCOMPRESSED_BYTES);
                if (!decompressed) {
                    return std::move(decompressed).error();
                }
                decoded = std::move(*decompressed);
            }
            if (auto valid = require(decoded.size() == commit.index_uncompressed_bytes,
                                     file.path(), commit.offset + 152,
                                     "commit.index_uncompressed_bytes",
                                     std::to_string(commit.index_uncompressed_bytes),
                                     std::to_string(decoded.size()));
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t decoded_crc = crc32c(0, decoded.data(), decoded.size());
            if (auto valid =
                    require(decoded_crc == commit.index_uncompressed_crc32c, file.path(),
                            commit.offset + 164, "commit.index_uncompressed_crc32c",
                            std::format("0x{:08x}", decoded_crc),
                            std::format("0x{:08x}", commit.index_uncompressed_crc32c));
                !valid) {
                return std::move(valid).error();
            }
            const std::span<const std::byte> bytes(decoded);
            if (auto valid = require(bytes_equal(bytes, 0, INDEX_MAGIC), file.path(),
                                     commit.index_offset, "index.magic", "LFSINDEX",
                                     "different bytes");
                !valid) {
                return std::move(valid).error();
            }
            const std::array<std::tuple<bool, std::uint64_t, std::string_view, std::string,
                                        std::string, lfs::ErrorCode>,
                             4>
                fixed_checks = {{
                    {read_u16(bytes, 8) == 1, commit.index_offset + 8,
                     "index.index_version", "1", std::to_string(read_u16(bytes, 8)),
                     lfs::ErrorCode::Unsupported},
                    {read_u16(bytes, 10) == INDEX_HEADER_BYTES, commit.index_offset + 10,
                     "index.header_bytes", "64", std::to_string(read_u16(bytes, 10)),
                     lfs::ErrorCode::DataLoss},
                    {read_u16(bytes, 12) == INDEX_ROW_BYTES, commit.index_offset + 12,
                     "index.row_bytes", "96", std::to_string(read_u16(bytes, 12)),
                     lfs::ErrorCode::DataLoss},
                    {read_u16(bytes, 14) == 1, commit.index_offset + 14,
                     "index.sort_order", "1", std::to_string(read_u16(bytes, 14)),
                     lfs::ErrorCode::Unsupported},
                }};
            for (const auto& [condition, check_offset, field, expected, got, code] :
                 fixed_checks) {
                if (auto valid =
                        require(condition, file.path(), check_offset, field, expected, got, code);
                    !valid) {
                    return std::move(valid).error();
                }
            }
            const std::uint64_t row_count = read_u64(bytes, 16);
            auto rows_bytes = detail::checked_multiply(
                row_count, INDEX_ROW_BYTES, file.path(), commit.index_offset + 16,
                "index.row_count");
            if (!rows_bytes) {
                return std::move(rows_bytes).error();
            }
            auto expected_size =
                detail::checked_add(INDEX_HEADER_BYTES, *rows_bytes, file.path(),
                                    commit.index_offset + 16, "index.decoded_size_from_row_count");
            if (!expected_size) {
                return std::move(expected_size).error();
            }
            if (auto valid = require(decoded.size() == *expected_size, file.path(),
                                     commit.index_offset + 16,
                                     "index.decoded_size_from_row_count",
                                     std::to_string(*expected_size),
                                     std::to_string(decoded.size()));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_u64(bytes, 24) == commit.generation, file.path(),
                                     commit.index_offset + 24, "index.generation",
                                     std::to_string(commit.generation),
                                     std::to_string(read_u64(bytes, 24)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(read_uuid(bytes, 32) == commit.commit_uuid, file.path(),
                                     commit.index_offset + 32, "index.commit_uuid",
                                     uuid_string(commit.commit_uuid),
                                     uuid_string(read_uuid(bytes, 32)));
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t index_flags = read_u32(bytes, 48);
            if (auto valid = require((index_flags & ~0x3u) == 0, file.path(),
                                     commit.index_offset + 48, "index.index_flags",
                                     "only bits 0 and 1",
                                     std::format("0x{:x}", index_flags));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require_zero(bytes, 52, 64, file.path(), commit.index_offset,
                                          "index.reserved");
                !valid) {
                return std::move(valid).error();
            }

            std::vector<ChunkInfo> rows;
            if (row_count > rows.max_size()) {
                return format_error(
                    file.path(), commit.index_offset + 16, "index.row_count",
                    std::format("<= vector maximum {}", rows.max_size()),
                    std::to_string(row_count), lfs::ErrorCode::BoundsViolation);
            }
            try {
                rows.reserve(static_cast<std::size_t>(row_count));
            } catch (const std::bad_alloc&) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to index this project.",
                    std::format("allocation of {} index rows failed", row_count),
                    file.path(), commit.index_offset + 16, "index.row_count");
            } catch (const std::length_error& error) {
                return detail::project_error(
                    lfs::ErrorCode::BoundsViolation,
                    "The project index cannot be represented by this build.",
                    std::format("allocation of {} rows failed: {}", row_count,
                                error.what()),
                    file.path(), commit.index_offset + 16, "index.row_count");
            }
            std::optional<std::array<std::uint8_t, 20>> previous_key;
            bool has_tombstone = false;
            bool has_base_reference = false;
            for (std::uint64_t row_index = 0; row_index < row_count; ++row_index) {
                const std::size_t relative = static_cast<std::size_t>(
                    INDEX_HEADER_BYTES + row_index * INDEX_ROW_BYTES);
                const std::uint64_t absolute = commit.index_offset + relative;
                const auto raw = bytes.subspan(relative, INDEX_ROW_BYTES);
                const Fourcc fourcc = read_fourcc(raw, 0);
                if (auto valid = require(fourcc.valid(), file.path(), absolute,
                                         std::format("index.rows[{}].fourcc", row_index),
                                         "[A-Z0-9]{4}", fourcc.to_string());
                    !valid) {
                    return std::move(valid).error();
                }
                const lfs::core::Uuid instance_uuid = read_uuid(raw, 16);
                if (auto valid =
                        require(!instance_uuid.is_nil(), file.path(), absolute + 16,
                                std::format("index.rows[{}].instance_uuid", row_index),
                                "non-null UUID", uuid_string(instance_uuid));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require_zero(raw, 12, 16, file.path(), absolute,
                                     std::format("index.rows[{}].reserved_0", row_index));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require_zero(raw, 80, 96, file.path(), absolute,
                                     std::format("index.rows[{}].reserved_1", row_index));
                    !valid) {
                    return std::move(valid).error();
                }
                const std::uint8_t row_kind_value = read_u8(raw, 6);
                if (auto valid =
                        require(row_kind_value <= 2, file.path(), absolute + 6,
                                std::format("index.rows[{}].row_kind", row_index),
                                "0, 1, or 2", std::to_string(row_kind_value));
                    !valid) {
                    return std::move(valid).error();
                }
                const std::uint8_t compression_value = read_u8(raw, 7);
                ChunkInfo row{
                    .key = ChunkKey{fourcc, instance_uuid},
                    .chunk_version = read_u16(raw, 4),
                    .row_kind = static_cast<RowKind>(row_kind_value),
                    .compression = static_cast<Compression>(compression_value),
                    .flags = read_u32(raw, 8),
                    .header_offset = read_u64(raw, 32),
                    .payload_offset = read_u64(raw, 40),
                    .stored_bytes = read_u64(raw, 48),
                    .uncompressed_bytes = read_u64(raw, 56),
                    .source_generation = read_u64(raw, 64),
                    .payload_crc32c = read_u32(raw, 72),
                    .header_crc32c = read_u32(raw, 76),
                    .block_crc_table = std::nullopt,
                };
                const auto canonical_key = key_bytes(row.key);
                if (previous_key.has_value()) {
                    if (auto valid =
                            require(canonical_key > *previous_key, file.path(), absolute,
                                    std::format("index.rows[{}].canonical_key_order",
                                                row_index),
                                    "strict unsigned bytewise increase",
                                    row.key_string());
                        !valid) {
                        return std::move(valid).error();
                    }
                }
                previous_key = canonical_key;

                const std::string prefix =
                    std::format("index.row[{}]", row.key_string());
                if (row.row_kind == RowKind::Live) {
                    auto validated =
                        validate_live_row(file, physical_size, commit, row, absolute);
                    if (!validated) {
                        return std::move(validated).error();
                    }
                    row = std::move(*validated);
                } else if (row.row_kind == RowKind::Tombstone) {
                    has_tombstone = true;
                    const bool sentinel =
                        row.chunk_version == 0 && compression_value == 0 && row.flags == 0 &&
                        row.header_offset == 0 && row.payload_offset == 0 &&
                        row.stored_bytes == 0 && row.uncompressed_bytes == 0 &&
                        row.payload_crc32c == 0 && row.header_crc32c == 0;
                    if (auto valid =
                            require(sentinel, file.path(), absolute + 4,
                                    prefix + ".tombstone_encoding",
                                    "all sentinel fields zero", "nonzero sentinel field");
                        !valid) {
                        return std::move(valid).error();
                    }
                    if (auto valid =
                            require(row.source_generation >= 1 &&
                                        row.source_generation <= commit.generation,
                                    file.path(), absolute + 64,
                                    prefix + ".source_generation",
                                    std::format("[1,{}]", commit.generation),
                                    std::to_string(row.source_generation));
                        !valid) {
                        return std::move(valid).error();
                    }
                } else {
                    has_base_reference = true;
                    if (auto valid =
                            require(superblock.role == ContainerRole::AutosaveSidecar,
                                    file.path(), absolute + 6, prefix + ".row_kind",
                                    "base reference only in sidecar",
                                    "base reference in master");
                        !valid) {
                        return std::move(valid).error();
                    }
                    if (auto valid = require(row.chunk_version >= 1, file.path(),
                                             absolute + 4, prefix + ".chunk_version",
                                             ">= 1", std::to_string(row.chunk_version));
                        !valid) {
                        return std::move(valid).error();
                    }
                    if (auto valid =
                            require(compression_value <= 2, file.path(), absolute + 7,
                                    prefix + ".compression", "0, 1, or 2",
                                    std::to_string(compression_value));
                        !valid) {
                        return std::move(valid).error();
                    }
                    if (auto valid =
                            require((row.flags & ~KNOWN_CHUNK_FLAGS) == 0, file.path(),
                                    absolute + 8, prefix + ".chunk_flags",
                                    std::format("no bits outside 0x{:x}",
                                                KNOWN_CHUNK_FLAGS),
                                    std::format("0x{:x}", row.flags));
                        !valid) {
                        return std::move(valid).error();
                    }
                    if (auto valid =
                            require(row.header_offset == 0 && row.payload_offset == 0,
                                    file.path(), absolute + 32,
                                    prefix + ".base_ref_offsets", "both zero",
                                    std::format("({}, {})", row.header_offset,
                                                row.payload_offset));
                        !valid) {
                        return std::move(valid).error();
                    }
                    if (auto valid = require(row.source_generation >= 1, file.path(),
                                             absolute + 64,
                                             prefix + ".source_generation", ">= 1",
                                             std::to_string(row.source_generation));
                        !valid) {
                        return std::move(valid).error();
                    }
                    if (row.compression == Compression::Stored) {
                        if (auto valid =
                                require(row.stored_bytes == row.uncompressed_bytes,
                                        file.path(), absolute + 48,
                                        prefix + ".stored_uncompressed_sizes",
                                        "equal in stored mode",
                                        std::format("({}, {})", row.stored_bytes,
                                                    row.uncompressed_bytes));
                            !valid) {
                            return std::move(valid).error();
                        }
                    }
                }
                rows.push_back(std::move(row));
            }

            const std::uint32_t derived_flags =
                (has_tombstone ? 1u : 0u) | (has_base_reference ? 2u : 0u);
            if (auto valid = require(index_flags == derived_flags, file.path(),
                                     commit.index_offset + 48, "index.index_flags",
                                     std::format("derived value 0x{:x}", derived_flags),
                                     std::format("0x{:x}", index_flags));
                !valid) {
                return std::move(valid).error();
            }

            struct Span {
                std::uint64_t begin;
                std::uint64_t end;
                const ChunkInfo* row;
            };
            std::vector<Span> spans;
            for (const auto& row : rows) {
                if (row.row_kind == RowKind::Live) {
                    spans.push_back(
                        Span{row.header_offset, row.payload_offset + row.stored_bytes, &row});
                }
            }
            std::sort(spans.begin(), spans.end(),
                      [](const Span& lhs, const Span& rhs) {
                          return std::tie(lhs.begin, lhs.end) <
                                 std::tie(rhs.begin, rhs.end);
                      });
            for (std::size_t index = 1; index < spans.size(); ++index) {
                const Span& previous = spans[index - 1];
                const Span& current = spans[index];
                if (auto valid =
                        require(current.begin >= previous.end, file.path(),
                                current.row->header_offset,
                                std::format("index.row[{}].occupied_span",
                                            current.row->key_string()),
                                std::format("start >= prior end 0x{:x} ({})",
                                            previous.end, previous.row->key_string()),
                                std::format("[0x{:x},0x{:x})", current.begin,
                                            current.end));
                    !valid) {
                    return std::move(valid).error();
                }
            }

            std::map<std::uint64_t, const ParsedCommit*> lineage_by_generation;
            for (const auto& ancestor : lineage) {
                lineage_by_generation.emplace(ancestor.info.generation, &ancestor);
            }
            for (const auto& row : rows) {
                if (row.row_kind != RowKind::Live) {
                    continue;
                }
                const auto source = lineage_by_generation.find(row.source_generation);
                if (auto valid =
                        require(source != lineage_by_generation.end(), file.path(),
                                row.header_offset,
                                std::format("index.row[{}].source_generation",
                                            row.key_string()),
                                "generation present in selected lineage",
                                std::to_string(row.source_generation));
                    !valid) {
                    return std::move(valid).error();
                }
                const std::uint64_t source_start =
                    row.source_generation == 1
                        ? APPEND_REGION_OFFSET
                        : lineage_by_generation.at(row.source_generation - 1)
                              ->info.committed_file_end;
                const std::uint64_t source_end = source->second->info.index_offset;
                const std::uint64_t row_end = row.payload_offset + row.stored_bytes;
                if (auto valid =
                        require(row.header_offset >= source_start && row_end <= source_end,
                                file.path(), row.header_offset,
                                std::format("index.row[{}].source_generation_span",
                                            row.key_string()),
                                std::format("subset of generation {} payload region "
                                            "[0x{:x},0x{:x})",
                                            row.source_generation, source_start, source_end),
                                std::format("[0x{:x},0x{:x})", row.header_offset,
                                            row_end));
                    !valid) {
                    return std::move(valid).error();
                }
            }

            CapabilitySet derived_capabilities;
            if (commit.index_compression == Compression::ZstdFramed) {
                derived_capabilities.set(INDEX_ZSTD_V1);
            }
            if (std::any_of(rows.begin(), rows.end(), [](const ChunkInfo& row) {
                    return row.row_kind != RowKind::Tombstone &&
                           row.compression == Compression::ZstdFramed;
                })) {
                derived_capabilities.set(CHUNK_ZSTD_V1);
            }
            if (std::any_of(rows.begin(), rows.end(), [](const ChunkInfo& row) {
                    return row.row_kind != RowKind::Tombstone &&
                           row.compression == Compression::ByteShuffleZstdFramed;
                })) {
                derived_capabilities.set(CHUNK_BYTESHUFFLE_ZSTD_V1);
            }
            if (std::any_of(rows.begin(), rows.end(), [](const ChunkInfo& row) {
                    return row.block_crc_table.has_value();
                })) {
                derived_capabilities.set(BLOCK_CRC32C_V1);
            }
            if (has_tombstone) {
                derived_capabilities.set(INDEX_TOMBSTONES_V1);
            }
            if (superblock.role == ContainerRole::AutosaveSidecar ||
                has_base_reference) {
                derived_capabilities.set(SIDECAR_OVERLAY_V1);
            }
            if (!commit.required_reader_capabilities.contains_all(
                    derived_capabilities)) {
                return format_error(
                    file.path(), commit.offset + 192,
                    "commit.required_reader_capabilities",
                    std::format("include feature bits {}",
                                derived_capabilities.to_hex()),
                    commit.required_reader_capabilities.to_hex());
            }
            return ParsedIndex{.chunks = std::move(rows), .flags = index_flags};
        }

        std::optional<lfs::Error>
        compatibility_error(const std::filesystem::path& path, const ParsedCommit& commit,
                            const ReaderOptions& options) {
            if (commit.info.min_reader_version > options.reader_version) {
                return format_error(
                    path, commit.info.offset + 184, "commit.min_reader_version",
                    std::format("<= {}.{}", options.reader_version.major,
                                options.reader_version.minor),
                    std::format("{}.{}", commit.info.min_reader_version.major,
                                commit.info.min_reader_version.minor),
                    lfs::ErrorCode::Unsupported);
            }
            const CapabilitySet missing =
                commit.info.required_reader_capabilities.missing_from(
                    options.reader_capabilities);
            if (!missing.empty()) {
                return format_error(
                    path, commit.info.offset + 192,
                    "commit.required_reader_capabilities",
                    std::format("subset of {}", options.reader_capabilities.to_hex()),
                    std::format("unsupported bits {}", missing.to_hex()),
                    lfs::ErrorCode::Unsupported);
            }
            return std::nullopt;
        }

        lfs::Result<ParsedHead>
        parse_head(const detail::NativeFile& file, const std::uint64_t physical_size,
                   const SuperblockInfo& superblock, const ReaderOptions& options,
                   const std::uint32_t slot_id,
                   const std::array<std::byte, HEAD_SLOT_BYTES>& raw,
                   CommitRecordCache* cache = nullptr) {
            const std::uint64_t offset = HEAD_SLOT_OFFSETS[slot_id];
            const auto bytes = byte_span(raw);
            if (auto valid = require(bytes_equal(bytes, 0, HEAD_MAGIC), file.path(), offset,
                                     std::format("head[{}].magic", slot_id),
                                     "LFSHEAD\\0", "different bytes");
                !valid) {
                return std::move(valid).error();
            }
            const std::uint32_t expected_head_crc = crc32c(0, raw.data(), 4092);
            if (auto valid =
                    require(read_u32(bytes, 4092) == expected_head_crc, file.path(),
                            offset + 4092, std::format("head[{}].crc32c", slot_id),
                            std::format("0x{:08x}", expected_head_crc),
                            std::format("0x{:08x}", read_u32(bytes, 4092)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(read_u32(bytes, 8) == slot_id, file.path(), offset + 8,
                            std::format("head[{}].slot_id", slot_id),
                            std::to_string(slot_id), std::to_string(read_u32(bytes, 8)));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(read_u32(bytes, 12) == HEAD_SLOT_BYTES, file.path(), offset + 12,
                            std::format("head[{}].head_bytes", slot_id), "4096",
                            std::to_string(read_u32(bytes, 12)));
                !valid) {
                return std::move(valid).error();
            }
            const std::uint64_t sequence = read_u64(bytes, 16);
            const std::uint64_t generation = read_u64(bytes, 24);
            const lfs::core::Uuid project_uuid = read_uuid(bytes, 32);
            const lfs::core::Uuid file_uuid = read_uuid(bytes, 48);
            const lfs::core::Uuid commit_uuid = read_uuid(bytes, 64);
            const std::uint64_t commit_offset = read_u64(bytes, 80);
            const std::uint64_t commit_bytes = read_u64(bytes, 88);
            const std::uint64_t committed_end = read_u64(bytes, 96);
            const std::uint32_t commit_crc_echo = read_u32(bytes, 104);
            const std::uint64_t preview_offset = read_u64(bytes, 112);
            const std::uint32_t preview_bytes = read_u32(bytes, 120);
            const std::uint32_t preview_format = read_u32(bytes, 124);

            if (auto valid = require(sequence >= 1, file.path(), offset + 16,
                                     std::format("head[{}].head_sequence", slot_id),
                                     ">= 1", std::to_string(sequence));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(generation >= 1, file.path(), offset + 24,
                                     std::format("head[{}].generation", slot_id),
                                     ">= 1", std::to_string(generation));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(project_uuid == superblock.project_uuid, file.path(), offset + 32,
                            std::format("head[{}].project_uuid", slot_id),
                            uuid_string(superblock.project_uuid), uuid_string(project_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(file_uuid == superblock.file_uuid, file.path(), offset + 48,
                            std::format("head[{}].file_uuid", slot_id),
                            uuid_string(superblock.file_uuid), uuid_string(file_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(!commit_uuid.is_nil(), file.path(), offset + 64,
                                     std::format("head[{}].commit_uuid", slot_id),
                                     "non-null UUID", uuid_string(commit_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(commit_offset >= APPEND_REGION_OFFSET &&
                                commit_offset % CHUNK_ALIGNMENT == 0,
                            file.path(), offset + 80,
                            std::format("head[{}].commit_offset", slot_id),
                            "64-byte-aligned offset >= 65536",
                            std::format("0x{:x}", commit_offset));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(commit_bytes == COMMIT_RECORD_BYTES, file.path(), offset + 88,
                            std::format("head[{}].commit_bytes", slot_id), "256",
                            std::to_string(commit_bytes));
                !valid) {
                return std::move(valid).error();
            }
            auto expected_end = detail::checked_add(commit_offset, COMMIT_RECORD_BYTES,
                                                    file.path(), offset + 96,
                                                    "head.committed_file_end");
            if (!expected_end) {
                return std::move(expected_end).error();
            }
            if (auto valid =
                    require(committed_end == *expected_end, file.path(), offset + 96,
                            std::format("head[{}].committed_file_end", slot_id),
                            std::format("0x{:x}", *expected_end),
                            std::format("0x{:x}", committed_end));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(committed_end <= physical_size, file.path(), offset + 96,
                            std::format("head[{}].committed_file_end", slot_id),
                            std::format("<= physical size 0x{:x}", physical_size),
                            std::format("0x{:x}", committed_end));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(read_u32(bytes, 108) == 0, file.path(), offset + 108,
                            std::format("head[{}].head_flags", slot_id), "0",
                            std::to_string(read_u32(bytes, 108)));
                !valid) {
                return std::move(valid).error();
            }
            std::optional<PreviewLocator> preview;
            if (preview_offset == 0) {
                if (auto valid =
                        require(preview_bytes == 0, file.path(), offset + 120,
                                std::format("head[{}].preview_bytes", slot_id),
                                "0 when preview_offset is zero",
                                std::to_string(preview_bytes));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(preview_format == 0, file.path(), offset + 124,
                                std::format("head[{}].preview_format", slot_id),
                                "0 when preview_offset is zero",
                                std::to_string(preview_format));
                    !valid) {
                    return std::move(valid).error();
                }
            } else {
                if (auto valid =
                        require(preview_bytes >= 1 &&
                                    preview_bytes <= MAX_PREVIEW_BYTES,
                                file.path(), offset + 120,
                                std::format("head[{}].preview_bytes", slot_id),
                                std::format("between 1 and {}", MAX_PREVIEW_BYTES),
                                std::to_string(preview_bytes));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(preview_format ==
                                    static_cast<std::uint32_t>(PreviewFormat::Png),
                                file.path(), offset + 124,
                                std::format("head[{}].preview_format", slot_id),
                                "1 (PNG)", std::to_string(preview_format));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(preview_offset % CHUNK_ALIGNMENT == 0, file.path(),
                                offset + 112,
                                std::format("head[{}].preview_offset", slot_id),
                                "64-byte aligned",
                                std::format("0x{:x}", preview_offset));
                    !valid) {
                    return std::move(valid).error();
                }
                auto preview_end = detail::checked_add(
                    preview_offset, preview_bytes, file.path(), offset + 112,
                    std::format("head[{}].preview_range", slot_id));
                if (!preview_end) {
                    return std::move(preview_end).error();
                }
                if (auto valid =
                        require(*preview_end <= committed_end, file.path(),
                                offset + 112,
                                std::format("head[{}].preview_range", slot_id),
                                std::format("end <= committed_file_end 0x{:x}",
                                            committed_end),
                                std::format("[0x{:x}, 0x{:x})", preview_offset,
                                            *preview_end));
                    !valid) {
                    return std::move(valid).error();
                }
                preview = PreviewLocator{
                    .offset = preview_offset,
                    .bytes = preview_bytes,
                    .format = PreviewFormat::Png,
                };
            }
            if (auto valid = require_zero(bytes, 128, 4092, file.path(), offset,
                                          std::format("head[{}].reserved", slot_id));
                !valid) {
                return std::move(valid).error();
            }

            auto commit =
                parse_commit(file, physical_size, superblock, commit_offset,
                             committed_end, cache);
            if (!commit) {
                return std::move(commit).error();
            }
            if (auto valid =
                    require(commit->info.commit_uuid == commit_uuid, file.path(), offset + 64,
                            std::format("head[{}].commit_uuid", slot_id),
                            uuid_string(commit->info.commit_uuid), uuid_string(commit_uuid));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(commit->info.generation == generation, file.path(), offset + 24,
                            std::format("head[{}].generation", slot_id),
                            std::to_string(commit->info.generation),
                            std::to_string(generation));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid = require(commit->info.committed_file_end == committed_end,
                                     file.path(), offset + 96,
                                     std::format("head[{}].committed_file_end_echo",
                                                 slot_id),
                                     std::format("0x{:x}",
                                                 commit->info.committed_file_end),
                                     std::format("0x{:x}", committed_end));
                !valid) {
                return std::move(valid).error();
            }
            if (auto valid =
                    require(commit->info.crc32c == commit_crc_echo, file.path(),
                            offset + 104,
                            std::format("head[{}].commit_crc32c_echo", slot_id),
                            std::format("0x{:08x}", commit->info.crc32c),
                            std::format("0x{:08x}", commit_crc_echo));
                !valid) {
                return std::move(valid).error();
            }
            auto lineage =
                validate_lineage(file, physical_size, superblock, *commit, cache);
            if (!lineage) {
                return std::move(lineage).error();
            }
            auto index = parse_index(file, physical_size, superblock, *commit, *lineage);
            if (!index) {
                return std::move(index).error();
            }
            if (preview.has_value()) {
                const auto matching_row = std::find_if(
                    index->chunks.begin(), index->chunks.end(),
                    [&](const ChunkInfo& row) {
                        return row.row_kind == RowKind::Live &&
                               row.key.fourcc == FOURCC_THMB &&
                               row.payload_offset == preview->offset &&
                               row.stored_bytes == preview->bytes;
                    });
                if (auto valid =
                        require(matching_row != index->chunks.end(), file.path(),
                                offset + 112,
                                std::format("head[{}].preview_locator", slot_id),
                                "exact stored span of a live THMB index row",
                                std::format("offset=0x{:x}, bytes={}",
                                            preview->offset, preview->bytes));
                    !valid) {
                    return std::move(valid).error();
                }
                if (auto valid =
                        require(matching_row->compression == Compression::Stored,
                                file.path(), offset + 112,
                                std::format("head[{}].preview_compression", slot_id),
                                "stored (none)",
                                std::to_string(static_cast<std::uint16_t>(
                                    matching_row->compression)));
                    !valid) {
                    return std::move(valid).error();
                }
            }

            ParsedHead result{
                .info =
                    HeadInfo{
                        .slot_id = slot_id,
                        .head_sequence = sequence,
                        .generation = generation,
                        .commit_uuid = commit_uuid,
                        .commit_offset = commit_offset,
                        .committed_file_end = committed_end,
                        .commit_crc32c_echo = commit_crc_echo,
                        .preview = preview,
                        .head_crc32c = read_u32(bytes, 4092),
                    },
                .commit = *commit,
                .lineage = std::move(*lineage),
                .chunks = std::move(index->chunks),
                .index_flags = index->flags,
                .compatibility_error = std::nullopt,
            };
            result.compatibility_error =
                compatibility_error(file.path(), result.commit, options);
            return result;
        }

        std::string error_detail(const lfs::Error& error) {
            return std::string(error.detail());
        }

        lfs::Error terminal_error(const std::filesystem::path& path, const OpenState state,
                                  const std::string& detail_text,
                                  const std::uint64_t offset = HEAD_SLOT_OFFSETS[0],
                                  const std::string_view field = "heads.authority") {
            const lfs::ErrorCode code =
                state == OpenState::UnsupportedNewer ? lfs::ErrorCode::Unsupported
                : state == OpenState::HardFail       ? lfs::ErrorCode::DataLoss
                                                     : lfs::ErrorCode::FailedPrecondition;
            const std::string user_message =
                state == OpenState::UnsupportedNewer
                    ? "This project requires a newer LichtFeld version."
                : state == OpenState::RepairOnly
                    ? "This project requires explicit repair."
                    : "The project has contradictory or corrupt authority metadata.";
            return detail::project_error(code, user_message, detail_text, path, offset, field);
        }

        lfs::Result<ParseOutcome>
        parse_path(const std::filesystem::path& path, const ReaderOptions& options) {
            auto file_result = detail::NativeFile::open_read(path);
            if (!file_result) {
                return std::move(file_result).error();
            }
            const auto file = *file_result;
            auto size_result = file->size();
            if (!size_result) {
                return std::move(size_result).error();
            }
            const std::uint64_t physical_size = *size_result;
            auto superblock = parse_superblock(*file, physical_size);
            if (!superblock) {
                ParseOutcome outcome;
                outcome.state = OpenState::HardFail;
                outcome.error = std::move(superblock).error();
                return outcome;
            }

            std::array<HeadAttempt, 2> attempts;
            CommitRecordCache commit_cache;
            for (std::uint32_t slot_id = 0; slot_id < attempts.size(); ++slot_id) {
                HeadAttempt& attempt = attempts[slot_id];
                attempt.slot_id = slot_id;
                auto raw = read_fixed<HEAD_SLOT_BYTES>(
                    *file, HEAD_SLOT_OFFSETS[slot_id], physical_size, physical_size,
                    std::format("head[{}]", slot_id));
                if (!raw) {
                    attempt.error = std::move(raw).error();
                    continue;
                }
                if (all_zero(byte_span(*raw))) {
                    attempt.blank = true;
                    continue;
                }
                attempt.sequence_hint = read_u64(byte_span(*raw), 16);
                auto head =
                    parse_head(*file, physical_size, *superblock, options, slot_id, *raw,
                               &commit_cache);
                if (!head) {
                    attempt.error = std::move(head).error();
                    continue;
                }
                attempt.head = std::move(*head);
            }

            std::vector<std::size_t> structural;
            for (std::size_t index = 0; index < attempts.size(); ++index) {
                if (attempts[index].head.has_value()) {
                    structural.push_back(index);
                }
            }
            if (structural.empty()) {
                std::string failures;
                for (const auto& attempt : attempts) {
                    if (!failures.empty()) {
                        failures += "; ";
                    }
                    failures += attempt.blank
                                    ? "blank"
                                : attempt.error.has_value()
                                    ? error_detail(*attempt.error)
                                    : "invalid";
                }
                ParseOutcome outcome;
                outcome.state = OpenState::RepairOnly;
                outcome.error = terminal_error(
                    path, OpenState::RepairOnly,
                    std::format("expected at least one fully valid head, got {}", failures));
                return outcome;
            }

            std::vector<std::string> warnings;
            std::size_t selected_index = structural.front();
            if (structural.size() == 2) {
                ParsedHead& first = *attempts[structural[0]].head;
                ParsedHead& second = *attempts[structural[1]].head;

                const ParsedHead* direct_parent = nullptr;
                const ParsedHead* direct_child = nullptr;
                if (first.commit.info.generation + 1 == second.commit.info.generation &&
                    second.commit.info.parent_commit_uuid == first.info.commit_uuid &&
                    second.commit.info.parent_commit_offset == first.info.commit_offset) {
                    direct_parent = &first;
                    direct_child = &second;
                } else if (second.commit.info.generation + 1 ==
                               first.commit.info.generation &&
                           first.commit.info.parent_commit_uuid == second.info.commit_uuid &&
                           first.commit.info.parent_commit_offset ==
                               second.info.commit_offset) {
                    direct_parent = &second;
                    direct_child = &first;
                }
                if (direct_parent != nullptr &&
                    direct_child->info.head_sequence <=
                        direct_parent->info.head_sequence) {
                    ParseOutcome outcome;
                    outcome.state = OpenState::HardFail;
                    outcome.error = terminal_error(
                        path, OpenState::HardFail,
                        std::format("expected child head_sequence > {}, got {}",
                                    direct_parent->info.head_sequence,
                                    direct_child->info.head_sequence),
                        HEAD_SLOT_OFFSETS[direct_child->info.slot_id] + 16,
                        "heads.head_sequence_regression");
                    return outcome;
                }

                if (first.info.head_sequence == second.info.head_sequence) {
                    if (first.info.commit_uuid != second.info.commit_uuid) {
                        ParseOutcome outcome;
                        outcome.state = OpenState::HardFail;
                        outcome.error = terminal_error(
                            path, OpenState::HardFail,
                            std::format("equal sequence {} has different commit UUIDs {} and {}",
                                        first.info.head_sequence,
                                        uuid_string(first.info.commit_uuid),
                                        uuid_string(second.info.commit_uuid)),
                            HEAD_SLOT_OFFSETS[0] + 16, "heads.equal_sequence");
                        return outcome;
                    }
                    if (first.info.commit_crc32c_echo !=
                        second.info.commit_crc32c_echo) {
                        ParseOutcome outcome;
                        outcome.state = OpenState::HardFail;
                        outcome.error = terminal_error(
                            path, OpenState::HardFail,
                            "equal-sequence slots have different commit CRC echoes",
                            HEAD_SLOT_OFFSETS[0] + 104,
                            "heads.equal_sequence_commit_crc");
                        return outcome;
                    }
                    const bool same_authority =
                        first.info.generation == second.info.generation &&
                        first.info.commit_offset == second.info.commit_offset &&
                        first.info.committed_file_end ==
                            second.info.committed_file_end;
                    if (!same_authority) {
                        ParseOutcome outcome;
                        outcome.state = OpenState::HardFail;
                        outcome.error = terminal_error(
                            path, OpenState::HardFail,
                            "equal-sequence slots have different authority tuples",
                            HEAD_SLOT_OFFSETS[0] + 24,
                            "heads.equal_sequence_authority_tuple");
                        return outcome;
                    }
                    selected_index = structural[0];
                    warnings.push_back(std::format(
                        "duplicate-slot write accepted: slots A/B both publish sequence {}, "
                        "commit {}",
                        first.info.head_sequence, uuid_string(first.info.commit_uuid)));
                } else {
                    selected_index =
                        first.info.head_sequence > second.info.head_sequence
                            ? structural[0]
                            : structural[1];
                }
            }

            ParsedHead& selected = *attempts[selected_index].head;
            for (const auto& attempt : attempts) {
                if (!attempt.head.has_value() && !attempt.blank &&
                    attempt.error.has_value() &&
                    attempt.error->code() == lfs::ErrorCode::Unsupported &&
                    (!attempt.sequence_hint.has_value() ||
                     *attempt.sequence_hint >= selected.info.head_sequence)) {
                    ParseOutcome outcome;
                    outcome.state = OpenState::HardFail;
                    outcome.error = terminal_error(
                        path, OpenState::HardFail,
                        std::format("newer structural metadata is unsupported; refusing "
                                    "rollback: {}",
                                    error_detail(*attempt.error)),
                        HEAD_SLOT_OFFSETS[attempt.slot_id],
                        "heads.newer_unsupported_head");
                    return outcome;
                }
            }

            bool has_supported_structural = false;
            for (const std::size_t index : structural) {
                if (!attempts[index].head->compatibility_error.has_value()) {
                    has_supported_structural = true;
                }
            }
            if (selected.compatibility_error.has_value() &&
                has_supported_structural) {
                ParseOutcome outcome;
                outcome.state = OpenState::HardFail;
                outcome.error = terminal_error(
                    path, OpenState::HardFail,
                    std::format("greater-sequence authority is unsupported while an older "
                                "head is supported: {}",
                                error_detail(*selected.compatibility_error)),
                    selected.commit.info.offset +
                        (selected.commit.info.min_reader_version >
                                 options.reader_version
                             ? 184
                             : 192),
                    "heads.newer_unsupported_head");
                return outcome;
            }

            for (const auto& attempt : attempts) {
                if (!attempt.head.has_value() && !attempt.blank &&
                    attempt.error.has_value()) {
                    warnings.push_back(std::format(
                        "recovery warning: rejected nonblank head slot {} at 0x{:x}: {}",
                        attempt.slot_id, HEAD_SLOT_OFFSETS[attempt.slot_id],
                        error_detail(*attempt.error)));
                } else if (attempt.head.has_value() &&
                           attempt.head->compatibility_error.has_value() &&
                           attempt.slot_id != selected.info.slot_id) {
                    warnings.push_back(std::format(
                        "compatibility warning: non-selected head slot {} requires a newer "
                        "reader: {}",
                        attempt.slot_id,
                        error_detail(*attempt.head->compatibility_error)));
                }
            }

            auto state = std::make_shared<ReaderState>();
            state->path = path;
            state->file = file;
            state->physical_size = physical_size;
            state->superblock = *superblock;
            state->selected = selected;
            state->warnings = std::move(warnings);
            state->options = options;
            state->open_state = selected.compatibility_error.has_value()
                                    ? OpenState::UnsupportedNewer
                                    : OpenState::Open;

            ParseOutcome outcome;
            outcome.state = state->open_state;
            outcome.reader = std::move(state);
            if (outcome.state == OpenState::UnsupportedNewer) {
                outcome.error = *selected.compatibility_error;
            }
            return outcome;
        }

    } // namespace

    struct ProjectReader::Impl {
        explicit Impl(std::shared_ptr<ReaderState> state_in)
            : state(std::move(state_in)) {}

        std::shared_ptr<ReaderState> state;
    };

    std::string Fourcc::to_string() const {
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

    std::optional<Fourcc> Fourcc::from_string(const std::string_view text) {
        if (text.size() != 4) {
            return std::nullopt;
        }
        Fourcc result;
        for (std::size_t index = 0; index < result.bytes.size(); ++index) {
            result.bytes[index] =
                static_cast<std::uint8_t>(static_cast<unsigned char>(text[index]));
        }
        return result.valid() ? std::optional<Fourcc>{result} : std::nullopt;
    }

    bool ChunkKeyLess::operator()(const ChunkKey& lhs, const ChunkKey& rhs) const noexcept {
        return key_bytes(lhs) < key_bytes(rhs);
    }

    CapabilitySet::CapabilitySet(std::array<std::uint8_t, 16> bytes) noexcept
        : bytes_(bytes) {}

    bool CapabilitySet::contains(const std::uint8_t bit) const noexcept {
        assert(bit < 128);
        return (bytes_[bit / 8] & (1u << (bit % 8))) != 0;
    }

    void CapabilitySet::set(const std::uint8_t bit, const bool enabled) noexcept {
        assert(bit < 128);
        const auto mask = static_cast<std::uint8_t>(1u << (bit % 8));
        if (enabled) {
            bytes_[bit / 8] |= mask;
        } else {
            bytes_[bit / 8] &= static_cast<std::uint8_t>(~mask);
        }
    }

    bool CapabilitySet::contains_all(const CapabilitySet& required) const noexcept {
        for (std::size_t index = 0; index < bytes_.size(); ++index) {
            if ((required.bytes_[index] & ~bytes_[index]) != 0) {
                return false;
            }
        }
        return true;
    }

    CapabilitySet CapabilitySet::missing_from(const CapabilitySet& supported) const noexcept {
        CapabilitySet missing;
        for (std::size_t index = 0; index < bytes_.size(); ++index) {
            missing.bytes_[index] =
                static_cast<std::uint8_t>(bytes_[index] & ~supported.bytes_[index]);
        }
        return missing;
    }

    bool CapabilitySet::empty() const noexcept {
        return std::all_of(bytes_.begin(), bytes_.end(),
                           [](const std::uint8_t byte) { return byte == 0; });
    }

    const std::array<std::uint8_t, 16>& CapabilitySet::bytes() const noexcept {
        return bytes_;
    }

    std::string CapabilitySet::to_hex() const {
        std::string result = "0x";
        result.reserve(34);
        constexpr std::string_view HEX = "0123456789abcdef";
        for (auto iterator = bytes_.rbegin(); iterator != bytes_.rend(); ++iterator) {
            result.push_back(HEX[*iterator >> 4]);
            result.push_back(HEX[*iterator & 0x0f]);
        }
        return result;
    }

    CapabilitySet& CapabilitySet::operator|=(const CapabilitySet& other) noexcept {
        for (std::size_t index = 0; index < bytes_.size(); ++index) {
            bytes_[index] |= other.bytes_[index];
        }
        return *this;
    }

    CapabilitySet supported_reader_capabilities() {
        CapabilitySet result;
        for (std::uint8_t bit = INDEX_ZSTD_V1;
             bit <= CHUNK_BYTESHUFFLE_ZSTD_V1; ++bit) {
            result.set(bit);
        }
        return result;
    }

    CapabilitySet supported_writer_capabilities() {
        return supported_reader_capabilities();
    }

    std::string ChunkInfo::key_string() const {
        return std::format("{}:{}", key.fourcc.to_string(),
                           key.instance_uuid.to_string());
    }

    ProjectReader::ProjectReader(std::shared_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    ProjectReader::ProjectReader(ProjectReader&&) noexcept = default;
    ProjectReader& ProjectReader::operator=(ProjectReader&&) noexcept = default;
    ProjectReader::~ProjectReader() = default;

    lfs::Result<ProjectReader>
    ProjectReader::open(const std::filesystem::path& path, const ReaderOptions& options) {
        auto parsed = parse_path(path, options);
        if (!parsed) {
            return std::move(parsed).error();
        }
        if (parsed->state == OpenState::Open ||
            (parsed->state == OpenState::UnsupportedNewer &&
             options.allow_unsupported_inspection)) {
            assert(parsed->reader != nullptr);
            return ProjectReader(
                std::make_shared<Impl>(std::move(parsed->reader)));
        }
        assert(parsed->error.has_value());
        return std::move(*parsed->error);
    }

    OpenClassification
    ProjectReader::classify(const std::filesystem::path& path,
                            const ReaderOptions& options) {
        auto parsed = parse_path(path, options);
        if (!parsed) {
            return OpenClassification{
                .state = OpenState::HardFail,
                .diagnostic = std::string(parsed.error().detail()),
            };
        }
        OpenClassification result{
            .state = parsed->state,
            .generation =
                parsed->reader != nullptr
                    ? parsed->reader->selected.commit.info.generation
                    : 0,
            .diagnostic = {},
        };
        if (parsed->error.has_value()) {
            result.diagnostic = error_detail(*parsed->error);
        } else if (parsed->reader != nullptr &&
                   !parsed->reader->warnings.empty()) {
            for (const auto& warning : parsed->reader->warnings) {
                if (!result.diagnostic.empty()) {
                    result.diagnostic += "; ";
                }
                result.diagnostic += warning;
            }
        }
        return result;
    }

    const std::filesystem::path& ProjectReader::path() const noexcept {
        return impl_->state->path;
    }

    std::uint64_t ProjectReader::physical_file_size() const noexcept {
        return impl_->state->physical_size;
    }

    OpenState ProjectReader::open_state() const noexcept {
        return impl_->state->open_state;
    }

    const SuperblockInfo& ProjectReader::superblock() const noexcept {
        return impl_->state->superblock;
    }

    const HeadInfo& ProjectReader::selected_head() const noexcept {
        return impl_->state->selected.info;
    }

    const CommitInfo& ProjectReader::commit() const noexcept {
        return impl_->state->selected.commit.info;
    }

    const std::vector<ChunkInfo>& ProjectReader::chunks() const noexcept {
        return impl_->state->selected.chunks;
    }

    const std::vector<std::string>& ProjectReader::warnings() const noexcept {
        return impl_->state->warnings;
    }

    const std::optional<PreviewLocator>& ProjectReader::preview() const noexcept {
        return impl_->state->selected.info.preview;
    }

    const ReaderOptions& ProjectReader::reader_options() const noexcept {
        return impl_->state->options;
    }

    WriteCompatibility ProjectReader::write_compatibility() const {
        const ReaderState& state = *impl_->state;
        const CommitInfo& selected_commit = state.selected.commit.info;
        WriteCompatibility result{
            .safe = true,
            .reasons = {},
        };
        if (selected_commit.min_safe_writer_version >
            state.options.writer_version) {
            result.safe = false;
            result.reasons.push_back(std::format(
                "min_safe_writer_version {}.{} exceeds declared writer {}.{}",
                selected_commit.min_safe_writer_version.major,
                selected_commit.min_safe_writer_version.minor,
                state.options.writer_version.major,
                state.options.writer_version.minor));
        }
        const CapabilitySet missing =
            selected_commit.required_writer_capabilities.missing_from(
                state.options.writer_capabilities);
        if (!missing.empty()) {
            result.safe = false;
            result.reasons.push_back(std::format(
                "required_writer_capabilities contains unsupported bits {}",
                missing.to_hex()));
        }
        return result;
    }

    const ChunkInfo* ProjectReader::find(const ChunkKey& key) const noexcept {
        const auto iterator = std::lower_bound(
            chunks().begin(), chunks().end(), key,
            [](const ChunkInfo& chunk, const ChunkKey& search) {
                return ChunkKeyLess{}(chunk.key, search);
            });
        return iterator != chunks().end() && iterator->key == key ? &*iterator : nullptr;
    }

    const ChunkInfo*
    ProjectReader::find(const Fourcc fourcc,
                        const lfs::core::Uuid& instance_uuid) const noexcept {
        return find(ChunkKey{fourcc, instance_uuid});
    }

    namespace {

        lfs::Result<const ChunkInfo*>
        resolve_chunk(const ReaderState& state, const ChunkInfo& requested,
                      const bool require_live = true) {
            if (state.open_state != OpenState::Open) {
                return detail::project_error(
                    lfs::ErrorCode::Unsupported,
                    "Chunk semantics are unavailable for this newer project.",
                    "payload access is forbidden in unsupported-newer inspection mode",
                    state.path, state.selected.commit.info.offset,
                    "reader.semantic_access");
            }
            const auto iterator = std::lower_bound(
                state.selected.chunks.begin(), state.selected.chunks.end(),
                requested.key,
                [](const ChunkInfo& chunk, const ChunkKey& search) {
                    return ChunkKeyLess{}(chunk.key, search);
                });
            if (iterator == state.selected.chunks.end() ||
                iterator->key != requested.key) {
                return detail::project_error(
                    lfs::ErrorCode::NotFound, "The requested project chunk was not found.",
                    std::format("{} is not in the pinned generation", requested.key_string()),
                    state.path);
            }
            const bool same_row =
                iterator->row_kind == requested.row_kind &&
                iterator->chunk_version == requested.chunk_version &&
                iterator->compression == requested.compression &&
                iterator->flags == requested.flags &&
                iterator->header_offset == requested.header_offset &&
                iterator->payload_offset == requested.payload_offset &&
                iterator->stored_bytes == requested.stored_bytes &&
                iterator->uncompressed_bytes == requested.uncompressed_bytes &&
                iterator->source_generation == requested.source_generation &&
                iterator->payload_crc32c == requested.payload_crc32c &&
                iterator->header_crc32c == requested.header_crc32c;
            if (!same_row) {
                return detail::project_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "The requested chunk handle is not from this pinned generation.",
                    std::format("{} metadata differs from the selected index",
                                requested.key_string()),
                    state.path);
            }
            if (require_live && iterator->row_kind != RowKind::Live) {
                return detail::project_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "The requested project row has no local payload.",
                    std::format("{} is not a live row", requested.key_string()),
                    state.path);
            }
            return &*iterator;
        }

        lfs::Result<void> verify_block_range(const ReaderState& state,
                                             const ChunkInfo& row,
                                             const std::uint64_t relative_offset,
                                             const std::uint64_t length) {
            if (!row.block_crc_table.has_value()) {
                return {};
            }
            if (length == 0) {
                return {};
            }
            const BlockCrcTable& table = *row.block_crc_table;
            const std::uint64_t first = relative_offset / table.block_size;
            const std::uint64_t last =
                (relative_offset + length - 1) / table.block_size;
            for (std::uint64_t block_index = first; block_index <= last; ++block_index) {
                const std::uint64_t block_offset =
                    block_index * table.block_size;
                const std::uint64_t block_bytes =
                    std::min<std::uint64_t>(table.block_size,
                                            row.stored_bytes - block_offset);
                auto bytes = read_vector(*state.file, row.payload_offset + block_offset,
                                         block_bytes, state.physical_size,
                                         state.selected.commit.info.committed_file_end,
                                         "payload.block", BLOCK_CRC_BYTES);
                if (!bytes) {
                    return status_failure(std::move(bytes).error());
                }
                const std::uint32_t actual =
                    crc32c(0, bytes->data(), bytes->size());
                const std::uint32_t expected =
                    table.entries[static_cast<std::size_t>(block_index)];
                if (actual != expected) {
                    return status_failure(format_error(
                        state.path, row.payload_offset + block_offset,
                        std::format("payload[{}].block[{}].crc32c",
                                    row.key_string(), block_index),
                        std::format("0x{:08x}", expected),
                        std::format("0x{:08x}", actual)));
                }
            }
            return {};
        }

        // Stored-only integrity: one bounded pass over the payload range.
        // Matches verify_chunk's payload CRC mismatch identity (field string
        // and expected/actual hex). Does not decode framed records.
        // payload_bytes_read is incremented before each block read, same as
        // verify_chunk's stored CRC loop (the subsequent framed materialize
        // in verify_chunk does not increment this counter).
        lfs::Result<void> verify_stored_payload_crc32c(const ReaderState& state,
                                                       const ChunkInfo& row) {
            if (row.stored_bytes == 0) {
                if (0u != row.payload_crc32c) {
                    return status_failure(format_error(
                        state.path, row.payload_offset,
                        std::format("payload[{}].crc32c", row.key_string()),
                        std::format("0x{:08x}", row.payload_crc32c),
                        "0x00000000"));
                }
                return {};
            }
            auto payload_end = detail::checked_add(
                row.payload_offset, row.stored_bytes, state.path,
                row.payload_offset, "payload.verify");
            if (!payload_end) {
                return status_failure(std::move(payload_end).error());
            }
            const std::uint64_t limit = std::min(
                state.physical_size,
                state.selected.commit.info.committed_file_end);
            if (*payload_end > limit) {
                return status_failure(format_error(
                    state.path, row.payload_offset, "payload.verify",
                    std::format("range within authority end 0x{:x}", limit),
                    std::format("[0x{:x},0x{:x}) size {}", row.payload_offset,
                                *payload_end, row.stored_bytes),
                    lfs::ErrorCode::BoundsViolation));
            }

            const auto max_workers =
                std::max(1u, std::thread::hardware_concurrency());
            const std::uint64_t range_count = std::min<std::uint64_t>(
                max_workers,
                std::max<std::uint64_t>(
                    1, (row.stored_bytes + BLOCK_CRC_BYTES - 1) /
                           BLOCK_CRC_BYTES));
            struct Range {
                std::uint64_t offset = 0;
                std::uint64_t size = 0;
            };
            std::vector<Range> ranges(static_cast<std::size_t>(range_count));
            const std::uint64_t base = row.stored_bytes / range_count;
            const std::uint64_t extra = row.stored_bytes % range_count;
            std::uint64_t cursor = 0;
            for (std::uint64_t i = 0; i < range_count; ++i) {
                const std::uint64_t size = base + (i < extra ? 1 : 0);
                ranges[static_cast<std::size_t>(i)] = Range{
                    .offset = cursor,
                    .size = size,
                };
                cursor += size;
            }

            std::vector<std::uint32_t> partial(ranges.size(), 0);
            std::mutex mutex;
            std::optional<lfs::Error> error;
            const auto crc_range = [&](const std::size_t index) {
                const Range& range = ranges[index];
                if (range.size == 0) {
                    return;
                }
                const std::size_t buffer_size = static_cast<std::size_t>(
                    std::min<std::uint64_t>(BLOCK_CRC_BYTES, range.size));
                std::vector<std::byte> buffer(buffer_size);
                std::uint32_t crc = 0;
                std::uint64_t remaining = range.size;
                std::uint64_t relative = 0;
                while (remaining > 0) {
                    const std::uint64_t count = std::min<std::uint64_t>(
                        buffer.size(), remaining);
                    if (state.options.payload_bytes_read) {
                        state.options.payload_bytes_read->fetch_add(
                            count, std::memory_order_relaxed);
                    }
                    auto dest = std::span<std::byte>(buffer.data(),
                                                     static_cast<std::size_t>(count));
                    if (auto read = state.file->read_exact(
                            row.payload_offset + range.offset + relative, dest);
                        !read) {
                        std::scoped_lock lock(mutex);
                        if (!error) {
                            error = std::move(read).error();
                        }
                        return;
                    }
                    crc = crc32c(crc, dest.data(), dest.size());
                    relative += count;
                    remaining -= count;
                }
                partial[index] = crc;
            };

            if (ranges.size() == 1) {
                try {
                    crc_range(0);
                } catch (const std::bad_alloc&) {
                    return status_failure(detail::project_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "There is not enough memory to verify this project region.",
                        "payload CRC buffer allocation failed", state.path,
                        row.payload_offset, "payload.verify"));
                } catch (const std::exception& exception) {
                    // LFS-CENSUS-OK(empty-catch): propagate the CRC exception as a project error
                    return status_failure(detail::project_error(
                        lfs::ErrorCode::Internal,
                        "The project payload CRC could not be verified.",
                        exception.what(), state.path, row.payload_offset,
                        "payload.verify"));
                }
            } else {
                std::vector<std::jthread> workers;
                try {
                    workers.reserve(ranges.size());
                    for (std::size_t i = 0; i < ranges.size(); ++i) {
                        workers.emplace_back([&, i] {
                            try {
                                crc_range(i);
                            } catch (const std::bad_alloc&) {
                                std::scoped_lock lock(mutex);
                                if (!error) {
                                    error = detail::project_error(
                                        lfs::ErrorCode::ResourceExhausted,
                                        "There is not enough memory to verify this project region.",
                                        "payload CRC buffer allocation failed",
                                        state.path, row.payload_offset,
                                        "payload.verify");
                                }
                            } catch (const std::exception& exception) {
                                // LFS-CENSUS-OK(empty-catch): record the CRC worker exception as the first payload error
                                std::scoped_lock lock(mutex);
                                if (!error) {
                                    error = detail::project_error(
                                        lfs::ErrorCode::Internal,
                                        "The project payload CRC could not be verified.",
                                        exception.what(), state.path,
                                        row.payload_offset, "payload.verify");
                                }
                            } catch (...) {
                                // LFS-CENSUS-OK(empty-catch): record an unknown CRC worker exception as the first payload error
                                std::scoped_lock lock(mutex);
                                if (!error) {
                                    error = detail::project_error(
                                        lfs::ErrorCode::Internal,
                                        "The project payload CRC could not be verified.",
                                        "unknown worker exception", state.path,
                                        row.payload_offset, "payload.verify");
                                }
                            }
                        });
                    }
                } catch (const std::exception& exception) {
                    // LFS-CENSUS-OK(empty-catch): propagate CRC worker construction failure as a project error
                    workers.clear();
                    return status_failure(detail::project_error(
                        lfs::ErrorCode::Internal,
                        "The project payload CRC could not be verified.",
                        exception.what(), state.path, row.payload_offset,
                        "payload.verify"));
                }
                workers.clear();
            }
            if (error) {
                return status_failure(std::move(*error));
            }

            std::uint32_t running_crc = 0;
            for (std::size_t i = 0; i < ranges.size(); ++i) {
                running_crc = crc32c_combine(running_crc, partial[i],
                                             ranges[i].size);
            }
            if (running_crc != row.payload_crc32c) {
                return status_failure(format_error(
                    state.path, row.payload_offset,
                    std::format("payload[{}].crc32c", row.key_string()),
                    std::format("0x{:08x}", row.payload_crc32c),
                    std::format("0x{:08x}", running_crc)));
            }
            return {};
        }

        lfs::Result<void>
        require_supported_payload_access(const ProjectReader& reader) {
            if (reader.open_state() == OpenState::Open) {
                return {};
            }
            return status_failure(detail::project_error(
                lfs::ErrorCode::Unsupported,
                "Payload access is unavailable while inspecting a newer project.",
                "allow_unsupported_inspection exposes only validated container metadata; "
                "verify, extract, map, stream, and clean-proof operations are refused",
                reader.path(), reader.commit().offset,
                "commit.read_compatibility"));
        }

        lfs::Result<void> decompress_framed_zstd(
            const std::filesystem::path& path, const std::uint64_t offset,
            const std::span<const std::byte> stored, const std::uint64_t expected_size,
            const std::uint64_t maximum_decoded_size,
            const std::span<std::byte> decoded,
            const std::function<void(std::size_t, std::size_t)>& progress = {}) {
            if (stored.size() < detail::FRAMED_HEADER_BYTES ||
                !std::equal(detail::FRAMED_MAGIC.begin(), detail::FRAMED_MAGIC.end(), stored.begin())) {
                return status_failure(format_error(path, offset, "payload.framed", "framed header", "invalid magic"));
            }
            const auto count = read_u32(stored, 12);
            if (count > (std::numeric_limits<std::size_t>::max() -
                         detail::FRAMED_HEADER_BYTES) /
                            detail::FRAMED_RECORD_BYTES) {
                return status_failure(format_error(path, offset, "payload.framed.header",
                                                   "representable record table", std::to_string(count)));
            }
            const auto table = detail::FRAMED_HEADER_BYTES +
                               static_cast<std::size_t>(count) * detail::FRAMED_RECORD_BYTES;
            if (read_u16(stored, 8) != detail::FRAMED_VERSION || read_u16(stored, 10) != 0 ||
                count == 0 || table > stored.size()) {
                return status_failure(format_error(path, offset, "payload.framed.header", "valid header", "invalid header"));
            }
            if (expected_size > maximum_decoded_size || expected_size > std::numeric_limits<std::size_t>::max()) {
                return status_failure(detail::project_error(lfs::ErrorCode::ResourceExhausted,
                                                            "The project payload is too large to decode.",
                                                            std::format("decoded size {} exceeds implementation maximum {}",
                                                                        expected_size, maximum_decoded_size),
                                                            path, offset, "payload.framed"));
            }
            if (count > expected_size || stored.size() - table < count) {
                return status_failure(format_error(path, offset, "payload.framed.records",
                                                   "at least one byte per record", "invalid record count"));
            }
            struct Record {
                std::size_t so, sb, uo, ub;
            };
            std::vector<Record> records;
            try {
                records.reserve(count);
            } catch (const std::bad_alloc&) {
                return status_failure(detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to decode this project region.",
                    std::format("allocation of {} record descriptors failed", count),
                    path, offset, "payload.framed.records"));
            } catch (const std::length_error& error) {
                return status_failure(detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "The framed project region cannot be represented by this build.",
                    std::format("allocation of {} record descriptors failed: {}", count,
                                error.what()),
                    path, offset, "payload.framed.records"));
            }
            std::size_t so = table, uo = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto at = detail::FRAMED_HEADER_BYTES + i * detail::FRAMED_RECORD_BYTES;
                const auto sb64 = read_u64(stored, at), ub64 = read_u64(stored, at + 8);
                if (sb64 == 0 || ub64 == 0 || sb64 > stored.size() - so ||
                    ub64 > expected_size - uo || sb64 > std::numeric_limits<std::size_t>::max() ||
                    ub64 > std::numeric_limits<std::size_t>::max()) {
                    return status_failure(format_error(path, offset + at, "payload.framed.record", "valid record range", "invalid range"));
                }
                records.push_back({so, static_cast<std::size_t>(sb64), uo, static_cast<std::size_t>(ub64)});
                so += static_cast<std::size_t>(sb64);
                uo += static_cast<std::size_t>(ub64);
            }
            if (so != stored.size() || uo != expected_size) {
                return status_failure(format_error(path, offset, "payload.framed.records", "exact stored and decoded sizes", "size mismatch"));
            }
            std::atomic<std::size_t> next{0}, done{0};
            std::mutex mutex;
            std::optional<lfs::Error> error;
            const auto publish_worker_error = [&](const lfs::ErrorCode code,
                                                  const std::string_view message,
                                                  const std::string_view detail) noexcept {
                try {
                    std::scoped_lock lock(mutex);
                    if (!error)
                        error = detail::project_error(code, std::string(message),
                                                      std::string(detail), path, offset,
                                                      "payload.framed.worker");
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): preserve the first worker error if formatting fails.
                }
            };
            const auto decode_one = [&](const std::size_t i) {
                const auto r = records[i];
                const auto frame = stored.subspan(r.so, r.sb);
                const auto size = ZSTD_getFrameContentSize(frame.data(), frame.size());
                const auto result = size == r.ub ? ZSTD_decompress(decoded.data() + r.uo, r.ub, frame.data(), r.sb) : ZSTD_CONTENTSIZE_ERROR;
                if (size != r.ub || ZSTD_isError(result) || result != r.ub) {
                    std::scoped_lock lock(mutex);
                    if (!error)
                        error = format_error(path, offset + r.so, "payload.framed.record", "valid zstd frame", "decode failed");
                    return;
                }
                if (progress)
                    progress(done.fetch_add(1) + 1, records.size());
            };
            if (records.size() == 1) {
                try {
                    decode_one(0);
                } catch (const std::bad_alloc&) {
                    publish_worker_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "There is not enough memory to decode this project region.",
                        "framed decode worker allocation failed");
                } catch (const std::exception& exception) {
                    // LFS-CENSUS-OK(empty-catch): record the single-record decode exception as a worker error
                    publish_worker_error(
                        lfs::ErrorCode::Internal,
                        "The framed project payload could not be decoded.",
                        exception.what());
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): record an unknown single-record decode exception as a worker error
                    publish_worker_error(
                        lfs::ErrorCode::Internal,
                        "The framed project payload could not be decoded.",
                        "unknown worker exception");
                }
            } else {
                const auto workers_count = std::min<std::size_t>(records.size(), std::max(1u, std::thread::hardware_concurrency()));
                std::vector<std::jthread> workers;
                for (std::size_t w = 0; w < workers_count; ++w)
                    workers.emplace_back([&] {
                        try {
                            while (true) {
                                const auto i = next.fetch_add(1);
                                if (i >= records.size())
                                    return;
                                decode_one(i);
                            }
                        } catch (const std::bad_alloc&) {
                            publish_worker_error(
                                lfs::ErrorCode::ResourceExhausted,
                                "There is not enough memory to decode this project region.",
                                "framed decode worker allocation failed");
                        } catch (const std::exception& exception) {
                            publish_worker_error(
                                lfs::ErrorCode::Internal,
                                "The framed project payload could not be decoded.",
                                exception.what());
                        } catch (...) {
                            publish_worker_error(
                                lfs::ErrorCode::Internal,
                                "The framed project payload could not be decoded.",
                                "unknown worker exception");
                        }
                    });
                workers.clear();
            }
            if (error)
                return status_failure(std::move(*error));
            return {};
        }

    } // namespace

    lfs::Result<std::vector<std::byte>>
    detail::decompress_framed_zstd_for_testing(
        const std::filesystem::path& path, const std::uint64_t offset,
        const std::span<const std::byte> stored, const std::uint64_t expected_size,
        const std::uint64_t maximum_decoded_size) {
        if (expected_size > maximum_decoded_size ||
            expected_size > std::numeric_limits<std::size_t>::max()) {
            return detail::project_error(
                lfs::ErrorCode::ResourceExhausted,
                "The project payload is too large to decode.",
                std::format("decoded size {} exceeds implementation maximum {}",
                            expected_size, maximum_decoded_size),
                path, offset, "payload.framed");
        }
        std::vector<std::byte> decoded;
        try {
            decoded.resize(static_cast<std::size_t>(expected_size));
        } catch (const std::bad_alloc&) {
            return detail::project_error(
                lfs::ErrorCode::ResourceExhausted,
                "There is not enough memory to decode this project region.",
                std::format("allocation of {} decoded bytes failed", expected_size),
                path, offset, "payload.framed");
        } catch (const std::length_error& error) {
            return detail::project_error(
                lfs::ErrorCode::ResourceExhausted,
                "The framed project region cannot be represented by this build.",
                std::format("allocation of {} decoded bytes failed: {}", expected_size,
                            error.what()),
                path, offset, "payload.framed");
        }
        if (auto status = decompress_framed_zstd(path, offset, stored, expected_size,
                                                 maximum_decoded_size, decoded);
            !status) {
            return std::move(status).error();
        }
        return decoded;
    }

    void detail::set_max_payload_materialized_bytes_for_testing(
        const std::optional<std::uint64_t> maximum_decoded_size) {
        payload_materialized_bytes_override = maximum_decoded_size;
    }

    std::uint64_t detail::max_materialized_bytes_for(const ChunkInfo& row) noexcept {
        if (is_payload_class_chunk(row)) {
            return payload_materialized_bytes_override.value_or(
                MAX_PAYLOAD_MATERIALIZED_BYTES);
        }
        return MAX_MATERIALIZED_CHUNK_BYTES;
    }

    lfs::Result<void>
    ProjectReader::read_stored_at(const ChunkInfo& chunk,
                                  const std::uint64_t relative_offset,
                                  const std::span<std::byte> destination) const {
        if (auto allowed = require_supported_payload_access(*this); !allowed) {
            return allowed;
        }
        auto resolved = resolve_chunk(*impl_->state, chunk);
        if (!resolved) {
            return status_failure(std::move(resolved).error());
        }
        auto range_end = detail::checked_add(relative_offset, destination.size(), path(),
                                             (*resolved)->payload_offset,
                                             "chunk.relative_read_range");
        if (!range_end) {
            return status_failure(std::move(range_end).error());
        }
        if (*range_end > (*resolved)->stored_bytes) {
            return status_failure(format_error(
                path(), (*resolved)->payload_offset, "chunk.relative_read_range",
                std::format("end <= stored bytes {}", (*resolved)->stored_bytes),
                std::to_string(*range_end), lfs::ErrorCode::BoundsViolation));
        }
        if (destination.empty()) {
            return {};
        }
        if (impl_->state->options
                .payload_bytes_read) {
            impl_->state->options
                .payload_bytes_read
                ->fetch_add(
                    destination.size(),
                    std::memory_order_relaxed);
        }
        if ((*resolved)->block_crc_table.has_value()) {
            if (auto verified = verify_block_range(
                    *impl_->state, **resolved, relative_offset,
                    destination.size());
                !verified) {
                return verified;
            }
        }
        return impl_->state->file->read_exact(
            (*resolved)->payload_offset + relative_offset, destination);
    }

    lfs::Result<void>
    ProjectReader::read_logical_prefix(const ChunkInfo& chunk,
                                       const std::span<std::byte> destination) const {
        if (auto allowed = require_supported_payload_access(*this); !allowed) {
            return allowed;
        }
        auto resolved = resolve_chunk(*impl_->state, chunk);
        if (!resolved) {
            return status_failure(std::move(resolved).error());
        }
        const ChunkInfo& row = **resolved;
        if (destination.size() > row.uncompressed_bytes) {
            return status_failure(format_error(
                path(), row.payload_offset, "payload.logical_prefix",
                std::format("end <= uncompressed bytes {}", row.uncompressed_bytes),
                std::to_string(destination.size()), lfs::ErrorCode::BoundsViolation));
        }
        if (destination.empty()) {
            return {};
        }
        if (row.compression == Compression::Stored) {
            return read_stored_at(row, 0, destination);
        }
        if (row.compression != Compression::ZstdFramed &&
            row.compression != Compression::ByteShuffleZstdFramed) {
            return status_failure(detail::project_error(
                lfs::ErrorCode::InvalidArgument,
                "This project chunk uses an unsupported compression encoding.",
                std::format("compression {}",
                            static_cast<std::uint16_t>(row.compression)),
                path(), row.payload_offset, "payload.compression"));
        }

        struct FramedRecord {
            std::uint64_t so = 0;
            std::uint64_t sb = 0;
            std::uint64_t uo = 0;
            std::uint64_t ub = 0;
        };
        const auto fail_frame = [&](const std::string_view detail) {
            return status_failure(format_error(
                path(), row.payload_offset, "payload.framed.header", "valid header",
                std::string(detail)));
        };
        if (row.stored_bytes < detail::FRAMED_HEADER_BYTES) {
            return fail_frame("invalid magic");
        }
        std::array<std::byte, detail::FRAMED_HEADER_BYTES> header{};
        if (auto read = read_stored_at(row, 0, byte_span(header)); !read) {
            return read;
        }
        if (!std::equal(detail::FRAMED_MAGIC.begin(), detail::FRAMED_MAGIC.end(),
                        header.begin())) {
            return fail_frame("invalid magic");
        }
        const auto count = read_u32(byte_span(header), 12);
        if (count == 0 ||
            count > (std::numeric_limits<std::size_t>::max() -
                     detail::FRAMED_HEADER_BYTES) /
                        detail::FRAMED_RECORD_BYTES) {
            return fail_frame("invalid header");
        }
        const auto table = detail::FRAMED_HEADER_BYTES +
                           static_cast<std::size_t>(count) *
                               detail::FRAMED_RECORD_BYTES;
        if (read_u16(byte_span(header), 8) != detail::FRAMED_VERSION ||
            read_u16(byte_span(header), 10) != 0 || table > row.stored_bytes) {
            return fail_frame("invalid header");
        }
        std::vector<std::byte> table_bytes(table);
        std::memcpy(table_bytes.data(), header.data(), header.size());
        if (table > header.size()) {
            if (auto read = read_stored_at(
                    row, header.size(),
                    std::span<std::byte>(table_bytes.data() + header.size(),
                                         table - header.size()));
                !read) {
                return read;
            }
        }
        std::vector<FramedRecord> records;
        records.reserve(count);
        std::uint64_t so = table;
        std::uint64_t uo = 0;
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto at = detail::FRAMED_HEADER_BYTES +
                            static_cast<std::size_t>(index) *
                                detail::FRAMED_RECORD_BYTES;
            const auto sb64 = read_u64(table_bytes, at);
            const auto ub64 = read_u64(table_bytes, at + 8);
            if (sb64 == 0 || ub64 == 0 || sb64 > row.stored_bytes - so ||
                ub64 > row.uncompressed_bytes - uo) {
                return fail_frame("invalid record range");
            }
            records.push_back({so, sb64, uo, ub64});
            so += sb64;
            uo += ub64;
        }
        if (so != row.stored_bytes || uo != row.uncompressed_bytes) {
            return fail_frame("record sizes do not cover payload");
        }

        const auto stream_record_prefix =
            [&](const FramedRecord& record, const std::uint64_t skip,
                const std::span<std::byte> dest) -> lfs::Result<void> {
            if (dest.empty()) {
                return {};
            }
            if (skip + dest.size() > record.ub) {
                return status_failure(format_error(
                    path(), row.payload_offset + record.so, "payload.framed.record",
                    "prefix within record", "range exceeds record"));
            }
            ZSTD_DCtx* dctx = ZSTD_createDCtx();
            if (dctx == nullptr) {
                return status_failure(detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to decode this project region.",
                    "zstd stream context allocation failed", path(),
                    row.payload_offset, "payload.framed.prefix"));
            }
            struct DCtxGuard {
                ZSTD_DCtx* ctx;
                ~DCtxGuard() { ZSTD_freeDCtx(ctx); }
            } guard{dctx};
            const std::size_t in_size = static_cast<std::size_t>(
                std::min<std::uint64_t>(BLOCK_CRC_BYTES, record.sb));
            std::vector<std::byte> inbuf(in_size == 0 ? 1 : in_size);
            std::uint64_t stored_off = 0;
            ZSTD_inBuffer input{nullptr, 0, 0};
            const auto pull = [&](const std::span<std::byte> out_span)
                -> lfs::Result<void> {
                ZSTD_outBuffer output{out_span.data(), out_span.size(), 0};
                while (output.pos < output.size) {
                    if (input.pos == input.size) {
                        if (stored_off >= record.sb) {
                            return status_failure(format_error(
                                path(), row.payload_offset + record.so,
                                "payload.framed.record", "complete zstd prefix",
                                "truncated frame"));
                        }
                        const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(
                            inbuf.size(), record.sb - stored_off));
                        auto buf = std::span<std::byte>(inbuf.data(), n);
                        if (auto read =
                                read_stored_at(row, record.so + stored_off, buf);
                            !read) {
                            return read;
                        }
                        stored_off += n;
                        input = {inbuf.data(), n, 0};
                    }
                    const auto ret =
                        ZSTD_decompressStream(dctx, &output, &input);
                    if (ZSTD_isError(ret)) {
                        return status_failure(format_error(
                            path(), row.payload_offset + record.so,
                            "payload.framed.record", "valid zstd frame",
                            ZSTD_getErrorName(ret)));
                    }
                }
                return {};
            };
            std::uint64_t remaining_skip = skip;
            std::array<std::byte, 64 * 1024> skipbuf{};
            while (remaining_skip > 0) {
                const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(
                    skipbuf.size(), remaining_skip));
                if (auto filled = pull(std::span<std::byte>(skipbuf.data(), n));
                    !filled) {
                    return filled;
                }
                remaining_skip -= n;
            }
            return pull(dest);
        };

        if (row.compression == Compression::ZstdFramed) {
            const FramedRecord& record = records.front();
            const auto take = std::min<std::uint64_t>(destination.size(), record.ub);
            return stream_record_prefix(
                record, 0, destination.first(static_cast<std::size_t>(take)));
        }
        if (row.uncompressed_bytes % 4 != 0) {
            return status_failure(format_error(
                path(), row.payload_offset, "payload.byteshuffle",
                "decoded size multiple of 4",
                std::to_string(row.uncompressed_bytes)));
        }
        const std::uint64_t n_words = row.uncompressed_bytes / 4;
        struct PlaneNeed {
            std::uint64_t shuffled = 0;
            std::size_t dest_index = 0;
        };
        std::vector<PlaneNeed> needs;
        needs.reserve(destination.size());
        for (std::size_t i = 0; i < destination.size(); ++i) {
            needs.push_back(PlaneNeed{
                .shuffled = (i % 4) * n_words + (i / 4),
                .dest_index = i,
            });
        }
        std::sort(needs.begin(), needs.end(),
                  [](const PlaneNeed& lhs, const PlaneNeed& rhs) {
                      return lhs.shuffled < rhs.shuffled;
                  });
        std::size_t cursor = 0;
        while (cursor < needs.size()) {
            const std::uint64_t pos = needs[cursor].shuffled;
            const FramedRecord* record = nullptr;
            for (const auto& candidate : records) {
                if (pos >= candidate.uo && pos < candidate.uo + candidate.ub) {
                    record = &candidate;
                    break;
                }
            }
            if (record == nullptr) {
                return fail_frame("prefix is outside framed records");
            }
            std::size_t end = cursor;
            while (end < needs.size() &&
                   needs[end].shuffled >= record->uo &&
                   needs[end].shuffled < record->uo + record->ub) {
                ++end;
            }
            const std::uint64_t lo = needs[cursor].shuffled;
            const std::uint64_t hi = needs[end - 1].shuffled + 1;
            std::vector<std::byte> slice(static_cast<std::size_t>(hi - lo));
            if (auto decoded = stream_record_prefix(
                    *record, lo - record->uo, slice);
                !decoded) {
                return decoded;
            }
            for (std::size_t i = cursor; i < end; ++i) {
                destination[needs[i].dest_index] =
                    slice[static_cast<std::size_t>(needs[i].shuffled - lo)];
            }
            cursor = end;
        }
        return {};
    }

    lfs::Result<void> ProjectReader::verify_chunk(const ChunkInfo& chunk) const {
        if (auto allowed = require_supported_payload_access(*this); !allowed) {
            return allowed;
        }
        auto resolved = resolve_chunk(*impl_->state, chunk);
        if (!resolved) {
            return status_failure(std::move(resolved).error());
        }
        const ChunkInfo& row = **resolved;
        std::uint32_t running_crc = 0;
        std::uint64_t relative = 0;
        std::size_t block_index = 0;
        while (relative < row.stored_bytes) {
            const std::uint64_t count =
                std::min<std::uint64_t>(BLOCK_CRC_BYTES,
                                        row.stored_bytes - relative);
            if (impl_->state->options
                    .payload_bytes_read) {
                impl_->state->options
                    .payload_bytes_read
                    ->fetch_add(
                        count,
                        std::memory_order_relaxed);
            }
            auto bytes = read_vector(*impl_->state->file,
                                     row.payload_offset + relative, count,
                                     impl_->state->physical_size,
                                     commit().committed_file_end, "payload.verify",
                                     BLOCK_CRC_BYTES);
            if (!bytes) {
                return status_failure(std::move(bytes).error());
            }
            const std::uint32_t block_crc =
                crc32c(0, bytes->data(), bytes->size());
            if (row.block_crc_table.has_value()) {
                const auto& entries = row.block_crc_table->entries;
                assert(block_index < entries.size());
                if (block_crc != entries[block_index]) {
                    return status_failure(format_error(
                        path(), row.payload_offset + relative,
                        std::format("payload[{}].block[{}].crc32c",
                                    row.key_string(), block_index),
                        std::format("0x{:08x}", entries[block_index]),
                        std::format("0x{:08x}", block_crc)));
                }
            }
            running_crc = crc32c(running_crc, bytes->data(), bytes->size());
            relative += count;
            ++block_index;
        }
        if (running_crc != row.payload_crc32c) {
            return status_failure(format_error(
                path(), row.payload_offset,
                std::format("payload[{}].crc32c", row.key_string()),
                std::format("0x{:08x}", row.payload_crc32c),
                std::format("0x{:08x}", running_crc)));
        }
        if (row.block_crc_table.has_value() &&
            block_index != row.block_crc_table->entries.size()) {
            return status_failure(format_error(
                path(), row.block_crc_table->offset + 40,
                std::format("payload[{}].block_count", row.key_string()),
                std::to_string(row.block_crc_table->entries.size()),
                std::to_string(block_index)));
        }
        if (row.compression == Compression::ZstdFramed ||
            row.compression == Compression::ByteShuffleZstdFramed) {
            if (row.stored_bytes > std::numeric_limits<std::size_t>::max()) {
                return status_failure(format_error(
                    path(), row.payload_offset, "payload.zstd",
                    "stored size addressable by this build",
                    std::to_string(row.stored_bytes)));
            }
            const std::uint64_t materialize_max =
                detail::max_materialized_bytes_for(row);
            auto stored = read_vector(*impl_->state->file, row.payload_offset,
                                      row.stored_bytes,
                                      impl_->state->physical_size,
                                      commit().committed_file_end,
                                      "payload.zstd",
                                      materialize_max,
                                      lfs::ErrorCode::ResourceExhausted);
            if (!stored) {
                return status_failure(std::move(stored).error());
            }
            if (row.uncompressed_bytes > materialize_max ||
                row.uncompressed_bytes > std::numeric_limits<std::size_t>::max()) {
                return status_failure(detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "The project payload is too large to decode.",
                    std::format("decoded size {} exceeds implementation maximum {}",
                                row.uncompressed_bytes, materialize_max),
                    path(), row.payload_offset, "payload.framed"));
            }
            auto decoded = allocate_uninitialized_decoded(
                static_cast<std::size_t>(row.uncompressed_bytes), path(),
                row.payload_offset);
            if (!decoded) {
                return status_failure(std::move(decoded).error());
            }
            if (auto inflated = decompress_framed_zstd(
                    path(), row.payload_offset, *stored, row.uncompressed_bytes,
                    materialize_max, decoded->bytes());
                !inflated) {
                return status_failure(std::move(inflated).error());
            }
            if (row.compression == Compression::ByteShuffleZstdFramed &&
                decoded->size % 4 != 0) {
                return status_failure(format_error(
                    path(), row.payload_offset, "payload.byteshuffle",
                    "decoded size multiple of 4",
                    std::to_string(decoded->size)));
            }
        }
        return {};
    }

    lfs::Result<void> ProjectReader::verify_all() const {
        for (const auto& chunk : chunks()) {
            if (chunk.row_kind != RowKind::Live) {
                continue;
            }
            if (auto verified = verify_chunk(chunk); !verified) {
                return verified;
            }
        }
        return {};
    }

    namespace {

        struct PayloadBlock {
            std::size_t index = 0;
            std::uint64_t relative = 0;
            std::size_t count = 0;
        };

        PayloadBlock payload_block_at(const ChunkInfo& row,
                                      const std::size_t index) noexcept {
            const auto relative =
                static_cast<std::uint64_t>(index) * BLOCK_CRC_BYTES;
            const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
                BLOCK_CRC_BYTES, row.stored_bytes - relative));
            return PayloadBlock{.index = index, .relative = relative, .count = count};
        }

        lfs::Result<void> read_one_payload_block(
            const detail::NativeFile& file, const ChunkInfo& row,
            const std::span<std::byte> stored, const PayloadBlock& block,
            std::uint32_t& crc_out, std::atomic_uint64_t* bytes_read,
            const std::filesystem::path& path) {
            auto dest = stored.subspan(static_cast<std::size_t>(block.relative),
                                       block.count);
            if (auto read = file.read_exact(row.payload_offset + block.relative,
                                            dest);
                !read) {
                return status_failure(std::move(read).error());
            }
            if (bytes_read != nullptr) {
                bytes_read->fetch_add(block.count, std::memory_order_relaxed);
            }
            crc_out = crc32c(0, dest.data(), dest.size());
            if (row.block_crc_table.has_value()) {
                const auto& entries = row.block_crc_table->entries;
                if (block.index >= entries.size()) {
                    return status_failure(detail::project_error(
                        lfs::ErrorCode::DataLoss,
                        "The project payload block table is inconsistent.",
                        std::format("payload {} has more blocks than its CRC table",
                                    row.key_string()),
                        path, row.block_crc_table->offset + 40,
                        "payload.block_count"));
                }
                if (crc_out != entries[block.index]) {
                    return status_failure(format_error(
                        path, row.payload_offset + block.relative,
                        std::format("payload[{}].block[{}].crc32c",
                                    row.key_string(), block.index),
                        std::format("0x{:08x}", entries[block.index]),
                        std::format("0x{:08x}", crc_out)));
                }
            }
            return {};
        }

        template <typename OnComplete>
        lfs::Result<void> read_payload_blocks(
            const detail::NativeFile& file, const ChunkInfo& row,
            const std::span<std::byte> stored, const std::size_t begin,
            const std::size_t end, const std::span<std::uint32_t> crcs,
            const std::span<std::uint8_t> validated,
            std::atomic_uint64_t* bytes_read, const std::filesystem::path& path,
            OnComplete on_complete,
            const std::size_t max_workers =
                std::max(1u, std::thread::hardware_concurrency())) {
            if (begin >= end) {
                return {};
            }
            const auto run_one = [&](const std::size_t index) -> lfs::Result<void> {
                const auto block = payload_block_at(row, index);
                std::uint32_t crc = 0;
                if (auto read = read_one_payload_block(file, row, stored, block, crc,
                                                       bytes_read, path);
                    !read) {
                    return read;
                }
                crcs[index] = crc;
                if (!validated.empty()) {
                    validated[index] = 1;
                }
                on_complete(index);
                return {};
            };
            if (end - begin == 1) {
                try {
                    return run_one(begin);
                } catch (const std::bad_alloc&) {
                    return status_failure(detail::project_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "There is not enough memory to read this project chunk.",
                        "payload block buffer allocation failed", path,
                        row.payload_offset, "payload.read"));
                } catch (const std::exception& exception) {
                    // LFS-CENSUS-OK(empty-catch): propagate the payload-read exception as a project error
                    return status_failure(detail::project_error(
                        lfs::ErrorCode::Internal,
                        "The project payload could not be read.", exception.what(),
                        path, row.payload_offset, "payload.read"));
                }
            }

            std::mutex mutex;
            std::optional<lfs::Error> error;
            std::size_t error_index = std::numeric_limits<std::size_t>::max();
            std::atomic<std::size_t> next{begin};
            const auto workers_count = std::min<std::size_t>(
                end - begin, std::max<std::size_t>(1, max_workers));
            const auto record_error = [&](const std::size_t index, lfs::Error err) {
                std::scoped_lock lock(mutex);
                if (!error || index < error_index) {
                    error = std::move(err);
                    error_index = index;
                }
            };
            std::vector<std::jthread> workers;
            try {
                workers.reserve(workers_count);
                for (std::size_t worker = 0; worker < workers_count; ++worker) {
                    workers.emplace_back([&] {
                        try {
                            while (true) {
                                const auto index = next.fetch_add(1);
                                if (index >= end) {
                                    return;
                                }
                                {
                                    std::scoped_lock lock(mutex);
                                    if (error) {
                                        return;
                                    }
                                }
                                if (auto read = run_one(index); !read) {
                                    record_error(index, std::move(read).error());
                                    return;
                                }
                            }
                        } catch (const std::bad_alloc&) {
                            record_error(
                                std::numeric_limits<std::size_t>::max(),
                                detail::project_error(
                                    lfs::ErrorCode::ResourceExhausted,
                                    "There is not enough memory to read this project chunk.",
                                    "payload block worker allocation failed", path,
                                    row.payload_offset, "payload.read"));
                        } catch (const std::exception& exception) {
                            // LFS-CENSUS-OK(empty-catch): record the payload-read worker exception as the first read error
                            record_error(
                                std::numeric_limits<std::size_t>::max(),
                                detail::project_error(
                                    lfs::ErrorCode::Internal,
                                    "The project payload could not be read.",
                                    exception.what(), path, row.payload_offset,
                                    "payload.read"));
                        } catch (...) {
                            // LFS-CENSUS-OK(empty-catch): record an unknown payload-read worker exception as the first read error
                            record_error(
                                std::numeric_limits<std::size_t>::max(),
                                detail::project_error(
                                    lfs::ErrorCode::Internal,
                                    "The project payload could not be read.",
                                    "unknown worker exception", path,
                                    row.payload_offset, "payload.read"));
                        }
                    });
                }
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): propagate payload-read worker construction failure as a project error
                workers.clear();
                return status_failure(detail::project_error(
                    lfs::ErrorCode::Internal,
                    "The project payload could not be read.", exception.what(),
                    path, row.payload_offset, "payload.read"));
            }
            workers.clear();
            if (error) {
                return status_failure(std::move(*error));
            }
            return {};
        }

        std::uint32_t combine_block_crcs(const ChunkInfo& row,
                                         const std::span<const std::uint32_t> crcs) {
            std::uint32_t running = 0;
            for (std::size_t index = 0; index < crcs.size(); ++index) {
                running = crc32c_combine(running, crcs[index],
                                         payload_block_at(row, index).count);
            }
            return running;
        }

        enum class LogicalInit : std::uint8_t { Zeroed,
                                                Uninitialized };

        struct MaterializedLogical {
            std::vector<std::byte> zeroed;
            UninitializedBuffer uninitialized;
            UninitializedBuffer stored;
            UninitializedBuffer shuffle_planes;
            LogicalInit init = LogicalInit::Zeroed;

            [[nodiscard]] std::span<std::byte> bytes() noexcept {
                if (init == LogicalInit::Uninitialized) {
                    return uninitialized.bytes();
                }
                return std::span<std::byte>(zeroed);
            }

            [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
                if (init == LogicalInit::Uninitialized) {
                    return uninitialized.bytes();
                }
                return std::span<const std::byte>(zeroed);
            }
        };

        lfs::Result<void> allocate_logical_bytes(
            MaterializedLogical& out, const LogicalInit init, const std::size_t bytes,
            const std::filesystem::path& path, const std::uint64_t offset,
            const std::string_view field) {
            out.init = init;
            if (init == LogicalInit::Uninitialized) {
                auto buffer =
                    allocate_uninitialized_buffer(bytes, path, offset, field);
                if (!buffer) {
                    return status_failure(std::move(buffer).error());
                }
                out.uninitialized = std::move(*buffer);
                return {};
            }
            auto vector = allocate_zeroed_vector(bytes, path, offset, field);
            if (!vector) {
                return status_failure(std::move(vector).error());
            }
            out.zeroed = std::move(*vector);
            return {};
        }

        lfs::Result<MaterializedLogical> materialize_chunk_payload(
            const ProjectReader& reader, const ReaderState& state,
            const ChunkInfo& chunk,
            std::function<void(std::size_t, std::size_t)> progress,
            const LogicalInit init) {
            const auto read_started = std::chrono::steady_clock::now();
            if (auto allowed = require_supported_payload_access(reader); !allowed) {
                return std::move(allowed).error();
            }
            auto resolved = resolve_chunk(state, chunk);
            if (!resolved) {
                return std::move(resolved).error();
            }
            const auto resolved_at = std::chrono::steady_clock::now();
            const ChunkInfo& row = **resolved;
            const std::filesystem::path& path = reader.path();
            const std::uint64_t materialize_max =
                detail::max_materialized_bytes_for(**resolved);
            if ((*resolved)->stored_bytes > materialize_max ||
                (*resolved)->uncompressed_bytes > materialize_max) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "This project chunk is too large to materialize in memory.",
                    std::format(
                        "stored bytes {}, decoded bytes {}, materialized-read maximum {}; "
                        "use the bounded stream or mapped-range API",
                        (*resolved)->stored_bytes, (*resolved)->uncompressed_bytes,
                        materialize_max),
                    path, (*resolved)->payload_offset, "payload.materialized_size");
            }
            if (row.stored_bytes > std::numeric_limits<std::size_t>::max()) {
                return detail::project_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "This project chunk is too large to materialize in memory.",
                    std::format("stored bytes {} exceed this build's size_t range",
                                row.stored_bytes),
                    path, row.payload_offset, "payload.read");
            }
            auto payload_end = detail::checked_add(row.payload_offset, row.stored_bytes,
                                                   path, row.payload_offset,
                                                   "payload.read");
            if (!payload_end) {
                return std::move(payload_end).error();
            }
            if (*payload_end > reader.commit().committed_file_end) {
                return format_error(
                    path, row.payload_offset, "payload.read",
                    std::format("end <= committed file end {}",
                                reader.commit().committed_file_end),
                    std::to_string(*payload_end), lfs::ErrorCode::BoundsViolation);
            }
            const bool framed_payload =
                row.compression == Compression::ZstdFramed ||
                row.compression == Compression::ByteShuffleZstdFramed;
            if (framed_payload) {
                auto stored_buffer = allocate_uninitialized_buffer(
                    static_cast<std::size_t>(row.stored_bytes), path,
                    row.payload_offset, "payload.read");
                if (!stored_buffer) {
                    return std::move(stored_buffer).error();
                }
                UninitializedBuffer stored = std::move(*stored_buffer);
                const auto stored_span = stored.bytes();
                struct Record {
                    std::size_t stored_offset;
                    std::size_t stored_bytes;
                    std::size_t decoded_offset;
                    std::size_t decoded_bytes;
                };
                const auto fail_header = [&](const std::string_view detail)
                    -> lfs::Result<MaterializedLogical> {
                    return format_error(path, row.payload_offset,
                                        "payload.framed.header", "valid header",
                                        std::string(detail));
                };
                const auto block_count = static_cast<std::size_t>(
                    (row.stored_bytes + BLOCK_CRC_BYTES - 1) / BLOCK_CRC_BYTES);
                std::vector<std::uint8_t> validated(block_count, 0);
                std::vector<std::uint32_t> block_crcs(block_count, 0);
                std::size_t loaded_blocks = 0;
                auto* bytes_read = state.options.payload_bytes_read.get();
                const auto read_block =
                    [&](const std::size_t index) -> lfs::Result<void> {
                    const auto block = payload_block_at(row, index);
                    std::uint32_t crc = 0;
                    if (auto read = read_one_payload_block(
                            *state.file, row, stored_span, block, crc, bytes_read,
                            path);
                        !read) {
                        return read;
                    }
                    block_crcs[index] = crc;
                    validated[index] = 1;
                    ++loaded_blocks;
                    return {};
                };

                while (loaded_blocks == 0 && row.stored_bytes != 0) {
                    if (auto read = read_block(0); !read)
                        return std::move(read).error();
                }
                if (stored.size < detail::FRAMED_HEADER_BYTES ||
                    !std::equal(detail::FRAMED_MAGIC.begin(),
                                detail::FRAMED_MAGIC.end(), stored_span.begin())) {
                    return fail_header("invalid magic");
                }
                const auto count = read_u32(stored_span, 12);
                if (count > (std::numeric_limits<std::size_t>::max() -
                             detail::FRAMED_HEADER_BYTES) /
                                detail::FRAMED_RECORD_BYTES)
                    return fail_header("record table is not representable");
                const auto table = detail::FRAMED_HEADER_BYTES +
                                   static_cast<std::size_t>(count) *
                                       detail::FRAMED_RECORD_BYTES;
                if (read_u16(stored_span, 8) != detail::FRAMED_VERSION ||
                    read_u16(stored_span, 10) != 0 || count == 0 ||
                    table > stored.size)
                    return fail_header("invalid header");
                if (row.uncompressed_bytes > materialize_max ||
                    row.uncompressed_bytes >
                        std::numeric_limits<std::size_t>::max()) {
                    return detail::project_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "The project payload is too large to decode.",
                        std::format(
                            "decoded size {} exceeds implementation maximum {}",
                            row.uncompressed_bytes, materialize_max),
                        path, row.payload_offset, "payload.framed");
                }
                if (count > row.uncompressed_bytes || stored.size - table < count)
                    return fail_header("invalid record count");
                while (loaded_blocks < block_count &&
                       loaded_blocks * BLOCK_CRC_BYTES < table) {
                    if (auto read = read_block(loaded_blocks); !read)
                        return std::move(read).error();
                }
                std::vector<Record> records;
                std::size_t stored_offset = table;
                std::size_t decoded_offset = 0;
                try {
                    records.reserve(count);
                    for (std::size_t index = 0; index < count; ++index) {
                        const auto at = detail::FRAMED_HEADER_BYTES +
                                        index * detail::FRAMED_RECORD_BYTES;
                        const auto stored_bytes = read_u64(stored_span, at);
                        const auto decoded_bytes = read_u64(stored_span, at + 8);
                        if (stored_bytes == 0 || decoded_bytes == 0 ||
                            stored_bytes > stored.size - stored_offset ||
                            decoded_bytes >
                                row.uncompressed_bytes - decoded_offset ||
                            stored_bytes > std::numeric_limits<std::size_t>::max() ||
                            decoded_bytes >
                                std::numeric_limits<std::size_t>::max()) {
                            return format_error(path, row.payload_offset + at,
                                                "payload.framed.record",
                                                "valid record range",
                                                "invalid range");
                        }
                        records.push_back({stored_offset,
                                           static_cast<std::size_t>(stored_bytes),
                                           decoded_offset,
                                           static_cast<std::size_t>(decoded_bytes)});
                        stored_offset += static_cast<std::size_t>(stored_bytes);
                        decoded_offset += static_cast<std::size_t>(decoded_bytes);
                    }
                } catch (const std::bad_alloc&) {
                    return detail::project_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "There is not enough memory to decode this project region.",
                        std::format("allocation of {} record descriptors failed",
                                    count),
                        path, row.payload_offset, "payload.framed.records");
                } catch (const std::length_error& error) {
                    return detail::project_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "The framed project region cannot be represented by this build.",
                        std::format(
                            "allocation of {} record descriptors failed: {}", count,
                            error.what()),
                        path, row.payload_offset, "payload.framed.records");
                }
                if (stored_offset != stored.size ||
                    decoded_offset != row.uncompressed_bytes)
                    return fail_header("record sizes do not cover payload");
                MaterializedLogical logical;
                UninitializedBuffer shuffled;
                std::span<std::byte> decode_dest;
                if (row.compression == Compression::ByteShuffleZstdFramed) {
                    auto planes = allocate_uninitialized_buffer(
                        static_cast<std::size_t>(row.uncompressed_bytes), path,
                        row.payload_offset, "payload.framed");
                    if (!planes) {
                        return std::move(planes).error();
                    }
                    shuffled = std::move(*planes);
                    decode_dest = shuffled.bytes();
                } else {
                    if (auto allocated = allocate_logical_bytes(
                            logical, init,
                            static_cast<std::size_t>(row.uncompressed_bytes), path,
                            row.payload_offset, "payload.framed");
                        !allocated) {
                        return std::move(allocated).error();
                    }
                    decode_dest = logical.bytes();
                }
                std::mutex worker_mutex;
                std::condition_variable worker_cv;
                std::vector<bool> ready(records.size(), false);
                std::vector<bool> claimed(records.size(), false);
                std::optional<lfs::Error> worker_error;
                bool read_failed = false;
                bool read_done = false;
                std::atomic<std::size_t> completed{0};
                const auto publish_worker_error =
                    [&](const lfs::ErrorCode code, const std::string_view message,
                        const std::string_view detail) noexcept {
                        try {
                            std::scoped_lock lock(worker_mutex);
                            if (!worker_error)
                                worker_error = detail::project_error(
                                    code, std::string(message), std::string(detail),
                                    path, row.payload_offset,
                                    "payload.framed.worker");
                        } catch (...) {
                            // LFS-CENSUS-OK(empty-catch): preserve the first worker error if formatting fails.
                        }
                        worker_cv.notify_all();
                    };
                std::vector<std::vector<std::size_t>> records_for_block(block_count);
                const auto pending_blocks =
                    std::make_unique<std::atomic<std::uint32_t>[]>(
                        records.size());
                for (std::size_t index = 0; index < records.size(); ++index) {
                    const auto first =
                        records[index].stored_offset / BLOCK_CRC_BYTES;
                    const auto last = (records[index].stored_offset +
                                       records[index].stored_bytes - 1) /
                                      BLOCK_CRC_BYTES;
                    pending_blocks[index].store(
                        static_cast<std::uint32_t>(last - first + 1),
                        std::memory_order_relaxed);
                    for (auto block = first; block <= last; ++block) {
                        records_for_block[block].push_back(index);
                    }
                }
                const auto complete_block = [&](const std::size_t block) {
                    if (block >= records_for_block.size()) {
                        return;
                    }
                    for (const auto record_index : records_for_block[block]) {
                        if (pending_blocks[record_index].fetch_sub(
                                1, std::memory_order_acq_rel) != 1) {
                            continue;
                        }
                        {
                            std::scoped_lock lock(worker_mutex);
                            ready[record_index] = true;
                        }
                        worker_cv.notify_one();
                    }
                };
                for (std::size_t block = 0; block < loaded_blocks; ++block) {
                    complete_block(block);
                }
                const auto worker_count = std::min<std::size_t>(
                    records.size(),
                    std::max(1u, std::thread::hardware_concurrency()));
                std::vector<std::jthread> workers;
                const auto workers_started = std::chrono::steady_clock::now();
                try {
                    workers.reserve(worker_count);
                    for (std::size_t worker = 0; worker < worker_count; ++worker) {
                        workers.emplace_back([&] {
                            try {
                                while (true) {
                                    std::size_t index = records.size();
                                    {
                                        std::unique_lock lock(worker_mutex);
                                        worker_cv.wait(lock, [&] {
                                            bool has_ready = false;
                                            for (std::size_t candidate = 0;
                                                 candidate < ready.size();
                                                 ++candidate) {
                                                has_ready |= ready[candidate] &&
                                                             !claimed[candidate];
                                            }
                                            return worker_error.has_value() ||
                                                   read_failed || read_done ||
                                                   has_ready;
                                        });
                                        if (worker_error || read_failed)
                                            return;
                                        for (std::size_t candidate = 0;
                                             candidate < records.size();
                                             ++candidate) {
                                            if (ready[candidate] &&
                                                !claimed[candidate]) {
                                                claimed[candidate] = true;
                                                index = candidate;
                                                break;
                                            }
                                        }
                                        if (index == records.size()) {
                                            if (read_done)
                                                return;
                                            continue;
                                        }
                                    }
                                    const auto& record = records[index];
                                    const auto frame = std::span<const std::byte>(
                                        stored_span.data() + record.stored_offset,
                                        record.stored_bytes);
                                    const auto frame_size = ZSTD_getFrameContentSize(
                                        frame.data(), frame.size());
                                    if (frame_size != record.decoded_bytes) {
                                        publish_worker_error(
                                            lfs::ErrorCode::DataLoss,
                                            "The framed project payload could not be decoded.",
                                            "record content size mismatch");
                                        return;
                                    }
                                    const auto result = ZSTD_decompress(
                                        decode_dest.data() + record.decoded_offset,
                                        record.decoded_bytes, frame.data(),
                                        record.stored_bytes);
                                    if (ZSTD_isError(result) ||
                                        result != record.decoded_bytes) {
                                        publish_worker_error(
                                            lfs::ErrorCode::DataLoss,
                                            "The framed project payload could not be decoded.",
                                            "record decompression failed");
                                        return;
                                    }
                                    const auto done = completed.fetch_add(1) + 1;
                                    if (progress)
                                        progress(done, records.size());
                                }
                            } catch (const std::bad_alloc&) {
                                publish_worker_error(
                                    lfs::ErrorCode::ResourceExhausted,
                                    "There is not enough memory to decode this project region.",
                                    "framed decode worker allocation failed");
                            } catch (const std::exception& exception) {
                                publish_worker_error(
                                    lfs::ErrorCode::Internal,
                                    "The framed project payload could not be decoded.",
                                    exception.what());
                            } catch (...) {
                                publish_worker_error(
                                    lfs::ErrorCode::Internal,
                                    "The framed project payload could not be decoded.",
                                    "unknown worker exception");
                            }
                        });
                    }
                } catch (const std::bad_alloc&) {
                    publish_worker_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "There is not enough memory to start framed decode workers.",
                        "worker allocation failed");
                    {
                        std::scoped_lock lock(worker_mutex);
                        read_failed = true;
                    }
                    worker_cv.notify_all();
                    workers.clear();
                    return fail_header("could not start framed decode workers");
                } catch (const std::exception& exception) {
                    publish_worker_error(
                        lfs::ErrorCode::Internal,
                        "The framed project payload could not be decoded.",
                        exception.what());
                    {
                        std::scoped_lock lock(worker_mutex);
                        read_failed = true;
                    }
                    worker_cv.notify_all();
                    workers.clear();
                    return fail_header("could not start framed decode workers");
                } catch (...) {
                    publish_worker_error(
                        lfs::ErrorCode::Internal,
                        "The framed project payload could not be decoded.",
                        "unknown worker construction exception");
                    {
                        std::scoped_lock lock(worker_mutex);
                        read_failed = true;
                    }
                    worker_cv.notify_all();
                    workers.clear();
                    return fail_header("could not start framed decode workers");
                }
                worker_cv.notify_all();
                const auto stored_io_started = std::chrono::steady_clock::now();
                const auto pipelined_read_workers = std::min<std::size_t>(
                    12, std::max<std::size_t>(
                            4, std::thread::hardware_concurrency() / 2));
                if (auto remaining = read_payload_blocks(
                        *state.file, row, stored_span, loaded_blocks, block_count,
                        block_crcs, std::span<std::uint8_t>(validated), bytes_read,
                        path,
                        [&](const std::size_t block) { complete_block(block); },
                        pipelined_read_workers);
                    !remaining) {
                    {
                        std::scoped_lock lock(worker_mutex);
                        read_failed = true;
                    }
                    worker_cv.notify_all();
                    workers.clear();
                    return std::move(remaining).error();
                }
                loaded_blocks = block_count;
                const auto stored_read_at = std::chrono::steady_clock::now();
                {
                    std::scoped_lock lock(worker_mutex);
                    read_done = true;
                }
                worker_cv.notify_all();
                workers.clear();
                const auto decompressed_at = std::chrono::steady_clock::now();
                if (row.block_crc_table.has_value() &&
                    block_count != row.block_crc_table->entries.size()) {
                    return format_error(
                        path, row.block_crc_table->offset + 40,
                        std::format("payload[{}].block_count", row.key_string()),
                        std::to_string(row.block_crc_table->entries.size()),
                        std::to_string(block_count));
                }
                const auto running_crc = combine_block_crcs(row, block_crcs);
                if (running_crc != row.payload_crc32c) {
                    return format_error(
                        path, row.payload_offset,
                        std::format("payload[{}].crc32c", row.key_string()),
                        std::format("0x{:08x}", row.payload_crc32c),
                        std::format("0x{:08x}", running_crc));
                }
                if (worker_error)
                    return std::move(*worker_error);
                if (completed != records.size())
                    return fail_header("not all records were decoded");
                logical.stored = std::move(stored);
                const auto milliseconds = [](const auto begin, const auto end) {
                    return std::chrono::duration<double, std::milli>(end - begin)
                        .count();
                };
                if (row.compression == Compression::ZstdFramed) {
                    LOG_DEBUG(
                        "Project chunk read stages: chunk={} stored_bytes={} uncompressed_bytes={} compression={} resolve={:.3f} ms stored_read={:.3f} ms decompress={:.3f} ms unshuffle=0.000 ms total={:.3f} ms",
                        row.key_string(), row.stored_bytes, row.uncompressed_bytes,
                        static_cast<std::uint16_t>(row.compression),
                        milliseconds(read_started, resolved_at),
                        milliseconds(stored_io_started, stored_read_at),
                        milliseconds(workers_started, decompressed_at),
                        milliseconds(read_started, decompressed_at));
                    return logical;
                }
                if (shuffled.size % 4 != 0)
                    return format_error(path, row.payload_offset,
                                        "payload.byteshuffle",
                                        "decoded size multiple of 4",
                                        std::to_string(shuffled.size));
                if (auto allocated = allocate_logical_bytes(
                        logical, init,
                        static_cast<std::size_t>(row.uncompressed_bytes), path,
                        row.payload_offset, "payload.framed");
                    !allocated) {
                    return std::move(allocated).error();
                }
                unbyte_plane_f32_words(shuffled.bytes(), logical.bytes());
                logical.shuffle_planes = std::move(shuffled);
                const auto unshuffled_at = std::chrono::steady_clock::now();
                LOG_DEBUG(
                    "Project chunk read stages: chunk={} stored_bytes={} uncompressed_bytes={} compression={} resolve={:.3f} ms stored_read={:.3f} ms decompress={:.3f} ms unshuffle={:.3f} ms total={:.3f} ms",
                    row.key_string(), row.stored_bytes, row.uncompressed_bytes,
                    static_cast<std::uint16_t>(row.compression),
                    milliseconds(read_started, resolved_at),
                    milliseconds(stored_io_started, stored_read_at),
                    milliseconds(workers_started, decompressed_at),
                    milliseconds(decompressed_at, unshuffled_at),
                    milliseconds(read_started, unshuffled_at));
                return logical;
            }
            MaterializedLogical logical;
            if (auto allocated = allocate_logical_bytes(
                    logical, init, static_cast<std::size_t>(row.stored_bytes), path,
                    row.payload_offset, "payload.read");
                !allocated) {
                return std::move(allocated).error();
            }
            const auto block_count = static_cast<std::size_t>(
                (row.stored_bytes + BLOCK_CRC_BYTES - 1) / BLOCK_CRC_BYTES);
            std::vector<std::uint32_t> block_crcs(block_count, 0);
            std::vector<std::uint8_t> validated(
                row.block_crc_table.has_value() ? block_count : 0, 0);
            if (auto read = read_payload_blocks(
                    *state.file, row, logical.bytes(), 0, block_count, block_crcs,
                    validated, state.options.payload_bytes_read.get(), path,
                    [](const std::size_t) {});
                !read) {
                return std::move(read).error();
            }
            const auto running_crc = combine_block_crcs(row, block_crcs);
            if (running_crc != row.payload_crc32c) {
                return format_error(
                    path, row.payload_offset,
                    std::format("payload[{}].crc32c", row.key_string()),
                    std::format("0x{:08x}", row.payload_crc32c),
                    std::format("0x{:08x}", running_crc));
            }
            if (row.block_crc_table.has_value() &&
                block_count != row.block_crc_table->entries.size()) {
                return format_error(
                    path, row.block_crc_table->offset + 40,
                    std::format("payload[{}].block_count", row.key_string()),
                    std::to_string(row.block_crc_table->entries.size()),
                    std::to_string(block_count));
            }
            const auto stored_read_at = std::chrono::steady_clock::now();
            const auto milliseconds = [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(end - begin)
                    .count();
            };
            if (row.compression == Compression::Stored) {
                LOG_DEBUG(
                    "Project chunk read stages: chunk={} stored_bytes={} uncompressed_bytes={} compression={} resolve={:.3f} ms stored_read={:.3f} ms decompress=0.000 ms unshuffle=0.000 ms total={:.3f} ms",
                    row.key_string(), row.stored_bytes, row.uncompressed_bytes,
                    static_cast<std::uint16_t>(row.compression),
                    milliseconds(read_started, resolved_at),
                    milliseconds(resolved_at, stored_read_at),
                    milliseconds(read_started, stored_read_at));
                return logical;
            }
            return detail::project_error(
                lfs::ErrorCode::InvalidArgument,
                "This project chunk uses an unsupported compression encoding.",
                std::format("compression {}",
                            static_cast<std::uint16_t>(row.compression)),
                path, row.payload_offset, "payload.compression");
        }

        void retire_materialized_buffers_async(MaterializedLogical buffers) {
            try {
                std::thread([retired = std::move(buffers)]() mutable {
                    retired = {};
                }).detach();
            } catch (const std::exception& e) {
                LOG_WARN(
                    "Failed to start asynchronous materialize buffer retirement: {}",
                    e.what());
                buffers = {};
            }
        }

        void adopt_materialized_buffers(MaterializeRetirementSink& retirement,
                                        MaterializedLogical&& buffers) {
            retirement.adopt(std::move(buffers.uninitialized.storage));
            retirement.adopt(std::move(buffers.stored.storage));
            retirement.adopt(std::move(buffers.shuffle_planes.storage));
            retirement.adopt(std::move(buffers.zeroed));
        }

    } // namespace

    void MaterializeRetirementSink::retire_async() {
        if (buffers_.empty() && vectors_.empty()) {
            return;
        }
        try {
            std::thread([buffers = std::move(buffers_),
                         vectors = std::move(vectors_)]() mutable {
                buffers.clear();
                vectors.clear();
            }).detach();
        } catch (const std::exception& e) {
            LOG_WARN(
                "Failed to start asynchronous materialize buffer retirement: {}",
                e.what());
            buffers_.clear();
            vectors_.clear();
        }
    }

    lfs::Result<std::vector<std::byte>>
    ProjectReader::read_chunk(
        const ChunkInfo& chunk,
        std::function<void(std::size_t, std::size_t)> progress) const {
        auto materialized = materialize_chunk_payload(
            *this, *impl_->state, chunk, std::move(progress), LogicalInit::Zeroed);
        if (!materialized) {
            return std::move(materialized).error();
        }
        return std::move(materialized->zeroed);
    }

    lfs::Result<void> ProjectReader::visit_materialized_chunk(
        const ChunkInfo& chunk,
        const std::function<lfs::Result<void>(std::span<const std::byte>)>& visitor,
        MaterializeRetirementSink* retirement) const {
        auto materialized = materialize_chunk_payload(
            *this, *impl_->state, chunk, {}, LogicalInit::Uninitialized);
        if (!materialized) {
            return lfs::Result<void>::failure(std::move(materialized).error());
        }
        auto result = visitor(materialized->bytes());
        // Deserialize synchronizes its upload stream before returning, so handing
        // the decoded buffer off after the visitor is safe for cudaMemcpyAsync.
        if (retirement) {
            adopt_materialized_buffers(*retirement, std::move(*materialized));
        } else {
            retire_materialized_buffers_async(std::move(*materialized));
        }
        return result;
    }

    lfs::Result<std::vector<std::byte>> ProjectReader::read_preview() const {
        if (auto allowed = require_supported_payload_access(*this); !allowed) {
            return std::move(allowed).error();
        }
        if (!preview().has_value()) {
            return detail::project_error(
                lfs::ErrorCode::NotFound,
                "This project generation has no preview.",
                "the selected head carries an all-zero preview locator", path(),
                selected_head().commit_offset, "head.preview_locator");
        }
        const PreviewLocator& locator = *preview();
        const auto row = std::find_if(
            chunks().begin(), chunks().end(), [&](const ChunkInfo& chunk) {
                return chunk.row_kind == RowKind::Live &&
                       chunk.key.fourcc == FOURCC_THMB &&
                       chunk.compression == Compression::Stored &&
                       chunk.payload_offset == locator.offset &&
                       chunk.stored_bytes == locator.bytes;
            });
        assert(row != chunks().end());
        return read_chunk(*row);
    }

    lfs::Result<CleanProof>
    ProjectReader::make_clean_proof(const ChunkInfo& chunk,
                                    const std::uint64_t mutation_epoch) const {
        if (auto allowed = require_supported_payload_access(*this); !allowed) {
            return std::move(allowed).error();
        }
        auto resolved = resolve_chunk(*impl_->state, chunk);
        if (!resolved) {
            return std::move(resolved).error();
        }
        const ChunkInfo& row = **resolved;
        CleanProof proof;
        proof.key_ = row.key;
        proof.file_uuid_ = superblock().file_uuid;
        proof.commit_uuid_ = commit().commit_uuid;
        proof.source_generation_ = row.source_generation;
        proof.payload_crc32c_ = row.payload_crc32c;
        proof.header_crc32c_ = row.header_crc32c;
        proof.mutation_epoch_ = mutation_epoch;
        return proof;
    }

    namespace {

        class PositionalStreambuf final : public std::streambuf {
        public:
            PositionalStreambuf(
                std::shared_ptr<detail::NativeFile> file,
                const std::uint64_t start,
                const std::uint64_t size,
                std::optional<BlockCrcTable> block_crc_table)
                : file_(std::move(file)),
                  start_(start),
                  size_(size),
                  block_crc_table_(std::move(block_crc_table)),
                  buffer_(
                      static_cast<std::size_t>(
                          block_crc_table_
                              ? std::min<std::uint64_t>(
                                    block_crc_table_->block_size,
                                    size_)
                              : std::min<std::uint64_t>(
                                    64 * 1024, size_))) {
                setg(buffer_.data(), buffer_.data(), buffer_.data());
            }

        protected:
            int_type underflow() override {
                if (gptr() < egptr()) {
                    return traits_type::to_int_type(*gptr());
                }
                sync_position();
                if (position_ >= size_) {
                    return traits_type::eof();
                }
                const std::uint64_t read_position =
                    block_crc_table_
                        ? (position_ / block_crc_table_->block_size) *
                              block_crc_table_->block_size
                        : position_;
                const std::uint64_t remaining = size_ - read_position;
                const std::size_t count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(remaining, buffer_.size()));
                auto destination =
                    std::span<std::byte>(reinterpret_cast<std::byte*>(buffer_.data()),
                                         count);
                if (auto read = file_->read_exact(
                        start_ + read_position, destination);
                    !read) {
                    failed_ = true;
                    return traits_type::eof();
                }
                if (block_crc_table_) {
                    const std::size_t block_index =
                        static_cast<std::size_t>(
                            read_position /
                            block_crc_table_->block_size);
                    if (block_index >=
                            block_crc_table_->entries.size() ||
                        crc32c(
                            0, destination.data(),
                            destination.size()) !=
                            block_crc_table_
                                ->entries[block_index]) {
                        failed_ = true;
                        return traits_type::eof();
                    }
                }
                const auto offset =
                    static_cast<std::ptrdiff_t>(
                        position_ - read_position);
                buffer_start_ = read_position;
                setg(
                    buffer_.data(),
                    buffer_.data() + offset,
                    buffer_.data() + count);
                return traits_type::to_int_type(*gptr());
            }

            pos_type seekoff(const off_type offset, const std::ios_base::seekdir direction,
                             const std::ios_base::openmode mode) override {
                if ((mode & std::ios_base::in) == 0) {
                    return pos_type(off_type(-1));
                }
                sync_position();
                std::int64_t base = 0;
                if (direction == std::ios_base::beg) {
                    base = 0;
                } else if (direction == std::ios_base::cur) {
                    if (position_ > static_cast<std::uint64_t>(
                                        std::numeric_limits<std::int64_t>::max())) {
                        return pos_type(off_type(-1));
                    }
                    base = static_cast<std::int64_t>(position_);
                } else if (direction == std::ios_base::end) {
                    if (size_ > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max())) {
                        return pos_type(off_type(-1));
                    }
                    base = static_cast<std::int64_t>(size_);
                } else {
                    return pos_type(off_type(-1));
                }
                if (offset > 0 &&
                    base > std::numeric_limits<std::int64_t>::max() - offset) {
                    return pos_type(off_type(-1));
                }
                if (offset < 0 &&
                    base < std::numeric_limits<std::int64_t>::min() - offset) {
                    return pos_type(off_type(-1));
                }
                const std::int64_t target = base + offset;
                if (target < 0 || static_cast<std::uint64_t>(target) > size_) {
                    return pos_type(off_type(-1));
                }
                position_ = static_cast<std::uint64_t>(target);
                buffer_start_ = position_;
                setg(buffer_.data(), buffer_.data(), buffer_.data());
                return pos_type(static_cast<off_type>(position_));
            }

            pos_type seekpos(const pos_type position,
                             const std::ios_base::openmode mode) override {
                return seekoff(static_cast<off_type>(position), std::ios_base::beg,
                               mode);
            }

        private:
            void sync_position() noexcept {
                if (eback() != nullptr && gptr() != nullptr) {
                    position_ =
                        buffer_start_ + static_cast<std::uint64_t>(gptr() - eback());
                }
            }

            std::shared_ptr<detail::NativeFile> file_;
            std::uint64_t start_ = 0;
            std::uint64_t size_ = 0;
            std::uint64_t position_ = 0;
            std::uint64_t buffer_start_ = 0;
            std::optional<BlockCrcTable> block_crc_table_;
            std::vector<char> buffer_;
            bool failed_ = false;
        };

        // Inverse of the f32 byte-plane shuffle for a logical byte range.
        // `plane_cursors[p]` points at the first in-range byte of plane p
        // (planes that the range does not touch may be null). Full words are
        // gathered without per-byte `% 4` / `/ 4`; unaligned head/tail bytes
        // use a short scalar path.
        void gather_unshuffle_words(char* dest, const std::uint64_t logical_pos,
                                    const std::size_t count,
                                    std::array<const char*, 4>& plane_cursors) {
            if (dest == nullptr || count == 0) {
                return;
            }
            std::size_t offset = 0;
            auto plane = static_cast<std::size_t>(logical_pos & 3u);
            while (offset < count && plane != 0) {
                dest[offset++] = *plane_cursors[plane]++;
                plane = (plane + 1) & 3u;
            }
            const std::size_t remaining = count - offset;
            const std::size_t words = remaining / 4;
            if (words != 0) {
                const char* p0 = plane_cursors[0];
                const char* p1 = plane_cursors[1];
                const char* p2 = plane_cursors[2];
                const char* p3 = plane_cursors[3];
                char* out = dest + offset;
                for (std::size_t word = 0; word < words; ++word) {
                    out[4 * word + 0] = p0[word];
                    out[4 * word + 1] = p1[word];
                    out[4 * word + 2] = p2[word];
                    out[4 * word + 3] = p3[word];
                }
                plane_cursors[0] = p0 + words;
                plane_cursors[1] = p1 + words;
                plane_cursors[2] = p2 + words;
                plane_cursors[3] = p3 + words;
                offset += words * 4;
            }
            plane = 0;
            while (offset < count) {
                dest[offset++] = *plane_cursors[plane]++;
                ++plane;
            }
        }

        std::atomic<std::uint64_t> g_framed_record_decode_calls{0};

        // Logical (decoded / unshuffled) view of a framed payload. Resident
        // decoded records are capped: ZstdFramed keeps current + one
        // read-ahead; ByteShuffleZstdFramed keeps at most one record per
        // byte plane plus a bounded gather window. Bulk `xsgetn` decodes at
        // most kMaxResidentDecodeRecords (~512MiB decoded) beyond dest.
        class FramedLogicalStreambuf final : public std::streambuf {
        public:
            FramedLogicalStreambuf(std::shared_ptr<ReaderState> state,
                                   ChunkInfo row, const bool byteshuffle)
                : state_(std::move(state)),
                  row_(std::move(row)),
                  byteshuffle_(byteshuffle),
                  size_(row_.uncompressed_bytes),
                  window_(byteshuffle_ ? kShuffleWindowBytes : 0) {
                if (row_.block_crc_table.has_value() && row_.stored_bytes != 0) {
                    const auto blocks =
                        (row_.stored_bytes + row_.block_crc_table->block_size - 1) /
                        row_.block_crc_table->block_size;
                    block_validated_.assign(static_cast<std::size_t>(blocks),
                                            std::uint8_t{0});
                }
                setg(nullptr, nullptr, nullptr);
            }

        protected:
            int_type underflow() override {
                if (failed_) {
                    return traits_type::eof();
                }
                if (gptr() < egptr()) {
                    return traits_type::to_int_type(*gptr());
                }
                sync_position();
                if (position_ >= size_) {
                    return traits_type::eof();
                }
                // Drop the get area before cache mutation so evicted records
                // cannot leave dangling eback/gptr pointers.
                setg(nullptr, nullptr, nullptr);
                const bool ready =
                    byteshuffle_ ? prepare_shuffle_window() : prepare_zstd_window();
                if (!ready) {
                    failed_ = true;
                    return traits_type::eof();
                }
                if (gptr() < egptr()) {
                    return traits_type::to_int_type(*gptr());
                }
                failed_ = true;
                return traits_type::eof();
            }

            std::streamsize xsgetn(char* s, std::streamsize count) override {
                if (failed_ || s == nullptr || count <= 0) {
                    return 0;
                }
                std::streamsize delivered = 0;
                if (gptr() < egptr()) {
                    // Leftover is window-relative (prepare_* caps the get area
                    // at kMaxGetAreaBytes), so gbump's int count cannot overflow.
                    const auto available =
                        static_cast<std::streamsize>(egptr() - gptr());
                    const auto take = std::min(available, count);
                    std::memcpy(s, gptr(), static_cast<std::size_t>(take));
                    gbump(static_cast<int>(take));
                    delivered += take;
                    if (delivered >= count) {
                        return delivered;
                    }
                }
                sync_position();
                setg(nullptr, nullptr, nullptr);
                if (position_ >= size_) {
                    return delivered;
                }
                const auto want = static_cast<std::uint64_t>(count - delivered);
                const auto take = std::min(want, size_ - position_);
                if (take == 0) {
                    return delivered;
                }
                if (!ensure_table()) {
                    failed_ = true;
                    return delivered;
                }
                const std::uint64_t before = position_;
                const bool ok =
                    byteshuffle_ ? bulk_copy_shuffle(s + delivered, take)
                                 : bulk_copy_zstd(s + delivered, take);
                delivered += static_cast<std::streamsize>(position_ - before);
                if (!ok) {
                    failed_ = true;
                }
                prune_cache_after_bulk();
                return delivered;
            }

            pos_type seekoff(const off_type offset, const std::ios_base::seekdir direction,
                             const std::ios_base::openmode mode) override {
                if ((mode & std::ios_base::in) == 0) {
                    return pos_type(off_type(-1));
                }
                sync_position();
                std::int64_t base = 0;
                if (direction == std::ios_base::beg) {
                    base = 0;
                } else if (direction == std::ios_base::cur) {
                    if (position_ > static_cast<std::uint64_t>(
                                        std::numeric_limits<std::int64_t>::max())) {
                        return pos_type(off_type(-1));
                    }
                    base = static_cast<std::int64_t>(position_);
                } else if (direction == std::ios_base::end) {
                    if (size_ > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max())) {
                        return pos_type(off_type(-1));
                    }
                    base = static_cast<std::int64_t>(size_);
                } else {
                    return pos_type(off_type(-1));
                }
                if (offset > 0 &&
                    base > std::numeric_limits<std::int64_t>::max() - offset) {
                    return pos_type(off_type(-1));
                }
                if (offset < 0 &&
                    base < std::numeric_limits<std::int64_t>::min() - offset) {
                    return pos_type(off_type(-1));
                }
                const std::int64_t target = base + offset;
                if (target < 0 || static_cast<std::uint64_t>(target) > size_) {
                    return pos_type(off_type(-1));
                }
                position_ = static_cast<std::uint64_t>(target);
                buffer_start_ = position_;
                setg(nullptr, nullptr, nullptr);
                return pos_type(static_cast<off_type>(position_));
            }

            pos_type seekpos(const pos_type position,
                             const std::ios_base::openmode mode) override {
                return seekoff(static_cast<off_type>(position), std::ios_base::beg,
                               mode);
            }

        private:
            static constexpr std::size_t kNpos = static_cast<std::size_t>(-1);
            static constexpr std::size_t kShuffleWindowBytes = 8ull * 1024 * 1024;
            // Get-area windows stay at most INT_MAX bytes because MSVC stores
            // the get-area length as int.
            static constexpr std::size_t kMaxGetAreaBytes =
                static_cast<std::size_t>(std::numeric_limits<int>::max());
            // Bulk decode cap: 8 records × 64MiB target ≈ 512MiB decoded
            // resident at once beyond the caller's destination buffer.
            static constexpr std::size_t kMaxResidentDecodeRecords = 8;

            struct FramedRecord {
                std::uint64_t so = 0;
                std::uint64_t sb = 0;
                std::uint64_t uo = 0;
                std::uint64_t ub = 0;
            };

            struct CachedRecord {
                std::size_t index = kNpos;
                std::vector<char> bytes;
            };

            void sync_position() noexcept {
                if (eback() != nullptr && gptr() != nullptr) {
                    position_ =
                        buffer_start_ + static_cast<std::uint64_t>(gptr() - eback());
                }
            }

            void account_payload_bytes(const std::uint64_t count) const {
                if (state_ && state_->options.payload_bytes_read) {
                    state_->options.payload_bytes_read->fetch_add(
                        count, std::memory_order_relaxed);
                }
            }

            bool read_stored_locked(const std::uint64_t relative,
                                    const std::span<std::byte> destination) {
                if (!state_ || destination.empty()) {
                    return destination.empty();
                }
                if (auto read = state_->file->read_exact(
                        row_.payload_offset + relative, destination);
                    !read) {
                    return false;
                }
                account_payload_bytes(destination.size());
                return true;
            }

            bool ensure_blocks_locked(const std::uint64_t relative,
                                      const std::uint64_t length) {
                if (!row_.block_crc_table.has_value() || length == 0) {
                    return true;
                }
                const BlockCrcTable& table = *row_.block_crc_table;
                if (table.block_size == 0 || relative >= row_.stored_bytes ||
                    length > row_.stored_bytes - relative) {
                    return false;
                }
                const std::uint64_t first = relative / table.block_size;
                const std::uint64_t last = (relative + length - 1) / table.block_size;
                for (std::uint64_t block_index = first; block_index <= last;
                     ++block_index) {
                    const auto index = static_cast<std::size_t>(block_index);
                    if (index < block_validated_.size() &&
                        block_validated_[index] != 0) {
                        continue;
                    }
                    if (index >= table.entries.size()) {
                        return false;
                    }
                    const std::uint64_t block_offset = block_index * table.block_size;
                    const std::uint64_t block_bytes = std::min<std::uint64_t>(
                        table.block_size, row_.stored_bytes - block_offset);
                    if (auto verified =
                            verify_block_range(*state_, row_, block_offset, block_bytes);
                        !verified) {
                        return false;
                    }
                    if (index < block_validated_.size()) {
                        block_validated_[index] = 1;
                    }
                }
                return true;
            }

            bool ensure_table() {
                if (table_ready_) {
                    return true;
                }
                if (size_ == 0) {
                    table_ready_ = true;
                    return true;
                }
                std::scoped_lock lock(io_mutex_);
                if (table_ready_) {
                    return true;
                }
                if (row_.stored_bytes < detail::FRAMED_HEADER_BYTES) {
                    return false;
                }
                if (!ensure_blocks_locked(0, detail::FRAMED_HEADER_BYTES)) {
                    return false;
                }
                std::array<std::byte, detail::FRAMED_HEADER_BYTES> header{};
                if (!read_stored_locked(0, byte_span(header))) {
                    return false;
                }
                if (!std::equal(detail::FRAMED_MAGIC.begin(), detail::FRAMED_MAGIC.end(),
                                header.begin())) {
                    return false;
                }
                const auto count = read_u32(byte_span(header), 12);
                if (count > (std::numeric_limits<std::size_t>::max() -
                             detail::FRAMED_HEADER_BYTES) /
                                detail::FRAMED_RECORD_BYTES) {
                    return false;
                }
                const auto table = detail::FRAMED_HEADER_BYTES +
                                   static_cast<std::size_t>(count) *
                                       detail::FRAMED_RECORD_BYTES;
                if (read_u16(byte_span(header), 8) != detail::FRAMED_VERSION ||
                    read_u16(byte_span(header), 10) != 0 || count == 0 ||
                    table > row_.stored_bytes) {
                    return false;
                }
                if (count > size_ || row_.stored_bytes - table < count) {
                    return false;
                }
                std::vector<std::byte> table_bytes;
                try {
                    table_bytes.resize(table);
                } catch (const std::bad_alloc&) {
                    return false;
                } catch (const std::length_error&) {
                    return false;
                }
                std::memcpy(table_bytes.data(), header.data(), header.size());
                if (table > header.size()) {
                    if (!ensure_blocks_locked(0, table)) {
                        return false;
                    }
                    if (!read_stored_locked(
                            header.size(),
                            std::span<std::byte>(table_bytes.data() + header.size(),
                                                 table - header.size()))) {
                        return false;
                    }
                }
                std::vector<FramedRecord> records;
                try {
                    records.reserve(count);
                } catch (const std::bad_alloc&) {
                    return false;
                } catch (const std::length_error&) {
                    return false;
                }
                std::uint64_t so = table;
                std::uint64_t uo = 0;
                for (std::uint32_t index = 0; index < count; ++index) {
                    const auto at = detail::FRAMED_HEADER_BYTES +
                                    static_cast<std::size_t>(index) *
                                        detail::FRAMED_RECORD_BYTES;
                    const auto sb64 = read_u64(table_bytes, at);
                    const auto ub64 = read_u64(table_bytes, at + 8);
                    if (sb64 == 0 || ub64 == 0 || sb64 > row_.stored_bytes - so ||
                        ub64 > size_ - uo ||
                        sb64 > std::numeric_limits<std::size_t>::max() ||
                        ub64 > std::numeric_limits<std::size_t>::max()) {
                        return false;
                    }
                    records.push_back({so, sb64, uo, ub64});
                    so += sb64;
                    uo += ub64;
                }
                if (so != row_.stored_bytes || uo != size_) {
                    return false;
                }
                records_ = std::move(records);
                table_ready_ = true;
                return true;
            }

            [[nodiscard]] std::size_t
            record_index_containing(const std::uint64_t decoded_pos) const {
                if (records_.empty() || decoded_pos >= size_) {
                    return kNpos;
                }
                const auto iterator = std::upper_bound(
                    records_.begin(), records_.end(), decoded_pos,
                    [](const std::uint64_t pos, const FramedRecord& record) {
                        return pos < record.uo;
                    });
                if (iterator == records_.begin()) {
                    return kNpos;
                }
                const auto index = static_cast<std::size_t>(
                    std::distance(records_.begin(), iterator) - 1);
                const auto& record = records_[index];
                if (decoded_pos < record.uo ||
                    decoded_pos >= record.uo + record.ub) {
                    return kNpos;
                }
                return index;
            }

            [[nodiscard]] CachedRecord* find_cached(const std::size_t index) {
                for (auto& cached : cache_) {
                    if (cached.index == index) {
                        return &cached;
                    }
                }
                return nullptr;
            }

            [[nodiscard]] CachedRecord* find_ready_cached(const std::size_t index) {
                if (index >= records_.size()) {
                    return nullptr;
                }
                CachedRecord* cached = find_cached(index);
                if (cached == nullptr || cached->bytes.size() != records_[index].ub) {
                    return nullptr;
                }
                return cached;
            }

            void copy_cached_zstd_range(char* dest, const std::uint64_t start,
                                        const std::uint64_t end,
                                        const FramedRecord& record,
                                        const CachedRecord& cached) const {
                const std::uint64_t lo = std::max(record.uo, start);
                const std::uint64_t hi = std::min(record.uo + record.ub, end);
                if (lo >= hi) {
                    return;
                }
                std::memcpy(dest + static_cast<std::size_t>(lo - start),
                            cached.bytes.data() +
                                static_cast<std::size_t>(lo - record.uo),
                            static_cast<std::size_t>(hi - lo));
            }

            bool prefetch_stored_records(const std::vector<std::size_t>& indices) {
                if (indices.empty()) {
                    return true;
                }
                if (stored_cache_.size() != records_.size()) {
                    try {
                        stored_cache_.assign(records_.size(), {});
                    } catch (const std::bad_alloc&) {
                        return false;
                    } catch (const std::length_error&) {
                        return false;
                    }
                }
                for (const auto index : indices) {
                    if (index >= records_.size()) {
                        return false;
                    }
                    if (!stored_cache_[index].empty()) {
                        continue;
                    }
                    if (!read_record_stored(index, stored_cache_[index])) {
                        stored_cache_[index].clear();
                        break;
                    }
                }
                return true;
            }

            bool read_record_stored(const std::size_t index,
                                    std::vector<std::byte>& stored) {
                if (index >= records_.size()) {
                    return false;
                }
                if (index < stored_cache_.size() &&
                    !stored_cache_[index].empty()) {
                    stored = stored_cache_[index];
                    return true;
                }
                const FramedRecord& record = records_[index];
                try {
                    stored.resize(static_cast<std::size_t>(record.sb));
                } catch (const std::bad_alloc&) {
                    return false;
                } catch (const std::length_error&) {
                    return false;
                }
                std::scoped_lock lock(io_mutex_);
                if (!ensure_blocks_locked(record.so, record.sb)) {
                    return false;
                }
                return read_stored_locked(
                    record.so, std::span<std::byte>(stored.data(), stored.size()));
            }

            bool decompress_record(const FramedRecord& record,
                                   const std::vector<std::byte>& stored, void* dest,
                                   const std::size_t dest_bytes) const {
                if (dest == nullptr || dest_bytes != record.ub) {
                    return false;
                }
                const auto frame_size =
                    ZSTD_getFrameContentSize(stored.data(), stored.size());
                if (frame_size != record.ub) {
                    return false;
                }
                g_framed_record_decode_calls.fetch_add(1, std::memory_order_relaxed);
                const auto result = ZSTD_decompress(dest, dest_bytes, stored.data(),
                                                    stored.size());
                return !ZSTD_isError(result) && result == record.ub;
            }

            bool decode_record(const std::size_t index, std::vector<char>& decoded) {
                if (index >= records_.size()) {
                    return false;
                }
                std::vector<std::byte> stored;
                if (!read_record_stored(index, stored)) {
                    return false;
                }
                try {
                    decoded.resize(static_cast<std::size_t>(records_[index].ub));
                } catch (const std::bad_alloc&) {
                    return false;
                } catch (const std::length_error&) {
                    return false;
                }
                return decompress_record(records_[index], stored, decoded.data(),
                                         decoded.size());
            }

            bool decode_record_direct(const std::size_t index, void* dest,
                                      const std::size_t dest_bytes) {
                if (index >= records_.size()) {
                    return false;
                }
                std::vector<std::byte> stored;
                if (!read_record_stored(index, stored)) {
                    return false;
                }
                return decompress_record(records_[index], stored, dest, dest_bytes);
            }

            struct DecodeItem {
                std::size_t index = kNpos;
                char* direct = nullptr;
            };

            bool decode_items(const std::vector<DecodeItem>& items,
                              std::vector<std::uint8_t>& ok) {
                if (items.empty()) {
                    return true;
                }
                ok.assign(items.size(), std::uint8_t{1});
                std::atomic<std::size_t> next{0};
                std::mutex cache_mutex;
                const auto run_one = [&](const std::size_t slot) {
                    const DecodeItem& item = items[slot];
                    if (item.index >= records_.size()) {
                        ok[slot] = 0;
                        return;
                    }
                    if (item.direct != nullptr) {
                        if (!decode_record_direct(
                                item.index, item.direct,
                                static_cast<std::size_t>(records_[item.index].ub))) {
                            ok[slot] = 0;
                        }
                        return;
                    }
                    CachedRecord decoded;
                    decoded.index = item.index;
                    if (!decode_record(item.index, decoded.bytes)) {
                        ok[slot] = 0;
                        return;
                    }
                    std::scoped_lock lock(cache_mutex);
                    if (find_cached(item.index) == nullptr) {
                        cache_.push_back(std::move(decoded));
                    }
                };
                if (items.size() == 1) {
                    try {
                        run_one(0);
                    } catch (...) {
                        ok[0] = 0;
                    }
                    return ok[0] != 0;
                }
                const auto workers_count = std::min<std::size_t>(
                    items.size(),
                    std::max(1u, std::thread::hardware_concurrency()));
                std::vector<std::jthread> workers;
                try {
                    workers.reserve(workers_count);
                    for (std::size_t worker = 0; worker < workers_count; ++worker) {
                        workers.emplace_back([&] {
                            while (true) {
                                const auto slot = next.fetch_add(1);
                                if (slot >= items.size()) {
                                    return;
                                }
                                try {
                                    run_one(slot);
                                } catch (...) {
                                    ok[slot] = 0;
                                }
                            }
                        });
                    }
                } catch (...) {
                    ok.assign(items.size(), std::uint8_t{0});
                }
                workers.clear();
                return std::find(ok.begin(), ok.end(), std::uint8_t{0}) == ok.end();
            }

            static std::uint64_t first_plane_word(const std::uint64_t plane,
                                                  const std::uint64_t lo,
                                                  const std::uint64_t hi,
                                                  const std::uint64_t n_words) noexcept {
                if (lo >= hi || plane >= 4 || n_words == 0) {
                    return kNpos;
                }
                const std::uint64_t first =
                    lo / 4 + (plane < (lo % 4) ? 1 : 0);
                const std::uint64_t last_byte = hi - 1;
                const std::uint64_t last_word = last_byte / 4;
                const std::uint64_t plane_last =
                    plane <= (last_byte % 4)
                        ? last_word
                        : (last_word == 0 ? static_cast<std::uint64_t>(kNpos)
                                          : last_word - 1);
                if (plane_last == kNpos || first > plane_last || first >= n_words) {
                    return kNpos;
                }
                return first;
            }

            static std::uint64_t last_plane_word(const std::uint64_t plane,
                                                 const std::uint64_t lo,
                                                 const std::uint64_t hi,
                                                 const std::uint64_t n_words) noexcept {
                if (first_plane_word(plane, lo, hi, n_words) == kNpos) {
                    return kNpos;
                }
                const std::uint64_t last_byte = hi - 1;
                const std::uint64_t last_word = last_byte / 4;
                const std::uint64_t plane_last =
                    plane <= (last_byte % 4) ? last_word : last_word - 1;
                return std::min(plane_last, n_words - 1);
            }

            [[nodiscard]] bool
            shuffle_record_covers(const std::size_t index, const std::uint64_t lo,
                                  const std::uint64_t hi) const {
                if (index >= records_.size() || lo >= hi || size_ % 4 != 0) {
                    return false;
                }
                const std::uint64_t n_words = size_ / 4;
                const FramedRecord& record = records_[index];
                for (std::uint64_t plane = 0; plane < 4; ++plane) {
                    const std::uint64_t word =
                        first_plane_word(plane, lo, hi, n_words);
                    const std::uint64_t last =
                        last_plane_word(plane, lo, hi, n_words);
                    if (word == kNpos || last == kNpos) {
                        continue;
                    }
                    const std::uint64_t dec_lo = plane * n_words + word;
                    const std::uint64_t dec_hi = plane * n_words + last + 1;
                    if (record.uo < dec_hi && record.uo + record.ub > dec_lo) {
                        return true;
                    }
                }
                return false;
            }

            void retain_cache(const std::vector<std::size_t>& keep) {
                cache_.erase(std::remove_if(cache_.begin(), cache_.end(),
                                            [&](const CachedRecord& cached) {
                                                return std::find(keep.begin(),
                                                                 keep.end(),
                                                                 cached.index) ==
                                                       keep.end();
                                            }),
                             cache_.end());
            }

            void append_shuffle_boundary_records(const std::uint64_t pos,
                                                 std::vector<std::size_t>& keep) const {
                if (pos >= size_ || size_ % 4 != 0) {
                    return;
                }
                const std::uint64_t n_words = size_ / 4;
                for (std::uint64_t plane = 0; plane < 4; ++plane) {
                    const std::uint64_t word =
                        first_plane_word(plane, pos, size_, n_words);
                    if (word == kNpos) {
                        continue;
                    }
                    const auto index =
                        record_index_containing(plane * n_words + word);
                    if (index != kNpos &&
                        std::find(keep.begin(), keep.end(), index) == keep.end()) {
                        keep.push_back(index);
                    }
                }
            }

            void prune_cache_after_bulk() {
                if (position_ >= size_ || !table_ready_) {
                    cache_.clear();
                    return;
                }
                std::vector<std::size_t> keep;
                if (byteshuffle_) {
                    if (size_ % 4 != 0) {
                        cache_.clear();
                        return;
                    }
                    append_shuffle_boundary_records(position_, keep);
                } else {
                    const auto index = record_index_containing(position_);
                    if (index != kNpos) {
                        keep.push_back(index);
                        if (index + 1 < records_.size()) {
                            keep.push_back(index + 1);
                        }
                    }
                }
                retain_cache(keep);
            }

            bool bulk_copy_zstd(char* dest, const std::uint64_t count) {
                const std::uint64_t start = position_;
                const std::uint64_t end = start + count;
                if (count == 0) {
                    return true;
                }
                const auto first = record_index_containing(start);
                const auto last = record_index_containing(end - 1);
                if (first == kNpos || last == kNpos || last < first) {
                    return false;
                }
                std::vector<std::size_t> needed;
                needed.reserve(last - first + 1);
                for (std::size_t index = first; index <= last; ++index) {
                    needed.push_back(index);
                }
                if (!prefetch_stored_records(needed)) {
                    return false;
                }
                for (std::size_t batch_begin = first; batch_begin <= last;
                     batch_begin += kMaxResidentDecodeRecords) {
                    const std::size_t batch_end = std::min(
                        batch_begin + kMaxResidentDecodeRecords, last + 1);
                    std::vector<DecodeItem> items;
                    items.reserve(batch_end - batch_begin);
                    for (std::size_t index = batch_begin; index < batch_end;
                         ++index) {
                        const FramedRecord& record = records_[index];
                        // Cache hits must bypass decode.
                        const CachedRecord* cached = find_ready_cached(index);
                        if (cached != nullptr) {
                            copy_cached_zstd_range(dest, start, end, record,
                                                   *cached);
                            continue;
                        }
                        const bool full = record.uo >= start &&
                                          record.uo + record.ub <= end;
                        DecodeItem item;
                        item.index = index;
                        if (full) {
                            item.direct =
                                dest + static_cast<std::size_t>(record.uo - start);
                        }
                        items.push_back(item);
                    }
                    std::vector<std::uint8_t> ok;
                    decode_items(items, ok);
                    for (std::size_t slot = 0; slot < items.size(); ++slot) {
                        const std::size_t index = items[slot].index;
                        const FramedRecord& record = records_[index];
                        if (slot >= ok.size() || ok[slot] == 0) {
                            position_ = std::max(start, record.uo);
                            return false;
                        }
                        if (items[slot].direct != nullptr) {
                            continue;
                        }
                        const CachedRecord* cached = find_cached(index);
                        if (cached == nullptr) {
                            position_ = std::max(start, record.uo);
                            return false;
                        }
                        copy_cached_zstd_range(dest, start, end, record, *cached);
                    }
                }
                position_ = end;
                return true;
            }

            bool bulk_copy_shuffle(char* dest, const std::uint64_t count) {
                if (size_ % 4 != 0) {
                    return false;
                }
                const std::uint64_t n_words = size_ / 4;
                const std::uint64_t start = position_;
                const std::uint64_t end = start + count;
                if (count == 0) {
                    return true;
                }
                std::vector<std::size_t> needed;
                needed.reserve(records_.size());
                for (std::size_t index = 0; index < records_.size(); ++index) {
                    if (shuffle_record_covers(index, start, end)) {
                        needed.push_back(index);
                    }
                }
                if (!prefetch_stored_records(needed)) {
                    return false;
                }
                // 0 = unknown, 1 = ready, 2 = failed
                std::vector<std::uint8_t> status(records_.size(), std::uint8_t{0});
                for (const auto& cached : cache_) {
                    if (cached.index < status.size()) {
                        status[cached.index] = 1;
                    }
                }
                const auto add_unique = [](std::vector<std::size_t>& batch,
                                           const std::size_t index) {
                    if (index == kNpos) {
                        return;
                    }
                    if (std::find(batch.begin(), batch.end(), index) !=
                        batch.end()) {
                        return;
                    }
                    if (batch.size() >= kMaxResidentDecodeRecords) {
                        return;
                    }
                    batch.push_back(index);
                };

                std::uint64_t logical = start;
                while (logical < end) {
                    std::vector<std::size_t> batch;
                    for (std::uint64_t plane = 0; plane < 4; ++plane) {
                        const std::uint64_t word =
                            first_plane_word(plane, logical, end, n_words);
                        if (word == kNpos) {
                            continue;
                        }
                        add_unique(batch,
                                   record_index_containing(plane * n_words + word));
                    }
                    for (std::uint64_t plane = 0; plane < 4; ++plane) {
                        const std::uint64_t word =
                            first_plane_word(plane, logical, end, n_words);
                        if (word == kNpos) {
                            continue;
                        }
                        const auto index =
                            record_index_containing(plane * n_words + word);
                        if (index == kNpos || index + 1 >= records_.size()) {
                            continue;
                        }
                        if (shuffle_record_covers(index + 1, logical, end)) {
                            add_unique(batch, index + 1);
                        }
                    }

                    std::vector<DecodeItem> items;
                    for (const auto index : batch) {
                        if (index >= status.size()) {
                            continue;
                        }
                        if (status[index] == 2) {
                            continue;
                        }
                        // Cache hits must bypass decode.
                        if (find_ready_cached(index) != nullptr) {
                            status[index] = 1;
                            continue;
                        }
                        items.push_back(DecodeItem{index, nullptr});
                    }
                    std::vector<std::uint8_t> ok;
                    decode_items(items, ok);
                    for (std::size_t slot = 0; slot < items.size(); ++slot) {
                        const auto index = items[slot].index;
                        if (index >= status.size()) {
                            continue;
                        }
                        status[index] =
                            (slot < ok.size() && ok[slot] != 0) ? 1 : 2;
                    }
                    for (const auto& cached : cache_) {
                        if (cached.index < status.size() &&
                            status[cached.index] != 2) {
                            status[cached.index] = 1;
                        }
                    }

                    std::uint64_t stripe_end = end;
                    for (std::uint64_t plane = 0; plane < 4; ++plane) {
                        const std::uint64_t word =
                            first_plane_word(plane, logical, end, n_words);
                        if (word == kNpos) {
                            continue;
                        }
                        const auto index =
                            record_index_containing(plane * n_words + word);
                        const std::uint64_t first_logical = word * 4 + plane;
                        if (index == kNpos || index >= status.size() ||
                            status[index] != 1) {
                            stripe_end = std::min(stripe_end, first_logical);
                            continue;
                        }
                        const FramedRecord& record = records_[index];
                        const std::uint64_t plane_base = plane * n_words;
                        const std::uint64_t overlap_hi = std::min(
                            record.uo + record.ub, plane_base + n_words);
                        if (overlap_hi <= plane_base + word) {
                            stripe_end = std::min(stripe_end, first_logical);
                            continue;
                        }
                        const std::uint64_t last_word = overlap_hi - plane_base - 1;
                        const std::uint64_t record_end = last_word * 4 + plane + 1;
                        stripe_end = std::min(stripe_end, record_end);
                    }

                    if (stripe_end <= logical) {
                        position_ = logical;
                        return false;
                    }

                    std::array<const char*, 4> cursors{};
                    for (std::uint64_t plane = 0; plane < 4; ++plane) {
                        const std::uint64_t word =
                            first_plane_word(plane, logical, stripe_end, n_words);
                        if (word == kNpos) {
                            continue;
                        }
                        const auto index =
                            record_index_containing(plane * n_words + word);
                        const CachedRecord* cached =
                            index == kNpos ? nullptr : find_cached(index);
                        if (cached == nullptr) {
                            position_ = logical;
                            return false;
                        }
                        const std::uint64_t decoded = plane * n_words + word;
                        const std::uint64_t local = decoded - records_[index].uo;
                        if (local >= cached->bytes.size()) {
                            position_ = logical;
                            return false;
                        }
                        cursors[static_cast<std::size_t>(plane)] =
                            cached->bytes.data() + static_cast<std::size_t>(local);
                    }
                    gather_unshuffle_words(
                        dest + static_cast<std::size_t>(logical - start), logical,
                        static_cast<std::size_t>(stripe_end - logical), cursors);
                    logical = stripe_end;

                    if (logical < end) {
                        std::vector<std::size_t> keep;
                        append_shuffle_boundary_records(end, keep);
                        cache_.erase(
                            std::remove_if(
                                cache_.begin(), cache_.end(),
                                [&](const CachedRecord& cached) {
                                    if (shuffle_record_covers(cached.index,
                                                              logical, end)) {
                                        return false;
                                    }
                                    return std::find(keep.begin(), keep.end(),
                                                     cached.index) == keep.end();
                                }),
                            cache_.end());
                    }
                }
                position_ = end;
                return true;
            }

            bool ensure_decoded(const std::vector<std::size_t>& needed) {
                std::vector<std::size_t> missing;
                missing.reserve(needed.size());
                for (const auto index : needed) {
                    if (index >= records_.size()) {
                        return false;
                    }
                    if (find_cached(index) == nullptr) {
                        missing.push_back(index);
                    }
                }
                if (!missing.empty()) {
                    std::atomic<std::size_t> next{0};
                    std::atomic<bool> ok{true};
                    std::mutex cache_mutex;
                    const auto decode_one = [&](const std::size_t index) {
                        CachedRecord decoded;
                        decoded.index = index;
                        if (!decode_record(index, decoded.bytes)) {
                            ok.store(false, std::memory_order_relaxed);
                            return;
                        }
                        std::scoped_lock lock(cache_mutex);
                        cache_.push_back(std::move(decoded));
                    };
                    if (missing.size() == 1) {
                        decode_one(missing.front());
                    } else {
                        const auto workers_count = std::min<std::size_t>(
                            missing.size(),
                            std::max(1u, std::thread::hardware_concurrency()));
                        std::vector<std::jthread> workers;
                        try {
                            workers.reserve(workers_count);
                            for (std::size_t worker = 0; worker < workers_count;
                                 ++worker) {
                                workers.emplace_back([&] {
                                    try {
                                        while (ok.load(std::memory_order_relaxed)) {
                                            const auto which = next.fetch_add(1);
                                            if (which >= missing.size()) {
                                                return;
                                            }
                                            decode_one(missing[which]);
                                        }
                                    } catch (...) {
                                        ok.store(false, std::memory_order_relaxed);
                                    }
                                });
                            }
                        } catch (...) {
                            ok.store(false, std::memory_order_relaxed);
                        }
                        workers.clear();
                    }
                    if (!ok.load(std::memory_order_relaxed)) {
                        return false;
                    }
                }
                cache_.erase(std::remove_if(cache_.begin(), cache_.end(),
                                            [&](const CachedRecord& cached) {
                                                return std::find(needed.begin(),
                                                                 needed.end(),
                                                                 cached.index) ==
                                                       needed.end();
                                            }),
                             cache_.end());
                return true;
            }

            bool prepare_zstd_window() {
                if (!ensure_table()) {
                    return false;
                }
                const auto index = record_index_containing(position_);
                if (index == kNpos) {
                    return false;
                }
                std::vector<std::size_t> needed;
                needed.push_back(index);
                if (index + 1 < records_.size()) {
                    needed.push_back(index + 1);
                }
                if (!ensure_decoded(needed)) {
                    return false;
                }
                auto* cached = find_cached(index);
                if (cached == nullptr ||
                    cached->bytes.size() != records_[index].ub) {
                    return false;
                }
                const auto frame_off = position_ - records_[index].uo;
                if (frame_off >=
                    static_cast<std::uint64_t>(cached->bytes.size())) {
                    return false;
                }
                const auto remaining =
                    static_cast<std::uint64_t>(cached->bytes.size()) - frame_off;
                const auto count = std::min(
                    remaining, static_cast<std::uint64_t>(kMaxGetAreaBytes));
                buffer_start_ = position_;
                char* const first =
                    cached->bytes.data() + static_cast<std::size_t>(frame_off);
                setg(first, first, first + static_cast<std::ptrdiff_t>(count));
                return true;
            }

            bool prepare_shuffle_window() {
                if (!ensure_table()) {
                    return false;
                }
                if (size_ % 4 != 0) {
                    return false;
                }
                const std::uint64_t n_words = size_ / 4;
                const std::uint64_t start_word = position_ / 4;
                const std::uint64_t start_plane = position_ % 4;
                std::array<std::size_t, 4> plane_record{};
                plane_record.fill(kNpos);
                std::vector<std::size_t> needed;
                needed.reserve(4);
                const auto add_needed = [&](const std::size_t index) {
                    if (index == kNpos) {
                        return false;
                    }
                    if (std::find(needed.begin(), needed.end(), index) ==
                        needed.end()) {
                        needed.push_back(index);
                    }
                    return true;
                };
                for (std::uint64_t plane = 0; plane < 4; ++plane) {
                    const std::uint64_t word =
                        plane >= start_plane ? start_word : start_word + 1;
                    if (word >= n_words) {
                        continue;
                    }
                    const std::uint64_t decoded = plane * n_words + word;
                    const auto index = record_index_containing(decoded);
                    plane_record[static_cast<std::size_t>(plane)] = index;
                    if (!add_needed(index)) {
                        return false;
                    }
                }
                if (needed.empty() || !ensure_decoded(needed)) {
                    return false;
                }

                std::uint64_t window_end = std::min(size_, position_ + window_.size());
                for (std::uint64_t plane = 0; plane < 4; ++plane) {
                    const auto index =
                        plane_record[static_cast<std::size_t>(plane)];
                    if (index == kNpos) {
                        continue;
                    }
                    const auto& record = records_[index];
                    const std::uint64_t plane_base = plane * n_words;
                    const std::uint64_t overlap_lo = std::max(record.uo, plane_base);
                    const std::uint64_t overlap_hi =
                        std::min(record.uo + record.ub, plane_base + n_words);
                    if (overlap_hi <= overlap_lo) {
                        return false;
                    }
                    const std::uint64_t last_word = overlap_hi - plane_base - 1;
                    const std::uint64_t plane_end = last_word * 4 + plane + 1;
                    window_end = std::min(window_end, plane_end);
                }
                if (window_end <= position_) {
                    return false;
                }

                std::array<const CachedRecord*, 4> plane_cache{};
                for (std::uint64_t plane = 0; plane < 4; ++plane) {
                    const auto index =
                        plane_record[static_cast<std::size_t>(plane)];
                    if (index == kNpos) {
                        continue;
                    }
                    plane_cache[static_cast<std::size_t>(plane)] = find_cached(index);
                    if (plane_cache[static_cast<std::size_t>(plane)] == nullptr) {
                        return false;
                    }
                }

                const auto count = static_cast<std::size_t>(window_end - position_);
                std::array<const char*, 4> cursors{};
                for (std::uint64_t plane = 0; plane < 4; ++plane) {
                    const auto index =
                        plane_record[static_cast<std::size_t>(plane)];
                    if (index == kNpos) {
                        continue;
                    }
                    const auto* cached = plane_cache[static_cast<std::size_t>(plane)];
                    const std::uint64_t word =
                        plane >= start_plane ? start_word : start_word + 1;
                    const std::uint64_t decoded = plane * n_words + word;
                    const std::uint64_t local = decoded - records_[index].uo;
                    if (cached == nullptr || local >= cached->bytes.size()) {
                        return false;
                    }
                    cursors[static_cast<std::size_t>(plane)] =
                        cached->bytes.data() + static_cast<std::size_t>(local);
                }
                gather_unshuffle_words(window_.data(), position_, count, cursors);
                buffer_start_ = position_;
                setg(window_.data(), window_.data(), window_.data() + count);
                return true;
            }

            std::shared_ptr<ReaderState> state_;
            ChunkInfo row_;
            bool byteshuffle_ = false;
            bool failed_ = false;
            bool table_ready_ = false;
            std::uint64_t size_ = 0;
            std::uint64_t position_ = 0;
            std::uint64_t buffer_start_ = 0;
            std::vector<FramedRecord> records_;
            std::vector<std::uint8_t> block_validated_;
            std::vector<CachedRecord> cache_;
            std::vector<std::vector<std::byte>> stored_cache_;
            std::vector<char> window_;
            std::mutex io_mutex_;
        };

    } // namespace

    namespace detail {
        void reset_framed_record_decode_calls_for_testing() {
            g_framed_record_decode_calls.store(0, std::memory_order_relaxed);
        }

        std::uint64_t framed_record_decode_calls_for_testing() {
            return g_framed_record_decode_calls.load(std::memory_order_relaxed);
        }
    } // namespace detail

    struct BoundedInputStream::Impl {
        Impl(std::unique_ptr<std::streambuf> owned_buffer,
             const std::uint64_t size)
            : buffer(std::move(owned_buffer)),
              input(buffer.get()),
              size_bytes(size) {}

        std::unique_ptr<std::streambuf> buffer;
        std::istream input;
        std::uint64_t size_bytes = 0;
    };

    BoundedInputStream::BoundedInputStream(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    BoundedInputStream::BoundedInputStream(BoundedInputStream&&) noexcept = default;
    BoundedInputStream&
    BoundedInputStream::operator=(BoundedInputStream&&) noexcept = default;
    BoundedInputStream::~BoundedInputStream() = default;

    std::istream& BoundedInputStream::stream() {
        return impl_->input;
    }

    std::uint64_t BoundedInputStream::size() const noexcept {
        return impl_->size_bytes;
    }

    lfs::Result<BoundedInputStream>
    ProjectReader::open_bounded_stream(const ChunkInfo& chunk) const {
        if (auto allowed = require_supported_payload_access(*this); !allowed) {
            return std::move(allowed).error();
        }
        auto resolved = resolve_chunk(*impl_->state, chunk);
        if (!resolved) {
            return std::move(resolved).error();
        }
        const ChunkInfo& row = **resolved;
        const bool framed = row.compression == Compression::ZstdFramed ||
                            row.compression == Compression::ByteShuffleZstdFramed;
        if (row.compression != Compression::Stored && !framed) {
            return detail::project_error(
                lfs::ErrorCode::FailedPrecondition,
                "A random-access stream requires a stored project chunk.",
                std::format("{} is compressed (not STORED)",
                            row.key_string()),
                path(), row.payload_offset,
                "bounded_stream.compression");
        }
        if (row.compression == Compression::ByteShuffleZstdFramed &&
            row.uncompressed_bytes % 4 != 0) {
            return format_error(
                path(), row.payload_offset, "payload.byteshuffle",
                "decoded size multiple of 4",
                std::to_string(row.uncompressed_bytes));
        }
        if (!row.block_crc_table.has_value()) {
            // Framed: CRC the stored range in BLOCK_CRC_BYTES windows. Do not
            // decode — ensure_table validates the record table on first use,
            // and decompress_record checks ZSTD_getFrameContentSize == ub and
            // decode size == ub when that record is read.
            // Stored: verify_chunk does not decode and stays as-is.
            if (framed) {
                if (auto verified =
                        verify_stored_payload_crc32c(*impl_->state, row);
                    !verified) {
                    return std::move(verified).error();
                }
            } else if (auto verified = verify_chunk(row); !verified) {
                return std::move(verified).error();
            }
        }
        if (framed) {
            return BoundedInputStream(std::make_unique<BoundedInputStream::Impl>(
                std::make_unique<FramedLogicalStreambuf>(
                    impl_->state, row,
                    row.compression == Compression::ByteShuffleZstdFramed),
                row.uncompressed_bytes));
        }
        return BoundedInputStream(std::make_unique<BoundedInputStream::Impl>(
            std::make_unique<PositionalStreambuf>(
                impl_->state->file, row.payload_offset, row.stored_bytes,
                row.block_crc_table),
            row.stored_bytes));
    }

    struct MappedRegion::Impl {
        std::shared_ptr<detail::NativeFile> file;
        void* mapped_base = nullptr;
        std::size_t mapped_bytes = 0;
        const std::byte* requested_data = nullptr;
        std::size_t requested_bytes = 0;
        std::uint64_t requested_offset = 0;
#ifdef _WIN32
        HANDLE mapping = nullptr;
#endif

        ~Impl() {
#ifdef _WIN32
            if (mapped_base != nullptr) {
                UnmapViewOfFile(mapped_base);
            }
            if (mapping != nullptr) {
                CloseHandle(mapping);
            }
#else
            if (mapped_base != nullptr) {
                ::munmap(mapped_base, mapped_bytes);
            }
#endif
        }
    };

    MappedRegion::MappedRegion(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    MappedRegion::MappedRegion(MappedRegion&&) noexcept = default;
    MappedRegion& MappedRegion::operator=(MappedRegion&&) noexcept = default;
    MappedRegion::~MappedRegion() = default;

    std::span<const std::byte> MappedRegion::bytes() const noexcept {
        return {impl_->requested_data, impl_->requested_bytes};
    }

    std::uint64_t MappedRegion::file_offset() const noexcept {
        return impl_->requested_offset;
    }

    lfs::Result<MappedRegion>
    ProjectReader::map_stored_range(const ChunkInfo& chunk,
                                    const std::uint64_t relative_offset,
                                    const std::uint64_t length) const {
        if (auto allowed = require_supported_payload_access(*this); !allowed) {
            return std::move(allowed).error();
        }
        auto resolved = resolve_chunk(*impl_->state, chunk);
        if (!resolved) {
            return std::move(resolved).error();
        }
        const ChunkInfo& row = **resolved;
        auto range_end = detail::checked_add(relative_offset, length, path(),
                                             row.payload_offset,
                                             "mapped_range");
        if (!range_end) {
            return std::move(range_end).error();
        }
        if (*range_end > row.stored_bytes) {
            return format_error(
                path(), row.payload_offset, "mapped_range",
                std::format("end <= stored bytes {}", row.stored_bytes),
                std::to_string(*range_end), lfs::ErrorCode::BoundsViolation);
        }
        if (row.block_crc_table.has_value()) {
            if (auto verified =
                    verify_block_range(*impl_->state, row, relative_offset, length);
                !verified) {
                return std::move(verified).error();
            }
        } else if (auto verified = verify_chunk(row); !verified) {
            return std::move(verified).error();
        }

        auto mapping = std::make_unique<MappedRegion::Impl>();
        mapping->file = impl_->state->file;
        mapping->requested_offset = row.payload_offset + relative_offset;
        if (length == 0) {
            return MappedRegion(std::move(mapping));
        }
        if (length > std::numeric_limits<std::size_t>::max()) {
            return format_error(path(), mapping->requested_offset, "mapped_range",
                                "length addressable by this build",
                                std::to_string(length),
                                lfs::ErrorCode::BoundsViolation);
        }

#ifdef _WIN32
        SYSTEM_INFO system_info{};
        GetSystemInfo(&system_info);
        const std::uint64_t granularity = system_info.dwAllocationGranularity;
        const std::uint64_t mapped_offset =
            (mapping->requested_offset / granularity) * granularity;
        const std::uint64_t adjustment = mapping->requested_offset - mapped_offset;
        auto mapped_length = detail::checked_add(adjustment, length, path(),
                                                 mapping->requested_offset,
                                                 "mapped_range.length");
        if (!mapped_length ||
            *mapped_length > std::numeric_limits<SIZE_T>::max()) {
            return !mapped_length
                       ? lfs::Result<MappedRegion>(std::move(mapped_length).error())
                       : lfs::Result<MappedRegion>(format_error(
                             path(), mapping->requested_offset,
                             "mapped_range.length", "SIZE_T range",
                             std::to_string(*mapped_length),
                             lfs::ErrorCode::BoundsViolation));
        }
        mapping->mapping = CreateFileMappingW(
            mapping->file->native_handle(), nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping->mapping == nullptr) {
            const DWORD error = GetLastError();
            return detail::project_error(
                lfs::ErrorCode::Internal, "The project chunk could not be mapped.",
                std::format("CreateFileMappingW failed with Windows error {}", error),
                path(), mapped_offset, "mapped_range",
                static_cast<std::int64_t>(error), "Win32");
        }
        mapping->mapped_base = MapViewOfFile(
            mapping->mapping, FILE_MAP_READ,
            static_cast<DWORD>(mapped_offset >> 32),
            static_cast<DWORD>(mapped_offset & 0xffffffffu),
            static_cast<SIZE_T>(*mapped_length));
        if (mapping->mapped_base == nullptr) {
            const DWORD error = GetLastError();
            return detail::project_error(
                lfs::ErrorCode::Internal, "The project chunk could not be mapped.",
                std::format("MapViewOfFile failed with Windows error {}", error),
                path(), mapped_offset, "mapped_range",
                static_cast<std::int64_t>(error), "Win32");
        }
        mapping->mapped_bytes = static_cast<std::size_t>(*mapped_length);
#else
        const long page_size_result = ::sysconf(_SC_PAGESIZE);
        const std::uint64_t page_size =
            page_size_result > 0 ? static_cast<std::uint64_t>(page_size_result) : 4096;
        const std::uint64_t mapped_offset =
            (mapping->requested_offset / page_size) * page_size;
        const std::uint64_t adjustment = mapping->requested_offset - mapped_offset;
        auto mapped_length = detail::checked_add(adjustment, length, path(),
                                                 mapping->requested_offset,
                                                 "mapped_range.length");
        if (!mapped_length ||
            *mapped_length > std::numeric_limits<std::size_t>::max() ||
            mapped_offset >
                static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
            return !mapped_length
                       ? lfs::Result<MappedRegion>(std::move(mapped_length).error())
                       : lfs::Result<MappedRegion>(format_error(
                             path(), mapping->requested_offset,
                             "mapped_range.length", "size_t/off_t range",
                             std::to_string(*mapped_length),
                             lfs::ErrorCode::BoundsViolation));
        }
        mapping->mapped_base =
            ::mmap(nullptr, static_cast<std::size_t>(*mapped_length), PROT_READ,
                   MAP_SHARED, mapping->file->native_handle(),
                   static_cast<off_t>(mapped_offset));
        if (mapping->mapped_base == MAP_FAILED) {
            mapping->mapped_base = nullptr;
            const int error = errno;
            return detail::project_error(
                lfs::ErrorCode::Internal, "The project chunk could not be mapped.",
                std::format("mmap failed: {}", std::strerror(error)), path(),
                mapped_offset, "mapped_range", error, std::strerror(error));
        }
        mapping->mapped_bytes = static_cast<std::size_t>(*mapped_length);
#endif
        const std::size_t adjustment_bytes = static_cast<std::size_t>(
            mapping->requested_offset -
            (mapping->requested_offset -
             (mapping->requested_offset %
#ifdef _WIN32
              static_cast<std::uint64_t>(system_info.dwAllocationGranularity)
#else
              page_size
#endif
                  )));
        mapping->requested_data =
            static_cast<const std::byte*>(mapping->mapped_base) + adjustment_bytes;
        mapping->requested_bytes = static_cast<std::size_t>(length);
        return MappedRegion(std::move(mapping));
    }

} // namespace lfs::io::project
