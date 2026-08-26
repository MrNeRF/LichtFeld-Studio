/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_container.hpp"

#include "core/logger.hpp"
#include "crc32c.hpp"
#include "project_container_internal.hpp"
#include "project_framing.hpp"
#include "project_recovery_internal.hpp"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <exception>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
        constexpr std::uint64_t MAX_INDEX_UNCOMPRESSED_BYTES = 512ull * 1024 * 1024;
        constexpr std::uint64_t METADATA_PREFLIGHT_BYTES = 8ull * 1024 * 1024;

        lfs::Result<void> status_failure(lfs::Error error) {
            return lfs::Result<void>::failure(std::move(error));
        }

        lfs::Error writer_error(const lfs::ErrorCode code,
                                const std::filesystem::path& path,
                                const std::string_view user_message,
                                std::string detail_message,
                                const std::string_view field = {}) {
            return detail::project_error(code, std::string(user_message),
                                         std::move(detail_message), path, std::nullopt,
                                         field);
        }

        template <std::size_t Size>
        std::span<std::byte> byte_span(std::array<std::byte, Size>& bytes) {
            return {bytes.data(), bytes.size()};
        }

        template <std::size_t Size>
        std::span<const std::byte> byte_span(
            const std::array<std::byte, Size>& bytes) {
            return {bytes.data(), bytes.size()};
        }

        void put_u8(const std::span<std::byte> bytes, const std::size_t offset,
                    const std::uint8_t value) noexcept {
            bytes[offset] = static_cast<std::byte>(value);
        }

        void put_u16(const std::span<std::byte> bytes, const std::size_t offset,
                     const std::uint16_t value) noexcept {
            for (std::size_t index = 0; index < 2; ++index) {
                put_u8(bytes, offset + index,
                       static_cast<std::uint8_t>(value >> (index * 8)));
            }
        }

        void put_u32(const std::span<std::byte> bytes, const std::size_t offset,
                     const std::uint32_t value) noexcept {
            for (std::size_t index = 0; index < 4; ++index) {
                put_u8(bytes, offset + index,
                       static_cast<std::uint8_t>(value >> (index * 8)));
            }
        }

        void put_u64(const std::span<std::byte> bytes, const std::size_t offset,
                     const std::uint64_t value) noexcept {
            for (std::size_t index = 0; index < 8; ++index) {
                put_u8(bytes, offset + index,
                       static_cast<std::uint8_t>(value >> (index * 8)));
            }
        }

        void put_uuid(const std::span<std::byte> bytes, const std::size_t offset,
                      const lfs::core::Uuid& uuid) noexcept {
            for (std::size_t index = 0; index < uuid.bytes.size(); ++index) {
                put_u8(bytes, offset + index, uuid.bytes[index]);
            }
        }

        void put_fourcc(const std::span<std::byte> bytes, const std::size_t offset,
                        const Fourcc fourcc) noexcept {
            for (std::size_t index = 0; index < fourcc.bytes.size(); ++index) {
                put_u8(bytes, offset + index, fourcc.bytes[index]);
            }
        }

        void put_capabilities(const std::span<std::byte> bytes,
                              const std::size_t offset,
                              const CapabilitySet& capabilities) noexcept {
            for (std::size_t index = 0; index < capabilities.bytes().size();
                 ++index) {
                put_u8(bytes, offset + index, capabilities.bytes()[index]);
            }
        }

        template <std::size_t Size>
        void put_bytes(const std::span<std::byte> destination,
                       const std::size_t offset,
                       const std::array<std::byte, Size>& source) {
            std::copy(source.begin(), source.end(), destination.begin() + offset);
        }

        lfs::Result<std::uint64_t>
        align_up(const std::uint64_t value, const std::uint64_t alignment,
                 const std::filesystem::path& path,
                 const std::string_view field) {
            assert(alignment != 0 && (alignment & (alignment - 1)) == 0);
            const std::uint64_t mask = alignment - 1;
            auto sum = detail::checked_add(value, mask, path, value, field);
            if (!sum) {
                return std::move(sum).error();
            }
            return *sum & ~mask;
        }

        lfs::Result<void> write_zeros(detail::NativeFile& file,
                                      const std::uint64_t begin,
                                      const std::uint64_t end) {
            if (end < begin) {
                return status_failure(writer_error(
                    lfs::ErrorCode::ContractViolation, file.path(),
                    "The project writer computed an invalid padding range.",
                    std::format("padding begins at {} and ends at {}", begin, end),
                    "padding"));
            }
            constexpr std::array<std::byte, 4096> ZEROS{};
            std::uint64_t cursor = begin;
            while (cursor < end) {
                const std::uint64_t remaining = end - cursor;
                const std::size_t count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(remaining, ZEROS.size()));
                if (auto write = file.write_exact(
                        cursor, std::span<const std::byte>(ZEROS.data(), count));
                    !write) {
                    return write;
                }
                cursor += count;
            }
            return {};
        }

        lfs::Result<void> validate_key(const ChunkKey& key,
                                       const std::filesystem::path& path) {
            if (!key.fourcc.valid()) {
                return status_failure(writer_error(
                    lfs::ErrorCode::InvalidArgument, path,
                    "The project chunk type is invalid.",
                    std::format("fourcc '{}' must match [A-Z0-9]{{4}}",
                                key.fourcc.to_string()),
                    "chunk_key.fourcc"));
            }
            if (key.instance_uuid.is_nil()) {
                return status_failure(writer_error(
                    lfs::ErrorCode::InvalidArgument, path,
                    "The project chunk identity is invalid.",
                    "instance UUID must be non-null",
                    "chunk_key.instance_uuid"));
            }
            return {};
        }

        lfs::Result<lfs::core::Uuid>
        assigned_uuid(const lfs::core::Uuid& requested,
                      const std::filesystem::path& path,
                      const std::string_view field) {
            if (!requested.is_nil()) {
                return requested;
            }
            try {
                return lfs::core::generate_uuid_v4();
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): translate the UUID boundary to lfs::Error.
                return writer_error(
                    lfs::ErrorCode::Internal, path,
                    "A project identity could not be generated.",
                    std::format("{} generation failed: {}", field,
                                exception.what()),
                    field);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): preserve Result-only public error plumbing.
                return writer_error(
                    lfs::ErrorCode::Internal, path,
                    "A project identity could not be generated.",
                    std::format("{} generation failed with an unknown exception",
                                field),
                    field);
            }
        }

        // Payload records use the fast level; index chunks retain level 3. Below
        // compress; above it, ZSTD_compressStream2 with a bounded output window
        // so peak transient RAM is ~input + final-compressed + window, not
        // input + ZSTD_compressBound(~input).
        constexpr std::size_t ZSTD_STREAM_THRESHOLD_BYTES = 8ull * 1024 * 1024;
        constexpr std::size_t ZSTD_STREAM_OUT_WINDOW_BYTES = 8ull * 1024 * 1024;
        constexpr int ZSTD_FIXED_LEVEL = 3;
        constexpr int ZSTD_PAYLOAD_LEVEL = 1;

        // f32-word byte-plane transpose: [b0 b1 b2 b3 | b0 b1 b2 b3 | ...]
        // -> plane0 || plane1 || plane2 || plane3. Requires size % 4 == 0.
        std::vector<std::byte>
        byte_plane_f32_words(const std::span<const std::byte> interleaved) {
            const std::size_t n_words = interleaved.size() / 4;
            std::vector<std::byte> out(interleaved.size());
            const auto* src =
                reinterpret_cast<const std::uint8_t*>(interleaved.data());
            auto* dst = reinterpret_cast<std::uint8_t*>(out.data());
            for (std::size_t w = 0; w < n_words; ++w) {
                dst[0 * n_words + w] = src[w * 4 + 0];
                dst[1 * n_words + w] = src[w * 4 + 1];
                dst[2 * n_words + w] = src[w * 4 + 2];
                dst[3 * n_words + w] = src[w * 4 + 3];
            }
            return out;
        }

        lfs::Result<std::vector<std::byte>>
        compress_zstd(const std::span<const std::byte> input,
                      const std::filesystem::path& path,
                      const std::string_view field,
                      const int compression_level = ZSTD_FIXED_LEVEL) {
            if (input.size() < ZSTD_STREAM_THRESHOLD_BYTES) {
                const std::size_t bound = ZSTD_compressBound(input.size());
                std::vector<std::byte> compressed(bound);
                const std::size_t result = ZSTD_compress(
                    compressed.data(), compressed.size(), input.data(),
                    input.size(), compression_level);
                if (ZSTD_isError(result)) {
                    return writer_error(
                        lfs::ErrorCode::Internal, path,
                        "Project data compression failed.",
                        std::format("ZSTD_compress failed: {}",
                                    ZSTD_getErrorName(result)),
                        field);
                }
                compressed.resize(result);
                return compressed;
            }

            ZSTD_CCtx* cctx = ZSTD_createCCtx();
            if (cctx == nullptr) {
                return writer_error(
                    lfs::ErrorCode::ResourceExhausted, path,
                    "Project data compression failed.",
                    "ZSTD_createCCtx returned null", field);
            }
            auto free_cctx = [&]() { ZSTD_freeCCtx(cctx); };

            if (const std::size_t rc = ZSTD_CCtx_setParameter(
                    cctx, ZSTD_c_compressionLevel, compression_level);
                ZSTD_isError(rc)) {
                free_cctx();
                return writer_error(
                    lfs::ErrorCode::Internal, path,
                    "Project data compression failed.",
                    std::format("ZSTD_CCtx_setParameter failed: {}",
                                ZSTD_getErrorName(rc)),
                    field);
            }
            // Readers require a single index frame with known content size.
            if (const std::size_t rc = ZSTD_CCtx_setParameter(
                    cctx, ZSTD_c_contentSizeFlag, 1);
                ZSTD_isError(rc)) {
                free_cctx();
                return writer_error(
                    lfs::ErrorCode::Internal, path,
                    "Project data compression failed.",
                    std::format("ZSTD_c_contentSizeFlag failed: {}",
                                ZSTD_getErrorName(rc)),
                    field);
            }
            if (const std::size_t rc = ZSTD_CCtx_setPledgedSrcSize(
                    cctx, input.size());
                ZSTD_isError(rc)) {
                free_cctx();
                return writer_error(
                    lfs::ErrorCode::Internal, path,
                    "Project data compression failed.",
                    std::format("ZSTD_CCtx_setPledgedSrcSize failed: {}",
                                ZSTD_getErrorName(rc)),
                    field);
            }

            std::vector<std::byte> compressed;
            // Conservative reserve: mature CKPT payloads compress ~10–20%.
            compressed.reserve(input.size());
            std::vector<std::byte> out_window(ZSTD_STREAM_OUT_WINDOW_BYTES);

            std::size_t in_pos = 0;
            while (in_pos < input.size()) {
                const std::size_t chunk = std::min(ZSTD_STREAM_OUT_WINDOW_BYTES,
                                                   input.size() - in_pos);
                ZSTD_inBuffer zin{input.data() + in_pos, chunk, 0};
                const ZSTD_EndDirective mode =
                    (in_pos + chunk == input.size()) ? ZSTD_e_end
                                                     : ZSTD_e_continue;
                std::size_t remaining = 1;
                while (zin.pos < zin.size ||
                       (mode == ZSTD_e_end && remaining != 0)) {
                    ZSTD_outBuffer zout{out_window.data(), out_window.size(), 0};
                    remaining =
                        ZSTD_compressStream2(cctx, &zout, &zin, mode);
                    if (ZSTD_isError(remaining)) {
                        free_cctx();
                        return writer_error(
                            lfs::ErrorCode::Internal, path,
                            "Project data compression failed.",
                            std::format("ZSTD_compressStream2 failed: {}",
                                        ZSTD_getErrorName(remaining)),
                            field);
                    }
                    compressed.insert(compressed.end(), out_window.begin(),
                                      out_window.begin() +
                                          static_cast<std::ptrdiff_t>(zout.pos));
                    if (mode != ZSTD_e_end && zin.pos == zin.size) {
                        break;
                    }
                }
                in_pos += chunk;
            }
            free_cctx();
            return compressed;
        }

        lfs::Result<std::vector<std::byte>> frame_zstd(
            const std::span<const std::byte> input,
            const std::filesystem::path& path, const std::string_view field,
            const int compression_level = ZSTD_FIXED_LEVEL) {
            const auto started = std::chrono::steady_clock::now();
            const std::size_t record_count = detail::framed_record_count(input.size());
            if (record_count == 0) {
                return writer_error(lfs::ErrorCode::InvalidArgument, path,
                                    "The framed project payload is empty.",
                                    "framed payloads require at least one record", field);
            }
            std::vector<std::vector<std::byte>> records(record_count);
            std::atomic<std::size_t> next{0};
            std::mutex error_mutex;
            std::optional<lfs::Error> first_error;
            const auto publish_worker_error = [&](const lfs::ErrorCode code,
                                                  const std::string_view message,
                                                  const std::string_view detail) noexcept {
                try {
                    std::scoped_lock lock(error_mutex);
                    if (!first_error)
                        first_error = writer_error(code, path, message,
                                                   std::string(detail), field);
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): preserve the first worker error if formatting fails.
                }
            };
            const std::size_t worker_count = std::min<std::size_t>(
                record_count, std::max(1u, std::thread::hardware_concurrency()));
            std::vector<std::jthread> workers;
            try {
                workers.reserve(worker_count);
                for (std::size_t worker = 0; worker < worker_count; ++worker) {
                    workers.emplace_back([&] {
                        try {
                            while (true) {
                                const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                                if (index >= record_count)
                                    return;
                                {
                                    std::scoped_lock lock(error_mutex);
                                    if (first_error)
                                        return;
                                }
                                const std::size_t offset = index * detail::FRAMED_RECORD_TARGET_BYTES;
                                const std::size_t size = std::min(
                                    detail::FRAMED_RECORD_TARGET_BYTES, input.size() - offset);
                                auto compressed = compress_zstd(
                                    input.subspan(offset, size), path, field,
                                    compression_level);
                                if (!compressed) {
                                    std::scoped_lock lock(error_mutex);
                                    if (!first_error)
                                        first_error = std::move(compressed).error();
                                    return;
                                }
                                records[index] = std::move(*compressed);
                            }
                        } catch (const std::bad_alloc&) {
                            publish_worker_error(
                                lfs::ErrorCode::ResourceExhausted,
                                "There is not enough memory to compress this project payload.",
                                "framed compression worker allocation failed");
                        } catch (const std::exception& exception) {
                            publish_worker_error(
                                lfs::ErrorCode::Internal,
                                "The project payload could not be compressed.",
                                exception.what());
                        } catch (...) {
                            publish_worker_error(
                                lfs::ErrorCode::Internal,
                                "The project payload could not be compressed.",
                                "unknown worker exception");
                        }
                    });
                }
            } catch (const std::bad_alloc&) {
                publish_worker_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "There is not enough memory to start framed compression workers.",
                    "worker allocation failed");
            } catch (const std::exception& exception) {
                publish_worker_error(
                    lfs::ErrorCode::Internal,
                    "The project payload could not be compressed.",
                    exception.what());
            } catch (...) {
                publish_worker_error(
                    lfs::ErrorCode::Internal,
                    "The project payload could not be compressed.",
                    "unknown worker construction exception");
            }
            workers.clear();
            if (first_error)
                return std::move(*first_error);
            const auto compressed_at = std::chrono::steady_clock::now();

            std::size_t total = detail::FRAMED_HEADER_BYTES +
                                record_count * detail::FRAMED_RECORD_BYTES;
            for (const auto& record : records) {
                auto checked = detail::checked_add(total, record.size(), path,
                                                   total, field);
                if (!checked)
                    return std::move(checked).error();
                total = *checked;
            }
            std::vector<std::byte> framed(total);
            const auto bytes = std::span<std::byte>(framed);
            put_bytes(bytes, 0, detail::FRAMED_MAGIC);
            put_u16(bytes, 8, detail::FRAMED_VERSION);
            put_u16(bytes, 10, 0);
            put_u32(bytes, 12, static_cast<std::uint32_t>(record_count));
            std::size_t cursor = detail::FRAMED_HEADER_BYTES +
                                 record_count * detail::FRAMED_RECORD_BYTES;
            for (std::size_t index = 0; index < record_count; ++index) {
                const std::size_t source_offset = index * detail::FRAMED_RECORD_TARGET_BYTES;
                const std::size_t uncompressed = std::min(
                    detail::FRAMED_RECORD_TARGET_BYTES, input.size() - source_offset);
                const std::size_t table_offset = detail::FRAMED_HEADER_BYTES +
                                                 index * detail::FRAMED_RECORD_BYTES;
                put_u64(bytes, table_offset, records[index].size());
                put_u64(bytes, table_offset + 8, uncompressed);
                std::copy(records[index].begin(), records[index].end(),
                          framed.begin() + static_cast<std::ptrdiff_t>(cursor));
                cursor += records[index].size();
            }
            const auto assembled_at = std::chrono::steady_clock::now();
            const auto milliseconds = [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(end - begin).count();
            };
            LOG_DEBUG(
                "Project payload save stages: field={} input_bytes={} records={} level={} compress={:.3f} ms assemble={:.3f} ms total={:.3f} ms stored_bytes={}",
                field, input.size(), record_count, compression_level,
                milliseconds(started, compressed_at),
                milliseconds(compressed_at, assembled_at),
                milliseconds(started, assembled_at), framed.size());
            return framed;
        }

        std::array<std::byte, SUPERBLOCK_BYTES>
        encode_superblock(const SuperblockInfo& info) {
            std::array<std::byte, SUPERBLOCK_BYTES> raw{};
            const auto bytes = byte_span(raw);
            put_bytes(bytes, 0, SUPERBLOCK_MAGIC);
            put_u16(bytes, 8, info.format.major);
            put_u16(bytes, 10, info.format.minor);
            put_u32(bytes, 12, BYTE_ORDER_TAG);
            put_u32(bytes, 16,
                    static_cast<std::uint32_t>(SUPERBLOCK_BYTES));
            put_u32(bytes, 20, static_cast<std::uint32_t>(info.role));
            put_uuid(bytes, 24, info.project_uuid);
            put_uuid(bytes, 40, info.file_uuid);
            put_u64(bytes, 56, HEAD_SLOT_OFFSETS[0]);
            put_u64(bytes, 64, HEAD_SLOT_OFFSETS[1]);
            put_u32(bytes, 72, static_cast<std::uint32_t>(HEAD_SLOT_BYTES));
            put_u32(bytes, 76, 2);
            put_u64(bytes, 80, APPEND_REGION_OFFSET);
            put_u64(bytes, 88, info.creation_time_unix_ns);
            put_uuid(bytes, 96, info.base_explicit_commit_uuid);
            put_u64(bytes, 112, info.autosave_sequence);
            put_uuid(bytes, 128, info.sidecar_snapshot_uuid);
            put_u32(bytes, 252, crc32c(0, raw.data(), 252));
            return raw;
        }

        std::array<std::byte, CHUNK_HEADER_BYTES>
        encode_chunk_header(const ChunkInfo& row) {
            assert(row.row_kind == RowKind::Live);
            std::array<std::byte, CHUNK_HEADER_BYTES> raw{};
            const auto bytes = byte_span(raw);
            put_fourcc(bytes, 0, row.key.fourcc);
            put_u16(bytes, 4, row.chunk_version);
            put_u16(bytes, 6,
                    static_cast<std::uint16_t>(CHUNK_HEADER_BYTES));
            put_uuid(bytes, 8, row.key.instance_uuid);
            put_u32(bytes, 24, row.flags);
            put_u16(bytes, 28, static_cast<std::uint16_t>(row.compression));
            put_u16(bytes, 30, row.block_crc_table.has_value() ? 1 : 0);
            put_u64(bytes, 32, row.stored_bytes);
            put_u64(bytes, 40, row.uncompressed_bytes);
            put_u64(bytes, 48,
                    row.block_crc_table.has_value()
                        ? row.block_crc_table->offset
                        : 0);
            put_u32(bytes, 56, row.payload_crc32c);
            put_u32(bytes, 60, crc32c(0, raw.data(), 60));
            return raw;
        }

        std::array<std::byte, BLOCK_CRC_HEADER_BYTES>
        encode_block_crc_header(const BlockCrcTable& table) {
            std::array<std::byte, BLOCK_CRC_HEADER_BYTES> raw{};
            const auto bytes = byte_span(raw);
            put_bytes(bytes, 0, BLOCK_CRC_MAGIC);
            put_u16(bytes, 8, 1);
            put_u16(bytes, 10,
                    static_cast<std::uint16_t>(BLOCK_CRC_HEADER_BYTES));
            put_u16(bytes, 12, 4);
            put_u16(bytes, 14, 1);
            put_u32(bytes, 16, static_cast<std::uint32_t>(BLOCK_CRC_BYTES));
            put_u64(bytes, 24, table.payload_offset);
            put_u64(bytes, 32, table.stored_bytes);
            put_u64(bytes, 40, table.entries.size());
            put_u32(bytes, 48, table.entries_crc32c);
            put_u32(bytes, 60, crc32c(0, raw.data(), 60));
            return raw;
        }

        std::vector<std::byte>
        encode_block_entries(const std::vector<std::uint32_t>& entries) {
            std::vector<std::byte> raw(entries.size() * sizeof(std::uint32_t));
            for (std::size_t index = 0; index < entries.size(); ++index) {
                put_u32(raw, index * sizeof(std::uint32_t), entries[index]);
            }
            return raw;
        }

        std::array<std::byte, INDEX_ROW_BYTES>
        encode_index_row(const ChunkInfo& row) {
            std::array<std::byte, INDEX_ROW_BYTES> raw{};
            const auto bytes = byte_span(raw);
            put_fourcc(bytes, 0, row.key.fourcc);
            put_u16(bytes, 4, row.chunk_version);
            put_u8(bytes, 6, static_cast<std::uint8_t>(row.row_kind));
            put_u8(bytes, 7, static_cast<std::uint8_t>(row.compression));
            put_u32(bytes, 8, row.flags);
            put_uuid(bytes, 16, row.key.instance_uuid);
            put_u64(bytes, 32, row.header_offset);
            put_u64(bytes, 40, row.payload_offset);
            put_u64(bytes, 48, row.stored_bytes);
            put_u64(bytes, 56, row.uncompressed_bytes);
            put_u64(bytes, 64, row.source_generation);
            put_u32(bytes, 72, row.payload_crc32c);
            put_u32(bytes, 76, row.header_crc32c);
            return raw;
        }

        lfs::Result<std::vector<std::byte>>
        encode_index(const std::map<ChunkKey, ChunkInfo, ChunkKeyLess>& rows,
                     const std::uint64_t generation,
                     const lfs::core::Uuid& commit_uuid,
                     const std::filesystem::path& path) {
            auto rows_bytes = detail::checked_multiply(
                rows.size(), INDEX_ROW_BYTES, path, 16, "index.row_count");
            if (!rows_bytes) {
                return std::move(rows_bytes).error();
            }
            auto total = detail::checked_add(INDEX_HEADER_BYTES, *rows_bytes, path,
                                             16, "index.decoded_bytes");
            if (!total) {
                return std::move(total).error();
            }
            if (*total > MAX_INDEX_UNCOMPRESSED_BYTES ||
                *total > std::numeric_limits<std::size_t>::max()) {
                return writer_error(
                    lfs::ErrorCode::ResourceExhausted, path,
                    "The project index is too large.",
                    std::format("decoded index size {} exceeds the {}-byte limit",
                                *total, MAX_INDEX_UNCOMPRESSED_BYTES),
                    "index.decoded_bytes");
            }
            std::vector<std::byte> raw(static_cast<std::size_t>(*total));
            const auto bytes = std::span<std::byte>(raw);
            put_bytes(bytes, 0, INDEX_MAGIC);
            put_u16(bytes, 8, 1);
            put_u16(bytes, 10,
                    static_cast<std::uint16_t>(INDEX_HEADER_BYTES));
            put_u16(bytes, 12, static_cast<std::uint16_t>(INDEX_ROW_BYTES));
            put_u16(bytes, 14, 1);
            put_u64(bytes, 16, rows.size());
            put_u64(bytes, 24, generation);
            put_uuid(bytes, 32, commit_uuid);
            std::uint32_t flags = 0;
            for (const auto& [key, row] : rows) {
                (void)key;
                if (row.row_kind == RowKind::Tombstone) {
                    flags |= 1u;
                } else if (row.row_kind == RowKind::SidecarBaseReference) {
                    flags |= 2u;
                }
            }
            put_u32(bytes, 48, flags);
            std::size_t offset = INDEX_HEADER_BYTES;
            for (const auto& [key, row] : rows) {
                (void)key;
                const auto encoded = encode_index_row(row);
                std::copy(encoded.begin(), encoded.end(), raw.begin() + offset);
                offset += encoded.size();
            }
            assert(offset == raw.size());
            return raw;
        }

        CapabilitySet derive_reader_capabilities(
            const std::map<ChunkKey, ChunkInfo, ChunkKeyLess>& rows,
            const ContainerRole role, const IndexCompression index_compression) {
            CapabilitySet capabilities;
            if (index_compression == IndexCompression::Zstd) {
                capabilities.set(INDEX_ZSTD_V1);
            }
            if (role == ContainerRole::AutosaveSidecar) {
                capabilities.set(SIDECAR_OVERLAY_V1);
            }
            for (const auto& [key, row] : rows) {
                (void)key;
                if (row.row_kind == RowKind::Live) {
                    if (row.compression == Compression::ZstdFramed) {
                        capabilities.set(CHUNK_ZSTD_V1);
                    }
                    if (row.compression == Compression::ByteShuffleZstdFramed) {
                        capabilities.set(CHUNK_BYTESHUFFLE_ZSTD_V1);
                    }
                    if ((row.flags & HAS_BLOCK_CRCS) != 0) {
                        capabilities.set(BLOCK_CRC32C_V1);
                    }
                } else if (row.row_kind == RowKind::Tombstone) {
                    capabilities.set(INDEX_TOMBSTONES_V1);
                } else if (row.row_kind ==
                           RowKind::SidecarBaseReference) {
                    capabilities.set(SIDECAR_OVERLAY_V1);
                }
            }
            return capabilities;
        }

        CapabilitySet derive_writer_capabilities(
            const ContainerRole role, const bool used_opaque_carry_forward,
            const bool used_clean_proof_reuse,
            const bool preserve_legacy_fixture_envelope) {
            CapabilitySet capabilities;
            capabilities.set(OPAQUE_CHUNK_PRESERVATION,
                             used_opaque_carry_forward ||
                                 preserve_legacy_fixture_envelope);
            capabilities.set(CLEAN_PROOF_REUSE,
                             used_clean_proof_reuse ||
                                 preserve_legacy_fixture_envelope);
            if (role == ContainerRole::AutosaveSidecar) {
                capabilities.set(SIDECAR_OVERLAY_V1);
            }
            return capabilities;
        }

        std::array<std::byte, COMMIT_RECORD_BYTES>
        encode_commit(const SuperblockInfo& superblock, const CommitInfo& commit) {
            std::array<std::byte, COMMIT_RECORD_BYTES> raw{};
            const auto bytes = byte_span(raw);
            put_bytes(bytes, 0, COMMIT_MAGIC);
            put_u16(bytes, 8,
                    static_cast<std::uint16_t>(COMMIT_RECORD_BYTES));
            put_u16(bytes, 10, 1);
            put_u32(bytes, 12, static_cast<std::uint32_t>(commit.kind));
            put_uuid(bytes, 16, superblock.project_uuid);
            put_uuid(bytes, 32, superblock.file_uuid);
            put_uuid(bytes, 48, commit.commit_uuid);
            put_u64(bytes, 64, commit.generation);
            put_uuid(bytes, 72, commit.parent_commit_uuid);
            put_u64(bytes, 88, commit.parent_commit_offset);
            put_uuid(bytes, 96, commit.explicit_ancestor_commit_uuid);
            put_uuid(bytes, 112, commit.snapshot_uuid);
            put_u64(bytes, 128, commit.wallclock_unix_ns);
            put_u64(bytes, 136, commit.index_offset);
            put_u64(bytes, 144, commit.index_stored_bytes);
            put_u64(bytes, 152, commit.index_uncompressed_bytes);
            put_u32(bytes, 160, commit.index_stored_crc32c);
            put_u32(bytes, 164, commit.index_uncompressed_crc32c);
            put_u32(bytes, 168,
                    static_cast<std::uint32_t>(commit.index_compression));
            put_u64(bytes, 176, commit.committed_file_end);
            put_u16(bytes, 184, commit.min_reader_version.major);
            put_u16(bytes, 186, commit.min_reader_version.minor);
            put_u16(bytes, 188, commit.min_safe_writer_version.major);
            put_u16(bytes, 190, commit.min_safe_writer_version.minor);
            put_capabilities(bytes, 192,
                             commit.required_reader_capabilities);
            put_capabilities(bytes, 208,
                             commit.required_writer_capabilities);
            put_u32(bytes, 252, crc32c(0, raw.data(), 252));
            return raw;
        }

        std::array<std::byte, HEAD_SLOT_BYTES>
        encode_head(const SuperblockInfo& superblock, const HeadInfo& head) {
            std::array<std::byte, HEAD_SLOT_BYTES> raw{};
            const auto bytes = byte_span(raw);
            put_bytes(bytes, 0, HEAD_MAGIC);
            put_u32(bytes, 8, head.slot_id);
            put_u32(bytes, 12, static_cast<std::uint32_t>(HEAD_SLOT_BYTES));
            put_u64(bytes, 16, head.head_sequence);
            put_u64(bytes, 24, head.generation);
            put_uuid(bytes, 32, superblock.project_uuid);
            put_uuid(bytes, 48, superblock.file_uuid);
            put_uuid(bytes, 64, head.commit_uuid);
            put_u64(bytes, 80, head.commit_offset);
            put_u64(bytes, 88, COMMIT_RECORD_BYTES);
            put_u64(bytes, 96, head.committed_file_end);
            put_u32(bytes, 104, head.commit_crc32c_echo);
            if (head.preview.has_value()) {
                put_u64(bytes, 112, head.preview->offset);
                put_u32(bytes, 120, head.preview->bytes);
                put_u32(bytes, 124,
                        static_cast<std::uint32_t>(head.preview->format));
            }
            put_u32(bytes, 4092, crc32c(0, raw.data(), 4092));
            return raw;
        }

        std::uint32_t crc_entries(
            const std::vector<std::uint32_t>& entries) {
            const auto encoded = encode_block_entries(entries);
            return crc32c(0, encoded.data(), encoded.size());
        }

        std::vector<std::uint32_t>
        calculate_block_crcs(const std::span<const std::byte> stored) {
            std::vector<std::uint32_t> entries;
            if (stored.empty()) {
                return entries;
            }
            const std::uint64_t count =
                stored.size() / BLOCK_CRC_BYTES +
                (stored.size() % BLOCK_CRC_BYTES != 0 ? 1 : 0);
            entries.reserve(static_cast<std::size_t>(count));
            std::size_t offset = 0;
            while (offset < stored.size()) {
                const std::size_t bytes = static_cast<std::size_t>(
                    std::min<std::uint64_t>(stored.size() - offset,
                                            BLOCK_CRC_BYTES));
                entries.push_back(
                    crc32c(0, stored.data() + offset, bytes));
                offset += bytes;
            }
            return entries;
        }

    } // namespace

    struct ProjectWriter::Impl {
        enum class Mode {
            Create,
            Append,
        };

        struct Crc32cCountingStreambuf final : std::streambuf {
            Crc32cCountingStreambuf(std::shared_ptr<detail::NativeFile> file_in,
                                    const std::uint64_t offset_in,
                                    const bool block_crcs_in)
                : file(std::move(file_in)),
                  start_offset(offset_in),
                  write_offset(offset_in),
                  collect_block_crcs(block_crcs_in) {
                setp(buffer.data(), buffer.data() + buffer.size());
            }

            [[nodiscard]] std::uint32_t crc() const noexcept {
                return running_crc;
            }

            [[nodiscard]] std::uint64_t count() const noexcept {
                return byte_count;
            }

            [[nodiscard]] const std::vector<std::uint32_t>&
            block_entries() const noexcept {
                return entries;
            }

            [[nodiscard]] const std::optional<lfs::Error>&
            error() const noexcept {
                return failure;
            }

        protected:
            int_type overflow(const int_type character) override {
                if (flush_buffer() != 0) {
                    return traits_type::eof();
                }
                if (!traits_type::eq_int_type(character, traits_type::eof())) {
                    *pptr() = traits_type::to_char_type(character);
                    pbump(1);
                    if (flush_buffer() != 0) {
                        return traits_type::eof();
                    }
                }
                return traits_type::not_eof(character);
            }

            std::streamsize xsputn(const char* source,
                                   const std::streamsize count) override {
                if (count <= 0 || failure.has_value()) {
                    return 0;
                }
                std::streamsize completed = 0;
                while (completed < count) {
                    const auto available = epptr() - pptr();
                    if (available == 0 && flush_buffer() != 0) {
                        break;
                    }
                    const std::streamsize step =
                        std::min<std::streamsize>(count - completed,
                                                  epptr() - pptr());
                    std::memcpy(pptr(), source + completed,
                                static_cast<std::size_t>(step));
                    pbump(static_cast<int>(step));
                    completed += step;
                }
                return completed;
            }

            int sync() override {
                return flush_buffer();
            }

        private:
            int flush_buffer() {
                const std::size_t count =
                    static_cast<std::size_t>(pptr() - pbase());
                if (count == 0) {
                    return failure.has_value() ? -1 : 0;
                }
                const auto bytes = std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(pbase()), count);
                if (auto write = file->write_exact(write_offset, bytes); !write) {
                    failure = std::move(write).error();
                    setp(buffer.data(), buffer.data() + buffer.size());
                    return -1;
                }
                running_crc = crc32c(running_crc, bytes.data(), bytes.size());
                if (collect_block_crcs) {
                    consume_blocks(bytes);
                }
                write_offset += bytes.size();
                byte_count += bytes.size();
                setp(buffer.data(), buffer.data() + buffer.size());
                return 0;
            }

            void consume_blocks(std::span<const std::byte> bytes) {
                while (!bytes.empty()) {
                    const std::size_t room = static_cast<std::size_t>(
                        BLOCK_CRC_BYTES - block_bytes);
                    const std::size_t step = std::min(room, bytes.size());
                    block_crc =
                        crc32c(block_crc, bytes.data(), step);
                    block_bytes += step;
                    bytes = bytes.subspan(step);
                    if (block_bytes == BLOCK_CRC_BYTES) {
                        entries.push_back(block_crc);
                        block_crc = 0;
                        block_bytes = 0;
                    }
                }
            }

        public:
            void finish_partial_block() {
                if (collect_block_crcs && block_bytes != 0) {
                    entries.push_back(block_crc);
                    block_crc = 0;
                    block_bytes = 0;
                }
            }

            std::shared_ptr<detail::NativeFile> file;
            std::uint64_t start_offset = 0;
            std::uint64_t write_offset = 0;
            std::uint64_t byte_count = 0;
            std::uint32_t running_crc = 0;
            bool collect_block_crcs = false;
            std::uint64_t block_bytes = 0;
            std::uint32_t block_crc = 0;
            std::vector<std::uint32_t> entries;
            std::optional<lfs::Error> failure;
            std::array<char, 256 * 1024> buffer{};
        };

        struct StreamingChunk {
            ChunkKey key;
            ChunkWriteOptions options;
            ChunkInfo row;
            std::uint64_t table_offset = 0;
            std::uint64_t table_bytes = 0;
            Crc32cCountingStreambuf buffer;
            std::ostream stream;

            StreamingChunk(ChunkKey key_in, ChunkWriteOptions options_in,
                           ChunkInfo row_in, const std::uint64_t table_offset_in,
                           const std::uint64_t table_bytes_in,
                           std::shared_ptr<detail::NativeFile> file)
                : key(std::move(key_in)),
                  options(std::move(options_in)),
                  row(std::move(row_in)),
                  table_offset(table_offset_in),
                  table_bytes(table_bytes_in),
                  buffer(std::move(file), row.payload_offset,
                         (row.flags & HAS_BLOCK_CRCS) != 0),
                  stream(&buffer) {}
        };

        Mode mode = Mode::Append;
        std::filesystem::path destination_path;
        std::filesystem::path active_path;
        std::optional<detail::WriterLock> lock;
        std::optional<WriterLockLease> lock_lease;
        std::shared_ptr<detail::NativeFile> file;
        std::optional<ProjectReader> prior_reader;
        SuperblockInfo superblock;
        std::uint64_t generation = 1;
        std::uint64_t head_sequence = 1;
        std::uint32_t head_slot = 0;
        lfs::core::Uuid parent_commit_uuid;
        std::uint64_t parent_commit_offset = 0;
        std::uint64_t cursor = APPEND_REGION_OFFSET;
        std::uint64_t original_physical_size = 0;
        std::uint64_t disk_reserve_bytes = 0;
        IndexCompression index_compression = IndexCompression::Zstd;
        CommitBoundaryObserver observer;
        std::map<ChunkKey, ChunkInfo, ChunkKeyLess> rows;
        std::set<ChunkKey, ChunkKeyLess> unresolved;
        std::set<ChunkKey, ChunkKeyLess> touched;
        std::set<ChunkKey, ChunkKeyLess> seeded_base_keys;
        std::optional<PreviewLocator> preview_locator;
        std::optional<CommitOptions> commit_plan;
        std::unique_ptr<StreamingChunk> streaming;
        bool preflight_complete = false;
        bool mutation_started = false;
        bool committed = false;
        bool poisoned = false;
        bool keep_temporary = false;
        bool used_clean_proof_reuse = false;
        bool used_opaque_carry_forward = false;
        std::optional<lfs::Error> post_publish_note;

        ~Impl() {
            if (mode == Mode::Create && !committed && !keep_temporary &&
                !active_path.empty()) {
                std::error_code ignored;
                std::filesystem::remove(active_path, ignored);
            }
        }

        [[nodiscard]] lfs::Result<void>
        notify(const CommitBoundary boundary) const {
            if (!observer) {
                return {};
            }
            try {
                observer(boundary);
                return {};
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): test hooks cannot escape the Result boundary.
                return status_failure(writer_error(
                    lfs::ErrorCode::Internal, destination_path,
                    "The project commit boundary observer failed.",
                    std::format("boundary {} observer threw: {}",
                                static_cast<unsigned>(boundary),
                                exception.what()),
                    "commit_boundary_observer"));
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): test hooks cannot escape the Result boundary.
                return status_failure(writer_error(
                    lfs::ErrorCode::Internal, destination_path,
                    "The project commit boundary observer failed.",
                    std::format("boundary {} observer threw an unknown exception",
                                static_cast<unsigned>(boundary)),
                    "commit_boundary_observer"));
            }
        }

        void record_post_publish_note(
            lfs::Error cause,
            const std::string_view operation) {
            if (post_publish_note) {
                post_publish_note =
                    std::move(*post_publish_note)
                        .with_suppressed(
                            std::move(cause));
                return;
            }
            auto note = writer_error(
                lfs::ErrorCode::Internal,
                destination_path,
                "The project was published, but post-publication cleanup reported a problem.",
                std::format(
                    "{} failed after the destination authority was independently validated; the published commit remains successful",
                    operation),
                "commit.post_publish_verification");
            post_publish_note =
                std::move(note).with_suppressed(
                    std::move(cause));
        }

        [[nodiscard]] lfs::Result<void> require_ready() const {
            if (committed) {
                return status_failure(writer_error(
                    lfs::ErrorCode::FailedPrecondition, destination_path,
                    "This project writer has already committed.",
                    "one ProjectWriter instance publishes exactly one generation",
                    "writer.state"));
            }
            if (poisoned) {
                return status_failure(writer_error(
                    lfs::ErrorCode::FailedPrecondition, destination_path,
                    "This project save cannot continue after an earlier write failure.",
                    "discard this writer and retry from the last committed generation",
                    "writer.state"));
            }
            if (!preflight_complete) {
                return status_failure(writer_error(
                    lfs::ErrorCode::FailedPrecondition, destination_path,
                    "Project save disk preflight has not completed.",
                    "call preflight() before writing chunks",
                    "writer.state"));
            }
            if (streaming) {
                return status_failure(writer_error(
                    lfs::ErrorCode::FailedPrecondition, destination_path,
                    "A streaming project chunk is still open.",
                    "call end_chunk() before starting another mutation",
                    "writer.streaming_chunk"));
            }
            return {};
        }

        [[nodiscard]] lfs::Result<void> touch_key(const ChunkKey& key) {
            if (touched.contains(key)) {
                return status_failure(writer_error(
                    lfs::ErrorCode::AlreadyExists, destination_path,
                    "The project chunk was resolved more than once.",
                    std::format("key {} may be written, reused, carried, or erased "
                                "only once per generation",
                                key.fourcc.to_string()),
                    "chunk_key"));
            }
            seeded_base_keys.erase(key);
            touched.insert(key);
            unresolved.erase(key);
            return {};
        }

        [[nodiscard]] lfs::Result<void>
        write_block_table(const ChunkInfo& row) {
            if (!row.block_crc_table.has_value()) {
                return {};
            }
            const BlockCrcTable& table = *row.block_crc_table;
            const auto entries = encode_block_entries(table.entries);
            const auto header = encode_block_crc_header(table);
            if (auto write =
                    file->write_exact(table.offset, byte_span(header));
                !write) {
                return write;
            }
            return file->write_exact(table.offset + BLOCK_CRC_HEADER_BYTES,
                                     entries);
        }

        [[nodiscard]] lfs::Result<ChunkInfo>
        place_stored_chunk(const ChunkKey& key,
                           const std::span<const std::byte> stored,
                           const std::uint64_t uncompressed_bytes,
                           const ChunkWriteOptions& options,
                           const bool preserve_stored_crc = false,
                           const std::uint32_t expected_stored_crc = 0) {
            const auto started = std::chrono::steady_clock::now();
            auto header_offset =
                align_up(cursor, CHUNK_ALIGNMENT, active_path,
                         "chunk.header_offset");
            if (!header_offset) {
                return std::move(header_offset).error();
            }
            auto table_offset = detail::checked_add(
                *header_offset, CHUNK_HEADER_BYTES, active_path, *header_offset,
                "chunk.table_offset");
            if (!table_offset) {
                return std::move(table_offset).error();
            }

            const bool with_blocks =
                options.block_crcs ||
                stored.size() >= BLOCK_CRC_REQUIRED_AT;
            std::vector<std::uint32_t> entries;
            std::uint64_t table_bytes = 0;
            if (with_blocks) {
                if (stored.empty()) {
                    return writer_error(
                        lfs::ErrorCode::InvalidArgument, destination_path,
                        "An empty project chunk cannot have a block CRC table.",
                        "block_count must be at least one",
                        "chunk.block_crcs");
                }
                entries = calculate_block_crcs(stored);
                auto entry_bytes = detail::checked_multiply(
                    entries.size(), sizeof(std::uint32_t), active_path,
                    *table_offset, "block_table.entries");
                if (!entry_bytes) {
                    return std::move(entry_bytes).error();
                }
                auto total = detail::checked_add(
                    BLOCK_CRC_HEADER_BYTES, *entry_bytes, active_path,
                    *table_offset, "block_table.bytes");
                if (!total) {
                    return std::move(total).error();
                }
                table_bytes = *total;
            }
            const auto crc_at = std::chrono::steady_clock::now();
            auto after_metadata = detail::checked_add(
                *table_offset, table_bytes, active_path, *table_offset,
                "chunk.payload_pre_alignment");
            if (!after_metadata) {
                return std::move(after_metadata).error();
            }
            const std::uint64_t alignment =
                options.tensor_payload ? TENSOR_PAYLOAD_ALIGNMENT
                                       : CHUNK_ALIGNMENT;
            auto payload_offset =
                align_up(*after_metadata, alignment, active_path,
                         "chunk.payload_offset");
            if (!payload_offset) {
                return std::move(payload_offset).error();
            }
            auto payload_end = detail::checked_add(
                *payload_offset, stored.size(), active_path, *payload_offset,
                "chunk.payload_end");
            if (!payload_end) {
                return std::move(payload_end).error();
            }

            if (auto zero = write_zeros(*file, cursor, *header_offset); !zero) {
                return std::move(zero).error();
            }
            if (auto zero =
                    write_zeros(*file, *header_offset, *payload_offset);
                !zero) {
                return std::move(zero).error();
            }
            if (auto write = file->write_exact(*payload_offset, stored); !write) {
                return std::move(write).error();
            }

            const std::uint32_t stored_crc =
                crc32c(0, stored.data(), stored.size());
            if (preserve_stored_crc && stored_crc != expected_stored_crc) {
                return writer_error(
                    lfs::ErrorCode::DataLoss, destination_path,
                    "A project chunk changed while it was being copied.",
                    std::format("expected stored CRC 0x{:08x}, got 0x{:08x}",
                                expected_stored_crc, stored_crc),
                    "chunk.payload_crc32c");
            }

            ChunkInfo row{
                .key = key,
                .chunk_version = options.chunk_version,
                .row_kind = RowKind::Live,
                .compression = options.compression,
                .flags = (options.tensor_payload ? TENSOR_PAYLOAD : 0u) |
                         (with_blocks ? HAS_BLOCK_CRCS : 0u),
                .header_offset = *header_offset,
                .payload_offset = *payload_offset,
                .stored_bytes = stored.size(),
                .uncompressed_bytes = uncompressed_bytes,
                .source_generation = generation,
                .payload_crc32c = stored_crc,
                .header_crc32c = 0,
                .block_crc_table = std::nullopt,
            };
            if (with_blocks) {
                row.block_crc_table = BlockCrcTable{
                    .offset = *table_offset,
                    .payload_offset = *payload_offset,
                    .stored_bytes = stored.size(),
                    .block_size =
                        static_cast<std::uint32_t>(BLOCK_CRC_BYTES),
                    .entries = std::move(entries),
                    .entries_crc32c = 0,
                    .header_crc32c = 0,
                };
                row.block_crc_table->entries_crc32c =
                    crc_entries(row.block_crc_table->entries);
                const auto table_header =
                    encode_block_crc_header(*row.block_crc_table);
                row.block_crc_table->header_crc32c =
                    crc32c(0, table_header.data(), 60);
                if (auto table = write_block_table(row); !table) {
                    return std::move(table).error();
                }
            }
            const auto header = encode_chunk_header(row);
            row.header_crc32c = crc32c(0, header.data(), 60);
            if (auto write =
                    file->write_exact(*header_offset, byte_span(header));
                !write) {
                return std::move(write).error();
            }
            const auto written_at = std::chrono::steady_clock::now();
            const auto milliseconds = [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(end - begin).count();
            };
            LOG_DEBUG(
                "Project chunk write stages: chunk={} stored_bytes={} block_crc={:.3f} ms disk_write={:.3f} ms total={:.3f} ms",
                key.fourcc.to_string(), stored.size(), milliseconds(started, crc_at),
                milliseconds(crc_at, written_at), milliseconds(started, written_at));
            cursor = *payload_end;
            mutation_started = true;
            return row;
        }

        [[nodiscard]] lfs::Result<ChunkInfo>
        copy_stored_chunk(const ProjectReader& source,
                          const ChunkInfo& source_row) {
            const auto started = std::chrono::steady_clock::now();
            assert(source_row.row_kind == RowKind::Live);
            auto header_offset =
                align_up(cursor, CHUNK_ALIGNMENT, active_path,
                         "chunk.header_offset");
            if (!header_offset) {
                return std::move(header_offset).error();
            }
            auto table_offset = detail::checked_add(
                *header_offset, CHUNK_HEADER_BYTES, active_path,
                *header_offset, "chunk.table_offset");
            if (!table_offset) {
                return std::move(table_offset).error();
            }
            const bool with_blocks =
                (source_row.flags & HAS_BLOCK_CRCS) != 0;
            const std::uint64_t block_count =
                with_blocks
                    ? source_row.stored_bytes / BLOCK_CRC_BYTES +
                          (source_row.stored_bytes % BLOCK_CRC_BYTES != 0 ? 1
                                                                          : 0)
                    : 0;
            if (with_blocks && block_count == 0) {
                return writer_error(
                    lfs::ErrorCode::DataLoss, destination_path,
                    "The source block CRC layout is invalid.",
                    "a present block table must cover a nonempty payload",
                    "block_table.block_count");
            }
            auto entry_bytes = detail::checked_multiply(
                block_count, sizeof(std::uint32_t), active_path, *table_offset,
                "block_table.entries");
            if (!entry_bytes) {
                return std::move(entry_bytes).error();
            }
            auto table_bytes = detail::checked_add(
                with_blocks ? BLOCK_CRC_HEADER_BYTES : 0, *entry_bytes,
                active_path, *table_offset, "block_table.bytes");
            if (!table_bytes) {
                return std::move(table_bytes).error();
            }
            auto after_metadata = detail::checked_add(
                *table_offset, *table_bytes, active_path, *table_offset,
                "chunk.payload_pre_alignment");
            if (!after_metadata) {
                return std::move(after_metadata).error();
            }
            auto payload_offset = align_up(
                *after_metadata,
                (source_row.flags & TENSOR_PAYLOAD) != 0
                    ? TENSOR_PAYLOAD_ALIGNMENT
                    : CHUNK_ALIGNMENT,
                active_path, "chunk.payload_offset");
            if (!payload_offset) {
                return std::move(payload_offset).error();
            }
            auto payload_end = detail::checked_add(
                *payload_offset, source_row.stored_bytes, active_path,
                *payload_offset, "chunk.payload_end");
            if (!payload_end) {
                return std::move(payload_end).error();
            }
            if (auto zero = write_zeros(*file, cursor, *payload_offset); !zero) {
                return std::move(zero).error();
            }

            constexpr std::size_t COPY_BUFFER_BYTES = 5 * 1024 * 1024;
            std::vector<std::byte> buffer(
                static_cast<std::size_t>(std::min<std::uint64_t>(
                    std::max<std::uint64_t>(source_row.stored_bytes, 1),
                    COPY_BUFFER_BYTES)));
            std::vector<std::uint32_t> entries;
            entries.reserve(static_cast<std::size_t>(block_count));
            std::uint64_t relative_offset = 0;
            std::uint32_t stored_crc = 0;
            std::uint32_t block_crc = 0;
            std::uint64_t block_bytes = 0;
            while (relative_offset < source_row.stored_bytes) {
                const std::size_t count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(
                        source_row.stored_bytes - relative_offset,
                        buffer.size()));
                auto block = std::span<std::byte>(buffer.data(), count);
                if (auto read = source.read_stored_at(
                        source_row, relative_offset, block);
                    !read) {
                    return std::move(read).error();
                }
                stored_crc =
                    crc32c(stored_crc, block.data(), block.size());
                if (with_blocks) {
                    auto remaining = std::span<const std::byte>(block);
                    while (!remaining.empty()) {
                        const std::size_t room = static_cast<std::size_t>(
                            BLOCK_CRC_BYTES - block_bytes);
                        const std::size_t step =
                            std::min(room, remaining.size());
                        block_crc =
                            crc32c(block_crc, remaining.data(), step);
                        block_bytes += step;
                        remaining = remaining.subspan(step);
                        if (block_bytes == BLOCK_CRC_BYTES) {
                            entries.push_back(block_crc);
                            block_crc = 0;
                            block_bytes = 0;
                        }
                    }
                }
                if (auto write = file->write_exact(
                        *payload_offset + relative_offset, block);
                    !write) {
                    return std::move(write).error();
                }
                relative_offset += count;
            }
            if (with_blocks && block_bytes != 0) {
                entries.push_back(block_crc);
            }
            assert(entries.size() == block_count);
            if (stored_crc != source_row.payload_crc32c) {
                return writer_error(
                    lfs::ErrorCode::DataLoss, destination_path,
                    "A source project chunk failed CRC validation during compaction.",
                    std::format("expected 0x{:08x}, got 0x{:08x}",
                                source_row.payload_crc32c, stored_crc),
                    "chunk.payload_crc32c");
            }

            ChunkInfo row{
                .key = source_row.key,
                .chunk_version = source_row.chunk_version,
                .row_kind = RowKind::Live,
                .compression = source_row.compression,
                .flags = source_row.flags,
                .header_offset = *header_offset,
                .payload_offset = *payload_offset,
                .stored_bytes = source_row.stored_bytes,
                .uncompressed_bytes = source_row.uncompressed_bytes,
                .source_generation = generation,
                .payload_crc32c = stored_crc,
                .header_crc32c = 0,
                .block_crc_table = std::nullopt,
            };
            if (with_blocks) {
                row.block_crc_table = BlockCrcTable{
                    .offset = *table_offset,
                    .payload_offset = *payload_offset,
                    .stored_bytes = row.stored_bytes,
                    .block_size =
                        static_cast<std::uint32_t>(BLOCK_CRC_BYTES),
                    .entries = std::move(entries),
                    .entries_crc32c = 0,
                    .header_crc32c = 0,
                };
                row.block_crc_table->entries_crc32c =
                    crc_entries(row.block_crc_table->entries);
                const auto table_header =
                    encode_block_crc_header(*row.block_crc_table);
                row.block_crc_table->header_crc32c =
                    crc32c(0, table_header.data(), 60);
                if (auto table = write_block_table(row); !table) {
                    return std::move(table).error();
                }
            }
            const auto header = encode_chunk_header(row);
            row.header_crc32c = crc32c(0, header.data(), 60);
            if (auto write =
                    file->write_exact(*header_offset, byte_span(header));
                !write) {
                return std::move(write).error();
            }
            const auto finished = std::chrono::steady_clock::now();
            LOG_DEBUG(
                "Project carried chunk stage: chunk={} stored_bytes={} total={:.3f} ms",
                source_row.key_string(), source_row.stored_bytes,
                std::chrono::duration<double, std::milli>(finished - started).count());
            cursor = *payload_end;
            mutation_started = true;
            return row;
        }
    };

    ProjectWriter::ProjectWriter(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    ProjectWriter::ProjectWriter(ProjectWriter&&) noexcept = default;
    ProjectWriter& ProjectWriter::operator=(ProjectWriter&&) noexcept = default;
    ProjectWriter::~ProjectWriter() = default;

    const std::optional<lfs::Error>&
    ProjectWriter::post_publish_note() const noexcept {
        return impl_->post_publish_note;
    }

    lfs::Result<ProjectWriter>
    ProjectWriter::create(const std::filesystem::path& path,
                          const CreateOptions& options) {
        if (path.empty()) {
            return writer_error(lfs::ErrorCode::InvalidArgument, path,
                                "The project path is empty.",
                                "create requires a destination path",
                                "project.path");
        }
        if (options.role ==
                ContainerRole::AutosaveSidecar &&
            !options.writer_lock_anchor) {
            return writer_error(
                lfs::ErrorCode::FailedPrecondition,
                path,
                "An autosave sidecar requires its master writer-lock anchor.",
                "role AUTOSAVE_SIDECAR must bind publication to a held master authority",
                "writer_lock_anchor");
        }
        if (auto parent = detail::ensure_parent_directory(path); !parent) {
            return std::move(parent).error();
        }
        const auto& lock_anchor =
            options.writer_lock_anchor
                ? *options.writer_lock_anchor
                : path;
        std::optional<detail::WriterLock> held_lock;
        std::optional<WriterLockLease> held_lease;
        if (options.writer_lock_lease) {
            if (!options.writer_lock_lease->owns(
                    lock_anchor)) {
                return writer_error(
                    lfs::ErrorCode::FailedPrecondition,
                    lock_anchor,
                    "The supplied project writer lock is not held for this destination.",
                    "writer_lock_lease must own the selected lock anchor",
                    "writer_lock");
            }
            held_lease =
                *options.writer_lock_lease;
        } else {
            auto lock_result =
                detail::WriterLock::acquire(
                    lock_anchor);
            if (!lock_result) {
                return std::move(lock_result)
                    .error();
            }
            held_lock.emplace(
                std::move(*lock_result));
        }

        if (options.role ==
                ContainerRole::AutosaveSidecar &&
            options.writer_lock_anchor) {
            auto master = ProjectReader::open(
                lock_anchor,
                options.writer_lock_anchor_compatibility);
            if (!master) {
                return std::move(master).error();
            }
            if (master->superblock().role !=
                    ContainerRole::Master ||
                master->superblock().project_uuid !=
                    options.project_uuid ||
                master->commit().commit_uuid !=
                    options.base_explicit_commit_uuid) {
                return writer_error(
                    lfs::ErrorCode::FailedPrecondition,
                    lock_anchor,
                    "The autosave base changed before publication.",
                    "the held master authority no longer matches the "
                    "sidecar project/base binding",
                    "superblock.sidecar_binding");
            }
            auto valid =
                detail::valid_bound_autosaves_locked(
                    lock_anchor, *master);
            if (!valid) {
                return std::move(valid).error();
            }
            const auto highest =
                std::ranges::max_element(
                    *valid, {},
                    &detail::ValidBoundAutosave::sequence);
            if (highest != valid->end() &&
                options.autosave_sequence <=
                    highest->sequence) {
                return writer_error(
                    lfs::ErrorCode::FailedPrecondition,
                    path,
                    "The autosave sequence is not newer than the current recovery candidate.",
                    std::format(
                        "requested sequence {} must be greater than held candidate sequence {}",
                        options.autosave_sequence,
                        highest->sequence),
                    "autosave.sequence");
            }
        }

        if (options.role == ContainerRole::Master) {
            std::error_code exists_error;
            const bool destination_exists =
                std::filesystem::exists(
                    path, exists_error);
            if (exists_error) {
                return writer_error(
                    lfs::ErrorCode::PermissionDenied,
                    path,
                    "The existing project could not be inspected.",
                    std::format(
                        "filesystem::exists failed: {}",
                        exists_error.message()),
                    "project.path");
            }
            if (destination_exists) {
                auto existing =
                    ProjectReader::open(path);
                if (!existing) {
                    return std::move(existing)
                        .error();
                }
                const WriteCompatibility
                    compatibility =
                        existing
                            ->write_compatibility();
                if (!compatibility.safe) {
                    return writer_error(
                        lfs::ErrorCode::Unsupported,
                        path,
                        "This project is read-only in the current LichtFeld version.",
                        std::format(
                            "replacement refused before writing project bytes: {}",
                            compatibility.reasons
                                    .empty()
                                ? std::string{
                                      "unknown writer incompatibility"}
                                : compatibility.reasons.front()),
                        "commit.write_compatibility");
                }
            }
        }

        auto project_uuid =
            assigned_uuid(options.project_uuid, path, "project_uuid");
        if (!project_uuid) {
            return std::move(project_uuid).error();
        }
        auto file_uuid = assigned_uuid(options.file_uuid, path, "file_uuid");
        if (!file_uuid) {
            return std::move(file_uuid).error();
        }
        if (options.role == ContainerRole::Master) {
            if (!options.base_explicit_commit_uuid.is_nil() ||
                options.autosave_sequence != 0 ||
                !options.sidecar_snapshot_uuid.is_nil()) {
                return writer_error(
                    lfs::ErrorCode::InvalidArgument, path,
                    "Master project sidecar bindings must be empty.",
                    "base commit UUID, autosave sequence, and sidecar snapshot "
                    "UUID must all be zero for role MASTER",
                    "superblock.sidecar_binding");
            }
        } else {
            if (options.base_explicit_commit_uuid.is_nil() ||
                options.autosave_sequence == 0 ||
                options.sidecar_snapshot_uuid.is_nil()) {
                return writer_error(
                    lfs::ErrorCode::InvalidArgument, path,
                    "Autosave sidecar bindings are incomplete.",
                    "role AUTOSAVE_SIDECAR requires a base commit UUID, a "
                    "positive sequence, and a snapshot UUID",
                    "superblock.sidecar_binding");
            }
        }

        auto impl = std::make_unique<Impl>();
        impl->mode = Impl::Mode::Create;
        impl->destination_path = path;
        impl->active_path =
            detail::make_sibling_temp_path(path, "project-write");
        impl->lock = std::move(held_lock);
        impl->lock_lease =
            std::move(held_lease);
        impl->superblock = SuperblockInfo{
            .format = CURRENT_CONTAINER_VERSION,
            .role = options.role,
            .project_uuid = *project_uuid,
            .file_uuid = *file_uuid,
            .creation_time_unix_ns =
                options.creation_time_unix_ns != 0
                    ? options.creation_time_unix_ns
                    : detail::unix_time_ns(),
            .base_explicit_commit_uuid =
                options.base_explicit_commit_uuid,
            .autosave_sequence = options.autosave_sequence,
            .sidecar_snapshot_uuid = options.sidecar_snapshot_uuid,
            .crc32c = 0,
        };
        impl->disk_reserve_bytes = options.disk_reserve_bytes;
        impl->index_compression = options.index_compression;
        impl->observer = options.boundary_observer;
        impl->generation = 1;
        impl->head_sequence = 1;
        impl->head_slot = 0;
        impl->cursor = APPEND_REGION_OFFSET;

        if (auto current = impl->notify(CommitBoundary::CurrentHeadValidated);
            !current) {
            return std::move(current).error();
        }
        auto file_result = detail::NativeFile::create_new(impl->active_path);
        if (!file_result) {
            return std::move(file_result).error();
        }
        impl->file = *file_result;
        if (auto truncate = impl->file->truncate(APPEND_REGION_OFFSET);
            !truncate) {
            return std::move(truncate).error();
        }
        auto raw_superblock = encode_superblock(impl->superblock);
        impl->superblock.crc32c =
            crc32c(0, raw_superblock.data(), 252);
        if (auto write =
                impl->file->write_exact(0, byte_span(raw_superblock));
            !write) {
            return std::move(write).error();
        }
        impl->original_physical_size = APPEND_REGION_OFFSET;
        return ProjectWriter(std::move(impl));
    }

    lfs::Result<ProjectWriter>
    ProjectWriter::append(const std::filesystem::path& path,
                          const AppendOptions& options) {
        if (path.empty()) {
            return writer_error(lfs::ErrorCode::InvalidArgument, path,
                                "The project path is empty.",
                                "append requires a destination path",
                                "project.path");
        }
        std::optional<detail::WriterLock> held_lock;
        std::optional<WriterLockLease> held_lease;
        if (options.writer_lock_lease) {
            if (!options.writer_lock_lease->owns(path)) {
                return writer_error(
                    lfs::ErrorCode::FailedPrecondition,
                    path,
                    "The supplied project writer lock is not held for this destination.",
                    "writer_lock_lease must own the append destination",
                    "writer_lock");
            }
            held_lease =
                *options.writer_lock_lease;
        } else {
            auto lock_result =
                detail::WriterLock::acquire(
                    path,
                    options.writer_lock_wait);
            if (!lock_result) {
                return std::move(lock_result)
                    .error();
            }
            held_lock.emplace(
                std::move(*lock_result));
        }
        auto reader_result = ProjectReader::open(path, options.compatibility);
        if (!reader_result) {
            return std::move(reader_result).error();
        }
        if (reader_result->open_state() != OpenState::Open) {
            return writer_error(
                lfs::ErrorCode::Unsupported, path,
                "This project cannot be modified by the current LichtFeld version.",
                "semantic open did not select a supported generation",
                "commit.read_compatibility");
        }
        const WriteCompatibility compatibility =
            reader_result->write_compatibility();
        if (!compatibility.safe) {
            std::string reasons;
            for (std::size_t index = 0; index < compatibility.reasons.size();
                 ++index) {
                if (index != 0) {
                    reasons += "; ";
                }
                reasons += compatibility.reasons[index];
            }
            return writer_error(
                lfs::ErrorCode::Unsupported, path,
                "This project is read-only in the current LichtFeld version.",
                std::format("append refused before writing project bytes: {}",
                            reasons),
                "commit.write_compatibility");
        }
        if (reader_result->superblock().role != ContainerRole::Master) {
            return writer_error(
                lfs::ErrorCode::FailedPrecondition, path,
                "Autosave sidecars are replaced, not appended.",
                "role AUTOSAVE_SIDECAR must be recreated as a one-generation "
                "temporary file",
                "superblock.container_role");
        }
        if (reader_result->commit().generation ==
                std::numeric_limits<std::uint64_t>::max() ||
            reader_result->selected_head().head_sequence ==
                std::numeric_limits<std::uint64_t>::max()) {
            return writer_error(
                lfs::ErrorCode::ResourceExhausted, path,
                "The project generation counter is exhausted.",
                "generation and head_sequence must increase monotonically",
                "head.sequence");
        }

        auto file_result = detail::NativeFile::open_read_write(path);
        if (!file_result) {
            return std::move(file_result).error();
        }
        auto physical_size = (*file_result)->size();
        if (!physical_size) {
            return std::move(physical_size).error();
        }

        auto impl = std::make_unique<Impl>();
        impl->mode = Impl::Mode::Append;
        impl->destination_path = path;
        impl->active_path = path;
        impl->lock = std::move(held_lock);
        impl->lock_lease =
            std::move(held_lease);
        impl->file = *file_result;
        impl->superblock = reader_result->superblock();
        impl->generation = reader_result->commit().generation + 1;
        impl->head_sequence =
            reader_result->selected_head().head_sequence + 1;
        impl->head_slot = 1u - reader_result->selected_head().slot_id;
        impl->parent_commit_uuid = reader_result->commit().commit_uuid;
        impl->parent_commit_offset = reader_result->commit().offset;
        impl->cursor = reader_result->commit().committed_file_end;
        impl->original_physical_size = *physical_size;
        impl->disk_reserve_bytes = options.disk_reserve_bytes;
        impl->index_compression = options.index_compression;
        impl->observer = options.boundary_observer;
        for (const ChunkInfo& row : reader_result->chunks()) {
            if (row.row_kind == RowKind::Live) {
                impl->unresolved.insert(row.key);
            } else {
                impl->rows.emplace(row.key, row);
            }
        }
        impl->prior_reader.emplace(std::move(*reader_result));

        if (auto current = impl->notify(CommitBoundary::CurrentHeadValidated);
            !current) {
            return std::move(current).error();
        }
        return ProjectWriter(std::move(impl));
    }

    lfs::Result<void>
    ProjectWriter::plan_commit(const CommitOptions& requested_options) {
        if (impl_->committed || impl_->preflight_complete ||
            impl_->mutation_started || impl_->streaming ||
            impl_->commit_plan.has_value()) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition,
                impl_->destination_path,
                "The project commit plan is out of sequence.",
                "plan_commit must run exactly once after authority validation "
                "and before disk preflight",
                "writer.state"));
        }
        if (impl_->poisoned) {
            return impl_->require_ready();
        }
        if (impl_->superblock.role == ContainerRole::Master) {
            if (requested_options.kind == CommitKind::Autosave) {
                return status_failure(writer_error(
                    lfs::ErrorCode::InvalidArgument,
                    impl_->destination_path,
                    "An autosave commit cannot be published in a master file.",
                    "AUTOSAVE is valid only in role AUTOSAVE_SIDECAR",
                    "commit.kind"));
            }
            if (requested_options.kind == CommitKind::Compaction &&
                impl_->mode != Impl::Mode::Create) {
                return status_failure(writer_error(
                    lfs::ErrorCode::InvalidArgument,
                    impl_->destination_path,
                    "A compaction commit must start a new file incarnation.",
                    "COMPACTION is a generation-1 root",
                    "commit.kind"));
            }
        } else if (requested_options.kind != CommitKind::Autosave ||
                   impl_->generation != 1) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument,
                impl_->destination_path,
                "An autosave sidecar must contain one autosave root.",
                "sidecar commit kind must be AUTOSAVE at generation 1",
                "commit.kind"));
        }
        if (impl_->index_compression == IndexCompression::Zstd &&
            (requested_options.min_reader_version >
                 CURRENT_CONTAINER_VERSION ||
             requested_options.min_safe_writer_version >
                 CURRENT_CONTAINER_VERSION ||
             !supported_reader_capabilities().contains_all(
                 requested_options.extra_reader_capabilities) ||
             !supported_writer_capabilities().contains_all(
                 requested_options.extra_writer_capabilities))) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument,
                impl_->destination_path,
                "The production writer cannot publish requirements it does "
                "not implement.",
                "future version/capability envelopes are available only to "
                "the deterministic conformance-fixture encoding",
                "commit.compatibility"));
        }

        auto commit_uuid = assigned_uuid(
            requested_options.commit_uuid, impl_->destination_path,
            "commit_uuid");
        if (!commit_uuid) {
            return status_failure(std::move(commit_uuid).error());
        }
        lfs::Result<lfs::core::Uuid> snapshot_result =
            impl_->superblock.role == ContainerRole::AutosaveSidecar
                ? lfs::Result<lfs::core::Uuid>(
                      impl_->superblock.sidecar_snapshot_uuid)
                : assigned_uuid(requested_options.snapshot_uuid,
                                impl_->destination_path, "snapshot_uuid");
        if (!snapshot_result) {
            return status_failure(std::move(snapshot_result).error());
        }
        if (impl_->superblock.role == ContainerRole::AutosaveSidecar &&
            !requested_options.snapshot_uuid.is_nil() &&
            requested_options.snapshot_uuid != *snapshot_result) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument,
                impl_->destination_path,
                "The sidecar snapshot identity is inconsistent.",
                "commit snapshot UUID must equal the superblock sidecar "
                "snapshot UUID",
                "commit.snapshot_uuid"));
        }

        CommitOptions normalized = requested_options;
        normalized.commit_uuid = *commit_uuid;
        normalized.snapshot_uuid = *snapshot_result;
        impl_->commit_plan = std::move(normalized);
        return impl_->notify(CommitBoundary::IdentitiesAssigned);
    }

    lfs::Result<void>
    ProjectWriter::preflight(const std::uint64_t planned_stored_payload_bytes) {
        if (impl_->committed || impl_->mutation_started ||
            impl_->preflight_complete) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition,
                impl_->destination_path,
                "Project save disk preflight is out of sequence.",
                "preflight must run exactly once before the first chunk mutation",
                "writer.state"));
        }
        if (!impl_->commit_plan.has_value()) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition,
                impl_->destination_path,
                "The project commit identities have not been assigned.",
                "call plan_commit() before disk preflight",
                "writer.state"));
        }
        auto with_metadata = detail::checked_add(
            planned_stored_payload_bytes, METADATA_PREFLIGHT_BYTES,
            impl_->destination_path, 0, "preflight.metadata_allowance");
        if (!with_metadata) {
            return status_failure(std::move(with_metadata).error());
        }
        auto required = detail::checked_add(
            *with_metadata, impl_->disk_reserve_bytes,
            impl_->destination_path, 0, "preflight.reserve");
        if (!required) {
            return status_failure(std::move(required).error());
        }
        if (auto space =
                detail::preflight_disk_space(impl_->active_path, *required);
            !space) {
            return space;
        }
        if (impl_->mode == Impl::Mode::Append &&
            impl_->original_physical_size > impl_->cursor) {
            if (auto truncate = impl_->file->truncate(impl_->cursor);
                !truncate) {
                return truncate;
            }
            impl_->original_physical_size = impl_->cursor;
        }
        impl_->preflight_complete = true;
        return impl_->notify(CommitBoundary::PreflightComplete);
    }

    lfs::Result<void>
    ProjectWriter::write_chunk(const ChunkKey& key,
                               const std::span<const std::byte> payload,
                               const ChunkWriteOptions& options) {
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (auto valid = validate_key(key, impl_->destination_path); !valid) {
            return valid;
        }
        if (options.chunk_version == 0) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "The project chunk version is invalid.",
                "live chunks require chunk_version >= 1",
                "chunk.chunk_version"));
        }
        if (impl_->touched.contains(key)) {
            return status_failure(writer_error(
                lfs::ErrorCode::AlreadyExists, impl_->destination_path,
                "The project chunk was resolved more than once.",
                "a key may be written only once per generation",
                "chunk_key"));
        }

        std::vector<std::byte> compressed;
        std::span<const std::byte> stored = payload;
        ChunkWriteOptions placed_options = options;
        if (options.compression == Compression::ByteShuffleZstdFramed) {
            // Deterministic fallback: non-multiple-of-4 payloads cannot be
            // f32-word plane-shuffled; emit framed Zstd instead (no knob).
            if (payload.size() % 4 != 0) {
                placed_options.compression = Compression::ZstdFramed;
            } else {
                const std::vector<std::byte> planes = byte_plane_f32_words(payload);
                auto result = frame_zstd(planes, impl_->active_path,
                                         "chunk.byteshuffle_zstd_framed",
                                         ZSTD_PAYLOAD_LEVEL);
                if (!result) {
                    return status_failure(std::move(result).error());
                }
                compressed = std::move(*result);
                stored = compressed;
            }
        }
        if (placed_options.compression == Compression::ZstdFramed) {
            auto result = frame_zstd(payload, impl_->active_path, "chunk.zstd_framed",
                                     ZSTD_PAYLOAD_LEVEL);
            if (!result) {
                return status_failure(std::move(result).error());
            }
            compressed = std::move(*result);
            stored = compressed;
        } else if (placed_options.compression != Compression::Stored &&
                   placed_options.compression != Compression::ZstdFramed &&
                   placed_options.compression != Compression::ByteShuffleZstdFramed) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "The project chunk compression is invalid.",
                "compression must be STORED, ZSTD_FRAMED, or BYTESHUFFLE_ZSTD_FRAMED",
                "chunk.compression"));
        }

        auto placed = impl_->place_stored_chunk(key, stored, payload.size(),
                                                placed_options);
        if (!placed) {
            impl_->poisoned = true;
            return status_failure(std::move(placed).error());
        }
        if (auto touched = impl_->touch_key(key); !touched) {
            impl_->poisoned = true;
            return touched;
        }
        impl_->rows[key] = std::move(*placed);
        return {};
    }

    lfs::Result<void>
    ProjectWriter::set_preview(const std::span<const std::byte> png_bytes) {
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (impl_->superblock.role != ContainerRole::Master) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "Autosave sidecars cannot regenerate the project preview.",
                "publish the preview in a master generation; sidecars retain no "
                "head locator",
                "head.preview_locator"));
        }
        if (png_bytes.empty() || png_bytes.size() > MAX_PREVIEW_BYTES) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "The project preview size is invalid.",
                std::format("PNG preview bytes must be between 1 and {}, got {}",
                            MAX_PREVIEW_BYTES, png_bytes.size()),
                "head.preview_bytes"));
        }

        ChunkKey key{.fourcc = FOURCC_THMB,
                     .instance_uuid = impl_->superblock.project_uuid};
        if (impl_->prior_reader.has_value() &&
            impl_->prior_reader->preview().has_value()) {
            const PreviewLocator& prior_locator =
                *impl_->prior_reader->preview();
            const auto prior_row = std::find_if(
                impl_->prior_reader->chunks().begin(),
                impl_->prior_reader->chunks().end(),
                [&](const ChunkInfo& row) {
                    return row.row_kind == RowKind::Live &&
                           row.key.fourcc == FOURCC_THMB &&
                           row.payload_offset == prior_locator.offset &&
                           row.stored_bytes == prior_locator.bytes;
                });
            assert(prior_row != impl_->prior_reader->chunks().end());
            key = prior_row->key;
        }

        ChunkWriteOptions options{
            .chunk_version = 1,
            .compression = Compression::Stored,
            .tensor_payload = false,
            .block_crcs = false,
            .expected_stream_bytes = std::nullopt,
        };
        if (auto written = write_chunk(key, png_bytes, options); !written) {
            return written;
        }
        const ChunkInfo& row = impl_->rows.at(key);
        assert(row.payload_offset % CHUNK_ALIGNMENT == 0);
        assert(row.stored_bytes == png_bytes.size());
        impl_->preview_locator = PreviewLocator{
            .offset = row.payload_offset,
            .bytes = static_cast<std::uint32_t>(row.stored_bytes),
            .format = PreviewFormat::Png,
        };
        return {};
    }

    lfs::Result<std::ostream*>
    ProjectWriter::begin_chunk(const ChunkKey& key,
                               const ChunkWriteOptions& options) {
        if (auto ready = impl_->require_ready(); !ready) {
            return std::move(ready).error();
        }
        if (auto valid = validate_key(key, impl_->destination_path); !valid) {
            return std::move(valid).error();
        }
        if (options.chunk_version == 0) {
            return writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "The project chunk version is invalid.",
                "live chunks require chunk_version >= 1",
                "chunk.chunk_version");
        }
        if (options.compression != Compression::Stored) {
            return writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "Streaming project chunks must use stored encoding.",
                "begin_chunk streams bytes directly; use write_chunk for compressed encodings",
                "chunk.compression");
        }
        if (impl_->touched.contains(key)) {
            return writer_error(
                lfs::ErrorCode::AlreadyExists, impl_->destination_path,
                "The project chunk was resolved more than once.",
                "a key may be written only once per generation",
                "chunk_key");
        }

        const bool mandatory_blocks =
            options.expected_stream_bytes.has_value() &&
            *options.expected_stream_bytes >= BLOCK_CRC_REQUIRED_AT;
        const bool with_blocks = options.block_crcs || mandatory_blocks;
        if (with_blocks &&
            (!options.expected_stream_bytes.has_value() ||
             *options.expected_stream_bytes == 0)) {
            return writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "Streaming block CRC layout needs the payload size in advance.",
                "set expected_stream_bytes to a positive exact byte count",
                "chunk.expected_stream_bytes");
        }

        auto header_offset =
            align_up(impl_->cursor, CHUNK_ALIGNMENT, impl_->active_path,
                     "chunk.header_offset");
        if (!header_offset) {
            return std::move(header_offset).error();
        }
        auto table_offset = detail::checked_add(
            *header_offset, CHUNK_HEADER_BYTES, impl_->active_path,
            *header_offset, "chunk.table_offset");
        if (!table_offset) {
            return std::move(table_offset).error();
        }
        std::uint64_t table_bytes = 0;
        if (with_blocks) {
            const std::uint64_t block_count =
                *options.expected_stream_bytes / BLOCK_CRC_BYTES +
                (*options.expected_stream_bytes % BLOCK_CRC_BYTES != 0 ? 1
                                                                       : 0);
            auto entries_bytes = detail::checked_multiply(
                block_count, sizeof(std::uint32_t), impl_->active_path,
                *table_offset, "block_table.entries");
            if (!entries_bytes) {
                return std::move(entries_bytes).error();
            }
            auto total = detail::checked_add(
                BLOCK_CRC_HEADER_BYTES, *entries_bytes, impl_->active_path,
                *table_offset, "block_table.bytes");
            if (!total) {
                return std::move(total).error();
            }
            table_bytes = *total;
        }
        auto after_metadata = detail::checked_add(
            *table_offset, table_bytes, impl_->active_path, *table_offset,
            "chunk.payload_pre_alignment");
        if (!after_metadata) {
            return std::move(after_metadata).error();
        }
        auto payload_offset = align_up(
            *after_metadata,
            options.tensor_payload ? TENSOR_PAYLOAD_ALIGNMENT : CHUNK_ALIGNMENT,
            impl_->active_path, "chunk.payload_offset");
        if (!payload_offset) {
            return std::move(payload_offset).error();
        }
        if (auto zero =
                write_zeros(*impl_->file, impl_->cursor, *payload_offset);
            !zero) {
            impl_->poisoned = true;
            return std::move(zero).error();
        }

        ChunkInfo row{
            .key = key,
            .chunk_version = options.chunk_version,
            .row_kind = RowKind::Live,
            .compression = Compression::Stored,
            .flags = (options.tensor_payload ? TENSOR_PAYLOAD : 0u) |
                     (with_blocks ? HAS_BLOCK_CRCS : 0u),
            .header_offset = *header_offset,
            .payload_offset = *payload_offset,
            .stored_bytes = 0,
            .uncompressed_bytes = 0,
            .source_generation = impl_->generation,
            .payload_crc32c = 0,
            .header_crc32c = 0,
            .block_crc_table = std::nullopt,
        };
        impl_->streaming = std::make_unique<Impl::StreamingChunk>(
            key, options, row, *table_offset, table_bytes, impl_->file);
        impl_->mutation_started = true;
        return &impl_->streaming->stream;
    }

    lfs::Result<void> ProjectWriter::end_chunk() {
        if (!impl_->streaming) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "No streaming project chunk is open.",
                "begin_chunk() must precede end_chunk()",
                "writer.streaming_chunk"));
        }
        if (impl_->poisoned) {
            return impl_->require_ready();
        }
        Impl::StreamingChunk& streaming = *impl_->streaming;
        streaming.stream.flush();
        if (!streaming.stream || streaming.buffer.error().has_value()) {
            impl_->poisoned = true;
            if (streaming.buffer.error().has_value()) {
                return status_failure(*streaming.buffer.error());
            }
            return status_failure(writer_error(
                lfs::ErrorCode::Internal, impl_->destination_path,
                "The streaming project chunk could not be finalized.",
                "ostream flush failed",
                "writer.streaming_chunk"));
        }
        streaming.buffer.finish_partial_block();
        const std::uint64_t stored_bytes = streaming.buffer.count();
        if (streaming.options.expected_stream_bytes.has_value() &&
            stored_bytes != *streaming.options.expected_stream_bytes) {
            impl_->poisoned = true;
            return status_failure(writer_error(
                lfs::ErrorCode::ContractViolation,
                impl_->destination_path,
                "The streaming project chunk size did not match its plan.",
                std::format("expected {} bytes, streamed {}",
                            *streaming.options.expected_stream_bytes,
                            stored_bytes),
                "chunk.expected_stream_bytes"));
        }
        if (stored_bytes >= BLOCK_CRC_REQUIRED_AT &&
            (streaming.row.flags & HAS_BLOCK_CRCS) == 0) {
            impl_->poisoned = true;
            return status_failure(writer_error(
                lfs::ErrorCode::ContractViolation,
                impl_->destination_path,
                "The large project chunk is missing mandatory block CRCs.",
                "streams that may reach 1 GiB must declare expected_stream_bytes",
                "chunk.block_crcs"));
        }
        auto payload_end = detail::checked_add(
            streaming.row.payload_offset, stored_bytes, impl_->active_path,
            streaming.row.payload_offset, "chunk.payload_end");
        if (!payload_end) {
            impl_->poisoned = true;
            return status_failure(std::move(payload_end).error());
        }

        streaming.row.stored_bytes = stored_bytes;
        streaming.row.uncompressed_bytes = stored_bytes;
        streaming.row.payload_crc32c = streaming.buffer.crc();
        if ((streaming.row.flags & HAS_BLOCK_CRCS) != 0) {
            const auto& entries = streaming.buffer.block_entries();
            const std::uint64_t expected_entries =
                stored_bytes / BLOCK_CRC_BYTES +
                (stored_bytes % BLOCK_CRC_BYTES != 0 ? 1 : 0);
            if (entries.size() != expected_entries) {
                impl_->poisoned = true;
                return status_failure(writer_error(
                    lfs::ErrorCode::ContractViolation,
                    impl_->destination_path,
                    "The streaming block CRC table is incomplete.",
                    std::format("expected {} entries, accumulated {}",
                                expected_entries, entries.size()),
                    "block_table.entries"));
            }
            streaming.row.block_crc_table = BlockCrcTable{
                .offset = streaming.table_offset,
                .payload_offset = streaming.row.payload_offset,
                .stored_bytes = stored_bytes,
                .block_size =
                    static_cast<std::uint32_t>(BLOCK_CRC_BYTES),
                .entries = entries,
                .entries_crc32c = crc_entries(entries),
                .header_crc32c = 0,
            };
            const auto table_header =
                encode_block_crc_header(*streaming.row.block_crc_table);
            streaming.row.block_crc_table->header_crc32c =
                crc32c(0, table_header.data(), 60);
            if (auto table = impl_->write_block_table(streaming.row); !table) {
                impl_->poisoned = true;
                return table;
            }
        }
        const auto header = encode_chunk_header(streaming.row);
        streaming.row.header_crc32c = crc32c(0, header.data(), 60);
        if (auto write = impl_->file->write_exact(streaming.row.header_offset,
                                                  byte_span(header));
            !write) {
            impl_->poisoned = true;
            return write;
        }
        impl_->cursor = *payload_end;
        const ChunkKey key = streaming.key;
        ChunkInfo row = std::move(streaming.row);
        impl_->streaming.reset();
        if (auto touched = impl_->touch_key(key); !touched) {
            impl_->poisoned = true;
            return touched;
        }
        impl_->rows[key] = std::move(row);
        return {};
    }

    lfs::Result<void>
    ProjectWriter::reuse_if_clean(const CleanProof& proof,
                                  const std::uint64_t current_mutation_epoch) {
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (!impl_->prior_reader.has_value()) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "A new project has no prior payload span to reuse.",
                "clean proofs apply only to a validated same-file append",
                "clean_proof"));
        }
        if (proof.mutation_epoch_ != current_mutation_epoch) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "The project chapter changed after its clean proof was issued.",
                std::format("proof epoch {} does not match current epoch {}; "
                            "the payload must be rewritten",
                            proof.mutation_epoch_, current_mutation_epoch),
                "clean_proof.mutation_epoch"));
        }
        if (proof.file_uuid_ != impl_->superblock.file_uuid ||
            proof.commit_uuid_ != impl_->parent_commit_uuid) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "The clean proof belongs to a different project generation.",
                "file UUID and selected commit UUID must match the locked "
                "append authority",
                "clean_proof.authority"));
        }
        const ChunkInfo* current =
            impl_->prior_reader->find(proof.key_);
        if (current == nullptr || current->row_kind != RowKind::Live ||
            current->source_generation != proof.source_generation_ ||
            current->payload_crc32c != proof.payload_crc32c_ ||
            current->header_crc32c != proof.header_crc32c_) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "The clean proof no longer identifies the selected payload.",
                "key, source generation, payload CRC, and header CRC must all "
                "match the pinned reader",
                "clean_proof.content"));
        }
        if (impl_->touched.contains(proof.key_)) {
            return status_failure(writer_error(
                lfs::ErrorCode::AlreadyExists, impl_->destination_path,
                "The project chunk was resolved more than once.",
                "a clean proof cannot replace another resolution in the same "
                "generation",
                "chunk_key"));
        }
        impl_->rows[proof.key_] = *current;
        if (auto touched = impl_->touch_key(proof.key_); !touched) {
            return touched;
        }
        impl_->used_clean_proof_reuse = true;
        if (impl_->prior_reader->preview().has_value()) {
            const PreviewLocator& prior_preview =
                *impl_->prior_reader->preview();
            if (current->key.fourcc == FOURCC_THMB &&
                current->compression == Compression::Stored &&
                current->payload_offset == prior_preview.offset &&
                current->stored_bytes == prior_preview.bytes) {
                impl_->preview_locator = prior_preview;
            }
        }
        impl_->mutation_started = true;
        return {};
    }

    lfs::Result<void>
    ProjectWriter::carry_forward_opaque(
        const ChunkInfo& chunk, const CleanProof& proof,
        const std::uint64_t current_mutation_epoch) {
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (!impl_->prior_reader.has_value()) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "A new project has no opaque payload to carry forward.",
                "opaque carry-forward applies only to a validated same-file "
                "append",
                "opaque_chunk"));
        }
        const ChunkInfo* current = impl_->prior_reader->find(chunk.key);
        const bool exact_match =
            current != nullptr && current->row_kind == RowKind::Live &&
            current->chunk_version == chunk.chunk_version &&
            current->compression == chunk.compression &&
            current->flags == chunk.flags &&
            current->header_offset == chunk.header_offset &&
            current->payload_offset == chunk.payload_offset &&
            current->stored_bytes == chunk.stored_bytes &&
            current->uncompressed_bytes == chunk.uncompressed_bytes &&
            current->source_generation == chunk.source_generation &&
            current->payload_crc32c == chunk.payload_crc32c &&
            current->header_crc32c == chunk.header_crc32c;
        if (!exact_match) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "The opaque chunk is not the pinned generation's exact row.",
                "opaque carry-forward never accepts reconstructed or cross-file "
                "metadata",
                "opaque_chunk"));
        }
        if (proof.key_ != chunk.key) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "The opaque clean proof identifies a different chunk.",
                "opaque carry-forward requires an epoch-bound proof for the "
                "exact pinned row",
                "opaque_chunk.clean_proof"));
        }
        if (auto reused = reuse_if_clean(proof, current_mutation_epoch);
            !reused) {
            return reused;
        }
        impl_->used_opaque_carry_forward = true;
        return {};
    }

    lfs::Result<void>
    ProjectWriter::add_sidecar_base_reference(const ChunkInfo& base) {
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (impl_->superblock.role != ContainerRole::AutosaveSidecar ||
            impl_->mode != Impl::Mode::Create) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, impl_->destination_path,
                "Base references are valid only in a new autosave sidecar.",
                "master indexes and appended generations cannot contain row "
                "kind SIDECAR_BASE_REFERENCE",
                "index.row_kind"));
        }
        if (auto valid = validate_key(base.key, impl_->destination_path); !valid) {
            return valid;
        }
        if (base.row_kind != RowKind::Live || base.chunk_version == 0 ||
            base.source_generation == 0) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "The sidecar base row is not a live master chunk.",
                "base references must echo a live row from the bound master "
                "commit",
                "index.base_reference"));
        }
        if (impl_->rows.contains(base.key) ||
            impl_->touched.contains(base.key)) {
            return status_failure(writer_error(
                lfs::ErrorCode::AlreadyExists, impl_->destination_path,
                "The sidecar key was seeded more than once.",
                "each bound-master key has exactly one overlay row",
                "chunk_key"));
        }
        ChunkInfo reference = base;
        reference.row_kind = RowKind::SidecarBaseReference;
        reference.header_offset = 0;
        reference.payload_offset = 0;
        reference.block_crc_table.reset();
        impl_->rows.emplace(reference.key, std::move(reference));
        impl_->seeded_base_keys.insert(base.key);
        impl_->mutation_started = true;
        return {};
    }

    lfs::Result<void>
    ProjectWriter::copy_chunk_verbatim(
        const ProjectReader& source,
        const ChunkInfo& chunk) {
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (chunk.row_kind != RowKind::Live) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument,
                impl_->destination_path,
                "Only a live chunk can be copied verbatim.",
                "tombstones and base references have no stored payload",
                "chunk.row_kind"));
        }
        if (impl_->rows.contains(chunk.key) ||
            impl_->touched.contains(chunk.key)) {
            return status_failure(writer_error(
                lfs::ErrorCode::AlreadyExists,
                impl_->destination_path,
                "The project chunk was resolved more than once.",
                "verbatim copy requires one untouched logical key",
                "chunk_key"));
        }
        auto copied =
            impl_->copy_stored_chunk(source, chunk);
        if (!copied) {
            impl_->poisoned = true;
            return status_failure(
                std::move(copied).error());
        }
        if (source.preview().has_value() &&
            chunk.key.fourcc == FOURCC_THMB &&
            chunk.payload_offset ==
                source.preview()->offset &&
            chunk.stored_bytes ==
                source.preview()->bytes) {
            impl_->preview_locator = PreviewLocator{
                .offset = copied->payload_offset,
                .bytes = static_cast<std::uint32_t>(
                    copied->stored_bytes),
                .format = PreviewFormat::Png,
            };
        }
        impl_->rows[copied->key] =
            std::move(*copied);
        if (auto touched = impl_->touch_key(chunk.key);
            !touched) {
            return touched;
        }
        return {};
    }

    lfs::Result<void> ProjectWriter::erase(const ChunkKey& key) {
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (auto valid = validate_key(key, impl_->destination_path); !valid) {
            return valid;
        }
        if (impl_->touched.contains(key)) {
            return status_failure(writer_error(
                lfs::ErrorCode::AlreadyExists, impl_->destination_path,
                "The project chunk was resolved more than once.",
                "a key may be erased only once per generation",
                "chunk_key"));
        }
        const bool prior_live = impl_->unresolved.contains(key);
        const bool seeded_base = impl_->seeded_base_keys.contains(key);
        const auto existing = impl_->rows.find(key);
        if (!prior_live && !seeded_base &&
            (existing == impl_->rows.end() ||
             existing->second.row_kind == RowKind::Tombstone)) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument, impl_->destination_path,
                "The project chunk cannot be deleted because it is not live.",
                "tombstones identify a live same-file or bound-base key",
                "chunk_key"));
        }
        impl_->rows[key] = ChunkInfo{
            .key = key,
            .chunk_version = 0,
            .row_kind = RowKind::Tombstone,
            .compression = Compression::Stored,
            .flags = 0,
            .header_offset = 0,
            .payload_offset = 0,
            .stored_bytes = 0,
            .uncompressed_bytes = 0,
            .source_generation = impl_->generation,
            .payload_crc32c = 0,
            .header_crc32c = 0,
            .block_crc_table = std::nullopt,
        };
        if (auto touched = impl_->touch_key(key); !touched) {
            return touched;
        }
        impl_->mutation_started = true;
        return {};
    }

    lfs::Result<void>
    ProjectWriter::commit() {
        const auto commit_started = std::chrono::steady_clock::now();
        if (auto ready = impl_->require_ready(); !ready) {
            return ready;
        }
        if (!impl_->commit_plan.has_value()) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition,
                impl_->destination_path,
                "The project commit identities have not been assigned.",
                "call plan_commit() before publication",
                "writer.state"));
        }
        const CommitOptions& options = *impl_->commit_plan;
        if (!impl_->unresolved.empty()) {
            const auto dirty_checkpoint = std::find_if(
                impl_->unresolved.begin(), impl_->unresolved.end(),
                [](const ChunkKey& key) { return key.fourcc == FOURCC_CKPT; });
            if (dirty_checkpoint != impl_->unresolved.end()) {
                return status_failure(writer_error(
                    lfs::ErrorCode::FailedPrecondition,
                    impl_->destination_path,
                    "The save requires a fresh training checkpoint.",
                    "metadata-only publication was escalated because CKPT has "
                    "neither rewritten bytes nor an epoch-bound clean proof",
                    "commit.dirty_ckpt"));
            }
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition,
                impl_->destination_path,
                "The save has unresolved project chapters.",
                std::format("{} prior live row(s) have neither rewritten bytes, "
                            "an epoch-bound clean proof, nor audited opaque "
                            "carry-forward",
                            impl_->unresolved.size()),
                "commit.clean_proof"));
        }
        const lfs::core::Uuid commit_uuid = options.commit_uuid;
        const lfs::core::Uuid snapshot_uuid = options.snapshot_uuid;

        CapabilitySet reader_capabilities = derive_reader_capabilities(
            impl_->rows, impl_->superblock.role, impl_->index_compression);
        const bool preserve_legacy_fixture_envelope =
            impl_->index_compression ==
            IndexCompression::StoredForDeterministicTests;
        CapabilitySet writer_capabilities = derive_writer_capabilities(
            impl_->superblock.role, impl_->used_opaque_carry_forward,
            impl_->used_clean_proof_reuse, preserve_legacy_fixture_envelope);
        reader_capabilities |= options.extra_reader_capabilities;
        writer_capabilities |= options.extra_writer_capabilities;
        if (impl_->prior_reader.has_value()) {
            reader_capabilities |=
                impl_->prior_reader->commit().required_reader_capabilities;
            CapabilitySet inherited_writer_capabilities =
                impl_->prior_reader->commit().required_writer_capabilities;
            inherited_writer_capabilities.set(OPAQUE_CHUNK_PRESERVATION, false);
            inherited_writer_capabilities.set(CLEAN_PROOF_REUSE, false);
            writer_capabilities |= inherited_writer_capabilities;
        }
        if (impl_->index_compression == IndexCompression::Zstd) {
            if (options.min_reader_version > CURRENT_CONTAINER_VERSION ||
                options.min_safe_writer_version >
                    CURRENT_CONTAINER_VERSION ||
                !supported_reader_capabilities().contains_all(
                    reader_capabilities) ||
                !supported_writer_capabilities().contains_all(
                    writer_capabilities)) {
                return status_failure(writer_error(
                    lfs::ErrorCode::InvalidArgument,
                    impl_->destination_path,
                    "The production writer cannot publish requirements it does "
                    "not implement.",
                    "future version/capability envelopes are available only to "
                    "the deterministic conformance-fixture encoding",
                    "commit.compatibility"));
            }
        }

        if (auto chunks =
                impl_->notify(CommitBoundary::ChunksWritten);
            !chunks) {
            return chunks;
        }
        auto decoded_index = encode_index(impl_->rows, impl_->generation,
                                          commit_uuid,
                                          impl_->active_path);
        if (!decoded_index) {
            return status_failure(std::move(decoded_index).error());
        }
        std::vector<std::byte> stored_index;
        Compression index_compression = Compression::Stored;
        if (impl_->index_compression == IndexCompression::Zstd) {
            // The index is intentionally one legacy zstd frame; do not apply
            // the payload framed-record layout to this metadata encoding.
            auto compressed = compress_zstd(
                *decoded_index, impl_->active_path, "index.zstd");
            if (!compressed) {
                return status_failure(std::move(compressed).error());
            }
            stored_index = std::move(*compressed);
            index_compression = Compression::ZstdFramed;
        } else {
            stored_index = *decoded_index;
        }

        auto index_offset =
            align_up(impl_->cursor, CHUNK_ALIGNMENT, impl_->active_path,
                     "commit.index_offset");
        if (!index_offset) {
            return status_failure(std::move(index_offset).error());
        }
        if (auto zero =
                write_zeros(*impl_->file, impl_->cursor, *index_offset);
            !zero) {
            impl_->poisoned = true;
            return zero;
        }
        if (auto write =
                impl_->file->write_exact(*index_offset, stored_index);
            !write) {
            impl_->poisoned = true;
            return write;
        }
        auto index_end = detail::checked_add(
            *index_offset, stored_index.size(), impl_->active_path,
            *index_offset, "commit.index_end");
        if (!index_end) {
            impl_->poisoned = true;
            return status_failure(std::move(index_end).error());
        }
        if (auto boundary =
                impl_->notify(CommitBoundary::IndexWritten);
            !boundary) {
            return boundary;
        }

        auto commit_offset =
            align_up(*index_end, CHUNK_ALIGNMENT, impl_->active_path,
                     "commit.offset");
        if (!commit_offset) {
            impl_->poisoned = true;
            return status_failure(std::move(commit_offset).error());
        }
        if (auto zero =
                write_zeros(*impl_->file, *index_end, *commit_offset);
            !zero) {
            impl_->poisoned = true;
            return zero;
        }
        auto committed_end = detail::checked_add(
            *commit_offset, COMMIT_RECORD_BYTES, impl_->active_path,
            *commit_offset, "commit.committed_file_end");
        if (!committed_end) {
            impl_->poisoned = true;
            return status_failure(std::move(committed_end).error());
        }

        const bool root = impl_->generation == 1;
        CommitInfo commit_info{
            .offset = *commit_offset,
            .kind = options.kind,
            .commit_uuid = commit_uuid,
            .generation = impl_->generation,
            .parent_commit_uuid =
                root ? lfs::core::Uuid{} : impl_->parent_commit_uuid,
            .parent_commit_offset =
                root ? 0 : impl_->parent_commit_offset,
            .explicit_ancestor_commit_uuid =
                impl_->superblock.role == ContainerRole::AutosaveSidecar
                    ? impl_->superblock.base_explicit_commit_uuid
                    : commit_uuid,
            .snapshot_uuid = snapshot_uuid,
            .wallclock_unix_ns =
                options.wallclock_unix_ns != 0
                    ? options.wallclock_unix_ns
                    : detail::unix_time_ns(),
            .index_offset = *index_offset,
            .index_stored_bytes = stored_index.size(),
            .index_uncompressed_bytes = decoded_index->size(),
            .index_stored_crc32c = crc32c(
                0, stored_index.data(), stored_index.size()),
            .index_uncompressed_crc32c = crc32c(
                0, decoded_index->data(), decoded_index->size()),
            .index_compression = index_compression,
            .committed_file_end = *committed_end,
            .min_reader_version = options.min_reader_version,
            .min_safe_writer_version = options.min_safe_writer_version,
            .required_reader_capabilities = reader_capabilities,
            .required_writer_capabilities = writer_capabilities,
            .crc32c = 0,
        };
        const auto raw_commit =
            encode_commit(impl_->superblock, commit_info);
        commit_info.crc32c = crc32c(0, raw_commit.data(), 252);
        if (auto write =
                impl_->file->write_exact(*commit_offset, byte_span(raw_commit));
            !write) {
            impl_->poisoned = true;
            return write;
        }
        if (auto boundary =
                impl_->notify(CommitBoundary::CommitWritten);
            !boundary) {
            return boundary;
        }
        const auto commit_written = std::chrono::steady_clock::now();
        if (auto flush = impl_->file->sync_data(); !flush) {
            impl_->poisoned = true;
            return flush;
        }
        const auto data_flushed = std::chrono::steady_clock::now();
        if (auto boundary =
                impl_->notify(CommitBoundary::AppendFlushed);
            !boundary) {
            return boundary;
        }

        HeadInfo head{
            .slot_id = impl_->head_slot,
            .head_sequence = impl_->head_sequence,
            .generation = impl_->generation,
            .commit_uuid = commit_uuid,
            .commit_offset = *commit_offset,
            .committed_file_end = *committed_end,
            .commit_crc32c_echo = commit_info.crc32c,
            .preview = impl_->preview_locator,
            .head_crc32c = 0,
        };
        const auto raw_head = encode_head(impl_->superblock, head);
        head.head_crc32c = crc32c(0, raw_head.data(), 4092);
        if (auto write = impl_->file->write_single(
                HEAD_SLOT_OFFSETS[impl_->head_slot], byte_span(raw_head));
            !write) {
            impl_->poisoned = true;
            return write;
        }
        if (auto boundary = impl_->notify(CommitBoundary::HeadWritten);
            !boundary) {
            return boundary;
        }
        if (auto flush = impl_->file->sync_all(); !flush) {
            impl_->poisoned = true;
            return flush;
        }
        const auto all_flushed = std::chrono::steady_clock::now();
        LOG_DEBUG(
            "Project commit stages: write_commit={:.3f} ms sync_data={:.3f} ms sync_all={:.3f} ms total={:.3f} ms",
            std::chrono::duration<double, std::milli>(commit_written - commit_started).count(),
            std::chrono::duration<double, std::milli>(data_flushed - commit_written).count(),
            std::chrono::duration<double, std::milli>(all_flushed - data_flushed).count(),
            std::chrono::duration<double, std::milli>(all_flushed - commit_started).count());
        if (auto boundary = impl_->notify(CommitBoundary::HeadFlushed);
            !boundary) {
            return boundary;
        }

        ReaderOptions validation_options;
        validation_options.reader_version =
            std::max(CURRENT_CONTAINER_VERSION,
                     options.min_reader_version);
        validation_options.writer_version =
            std::max(CURRENT_CONTAINER_VERSION,
                     options.min_safe_writer_version);
        validation_options.reader_capabilities =
            supported_reader_capabilities() | reader_capabilities;
        validation_options.writer_capabilities =
            supported_writer_capabilities() | writer_capabilities;

        auto validate_authority = [&](const std::filesystem::path& candidate)
            -> lfs::Result<void> {
            auto reader =
                ProjectReader::open(candidate, validation_options);
            if (!reader) {
                return status_failure(std::move(reader).error());
            }
            if (reader->commit().commit_uuid != commit_uuid ||
                reader->commit().generation != impl_->generation ||
                reader->commit().committed_file_end != *committed_end) {
                return status_failure(writer_error(
                    lfs::ErrorCode::DataLoss, candidate,
                    "The published project selected an unexpected generation.",
                    "post-write validation did not recover the just-published "
                    "authority tuple",
                    "commit.validation"));
            }
            return {};
        };

        auto validate_path = [&](const std::filesystem::path& candidate)
            -> lfs::Result<void> {
            const auto validation_started = std::chrono::steady_clock::now();
            if (auto authority = validate_authority(candidate); !authority) {
                return authority;
            }
            auto reader =
                ProjectReader::open(candidate, validation_options);
            if (!reader) {
                return status_failure(std::move(reader).error());
            }
            for (const ChunkInfo& row : reader->chunks()) {
                if (!row.is_live() ||
                    row.source_generation != impl_->generation) {
                    continue;
                }
                const bool payload_class =
                    (row.flags & TENSOR_PAYLOAD) != 0 ||
                    row.key.fourcc == FOURCC_CKPT ||
                    row.key.fourcc == FOURCC_PPIS;
                if (!payload_class) {
                    if (auto verified = reader->verify_chunk(row); !verified)
                        return verified;
                    continue;
                }
                std::vector<std::byte> buffer(BLOCK_CRC_BYTES);
                std::uint32_t stored_crc = 0;
                std::size_t block_index = 0;
                for (std::uint64_t relative = 0; relative < row.stored_bytes;
                     relative += std::min<std::uint64_t>(
                         BLOCK_CRC_BYTES, row.stored_bytes - relative)) {
                    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
                        BLOCK_CRC_BYTES, row.stored_bytes - relative));
                    auto block = std::span<std::byte>(buffer.data(), count);
                    if (auto read = reader->read_stored_at(row, relative, block);
                        !read) {
                        return read;
                    }
                    const auto block_crc = crc32c(0, block.data(), block.size());
                    if (row.block_crc_table.has_value() &&
                        (block_index >= row.block_crc_table->entries.size() ||
                         block_crc != row.block_crc_table->entries[block_index])) {
                        return status_failure(writer_error(
                            lfs::ErrorCode::DataLoss, candidate,
                            "The published project failed block CRC validation.",
                            std::format("chunk {} block {} does not match",
                                        row.key_string(), block_index),
                            "commit.validation"));
                    }
                    stored_crc = crc32c(stored_crc, block.data(), block.size());
                    ++block_index;
                }
                if (stored_crc != row.payload_crc32c) {
                    return status_failure(writer_error(
                        lfs::ErrorCode::DataLoss, candidate,
                        "The published project failed payload CRC validation.",
                        std::format("chunk {} does not match", row.key_string()),
                        "commit.validation"));
                }
            }
            LOG_DEBUG(
                "Project commit validation stage: path={} crc_only={:.3f} ms",
                candidate.string(),
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - validation_started)
                    .count());
            return {};
        };

        auto validation = impl_->mode == Impl::Mode::Append
                              ? validate_authority(impl_->active_path)
                              : validate_path(impl_->active_path);
        if (!validation) {
            impl_->poisoned = true;
            impl_->keep_temporary = impl_->mode == Impl::Mode::Create;
            if (impl_->mode == Impl::Mode::Append) {
                impl_->committed = true;
                impl_->cursor = *committed_end;
                lfs::Error cause = std::move(validation).error();
                lfs::Error published = writer_error(
                    lfs::ErrorCode::DataLoss, impl_->destination_path,
                    "The durable project generation failed post-publication verification.",
                    std::format(
                        "published generation={}, commit_uuid={}, "
                        "committed_file_end={}; the file is authoritative and "
                        "the caller must not retry from stale state",
                        impl_->generation, commit_uuid.to_string(),
                        *committed_end),
                    "commit.post_publish_verification");
                published =
                    std::move(published).with_suppressed(std::move(cause));
                return status_failure(std::move(published));
            }
            return validation;
        }
        if (impl_->mode == Impl::Mode::Create) {
            impl_->file.reset();
            if (auto boundary =
                    impl_->notify(
                        CommitBoundary::
                            ReplacementReady);
                !boundary) {
                return boundary;
            }
            auto replacement =
                detail::atomic_replace(impl_->active_path,
                                       impl_->destination_path);
            if (!replacement) {
                impl_->keep_temporary = true;
                return status_failure(std::move(replacement).error());
            }
            if (auto boundary =
                    impl_->notify(
                        CommitBoundary::
                            ReplacementPublished);
                !boundary) {
                impl_->record_post_publish_note(
                    std::move(boundary).error(),
                    "replacement-published observer");
            }
            if (auto validation =
                    validate_authority(impl_->destination_path);
                !validation) {
                auto rollback = detail::rollback_atomic_replace(
                    *replacement, impl_->destination_path);
                if (!rollback) {
                    lfs::Error error = std::move(validation).error();
                    error = std::move(error).with_suppressed(
                        std::move(rollback).error());
                    return status_failure(std::move(error));
                }
                return validation;
            }
            impl_->committed = true;
            impl_->cursor = *committed_end;
            if (auto boundary =
                    impl_->notify(
                        CommitBoundary::
                            ReplacementValidated);
                !boundary) {
                impl_->record_post_publish_note(
                    std::move(boundary).error(),
                    "replacement-validated observer");
            }
            if (auto finish = detail::finish_atomic_replace(
                    *replacement, impl_->destination_path);
                !finish) {
                impl_->record_post_publish_note(
                    std::move(finish).error(),
                    "replacement backup cleanup");
            }
        }

        impl_->committed = true;
        impl_->cursor = *committed_end;
        if (auto boundary = impl_->notify(CommitBoundary::Committed);
            !boundary) {
            impl_->record_post_publish_note(
                std::move(boundary).error(),
                "committed observer");
        }
        return {};
    }

    lfs::Result<void>
    ProjectWriter::compact(const std::filesystem::path& path,
                           const CompactionOptions& options) {
        if (path.empty()) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument, path,
                "The project path is empty.",
                "compact requires an existing destination path",
                "project.path"));
        }
        auto lock_result = detail::WriterLock::acquire(path);
        if (!lock_result) {
            return status_failure(std::move(lock_result).error());
        }
        auto source_result =
            ProjectReader::open(path, options.compatibility);
        if (!source_result) {
            return status_failure(std::move(source_result).error());
        }
        if (source_result->open_state() != OpenState::Open) {
            return status_failure(writer_error(
                lfs::ErrorCode::Unsupported, path,
                "This project cannot be compacted by the current LichtFeld version.",
                "semantic open did not select a supported generation",
                "commit.read_compatibility"));
        }
        const WriteCompatibility compatibility =
            source_result->write_compatibility();
        if (!compatibility.safe) {
            return status_failure(writer_error(
                lfs::ErrorCode::Unsupported, path,
                "This project is read-only in the current LichtFeld version.",
                std::format("compaction refused before writing project bytes: {}",
                            compatibility.reasons.empty()
                                ? std::string{"unknown writer incompatibility"}
                                : compatibility.reasons.front()),
                "commit.write_compatibility"));
        }
        if (source_result->superblock().role != ContainerRole::Master) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition, path,
                "Autosave sidecars are replaced, not compacted.",
                "compaction creates a master COMPACTION root",
                "superblock.container_role"));
        }
        auto bound_autosaves =
            detail::valid_bound_autosaves_locked(
                path, *source_result);
        if (!bound_autosaves) {
            return status_failure(
                std::move(bound_autosaves).error());
        }
        if (!bound_autosaves->empty()) {
            return status_failure(writer_error(
                lfs::ErrorCode::FailedPrecondition,
                path,
                "A current autosave must be merged or discarded before compaction.",
                std::format(
                    "{} valid autosave candidate(s) bind to master head {}",
                    bound_autosaves->size(),
                    source_result->commit()
                        .commit_uuid.to_string()),
                "compaction.autosave_binding"));
        }
        if (auto verify = source_result->verify_all(); !verify) {
            return verify;
        }

        auto file_uuid =
            assigned_uuid(options.new_file_uuid, path, "file_uuid");
        if (!file_uuid) {
            return status_failure(std::move(file_uuid).error());
        }
        if (*file_uuid == source_result->superblock().file_uuid) {
            return status_failure(writer_error(
                lfs::ErrorCode::InvalidArgument, path,
                "Compaction requires a new file identity.",
                "new_file_uuid must differ from the current file incarnation",
                "superblock.file_uuid"));
        }

        auto impl = std::make_unique<Impl>();
        impl->mode = Impl::Mode::Create;
        impl->destination_path = path;
        impl->active_path =
            detail::make_sibling_temp_path(path, "compact");
        impl->lock.emplace(std::move(*lock_result));
        impl->superblock = SuperblockInfo{
            .format = CURRENT_CONTAINER_VERSION,
            .role = ContainerRole::Master,
            .project_uuid =
                options.new_project_uuid.is_nil()
                    ? source_result->superblock().project_uuid
                    : options.new_project_uuid,
            .file_uuid = *file_uuid,
            .creation_time_unix_ns =
                options.creation_time_unix_ns != 0
                    ? options.creation_time_unix_ns
                    : detail::unix_time_ns(),
            .base_explicit_commit_uuid = {},
            .autosave_sequence = 0,
            .sidecar_snapshot_uuid = {},
            .crc32c = 0,
        };
        impl->generation = 1;
        impl->head_sequence = 1;
        impl->head_slot = 0;
        impl->cursor = APPEND_REGION_OFFSET;
        impl->disk_reserve_bytes = options.disk_reserve_bytes;
        impl->index_compression = IndexCompression::Zstd;
        impl->observer = options.boundary_observer;

        if (auto current = impl->notify(CommitBoundary::CurrentHeadValidated);
            !current) {
            return current;
        }
        auto file_result = detail::NativeFile::create_new(impl->active_path);
        if (!file_result) {
            return status_failure(std::move(file_result).error());
        }
        impl->file = std::move(*file_result);
        if (auto truncate = impl->file->truncate(APPEND_REGION_OFFSET);
            !truncate) {
            return truncate;
        }
        auto raw_superblock = encode_superblock(impl->superblock);
        impl->superblock.crc32c =
            crc32c(0, raw_superblock.data(), 252);
        if (auto write =
                impl->file->write_exact(0, byte_span(raw_superblock));
            !write) {
            return write;
        }
        impl->original_physical_size = APPEND_REGION_OFFSET;
        ProjectWriter writer(std::move(impl));

        CommitOptions commit_options{
            .kind = CommitKind::Compaction,
            .commit_uuid = options.commit_uuid,
            .snapshot_uuid = options.snapshot_uuid,
            .wallclock_unix_ns = options.wallclock_unix_ns,
            .min_reader_version =
                source_result->commit().min_reader_version,
            .min_safe_writer_version =
                source_result->commit().min_safe_writer_version,
            .extra_reader_capabilities =
                source_result->commit().required_reader_capabilities,
            .extra_writer_capabilities =
                source_result->commit().required_writer_capabilities,
        };
        if (auto plan = writer.plan_commit(commit_options); !plan) {
            return plan;
        }
        std::uint64_t planned_bytes = 0;
        for (const ChunkInfo& row : source_result->chunks()) {
            if (row.row_kind != RowKind::Live) {
                continue;
            }
            auto next = detail::checked_add(
                planned_bytes, row.stored_bytes, path, row.payload_offset,
                "compaction.planned_payload_bytes");
            if (!next) {
                return status_failure(std::move(next).error());
            }
            planned_bytes = *next;
        }
        if (auto preflight = writer.preflight(planned_bytes); !preflight) {
            return preflight;
        }

        for (const ChunkInfo& source_row : source_result->chunks()) {
            if (source_row.row_kind == RowKind::Live) {
                auto copied =
                    writer.impl_->copy_stored_chunk(*source_result, source_row);
                if (!copied) {
                    writer.impl_->poisoned = true;
                    return status_failure(std::move(copied).error());
                }
                if (source_result->preview().has_value() &&
                    source_row.key.fourcc == FOURCC_THMB &&
                    source_row.payload_offset ==
                        source_result->preview()->offset &&
                    source_row.stored_bytes ==
                        source_result->preview()->bytes) {
                    assert(copied->compression == Compression::Stored);
                    writer.impl_->preview_locator = PreviewLocator{
                        .offset = copied->payload_offset,
                        .bytes =
                            static_cast<std::uint32_t>(copied->stored_bytes),
                        .format = PreviewFormat::Png,
                    };
                }
                writer.impl_->rows[copied->key] = std::move(*copied);
                writer.impl_->touched.insert(source_row.key);
            }
            // Tombstones are discarded on compaction (spec MAY; no keep_tombstones producer).
        }

        return writer.commit();
    }

} // namespace lfs::io::project
