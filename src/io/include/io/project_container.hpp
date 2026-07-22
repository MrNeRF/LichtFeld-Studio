/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/error.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <ostream>
#include <span>
#include <vector>

// .licht TLV container: [file header][chunk]...[INDX chunk][footer].
// Little-endian by fiat (matching every existing serializer); packed on-disk
// structs, no endian swapping. All chunk headers start 64-byte aligned;
// page-aligned chunks additionally place their payload on a 4096-byte
// boundary (rule documented at chunk_payload_offset).
namespace lfs::io::project {

    constexpr std::uint32_t make_fourcc(const char a, const char b, const char c, const char d) {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24);
    }

    constexpr std::uint32_t FOURCC_INDX = make_fourcc('I', 'N', 'D', 'X');

    constexpr std::uint32_t CONTAINER_VERSION = 1;
    constexpr std::uint32_t CONTAINER_MIN_READER_VERSION = 1;

    constexpr std::array<std::uint8_t, 8> FILE_MAGIC = {0x89, 'L', 'F', 'S', '\r', '\n', 0x1a, '\n'};
    constexpr std::uint32_t FOOTER_MAGIC = make_fourcc('L', 'F', 'S', 'E');

    constexpr std::uint64_t CHUNK_ALIGNMENT = 64;
    constexpr std::uint64_t PAGE_ALIGNMENT = 4096;

    enum ChunkFlags : std::uint8_t {
        CHUNK_FLAG_CRITICAL = 1u << 0,
        CHUNK_FLAG_SUPERSEDED = 1u << 1,
        // Written by the container itself for page_align requests so a reader
        // can recompute the payload offset from the header offset alone.
        CHUNK_FLAG_PAGE_ALIGNED = 1u << 2,
    };

    enum class Compression : std::uint8_t {
        Raw = 0,
        Zstd = 1,
    };

#pragma pack(push, 1)

    // The container spec's prose says "64 bytes" but its frozen field list
    // sums to 72; every field is kept, sizeof is 72, and the first chunk
    // starts at the next 64-byte boundary (offset 128).
    struct FileHeader {
        std::uint8_t magic[8];
        std::uint32_t container_version;
        std::uint32_t min_reader_version;
        std::uint64_t declared_file_size;
        std::uint64_t index_offset;
        std::uint64_t index_size;
        std::uint32_t index_crc32c;
        std::uint16_t header_flags;
        std::uint16_t reserved;
        std::uint64_t save_generation;
        std::uint8_t project_uuid[16];
    };
    static_assert(sizeof(FileHeader) == 72);

    constexpr std::uint64_t FIRST_CHUNK_OFFSET = 128;

    // The frozen field list sums to 36; the trailing reserved word pads to
    // the mandated 40 bytes. payload_crc32c covers the STORED payload bytes.
    struct ChunkHeader {
        std::uint32_t fourcc;
        std::uint32_t chunk_version;
        std::uint64_t payload_bytes;
        std::uint64_t uncompressed_bytes;
        std::uint32_t instance_id;
        std::uint32_t payload_crc32c;
        std::uint8_t compression;
        std::uint8_t flags;
        std::uint16_t pad;
        std::uint32_t reserved;
    };
    static_assert(sizeof(ChunkHeader) == 40);

    struct IndexPrologue {
        std::uint64_t prev_index_offset;
        std::uint32_t row_count;
        std::uint32_t reserved;
    };
    static_assert(sizeof(IndexPrologue) == 16);

    struct IndexRow {
        std::uint32_t fourcc;
        std::uint32_t instance_id;
        std::uint32_t chunk_version;
        std::uint32_t flags;
        std::uint64_t offset;
        std::uint64_t stored_size;
        std::uint64_t uncompressed_size;
        std::uint64_t generation;
    };
    static_assert(sizeof(IndexRow) == 48);

    struct Footer {
        std::uint32_t magic;
        std::uint32_t reserved;
        std::uint64_t index_offset;
        std::uint64_t index_size;
        std::uint32_t index_crc32c;
        std::uint64_t generation;
        std::uint32_t footer_crc32c;
    };
    static_assert(sizeof(Footer) == 40);

#pragma pack(pop)

    constexpr std::uint64_t align_up(const std::uint64_t value, const std::uint64_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // Page-align rule: the writer places the payload at
    //   payload_offset = align_up(align_up(cursor, 64) + sizeof(ChunkHeader), 4096)
    // and the chunk header at payload_offset - 64 (64-byte aligned by
    // construction, 24 zero bytes between header end and payload). A reader
    // recomputes the identical payload offset from the header offset and the
    // PAGE_ALIGNED flag; non-page-aligned payloads follow the header
    // immediately.
    constexpr std::uint64_t chunk_payload_offset(const std::uint64_t header_offset, const std::uint32_t flags) {
        const std::uint64_t header_end = header_offset + sizeof(ChunkHeader);
        return (flags & CHUNK_FLAG_PAGE_ALIGNED) ? align_up(header_end, PAGE_ALIGNMENT) : header_end;
    }

    struct ChunkOptions {
        std::uint32_t chunk_version = 1;
        std::uint32_t instance_id = 0;
        Compression compression = Compression::Raw;
        bool page_align = false;
        std::uint8_t flags = 0;
        std::uint64_t generation = 0;
    };

    struct ChunkInfo {
        std::uint32_t fourcc = 0;
        std::uint32_t instance_id = 0;
        std::uint32_t chunk_version = 0;
        std::uint32_t flags = 0;
        std::uint64_t offset = 0;
        std::uint64_t stored_size = 0;
        std::uint64_t uncompressed_size = 0;
        std::uint64_t generation = 0;
        Compression compression = Compression::Raw;
        std::uint32_t payload_crc32c = 0;
    };

    constexpr std::uint64_t chunk_payload_offset(const ChunkInfo& info) {
        return chunk_payload_offset(info.offset, info.flags);
    }

    class ProjectWriter;

    class LFS_IO_API ProjectReader {
    public:
        [[nodiscard]] static Result<ProjectReader> open(const std::filesystem::path& path);

        ProjectReader(ProjectReader&&) noexcept;
        ProjectReader& operator=(ProjectReader&&) noexcept;
        ProjectReader(const ProjectReader&) = delete;
        ProjectReader& operator=(const ProjectReader&) = delete;
        ~ProjectReader();

        [[nodiscard]] const std::filesystem::path& path() const;
        [[nodiscard]] std::uint32_t container_version() const;
        [[nodiscard]] std::uint32_t min_reader_version() const;
        [[nodiscard]] std::uint64_t save_generation() const;
        [[nodiscard]] const std::array<std::uint8_t, 16>& project_uuid() const;

        // Rows of the newest valid INDX, current INDX excluded. Historical
        // INDX chunks preserved for the prev_index_offset chain do appear as
        // rows (fourcc INDX).
        [[nodiscard]] const std::vector<ChunkInfo>& chunks() const;
        [[nodiscard]] const ChunkInfo* find(std::uint32_t fourcc, std::uint32_t instance_id = 0) const;
        // Synthesized info for the current INDX chunk itself, so it can be
        // copied through to preserve this generation's index in the next file.
        [[nodiscard]] const ChunkInfo& index_chunk_info() const;
        [[nodiscard]] std::uint64_t prev_index_offset() const;

        // Decompresses and CRC-verifies. A failure here is chunk-level: other
        // chunks of the same reader stay readable.
        [[nodiscard]] Result<std::vector<std::byte>> read_chunk(const ChunkInfo& info) const;

        // Reads a previous generation's INDX chunk at prev_offset (obtained
        // from prev_index_offset() or a previous call's prev_out).
        [[nodiscard]] Result<std::vector<ChunkInfo>> open_prev_index(std::uint64_t prev_offset,
                                                                     std::uint64_t* prev_out = nullptr) const;

    private:
        friend class ProjectWriter;
        struct Impl;
        explicit ProjectReader(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    class LFS_IO_API ProjectWriter {
    public:
        [[nodiscard]] static Result<ProjectWriter> create(const std::filesystem::path& path,
                                                          std::uint64_t save_generation,
                                                          const std::array<std::uint8_t, 16>& project_uuid,
                                                          std::uint64_t prev_index_offset_hint = 0);

        ProjectWriter(ProjectWriter&&) noexcept;
        ProjectWriter& operator=(ProjectWriter&&) noexcept;
        ProjectWriter(const ProjectWriter&) = delete;
        ProjectWriter& operator=(const ProjectWriter&) = delete;
        // Destroying without finalize() discards the temp file; the
        // destination is never observed torn.
        ~ProjectWriter();

        [[nodiscard]] Result<void> write_chunk(std::uint32_t fourcc,
                                               std::span<const std::byte> payload,
                                               const ChunkOptions& options = {});

        // Streaming protocol for payloads that must not be buffered in RAM:
        // a placeholder header is written up front, bytes stream through a
        // CRC32c-counting filter straight to disk, and end_chunk backpatches
        // payload_bytes/uncompressed_bytes/payload_crc32c. Raw-only.
        [[nodiscard]] Result<std::ostream*> begin_chunk(std::uint32_t fourcc, const ChunkOptions& options = {});
        [[nodiscard]] Result<void> end_chunk();

        // Byte-exact copy of the stored payload (no decode/re-encode, CRC
        // preserved); offset and alignment are recomputed for this file.
        // Copying source.index_chunk_info() preserves the previous
        // generation's index for the prev_index_offset chain. Returns the
        // chunk's placement in this file.
        [[nodiscard]] Result<ChunkInfo> copy_chunk(const ProjectReader& source, const ChunkInfo& info);

        void set_prev_index_offset(std::uint64_t prev_index_offset);

        // Writes INDX + footer, backpatches the file header, and atomically
        // commits to the destination path.
        [[nodiscard]] Result<void> finalize();

    private:
        struct Impl;
        explicit ProjectWriter(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io::project
