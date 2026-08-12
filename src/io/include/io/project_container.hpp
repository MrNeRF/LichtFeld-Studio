/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/uuid.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::io::project {

    inline constexpr std::uint64_t SUPERBLOCK_BYTES = 256;
    inline constexpr std::uint64_t HEAD_SLOT_BYTES = 4096;
    inline constexpr std::array<std::uint64_t, 2> HEAD_SLOT_OFFSETS = {4096, 8192};
    inline constexpr std::uint64_t APPEND_REGION_OFFSET = 65536;
    inline constexpr std::uint64_t COMMIT_RECORD_BYTES = 256;
    inline constexpr std::uint64_t CHUNK_HEADER_BYTES = 64;
    inline constexpr std::uint64_t INDEX_HEADER_BYTES = 64;
    inline constexpr std::uint64_t INDEX_ROW_BYTES = 96;
    inline constexpr std::uint64_t BLOCK_CRC_HEADER_BYTES = 64;
    inline constexpr std::uint64_t CHUNK_ALIGNMENT = 64;
    inline constexpr std::uint64_t TENSOR_PAYLOAD_ALIGNMENT = 4096;
    inline constexpr std::uint64_t BLOCK_CRC_BYTES = 4ull * 1024 * 1024;
    inline constexpr std::uint64_t BLOCK_CRC_REQUIRED_AT = 1ull * 1024 * 1024 * 1024;
    inline constexpr std::uint32_t MAX_PREVIEW_BYTES = 16u * 1024 * 1024;

    struct Version {
        std::uint16_t major = 1;
        std::uint16_t minor = 0;

        friend constexpr auto operator<=>(const Version&, const Version&) = default;
    };

    inline constexpr Version CURRENT_CONTAINER_VERSION{1, 0};

    struct LFS_IO_API Fourcc {
        std::array<std::uint8_t, 4> bytes{};

        [[nodiscard]] constexpr bool valid() const noexcept {
            for (const std::uint8_t byte : bytes) {
                if (!((byte >= static_cast<std::uint8_t>('A') &&
                       byte <= static_cast<std::uint8_t>('Z')) ||
                      (byte >= static_cast<std::uint8_t>('0') &&
                       byte <= static_cast<std::uint8_t>('9')))) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] std::string to_string() const;
        [[nodiscard]] static std::optional<Fourcc> from_string(std::string_view text);

        friend constexpr auto operator<=>(const Fourcc&, const Fourcc&) = default;
    };

    [[nodiscard]] constexpr Fourcc make_fourcc(const char a, const char b, const char c,
                                               const char d) noexcept {
        return Fourcc{{
            static_cast<std::uint8_t>(static_cast<unsigned char>(a)),
            static_cast<std::uint8_t>(static_cast<unsigned char>(b)),
            static_cast<std::uint8_t>(static_cast<unsigned char>(c)),
            static_cast<std::uint8_t>(static_cast<unsigned char>(d)),
        }};
    }

    inline constexpr Fourcc FOURCC_PROJ = make_fourcc('P', 'R', 'O', 'J');
    inline constexpr Fourcc FOURCC_PRMS = make_fourcc('P', 'R', 'M', 'S');
    inline constexpr Fourcc FOURCC_SCNG = make_fourcc('S', 'C', 'N', 'G');
    inline constexpr Fourcc FOURCC_SELM = make_fourcc('S', 'E', 'L', 'M');
    inline constexpr Fourcc FOURCC_REFS = make_fourcc('R', 'E', 'F', 'S');
    inline constexpr Fourcc FOURCC_SPLT = make_fourcc('S', 'P', 'L', 'T');
    inline constexpr Fourcc FOURCC_PCLD = make_fourcc('P', 'C', 'L', 'D');
    inline constexpr Fourcc FOURCC_MESH = make_fourcc('M', 'E', 'S', 'H');
    inline constexpr Fourcc FOURCC_CKPT = make_fourcc('C', 'K', 'P', 'T');
    inline constexpr Fourcc FOURCC_PPIS = make_fourcc('P', 'P', 'I', 'S');
    inline constexpr Fourcc FOURCC_GUIL = make_fourcc('G', 'U', 'I', 'L');
    inline constexpr Fourcc FOURCC_VIEW = make_fourcc('V', 'I', 'E', 'W');
    inline constexpr Fourcc FOURCC_EDTR = make_fourcc('E', 'D', 'T', 'R');
    inline constexpr Fourcc FOURCC_SEQR = make_fourcc('S', 'E', 'Q', 'R');
    inline constexpr Fourcc FOURCC_METR = make_fourcc('M', 'E', 'T', 'R');
    inline constexpr Fourcc FOURCC_THMB = make_fourcc('T', 'H', 'M', 'B');

    struct ChunkKey {
        Fourcc fourcc;
        lfs::core::Uuid instance_uuid;

        friend bool operator==(const ChunkKey&, const ChunkKey&) = default;
    };

    struct LFS_IO_API ChunkKeyLess {
        [[nodiscard]] bool operator()(const ChunkKey& lhs, const ChunkKey& rhs) const noexcept;
    };

    class LFS_IO_API CapabilitySet {
    public:
        CapabilitySet() = default;
        explicit CapabilitySet(std::array<std::uint8_t, 16> bytes) noexcept;

        [[nodiscard]] bool contains(std::uint8_t bit) const noexcept;
        void set(std::uint8_t bit, bool enabled = true) noexcept;
        [[nodiscard]] bool contains_all(const CapabilitySet& required) const noexcept;
        [[nodiscard]] CapabilitySet missing_from(const CapabilitySet& supported) const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] const std::array<std::uint8_t, 16>& bytes() const noexcept;
        [[nodiscard]] std::string to_hex() const;

        CapabilitySet& operator|=(const CapabilitySet& other) noexcept;
        friend CapabilitySet operator|(CapabilitySet lhs, const CapabilitySet& rhs) noexcept {
            lhs |= rhs;
            return lhs;
        }
        friend bool operator==(const CapabilitySet&, const CapabilitySet&) = default;

    private:
        std::array<std::uint8_t, 16> bytes_{};
    };

    enum CapabilityBit : std::uint8_t {
        INDEX_ZSTD_V1 = 0,
        CHUNK_ZSTD_V1 = 1,
        BLOCK_CRC32C_V1 = 2,
        INDEX_TOMBSTONES_V1 = 3,
        SIDECAR_OVERLAY_V1 = 4,
        OPAQUE_CHUNK_PRESERVATION = 5,
        RETAINED_JSON_FIELDS = 6,
        CLEAN_PROOF_REUSE = 7,
        // Byte-plane (f32-word) prefilter + zstd. Distinct wire encoding from
        // plain CHUNK_ZSTD_V1; readers without this bit refuse the generation.
        CHUNK_BYTESHUFFLE_ZSTD_V1 = 8,
    };

    [[nodiscard]] LFS_IO_API CapabilitySet supported_reader_capabilities();
    [[nodiscard]] LFS_IO_API CapabilitySet supported_writer_capabilities();

    enum class ContainerRole : std::uint32_t {
        Master = 0,
        AutosaveSidecar = 1,
    };

    enum class CommitKind : std::uint32_t {
        Explicit = 1,
        Autosave = 2,
        Recovered = 3,
        Compaction = 4,
    };

    // Chunk payload entropy encodings (wire u16 / index-row u8).
    // Framed payloads contain independently compressed zstd records.
    enum class Compression : std::uint16_t {
        Stored = 0,
        ZstdFramed = 1,
        ByteShuffleZstdFramed = 2,
    };

    enum class RowKind : std::uint8_t {
        Live = 0,
        Tombstone = 1,
        SidecarBaseReference = 2,
    };

    enum ChunkFlag : std::uint32_t {
        TENSOR_PAYLOAD = 1u << 0,
        HAS_BLOCK_CRCS = 1u << 1,
    };

    enum class OpenState {
        Open,
        UnsupportedNewer,
        RepairOnly,
        HardFail,
    };

    enum class PreviewFormat : std::uint32_t {
        Png = 1,
    };

    struct PreviewLocator {
        std::uint64_t offset = 0;
        std::uint32_t bytes = 0;
        PreviewFormat format = PreviewFormat::Png;

        friend constexpr auto operator<=>(const PreviewLocator&,
                                          const PreviewLocator&) = default;
    };

    struct LFS_IO_API OpenClassification {
        OpenState state = OpenState::HardFail;
        std::uint64_t generation = 0;
        std::string diagnostic;
    };

    struct ReaderOptions {
        Version reader_version = CURRENT_CONTAINER_VERSION;
        CapabilitySet reader_capabilities = supported_reader_capabilities();
        Version writer_version = CURRENT_CONTAINER_VERSION;
        CapabilitySet writer_capabilities = supported_writer_capabilities();
        bool allow_unsupported_inspection = false;
        // Optional read telemetry. Structural open-time reads are deliberately
        // excluded; only chunk payload bytes increment this counter.
        std::shared_ptr<std::atomic_uint64_t>
            payload_bytes_read = {};
    };

    struct SuperblockInfo {
        Version format;
        ContainerRole role = ContainerRole::Master;
        lfs::core::Uuid project_uuid;
        lfs::core::Uuid file_uuid;
        std::uint64_t creation_time_unix_ns = 0;
        lfs::core::Uuid base_explicit_commit_uuid;
        std::uint64_t autosave_sequence = 0;
        lfs::core::Uuid sidecar_snapshot_uuid;
        std::uint32_t crc32c = 0;
    };

    struct CommitInfo {
        std::uint64_t offset = 0;
        CommitKind kind = CommitKind::Explicit;
        lfs::core::Uuid commit_uuid;
        std::uint64_t generation = 0;
        lfs::core::Uuid parent_commit_uuid;
        std::uint64_t parent_commit_offset = 0;
        lfs::core::Uuid explicit_ancestor_commit_uuid;
        lfs::core::Uuid snapshot_uuid;
        std::uint64_t wallclock_unix_ns = 0;
        std::uint64_t index_offset = 0;
        std::uint64_t index_stored_bytes = 0;
        std::uint64_t index_uncompressed_bytes = 0;
        std::uint32_t index_stored_crc32c = 0;
        std::uint32_t index_uncompressed_crc32c = 0;
        Compression index_compression = Compression::Stored;
        std::uint64_t committed_file_end = 0;
        Version min_reader_version;
        Version min_safe_writer_version;
        CapabilitySet required_reader_capabilities;
        CapabilitySet required_writer_capabilities;
        std::uint32_t crc32c = 0;
    };

    struct HeadInfo {
        std::uint32_t slot_id = 0;
        std::uint64_t head_sequence = 0;
        std::uint64_t generation = 0;
        lfs::core::Uuid commit_uuid;
        std::uint64_t commit_offset = 0;
        std::uint64_t committed_file_end = 0;
        std::uint32_t commit_crc32c_echo = 0;
        std::optional<PreviewLocator> preview;
        std::uint32_t head_crc32c = 0;
    };

    struct BlockCrcTable {
        std::uint64_t offset = 0;
        std::uint64_t payload_offset = 0;
        std::uint64_t stored_bytes = 0;
        std::uint32_t block_size = 0;
        std::vector<std::uint32_t> entries;
        std::uint32_t entries_crc32c = 0;
        std::uint32_t header_crc32c = 0;
    };

    struct LFS_IO_API ChunkInfo {
        ChunkKey key;
        std::uint16_t chunk_version = 0;
        RowKind row_kind = RowKind::Live;
        Compression compression = Compression::Stored;
        std::uint32_t flags = 0;
        std::uint64_t header_offset = 0;
        std::uint64_t payload_offset = 0;
        std::uint64_t stored_bytes = 0;
        std::uint64_t uncompressed_bytes = 0;
        std::uint64_t source_generation = 0;
        std::uint32_t payload_crc32c = 0;
        std::uint32_t header_crc32c = 0;
        std::optional<BlockCrcTable> block_crc_table;

        [[nodiscard]] bool is_live() const noexcept { return row_kind == RowKind::Live; }
        [[nodiscard]] std::string key_string() const;
    };

    struct WriteCompatibility {
        bool safe = false;
        std::vector<std::string> reasons;
    };

    class LFS_IO_API CleanProof {
    public:
        CleanProof(const CleanProof&) = default;
        CleanProof(CleanProof&&) noexcept = default;
        CleanProof& operator=(const CleanProof&) = default;
        CleanProof& operator=(CleanProof&&) noexcept = default;
        ~CleanProof() = default;

        [[nodiscard]] const ChunkKey& key() const noexcept { return key_; }
        [[nodiscard]] std::uint64_t mutation_epoch() const noexcept { return mutation_epoch_; }

    private:
        friend class ProjectReader;
        friend class ProjectWriter;

        CleanProof() = default;

        ChunkKey key_;
        lfs::core::Uuid file_uuid_;
        lfs::core::Uuid commit_uuid_;
        std::uint64_t source_generation_ = 0;
        std::uint32_t payload_crc32c_ = 0;
        std::uint32_t header_crc32c_ = 0;
        std::uint64_t mutation_epoch_ = 0;
    };

    class LFS_IO_API BoundedInputStream {
    public:
        BoundedInputStream(BoundedInputStream&&) noexcept;
        BoundedInputStream& operator=(BoundedInputStream&&) noexcept;
        BoundedInputStream(const BoundedInputStream&) = delete;
        BoundedInputStream& operator=(const BoundedInputStream&) = delete;
        ~BoundedInputStream();

        [[nodiscard]] std::istream& stream();
        [[nodiscard]] std::uint64_t size() const noexcept;

    private:
        friend class ProjectReader;
        struct Impl;
        explicit BoundedInputStream(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    class LFS_IO_API MappedRegion {
    public:
        MappedRegion(MappedRegion&&) noexcept;
        MappedRegion& operator=(MappedRegion&&) noexcept;
        MappedRegion(const MappedRegion&) = delete;
        MappedRegion& operator=(const MappedRegion&) = delete;
        ~MappedRegion();

        [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
        [[nodiscard]] std::uint64_t file_offset() const noexcept;

    private:
        friend class ProjectReader;
        struct Impl;
        explicit MappedRegion(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    class ProjectWriter;

    class LFS_IO_API ProjectReader {
    public:
        [[nodiscard]] static lfs::Result<ProjectReader>
        open(const std::filesystem::path& path, const ReaderOptions& options = {});
        [[nodiscard]] static OpenClassification
        classify(const std::filesystem::path& path, const ReaderOptions& options = {});

        ProjectReader(ProjectReader&&) noexcept;
        ProjectReader& operator=(ProjectReader&&) noexcept;
        ProjectReader(const ProjectReader&) = delete;
        ProjectReader& operator=(const ProjectReader&) = delete;
        ~ProjectReader();

        [[nodiscard]] const std::filesystem::path& path() const noexcept;
        [[nodiscard]] std::uint64_t physical_file_size() const noexcept;
        [[nodiscard]] OpenState open_state() const noexcept;
        [[nodiscard]] const SuperblockInfo& superblock() const noexcept;
        [[nodiscard]] const HeadInfo& selected_head() const noexcept;
        [[nodiscard]] const CommitInfo& commit() const noexcept;
        [[nodiscard]] const std::vector<ChunkInfo>& chunks() const noexcept;
        [[nodiscard]] const std::vector<std::string>& warnings() const noexcept;
        [[nodiscard]] const std::optional<PreviewLocator>& preview() const noexcept;
        [[nodiscard]] const ReaderOptions& reader_options() const noexcept;
        [[nodiscard]] WriteCompatibility write_compatibility() const;

        [[nodiscard]] const ChunkInfo* find(const ChunkKey& key) const noexcept;
        [[nodiscard]] const ChunkInfo* find(Fourcc fourcc,
                                            const lfs::core::Uuid& instance_uuid) const noexcept;

        [[nodiscard]] lfs::Result<std::vector<std::byte>>
        read_chunk(const ChunkInfo& chunk,
                   // For framed payloads, progress may be invoked concurrently
                   // from decompression worker threads.
                   std::function<void(std::size_t, std::size_t)> progress = {}) const;
        [[nodiscard]] lfs::Result<void>
        read_stored_at(const ChunkInfo& chunk, std::uint64_t relative_offset,
                       std::span<std::byte> destination) const;
        [[nodiscard]] lfs::Result<void> verify_chunk(const ChunkInfo& chunk) const;
        [[nodiscard]] lfs::Result<void> verify_all() const;
        [[nodiscard]] lfs::Result<std::vector<std::byte>> read_preview() const;
        [[nodiscard]] lfs::Result<BoundedInputStream>
        open_bounded_stream(const ChunkInfo& chunk) const;
        [[nodiscard]] lfs::Result<MappedRegion>
        map_stored_range(const ChunkInfo& chunk, std::uint64_t relative_offset,
                         std::uint64_t length) const;

        [[nodiscard]] lfs::Result<CleanProof>
        make_clean_proof(const ChunkInfo& chunk, std::uint64_t mutation_epoch) const;

    private:
        friend class ProjectWriter;
        struct Impl;
        explicit ProjectReader(std::shared_ptr<Impl> impl);
        std::shared_ptr<Impl> impl_;
    };

    enum class IndexCompression {
        Zstd,
        StoredForDeterministicTests,
    };

    enum class CommitBoundary {
        CurrentHeadValidated = 1,
        IdentitiesAssigned = 2,
        PreflightComplete = 3,
        ChunksWritten = 4,
        IndexWritten = 5,
        CommitWritten = 6,
        AppendFlushed = 7,
        HeadWritten = 8,
        HeadFlushed = 9,
        ReplacementReady = 10,
        ReplacementPublished = 11,
        ReplacementValidated = 12,
        Committed = 13,
    };

    using CommitBoundaryObserver = std::function<void(CommitBoundary)>;

    // A copyable lease for one held project writer lock. Recovery sessions use
    // it to keep the original master exclusive while a staging document is
    // open, and pass the same OS lock through the eventual merge writer.
    class LFS_IO_API WriterLockLease {
    public:
        WriterLockLease() noexcept;
        WriterLockLease(const WriterLockLease&) noexcept;
        WriterLockLease& operator=(const WriterLockLease&) noexcept;
        WriterLockLease(WriterLockLease&&) noexcept;
        WriterLockLease& operator=(WriterLockLease&&) noexcept;
        ~WriterLockLease();

        [[nodiscard]] static lfs::Result<WriterLockLease>
        acquire(const std::filesystem::path& project_path);
        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] bool owns(
            const std::filesystem::path& project_path) const noexcept;
        // Requests release after any in-flight writer copies are destroyed.
        void release() noexcept;

    private:
        struct Impl;
        explicit WriterLockLease(std::shared_ptr<Impl> impl) noexcept;
        std::shared_ptr<Impl> impl_;
    };

    struct CreateOptions {
        lfs::core::Uuid project_uuid;
        lfs::core::Uuid file_uuid;
        ContainerRole role = ContainerRole::Master;
        lfs::core::Uuid base_explicit_commit_uuid;
        std::uint64_t autosave_sequence = 0;
        lfs::core::Uuid sidecar_snapshot_uuid;
        std::uint64_t creation_time_unix_ns = 0;
        IndexCompression index_compression = IndexCompression::Zstd;
        std::uint64_t disk_reserve_bytes = 64ull * 1024 * 1024;
        CommitBoundaryObserver boundary_observer;
        // Sidecars use the master path here so creation, replacement, and
        // recovery cleanup all share the master's exclusive writer lock.
        ReaderOptions writer_lock_anchor_compatibility = {};
        std::optional<std::filesystem::path> writer_lock_anchor;
        std::optional<WriterLockLease> writer_lock_lease =
            std::nullopt;
    };

    struct AppendOptions {
        ReaderOptions compatibility;
        IndexCompression index_compression = IndexCompression::Zstd;
        std::uint64_t disk_reserve_bytes = 64ull * 1024 * 1024;
        CommitBoundaryObserver boundary_observer;
        std::optional<WriterLockLease> writer_lock_lease =
            std::nullopt;
        // Writers that must not lose their generation to a transient
        // in-process lock holder wait instead of failing immediately.
        std::chrono::milliseconds writer_lock_wait{0};
    };

    struct ChunkWriteOptions {
        std::uint16_t chunk_version = 1;
        Compression compression = Compression::Stored;
        bool tensor_payload = false;
        bool block_crcs = false;
        std::optional<std::uint64_t> expected_stream_bytes;
    };

    struct CommitOptions {
        CommitKind kind = CommitKind::Explicit;
        lfs::core::Uuid commit_uuid;
        lfs::core::Uuid snapshot_uuid;
        std::uint64_t wallclock_unix_ns = 0;
        Version min_reader_version = CURRENT_CONTAINER_VERSION;
        Version min_safe_writer_version = CURRENT_CONTAINER_VERSION;
        CapabilitySet extra_reader_capabilities;
        CapabilitySet extra_writer_capabilities;
    };

    struct CompactionOptions {
        ReaderOptions compatibility;
        lfs::core::Uuid new_file_uuid;
        lfs::core::Uuid commit_uuid;
        lfs::core::Uuid snapshot_uuid;
        std::uint64_t creation_time_unix_ns = 0;
        std::uint64_t wallclock_unix_ns = 0;
        std::uint64_t disk_reserve_bytes = 64ull * 1024 * 1024;
        CommitBoundaryObserver boundary_observer;
    };

    class LFS_IO_API ProjectWriter {
    public:
        [[nodiscard]] static lfs::Result<ProjectWriter>
        create(const std::filesystem::path& path, const CreateOptions& options);
        [[nodiscard]] static lfs::Result<ProjectWriter>
        append(const std::filesystem::path& path, const AppendOptions& options = {});
        [[nodiscard]] static lfs::Result<void>
        compact(const std::filesystem::path& path, const CompactionOptions& options = {});

        ProjectWriter(ProjectWriter&&) noexcept;
        ProjectWriter& operator=(ProjectWriter&&) noexcept;
        ProjectWriter(const ProjectWriter&) = delete;
        ProjectWriter& operator=(const ProjectWriter&) = delete;
        ~ProjectWriter();

        [[nodiscard]] lfs::Result<void>
        plan_commit(const CommitOptions& options = {});
        // Must be called before the first chunk mutation. The estimate covers
        // stored payload bytes; the writer adds bounded metadata, alignment,
        // and the configured reserve before checking the containing volume.
        [[nodiscard]] lfs::Result<void>
        preflight(std::uint64_t planned_stored_payload_bytes);

        [[nodiscard]] lfs::Result<void>
        write_chunk(const ChunkKey& key, std::span<const std::byte> payload,
                    const ChunkWriteOptions& options = {});
        [[nodiscard]] lfs::Result<void>
        set_preview(std::span<const std::byte> png_bytes);
        [[nodiscard]] lfs::Result<std::ostream*>
        begin_chunk(const ChunkKey& key, const ChunkWriteOptions& options = {});
        [[nodiscard]] lfs::Result<void> end_chunk();

        [[nodiscard]] lfs::Result<void>
        reuse_if_clean(const CleanProof& proof, std::uint64_t current_mutation_epoch);
        [[nodiscard]] lfs::Result<void>
        carry_forward_opaque(const ChunkInfo& chunk, const CleanProof& proof,
                             std::uint64_t current_mutation_epoch);
        // Copies the exact stored representation through bounded windows.
        // Used by compaction/recovery so unknown fourccs and newer known
        // chapter versions remain byte-for-byte opaque.
        [[nodiscard]] lfs::Result<void>
        copy_chunk_verbatim(const ProjectReader& source,
                            const ChunkInfo& chunk);
        [[nodiscard]] lfs::Result<void> add_sidecar_base_reference(const ChunkInfo& base);
        [[nodiscard]] lfs::Result<void> erase(const ChunkKey& key);

        // DataLoss with field "commit.post_publish_verification" has special
        // append semantics: the second flush completed, so the generation is
        // already published and durable. The file is authoritative; callers
        // must not retry the same logical save from stale in-memory state.
        // Create-mode cleanup failures after validated atomic publication do
        // not fail commit(); they are exposed through post_publish_note().
        [[nodiscard]] lfs::Result<void> commit();
        [[nodiscard]] const std::optional<lfs::Error>&
        post_publish_note() const noexcept;

    private:
        struct Impl;
        explicit ProjectWriter(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io::project
