/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_container.hpp"
#include "crc32c.hpp"
#include "io/atomic_output.hpp"

#include <zstd.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <format>
#include <fstream>
#include <string>
#include <utility>

namespace lfs::io::project {

    namespace {

        constexpr int ZSTD_COMPRESSION_LEVEL = 3;
        constexpr std::uint64_t COPY_BLOCK_BYTES = 4ull * 1024 * 1024;

        std::string fourcc_to_string(const std::uint32_t fourcc) {
            std::string result(4, '.');
            for (int i = 0; i < 4; ++i) {
                const char c = static_cast<char>((fourcc >> (8 * i)) & 0xFFu);
                if (c >= 0x20 && c < 0x7F) {
                    result[static_cast<std::size_t>(i)] = c;
                }
            }
            return result;
        }

        Result<void> write_bytes_at(std::fstream& file, const std::uint64_t offset,
                                    const void* data, const std::size_t size) {
            file.seekp(static_cast<std::streamoff>(offset));
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
            if (!file) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Write of {} bytes at offset {} failed", size, offset));
            }
            return {};
        }

        Result<void> zero_fill(std::fstream& file, const std::uint64_t from, const std::uint64_t to) {
            assert(from <= to);
            if (from == to) {
                return {};
            }
            static constexpr std::array<char, 4096> ZEROS{};
            file.seekp(static_cast<std::streamoff>(from));
            for (std::uint64_t remaining = to - from; remaining > 0;) {
                const auto step = std::min<std::uint64_t>(remaining, ZEROS.size());
                file.write(ZEROS.data(), static_cast<std::streamsize>(step));
                remaining -= step;
            }
            if (!file) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Zero-fill padding [{}, {}) failed", from, to));
            }
            return {};
        }

        // Streams payload bytes through to the file while accumulating the
        // CRC32c and byte count end_chunk backpatches into the chunk header.
        // Nothing is buffered here beyond the CRC state.
        class Crc32cCountingStreambuf final : public std::streambuf {
        public:
            explicit Crc32cCountingStreambuf(std::streambuf* sink)
                : sink_(sink) {}

            [[nodiscard]] std::uint32_t crc() const { return crc_; }
            [[nodiscard]] std::uint64_t count() const { return count_; }

        protected:
            int_type overflow(const int_type ch) override {
                if (traits_type::eq_int_type(ch, traits_type::eof())) {
                    return traits_type::not_eof(ch);
                }
                const char c = traits_type::to_char_type(ch);
                if (traits_type::eq_int_type(sink_->sputc(c), traits_type::eof())) {
                    return traits_type::eof();
                }
                crc_ = crc32c(crc_, &c, 1);
                ++count_;
                return ch;
            }

            std::streamsize xsputn(const char* data, const std::streamsize size) override {
                const std::streamsize written = sink_->sputn(data, size);
                if (written > 0) {
                    crc_ = crc32c(crc_, data, static_cast<std::size_t>(written));
                    count_ += static_cast<std::uint64_t>(written);
                }
                return written;
            }

        private:
            std::streambuf* sink_;
            std::uint32_t crc_ = 0;
            std::uint64_t count_ = 0;
        };

        struct StreamingChunk {
            std::uint64_t header_offset;
            std::uint64_t payload_offset;
            ChunkHeader header;
            std::uint64_t generation;
            Crc32cCountingStreambuf buffer;
            std::ostream stream;

            StreamingChunk(const std::uint64_t header_offset_in, const std::uint64_t payload_offset_in,
                           const ChunkHeader& header_in, const std::uint64_t generation_in,
                           std::streambuf* sink)
                : header_offset(header_offset_in),
                  payload_offset(payload_offset_in),
                  header(header_in),
                  generation(generation_in),
                  buffer(sink),
                  stream(&buffer) {}
        };

    } // namespace

    // ------------------------------------------------------------------ reader

    struct ProjectReader::Impl {
        std::filesystem::path path;
        mutable std::ifstream file;
        std::uint64_t file_size = 0;
        FileHeader header{};
        std::array<std::uint8_t, 16> uuid{};
        std::uint64_t prev_index = 0;
        std::vector<ChunkInfo> chunk_infos;
        ChunkInfo index_info{};

        Result<void> read_at(const std::uint64_t offset, void* dst, const std::size_t size) const {
            file.clear();
            file.seekg(static_cast<std::streamoff>(offset));
            file.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
            if (file.gcount() != static_cast<std::streamsize>(size)) {
                return make_error(ErrorCode::READ_FAILURE,
                                  std::format("Read of {} bytes at offset {} failed", size, offset), path);
            }
            return {};
        }

        struct ParsedIndex {
            ChunkHeader index_header{};
            IndexPrologue prologue{};
            std::vector<ChunkInfo> rows;
        };

        Result<ParsedIndex> load_index_at(const std::uint64_t index_offset, const std::uint64_t index_size,
                                          const std::uint32_t index_crc) const {
            if (index_offset < FIRST_CHUNK_OFFSET || index_offset % CHUNK_ALIGNMENT != 0 ||
                index_size < sizeof(IndexPrologue) ||
                index_offset + sizeof(ChunkHeader) + index_size > file_size) {
                return make_error(ErrorCode::CORRUPTED_DATA, "index location out of bounds", path);
            }

            ParsedIndex parsed;
            if (auto read = read_at(index_offset, &parsed.index_header, sizeof(ChunkHeader)); !read) {
                return std::unexpected(read.error());
            }
            const ChunkHeader& index_header = parsed.index_header;
            if (index_header.fourcc != FOURCC_INDX ||
                index_header.compression != static_cast<std::uint8_t>(Compression::Raw) ||
                index_header.payload_bytes != index_size ||
                index_header.uncompressed_bytes != index_size ||
                index_header.payload_crc32c != index_crc) {
                return make_error(ErrorCode::CORRUPTED_DATA, "INDX chunk header mismatch", path);
            }

            std::vector<std::byte> payload(index_size);
            const std::uint64_t payload_offset = chunk_payload_offset(index_offset, index_header.flags);
            if (payload_offset + index_size > file_size) {
                return make_error(ErrorCode::CORRUPTED_DATA, "index payload out of bounds", path);
            }
            if (auto read = read_at(payload_offset, payload.data(), payload.size()); !read) {
                return std::unexpected(read.error());
            }
            if (crc32c(0, payload.data(), payload.size()) != index_crc) {
                return make_error(ErrorCode::CORRUPTED_DATA, "index CRC mismatch", path);
            }

            std::memcpy(&parsed.prologue, payload.data(), sizeof(IndexPrologue));
            if (sizeof(IndexPrologue) + static_cast<std::uint64_t>(parsed.prologue.row_count) * sizeof(IndexRow) != index_size) {
                return make_error(ErrorCode::CORRUPTED_DATA, "index row count does not match index size", path);
            }

            parsed.rows.reserve(parsed.prologue.row_count);
            for (std::uint32_t i = 0; i < parsed.prologue.row_count; ++i) {
                IndexRow row{};
                std::memcpy(&row, payload.data() + sizeof(IndexPrologue) + i * sizeof(IndexRow), sizeof(IndexRow));

                if (row.offset < FIRST_CHUNK_OFFSET || row.offset % CHUNK_ALIGNMENT != 0 ||
                    chunk_payload_offset(row.offset, row.flags) + row.stored_size > file_size) {
                    return make_error(ErrorCode::CORRUPTED_DATA,
                                      std::format("index row {} ('{}') out of bounds", i, fourcc_to_string(row.fourcc)),
                                      path);
                }

                // The chunk's own header is the authority for compression and
                // payload CRC (rows do not carry them). A mismatching or
                // garbage header is a chunk-level condition reported by
                // read_chunk, never an open failure.
                ChunkHeader chunk_header{};
                if (auto read = read_at(row.offset, &chunk_header, sizeof(ChunkHeader)); !read) {
                    return std::unexpected(read.error());
                }

                ChunkInfo info;
                info.fourcc = row.fourcc;
                info.instance_id = row.instance_id;
                info.chunk_version = row.chunk_version;
                info.flags = row.flags;
                info.offset = row.offset;
                info.stored_size = row.stored_size;
                info.uncompressed_size = row.uncompressed_size;
                info.generation = row.generation;
                info.compression = static_cast<Compression>(chunk_header.compression);
                info.payload_crc32c = chunk_header.payload_crc32c;
                parsed.rows.push_back(info);
            }
            return parsed;
        }
    };

    ProjectReader::ProjectReader(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    ProjectReader::ProjectReader(ProjectReader&&) noexcept = default;
    ProjectReader& ProjectReader::operator=(ProjectReader&&) noexcept = default;
    ProjectReader::~ProjectReader() = default;

    Result<ProjectReader> ProjectReader::open(const std::filesystem::path& path) {
        auto impl = std::make_unique<Impl>();
        impl->path = path;

        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              std::format("Cannot open project container: {}", ec.message()), path);
        }
        impl->file_size = size;

        impl->file.open(path, std::ios::binary);
        if (!impl->file.is_open()) {
            return make_error(ErrorCode::READ_FAILURE, "Cannot open project container", path);
        }

        if (size < sizeof(FileHeader) + sizeof(Footer)) {
            return make_error(ErrorCode::INVALID_HEADER,
                              "File too small to be a .licht project container", path);
        }

        if (auto read = impl->read_at(0, &impl->header, sizeof(FileHeader)); !read) {
            return std::unexpected(read.error());
        }
        if (std::memcmp(impl->header.magic, FILE_MAGIC.data(), FILE_MAGIC.size()) != 0) {
            return make_error(ErrorCode::INVALID_HEADER,
                              "Not a LichtFeld project container (bad magic)", path);
        }
        if (impl->header.min_reader_version > CONTAINER_VERSION) {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              std::format("Project requires a newer LichtFeld version: file needs container "
                                          "version >= {}, this build supports {}",
                                          impl->header.min_reader_version, CONTAINER_VERSION),
                              path);
        }
        if (impl->header.declared_file_size != size) {
            return make_error(
                ErrorCode::CORRUPTED_DATA,
                std::format("Declared project size {} does not match actual size {}",
                            impl->header.declared_file_size, size),
                path);
        }
        std::memcpy(impl->uuid.data(), impl->header.project_uuid, impl->uuid.size());

        // Newest index source is the footer; the file header's index fields
        // are the independent second source when the footer is torn. Scanning
        // is never attempted.
        std::string footer_failure;
        std::uint64_t index_offset = 0;
        std::uint64_t index_size = 0;
        std::uint32_t index_crc = 0;
        Impl::ParsedIndex parsed;
        bool index_loaded = false;

        Footer footer{};
        if (auto read = impl->read_at(size - sizeof(Footer), &footer, sizeof(Footer)); !read) {
            return std::unexpected(read.error());
        }
        if (crc32c(0, &footer, sizeof(Footer) - sizeof(std::uint32_t)) != footer.footer_crc32c) {
            footer_failure = "footer CRC mismatch";
        } else if (footer.magic != FOOTER_MAGIC) {
            footer_failure = "footer magic mismatch";
        } else if (auto loaded = impl->load_index_at(footer.index_offset, footer.index_size, footer.index_crc32c);
                   loaded) {
            parsed = std::move(*loaded);
            index_offset = footer.index_offset;
            index_size = footer.index_size;
            index_crc = footer.index_crc32c;
            index_loaded = true;
        } else {
            footer_failure = loaded.error().message;
        }

        if (!index_loaded) {
            auto loaded = impl->load_index_at(impl->header.index_offset, impl->header.index_size,
                                              impl->header.index_crc32c);
            if (!loaded) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Project container index unreadable ({}; header fallback: {})",
                                              footer_failure, loaded.error().message),
                                  path);
            }
            parsed = std::move(*loaded);
            index_offset = impl->header.index_offset;
            index_size = impl->header.index_size;
            index_crc = impl->header.index_crc32c;
        }

        impl->prev_index = parsed.prologue.prev_index_offset;
        impl->chunk_infos = std::move(parsed.rows);
        impl->index_info = ChunkInfo{
            .fourcc = FOURCC_INDX,
            .instance_id = 0,
            .chunk_version = parsed.index_header.chunk_version,
            .flags = parsed.index_header.flags,
            .offset = index_offset,
            .stored_size = index_size,
            .uncompressed_size = index_size,
            .generation = impl->header.save_generation,
            .compression = Compression::Raw,
            .payload_crc32c = index_crc,
        };

        return ProjectReader(std::move(impl));
    }

    const std::filesystem::path& ProjectReader::path() const { return impl_->path; }
    std::uint32_t ProjectReader::container_version() const { return impl_->header.container_version; }
    std::uint32_t ProjectReader::min_reader_version() const { return impl_->header.min_reader_version; }
    std::uint64_t ProjectReader::save_generation() const { return impl_->header.save_generation; }
    const std::array<std::uint8_t, 16>& ProjectReader::project_uuid() const { return impl_->uuid; }
    const std::vector<ChunkInfo>& ProjectReader::chunks() const { return impl_->chunk_infos; }
    const ChunkInfo& ProjectReader::index_chunk_info() const { return impl_->index_info; }
    std::uint64_t ProjectReader::prev_index_offset() const { return impl_->prev_index; }

    const ChunkInfo* ProjectReader::find(const std::uint32_t fourcc, const std::uint32_t instance_id) const {
        for (const auto& info : impl_->chunk_infos) {
            if (info.fourcc == fourcc && info.instance_id == instance_id) {
                return &info;
            }
        }
        return nullptr;
    }

    Result<std::vector<std::byte>> ProjectReader::read_chunk(const ChunkInfo& info) const {
        const auto& impl = *impl_;
        const std::string name = fourcc_to_string(info.fourcc);

        if (info.offset + sizeof(ChunkHeader) > impl.file_size ||
            chunk_payload_offset(info) + info.stored_size > impl.file_size) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("chunk '{}' (instance {}) out of bounds", name, info.instance_id),
                              impl.path);
        }

        ChunkHeader header{};
        if (auto read = impl.read_at(info.offset, &header, sizeof(ChunkHeader)); !read) {
            return std::unexpected(read.error());
        }
        if (header.fourcc != info.fourcc || header.instance_id != info.instance_id ||
            header.chunk_version != info.chunk_version ||
            header.payload_bytes != info.stored_size ||
            header.uncompressed_bytes != info.uncompressed_size ||
            static_cast<std::uint32_t>(header.flags) != info.flags) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("chunk '{}' (instance {}) header does not match index row",
                                          name, info.instance_id),
                              impl.path);
        }
        if (header.compression > static_cast<std::uint8_t>(Compression::Zstd)) {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              std::format("chunk '{}' has unknown compression {}", name, header.compression),
                              impl.path);
        }
        const auto compression = static_cast<Compression>(header.compression);
        if (compression == Compression::Raw && header.payload_bytes != header.uncompressed_bytes) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("raw chunk '{}' stored/uncompressed size mismatch", name),
                              impl.path);
        }

        std::vector<std::byte> stored(info.stored_size);
        if (auto read = impl.read_at(chunk_payload_offset(info.offset, header.flags), stored.data(), stored.size());
            !read) {
            return std::unexpected(read.error());
        }
        if (crc32c(0, stored.data(), stored.size()) != header.payload_crc32c) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("chunk '{}' (instance {}) payload CRC mismatch", name, info.instance_id),
                              impl.path);
        }

        if (compression == Compression::Raw) {
            return stored;
        }
        if (header.uncompressed_bytes == 0) {
            return std::vector<std::byte>{};
        }
        std::vector<std::byte> decompressed(header.uncompressed_bytes);
        const std::size_t result = ZSTD_decompress(decompressed.data(), decompressed.size(),
                                                   stored.data(), stored.size());
        if (ZSTD_isError(result) || result != header.uncompressed_bytes) {
            return make_error(ErrorCode::DECODING_FAILED,
                              std::format("chunk '{}' zstd decompression failed: {}", name,
                                          ZSTD_isError(result) ? ZSTD_getErrorName(result) : "size mismatch"),
                              impl.path);
        }
        return decompressed;
    }

    Result<std::vector<ChunkInfo>> ProjectReader::open_prev_index(const std::uint64_t prev_offset,
                                                                  std::uint64_t* prev_out) const {
        const auto& impl = *impl_;
        if (prev_offset == 0) {
            return make_error(ErrorCode::READ_FAILURE, "no previous index (offset 0)", impl.path);
        }
        if (prev_offset < FIRST_CHUNK_OFFSET || prev_offset % CHUNK_ALIGNMENT != 0 ||
            prev_offset + sizeof(ChunkHeader) > impl.file_size) {
            return make_error(ErrorCode::CORRUPTED_DATA, "previous index offset out of bounds", impl.path);
        }

        ChunkHeader header{};
        if (auto read = impl.read_at(prev_offset, &header, sizeof(ChunkHeader)); !read) {
            return std::unexpected(read.error());
        }
        if (header.fourcc != FOURCC_INDX) {
            return make_error(ErrorCode::CORRUPTED_DATA, "previous index offset is not an INDX chunk", impl.path);
        }

        auto parsed = impl.load_index_at(prev_offset, header.payload_bytes, header.payload_crc32c);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        if (prev_out != nullptr) {
            *prev_out = parsed->prologue.prev_index_offset;
        }
        return std::move(parsed->rows);
    }

    // ------------------------------------------------------------------ writer

    struct ProjectWriter::Impl {
        // Declared before the stream: destruction closes the file first, then
        // the uncommitted temp is removed.
        std::unique_ptr<ScopedAtomicOutputFile> atomic_output;
        std::fstream file;
        std::uint64_t cursor = FIRST_CHUNK_OFFSET;
        std::uint64_t save_generation = 0;
        std::array<std::uint8_t, 16> project_uuid{};
        std::uint64_t prev_index_offset = 0;
        std::vector<IndexRow> rows;
        std::unique_ptr<StreamingChunk> streaming;
        bool finalized = false;

        [[nodiscard]] Result<void> guard_writable(const char* operation) const {
            if (finalized || streaming) {
                assert(false && "ProjectWriter misuse");
                return make_error(ErrorCode::INTERNAL_ERROR,
                                  std::format("{} on a {} writer", operation,
                                              finalized ? "finalized" : "streaming"));
            }
            return {};
        }

        struct ChunkPlacement {
            std::uint64_t header_offset;
            std::uint64_t payload_offset;
        };

        // Writes padding + chunk header and leaves the put position at the
        // payload start. See chunk_payload_offset for the page-align rule.
        Result<ChunkPlacement> place_and_write_header(const ChunkHeader& header) {
            std::uint64_t header_offset = align_up(cursor, CHUNK_ALIGNMENT);
            std::uint64_t payload_offset = header_offset + sizeof(ChunkHeader);
            if (header.flags & CHUNK_FLAG_PAGE_ALIGNED) {
                payload_offset = align_up(header_offset + sizeof(ChunkHeader), PAGE_ALIGNMENT);
                header_offset = payload_offset - CHUNK_ALIGNMENT;
            }
            assert(header_offset % CHUNK_ALIGNMENT == 0);
            assert(header_offset >= cursor);
            assert(!(header.flags & CHUNK_FLAG_PAGE_ALIGNED) || payload_offset % PAGE_ALIGNMENT == 0);

            if (auto fill = zero_fill(file, cursor, header_offset); !fill) {
                return std::unexpected(fill.error());
            }
            if (auto write = write_bytes_at(file, header_offset, &header, sizeof(ChunkHeader)); !write) {
                return std::unexpected(write.error());
            }
            if (auto fill = zero_fill(file, header_offset + sizeof(ChunkHeader), payload_offset); !fill) {
                return std::unexpected(fill.error());
            }
            file.seekp(static_cast<std::streamoff>(payload_offset));
            return ChunkPlacement{header_offset, payload_offset};
        }

        void record_row(const ChunkHeader& header, const std::uint64_t header_offset,
                        const std::uint64_t generation) {
            rows.push_back(IndexRow{
                .fourcc = header.fourcc,
                .instance_id = header.instance_id,
                .chunk_version = header.chunk_version,
                .flags = header.flags,
                .offset = header_offset,
                .stored_size = header.payload_bytes,
                .uncompressed_size = header.uncompressed_bytes,
                .generation = generation,
            });
        }

        [[nodiscard]] FileHeader make_file_header(const std::uint64_t declared_file_size,
                                                  const std::uint64_t index_offset,
                                                  const std::uint64_t index_size,
                                                  const std::uint32_t index_crc) const {
            FileHeader header{};
            std::memcpy(header.magic, FILE_MAGIC.data(), FILE_MAGIC.size());
            header.container_version = CONTAINER_VERSION;
            header.min_reader_version = CONTAINER_MIN_READER_VERSION;
            header.declared_file_size = declared_file_size;
            header.index_offset = index_offset;
            header.index_size = index_size;
            header.index_crc32c = index_crc;
            header.save_generation = save_generation;
            std::memcpy(header.project_uuid, project_uuid.data(), project_uuid.size());
            return header;
        }
    };

    ProjectWriter::ProjectWriter(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    ProjectWriter::ProjectWriter(ProjectWriter&&) noexcept = default;
    ProjectWriter& ProjectWriter::operator=(ProjectWriter&&) noexcept = default;
    ProjectWriter::~ProjectWriter() = default;

    Result<ProjectWriter> ProjectWriter::create(const std::filesystem::path& path,
                                                const std::uint64_t save_generation,
                                                const std::array<std::uint8_t, 16>& project_uuid,
                                                const std::uint64_t prev_index_offset_hint) {
        if (auto parent = ensure_output_parent_directory(path); !parent) {
            return std::unexpected(parent.error());
        }

        auto impl = std::make_unique<Impl>();
        impl->atomic_output = std::make_unique<ScopedAtomicOutputFile>(
            path, AtomicOutputTempName::PreserveExtension, AtomicOutputDurability::Durable);
        impl->save_generation = save_generation;
        impl->project_uuid = project_uuid;
        impl->prev_index_offset = prev_index_offset_hint;

        impl->file.open(impl->atomic_output->temp_path(),
                        std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!impl->file.is_open()) {
            return make_error(ErrorCode::WRITE_FAILURE, "Cannot create project container temp file",
                              impl->atomic_output->temp_path());
        }

        const FileHeader header = impl->make_file_header(0, 0, 0, 0);
        if (auto write = write_bytes_at(impl->file, 0, &header, sizeof(FileHeader)); !write) {
            return std::unexpected(write.error());
        }
        if (auto fill = zero_fill(impl->file, sizeof(FileHeader), FIRST_CHUNK_OFFSET); !fill) {
            return std::unexpected(fill.error());
        }

        return ProjectWriter(std::move(impl));
    }

    Result<void> ProjectWriter::write_chunk(const std::uint32_t fourcc,
                                            const std::span<const std::byte> payload,
                                            const ChunkOptions& options) {
        auto& impl = *impl_;
        if (auto guard = impl.guard_writable("write_chunk"); !guard) {
            return guard;
        }
        if (fourcc == 0 || fourcc == FOURCC_INDX) {
            assert(false && "reserved fourcc");
            return make_error(ErrorCode::INTERNAL_ERROR, "write_chunk with a reserved fourcc");
        }
        assert((options.flags & CHUNK_FLAG_PAGE_ALIGNED) == 0 && "PAGE_ALIGNED is container-managed");

        std::vector<std::byte> compressed;
        std::span<const std::byte> stored = payload;
        if (options.compression == Compression::Zstd) {
            compressed.resize(ZSTD_compressBound(payload.size()));
            const void* src = payload.empty() ? static_cast<const void*>("") : payload.data();
            const std::size_t written = ZSTD_compress(compressed.data(), compressed.size(),
                                                      src, payload.size(), ZSTD_COMPRESSION_LEVEL);
            if (ZSTD_isError(written)) {
                return make_error(ErrorCode::ENCODING_FAILED,
                                  std::format("chunk '{}' zstd compression failed: {}",
                                              fourcc_to_string(fourcc), ZSTD_getErrorName(written)));
            }
            compressed.resize(written);
            stored = compressed;
        }

        const auto chunk_flags = static_cast<std::uint8_t>(
            options.flags | (options.page_align ? CHUNK_FLAG_PAGE_ALIGNED : 0u));
        const ChunkHeader header{
            .fourcc = fourcc,
            .chunk_version = options.chunk_version,
            .payload_bytes = stored.size(),
            .uncompressed_bytes = payload.size(),
            .instance_id = options.instance_id,
            .payload_crc32c = crc32c(0, stored.data(), stored.size()),
            .compression = static_cast<std::uint8_t>(options.compression),
            .flags = chunk_flags,
            .pad = 0,
            .reserved = 0,
        };

        auto placement = impl.place_and_write_header(header);
        if (!placement) {
            return std::unexpected(placement.error());
        }
        if (!stored.empty()) {
            if (auto write = write_bytes_at(impl.file, placement->payload_offset, stored.data(), stored.size());
                !write) {
                return write;
            }
        }

        impl.record_row(header, placement->header_offset, options.generation);
        impl.cursor = placement->payload_offset + stored.size();
        return {};
    }

    Result<std::ostream*> ProjectWriter::begin_chunk(const std::uint32_t fourcc, const ChunkOptions& options) {
        auto& impl = *impl_;
        if (auto guard = impl.guard_writable("begin_chunk"); !guard) {
            return std::unexpected(guard.error());
        }
        if (fourcc == 0 || fourcc == FOURCC_INDX) {
            assert(false && "reserved fourcc");
            return make_error(ErrorCode::INTERNAL_ERROR, "begin_chunk with a reserved fourcc");
        }
        assert(options.compression == Compression::Raw && "streaming chunks are raw-only");
        if (options.compression != Compression::Raw) {
            return make_error(ErrorCode::INTERNAL_ERROR, "streaming chunks are raw-only");
        }
        assert((options.flags & CHUNK_FLAG_PAGE_ALIGNED) == 0 && "PAGE_ALIGNED is container-managed");

        const auto chunk_flags = static_cast<std::uint8_t>(
            options.flags | (options.page_align ? CHUNK_FLAG_PAGE_ALIGNED : 0u));
        const ChunkHeader header{
            .fourcc = fourcc,
            .chunk_version = options.chunk_version,
            .payload_bytes = 0,
            .uncompressed_bytes = 0,
            .instance_id = options.instance_id,
            .payload_crc32c = 0,
            .compression = static_cast<std::uint8_t>(Compression::Raw),
            .flags = chunk_flags,
            .pad = 0,
            .reserved = 0,
        };

        auto placement = impl.place_and_write_header(header);
        if (!placement) {
            return std::unexpected(placement.error());
        }
        impl.streaming = std::make_unique<StreamingChunk>(placement->header_offset, placement->payload_offset,
                                                          header, options.generation, impl.file.rdbuf());
        return &impl.streaming->stream;
    }

    Result<void> ProjectWriter::end_chunk() {
        auto& impl = *impl_;
        if (!impl.streaming) {
            assert(false && "end_chunk without begin_chunk");
            return make_error(ErrorCode::INTERNAL_ERROR, "end_chunk without begin_chunk");
        }
        auto streaming = std::move(impl.streaming);

        streaming->stream.flush();
        if (!streaming->stream.good() || !impl.file.good()) {
            return make_error(ErrorCode::WRITE_FAILURE, "streamed chunk write failed",
                              impl.atomic_output->temp_path());
        }

        streaming->header.payload_bytes = streaming->buffer.count();
        streaming->header.uncompressed_bytes = streaming->buffer.count();
        streaming->header.payload_crc32c = streaming->buffer.crc();
        if (auto write = write_bytes_at(impl.file, streaming->header_offset,
                                        &streaming->header, sizeof(ChunkHeader));
            !write) {
            return write;
        }

        impl.record_row(streaming->header, streaming->header_offset, streaming->generation);
        impl.cursor = streaming->payload_offset + streaming->header.payload_bytes;
        impl.file.seekp(static_cast<std::streamoff>(impl.cursor));
        return {};
    }

    Result<ChunkInfo> ProjectWriter::copy_chunk(const ProjectReader& source, const ChunkInfo& info) {
        auto& impl = *impl_;
        if (auto guard = impl.guard_writable("copy_chunk"); !guard) {
            return std::unexpected(guard.error());
        }

        const auto& source_impl = *source.impl_;
        const std::uint64_t source_payload = chunk_payload_offset(info);
        if (source_payload + info.stored_size > source_impl.file_size) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("source chunk '{}' out of bounds", fourcc_to_string(info.fourcc)),
                              source_impl.path);
        }

        const ChunkHeader header{
            .fourcc = info.fourcc,
            .chunk_version = info.chunk_version,
            .payload_bytes = info.stored_size,
            .uncompressed_bytes = info.uncompressed_size,
            .instance_id = info.instance_id,
            .payload_crc32c = info.payload_crc32c,
            .compression = static_cast<std::uint8_t>(info.compression),
            .flags = static_cast<std::uint8_t>(info.flags),
            .pad = 0,
            .reserved = 0,
        };

        auto placement = impl.place_and_write_header(header);
        if (!placement) {
            return std::unexpected(placement.error());
        }

        source_impl.file.clear();
        source_impl.file.seekg(static_cast<std::streamoff>(source_payload));
        std::vector<char> buffer(static_cast<std::size_t>(std::min(info.stored_size, COPY_BLOCK_BYTES)));
        std::uint32_t crc = 0;
        for (std::uint64_t remaining = info.stored_size; remaining > 0;) {
            const auto step = std::min<std::uint64_t>(remaining, buffer.size());
            source_impl.file.read(buffer.data(), static_cast<std::streamsize>(step));
            if (source_impl.file.gcount() != static_cast<std::streamsize>(step)) {
                return make_error(ErrorCode::READ_FAILURE, "source chunk read failed during copy",
                                  source_impl.path);
            }
            crc = crc32c(crc, buffer.data(), static_cast<std::size_t>(step));
            impl.file.write(buffer.data(), static_cast<std::streamsize>(step));
            if (!impl.file) {
                return make_error(ErrorCode::WRITE_FAILURE, "chunk copy write failed",
                                  impl.atomic_output->temp_path());
            }
            remaining -= step;
        }
        if (crc != info.payload_crc32c) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("source chunk '{}' (instance {}) CRC mismatch during copy",
                                          fourcc_to_string(info.fourcc), info.instance_id),
                              source_impl.path);
        }

        impl.record_row(header, placement->header_offset, info.generation);
        impl.cursor = placement->payload_offset + info.stored_size;

        ChunkInfo copied = info;
        copied.offset = placement->header_offset;
        return copied;
    }

    void ProjectWriter::set_prev_index_offset(const std::uint64_t prev_index_offset) {
        impl_->prev_index_offset = prev_index_offset;
    }

    Result<void> ProjectWriter::finalize() {
        auto& impl = *impl_;
        if (auto guard = impl.guard_writable("finalize"); !guard) {
            return guard;
        }

        const std::uint64_t index_header_offset = align_up(impl.cursor, CHUNK_ALIGNMENT);

        std::vector<std::byte> index_payload(sizeof(IndexPrologue) + impl.rows.size() * sizeof(IndexRow));
        const IndexPrologue prologue{
            .prev_index_offset = impl.prev_index_offset,
            .row_count = static_cast<std::uint32_t>(impl.rows.size()),
            .reserved = 0,
        };
        std::memcpy(index_payload.data(), &prologue, sizeof(IndexPrologue));
        if (!impl.rows.empty()) {
            std::memcpy(index_payload.data() + sizeof(IndexPrologue), impl.rows.data(),
                        impl.rows.size() * sizeof(IndexRow));
        }
        const std::uint32_t index_crc = crc32c(0, index_payload.data(), index_payload.size());

        const std::uint64_t remaining_bytes = (index_header_offset - impl.cursor) + sizeof(ChunkHeader) +
                                              index_payload.size() + sizeof(Footer);
        if (auto space = check_disk_space(impl.atomic_output->output_path(), remaining_bytes); !space) {
            return std::unexpected(space.error());
        }

        if (auto fill = zero_fill(impl.file, impl.cursor, index_header_offset); !fill) {
            return fill;
        }
        const ChunkHeader index_header{
            .fourcc = FOURCC_INDX,
            .chunk_version = 1,
            .payload_bytes = index_payload.size(),
            .uncompressed_bytes = index_payload.size(),
            .instance_id = 0,
            .payload_crc32c = index_crc,
            .compression = static_cast<std::uint8_t>(Compression::Raw),
            .flags = CHUNK_FLAG_CRITICAL,
            .pad = 0,
            .reserved = 0,
        };
        if (auto write = write_bytes_at(impl.file, index_header_offset, &index_header, sizeof(ChunkHeader));
            !write) {
            return write;
        }
        const std::uint64_t index_payload_offset = index_header_offset + sizeof(ChunkHeader);
        if (auto write = write_bytes_at(impl.file, index_payload_offset, index_payload.data(),
                                        index_payload.size());
            !write) {
            return write;
        }

        const std::uint64_t footer_offset = index_payload_offset + index_payload.size();
        Footer footer{
            .magic = FOOTER_MAGIC,
            .reserved = 0,
            .index_offset = index_header_offset,
            .index_size = index_payload.size(),
            .index_crc32c = index_crc,
            .generation = impl.save_generation,
            .footer_crc32c = 0,
        };
        footer.footer_crc32c = crc32c(0, &footer, sizeof(Footer) - sizeof(std::uint32_t));
        if (auto write = write_bytes_at(impl.file, footer_offset, &footer, sizeof(Footer)); !write) {
            return write;
        }

        const FileHeader file_header = impl.make_file_header(footer_offset + sizeof(Footer), index_header_offset,
                                                             index_payload.size(), index_crc);
        if (auto write = write_bytes_at(impl.file, 0, &file_header, sizeof(FileHeader)); !write) {
            return write;
        }

        impl.file.flush();
        if (!impl.file) {
            return make_error(ErrorCode::WRITE_FAILURE, "Flushing project container failed",
                              impl.atomic_output->temp_path());
        }
        impl.file.close();
        if (impl.file.fail()) {
            return make_error(ErrorCode::WRITE_FAILURE, "Closing project container failed",
                              impl.atomic_output->temp_path());
        }

        if (auto commit = impl.atomic_output->commit(); !commit) {
            return commit;
        }
        impl.finalized = true;
        return {};
    }

} // namespace lfs::io::project
