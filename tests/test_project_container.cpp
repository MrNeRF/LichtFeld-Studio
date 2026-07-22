/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Exhaustive grammar tests for the .licht project container (P1): TLV
// round-trip, alignment, streaming backpatch, torn-write fallbacks,
// copy-through preservation, and the prev-index generation chain.

#include "io/project/crc32c.hpp"
#include "io/project_container.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;

    constexpr std::array<std::uint8_t, 16> TEST_UUID = {
        0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04,
        0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c};

    constexpr std::uint32_t FOURCC_AAAA = make_fourcc('A', 'A', 'A', 'A');
    constexpr std::uint32_t FOURCC_BBBB = make_fourcc('B', 'B', 'B', 'B');
    constexpr std::uint32_t FOURCC_ZSTD = make_fourcc('Z', 'S', 'T', 'D');
    constexpr std::uint32_t FOURCC_PAGE = make_fourcc('P', 'A', 'G', 'E');
    constexpr std::uint32_t FOURCC_STRM = make_fourcc('S', 'T', 'R', 'M');

    std::vector<std::byte> pattern_bytes(const std::size_t size, const std::uint32_t seed) {
        std::vector<std::byte> bytes(size);
        std::uint32_t state = seed * 2654435761u + 1u;
        for (auto& byte : bytes) {
            state = state * 1664525u + 1013904223u;
            byte = static_cast<std::byte>(state >> 24);
        }
        return bytes;
    }

    std::vector<std::byte> compressible_bytes(const std::size_t size) {
        const std::string phrase = "LichtFeld .licht container zstd chapter payload. ";
        std::vector<std::byte> bytes(size);
        for (std::size_t i = 0; i < size; ++i) {
            bytes[i] = static_cast<std::byte>(phrase[i % phrase.size()]);
        }
        return bytes;
    }

    void flip_byte_at(const fs::path& path, const std::uint64_t offset) {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file.is_open());
        file.seekg(static_cast<std::streamoff>(offset));
        char value = 0;
        file.read(&value, 1);
        ASSERT_TRUE(file.good());
        value = static_cast<char>(value ^ 0x5A);
        file.seekp(static_cast<std::streamoff>(offset));
        file.write(&value, 1);
        ASSERT_TRUE(file.good());
    }

    void write_u32_at(const fs::path& path, const std::uint64_t offset, const std::uint32_t value) {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file.is_open());
        file.seekp(static_cast<std::streamoff>(offset));
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        ASSERT_TRUE(file.good());
    }

    std::vector<std::byte> read_file_range(const fs::path& path, const std::uint64_t offset,
                                           const std::uint64_t size) {
        std::ifstream file(path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        std::vector<std::byte> bytes(size);
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        EXPECT_EQ(file.gcount(), static_cast<std::streamsize>(size));
        return bytes;
    }

    class ProjectContainerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
            dir_ = fs::temp_directory_path() /
                   std::format("licht_container_{}_{}", info->name(), ticks);
            std::error_code ec;
            fs::remove_all(dir_, ec);
            ASSERT_TRUE(fs::create_directories(dir_));
        }

        void TearDown() override {
            std::error_code ec;
            fs::remove_all(dir_, ec);
        }

        [[nodiscard]] fs::path file_path(const std::string& name = "project.licht") const {
            return dir_ / name;
        }

        fs::path dir_;
    };

    TEST(ProjectContainerFormat, FourccConstantsAndLayout) {
        EXPECT_EQ(FOURCC_INDX, make_fourcc('I', 'N', 'D', 'X'));
        EXPECT_EQ(FOOTER_MAGIC, make_fourcc('L', 'F', 'S', 'E'));

        std::array<char, 4> encoded{};
        std::memcpy(encoded.data(), &FOURCC_INDX, sizeof(FOURCC_INDX));
        EXPECT_EQ(encoded[0], 'I');
        EXPECT_EQ(encoded[1], 'N');
        EXPECT_EQ(encoded[2], 'D');
        EXPECT_EQ(encoded[3], 'X');

        EXPECT_EQ(sizeof(FileHeader), 72u);
        EXPECT_EQ(sizeof(ChunkHeader), 40u);
        EXPECT_EQ(sizeof(IndexPrologue), 16u);
        EXPECT_EQ(sizeof(IndexRow), 48u);
        EXPECT_EQ(sizeof(Footer), 40u);
        EXPECT_EQ(FIRST_CHUNK_OFFSET % CHUNK_ALIGNMENT, 0u);
    }

    TEST(ProjectContainerFormat, Crc32cKnownVectors) {
        const char* check = "123456789";
        EXPECT_EQ(crc32c(0, check, 9), 0xE3069283u);
        EXPECT_EQ(crc32c(0, check, 0), 0u);

        const std::uint32_t split_short = crc32c(crc32c(0, check, 3), check + 3, 6);
        EXPECT_EQ(split_short, 0xE3069283u);
        const std::uint32_t split_at_slice = crc32c(crc32c(0, check, 8), check + 8, 1);
        EXPECT_EQ(split_at_slice, 0xE3069283u);
    }

    TEST_F(ProjectContainerTest, MultiChunkRoundTrip) {
        const auto payload_a = pattern_bytes(1000, 1);
        const auto payload_splat_1 = pattern_bytes(4096 + 17, 2);
        const auto payload_splat_2 = pattern_bytes(333, 3);
        const auto payload_zstd = compressible_bytes(50000);

        auto writer = ProjectWriter::create(file_path(), 42, TEST_UUID);
        ASSERT_TRUE(writer) << writer.error().format();
        EXPECT_FALSE(fs::exists(file_path()));

        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, payload_a,
                                        {.chunk_version = 3, .flags = CHUNK_FLAG_SUPERSEDED, .generation = 7}));
        ASSERT_TRUE(writer->write_chunk(FOURCC_BBBB, payload_splat_1, {.instance_id = 1, .generation = 42}));
        ASSERT_TRUE(writer->write_chunk(FOURCC_BBBB, payload_splat_2, {.instance_id = 2, .generation = 42}));
        ASSERT_TRUE(writer->write_chunk(FOURCC_ZSTD, payload_zstd, {.compression = Compression::Zstd}));
        ASSERT_TRUE(writer->finalize());
        EXPECT_TRUE(fs::exists(file_path()));

        auto reader = ProjectReader::open(file_path());
        ASSERT_TRUE(reader) << reader.error().format();
        EXPECT_EQ(reader->container_version(), CONTAINER_VERSION);
        EXPECT_EQ(reader->min_reader_version(), CONTAINER_MIN_READER_VERSION);
        EXPECT_EQ(reader->save_generation(), 42u);
        EXPECT_EQ(reader->project_uuid(), TEST_UUID);
        EXPECT_EQ(reader->prev_index_offset(), 0u);
        ASSERT_EQ(reader->chunks().size(), 4u);

        const ChunkInfo* info_a = reader->find(FOURCC_AAAA);
        ASSERT_NE(info_a, nullptr);
        EXPECT_EQ(info_a->chunk_version, 3u);
        EXPECT_EQ(info_a->flags, CHUNK_FLAG_SUPERSEDED);
        EXPECT_EQ(info_a->generation, 7u);
        EXPECT_EQ(info_a->compression, Compression::Raw);
        EXPECT_EQ(info_a->stored_size, payload_a.size());
        auto read_a = reader->read_chunk(*info_a);
        ASSERT_TRUE(read_a) << read_a.error().format();
        EXPECT_EQ(*read_a, payload_a);

        const ChunkInfo* info_s1 = reader->find(FOURCC_BBBB, 1);
        const ChunkInfo* info_s2 = reader->find(FOURCC_BBBB, 2);
        ASSERT_NE(info_s1, nullptr);
        ASSERT_NE(info_s2, nullptr);
        EXPECT_NE(info_s1->offset, info_s2->offset);
        auto read_s1 = reader->read_chunk(*info_s1);
        auto read_s2 = reader->read_chunk(*info_s2);
        ASSERT_TRUE(read_s1);
        ASSERT_TRUE(read_s2);
        EXPECT_EQ(*read_s1, payload_splat_1);
        EXPECT_EQ(*read_s2, payload_splat_2);

        const ChunkInfo* info_z = reader->find(FOURCC_ZSTD);
        ASSERT_NE(info_z, nullptr);
        EXPECT_EQ(info_z->compression, Compression::Zstd);
        EXPECT_EQ(info_z->uncompressed_size, payload_zstd.size());
        EXPECT_LT(info_z->stored_size, info_z->uncompressed_size);
        auto read_z = reader->read_chunk(*info_z);
        ASSERT_TRUE(read_z) << read_z.error().format();
        EXPECT_EQ(*read_z, payload_zstd);
    }

    TEST_F(ProjectContainerTest, AlignmentInvariants) {
        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        std::uint32_t seed = 1;
        for (const std::size_t size : {std::size_t{1}, std::size_t{63}, std::size_t{64},
                                       std::size_t{65}, std::size_t{4095}}) {
            ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, pattern_bytes(size, seed), {.instance_id = seed}));
            ++seed;
        }
        const auto page_payload = pattern_bytes(10000, 99);
        ASSERT_TRUE(writer->write_chunk(FOURCC_PAGE, page_payload, {.page_align = true}));
        ASSERT_TRUE(writer->write_chunk(FOURCC_BBBB, pattern_bytes(100, 100), {}));
        ASSERT_TRUE(writer->finalize());

        auto reader = ProjectReader::open(file_path());
        ASSERT_TRUE(reader) << reader.error().format();
        ASSERT_EQ(reader->chunks().size(), 7u);

        for (const auto& info : reader->chunks()) {
            EXPECT_EQ(info.offset % CHUNK_ALIGNMENT, 0u);
            EXPECT_GE(info.offset, FIRST_CHUNK_OFFSET);
            auto payload = reader->read_chunk(info);
            ASSERT_TRUE(payload) << payload.error().format();
            if (info.fourcc == FOURCC_PAGE) {
                EXPECT_NE(info.flags & CHUNK_FLAG_PAGE_ALIGNED, 0u);
                EXPECT_EQ(chunk_payload_offset(info) % PAGE_ALIGNMENT, 0u);
                EXPECT_EQ(*payload, page_payload);
            } else {
                EXPECT_EQ(chunk_payload_offset(info), info.offset + sizeof(ChunkHeader));
            }
        }
        EXPECT_EQ(reader->index_chunk_info().offset % CHUNK_ALIGNMENT, 0u);
    }

    TEST_F(ProjectContainerTest, EmptyPayloadChunk) {
        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, {}, {}));
        ASSERT_TRUE(writer->finalize());

        auto reader = ProjectReader::open(file_path());
        ASSERT_TRUE(reader) << reader.error().format();
        const ChunkInfo* info = reader->find(FOURCC_AAAA);
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->stored_size, 0u);
        EXPECT_EQ(info->uncompressed_size, 0u);
        EXPECT_EQ(info->payload_crc32c, 0u);
        auto payload = reader->read_chunk(*info);
        ASSERT_TRUE(payload) << payload.error().format();
        EXPECT_TRUE(payload->empty());
    }

    TEST_F(ProjectContainerTest, StreamedChunkBackpatchesSizesAndCrc) {
        constexpr std::size_t BLOCK_BYTES = 1024 * 1024;
        constexpr std::size_t BLOCK_COUNT = 64;

        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);

        auto stream = writer->begin_chunk(FOURCC_STRM, {.chunk_version = 2, .generation = 5});
        ASSERT_TRUE(stream) << stream.error().format();
        std::uint32_t expected_crc = 0;
        for (std::size_t block = 0; block < BLOCK_COUNT; ++block) {
            const auto bytes = pattern_bytes(BLOCK_BYTES, static_cast<std::uint32_t>(block));
            expected_crc = crc32c(expected_crc, bytes.data(), bytes.size());
            (*stream)->write(reinterpret_cast<const char*>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()));
            ASSERT_TRUE((*stream)->good());
        }
        ASSERT_TRUE(writer->end_chunk());
        const auto trailing = pattern_bytes(777, 4242);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, trailing, {}));
        ASSERT_TRUE(writer->finalize());

        auto reader = ProjectReader::open(file_path());
        ASSERT_TRUE(reader) << reader.error().format();
        const ChunkInfo* info = reader->find(FOURCC_STRM);
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->stored_size, BLOCK_BYTES * BLOCK_COUNT);
        EXPECT_EQ(info->uncompressed_size, BLOCK_BYTES * BLOCK_COUNT);
        EXPECT_EQ(info->payload_crc32c, expected_crc);
        EXPECT_EQ(info->chunk_version, 2u);
        EXPECT_EQ(info->generation, 5u);

        auto payload = reader->read_chunk(*info);
        ASSERT_TRUE(payload) << payload.error().format();
        ASSERT_EQ(payload->size(), BLOCK_BYTES * BLOCK_COUNT);
        for (std::size_t block = 0; block < BLOCK_COUNT; ++block) {
            const auto bytes = pattern_bytes(BLOCK_BYTES, static_cast<std::uint32_t>(block));
            ASSERT_EQ(std::memcmp(payload->data() + block * BLOCK_BYTES, bytes.data(), BLOCK_BYTES), 0)
                << "block " << block;
        }

        const ChunkInfo* info_trailing = reader->find(FOURCC_AAAA);
        ASSERT_NE(info_trailing, nullptr);
        auto trailing_read = reader->read_chunk(*info_trailing);
        ASSERT_TRUE(trailing_read);
        EXPECT_EQ(*trailing_read, trailing);
    }

    TEST_F(ProjectContainerTest, StreamedEmptyChunk) {
        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        auto stream = writer->begin_chunk(FOURCC_STRM, {});
        ASSERT_TRUE(stream);
        ASSERT_TRUE(writer->end_chunk());
        ASSERT_TRUE(writer->finalize());

        auto reader = ProjectReader::open(file_path());
        ASSERT_TRUE(reader) << reader.error().format();
        const ChunkInfo* info = reader->find(FOURCC_STRM);
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->stored_size, 0u);
        auto payload = reader->read_chunk(*info);
        ASSERT_TRUE(payload);
        EXPECT_TRUE(payload->empty());
    }

    TEST_F(ProjectContainerTest, TruncatedFileFailsLoudly) {
        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, pattern_bytes(100 * 1024, 1), {}));
        ASSERT_TRUE(writer->finalize());

        std::uint64_t payload_offset = 0;
        {
            auto reader = ProjectReader::open(file_path());
            ASSERT_TRUE(reader);
            payload_offset = chunk_payload_offset(*reader->find(FOURCC_AAAA));
        }

        const auto full_size = fs::file_size(file_path());

        fs::copy_file(file_path(), file_path("mid_payload.licht"));
        fs::resize_file(file_path("mid_payload.licht"), payload_offset + 10);
        auto mid_payload = ProjectReader::open(file_path("mid_payload.licht"));
        ASSERT_FALSE(mid_payload);
        EXPECT_EQ(mid_payload.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);

        fs::copy_file(file_path(), file_path("mid_footer.licht"));
        fs::resize_file(file_path("mid_footer.licht"), full_size - 10);
        auto mid_footer = ProjectReader::open(file_path("mid_footer.licht"));
        ASSERT_FALSE(mid_footer);
        EXPECT_EQ(mid_footer.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
    }

    TEST_F(ProjectContainerTest, CorruptFooterFallsBackToHeaderIndex) {
        const auto payload = pattern_bytes(2048, 9);
        auto writer = ProjectWriter::create(file_path(), 3, TEST_UUID);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, payload, {}));
        ASSERT_TRUE(writer->finalize());

        const auto size = fs::file_size(file_path());
        flip_byte_at(file_path(), size - sizeof(Footer) + offsetof(Footer, index_offset));

        auto reader = ProjectReader::open(file_path());
        ASSERT_TRUE(reader) << reader.error().format();
        EXPECT_EQ(reader->save_generation(), 3u);
        const ChunkInfo* info = reader->find(FOURCC_AAAA);
        ASSERT_NE(info, nullptr);
        auto read = reader->read_chunk(*info);
        ASSERT_TRUE(read) << read.error().format();
        EXPECT_EQ(*read, payload);
    }

    TEST_F(ProjectContainerTest, CorruptIndexFailsBothSourcesLoudly) {
        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, pattern_bytes(256, 1), {}));
        ASSERT_TRUE(writer->finalize());

        std::uint64_t index_payload_offset = 0;
        {
            auto reader = ProjectReader::open(file_path());
            ASSERT_TRUE(reader);
            index_payload_offset = chunk_payload_offset(reader->index_chunk_info());
        }
        flip_byte_at(file_path(), index_payload_offset + sizeof(IndexPrologue) + 4);

        auto reader = ProjectReader::open(file_path());
        ASSERT_FALSE(reader);
        EXPECT_EQ(reader.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
        EXPECT_NE(reader.error().message.find("header fallback"), std::string::npos);
    }

    TEST_F(ProjectContainerTest, CorruptChunkCrcIsIsolated) {
        const auto payload_a = pattern_bytes(512, 1);
        const auto payload_b = pattern_bytes(512, 2);
        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, payload_a, {}));
        ASSERT_TRUE(writer->write_chunk(FOURCC_BBBB, payload_b, {}));
        ASSERT_TRUE(writer->finalize());

        std::uint64_t corrupt_offset = 0;
        {
            auto reader = ProjectReader::open(file_path());
            ASSERT_TRUE(reader);
            corrupt_offset = chunk_payload_offset(*reader->find(FOURCC_BBBB)) + 100;
        }
        flip_byte_at(file_path(), corrupt_offset);

        auto reader = ProjectReader::open(file_path());
        ASSERT_TRUE(reader) << reader.error().format();

        auto read_b = reader->read_chunk(*reader->find(FOURCC_BBBB));
        ASSERT_FALSE(read_b);
        EXPECT_EQ(read_b.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
        EXPECT_NE(read_b.error().message.find("CRC"), std::string::npos);

        auto read_a = reader->read_chunk(*reader->find(FOURCC_AAAA));
        ASSERT_TRUE(read_a) << read_a.error().format();
        EXPECT_EQ(*read_a, payload_a);
    }

    TEST_F(ProjectContainerTest, UnknownChunkCopyThroughIsByteExact) {
        const auto unknown_payload = compressible_bytes(30000);
        const auto page_payload = pattern_bytes(8192, 5);
        const std::uint32_t unknown_fourcc = make_fourcc('W', 'H', 'A', 'T');

        auto writer_a = ProjectWriter::create(file_path("a.licht"), 1, TEST_UUID);
        ASSERT_TRUE(writer_a);
        ASSERT_TRUE(writer_a->write_chunk(unknown_fourcc, unknown_payload,
                                          {.chunk_version = 9, .instance_id = 4, .compression = Compression::Zstd, .generation = 11}));
        ASSERT_TRUE(writer_a->write_chunk(FOURCC_PAGE, page_payload, {.page_align = true}));
        ASSERT_TRUE(writer_a->finalize());

        auto reader_a = ProjectReader::open(file_path("a.licht"));
        ASSERT_TRUE(reader_a);
        const ChunkInfo* info_a = reader_a->find(unknown_fourcc, 4);
        const ChunkInfo* page_a = reader_a->find(FOURCC_PAGE);
        ASSERT_NE(info_a, nullptr);
        ASSERT_NE(page_a, nullptr);

        auto writer_b = ProjectWriter::create(file_path("b.licht"), 2, TEST_UUID);
        ASSERT_TRUE(writer_b);
        ASSERT_TRUE(writer_b->write_chunk(FOURCC_AAAA, pattern_bytes(50, 1), {}));
        auto copied = writer_b->copy_chunk(*reader_a, *info_a);
        ASSERT_TRUE(copied) << copied.error().format();
        auto copied_page = writer_b->copy_chunk(*reader_a, *page_a);
        ASSERT_TRUE(copied_page) << copied_page.error().format();
        ASSERT_TRUE(writer_b->finalize());

        auto reader_b = ProjectReader::open(file_path("b.licht"));
        ASSERT_TRUE(reader_b) << reader_b.error().format();
        const ChunkInfo* info_b = reader_b->find(unknown_fourcc, 4);
        ASSERT_NE(info_b, nullptr);
        EXPECT_EQ(info_b->offset, copied->offset);
        EXPECT_EQ(info_b->chunk_version, 9u);
        EXPECT_EQ(info_b->generation, 11u);
        EXPECT_EQ(info_b->compression, Compression::Zstd);
        EXPECT_EQ(info_b->stored_size, info_a->stored_size);
        EXPECT_EQ(info_b->payload_crc32c, info_a->payload_crc32c);

        const auto stored_a = read_file_range(file_path("a.licht"), chunk_payload_offset(*info_a),
                                              info_a->stored_size);
        const auto stored_b = read_file_range(file_path("b.licht"), chunk_payload_offset(*info_b),
                                              info_b->stored_size);
        EXPECT_EQ(stored_a, stored_b);

        auto decoded = reader_b->read_chunk(*info_b);
        ASSERT_TRUE(decoded) << decoded.error().format();
        EXPECT_EQ(*decoded, unknown_payload);

        const ChunkInfo* page_b = reader_b->find(FOURCC_PAGE);
        ASSERT_NE(page_b, nullptr);
        EXPECT_EQ(chunk_payload_offset(*page_b) % PAGE_ALIGNMENT, 0u);
        auto page_decoded = reader_b->read_chunk(*page_b);
        ASSERT_TRUE(page_decoded);
        EXPECT_EQ(*page_decoded, page_payload);
    }

    TEST_F(ProjectContainerTest, PrevIndexChainAcrossTwoGenerations) {
        const auto payload_a = pattern_bytes(300, 1);
        const auto payload_b = pattern_bytes(5000, 2);
        const auto payload_c = pattern_bytes(120, 3);
        const std::uint32_t fourcc_c = make_fourcc('C', 'C', 'C', 'C');

        auto writer_1 = ProjectWriter::create(file_path("gen1.licht"), 1, TEST_UUID);
        ASSERT_TRUE(writer_1);
        ASSERT_TRUE(writer_1->write_chunk(FOURCC_AAAA, payload_a, {.generation = 1}));
        ASSERT_TRUE(writer_1->write_chunk(FOURCC_BBBB, payload_b, {.generation = 1}));
        ASSERT_TRUE(writer_1->finalize());

        auto reader_1 = ProjectReader::open(file_path("gen1.licht"));
        ASSERT_TRUE(reader_1);
        EXPECT_EQ(reader_1->prev_index_offset(), 0u);
        const std::uint64_t index_1_offset = reader_1->index_chunk_info().offset;

        // Generation 2 reproduces generation 1's layout by copying its chunks
        // in order (identical alignment => identical offsets), so the
        // preserved INDX chunk's absolute row offsets stay valid.
        auto writer_2 = ProjectWriter::create(file_path("gen2.licht"), 2, TEST_UUID);
        ASSERT_TRUE(writer_2);
        for (const auto& info : reader_1->chunks()) {
            auto copied = writer_2->copy_chunk(*reader_1, info);
            ASSERT_TRUE(copied) << copied.error().format();
            EXPECT_EQ(copied->offset, info.offset);
        }
        auto copied_index = writer_2->copy_chunk(*reader_1, reader_1->index_chunk_info());
        ASSERT_TRUE(copied_index) << copied_index.error().format();
        EXPECT_EQ(copied_index->offset, index_1_offset);
        writer_2->set_prev_index_offset(copied_index->offset);
        ASSERT_TRUE(writer_2->write_chunk(fourcc_c, payload_c, {.generation = 2}));
        ASSERT_TRUE(writer_2->finalize());

        auto reader_2 = ProjectReader::open(file_path("gen2.licht"));
        ASSERT_TRUE(reader_2) << reader_2.error().format();
        EXPECT_EQ(reader_2->save_generation(), 2u);
        EXPECT_EQ(reader_2->prev_index_offset(), index_1_offset);
        ASSERT_EQ(reader_2->chunks().size(), 4u);

        std::uint64_t prev_prev = 42;
        auto prev_rows = reader_2->open_prev_index(reader_2->prev_index_offset(), &prev_prev);
        ASSERT_TRUE(prev_rows) << prev_rows.error().format();
        EXPECT_EQ(prev_prev, 0u);
        ASSERT_EQ(prev_rows->size(), 2u);

        for (const auto& row : *prev_rows) {
            auto payload = reader_2->read_chunk(row);
            ASSERT_TRUE(payload) << payload.error().format();
            if (row.fourcc == FOURCC_AAAA) {
                EXPECT_EQ(*payload, payload_a);
            } else {
                ASSERT_EQ(row.fourcc, FOURCC_BBBB);
                EXPECT_EQ(*payload, payload_b);
            }
        }

        auto read_c = reader_2->read_chunk(*reader_2->find(fourcc_c));
        ASSERT_TRUE(read_c);
        EXPECT_EQ(*read_c, payload_c);
    }

    TEST_F(ProjectContainerTest, RejectsGarbageFiles) {
        {
            std::ofstream empty(file_path("empty.licht"), std::ios::binary);
        }
        auto empty = ProjectReader::open(file_path("empty.licht"));
        ASSERT_FALSE(empty);
        EXPECT_EQ(empty.error().code, lfs::io::ErrorCode::INVALID_HEADER);

        {
            std::ofstream garbage(file_path("garbage.licht"), std::ios::binary);
            const auto bytes = pattern_bytes(4096, 77);
            garbage.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
        }
        auto garbage = ProjectReader::open(file_path("garbage.licht"));
        ASSERT_FALSE(garbage);
        EXPECT_EQ(garbage.error().code, lfs::io::ErrorCode::INVALID_HEADER);

        auto writer = ProjectWriter::create(file_path("magic.licht"), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, pattern_bytes(64, 1), {}));
        ASSERT_TRUE(writer->finalize());
        flip_byte_at(file_path("magic.licht"), 0);
        auto wrong_magic = ProjectReader::open(file_path("magic.licht"));
        ASSERT_FALSE(wrong_magic);
        EXPECT_EQ(wrong_magic.error().code, lfs::io::ErrorCode::INVALID_HEADER);

        auto missing = ProjectReader::open(file_path("does_not_exist.licht"));
        ASSERT_FALSE(missing);
        EXPECT_EQ(missing.error().code, lfs::io::ErrorCode::PATH_NOT_FOUND);
    }

    TEST_F(ProjectContainerTest, MinReaderVersionRefused) {
        auto writer = ProjectWriter::create(file_path(), 1, TEST_UUID);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->write_chunk(FOURCC_AAAA, pattern_bytes(64, 1), {}));
        ASSERT_TRUE(writer->finalize());

        write_u32_at(file_path(), offsetof(FileHeader, min_reader_version), CONTAINER_VERSION + 1);

        auto reader = ProjectReader::open(file_path());
        ASSERT_FALSE(reader);
        EXPECT_EQ(reader.error().code, lfs::io::ErrorCode::UNSUPPORTED_FORMAT);
        EXPECT_NE(reader.error().message.find("newer"), std::string::npos);
    }

} // namespace
