/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/uuid.hpp"
#include "io/project/crc32c.hpp"
#include "io/project/project_container_internal.hpp"
#include "io/project_container.hpp"
#include "io/project_path.hpp"
#include "io/project_recovery.hpp"
#include "licht_test_support.hpp"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <limits.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::io::project::detail {
    void reset_framed_record_decode_calls_for_testing();
    std::uint64_t framed_record_decode_calls_for_testing();
} // namespace lfs::io::project::detail

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    using namespace std::string_view_literals;

    constexpr std::uint64_t FIXED_CREATION_TIME_NS =
        1'735'689'600'000'000'000;
    constexpr std::uint64_t FIXED_COMMIT_TIME_NS =
        1'735'689'601'000'000'000;

    CreateOptions fixture_create_options(const std::uint64_t file_tag) {
        return {
            .project_uuid = fixed_uuid(1),
            .file_uuid = fixed_uuid(file_tag),
            .role = ContainerRole::Master,
            .creation_time_unix_ns = FIXED_CREATION_TIME_NS,
            .index_compression = IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    CommitOptions fixture_commit_options(const std::uint64_t commit_tag,
                                         const std::uint64_t snapshot_tag,
                                         const std::uint64_t generation) {
        return {
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(commit_tag),
            .snapshot_uuid = fixed_uuid(snapshot_tag),
            .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + generation,
        };
    }

    AppendOptions fixture_append_options() {
        return {
            .compatibility = {},
            .index_compression = IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

#ifndef _WIN32
    template <typename Work>
    int run_child_process(Work&& work, const int exception_exit = 126,
                          const int normal_exit = 0) {
        const pid_t child = ::fork();
        if (child < 0) {
            throw std::runtime_error("fork failed in test driver");
        }
        if (child == 0) {
            try {
                work();
            } catch (...) {
                ::_exit(exception_exit);
            }
            ::_exit(normal_exit);
        }
        int status = 0;
        if (::waitpid(child, &status, 0) != child) {
            throw std::runtime_error("waitpid failed in test driver");
        }
        return status;
    }
#endif

    std::uint32_t crc_range(const std::span<const std::byte> bytes,
                            const std::size_t offset,
                            const std::size_t count) {
        return crc32c(0, bytes.data() + offset, count);
    }

    std::vector<std::byte> raw_framed(
        const std::uint32_t count,
        const std::vector<std::pair<std::uint64_t, std::uint64_t>>& records,
        const std::size_t tail_bytes = 0) {
        const std::size_t table = 16 + static_cast<std::size_t>(count) * 16;
        std::vector<std::byte> bytes(table + tail_bytes);
        const std::array magic = {std::byte{'L'}, std::byte{'F'}, std::byte{'S'},
                                  std::byte{'Z'}, std::byte{'F'}, std::byte{'R'},
                                  std::byte{'M'}, std::byte{0}};
        std::copy(magic.begin(), magic.end(), bytes.begin());
        auto put16 = [&](const std::size_t at, const std::uint16_t value) {
            bytes[at] = static_cast<std::byte>(value & 0xffu);
            bytes[at + 1] = static_cast<std::byte>(value >> 8);
        };
        auto put32 = [&](const std::size_t at, const std::uint32_t value) {
            for (std::size_t i = 0; i < 4; ++i)
                bytes[at + i] = static_cast<std::byte>(value >> (8 * i));
        };
        auto put64 = [&](const std::size_t at, const std::uint64_t value) {
            for (std::size_t i = 0; i < 8; ++i)
                bytes[at + i] = static_cast<std::byte>(value >> (8 * i));
        };
        put16(8, 1);
        put16(10, 0);
        put32(12, count);
        for (std::size_t i = 0; i < records.size(); ++i) {
            put64(16 + i * 16, records[i].first);
            put64(24 + i * 16, records[i].second);
        }
        return bytes;
    }

    std::vector<std::byte>
    create_single_chunk_fixture(const fs::path& path,
                                const std::uint64_t file_tag,
                                const std::uint64_t commit_tag,
                                const std::uint64_t snapshot_tag,
                                const ChunkKey& key,
                                const std::string_view payload_text,
                                const CommitOptions* custom_commit = nullptr) {
        ProjectWriter writer = require_result(
            ProjectWriter::create(path, fixture_create_options(file_tag)));
        const auto payload = byte_vector(payload_text);
        const CommitOptions commit_options =
            custom_commit != nullptr
                ? *custom_commit
                : fixture_commit_options(commit_tag, snapshot_tag, 1);
        require_status(writer.plan_commit(commit_options));
        require_status(writer.preflight(payload.size()));
        require_status(writer.write_chunk(key, payload));
        require_status(writer.commit());
        return read_file_bytes(path);
    }

    std::string recoverable_scene_graph_payload() {
        SceneGraphChapter graph;
        SceneNodeRecord node;
        node.uuid = lfs::core::generate_uuid_v4();
        node.type = "group";
        node.name = "Recoverable";
        require_status(graph.upsert_node(node));
        const auto bytes = graph.to_bytes();
        return std::string(
            reinterpret_cast<const char*>(bytes.data()),
            bytes.size());
    }

    void publish_complete_sidecar(
        const fs::path& master_path,
        const fs::path& sidecar_path,
        const std::uint64_t sequence,
        const std::uint64_t file_tag,
        const std::uint64_t commit_tag,
        const std::uint64_t snapshot_tag,
        CommitOptions commit = {},
        CommitBoundaryObserver observer = {}) {
        ProjectReader master =
            require_result(ProjectReader::open(master_path));
        const auto snapshot_uuid = fixed_uuid(snapshot_tag);
        ProjectWriter writer = require_result(ProjectWriter::create(
            sidecar_path,
            CreateOptions{
                .project_uuid = master.superblock().project_uuid,
                .file_uuid = fixed_uuid(file_tag),
                .role = ContainerRole::AutosaveSidecar,
                .base_explicit_commit_uuid = master.commit().commit_uuid,
                .autosave_sequence = sequence,
                .sidecar_snapshot_uuid = snapshot_uuid,
                .creation_time_unix_ns = FIXED_CREATION_TIME_NS + sequence,
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
                .boundary_observer = std::move(observer),
                .writer_lock_anchor = master_path,
            }));
        if (commit.commit_uuid.is_nil()) {
            commit = fixture_commit_options(commit_tag, snapshot_tag, 1);
        }
        commit.kind = CommitKind::Autosave;
        commit.snapshot_uuid = snapshot_uuid;
        require_status(writer.plan_commit(commit));
        require_status(writer.preflight(0));
        for (const ChunkInfo& row : master.chunks()) {
            if (row.row_kind == RowKind::Live) {
                require_status(writer.add_sidecar_base_reference(row));
            }
        }
        require_status(writer.commit());
    }

    [[nodiscard]] std::vector<fs::path> corrupt_asides_of(
        const fs::path& original) {
        std::vector<fs::path> found;
        const auto prefix = original.filename().string() + ".corrupt-";
        const auto directory = original.parent_path().empty()
                                   ? fs::path{"."}
                                   : original.parent_path();
        std::error_code error;
        for (fs::directory_iterator iterator(directory, error), end;
             !error && iterator != end; iterator.increment(error)) {
            const auto name = iterator->path().filename().string();
            if (name.starts_with(prefix)) {
                found.push_back(iterator->path());
            }
        }
        return found;
    }

    TEST(ProjectContainerFormat, Crc32cKnownVector) {
        constexpr std::string_view CHECK = "123456789";
        EXPECT_EQ(crc32c(0, CHECK.data(), CHECK.size()), 0xe3069283u);
        EXPECT_EQ(crc32c(0, CHECK.data(), 0), 0u);
        EXPECT_EQ(crc32c(crc32c(0, CHECK.data(), 3), CHECK.data() + 3, 6),
                  0xe3069283u);
    }

    TEST(ProjectContainerFormat, Crc32cKnownAnswersLengths0To9) {
        // Kills: crc32c const/arithmetic survivors on short lengths.
        // Castagnoli CRC32C of bytes {0,1,...,n-1} for n in 0..9 (seed 0).
        constexpr std::array<std::uint32_t, 10> kExpected{
            0x00000000u,
            0x527d5351u,
            0x030af4d1u,
            0x92fd4bfau,
            0xd9331aa3u,
            0x2425b106u,
            0x41098514u,
            0xa359ed4cu,
            0x8a2cbc3bu,
            0x7144c5a8u,
        };
        std::array<std::uint8_t, 9> sequential{};
        for (std::size_t n = 0; n < sequential.size(); ++n) {
            sequential[n] = static_cast<std::uint8_t>(n);
        }
        for (std::size_t n = 0; n <= 9; ++n) {
            EXPECT_EQ(crc32c(0, sequential.data(), n), kExpected[n]) << "n=" << n;
        }
    }

    TEST(ProjectContainerFormat, Crc32cCombineMatchesConcatenation) {
        constexpr std::string_view kKnown = "123456789";
        EXPECT_EQ(crc32c_combine(crc32c(0, kKnown.data(), 3),
                                 crc32c(0, kKnown.data() + 3, 6), 6),
                  crc32c(0, kKnown.data(), kKnown.size()));
        EXPECT_EQ(crc32c_combine(crc32c(0, kKnown.data(), kKnown.size()),
                                 crc32c(0, kKnown.data(), 0), 0),
                  crc32c(0, kKnown.data(), kKnown.size()));

        const std::array<std::size_t, 6> suffixes{
            0, 1, 17, 1024, BLOCK_CRC_BYTES + 17,
            static_cast<std::size_t>(BLOCK_CRC_BYTES) + 256u * 1024u};
        for (const std::size_t n2 : suffixes) {
            const std::size_t n1 = 64;
            std::vector<std::uint8_t> prefix(n1);
            std::vector<std::uint8_t> suffix(n2);
            for (std::size_t i = 0; i < n1; ++i) {
                prefix[i] = static_cast<std::uint8_t>(i * 3u + 1u);
            }
            for (std::size_t i = 0; i < n2; ++i) {
                suffix[i] = static_cast<std::uint8_t>(i * 7u + 11u);
            }
            std::vector<std::uint8_t> joined;
            joined.reserve(n1 + n2);
            joined.insert(joined.end(), prefix.begin(), prefix.end());
            joined.insert(joined.end(), suffix.begin(), suffix.end());
            const auto crc_a = crc32c(0, prefix.data(), prefix.size());
            const auto crc_b = crc32c(0, suffix.data(), suffix.size());
            const auto crc_ab = crc32c(0, joined.data(), joined.size());
            EXPECT_EQ(crc32c_combine(crc_a, crc_b, suffix.size()), crc_ab)
                << "n2=" << n2;
        }
    }

    TEST(ProjectContainerReader, ThreeGenerationOpenSelectsNewestHead) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "three-generations.licht";
        const ChunkKey key = fixed_key("PROJ", 941);
        for (std::uint64_t generation = 1; generation <= 3; ++generation) {
            ProjectWriter writer = require_result(
                generation == 1
                    ? ProjectWriter::create(path, fixture_create_options(940))
                    : ProjectWriter::append(path, fixture_append_options()));
            const auto payload = byte_vector(
                std::format(R"({{"generation":{}}})", generation));
            require_status(writer.plan_commit(fixture_commit_options(
                940 + generation * 3, 941 + generation * 3, generation)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(key, payload));
            require_status(writer.commit());
        }
        ProjectReader reader = require_result(ProjectReader::open(path));
        EXPECT_EQ(reader.commit().generation, 3u);
        EXPECT_EQ(reader.selected_head().generation, 3u);
        EXPECT_EQ(reader.selected_head().head_sequence, 3u);
        EXPECT_EQ(reader.commit().commit_uuid, fixed_uuid(949));
        const ChunkInfo* row = reader.find(key);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(require_result(reader.read_chunk(*row)),
                  byte_vector(R"({"generation":3})"));
    }

    TEST(ProjectContainerReader,
         PositionalReadValidatesOnlyTouchedBlockCrcRanges) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "partial-block-crc.licht";
        const std::size_t payload_bytes =
            static_cast<std::size_t>(BLOCK_CRC_BYTES * 3);
        std::vector<std::byte> payload(payload_bytes);
        for (std::size_t index = 0; index < payload.size(); ++index) {
            payload[index] = static_cast<std::byte>(index * 131u);
        }
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path, fixture_create_options(810)));
            require_status(writer.plan_commit(
                fixture_commit_options(811, 812, 1)));
            require_status(writer.preflight(payload.size()));
            ChunkWriteOptions options{
                .chunk_version = 1,
                .compression = Compression::Stored,
                .tensor_payload = false,
                .block_crcs = true,
            };
            require_status(writer.write_chunk(fixed_key("SPLT", 813), payload,
                                              options));
            require_status(writer.commit());
        }

        ProjectReader reader = require_result(ProjectReader::open(path));
        ASSERT_EQ(reader.chunks().size(), 1u);
        const ChunkInfo& chunk = reader.chunks().front();
        ASSERT_TRUE(chunk.block_crc_table.has_value());
        const std::uint64_t corrupt_offset =
            chunk.payload_offset + BLOCK_CRC_BYTES + 23;
        const std::array corruption = {std::byte{0xff}};
        write_file_range(path, corrupt_offset, corruption);

        auto full = reader.read_chunk(chunk);
        EXPECT_FALSE(full);
        std::array<std::byte, 64> touched{};
        auto touched_result =
            reader.read_stored_at(chunk, BLOCK_CRC_BYTES + 8, touched);
        EXPECT_FALSE(touched_result);
        std::array<std::byte, 64> untouched{};
        auto untouched_result = reader.read_stored_at(chunk, 128, untouched);
        EXPECT_TRUE(untouched_result)
            << (untouched_result
                    ? std::string{}
                    : lfs::format_for_developer(untouched_result.error()));
        EXPECT_TRUE(std::equal(untouched.begin(), untouched.end(),
                               payload.begin() + 128));
    }

    TEST(ProjectContainerReader, ReadChunkRejectsPayloadCrcCorruption) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "payload-crc.licht";
        std::vector<std::byte> payload(4096);
        for (std::size_t index = 0; index < payload.size(); ++index) {
            payload[index] = static_cast<std::byte>((index * 17u) ^ (index >> 3));
        }
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path, fixture_create_options(818)));
            require_status(writer.plan_commit(fixture_commit_options(819, 820, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(
                fixed_key("PROJ", 821), payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::Stored,
                    .tensor_payload = false,
                    .block_crcs = false,
                }));
            require_status(writer.commit());
        }
        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo& chunk = reader.chunks().front();
        write_file_range(path, chunk.payload_offset + 137,
                         std::array{std::byte{0x7f}});
        auto result = reader.read_chunk(chunk);
        EXPECT_FALSE(result);
        if (!result) {
            EXPECT_NE(lfs::format_for_developer(result.error()).find("crc32c"),
                      std::string::npos);
        }
    }

    TEST(ProjectContainerReader, ReadChunkRejectsTruncatedChunk) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "truncated-chunk.licht";
        const auto payload = byte_vector("payload that is longer than one byte");
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path, fixture_create_options(822)));
            require_status(writer.plan_commit(fixture_commit_options(823, 824, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(fixed_key("PROJ", 825), payload));
            require_status(writer.commit());
        }
        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo& chunk = reader.chunks().front();
        ASSERT_GT(chunk.stored_bytes, 1u);
        fs::resize_file(path, chunk.payload_offset + chunk.stored_bytes - 1);
        EXPECT_FALSE(reader.read_chunk(chunk));
    }

    TEST(ProjectContainerReader, FramedPayloadCorruptionIsRejected) {
        const auto check = [](const std::string_view name,
                              const std::function<void(const ChunkInfo&, const fs::path&)>& corrupt) {
            TemporaryDirectory temporary;
            const fs::path path = temporary.path / (std::string(name) + ".licht");
            const std::size_t payload_size = 64ull * 1024 * 1024 + 4096;
            std::vector<std::byte> payload(payload_size);
            for (std::size_t index = 0; index < payload.size(); ++index)
                payload[index] = static_cast<std::byte>((index * 31u) ^ (index >> 7));
            ProjectWriter writer = require_result(ProjectWriter::create(
                path, fixture_create_options(830)));
            require_status(writer.plan_commit(fixture_commit_options(831, 832, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(
                fixed_key("SPLT", 833), payload,
                ChunkWriteOptions{.chunk_version = 1,
                                  .compression = Compression::ZstdFramed,
                                  .tensor_payload = true,
                                  .block_crcs = true}));
            require_status(writer.commit());
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            corrupt(chunk, path);
            EXPECT_FALSE(reader.read_chunk(chunk)) << name;
        };
        check("framed-count", [](const ChunkInfo& chunk, const fs::path& path) {
            write_file_range(path, chunk.payload_offset + 12,
                             std::array{std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}});
        });
        check("framed-size", [](const ChunkInfo& chunk, const fs::path& path) {
            write_file_range(path, chunk.payload_offset + 24,
                             std::array{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}});
        });
        check("framed-truncated", [](const ChunkInfo& chunk, const fs::path& path) {
            fs::resize_file(path, chunk.payload_offset + chunk.stored_bytes - 1);
        });
        check("framed-crc", [](const ChunkInfo& chunk, const fs::path& path) {
            write_file_range(path, chunk.payload_offset + 4096,
                             std::array{std::byte{0x7f}});
        });
    }

    TEST(ProjectContainerReader, FramedParserRejectsMalformedHeadersAndTables) {
        const auto reject = [](std::vector<std::byte> bytes,
                               const std::uint64_t expected) {
            auto result = detail::decompress_framed_zstd_for_testing(
                "framed-corruption.licht", 0, bytes, expected, 64ull * 1024 * 1024);
            EXPECT_FALSE(result);
        };

        auto bad_magic = raw_framed(1, {{1, 1}}, 1);
        bad_magic[0] = std::byte{'X'};
        reject(std::move(bad_magic), 1);

        auto bad_version = raw_framed(1, {{1, 1}}, 1);
        bad_version[8] = std::byte{2};
        reject(std::move(bad_version), 1);

        auto bad_reserved = raw_framed(1, {{1, 1}}, 1);
        bad_reserved[10] = std::byte{1};
        reject(std::move(bad_reserved), 1);

        auto bad_count = raw_framed(0, {}, 1);
        reject(std::move(bad_count), 1);

        auto count_bomb = raw_framed(2, {{1, 1}, {1, 1}}, 2);
        auto count_result = detail::decompress_framed_zstd_for_testing(
            "framed-corruption.licht", 0, count_bomb, 1, 64ull * 1024 * 1024);
        ASSERT_FALSE(count_result);
        EXPECT_NE(lfs::format_for_developer(count_result.error()).find("invalid record count"),
                  std::string::npos);

        auto table_too_large = raw_framed(2, {}, 0);
        table_too_large.resize(16);
        reject(std::move(table_too_large), 0);

        auto zero_stored = raw_framed(1, {{0, 1}}, 1);
        reject(std::move(zero_stored), 1);

        auto zero_uncompressed = raw_framed(1, {{1, 0}}, 1);
        reject(std::move(zero_uncompressed), 1);

        auto stored_short = raw_framed(1, {{1, 1}}, 2);
        reject(std::move(stored_short), 1);

        auto stored_long = raw_framed(1, {{3, 1}}, 2);
        reject(std::move(stored_long), 1);

        auto decoded_short = raw_framed(1, {{1, 1}}, 1);
        reject(std::move(decoded_short), 2);

        const std::array source = {std::byte{0x42}};
        std::vector<std::byte> compressed(ZSTD_compressBound(source.size()));
        const auto compressed_size = ZSTD_compress(
            compressed.data(), compressed.size(), source.data(), source.size(), 1);
        ASSERT_FALSE(ZSTD_isError(compressed_size));
        ASSERT_GT(compressed_size, 1u);
        auto truncated = raw_framed(1, {{compressed_size - 1, 1}}, compressed_size - 1);
        std::copy_n(compressed.begin(), compressed_size - 1,
                    truncated.begin() + 32);
        reject(std::move(truncated), 1);
    }

    TEST(ProjectContainerReader, HardwareAndSoftwareCrc32cAgree) {
        std::vector<std::byte> bytes(1 << 20);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<std::byte>((i * 37u) ^ (i >> 9));
        const auto software = crc32c_software(0, bytes.data(), bytes.size());
        EXPECT_EQ(crc32c(0, bytes.data(), bytes.size()), software);
#if defined(__x86_64__) || defined(_M_X64)
        EXPECT_EQ(crc32c_sse42(0, bytes.data(), bytes.size()), software);
#endif
    }

    TEST(ProjectContainerReader,
         BoundedStreamValidatesOnlyTouchedBlockCrcRanges) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "partial-stream-crc.licht";
        const std::size_t payload_bytes =
            static_cast<std::size_t>(BLOCK_CRC_BYTES * 2);
        std::vector<std::byte> payload(payload_bytes);
        for (std::size_t index = 0; index < payload.size(); ++index) {
            payload[index] = static_cast<std::byte>(index * 113u);
        }
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path, fixture_create_options(814)));
            require_status(writer.plan_commit(
                fixture_commit_options(815, 816, 1)));
            require_status(writer.preflight(payload.size()));
            ChunkWriteOptions options{
                .chunk_version = 1,
                .compression = Compression::Stored,
                .tensor_payload = false,
                .block_crcs = true,
            };
            require_status(writer.write_chunk(fixed_key("SPLT", 817), payload,
                                              options));
            require_status(writer.commit());
        }

        ProjectReader reader = require_result(ProjectReader::open(path));
        ASSERT_EQ(reader.chunks().size(), 1u);
        const ChunkInfo& chunk = reader.chunks().front();
        ASSERT_TRUE(chunk.block_crc_table.has_value());
        const std::uint64_t corrupt_offset =
            chunk.payload_offset + BLOCK_CRC_BYTES + 23;
        const std::array corruption = {std::byte{0xff}};
        write_file_range(path, corrupt_offset, corruption);

        auto bounded = reader.open_bounded_stream(chunk);
        ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
        std::array<std::byte, 64> untouched{};
        bounded->stream().read(
            reinterpret_cast<char*>(untouched.data()),
            static_cast<std::streamsize>(untouched.size()));
        ASSERT_EQ(bounded->stream().gcount(),
                  static_cast<std::streamsize>(untouched.size()));
        EXPECT_TRUE(std::equal(untouched.begin(), untouched.end(),
                               payload.begin()));

        bounded->stream().seekg(
            static_cast<std::streamoff>(BLOCK_CRC_BYTES + 8),
            std::ios::beg);
        ASSERT_TRUE(bounded->stream());
        std::array<std::byte, 64> corrupt{};
        bounded->stream().read(
            reinterpret_cast<char*>(corrupt.data()),
            static_cast<std::streamsize>(corrupt.size()));
        EXPECT_TRUE(bounded->stream().fail());
        EXPECT_EQ(bounded->stream().gcount(), 0);
    }

    std::vector<std::byte> patterned_payload(const std::size_t bytes) {
        std::vector<std::byte> payload(bytes);
        for (std::size_t index = 0; index + 4 <= bytes; index += 4) {
            const auto word =
                static_cast<std::uint32_t>(index / 4) * 2654435761u;
            payload[index + 0] = static_cast<std::byte>(word);
            payload[index + 1] = static_cast<std::byte>(word >> 8);
            payload[index + 2] = static_cast<std::byte>(word >> 16);
            payload[index + 3] = static_cast<std::byte>(word >> 24);
        }
        for (std::size_t index = (bytes / 4) * 4; index < bytes; ++index) {
            payload[index] =
                static_cast<std::byte>(0xA5u ^ static_cast<unsigned>(index));
        }
        return payload;
    }

    std::vector<std::byte> compressible_payload(const std::size_t bytes) {
        std::vector<std::byte> payload(bytes);
        std::array<std::byte, 256> tile{};
        for (std::size_t index = 0; index < tile.size(); ++index) {
            tile[index] =
                static_cast<std::byte>(static_cast<unsigned>(index * 13u + 7u));
        }
        for (std::size_t offset = 0; offset < bytes;) {
            const auto n = std::min(tile.size(), bytes - offset);
            std::memcpy(payload.data() + offset, tile.data(), n);
            offset += n;
        }
        constexpr std::size_t kStamp = 1024 * 1024;
        for (std::size_t block = 0; block * kStamp < bytes; ++block) {
            payload[block * kStamp] =
                static_cast<std::byte>(static_cast<unsigned>(block & 0xffu));
        }
        return payload;
    }

    std::uint64_t read_le_u64(const std::vector<std::byte>& bytes,
                              const std::size_t at) {
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < 8; ++index) {
            value |= static_cast<std::uint64_t>(
                         std::to_integer<std::uint8_t>(bytes[at + index]))
                     << (8 * index);
        }
        return value;
    }

    struct FramedTableRecord {
        std::uint64_t stored_offset = 0;
        std::uint64_t stored_bytes = 0;
        std::uint64_t decoded_offset = 0;
        std::uint64_t decoded_bytes = 0;
    };

    std::vector<FramedTableRecord> read_framed_table(const fs::path& path,
                                                     const ChunkInfo& chunk) {
        const auto header = read_file_range(path, chunk.payload_offset, 16);
        const auto count = static_cast<std::uint32_t>(header[12]) |
                           (static_cast<std::uint32_t>(header[13]) << 8) |
                           (static_cast<std::uint32_t>(header[14]) << 16) |
                           (static_cast<std::uint32_t>(header[15]) << 24);
        const auto table = 16u + static_cast<std::size_t>(count) * 16u;
        const auto bytes = read_file_range(path, chunk.payload_offset, table);
        std::vector<FramedTableRecord> records;
        records.reserve(count);
        std::uint64_t stored_offset = table;
        std::uint64_t decoded_offset = 0;
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto at = 16u + static_cast<std::size_t>(index) * 16u;
            const auto stored_bytes = read_le_u64(bytes, at);
            const auto decoded_bytes = read_le_u64(bytes, at + 8);
            records.push_back({stored_offset, stored_bytes, decoded_offset,
                               decoded_bytes});
            stored_offset += stored_bytes;
            decoded_offset += decoded_bytes;
        }
        return records;
    }

    std::uint64_t first_logical_byte_of_record(const FramedTableRecord& record,
                                               const std::uint64_t uncompressed,
                                               const bool byteshuffle) {
        if (!byteshuffle) {
            return record.decoded_offset;
        }
        if (uncompressed < 4 || uncompressed % 4 != 0) {
            return record.decoded_offset;
        }
        const std::uint64_t n_words = uncompressed / 4;
        std::uint64_t first = uncompressed;
        for (std::uint64_t plane = 0; plane < 4; ++plane) {
            const std::uint64_t plane_lo = plane * n_words;
            const std::uint64_t plane_hi = plane_lo + n_words;
            const std::uint64_t overlap_lo =
                std::max(record.decoded_offset, plane_lo);
            const std::uint64_t overlap_hi = std::min(
                record.decoded_offset + record.decoded_bytes, plane_hi);
            if (overlap_hi <= overlap_lo) {
                continue;
            }
            const std::uint64_t word = overlap_lo - plane_lo;
            first = std::min(first, word * 4 + plane);
        }
        return first;
    }

    [[nodiscard]] std::optional<std::string>
    error_field_string(const lfs::Error& error, std::string_view key);

    struct PayloadCapGuard {
        explicit PayloadCapGuard(const std::optional<std::uint64_t> value) {
            detail::set_max_payload_materialized_bytes_for_testing(value);
        }
        PayloadCapGuard(const PayloadCapGuard&) = delete;
        PayloadCapGuard& operator=(const PayloadCapGuard&) = delete;
        ~PayloadCapGuard() {
            detail::set_max_payload_materialized_bytes_for_testing(std::nullopt);
        }
    };

    void expect_bounded_stream_matches(BoundedInputStream& bounded,
                                       const std::vector<std::byte>& expected) {
        auto& stream = bounded.stream();
        ASSERT_EQ(bounded.size(), expected.size());
        EXPECT_EQ(static_cast<std::uint64_t>(stream.tellg()), 0u);

        std::vector<std::byte> got(expected.size());
        stream.read(reinterpret_cast<char*>(got.data()),
                    static_cast<std::streamsize>(got.size()));
        ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(got.size()));
        EXPECT_EQ(got, expected);

        stream.clear();
        stream.seekg(0);
        ASSERT_TRUE(stream);
        EXPECT_EQ(static_cast<std::uint64_t>(stream.tellg()), 0u);
        std::vector<std::byte> again(expected.size());
        stream.read(reinterpret_cast<char*>(again.data()),
                    static_cast<std::streamsize>(again.size()));
        ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(again.size()));
        EXPECT_EQ(again, expected);

        const auto tail = std::min<std::size_t>(64, expected.size());
        ASSERT_GT(tail, 0u);
        const auto tail_offset = expected.size() - tail;
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(tail_offset), std::ios::beg);
        ASSERT_TRUE(stream);
        EXPECT_EQ(static_cast<std::uint64_t>(stream.tellg()), tail_offset);
        std::vector<std::byte> tail_bytes(tail);
        stream.read(reinterpret_cast<char*>(tail_bytes.data()),
                    static_cast<std::streamsize>(tail_bytes.size()));
        ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(tail));
        EXPECT_TRUE(std::equal(tail_bytes.begin(), tail_bytes.end(),
                               expected.end() - static_cast<std::ptrdiff_t>(tail)));

        if (tail_offset >= 8) {
            stream.clear();
            stream.seekg(static_cast<std::streamoff>(tail_offset), std::ios::beg);
            ASSERT_TRUE(stream);
            stream.seekg(-8, std::ios::cur);
            ASSERT_TRUE(stream);
            EXPECT_EQ(static_cast<std::uint64_t>(stream.tellg()), tail_offset - 8);
            std::array<std::byte, 8> relative{};
            stream.read(reinterpret_cast<char*>(relative.data()),
                        static_cast<std::streamsize>(relative.size()));
            ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(relative.size()));
            EXPECT_TRUE(std::equal(relative.begin(), relative.end(),
                                   expected.begin() +
                                       static_cast<std::ptrdiff_t>(tail_offset - 8)));
        }

        stream.clear();
        stream.seekg(-static_cast<std::streamoff>(tail), std::ios::end);
        ASSERT_TRUE(stream);
        EXPECT_EQ(static_cast<std::uint64_t>(stream.tellg()), tail_offset);
        std::vector<std::byte> from_end(tail);
        stream.read(reinterpret_cast<char*>(from_end.data()),
                    static_cast<std::streamsize>(from_end.size()));
        ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(tail));
        EXPECT_EQ(from_end, tail_bytes);

        stream.clear();
        stream.seekg(static_cast<std::streamoff>(expected.size() + 1),
                     std::ios::beg);
        EXPECT_TRUE(stream.fail());
    }

    void write_framed_fixture(const fs::path& path, const Fourcc fourcc,
                              const std::uint64_t key_tag,
                              const std::vector<std::byte>& payload,
                              const Compression compression,
                              const bool tensor_payload, const bool block_crcs,
                              const std::uint64_t file_tag) {
        ProjectWriter writer = require_result(
            ProjectWriter::create(path, fixture_create_options(file_tag)));
        require_status(writer.plan_commit(
            fixture_commit_options(file_tag + 1, file_tag + 2, 1)));
        require_status(writer.preflight(payload.size()));
        require_status(writer.write_chunk(
            ChunkKey{.fourcc = fourcc, .instance_uuid = fixed_uuid(key_tag)},
            payload,
            ChunkWriteOptions{
                .chunk_version = 1,
                .compression = compression,
                .tensor_payload = tensor_payload,
                .block_crcs = block_crcs,
            }));
        require_status(writer.commit());
    }

    TEST(ProjectContainerReader, BoundedStreamMatchesReadChunkForFramedPayloads) {
        constexpr std::size_t kBytes =
            2ull * 64ull * 1024ull * 1024ull + 32ull * 1024ull * 1024ull;
        const auto payload = patterned_payload(kBytes);
        const auto check = [&](const Compression compression,
                               const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-multi.licht");
            write_framed_fixture(path, FOURCC_CKPT, 840, payload, compression,
                                 true, true, 840);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            EXPECT_EQ(chunk.compression, compression) << name;
            EXPECT_GE(chunk.uncompressed_bytes,
                      2ull * 64ull * 1024ull * 1024ull + 1ull)
                << name;
            const auto header = read_file_range(path, chunk.payload_offset, 16);
            const auto record_count =
                static_cast<std::uint32_t>(header[12]) |
                (static_cast<std::uint32_t>(header[13]) << 8) |
                (static_cast<std::uint32_t>(header[14]) << 16) |
                (static_cast<std::uint32_t>(header[15]) << 24);
            ASSERT_GE(record_count, 3u) << name;
            const auto materialized = require_result(reader.read_chunk(chunk));
            ASSERT_EQ(materialized, payload) << name;
            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            expect_bounded_stream_matches(*bounded, materialized);
        };
        check(Compression::ZstdFramed, "zstd-framed");
        check(Compression::ByteShuffleZstdFramed, "byteshuffle-framed");
    }

    TEST(ProjectContainerReader, BoundedStreamMatchesReadChunkForSmallFramedPayloads) {
        const auto check = [](const std::string_view name,
                              const std::vector<std::byte>& payload,
                              const Compression requested) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-small.licht");
            write_framed_fixture(path, FOURCC_SPLT, 850, payload, requested, true,
                                 false, 850);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            if (requested == Compression::ByteShuffleZstdFramed &&
                payload.size() % 4 != 0) {
                EXPECT_EQ(chunk.compression, Compression::ZstdFramed) << name;
            } else {
                EXPECT_EQ(chunk.compression, requested) << name;
            }
            const auto materialized = require_result(reader.read_chunk(chunk));
            ASSERT_EQ(materialized, payload) << name;
            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            expect_bounded_stream_matches(*bounded, materialized);
        };
        check("zstd-single", patterned_payload(4096), Compression::ZstdFramed);
        check("byteshuffle-single", patterned_payload(4096),
              Compression::ByteShuffleZstdFramed);
        check("zstd-fallback-non-multiple-of-4", patterned_payload(4097),
              Compression::ByteShuffleZstdFramed);
    }

    // 7-byte chunked drain of a framed payload. Get-area windows default to
    // INT_MAX, so this exercises underflow / leftover-get-area drain rather than
    // an INT_MAX split. Multi-record coverage is BoundedStreamMatchesReadChunkForFramedPayloads.
    TEST(ProjectContainerReader, BoundedStreamChunkedReadsOnFramedPayload) {
        const auto payload = patterned_payload(4096);
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "framed-chunked.licht";
        write_framed_fixture(path, FOURCC_SPLT, 851, payload, Compression::ZstdFramed,
                             true, false, 851);
        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo& chunk = reader.chunks().front();
        auto bounded = reader.open_bounded_stream(chunk);
        ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
        auto& stream = bounded->stream();
        std::vector<std::byte> got;
        got.reserve(payload.size());
        while (stream) {
            std::array<char, 7> buf{};
            stream.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            const auto n = stream.gcount();
            if (n <= 0) {
                break;
            }
            const auto* begin = reinterpret_cast<const std::byte*>(buf.data());
            got.insert(got.end(), begin, begin + static_cast<std::size_t>(n));
        }
        EXPECT_EQ(got, payload);
    }

    TEST(ProjectContainerReader, BoundedStreamReadsPayloadClassBeyondMaterializeCap) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "cap-stream.licht";
        const auto payload = patterned_payload(8192);
        write_framed_fixture(path, FOURCC_CKPT, 860, payload,
                             Compression::ZstdFramed, true, true, 860);
        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo& chunk = reader.chunks().front();
        ASSERT_GT(chunk.uncompressed_bytes, 64u);
        const PayloadCapGuard cap(chunk.uncompressed_bytes - 1);
        auto materialized = reader.read_chunk(chunk);
        ASSERT_FALSE(materialized);
        EXPECT_EQ(materialized.error().code(), lfs::ErrorCode::ResourceExhausted);
        const auto detail = lfs::format_for_developer(materialized.error());
        EXPECT_NE(detail.find("materialized-read maximum"), std::string::npos)
            << detail;
        auto bounded = reader.open_bounded_stream(chunk);
        ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
        std::vector<std::byte> got(static_cast<std::size_t>(bounded->size()));
        bounded->stream().read(reinterpret_cast<char*>(got.data()),
                               static_cast<std::streamsize>(got.size()));
        ASSERT_EQ(bounded->stream().gcount(),
                  static_cast<std::streamsize>(got.size()));
        EXPECT_EQ(got, payload);
    }

    TEST(ProjectContainerReader,
         BoundedStreamReadsNoTableFramedPayloadBeyondMaterializeCap) {
        const auto check = [](const Compression compression,
                              const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-no-table-cap.licht");
            const auto payload = patterned_payload(64 * 1024);
            write_framed_fixture(path, FOURCC_CKPT, 1200, payload, compression,
                                 true, false, 1200);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            EXPECT_EQ(chunk.compression, compression) << name;
            ASSERT_FALSE(chunk.block_crc_table.has_value()) << name;
            ASSERT_LT(chunk.stored_bytes, BLOCK_CRC_REQUIRED_AT) << name;
            ASSERT_GT(chunk.uncompressed_bytes, 64u) << name;
            const PayloadCapGuard cap(chunk.uncompressed_bytes - 1);
            auto materialized = reader.read_chunk(chunk);
            ASSERT_FALSE(materialized) << name;
            EXPECT_EQ(materialized.error().code(),
                      lfs::ErrorCode::ResourceExhausted)
                << name;
            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            expect_bounded_stream_matches(*bounded, payload);
        };
        check(Compression::ZstdFramed, "zstd-framed");
        check(Compression::ByteShuffleZstdFramed, "byteshuffle-framed");
    }

    TEST(ProjectContainerReader, BoundedStreamRejectsNoTableFramedPayloadCrcMismatch) {
        const auto check = [](const Compression compression,
                              const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-no-table-crc.licht");
            const auto payload = patterned_payload(64 * 1024);
            write_framed_fixture(path, FOURCC_CKPT, 1201, payload, compression,
                                 true, false, 1201);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            ASSERT_FALSE(chunk.block_crc_table.has_value()) << name;
            ASSERT_GT(chunk.stored_bytes, 0u) << name;
            const std::uint64_t corrupt_offset =
                chunk.payload_offset + chunk.stored_bytes / 2;
            const auto original = read_file_range(path, corrupt_offset, 1);
            ASSERT_EQ(original.size(), 1u) << name;
            write_file_range(path, corrupt_offset,
                             std::array{original.front() ^ std::byte{0xff}});
            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_FALSE(bounded) << name;
            const auto formatted = lfs::format_for_developer(bounded.error());
            const auto field =
                std::format("payload[{}].crc32c", chunk.key_string());
            EXPECT_EQ(error_field_string(bounded.error(), "field"), field)
                << formatted;
            EXPECT_EQ(bounded.error().code(), lfs::ErrorCode::DataLoss) << name;
            EXPECT_NE(formatted.find(std::format("expected 0x{:08x}",
                                                 chunk.payload_crc32c)),
                      std::string::npos)
                << formatted;
            EXPECT_NE(formatted.find("got 0x"), std::string::npos) << formatted;
        };
        check(Compression::ZstdFramed, "zstd-framed");
        check(Compression::ByteShuffleZstdFramed, "byteshuffle-framed");
    }

    TEST(ProjectContainerReader,
         BoundedStreamOpenAccountsStoredPrePassOnlyWithoutBlockTable) {
        const auto measure_open = [](const bool with_table,
                                     const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-open-account.licht");
            const auto payload = patterned_payload(64 * 1024);
            write_framed_fixture(path, FOURCC_CKPT, 1202, payload,
                                 Compression::ZstdFramed, true, with_table,
                                 1202);
            auto payload_reads = std::make_shared<std::atomic_uint64_t>(0);
            ReaderOptions options;
            options.payload_bytes_read = payload_reads;
            ProjectReader reader =
                require_result(ProjectReader::open(path, options));
            const ChunkInfo& chunk = reader.chunks().front();
            EXPECT_EQ(chunk.block_crc_table.has_value(), with_table) << name;
            EXPECT_GT(chunk.stored_bytes, 0u) << name;
            EXPECT_EQ(payload_reads->load(), 0u) << name;
            auto bounded = reader.open_bounded_stream(chunk);
            EXPECT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            return std::pair{payload_reads->load(), chunk.stored_bytes};
        };

        const auto [with_table_reads, with_table_stored] =
            measure_open(true, "with-table");
        EXPECT_LT(with_table_reads, with_table_stored);
        EXPECT_EQ(with_table_reads, 0u);

        const auto [no_table_reads, no_table_stored] =
            measure_open(false, "no-table");
        EXPECT_EQ(no_table_reads, no_table_stored);
        EXPECT_GT(no_table_stored, 0u);
    }

    TEST(ProjectContainerReader, BoundedStreamRejectsCorruptFramedStoredBytes) {
        const auto write_and_open =
            [](const std::string_view name,
               const std::function<void(const ChunkInfo&, const fs::path&)>&
                   corrupt) {
                TemporaryDirectory temporary;
                const fs::path path =
                    temporary.path / (std::string(name) + "-corrupt.licht");
                const auto payload = patterned_payload(256 * 1024);
                write_framed_fixture(path, FOURCC_CKPT, 870, payload,
                                     Compression::ZstdFramed, true, true, 870);
                ProjectReader reader = require_result(ProjectReader::open(path));
                const ChunkInfo& chunk = reader.chunks().front();
                ASSERT_TRUE(chunk.block_crc_table.has_value());
                corrupt(chunk, path);
                auto bounded = reader.open_bounded_stream(chunk);
                ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
                std::vector<char> sink(64);
                bounded->stream().read(sink.data(),
                                       static_cast<std::streamsize>(sink.size()));
                EXPECT_TRUE(bounded->stream().fail()) << name;
                EXPECT_EQ(bounded->stream().gcount(), 0) << name;
            };
        write_and_open("framed-record-bytes",
                       [](const ChunkInfo& chunk, const fs::path& path) {
                           const auto header =
                               read_file_range(path, chunk.payload_offset, 16);
                           const auto record_count =
                               static_cast<std::uint32_t>(header[12]) |
                               (static_cast<std::uint32_t>(header[13]) << 8) |
                               (static_cast<std::uint32_t>(header[14]) << 16) |
                               (static_cast<std::uint32_t>(header[15]) << 24);
                           const auto table =
                               16u + static_cast<std::uint64_t>(record_count) * 16u;
                           write_file_range(
                               path, chunk.payload_offset + table + 8,
                               std::array{std::byte{0x7f}});
                       });
        write_and_open("framed-record-table",
                       [](const ChunkInfo& chunk, const fs::path& path) {
                           write_file_range(
                               path, chunk.payload_offset + 20,
                               std::array{std::byte{0xff}});
                       });
    }

    TEST(ProjectContainerReader, BoundedStreamBulkReadEquivalence) {
        constexpr std::size_t kBytes =
            2ull * 64ull * 1024ull * 1024ull + 32ull * 1024ull * 1024ull;
        const auto payload = patterned_payload(kBytes);
        const auto check = [&](const Compression compression,
                               const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-bulk.licht");
            write_framed_fixture(path, FOURCC_CKPT, 1100, payload, compression,
                                 true, true, 1100);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            EXPECT_EQ(chunk.compression, compression) << name;
            const auto records = read_framed_table(path, chunk);
            ASSERT_GE(records.size(), 3u) << name;
            const auto materialized = require_result(reader.read_chunk(chunk));
            ASSERT_EQ(materialized, payload) << name;

            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            auto& stream = bounded->stream();

            std::vector<std::byte> whole(payload.size());
            stream.read(reinterpret_cast<char*>(whole.data()),
                        static_cast<std::streamsize>(whole.size()));
            ASSERT_EQ(stream.gcount(),
                      static_cast<std::streamsize>(whole.size()))
                << name;
            EXPECT_EQ(whole, materialized) << name;
            EXPECT_EQ(static_cast<std::uint64_t>(stream.tellg()), payload.size())
                << name;

            struct Slice {
                std::uint64_t offset;
                std::uint64_t length;
            };
            const std::array<Slice, 6> slices{{
                {1, 17},
                {3, 64ull * 1024ull * 1024ull + 9},
                {7, 2ull * 64ull * 1024ull * 1024ull + 13},
                {64ull * 1024ull * 1024ull - 1, 32},
                {64ull * 1024ull * 1024ull + 3, 64ull * 1024ull * 1024ull + 5},
                {2ull * 64ull * 1024ull * 1024ull + 1, 1000},
            }};
            for (const auto& slice : slices) {
                ASSERT_LE(slice.offset + slice.length, payload.size()) << name;
                stream.clear();
                stream.seekg(static_cast<std::streamoff>(slice.offset));
                ASSERT_TRUE(stream) << name;
                std::vector<std::byte> got(static_cast<std::size_t>(slice.length));
                stream.read(reinterpret_cast<char*>(got.data()),
                            static_cast<std::streamsize>(got.size()));
                ASSERT_EQ(stream.gcount(),
                          static_cast<std::streamsize>(got.size()))
                    << name << " @" << slice.offset;
                EXPECT_TRUE(std::equal(
                    got.begin(), got.end(),
                    materialized.begin() +
                        static_cast<std::ptrdiff_t>(slice.offset)))
                    << name << " @" << slice.offset;
            }

            stream.clear();
            stream.seekg(5);
            ASSERT_TRUE(stream) << name;
            constexpr std::size_t kLarge1 = 64ull * 1024ull * 1024ull + 100;
            constexpr std::size_t kLarge2 = 64ull * 1024ull * 1024ull + 77;
            std::vector<std::byte> first(kLarge1);
            stream.read(reinterpret_cast<char*>(first.data()),
                        static_cast<std::streamsize>(first.size()));
            ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(kLarge1))
                << name;
            EXPECT_TRUE(std::equal(first.begin(), first.end(),
                                   materialized.begin() + 5))
                << name;
            const int mid = stream.get();
            ASSERT_NE(mid, std::char_traits<char>::eof()) << name;
            EXPECT_EQ(static_cast<unsigned char>(mid),
                      std::to_integer<unsigned char>(materialized[5 + kLarge1]))
                << name;
            std::vector<std::byte> second(kLarge2);
            stream.read(reinterpret_cast<char*>(second.data()),
                        static_cast<std::streamsize>(second.size()));
            ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(kLarge2))
                << name;
            EXPECT_TRUE(std::equal(
                second.begin(), second.end(),
                materialized.begin() +
                    static_cast<std::ptrdiff_t>(5 + kLarge1 + 1)))
                << name;

            stream.clear();
            stream.seekg(0);
            ASSERT_TRUE(stream) << name;
            std::vector<std::byte> extra(payload.size() + 32);
            stream.read(reinterpret_cast<char*>(extra.data()),
                        static_cast<std::streamsize>(extra.size()));
            ASSERT_EQ(stream.gcount(),
                      static_cast<std::streamsize>(payload.size()))
                << name;
            EXPECT_TRUE(std::equal(extra.begin(),
                                   extra.begin() +
                                       static_cast<std::ptrdiff_t>(payload.size()),
                                   materialized.begin()))
                << name;
            EXPECT_TRUE(stream.eof()) << name;
            EXPECT_FALSE(stream.bad()) << name;
            stream.clear();
            EXPECT_EQ(static_cast<std::uint64_t>(stream.tellg()), payload.size())
                << name;
            stream.seekg(0);
            ASSERT_TRUE(stream) << name;
            std::byte first_byte{};
            stream.read(reinterpret_cast<char*>(&first_byte), 1);
            ASSERT_EQ(stream.gcount(), 1) << name;
            EXPECT_EQ(first_byte, materialized.front()) << name;
        };
        check(Compression::ZstdFramed, "zstd-framed");
        check(Compression::ByteShuffleZstdFramed, "byteshuffle-framed");
    }

    std::vector<std::byte> incompressible_payload(const std::size_t bytes) {
        std::vector<std::byte> payload(bytes);
        std::uint64_t state = 0x9e3779b97f4a7c15ull;
        auto* words = reinterpret_cast<std::uint64_t*>(payload.data());
        const std::size_t n_words = bytes / sizeof(std::uint64_t);
        for (std::size_t index = 0; index < n_words; ++index) {
            state = state * 6364136223846793005ull + 1;
            words[index] = state;
        }
        for (std::size_t index = n_words * sizeof(std::uint64_t); index < bytes;
             ++index) {
            state = state * 6364136223846793005ull + 1;
            payload[index] = static_cast<std::byte>(state >> 56);
        }
        return payload;
    }

    TEST(ProjectContainerReader, BoundedStreamBulkReadStopsAtCorruptRecord) {
        constexpr std::size_t kBytes =
            2ull * 64ull * 1024ull * 1024ull + 32ull * 1024ull * 1024ull;
        const auto payload = incompressible_payload(kBytes);
        const auto check = [&](const Compression compression,
                               const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-bulk-corrupt.licht");
            write_framed_fixture(path, FOURCC_CKPT, 1110, payload, compression,
                                 true, true, 1110);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            const auto records = read_framed_table(path, chunk);
            ASSERT_GE(records.size(), 3u) << name;
            const auto& victim = records[2];
            ASSERT_GT(victim.stored_bytes, 8u) << name;
            const std::uint64_t corrupt_rel =
                victim.stored_offset + victim.stored_bytes - 1;
            write_file_range(path, chunk.payload_offset + corrupt_rel,
                             std::array{std::byte{0x7f}});
            const std::uint64_t block = corrupt_rel / BLOCK_CRC_BYTES;
            const std::uint64_t block_lo = block * BLOCK_CRC_BYTES;
            const std::uint64_t block_hi = block_lo + BLOCK_CRC_BYTES;
            std::size_t first_bad = records.size();
            for (std::size_t index = 0; index < records.size(); ++index) {
                const auto& record = records[index];
                if (record.stored_offset < block_hi &&
                    record.stored_offset + record.stored_bytes > block_lo) {
                    first_bad = index;
                    break;
                }
            }
            ASSERT_LT(first_bad, records.size()) << name;
            const bool byteshuffle =
                compression == Compression::ByteShuffleZstdFramed;
            const auto prefix = first_logical_byte_of_record(
                records[first_bad], chunk.uncompressed_bytes, byteshuffle);
            ASSERT_GT(prefix, 0u) << name;
            ASSERT_LT(prefix, payload.size()) << name;

            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            auto& stream = bounded->stream();
            std::vector<std::byte> got(payload.size(), std::byte{0x3c});
            stream.read(reinterpret_cast<char*>(got.data()),
                        static_cast<std::streamsize>(got.size()));
            EXPECT_EQ(stream.gcount(), static_cast<std::streamsize>(prefix))
                << name;
            EXPECT_TRUE(stream.fail()) << name;
            EXPECT_TRUE(std::equal(got.begin(),
                                   got.begin() + static_cast<std::ptrdiff_t>(prefix),
                                   payload.begin()))
                << name;

            stream.clear();
            stream.seekg(0);
            std::byte probe{};
            stream.read(reinterpret_cast<char*>(&probe), 1);
            EXPECT_EQ(stream.gcount(), 0) << name;
            EXPECT_TRUE(stream.fail()) << name;
        };
        check(Compression::ZstdFramed, "zstd-framed");
        check(Compression::ByteShuffleZstdFramed, "byteshuffle-framed");
    }

    TEST(ProjectContainerReader, BoundedStreamSmallReadsDoNotRedecodeFramedRecords) {
        constexpr std::size_t kBytes =
            2ull * 64ull * 1024ull * 1024ull + 32ull * 1024ull * 1024ull;
        const auto payload = patterned_payload(kBytes);
        const auto check = [&](const Compression compression,
                               const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-small-reads.licht");
            write_framed_fixture(path, FOURCC_CKPT, 1130, payload, compression,
                                 true, true, 1130);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            EXPECT_EQ(chunk.compression, compression) << name;
            const auto records = read_framed_table(path, chunk);
            ASSERT_GE(records.size(), 3u) << name;
            const auto materialized = require_result(reader.read_chunk(chunk));
            ASSERT_EQ(materialized, payload) << name;

            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            std::streambuf* buf = bounded->stream().rdbuf();
            ASSERT_NE(buf, nullptr) << name;
            const auto step = std::max<std::uint64_t>(
                records.front().decoded_bytes / 4, 1);
            detail::reset_framed_record_decode_calls_for_testing();
            std::vector<std::byte> got(payload.size());
            std::size_t filled = 0;
            while (filled < got.size()) {
                const auto want = std::min<std::size_t>(
                    static_cast<std::size_t>(step), got.size() - filled);
                const auto n = buf->sgetn(
                    reinterpret_cast<char*>(got.data() + filled),
                    static_cast<std::streamsize>(want));
                ASSERT_GT(n, 0) << name << " @" << filled;
                filled += static_cast<std::size_t>(n);
            }
            EXPECT_EQ(got, materialized) << name;
            EXPECT_EQ(detail::framed_record_decode_calls_for_testing(),
                      records.size())
                << name;
        };
        check(Compression::ZstdFramed, "zstd-framed");
        check(Compression::ByteShuffleZstdFramed, "byteshuffle-framed");
    }

    TEST(ProjectContainerReader, DISABLED_FramedStreamThroughputBenchmark) {
        constexpr std::size_t kBytes = 24ull * 64ull * 1024ull * 1024ull;
        std::vector<std::byte> payload;
        try {
            payload = compressible_payload(kBytes);
        } catch (const std::bad_alloc&) {
            GTEST_SKIP() << "not enough memory for 1.5GiB framed stream benchmark";
        }

        const auto run = [&](const Compression compression,
                             const std::string_view name) {
            TemporaryDirectory temporary;
            const fs::path path =
                temporary.path / (std::string(name) + "-bench.licht");
            write_framed_fixture(path, FOURCC_CKPT, 1120, payload, compression,
                                 true, true, 1120);
            ProjectReader reader = require_result(ProjectReader::open(path));
            const ChunkInfo& chunk = reader.chunks().front();
            EXPECT_EQ(chunk.compression, compression) << name;
            const auto records = read_framed_table(path, chunk);
            ASSERT_GE(records.size(), 3u) << name;

            const auto started_chunk = std::chrono::steady_clock::now();
            const auto materialized = require_result(reader.read_chunk(chunk));
            const auto chunk_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() -
                                      started_chunk)
                                      .count();
            ASSERT_EQ(materialized, payload) << name;

            auto bounded = reader.open_bounded_stream(chunk);
            ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
            std::vector<std::byte> streamed;
            try {
                streamed.resize(payload.size());
            } catch (const std::bad_alloc&) {
                GTEST_SKIP() << "not enough memory for streamed 1.5GiB buffer";
            }
            const auto started_stream = std::chrono::steady_clock::now();
            bounded->stream().read(reinterpret_cast<char*>(streamed.data()),
                                   static_cast<std::streamsize>(streamed.size()));
            const auto stream_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() -
                                       started_stream)
                                       .count();
            ASSERT_EQ(bounded->stream().gcount(),
                      static_cast<std::streamsize>(streamed.size()))
                << name;
            ASSERT_EQ(streamed, payload) << name;

            const double megabytes = static_cast<double>(kBytes) / (1024.0 * 1024.0);
            const auto mbps = [&](const double milliseconds) {
                return milliseconds <= 0.0 ? 0.0
                                           : megabytes / (milliseconds / 1000.0);
            };
            std::cout << "FramedStreamThroughputBenchmark " << name
                      << " read_chunk=" << mbps(chunk_ms) << " MB/s ("
                      << chunk_ms << " ms) stream=" << mbps(stream_ms)
                      << " MB/s (" << stream_ms << " ms)\n";
        };
        run(Compression::ByteShuffleZstdFramed, "byteshuffle-framed");
        run(Compression::ZstdFramed, "zstd-framed");
    }

    TEST(ProjectContainerWriter, CleanProofRejectsForcedFalseCleanAndReaderStaysPinned) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "clean-proof.licht";
        create_single_chunk_fixture(
            path, 401, 501, 601, fixed_key("PROJ", 701),
            R"({"state":"before"})");

        ProjectReader pinned =
            require_result(ProjectReader::open(path));
        const ChunkInfo* old_row =
            pinned.find(fixed_key("PROJ", 701));
        ASSERT_NE(old_row, nullptr);
        const std::uint64_t old_payload_offset = old_row->payload_offset;
        CleanProof proof =
            require_result(pinned.make_clean_proof(*old_row, 41));

        ProjectWriter writer = require_result(
            ProjectWriter::append(path, fixture_append_options()));
        const auto changed = byte_vector(R"({"state":"after"})");
        require_status(
            writer.plan_commit(fixture_commit_options(502, 602, 2)));
        require_status(writer.preflight(changed.size()));
        auto false_clean = writer.reuse_if_clean(proof, 42);
        ASSERT_FALSE(false_clean);
        EXPECT_EQ(false_clean.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        auto premature_commit = writer.commit();
        ASSERT_FALSE(premature_commit);
        EXPECT_EQ(premature_commit.error().code(),
                  lfs::ErrorCode::FailedPrecondition);

        require_status(
            writer.write_chunk(fixed_key("PROJ", 701), changed));
        require_status(writer.commit());

        EXPECT_EQ(pinned.commit().generation, 1u);
        const auto pinned_payload =
            require_result(pinned.read_chunk(*old_row));
        EXPECT_EQ(pinned_payload, byte_vector(R"({"state":"before"})"));

        ProjectReader current =
            require_result(ProjectReader::open(path));
        EXPECT_EQ(current.commit().generation, 2u);
        const ChunkInfo* new_row =
            current.find(fixed_key("PROJ", 701));
        ASSERT_NE(new_row, nullptr);
        EXPECT_NE(new_row->payload_offset, old_payload_offset);
        EXPECT_EQ(require_result(current.read_chunk(*new_row)), changed);
    }

    TEST(ProjectContainerWriter, DirtyCheckpointEscalatesMetadataOnlySave) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "dirty-ckpt.licht";
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path, fixture_create_options(402)));
            const auto checkpoint = byte_vector("checkpoint-generation-one");
            const auto metadata = byte_vector(R"({"metadata":1})");
            require_status(writer.plan_commit(
                fixture_commit_options(503, 603, 1)));
            require_status(writer.preflight(checkpoint.size() +
                                            metadata.size()));
            ChunkWriteOptions tensor{
                .chunk_version = 1,
                .compression = Compression::Stored,
                .tensor_payload = true,
            };
            require_status(writer.write_chunk(fixed_key("CKPT", 702),
                                              checkpoint, tensor));
            require_status(writer.write_chunk(fixed_key("PROJ", 703),
                                              metadata));
            require_status(writer.commit());
        }

        ProjectReader prior =
            require_result(ProjectReader::open(path));
        const ChunkInfo* metadata_row =
            prior.find(fixed_key("PROJ", 703));
        ASSERT_NE(metadata_row, nullptr);
        CleanProof metadata_proof =
            require_result(prior.make_clean_proof(*metadata_row, 9));
        ProjectWriter writer = require_result(
            ProjectWriter::append(path, fixture_append_options()));
        const auto checkpoint = byte_vector("checkpoint-generation-two");
        require_status(
            writer.plan_commit(fixture_commit_options(504, 604, 2)));
        require_status(writer.preflight(checkpoint.size()));
        require_status(writer.reuse_if_clean(metadata_proof, 9));
        auto metadata_only = writer.commit();
        ASSERT_FALSE(metadata_only);
        EXPECT_EQ(metadata_only.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        EXPECT_NE(lfs::format_for_developer(metadata_only.error())
                      .find("dirty_ckpt"),
                  std::string::npos);

        ChunkWriteOptions tensor{
            .chunk_version = 1,
            .compression = Compression::Stored,
            .tensor_payload = true,
        };
        require_status(writer.write_chunk(fixed_key("CKPT", 702),
                                          checkpoint, tensor));
        require_status(writer.commit());
        ProjectReader current =
            require_result(ProjectReader::open(path));
        EXPECT_EQ(current.commit().generation, 2u);
    }

    TEST(ProjectContainerWriter,
         ProductionCapabilitiesDescribeOnlyOperationsUsedByGeneration) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "capabilities.licht";
        {
            CreateOptions create = fixture_create_options(820);
            create.index_compression = IndexCompression::Zstd;
            ProjectWriter writer =
                require_result(ProjectWriter::create(path, create));
            const auto payload = byte_vector("known project metadata");
            require_status(writer.plan_commit(
                fixture_commit_options(821, 822, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(
                writer.write_chunk(fixed_key("X999", 823), payload));
            require_status(writer.commit());
        }
        ProjectReader first = require_result(ProjectReader::open(path));
        EXPECT_FALSE(first.commit().required_writer_capabilities.contains(
            OPAQUE_CHUNK_PRESERVATION));
        EXPECT_FALSE(first.commit().required_writer_capabilities.contains(
            CLEAN_PROOF_REUSE));
        const ChunkInfo* row = first.find(fixed_key("X999", 823));
        ASSERT_NE(row, nullptr);
        CleanProof proof =
            require_result(first.make_clean_proof(*row, 41));

        {
            AppendOptions append;
            append.index_compression = IndexCompression::Zstd;
            append.disk_reserve_bytes = 0;
            ProjectWriter writer =
                require_result(ProjectWriter::append(path, append));
            require_status(writer.plan_commit(
                fixture_commit_options(824, 825, 2)));
            require_status(writer.preflight(0));
            require_status(writer.reuse_if_clean(proof, 41));
            require_status(writer.commit());
        }
        ProjectReader second = require_result(ProjectReader::open(path));
        EXPECT_FALSE(second.commit().required_writer_capabilities.contains(
            OPAQUE_CHUNK_PRESERVATION));
        EXPECT_TRUE(second.commit().required_writer_capabilities.contains(
            CLEAN_PROOF_REUSE));
        const ChunkInfo* carried_row =
            second.find(fixed_key("X999", 823));
        ASSERT_NE(carried_row, nullptr);
        CleanProof opaque_proof =
            require_result(second.make_clean_proof(*carried_row, 42));
        {
            AppendOptions append;
            append.index_compression = IndexCompression::Zstd;
            append.disk_reserve_bytes = 0;
            ProjectWriter writer =
                require_result(ProjectWriter::append(path, append));
            require_status(writer.plan_commit(
                fixture_commit_options(826, 827, 3)));
            require_status(writer.preflight(0));
            require_status(writer.carry_forward_opaque(
                *carried_row, opaque_proof, 42));
            require_status(writer.commit());
        }
        ProjectReader third = require_result(ProjectReader::open(path));
        EXPECT_TRUE(third.commit().required_writer_capabilities.contains(
            OPAQUE_CHUNK_PRESERVATION));
        EXPECT_TRUE(third.commit().required_writer_capabilities.contains(
            CLEAN_PROOF_REUSE));
    }

    TEST(ProjectContainerWriter,
         AutosaveCreateReplacesTornAndWrongMagicStableSidecars) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "disposable-sidecar.licht";
        const fs::path sidecar = autosave_sidecar_path(master);
        create_single_chunk_fixture(
            master, 840, 841, 842, fixed_key("PROJ", 843),
            R"({"master":"unchanged"})");
        const auto master_before = read_file_bytes(master);

        publish_complete_sidecar(master, sidecar, 1, 844, 845, 846);
        fs::resize_file(sidecar, SUPERBLOCK_BYTES / 2);
        ASSERT_FALSE(ProjectReader::open(sidecar));
        publish_complete_sidecar(master, sidecar, 2, 847, 848, 849);
        {
            ProjectReader replacement =
                require_result(ProjectReader::open(sidecar));
            EXPECT_EQ(replacement.superblock().autosave_sequence, 2u);
            require_status(replacement.verify_all());
        }

        auto wrong_magic = read_file_bytes(sidecar);
        ASSERT_GE(wrong_magic.size(), 8u);
        wrong_magic[0] ^= std::byte{0xff};
        write_file_bytes(sidecar, wrong_magic);
        ASSERT_FALSE(ProjectReader::open(sidecar));
        publish_complete_sidecar(master, sidecar, 3, 850, 851, 852);
        ProjectReader replacement =
            require_result(ProjectReader::open(sidecar));
        EXPECT_EQ(replacement.superblock().autosave_sequence, 3u);
        require_status(replacement.verify_all());
        EXPECT_EQ(read_file_bytes(master), master_before);
    }

    TEST(ProjectContainerWriter,
         AutosaveCreateReplacesValidSidecarWithFutureWriteRequirements) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "future-sidecar.licht";
        const fs::path sidecar = autosave_sidecar_path(master);
        create_single_chunk_fixture(
            master, 853, 854, 855, fixed_key("PROJ", 856),
            R"({"master":"future-sidecar-base"})");

        CommitOptions future = fixture_commit_options(857, 858, 1);
        future.min_reader_version = Version{1, 1};
        future.min_safe_writer_version = Version{1, 1};
        future.extra_reader_capabilities.set(100);
        future.extra_writer_capabilities.set(101);
        publish_complete_sidecar(master, sidecar, 1, 859, 857, 858,
                                 future);
        const auto classification = ProjectReader::classify(sidecar);
        EXPECT_EQ(classification.state, OpenState::UnsupportedNewer);

        publish_complete_sidecar(master, sidecar, 2, 860, 861, 862);
        ProjectReader replacement =
            require_result(ProjectReader::open(sidecar));
        EXPECT_EQ(replacement.open_state(), OpenState::Open);
        EXPECT_EQ(replacement.superblock().autosave_sequence, 2u);
        require_status(replacement.verify_all());
    }

    TEST(ProjectContainerWriter,
         BackupCleanupFailureAfterValidatedReplaceReportsSuccessAndNote) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "cleanup-note.licht";
        create_single_chunk_fixture(
            path, 865, 866, 867, fixed_key("PROJ", 868),
            R"({"generation":"old"})");
        bool backup_removed = false;
        auto options = fixture_create_options(869);
        options.boundary_observer = [&](const CommitBoundary boundary) {
            if (boundary != CommitBoundary::ReplacementValidated) {
                return;
            }
            for (const auto& entry :
                 fs::directory_iterator(temporary.path)) {
                const auto name = entry.path().filename().string();
                if (name.starts_with(path.stem().string()) &&
                    name.find(".replace-backup.") != std::string::npos) {
                    backup_removed = fs::remove(entry.path());
                }
            }
        };
        ProjectWriter writer =
            require_result(ProjectWriter::create(path, options));
        const auto payload = byte_vector(R"({"generation":"new"})");
        require_status(
            writer.plan_commit(fixture_commit_options(870, 871, 1)));
        require_status(writer.preflight(payload.size()));
        require_status(
            writer.write_chunk(fixed_key("PROJ", 872), payload));
        auto published = writer.commit();
        ASSERT_TRUE(published)
            << lfs::format_for_developer(published.error());
        EXPECT_TRUE(backup_removed);
        ASSERT_TRUE(writer.post_publish_note().has_value());
        EXPECT_NE(lfs::format_for_developer(*writer.post_publish_note())
                      .find("commit.post_publish_verification"),
                  std::string::npos);

        ProjectReader reopened = require_result(ProjectReader::open(path));
        EXPECT_EQ(reopened.commit().commit_uuid, fixed_uuid(870));
        const ChunkInfo* row = reopened.find(fixed_key("PROJ", 872));
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(require_result(reopened.read_chunk(*row)), payload);
    }

    TEST(ProjectContainerWriter,
         StorageStatsMatchesCompactedKnownLayoutWithinTwoPercent) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "storage-stats.licht";
        const ChunkKey key = fixed_key("SPLT", 873);
        const std::vector<std::byte> old_payload(
            9 * 1024 * 1024 + 137, std::byte{0x31});
        const std::vector<std::byte> live_payload(
            7 * 1024 * 1024 + 79, std::byte{0x52});
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(path, fixture_create_options(874)));
            require_status(
                writer.plan_commit(fixture_commit_options(875, 876, 1)));
            require_status(writer.preflight(old_payload.size()));
            require_status(writer.write_chunk(
                key, old_payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::Stored,
                    .tensor_payload = true,
                    .block_crcs = true,
                }));
            require_status(writer.commit());
        }
        {
            ProjectWriter writer = require_result(
                ProjectWriter::append(path, fixture_append_options()));
            require_status(
                writer.plan_commit(fixture_commit_options(877, 878, 2)));
            require_status(writer.preflight(live_payload.size()));
            require_status(writer.write_chunk(
                key, live_payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::Stored,
                    .tensor_payload = true,
                    .block_crcs = true,
                }));
            require_status(writer.commit());
        }

        const ProjectStorageStats stats =
            require_result(project_storage_stats(path));
        {
            const auto reader =
                require_result(ProjectReader::open(path));
            const auto from_reader =
                require_result(project_storage_stats(reader));
            EXPECT_EQ(
                from_reader.physical_bytes,
                stats.physical_bytes);
            EXPECT_EQ(
                from_reader.estimated_live_bytes,
                stats.estimated_live_bytes);
            EXPECT_EQ(from_reader.dead_bytes, stats.dead_bytes);
            EXPECT_DOUBLE_EQ(
                from_reader.dead_ratio, stats.dead_ratio);
        }
        const auto physical_before = fs::file_size(path);
        require_status(ProjectWriter::compact(
            path,
            CompactionOptions{
                .new_file_uuid = fixed_uuid(879),
                .commit_uuid = fixed_uuid(880),
                .snapshot_uuid = fixed_uuid(881),
                .creation_time_unix_ns = FIXED_CREATION_TIME_NS + 40,
                .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 40,
                .disk_reserve_bytes = 0,
            }));
        const auto compacted_bytes = fs::file_size(path);
        const double compacted_oracle =
            static_cast<double>(physical_before - compacted_bytes) /
            static_cast<double>(physical_before);
        std::cout << std::format(
            "P7_STORAGE_STATS_KNOWN physical_bytes={} compacted_bytes={} "
            "estimated_live_bytes={} dead_bytes={} stats_ratio={:.6f} "
            "oracle_ratio={:.6f} absolute_error={:.6f}\n",
            physical_before, compacted_bytes, stats.estimated_live_bytes,
            stats.dead_bytes, stats.dead_ratio, compacted_oracle,
            std::abs(stats.dead_ratio - compacted_oracle));
        EXPECT_LT(std::abs(stats.dead_ratio - compacted_oracle), 0.02)
            << "stats=" << stats.dead_ratio
            << " compacted_oracle=" << compacted_oracle;
    }

    TEST(ProjectContainerWriter,
         StorageStatsCrossesSuggestionThresholdAtGenuineHalfDeadLayout) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "half-dead.licht";
        constexpr std::size_t ROWS = 8;
        const std::vector<std::byte> old_payload(
            4 * 1024 * 1024 + 512 * 1024,
            std::byte{0x63});
        const std::vector<std::byte> live_payload(
            4 * 1024 * 1024, std::byte{0x74});
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(path, fixture_create_options(882)));
            require_status(
                writer.plan_commit(fixture_commit_options(883, 884, 1)));
            require_status(writer.preflight(ROWS * old_payload.size()));
            for (std::size_t index = 0; index < ROWS; ++index) {
                require_status(writer.write_chunk(
                    fixed_key("SPLT", 10'000 + index), old_payload,
                    ChunkWriteOptions{
                        .chunk_version = 1,
                        .compression = Compression::Stored,
                        .tensor_payload = true,
                        .block_crcs = true,
                    }));
            }
            require_status(writer.commit());
        }
        {
            ProjectWriter writer = require_result(
                ProjectWriter::append(path, fixture_append_options()));
            require_status(
                writer.plan_commit(fixture_commit_options(885, 886, 2)));
            require_status(writer.preflight(ROWS * live_payload.size()));
            for (std::size_t index = 0; index < ROWS; ++index) {
                require_status(writer.write_chunk(
                    fixed_key("SPLT", 10'000 + index), live_payload,
                    ChunkWriteOptions{
                        .chunk_version = 1,
                        .compression = Compression::Stored,
                        .tensor_payload = true,
                        .block_crcs = true,
                    }));
            }
            require_status(writer.commit());
        }

        const ProjectStorageStats stats =
            require_result(project_storage_stats(path));
        const auto physical_before = fs::file_size(path);
        require_status(ProjectWriter::compact(
            path,
            CompactionOptions{
                .new_file_uuid = fixed_uuid(887),
                .commit_uuid = fixed_uuid(888),
                .snapshot_uuid = fixed_uuid(889),
                .creation_time_unix_ns = FIXED_CREATION_TIME_NS + 41,
                .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 41,
                .disk_reserve_bytes = 0,
            }));
        const auto compacted_bytes = fs::file_size(path);
        const double compacted_oracle =
            static_cast<double>(physical_before - compacted_bytes) /
            static_cast<double>(physical_before);
        std::cout << std::format(
            "P7_STORAGE_STATS_THRESHOLD physical_bytes={} compacted_bytes={} "
            "estimated_live_bytes={} dead_bytes={} stats_ratio={:.6f} "
            "oracle_ratio={:.6f} absolute_error={:.6f}\n",
            physical_before, compacted_bytes, stats.estimated_live_bytes,
            stats.dead_bytes, stats.dead_ratio, compacted_oracle,
            std::abs(stats.dead_ratio - compacted_oracle));
        EXPECT_GE(compacted_oracle, 0.50);
        EXPECT_LE(compacted_oracle, 0.54);
        EXPECT_LT(std::abs(stats.dead_ratio - compacted_oracle), 0.02);
        EXPECT_GE(stats.dead_ratio, 0.50)
            << "the lifecycle's 50% compaction suggestion must fire";
    }

    TEST(ProjectContainerWriter,
         CompactionRefusesCurrentBoundSidecarUntilExplicitHeadAdvances) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "compact-bound.licht";
        const fs::path sidecar = autosave_sidecar_path(master);
        const ChunkKey key = fixed_key("PROJ", 890);
        create_single_chunk_fixture(
            master, 891, 892, 893, key, R"({"generation":1})");
        publish_complete_sidecar(master, sidecar, 1, 894, 895, 896);

        const CompactionOptions first_options{
            .new_file_uuid = fixed_uuid(897),
            .commit_uuid = fixed_uuid(898),
            .snapshot_uuid = fixed_uuid(899),
            .creation_time_unix_ns = FIXED_CREATION_TIME_NS + 42,
            .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 42,
            .disk_reserve_bytes = 0,
        };
        auto refused = ProjectWriter::compact(master, first_options);
        ASSERT_FALSE(refused);
        EXPECT_EQ(refused.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        EXPECT_NE(lfs::format_for_developer(refused.error())
                      .find("compaction.autosave_binding"),
                  std::string::npos);
        EXPECT_TRUE(fs::exists(sidecar));

        {
            ProjectWriter writer = require_result(
                ProjectWriter::append(master, fixture_append_options()));
            const auto payload = byte_vector(R"({"generation":2})");
            require_status(
                writer.plan_commit(fixture_commit_options(900, 901, 2)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(key, payload));
            require_status(writer.commit());
        }
        auto stale = inspect_autosave_recovery(master);
        ASSERT_TRUE(stale)
            << lfs::format_for_developer(stale.error());
        EXPECT_EQ(stale->disposition, RecoveryDisposition::StaleDeleted);
        EXPECT_FALSE(fs::exists(sidecar));

        require_status(ProjectWriter::compact(
            master,
            CompactionOptions{
                .new_file_uuid = fixed_uuid(902),
                .commit_uuid = fixed_uuid(903),
                .snapshot_uuid = fixed_uuid(904),
                .creation_time_unix_ns = FIXED_CREATION_TIME_NS + 43,
                .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 43,
                .disk_reserve_bytes = 0,
            }));
        ProjectReader compacted =
            require_result(ProjectReader::open(master));
        EXPECT_EQ(compacted.commit().kind, CommitKind::Compaction);
    }

    TEST(ProjectContainerWriter,
         RecoveryInspectionOfMultiGigabyteShapedMasterReadsNoPayload) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "open-cost.licht";
        constexpr std::size_t PAYLOAD_BYTES = 8 * 1024 * 1024;
        const std::vector<std::byte> payload(PAYLOAD_BYTES,
                                             std::byte{0x45});
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(master, fixture_create_options(905)));
            require_status(
                writer.plan_commit(fixture_commit_options(906, 907, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(
                fixed_key("SPLT", 908), payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::Stored,
                    .tensor_payload = true,
                    .block_crcs = true,
                }));
            require_status(writer.commit());
        }

        ProjectReader seed = require_result(ProjectReader::open(master));
        ASSERT_EQ(seed.chunks().size(), 1u);
        const ChunkInfo& seed_row = seed.chunks().front();
        ASSERT_TRUE(seed_row.block_crc_table.has_value());

        constexpr std::uint64_t SHAPED_PAYLOAD_BYTES =
            3ull * 1024 * 1024 * 1024 + 17;
        const std::uint64_t block_count =
            SHAPED_PAYLOAD_BYTES / BLOCK_CRC_BYTES +
            (SHAPED_PAYLOAD_BYTES % BLOCK_CRC_BYTES != 0 ? 1 : 0);
        ASSERT_LT(block_count * sizeof(std::uint32_t),
                  seed_row.payload_offset -
                      (seed_row.block_crc_table->offset +
                       BLOCK_CRC_HEADER_BYTES));

        auto chunk_header =
            read_file_range(master, seed_row.header_offset,
                            CHUNK_HEADER_BYTES);
        write_u64_le(chunk_header, 32, SHAPED_PAYLOAD_BYTES);
        write_u64_le(chunk_header, 40, SHAPED_PAYLOAD_BYTES);
        const std::uint32_t chunk_header_crc =
            crc_range(chunk_header, 0, 60);
        write_u32_le(chunk_header, 60, chunk_header_crc);

        auto block_header = read_file_range(
            master, seed_row.block_crc_table->offset,
            BLOCK_CRC_HEADER_BYTES);
        std::vector<std::byte> block_entries(
            static_cast<std::size_t>(block_count * sizeof(std::uint32_t)));
        const std::vector<std::byte> zero_block(
            static_cast<std::size_t>(BLOCK_CRC_BYTES));
        const std::vector<std::byte> zero_tail(static_cast<std::size_t>(
            SHAPED_PAYLOAD_BYTES % BLOCK_CRC_BYTES));
        const std::uint32_t zero_block_crc =
            crc32c(0, zero_block.data(), zero_block.size());
        for (std::uint64_t block_index = 0; block_index < block_count;
             ++block_index) {
            std::uint32_t block_crc = zero_block_crc;
            if (block_index < PAYLOAD_BYTES / BLOCK_CRC_BYTES) {
                block_crc = crc32c(
                    0,
                    payload.data() + block_index * BLOCK_CRC_BYTES,
                    static_cast<std::size_t>(BLOCK_CRC_BYTES));
            } else if (block_index + 1 == block_count &&
                       !zero_tail.empty()) {
                block_crc =
                    crc32c(0, zero_tail.data(), zero_tail.size());
            }
            write_u32_le(block_entries,
                         static_cast<std::size_t>(block_index) *
                             sizeof(std::uint32_t),
                         block_crc);
        }
        write_u64_le(block_header, 32, SHAPED_PAYLOAD_BYTES);
        write_u64_le(block_header, 40, block_count);
        write_u32_le(block_header, 48,
                     crc32c(0, block_entries.data(), block_entries.size()));
        write_u32_le(block_header, 60, crc_range(block_header, 0, 60));

        auto index = read_file_range(
            master, seed.commit().index_offset,
            static_cast<std::size_t>(seed.commit().index_stored_bytes));
        constexpr std::size_t ROW_OFFSET = INDEX_HEADER_BYTES;
        write_u64_le(index, ROW_OFFSET + 48, SHAPED_PAYLOAD_BYTES);
        write_u64_le(index, ROW_OFFSET + 56, SHAPED_PAYLOAD_BYTES);
        write_u32_le(index, ROW_OFFSET + 76, chunk_header_crc);

        const auto align_chunk = [](const std::uint64_t value) {
            return (value + CHUNK_ALIGNMENT - 1) &
                   ~(CHUNK_ALIGNMENT - 1);
        };
        const std::uint64_t shaped_index_offset = align_chunk(
            seed_row.payload_offset + SHAPED_PAYLOAD_BYTES);
        const std::uint64_t shaped_commit_offset = align_chunk(
            shaped_index_offset + index.size());
        const std::uint64_t shaped_file_end =
            shaped_commit_offset + COMMIT_RECORD_BYTES;

        auto commit = read_file_range(master, seed.commit().offset,
                                      COMMIT_RECORD_BYTES);
        write_u64_le(commit, 136, shaped_index_offset);
        const std::uint32_t index_crc =
            crc32c(0, index.data(), index.size());
        write_u32_le(commit, 160, index_crc);
        write_u32_le(commit, 164, index_crc);
        write_u64_le(commit, 176, shaped_file_end);
        const std::uint32_t commit_crc = crc_range(commit, 0, 252);
        write_u32_le(commit, 252, commit_crc);

        auto head = read_file_range(
            master, HEAD_SLOT_OFFSETS[seed.selected_head().slot_id],
            HEAD_SLOT_BYTES);
        write_u64_le(head, 80, shaped_commit_offset);
        write_u64_le(head, 96, shaped_file_end);
        write_u32_le(head, 104, commit_crc);
        write_u32_le(head, 4092, crc_range(head, 0, 4092));

        fs::resize_file(master, shaped_file_end);
        write_file_range(master, seed_row.header_offset, chunk_header);
        write_file_range(master, seed_row.block_crc_table->offset,
                         block_header);
        write_file_range(
            master,
            seed_row.block_crc_table->offset + BLOCK_CRC_HEADER_BYTES,
            block_entries);
        write_file_range(master, shaped_index_offset, index);
        write_file_range(master, shaped_commit_offset, commit);
        write_file_range(
            master, HEAD_SLOT_OFFSETS[seed.selected_head().slot_id], head);

        auto payload_reads = std::make_shared<std::atomic_uint64_t>(0);
        ReaderOptions options;
        options.payload_bytes_read = payload_reads;
        auto inspection = inspect_autosave_recovery(master, options);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_EQ(inspection->disposition, RecoveryDisposition::None);
        EXPECT_EQ(payload_reads->load(), 0u);

        ProjectReader instrumented =
            require_result(ProjectReader::open(master, options));
        ASSERT_EQ(instrumented.chunks().size(), 1u);
        EXPECT_EQ(instrumented.chunks().front().stored_bytes,
                  SHAPED_PAYLOAD_BYTES);
        EXPECT_EQ(instrumented.commit().committed_file_end,
                  shaped_file_end);
        std::array<std::byte, 1> probe{};
        require_status(instrumented.read_stored_at(
            instrumented.chunks().front(), SHAPED_PAYLOAD_BYTES - 1,
            probe));
        EXPECT_EQ(payload_reads->load(), probe.size());
    }

    TEST(ProjectContainerWriter,
         RecoverySessionTempHygienePreservesLiveLockedSession) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "recovery-lock.licht";
        const fs::path sidecar = autosave_sidecar_path(master);
        create_single_chunk_fixture(
            master, 909, 910, 911, fixed_key("PROJ", 912),
            R"({"master":"recovery-lock"})");
        publish_complete_sidecar(master, sidecar, 1, 913, 914, 915);

        const fs::path orphan = recovery_session_temp_path(master);
        write_file_bytes(orphan, byte_vector("killed recovery session"));
        auto hygiene = inspect_autosave_recovery(master);
        ASSERT_TRUE(hygiene)
            << lfs::format_for_developer(hygiene.error());
        EXPECT_FALSE(fs::exists(orphan));
        EXPECT_NE(std::ranges::find(hygiene->deleted_paths, orphan),
                  hygiene->deleted_paths.end());

        RecoverySession session =
            require_result(begin_recovery_session(master, sidecar));
        const fs::path live_temp = recovery_session_temp_path(master);
        require_status(materialize_recovered_project(
            master, sidecar, live_temp, session));
        ASSERT_TRUE(fs::exists(live_temp));
        auto concurrent_inspection = inspect_autosave_recovery(master);
        ASSERT_FALSE(concurrent_inspection);
        EXPECT_EQ(concurrent_inspection.error().code(),
                  lfs::ErrorCode::Unavailable);
        EXPECT_TRUE(fs::exists(live_temp));
        auto concurrent_writer = ProjectWriter::append(master);
        ASSERT_FALSE(concurrent_writer);
        EXPECT_EQ(concurrent_writer.error().code(),
                  lfs::ErrorCode::Unavailable);

        {
            ProjectReader locked_base =
                require_result(ProjectReader::open(master));
            ASSERT_EQ(locked_base.chunks().size(), 1u);
            CleanProof proof = require_result(
                locked_base.make_clean_proof(
                    locked_base.chunks().front(), 1));
            AppendOptions append = fixture_append_options();
            append.writer_lock_lease = session.writer_lock();
            ProjectWriter writer = require_result(
                ProjectWriter::append(master, append));
            const auto payload = byte_vector("recovered merge marker");
            CommitOptions commit = fixture_commit_options(916, 917, 2);
            commit.kind = CommitKind::Recovered;
            require_status(writer.plan_commit(commit));
            require_status(writer.preflight(payload.size()));
            require_status(writer.reuse_if_clean(proof, 1));
            require_status(writer.write_chunk(
                fixed_key("VIEW", 918), payload));
            require_status(writer.commit());
        }
        require_status(session.release());
        EXPECT_FALSE(fs::exists(live_temp));
        auto stale = inspect_autosave_recovery(master);
        ASSERT_TRUE(stale)
            << lfs::format_for_developer(stale.error());
        EXPECT_EQ(stale->disposition, RecoveryDisposition::StaleDeleted);
    }

    TEST(ProjectContainerWriter,
         AutosaveSequenceMustExceedEveryValidBoundCandidate) {
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "sequence.licht";
        const fs::path stable = autosave_sidecar_path(master);
        const fs::path candidate =
            temporary.path / "sequence.licht.project-write.candidate.tmp.autosave";
        create_single_chunk_fixture(
            master, 919, 920, 921, fixed_key("PROJ", 922),
            R"({"master":"sequence"})");
        publish_complete_sidecar(master, stable, 5, 923, 924, 925);
        publish_complete_sidecar(master, candidate, 7, 926, 927, 928);

        ProjectReader base = require_result(ProjectReader::open(master));
        auto attempt = [&](const std::uint64_t sequence) {
            return ProjectWriter::create(
                stable,
                CreateOptions{
                    .project_uuid = base.superblock().project_uuid,
                    .file_uuid = fixed_uuid(929 + sequence),
                    .role = ContainerRole::AutosaveSidecar,
                    .base_explicit_commit_uuid = base.commit().commit_uuid,
                    .autosave_sequence = sequence,
                    .sidecar_snapshot_uuid = fixed_uuid(940 + sequence),
                    .creation_time_unix_ns = FIXED_CREATION_TIME_NS + sequence,
                    .index_compression =
                        IndexCompression::StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                    .writer_lock_anchor = master,
                });
        };
        auto lower = attempt(6);
        ASSERT_FALSE(lower);
        EXPECT_EQ(lower.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        auto equal = attempt(7);
        ASSERT_FALSE(equal);
        EXPECT_EQ(equal.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(require_result(ProjectReader::open(stable))
                      .superblock()
                      .autosave_sequence,
                  5u);

        publish_complete_sidecar(master, stable, 8, 950, 951, 952);
        EXPECT_EQ(require_result(ProjectReader::open(stable))
                      .superblock()
                      .autosave_sequence,
                  8u);
    }

    TEST(ProjectContainerWriter,
         AutosaveSidecarCreateRequiresMasterWriterLockAnchor) {
        TemporaryDirectory temporary;
        const fs::path sidecar =
            temporary.path / "raw.licht.autosave";
        auto created = ProjectWriter::create(
            sidecar,
            CreateOptions{
                .project_uuid = fixed_uuid(953),
                .file_uuid = fixed_uuid(954),
                .role = ContainerRole::AutosaveSidecar,
                .base_explicit_commit_uuid =
                    fixed_uuid(955),
                .autosave_sequence = 1,
                .sidecar_snapshot_uuid =
                    fixed_uuid(956),
                .creation_time_unix_ns =
                    FIXED_CREATION_TIME_NS,
                .index_compression =
                    IndexCompression::
                        StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_FALSE(created);
        EXPECT_EQ(created.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
        EXPECT_FALSE(fs::exists(sidecar));
    }

    TEST(ProjectContainerWriter, HeldOsLockNotLockfileExistenceControlsWriters) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "locked.licht";
        {
            ProjectWriter first = require_result(ProjectWriter::create(
                path, fixture_create_options(403)));
            auto second = ProjectWriter::create(
                path, fixture_create_options(404));
            ASSERT_FALSE(second);
            EXPECT_EQ(second.error().code(), lfs::ErrorCode::Unavailable);
        }
        const auto lock_path = fs::path(path.string() + ".lock");
        EXPECT_FALSE(fs::exists(lock_path));
        write_file_bytes(lock_path, byte_vector("stale lock is not authority"));

        ProjectWriter writer = require_result(ProjectWriter::create(
            path, fixture_create_options(403)));
        const auto payload = byte_vector("lockfile existence is not authority");
        require_status(
            writer.plan_commit(fixture_commit_options(505, 605, 1)));
        require_status(writer.preflight(payload.size()));
        require_status(
            writer.write_chunk(fixed_key("PROJ", 704), payload));
        require_status(writer.commit());
    }

    TEST(ProjectContainerWriter, ReclaimsOrphanTailAndCarriesOpaqueRowExactly) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "opaque-orphan.licht";
        create_single_chunk_fixture(path, 407, 508, 608,
                                    fixed_key("X999", 707),
                                    "opaque bytes owned by a future chapter");
        ProjectReader prior = require_result(ProjectReader::open(path));
        const ChunkInfo* opaque = prior.find(fixed_key("X999", 707));
        ASSERT_NE(opaque, nullptr);
        const std::uint64_t prior_end = prior.commit().committed_file_end;
        const std::uint64_t prior_header_offset = opaque->header_offset;
        CleanProof opaque_proof =
            require_result(prior.make_clean_proof(*opaque, 55));
        {
            std::ofstream orphan(path, std::ios::binary | std::ios::app);
            orphan << "unpublished orphan tail";
        }
        ASSERT_GT(fs::file_size(path), prior_end);

        ProjectWriter writer = require_result(
            ProjectWriter::append(path, fixture_append_options()));
        const auto new_payload = byte_vector("new known chapter");
        require_status(
            writer.plan_commit(fixture_commit_options(509, 609, 2)));
        require_status(writer.preflight(new_payload.size()));
        require_status(
            writer.carry_forward_opaque(*opaque, opaque_proof, 55));
        require_status(
            writer.write_chunk(fixed_key("VIEW", 708), new_payload));
        require_status(writer.commit());

        ProjectReader current = require_result(ProjectReader::open(path));
        const ChunkInfo* carried = current.find(fixed_key("X999", 707));
        const ChunkInfo* added = current.find(fixed_key("VIEW", 708));
        ASSERT_NE(carried, nullptr);
        ASSERT_NE(added, nullptr);
        EXPECT_EQ(carried->header_offset, prior_header_offset);
        EXPECT_EQ(carried->source_generation, 1u);
        EXPECT_EQ(added->header_offset, prior_end);
        EXPECT_EQ(require_result(current.read_chunk(*carried)),
                  byte_vector("opaque bytes owned by a future chapter"));
    }

    TEST(ProjectContainerWriter, ZstdStreamingAndCompactionRoundTrip) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "production.licht";
        const auto compressed_payload =
            byte_vector(std::string(32 * 1024, 'z'));
        const std::vector<std::byte> streamed_payload(
            static_cast<std::size_t>(BLOCK_CRC_BYTES * 2 + 137),
            std::byte{0x5a});
        {
            CreateOptions create = fixture_create_options(405);
            create.index_compression = IndexCompression::Zstd;
            ProjectWriter writer =
                require_result(ProjectWriter::create(path, create));
            require_status(writer.plan_commit(
                fixture_commit_options(506, 606, 1)));
            require_status(writer.preflight(compressed_payload.size() +
                                            streamed_payload.size()));
            ChunkWriteOptions compressed{
                .chunk_version = 1,
                .compression = Compression::ZstdFramed,
            };
            require_status(writer.write_chunk(fixed_key("PROJ", 705),
                                              compressed_payload,
                                              compressed));
            ChunkWriteOptions streaming{
                .chunk_version = 1,
                .compression = Compression::Stored,
                .tensor_payload = true,
                .block_crcs = true,
                .expected_stream_bytes = streamed_payload.size(),
            };
            std::ostream* stream = require_result(
                writer.begin_chunk(fixed_key("SPLT", 706), streaming));
            stream->write(
                reinterpret_cast<const char*>(streamed_payload.data()), 7);
            stream->write(
                reinterpret_cast<const char*>(streamed_payload.data() + 7),
                static_cast<std::streamsize>(streamed_payload.size() - 7));
            require_status(writer.end_chunk());
            require_status(writer.commit());
        }

        ProjectReader held =
            require_result(ProjectReader::open(path));
        EXPECT_EQ(held.commit().index_compression, Compression::ZstdFramed);
        const ChunkInfo* compressed_row =
            held.find(fixed_key("PROJ", 705));
        const ChunkInfo* streamed_row =
            held.find(fixed_key("SPLT", 706));
        ASSERT_NE(compressed_row, nullptr);
        ASSERT_NE(streamed_row, nullptr);
        EXPECT_EQ(require_result(held.read_chunk(*compressed_row)),
                  compressed_payload);
        EXPECT_EQ(require_result(held.read_chunk(*streamed_row)),
                  streamed_payload);
        ASSERT_TRUE(streamed_row->block_crc_table.has_value());
        MappedRegion held_mapping = require_result(
            held.map_stored_range(*streamed_row, 0,
                                  streamed_row->stored_bytes));
        const std::vector<std::byte> mapped_before(
            held_mapping.bytes().begin(), held_mapping.bytes().end());

        CompactionOptions compaction{
            .compatibility = {},
            .new_file_uuid = fixed_uuid(406),
            .commit_uuid = fixed_uuid(507),
            .snapshot_uuid = fixed_uuid(607),
            .creation_time_unix_ns = FIXED_CREATION_TIME_NS + 10,
            .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 10,
            .disk_reserve_bytes = 0,
        };
        require_status(ProjectWriter::compact(path, compaction));

        EXPECT_EQ(std::vector<std::byte>(held_mapping.bytes().begin(),
                                         held_mapping.bytes().end()),
                  mapped_before);
        EXPECT_EQ(held.commit().generation, 1u);
        EXPECT_EQ(require_result(held.read_chunk(*streamed_row)),
                  streamed_payload);

        ProjectReader compacted =
            require_result(ProjectReader::open(path));
        EXPECT_EQ(compacted.superblock().file_uuid, fixed_uuid(406));
        EXPECT_EQ(compacted.commit().generation, 1u);
        EXPECT_EQ(compacted.commit().kind, CommitKind::Compaction);
        EXPECT_EQ(compacted.chunks().size(), 2u);
        require_status(compacted.verify_all());
        const ChunkInfo* compacted_compressed =
            compacted.find(fixed_key("PROJ", 705));
        ASSERT_NE(compacted_compressed, nullptr);
        EXPECT_EQ(require_result(
                      compacted.read_chunk(*compacted_compressed)),
                  compressed_payload);
    }

    TEST(ProjectContainerWriter,
         CompactionPreservesUnknownAndNewerKnownChunksByteForByte) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "opaque-compaction.licht";
        const std::array keys = {
            fixed_key("X7Q9", 711), fixed_key("X7Q9", 712),
            fixed_key("SCNG", 713), fixed_key("SCNG", 714)};
        std::vector<std::vector<std::byte>> payloads(4);
        payloads[0].resize(11 * 1024 * 1024 + 37);
        payloads[1] = byte_vector("unknown fourcc without a block table");
        payloads[2].resize(5 * 1024 * 1024 + 19);
        payloads[3] = byte_vector(
            R"({"future_scene_graph_version":100,"block_table":false})");
        for (const std::size_t payload_index : {0u, 2u}) {
            for (std::size_t index = 0;
                 index < payloads[payload_index].size(); ++index) {
                payloads[payload_index][index] = static_cast<std::byte>(
                    (index * 131u + payload_index * 17u + 19u) & 0xffu);
            }
        }
        const std::array options = {
            ChunkWriteOptions{
                .chunk_version = 77,
                .compression = Compression::Stored,
                .tensor_payload = true,
                .block_crcs = true,
            },
            ChunkWriteOptions{
                .chunk_version = 78,
                .compression = Compression::ZstdFramed,
                .tensor_payload = false,
                .block_crcs = false,
            },
            ChunkWriteOptions{
                .chunk_version = 99,
                .compression = Compression::Stored,
                .tensor_payload = false,
                .block_crcs = true,
            },
            ChunkWriteOptions{
                .chunk_version = 100,
                .compression = Compression::ZstdFramed,
                .tensor_payload = false,
                .block_crcs = false,
            }};
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(path, fixture_create_options(410)));
            require_status(writer.plan_commit(
                fixture_commit_options(510, 610, 1)));
            std::uint64_t planned_bytes = 0;
            for (const auto& payload : payloads) {
                planned_bytes += payload.size();
            }
            require_status(writer.preflight(planned_bytes));
            for (std::size_t index = 0; index < keys.size(); ++index) {
                require_status(writer.write_chunk(
                    keys[index], payloads[index], options[index]));
            }
            require_status(writer.commit());
        }

        ProjectReader before = require_result(ProjectReader::open(path));
        struct StoredSnapshot {
            ChunkInfo metadata;
            std::vector<std::byte> payload;
        };
        std::vector<StoredSnapshot> snapshots;
        for (std::size_t index = 0; index < keys.size(); ++index) {
            const ChunkInfo* row = before.find(keys[index]);
            ASSERT_NE(row, nullptr);
            EXPECT_EQ(row->block_crc_table.has_value(),
                      index == 0 || index == 2);
            std::vector<std::byte> stored(row->stored_bytes);
            require_status(before.read_stored_at(*row, 0, stored));
            snapshots.push_back({.metadata = *row, .payload = std::move(stored)});
        }

        require_status(ProjectWriter::compact(
            path,
            CompactionOptions{
                .new_file_uuid = fixed_uuid(411),
                .commit_uuid = fixed_uuid(511),
                .snapshot_uuid = fixed_uuid(611),
                .creation_time_unix_ns = FIXED_CREATION_TIME_NS + 30,
                .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 30,
                .disk_reserve_bytes = 0,
            }));

        ProjectReader after = require_result(ProjectReader::open(path));
        require_status(after.verify_all());
        for (std::size_t index = 0; index < keys.size(); ++index) {
            const ChunkInfo* row = after.find(keys[index]);
            ASSERT_NE(row, nullptr);
            const ChunkInfo& original = snapshots[index].metadata;
            EXPECT_EQ(row->key, original.key);
            EXPECT_EQ(row->chunk_version, original.chunk_version);
            EXPECT_EQ(row->compression, original.compression);
            EXPECT_EQ(row->flags, original.flags);
            EXPECT_EQ(row->stored_bytes, original.stored_bytes);
            EXPECT_EQ(row->uncompressed_bytes, original.uncompressed_bytes);
            EXPECT_EQ(row->payload_crc32c, original.payload_crc32c);
            EXPECT_EQ(row->block_crc_table.has_value(),
                      original.block_crc_table.has_value());
            std::vector<std::byte> stored(row->stored_bytes);
            require_status(after.read_stored_at(*row, 0, stored));
            EXPECT_EQ(stored, snapshots[index].payload);
        }
    }

    TEST(ProjectContainerWriter,
         AcceleratedTwentyFourHourAutosaveScaleSimulation) {
        const char* enabled =
            std::getenv(
                "LFS_RUN_P7_SCALE_SIM");
        if (enabled == nullptr ||
            std::string_view(enabled) != "1") {
            GTEST_SKIP()
                << "set LFS_RUN_P7_SCALE_SIM=1 for the 288-cycle real-scale gate";
        }
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path /
            "p7-scale-master.licht";
        const fs::path sidecar =
            autosave_sidecar_path(master);
        constexpr std::uint64_t
            CHECKPOINT_BYTES =
                224ull * 1024 * 1024;
        constexpr std::uint64_t
            CYCLES = 24 * 60 / 5;
        constexpr std::size_t
            STREAM_WINDOW_BYTES =
                5 * 1024 * 1024;
        const ChunkKey checkpoint_key =
            fixed_key("CKPT", 1700);

        auto stream_checkpoint =
            [&](ProjectWriter& writer,
                const std::uint64_t cycle) {
                std::ostream* stream =
                    require_result(
                        writer.begin_chunk(
                            checkpoint_key,
                            ChunkWriteOptions{
                                .chunk_version =
                                    1,
                                .compression =
                                    Compression::
                                        Stored,
                                .tensor_payload =
                                    true,
                                .block_crcs =
                                    true,
                                .expected_stream_bytes =
                                    CHECKPOINT_BYTES,
                            }));
                std::vector<std::byte> window(
                    STREAM_WINDOW_BYTES);
                for (std::size_t index = 0;
                     index < window.size();
                     ++index) {
                    window[index] =
                        static_cast<std::byte>(
                            (index * 131u + 29u) &
                            0xffu);
                }
                std::uint64_t offset = 0;
                while (offset <
                       CHECKPOINT_BYTES) {
                    const auto count =
                        static_cast<std::size_t>(
                            std::min<
                                std::uint64_t>(
                                window.size(),
                                CHECKPOINT_BYTES -
                                    offset));
                    const auto marker =
                        cycle ^ offset;
                    std::memcpy(
                        window.data(), &marker,
                        std::min(
                            sizeof(marker),
                            count));
                    stream->write(
                        reinterpret_cast<
                            const char*>(
                            window.data()),
                        static_cast<
                            std::streamsize>(
                            count));
                    if (!*stream) {
                        throw std::runtime_error(
                            "scale-sim checkpoint stream failed");
                    }
                    offset += count;
                }
                require_status(
                    writer.end_chunk());
            };

        {
            ProjectWriter writer =
                require_result(
                    ProjectWriter::create(
                        master,
                        fixture_create_options(
                            1701)));
            require_status(
                writer.plan_commit(
                    fixture_commit_options(
                        1702, 1703, 1)));
            require_status(
                writer.preflight(
                    CHECKPOINT_BYTES));
            stream_checkpoint(writer, 0);
            require_status(writer.commit());
        }
        ProjectReader base =
            require_result(
                ProjectReader::open(master));
        require_status(base.verify_all());
        const std::uint64_t master_bytes =
            fs::file_size(master);

        const auto artifact_bytes =
            [&]() -> std::uint64_t {
            std::uint64_t total = 0;
            for (const auto& entry :
                 fs::directory_iterator(
                     temporary.path)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const auto name =
                    entry.path()
                        .filename()
                        .string();
                if (!name.starts_with(
                        master.stem()
                            .string()) ||
                    name.ends_with(
                        ".lock")) {
                    continue;
                }
                total +=
                    entry.file_size();
            }
            return total;
        };

        std::uint64_t transient_peak =
            master_bytes;
        std::uint64_t steady_peak =
            master_bytes;
        std::uint64_t autosave_min =
            std::numeric_limits<
                std::uint64_t>::max();
        std::uint64_t autosave_max = 0;
        const auto started =
            std::chrono::steady_clock::now();
        for (std::uint64_t cycle = 1;
             cycle <= CYCLES; ++cycle) {
            const auto snapshot_uuid =
                lfs::core::
                    generate_uuid_v4();
            ProjectWriter writer =
                require_result(
                    ProjectWriter::create(
                        sidecar,
                        CreateOptions{
                            .project_uuid =
                                base
                                    .superblock()
                                    .project_uuid,
                            .file_uuid =
                                lfs::core::
                                    generate_uuid_v4(),
                            .role =
                                ContainerRole::
                                    AutosaveSidecar,
                            .base_explicit_commit_uuid =
                                base.commit()
                                    .commit_uuid,
                            .autosave_sequence =
                                cycle,
                            .sidecar_snapshot_uuid =
                                snapshot_uuid,
                            .creation_time_unix_ns =
                                FIXED_CREATION_TIME_NS +
                                cycle,
                            .index_compression =
                                IndexCompression::
                                    Zstd,
                            .disk_reserve_bytes =
                                0,
                            .boundary_observer =
                                [&](const CommitBoundary) {
                                    transient_peak =
                                        std::max(
                                            transient_peak,
                                            artifact_bytes());
                                },
                            .writer_lock_anchor =
                                master,
                        }));
            require_status(
                writer.plan_commit(
                    CommitOptions{
                        .kind =
                            CommitKind::
                                Autosave,
                        .commit_uuid =
                            lfs::core::
                                generate_uuid_v4(),
                        .snapshot_uuid =
                            snapshot_uuid,
                        .wallclock_unix_ns =
                            FIXED_COMMIT_TIME_NS +
                            cycle,
                    }));
            require_status(
                writer.preflight(
                    CHECKPOINT_BYTES));
            stream_checkpoint(
                writer, cycle);
            require_status(writer.commit());

            const auto current_autosave =
                fs::file_size(sidecar);
            autosave_min =
                std::min(
                    autosave_min,
                    current_autosave);
            autosave_max =
                std::max(
                    autosave_max,
                    current_autosave);
            const auto steady =
                artifact_bytes();
            steady_peak =
                std::max(
                    steady_peak, steady);
            EXPECT_EQ(
                steady,
                master_bytes +
                    current_autosave);
        }
        auto recovery =
            inspect_autosave_recovery(
                master);
        ASSERT_TRUE(recovery)
            << lfs::format_for_developer(
                   recovery.error());
        EXPECT_EQ(
            recovery->disposition,
            RecoveryDisposition::Offer);
        EXPECT_EQ(
            recovery->autosave_sequence,
            CYCLES);
        const auto elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::
                    now() -
                started)
                .count();
        constexpr std::uint64_t EPSILON =
            2ull * 1024 * 1024;
        EXPECT_LE(
            steady_peak,
            master_bytes +
                autosave_max);
        EXPECT_LE(
            transient_peak,
            master_bytes +
                2 * autosave_max +
                EPSILON);
        std::cout
            << std::format(
                   "P7_AUTOSAVE_SCALE cycles={} checkpoint_bytes={} "
                   "master_bytes={} autosave_min_bytes={} "
                   "autosave_max_bytes={} steady_peak_bytes={} "
                   "transient_peak_bytes={} transient_bound_bytes={} "
                   "elapsed_seconds={:.3f}\n",
                   CYCLES, CHECKPOINT_BYTES,
                   master_bytes, autosave_min,
                   autosave_max, steady_peak,
                   transient_peak,
                   master_bytes +
                       2 * autosave_max +
                       EPSILON,
                   elapsed);
    }

#ifndef _WIN32

#endif

    [[nodiscard]] std::optional<std::uint64_t>
    error_field_u64(const lfs::Error& error, const std::string_view key) {
        if (error.frames().empty()) {
            return std::nullopt;
        }
        for (const auto& entry : error.frames().front().fields.entries()) {
            if (entry.key != key) {
                continue;
            }
            if (const auto* value = std::get_if<std::uint64_t>(&entry.value)) {
                return *value;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string>
    error_field_string(const lfs::Error& error, const std::string_view key) {
        if (error.frames().empty()) {
            return std::nullopt;
        }
        for (const auto& entry : error.frames().front().fields.entries()) {
            if (entry.key != key) {
                continue;
            }
            if (const auto* value = std::get_if<std::string>(&entry.value)) {
                return *value;
            }
        }
        return std::nullopt;
    }

    // Highly compressible zeros just over the historical 512 MiB bomb guard.
    // Runtime stays sane because zstd collapses the stored frame; materialize
    // still allocates the full logical size once (payload-class 16 GiB bound).
    TEST(ProjectContainerWriter, ZstdPayloadClassJustOver512MiBRoundTrip) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "payload-over-512mib.licht";
        constexpr std::size_t kLogical =
            static_cast<std::size_t>(512ull * 1024 * 1024) + 1u;
        const std::vector<std::byte> payload(kLogical, std::byte{0});
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(path, fixture_create_options(9300)));
            require_status(
                writer.plan_commit(fixture_commit_options(9301, 9302, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(
                fixed_key("CKPT", 9303), payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ZstdFramed,
                    .tensor_payload = true,
                }));
            // verify-before-publish must accept payload-class >512 MiB.
            require_status(writer.commit());
        }
        ProjectReader reader = require_result(ProjectReader::open(path));
        require_status(reader.verify_all());
        const ChunkInfo* row = reader.find(fixed_key("CKPT", 9303));
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->uncompressed_bytes, kLogical);
        EXPECT_LT(row->stored_bytes, row->uncompressed_bytes);
        EXPECT_EQ(require_result(reader.read_chunk(*row)), payload);
    }

    TEST(ProjectContainerWriter, ZstdNonPayloadClassOver512MiBRefused) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "json-over-512mib.licht";
        constexpr std::size_t kLogical =
            static_cast<std::size_t>(512ull * 1024 * 1024) + 1u;
        const std::vector<std::byte> payload(kLogical, std::byte{0});
        ProjectWriter writer = require_result(
            ProjectWriter::create(path, fixture_create_options(9310)));
        require_status(
            writer.plan_commit(fixture_commit_options(9311, 9312, 1)));
        require_status(writer.preflight(payload.size()));
        require_status(writer.write_chunk(
            fixed_key("PROJ", 9313), payload,
            ChunkWriteOptions{
                .chunk_version = 1,
                .compression = Compression::ZstdFramed,
                .tensor_payload = false,
            }));
        // Create-mode post-publish verify applies the 512 MiB small-class cap.
        auto published = writer.commit();
        ASSERT_FALSE(published);
        EXPECT_EQ(published.error().code(), lfs::ErrorCode::ResourceExhausted);
        const std::string detail = lfs::format_for_developer(published.error());
        EXPECT_TRUE(detail.find("implementation maximum") != std::string::npos ||
                    detail.find("536870912") != std::string::npos ||
                    detail.find("materialized") != std::string::npos)
            << detail;
    }

    TEST(ProjectContainerWriter, PreviewMaxBytesAtLimitAndOneOver) {
        // Kills: boundary mutants on MAX_* limit guards (</<=).
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "preview-max.licht";
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(path, fixture_create_options(9200)));
            require_status(writer.plan_commit(
                fixture_commit_options(9201, 9202, 1)));
            std::vector<std::byte> at_limit(MAX_PREVIEW_BYTES, std::byte{0xab});
            // PNG signature so convenience readers remain happy if accepted.
            constexpr std::array png_signature{
                std::byte{0x89}, std::byte{0x50}, std::byte{0x4e}, std::byte{0x47},
                std::byte{0x0d}, std::byte{0x0a}, std::byte{0x1a}, std::byte{0x0a}};
            std::copy(png_signature.begin(), png_signature.end(), at_limit.begin());
            require_status(writer.preflight(at_limit.size()));
            auto at_ok = writer.set_preview(at_limit);
            EXPECT_TRUE(at_ok) << lfs::format_for_developer(at_ok.error());

            std::vector<std::byte> one_over(MAX_PREVIEW_BYTES + 1u, std::byte{0xcd});
            auto over = writer.set_preview(one_over);
            ASSERT_FALSE(over);
            EXPECT_EQ(over.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_NE(std::string(over.error().detail()).find(std::to_string(MAX_PREVIEW_BYTES)),
                      std::string::npos);
            // Still publish the at-limit preview generation.
            require_status(writer.commit());
        }
        ProjectReader reader = require_result(ProjectReader::open(path));
        ASSERT_TRUE(reader.preview().has_value());
        EXPECT_EQ(reader.preview()->bytes, MAX_PREVIEW_BYTES);
    }

    TEST(ProjectContainerFile, CheckedAddAndMultiplyOverflow) {
        // Kills: project_file checked_add/checked_multiply overflow and short-write paths.
        // Short-write errno injection is not feasible without new deps/seams; overflow
        // oracles cover the structured BoundsViolation path in the same TU.
        const fs::path path = "overflow-probe.licht";
        constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
        {
            auto sum = detail::checked_add(kMax, 1, path, 0x10, "probe.add");
            ASSERT_FALSE(sum);
            EXPECT_EQ(sum.error().code(), lfs::ErrorCode::BoundsViolation);
            EXPECT_EQ(error_field_u64(sum.error(), "offset"), 0x10u);
            EXPECT_EQ(error_field_string(sum.error(), "field"), "probe.add");
            EXPECT_NE(std::string(sum.error().detail()).find("overflow"),
                      std::string::npos);
        }
        {
            auto product =
                detail::checked_multiply(kMax, 2, path, 0x20, "probe.mul");
            ASSERT_FALSE(product);
            EXPECT_EQ(product.error().code(), lfs::ErrorCode::BoundsViolation);
            EXPECT_EQ(error_field_u64(product.error(), "offset"), 0x20u);
            EXPECT_EQ(error_field_string(product.error(), "field"), "probe.mul");
            EXPECT_NE(std::string(product.error().detail()).find("overflow"),
                      std::string::npos);
        }
        {
            auto ok_add = detail::checked_add(kMax - 1, 1, path, 0, "probe.add.ok");
            ASSERT_TRUE(ok_add);
            EXPECT_EQ(*ok_add, kMax);
            auto ok_mul = detail::checked_multiply(1ull << 32, 1ull << 31, path, 0,
                                                   "probe.mul.ok");
            ASSERT_TRUE(ok_mul);
            EXPECT_EQ(*ok_mul, 1ull << 63);
        }
    }

    TEST(ProjectContainerWriter,
         InspectQuarantinesCorruptStableSidecarAndOpensMaster) {
        // Would fail if inspect still returned Invalid or left the sidecar
        // in place / unlinked it instead of renaming aside.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "corrupt-sidecar.licht";
        create_single_chunk_fixture(
            master, 960, 961, 962, fixed_key("PROJ", 963),
            R"({"master":"corrupt-sidecar"})");
        const auto original = read_file_bytes(master);
        const fs::path sidecar = autosave_sidecar_path(master);
        const auto garbage = byte_vector("truncated-autosave-garbage");
        write_file_bytes(sidecar, garbage);

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_NE(inspection->disposition, RecoveryDisposition::Invalid);
        EXPECT_NE(inspection->disposition, RecoveryDisposition::Ambiguous);
        EXPECT_TRUE(
            inspection->disposition == RecoveryDisposition::StaleDeleted ||
            inspection->disposition == RecoveryDisposition::None);
        EXPECT_FALSE(fs::exists(sidecar));
        const auto asides = corrupt_asides_of(sidecar);
        ASSERT_EQ(asides.size(), 1u);
        EXPECT_EQ(read_file_bytes(asides.front()), garbage);
        auto reader = ProjectReader::open(master);
        ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
        EXPECT_EQ(read_file_bytes(master), original);
    }

    TEST(ProjectContainerWriter,
         InspectQuarantinesWrongProjectUuidStableSidecar) {
        // Would fail if a leftover sidecar from another project still
        // produced Invalid or was unlinked instead of quarantined.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "foreign-uuid.licht";
        const fs::path foreign_master =
            temporary.path / "foreign-src.licht";
        create_single_chunk_fixture(
            master, 964, 965, 966, fixed_key("PROJ", 967),
            R"({"master":"foreign-uuid"})");
        CreateOptions foreign_create = fixture_create_options(1060);
        foreign_create.project_uuid = fixed_uuid(777);
        {
            ProjectWriter writer = require_result(
                ProjectWriter::create(foreign_master, foreign_create));
            const auto payload = byte_vector(R"({"master":"foreign-src"})");
            require_status(writer.plan_commit(
                fixture_commit_options(1061, 1062, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(
                fixed_key("PROJ", 1063), payload));
            require_status(writer.commit());
        }
        const fs::path sidecar = autosave_sidecar_path(master);
        publish_complete_sidecar(
            foreign_master, sidecar, 1, 968, 969, 970);
        const auto original = read_file_bytes(sidecar);

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_EQ(inspection->disposition, RecoveryDisposition::StaleDeleted);
        EXPECT_FALSE(fs::exists(sidecar));
        const auto asides = corrupt_asides_of(sidecar);
        ASSERT_EQ(asides.size(), 1u);
        EXPECT_EQ(read_file_bytes(asides.front()), original);
        ASSERT_TRUE(ProjectReader::open(master));
    }

    TEST(ProjectContainerWriter,
         InspectSucceedsWhenHygieneRemoveHitsNonEmptyDirectory) {
        // Would fail if inspect still returned PermissionDenied when
        // std::filesystem::remove cannot unlink a non-empty directory.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "hygiene-dir.licht";
        create_single_chunk_fixture(
            master, 971, 972, 973, fixed_key("PROJ", 974),
            R"({"master":"hygiene-dir"})");
        const fs::path blocker =
            temporary.path / "hygiene-dir.compact.killed.1.0.tmp.licht";
        ASSERT_TRUE(fs::create_directory(blocker));
        {
            std::ofstream child(blocker / "child");
            ASSERT_TRUE(child);
            child << "blocks remove";
        }

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_NE(inspection->disposition, RecoveryDisposition::Invalid);
        EXPECT_TRUE(
            inspection->disposition == RecoveryDisposition::None ||
            inspection->disposition == RecoveryDisposition::StaleDeleted);
        ASSERT_FALSE(inspection->diagnostics.empty());
        EXPECT_TRUE(std::ranges::any_of(
            inspection->diagnostics, [&](const std::string& diagnostic) {
                return diagnostic.find(blocker.filename().string()) !=
                       std::string::npos;
            }));
        EXPECT_TRUE(fs::is_directory(blocker));
    }

    TEST(ProjectContainerWriter,
         InspectPicksHigherWallclockOnSequenceTieAndQuarantinesLoser) {
        // Would fail if inspect returned Ambiguous or picked the older
        // wallclock when sequences match.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "tie-clock.licht";
        create_single_chunk_fixture(
            master, 975, 976, 977, fixed_key("PROJ", 978),
            R"({"master":"tie-clock"})");
        const fs::path stable = autosave_sidecar_path(master);
        const fs::path candidate =
            temporary.path /
            (master.filename().string() +
             ".project-write.tie.1.0.tmp.autosave");
        CommitOptions older = fixture_commit_options(979, 980, 1);
        older.wallclock_unix_ns = 1'000;
        CommitOptions newer = fixture_commit_options(981, 982, 1);
        newer.wallclock_unix_ns = 2'000;
        publish_complete_sidecar(
            master, stable, 4, 983, 979, 980, older);
        const auto parked = temporary.path / "parked-stable.aside";
        fs::rename(stable, parked);
        publish_complete_sidecar(
            master, candidate, 4, 984, 981, 982, newer);
        fs::rename(parked, stable);
        const auto loser_bytes = read_file_bytes(stable);

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_EQ(inspection->disposition, RecoveryDisposition::Offer);
        ASSERT_TRUE(inspection->selected_path);
        EXPECT_EQ(*inspection->selected_path, candidate);
        EXPECT_FALSE(fs::exists(stable));
        const auto asides = corrupt_asides_of(stable);
        ASSERT_EQ(asides.size(), 1u);
        EXPECT_EQ(read_file_bytes(asides.front()), loser_bytes);
        EXPECT_TRUE(fs::is_regular_file(candidate));
    }

    TEST(ProjectContainerWriter,
         InspectPicksLexicographicallyGreaterPathOnFullTieAndQuarantinesLoser) {
        // Would fail if inspect returned Ambiguous or used an unstable
        // path order. candidate filename is lexicographically greater
        // than the stable sidecar (".project-write." > ".autosave").
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "tie-path.licht";
        create_single_chunk_fixture(
            master, 985, 986, 987, fixed_key("PROJ", 988),
            R"({"master":"tie-path"})");
        const fs::path stable = autosave_sidecar_path(master);
        const fs::path candidate =
            temporary.path /
            (master.filename().string() +
             ".project-write.tie.1.0.tmp.autosave");
        ASSERT_GT(candidate.generic_string(), stable.generic_string());
        CommitOptions tied = fixture_commit_options(989, 990, 1);
        tied.wallclock_unix_ns = 3'000;
        publish_complete_sidecar(
            master, stable, 5, 991, 989, 990, tied);
        const auto parked = temporary.path / "parked-stable.aside";
        fs::rename(stable, parked);
        CommitOptions tied_again = fixture_commit_options(992, 993, 1);
        tied_again.wallclock_unix_ns = 3'000;
        publish_complete_sidecar(
            master, candidate, 5, 994, 992, 993, tied_again);
        fs::rename(parked, stable);
        const auto loser_bytes = read_file_bytes(stable);

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_EQ(inspection->disposition, RecoveryDisposition::Offer);
        ASSERT_TRUE(inspection->selected_path);
        EXPECT_EQ(*inspection->selected_path, candidate);
        EXPECT_FALSE(fs::exists(stable));
        const auto asides = corrupt_asides_of(stable);
        ASSERT_EQ(asides.size(), 1u);
        EXPECT_EQ(read_file_bytes(asides.front()), loser_bytes);
    }

    TEST(ProjectContainerWriter,
         InspectPicksHigherSequenceRegardlessOfWallclockOrPath) {
        // Would fail if wallclock or path could beat a higher sequence.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "seq-wins.licht";
        create_single_chunk_fixture(
            master, 995, 996, 997, fixed_key("PROJ", 998),
            R"({"master":"seq-wins"})");
        const fs::path stable = autosave_sidecar_path(master);
        const fs::path candidate =
            temporary.path /
            (master.filename().string() +
             ".project-write.tie.1.0.tmp.autosave");
        CommitOptions low_seq_new_clock =
            fixture_commit_options(999, 1000, 1);
        low_seq_new_clock.wallclock_unix_ns = 9'000;
        CommitOptions high_seq_old_clock =
            fixture_commit_options(1001, 1002, 1);
        high_seq_old_clock.wallclock_unix_ns = 1;
        publish_complete_sidecar(
            master, candidate, 1, 1003, 999, 1000,
            low_seq_new_clock);
        publish_complete_sidecar(
            master, stable, 8, 1004, 1001, 1002,
            high_seq_old_clock);

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_EQ(inspection->disposition, RecoveryDisposition::Offer);
        ASSERT_TRUE(inspection->selected_path);
        EXPECT_EQ(*inspection->selected_path, stable);
        EXPECT_EQ(inspection->autosave_sequence, 8u);
        EXPECT_FALSE(fs::exists(candidate));
        EXPECT_FALSE(corrupt_asides_of(candidate).empty());
    }

    TEST(ProjectContainerWriter, InspectDeletesFreeOwnedWriteTemp) {
        // Would fail if project-write master temps were still invisible
        // to inspect.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "free-write.licht";
        create_single_chunk_fixture(
            master, 1005, 1006, 1007, fixed_key("PROJ", 1008),
            R"({"master":"free-write"})");
        const fs::path free_temp =
            temporary.path / "free-write.project-write.free.1.0.tmp.licht";
        write_file_bytes(free_temp, byte_vector("orphan write temp"));

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_FALSE(fs::exists(free_temp));
        EXPECT_TRUE(corrupt_asides_of(free_temp).empty());
        EXPECT_NE(std::ranges::find(inspection->deleted_paths, free_temp),
                  inspection->deleted_paths.end());
    }

    TEST(ProjectContainerWriter,
         SweepQuarantinesFreeWriteTempWhenMasterAbsent) {
        // Would fail if flock-free first-publish temps were remove()d
        // when the destination master is absent, or left in place when
        // a published master exists and the temp flock is free.
        TemporaryDirectory temporary;
        const fs::path missing = temporary.path / "missing.licht";
        const fs::path missing_temp =
            temporary.path /
            "missing.project-write.crash.1.0.tmp.licht";
        const auto only_copy = byte_vector("only copy of crashed first publish");
        write_file_bytes(missing_temp, only_copy);
        ASSERT_FALSE(fs::exists(missing));

        RecoveryInspection absent;
        sweep_orphan_project_artifacts(missing, absent);
        EXPECT_FALSE(fs::exists(missing_temp));
        EXPECT_FALSE(fs::exists(missing));
        const auto asides = corrupt_asides_of(missing_temp);
        ASSERT_EQ(asides.size(), 1u);
        EXPECT_EQ(read_file_bytes(asides.front()), only_copy);
        EXPECT_NE(std::ranges::find(absent.deleted_paths, missing_temp),
                  absent.deleted_paths.end());
        const auto joined = [&] {
            std::string text;
            for (const auto& line : absent.diagnostics) {
                text += line;
                text += '\n';
            }
            return text;
        }();
        EXPECT_NE(joined.find("quarantined"), std::string::npos);
        EXPECT_EQ(joined.find("quarantine failed"), std::string::npos);

        const fs::path present = temporary.path / "present.licht";
        create_single_chunk_fixture(
            present, 1025, 1026, 1027, fixed_key("PROJ", 1028),
            R"({"master":"present"})");
        const fs::path present_temp =
            temporary.path /
            "present.project-write.free.1.0.tmp.licht";
        write_file_bytes(present_temp, byte_vector("superseded write temp"));

        RecoveryInspection published;
        sweep_orphan_project_artifacts(present, published);
        EXPECT_TRUE(fs::exists(present));
        EXPECT_FALSE(fs::exists(present_temp));
        EXPECT_TRUE(corrupt_asides_of(present_temp).empty());
        EXPECT_NE(std::ranges::find(published.deleted_paths, present_temp),
                  published.deleted_paths.end());
    }

    TEST(ProjectContainerWriter, InspectKeepsHeldWriteTempAndRemovesFreeSiblings) {
        // Would fail if sweep unlinked a temp whose WriterLockLease is held,
        // or stopped deleting free compact/replace-backup temps.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "held-write.licht";
        create_single_chunk_fixture(
            master, 1009, 1010, 1011, fixed_key("PROJ", 1012),
            R"({"master":"held-write"})");
        const fs::path held_temp =
            temporary.path / "held-write.project-write.held.1.0.tmp.licht";
        const fs::path compact_temp =
            temporary.path / "held-write.compact.free.1.0.tmp.licht";
        const fs::path backup_temp =
            temporary.path / "held-write.replace-backup.free.1.0.tmp.licht";
        write_file_bytes(held_temp, byte_vector("live write"));
        write_file_bytes(compact_temp, byte_vector("orphan compact"));
        write_file_bytes(backup_temp, byte_vector("orphan backup"));
        auto held = WriterLockLease::acquire(held_temp);
        ASSERT_TRUE(held) << lfs::format_for_developer(held.error());

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_TRUE(fs::exists(held_temp));
        EXPECT_FALSE(fs::exists(compact_temp));
        EXPECT_FALSE(fs::exists(backup_temp));
    }

    TEST(ProjectContainerWriter, InspectPrunesCorruptAsidesToNewestThree) {
        // Would fail if .corrupt-* asides grew without a per-stem cap.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "prune-corrupt.licht";
        create_single_chunk_fixture(
            master, 1013, 1014, 1015, fixed_key("PROJ", 1016),
            R"({"master":"prune-corrupt"})");
        const std::array stamps{100, 200, 300, 400, 500};
        for (const int stamp : stamps) {
            write_file_bytes(
                fs::path(master.string() +
                         std::format(".corrupt-{}", stamp)),
                byte_vector(std::format("aside-{}", stamp)));
        }

        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        EXPECT_FALSE(fs::exists(fs::path(master.string() + ".corrupt-100")));
        EXPECT_FALSE(fs::exists(fs::path(master.string() + ".corrupt-200")));
        EXPECT_TRUE(fs::exists(fs::path(master.string() + ".corrupt-300")));
        EXPECT_TRUE(fs::exists(fs::path(master.string() + ".corrupt-400")));
        EXPECT_TRUE(fs::exists(fs::path(master.string() + ".corrupt-500")));
    }

    TEST(ProjectContainerWriter, InspectLeavesMasterLockFile) {
        // Would fail if inspect left a permanent {master}.lock after
        // releasing the writer lock (O7: sweep still skips a held lock).
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "keep-lock.licht";
        create_single_chunk_fixture(
            master, 1017, 1018, 1019, fixed_key("PROJ", 1020),
            R"({"master":"keep-lock"})");
        auto inspection = inspect_autosave_recovery(master);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        auto lock_path = master;
        lock_path += ".lock";
        EXPECT_FALSE(fs::exists(lock_path));
    }

    TEST(ProjectContainerWriter, WriterLockReleaseDeletesLockFile) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "lock-hygiene.licht";
        auto lock_path = path;
        lock_path += ".lock";
        {
            auto lock = detail::WriterLock::acquire(path);
            ASSERT_TRUE(lock)
                << lfs::format_for_developer(lock.error());
            EXPECT_TRUE(fs::exists(lock_path));
        }
        EXPECT_FALSE(fs::exists(lock_path));
    }

    TEST(ProjectContainerWriter,
         WriterLockContentionLoserErrorsWinnerReleaseDeletes) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "lock-contend.licht";
        auto lock_path = path;
        lock_path += ".lock";
        std::optional<detail::WriterLock> winner;
        {
            auto acquired = detail::WriterLock::acquire(path);
            ASSERT_TRUE(acquired)
                << lfs::format_for_developer(acquired.error());
            winner.emplace(std::move(*acquired));
        }
        auto loser = detail::WriterLock::acquire(path);
        ASSERT_FALSE(loser);
        EXPECT_EQ(loser.error().code(), lfs::ErrorCode::Unavailable);
        EXPECT_TRUE(fs::exists(lock_path));
        winner.reset();
        EXPECT_FALSE(fs::exists(lock_path));
        {
            auto reacquired = detail::WriterLock::acquire(path);
            ASSERT_TRUE(reacquired)
                << lfs::format_for_developer(reacquired.error());
        }
        EXPECT_FALSE(fs::exists(lock_path));
    }

#ifndef _WIN32
    TEST(ProjectContainerWriter, WriterLockFdMatchesCurrentLockPath) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "lock-identity.licht";
        auto lock_path = path;
        lock_path += ".lock";
        const int fd =
            ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        ASSERT_GE(fd, 0);
        ASSERT_EQ(::flock(fd, LOCK_EX | LOCK_NB), 0);
        auto identity = detail::writer_lock_fd_matches_path(fd, lock_path);
        ::flock(fd, LOCK_UN);
        ::close(fd);
        ASSERT_TRUE(identity) << lfs::format_for_developer(identity.error());
        EXPECT_TRUE(*identity);
    }

    TEST(ProjectContainerWriter, WriterLockFdMismatchOnReplacedLockFile) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "lock-replaced.licht";
        auto lock_path = path;
        lock_path += ".lock";
        const int stale =
            ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        ASSERT_GE(stale, 0);
        ASSERT_EQ(::flock(stale, LOCK_EX | LOCK_NB), 0);
        ASSERT_EQ(::unlink(lock_path.c_str()), 0);
        const int current =
            ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        ASSERT_GE(current, 0);
        auto stale_identity =
            detail::writer_lock_fd_matches_path(stale, lock_path);
        auto current_identity =
            detail::writer_lock_fd_matches_path(current, lock_path);
        auto missing_identity = detail::writer_lock_fd_matches_path(
            stale, temporary.path / "missing.licht.lock");
        ::flock(stale, LOCK_UN);
        ::close(stale);
        ::close(current);
        ASSERT_TRUE(stale_identity)
            << lfs::format_for_developer(stale_identity.error());
        EXPECT_FALSE(*stale_identity);
        ASSERT_TRUE(current_identity)
            << lfs::format_for_developer(current_identity.error());
        EXPECT_TRUE(*current_identity);
        ASSERT_TRUE(missing_identity)
            << lfs::format_for_developer(missing_identity.error());
        EXPECT_FALSE(*missing_identity);
    }

    TEST(ProjectContainerWriter, WriterLockFdFstatFailureIsError) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "lock-fstat.licht";
        auto lock_path = path;
        lock_path += ".lock";
        const int fd =
            ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        ASSERT_GE(fd, 0);
        ASSERT_EQ(::close(fd), 0);
        auto identity = detail::writer_lock_fd_matches_path(fd, lock_path);
        ASSERT_FALSE(identity);
        EXPECT_EQ(identity.error().code(), lfs::ErrorCode::Internal);
        EXPECT_NE(std::string(identity.error().detail()).find("lockfile fstat failed"),
                  std::string::npos);
        ASSERT_TRUE(identity.error().native().has_value());
        EXPECT_EQ(identity.error().native()->code, EBADF);
    }

    TEST(ProjectContainerWriter, WriterLockAcquireFailsWhenLockPathIsDirectory) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "dir-lock.licht";
        auto lock_path = path;
        lock_path += ".lock";
        fs::create_directory(lock_path);
        auto lock = detail::WriterLock::acquire(path);
        ASSERT_FALSE(lock);
        EXPECT_NE(lock.error().code(), lfs::ErrorCode::Unavailable);
        EXPECT_NE(std::string(lock.error().detail()).find("lockfile open failed"),
                  std::string::npos);
        EXPECT_TRUE(fs::is_directory(lock_path));
    }
#endif

    TEST(ProjectContainerWriter,
         StartupSweepRemovesStaleMasterLockAndSaveasTemp) {
        TemporaryDirectory temporary;
        const fs::path master =
            temporary.path / "startup-hygiene.licht";
        create_single_chunk_fixture(
            master, 1101, 1102, 1103, fixed_key("PROJ", 1104),
            R"({"master":"startup-hygiene"})");
        auto master_lock = master;
        master_lock += ".lock";
        write_file_bytes(master_lock, byte_vector("stale master lock"));
        const fs::path saveas =
            temporary.path /
            ".startup-hygiene.licht.saveas-deadbeef.tmp";
        auto saveas_lock = saveas;
        saveas_lock += ".lock";
        write_file_bytes(saveas, byte_vector("stale saveas staging"));
        write_file_bytes(saveas_lock, byte_vector("stale saveas lock"));

        const fs::path other_master =
            temporary.path / "other.licht";
        write_file_bytes(other_master, byte_vector("foreign master"));
        const fs::path foreign =
            temporary.path / ".other.licht.saveas-bbbb.tmp";
        write_file_bytes(foreign, byte_vector("foreign saveas staging"));
        const fs::path foreign_compact =
            temporary.path / "other.compact.1.2.3.tmp.licht";
        write_file_bytes(foreign_compact, byte_vector("foreign compact"));

        const fs::path ghost =
            temporary.path / ".ghost.licht.saveas-xxxx.tmp";
        auto ghost_lock = ghost;
        ghost_lock += ".lock";
        write_file_bytes(ghost, byte_vector("unreferenced saveas staging"));
        write_file_bytes(ghost_lock, byte_vector("unreferenced saveas lock"));
        const fs::path ghost_compact =
            temporary.path / "ghost.compact.1.2.3.tmp.licht";
        write_file_bytes(ghost_compact, byte_vector("unreferenced compact"));

        const fs::path held_ghost =
            temporary.path / ".ghost.licht.saveas-held.tmp";
        auto held_ghost_lock = held_ghost;
        held_ghost_lock += ".lock";
        write_file_bytes(held_ghost, byte_vector("held ghost staging"));
        write_file_bytes(held_ghost_lock, byte_vector("held ghost lock"));
        auto held_ghost_lease = WriterLockLease::acquire(held_ghost);
        ASSERT_TRUE(held_ghost_lease)
            << lfs::format_for_developer(held_ghost_lease.error());

        const fs::path held_master =
            temporary.path / "startup-held.licht";
        create_single_chunk_fixture(
            held_master, 1105, 1106, 1107, fixed_key("PROJ", 1108),
            R"({"master":"startup-held"})");
        auto held = WriterLockLease::acquire(held_master);
        ASSERT_TRUE(held)
            << lfs::format_for_developer(held.error());
        auto held_lock = held_master;
        held_lock += ".lock";
        EXPECT_TRUE(fs::exists(held_lock));

        const fs::path missing = temporary.path / "missing.licht";
        sweep_stale_licht_artifacts_for_known_masters(
            {master, held_master, missing});

        EXPECT_FALSE(fs::exists(master_lock));
        EXPECT_FALSE(fs::exists(saveas));
        EXPECT_FALSE(fs::exists(saveas_lock));
        EXPECT_FALSE(fs::exists(ghost));
        EXPECT_FALSE(fs::exists(ghost_lock));
        EXPECT_FALSE(fs::exists(ghost_compact));
        EXPECT_TRUE(fs::exists(other_master));
        EXPECT_TRUE(fs::exists(foreign));
        EXPECT_TRUE(fs::exists(foreign_compact));
        EXPECT_TRUE(fs::exists(held_ghost));
        EXPECT_TRUE(fs::exists(held_ghost_lock));
        EXPECT_TRUE(fs::exists(held_lock));
        EXPECT_FALSE(fs::exists(missing));
    }

    TEST(ProjectContainerWriter,
         RemoveAutosaveArtifactsSweepsFreeWriteTemp) {
        // Would fail if remove_autosave_artifacts still skipped master
        // project-write temps.
        TemporaryDirectory temporary;
        const fs::path master = temporary.path / "remove-sweep.licht";
        create_single_chunk_fixture(
            master, 1021, 1022, 1023, fixed_key("PROJ", 1024),
            R"({"master":"remove-sweep"})");
        const fs::path free_temp =
            temporary.path /
            "remove-sweep.project-write.free.1.0.tmp.licht";
        write_file_bytes(free_temp, byte_vector("save-time orphan"));
        auto removed = remove_autosave_artifacts(master);
        ASSERT_TRUE(removed)
            << lfs::format_for_developer(removed.error());
        EXPECT_FALSE(fs::exists(free_temp));
    }

    TEST(ProjectRecoveryScratch, PathResolutionUsesSessionUuid) {
        TemporaryDirectory temporary;
        const auto uuid = lfs::core::generate_uuid_v4();
        const auto dir = temporary.path / "tmp";
        const auto path = scratch_autosave_path(dir, uuid);
        EXPECT_EQ(path.parent_path(), dir);
        EXPECT_EQ(
            path.filename().string(),
            uuid.to_string() + ".licht");
        EXPECT_TRUE(is_scratch_autosave_path(path, dir));
        EXPECT_FALSE(is_scratch_autosave_path(
            dir / "not-a-uuid.licht", dir));
        EXPECT_FALSE(is_scratch_autosave_path(
            path, temporary.path));
    }

    TEST(ProjectRecoveryScratch, ScanFindsScratchCandidate) {
        TemporaryDirectory temporary;
        const auto dir = temporary.path / "tmp";
        fs::create_directories(dir);
        const auto uuid = lfs::core::generate_uuid_v4();
        const auto path = scratch_autosave_path(dir, uuid);
        create_single_chunk_fixture(
            path, 2001, 2002, 2003,
            fixed_key("SCNG", 2004),
            recoverable_scene_graph_payload());
        auto found = scan_scratch_autosaves(dir);
        ASSERT_EQ(found.size(), 1u);
        EXPECT_EQ(
            found[0].disposition,
            RecoveryDisposition::Offer);
        EXPECT_TRUE(found[0].untitled_scratch);
        ASSERT_TRUE(found[0].selected_path);
        EXPECT_EQ(
            found[0].selected_path->lexically_normal(),
            path.lexically_normal());
        EXPECT_FALSE(found[0].commit_uuid.is_nil());
    }

    TEST(ProjectRecoveryScratch, RemoveDeletesUnlockedScratch) {
        TemporaryDirectory temporary;
        const auto dir = temporary.path / "tmp";
        fs::create_directories(dir);
        const auto uuid = lfs::core::generate_uuid_v4();
        const auto path = scratch_autosave_path(dir, uuid);
        create_single_chunk_fixture(
            path, 2011, 2012, 2013,
            fixed_key("SCNG", 2014),
            recoverable_scene_graph_payload());
        auto removed = remove_scratch_autosave(path);
        ASSERT_TRUE(removed)
            << lfs::format_for_developer(removed.error());
        EXPECT_FALSE(fs::exists(path));
    }

    TEST(ProjectRecoveryScratch, SweepSkipsLiveLockedScratch) {
        TemporaryDirectory temporary;
        const auto dir = temporary.path / "tmp";
        fs::create_directories(dir);
        const auto uuid = lfs::core::generate_uuid_v4();
        const auto path = scratch_autosave_path(dir, uuid);
        create_single_chunk_fixture(
            path, 2021, 2022, 2023,
            fixed_key("SCNG", 2024),
            recoverable_scene_graph_payload());
        auto lease = WriterLockLease::acquire(path);
        ASSERT_TRUE(lease)
            << lfs::format_for_developer(lease.error());
        sweep_stale_scratch_autosaves(dir);
        EXPECT_TRUE(fs::is_regular_file(path));
        auto found = scan_scratch_autosaves(dir);
        EXPECT_TRUE(found.empty());
    }

    TEST(ProjectRecoveryScratch, SweepRemovesEmptyUnlockedScratch) {
        TemporaryDirectory temporary;
        const auto dir = temporary.path / "tmp";
        fs::create_directories(dir);
        const auto uuid = lfs::core::generate_uuid_v4();
        const auto path = scratch_autosave_path(dir, uuid);
        create_single_chunk_fixture(
            path, 2031, 2032, 2033,
            fixed_key("PROJ", 2034),
            R"({"scratch":"empty"})");
        ASSERT_TRUE(fs::is_regular_file(path));
        sweep_stale_scratch_autosaves(dir);
        EXPECT_FALSE(fs::exists(path));
        auto found = scan_scratch_autosaves(dir);
        EXPECT_TRUE(found.empty());
    }

    TEST(ProjectPathTest, IsPublishedLichtPathTable) {
        // Would fail if write/compact/backup/recovery temps or .autosave
        // components were treated as published masters, or if
        // myautosave.licht were rejected as a substring false positive.
        const fs::path parent{"/tmp/licht-path-table"};
        const struct {
            const char* relative;
            bool published;
            const char* derived;
        } rows[] = {
            {"project.licht", true, nullptr},
            {"session.LICHT", true, nullptr},
            {"myautosave.licht", true, nullptr},
            {"project.project-write.1.2.3.tmp.licht", false, "project.licht"},
            {"project.compact.1.tmp.licht", false, "project.licht"},
            {"project.replace-backup.1.tmp.licht", false, "project.licht"},
            {"project.recovery-session.uuid.tmp.licht", false, "project.licht"},
            {"project.licht.autosave", false, "project.licht"},
            {"project.licht.corrupt-12", false, "project.licht"},
            {"foo.autosave.licht", false, nullptr},
        };
        for (const auto& row : rows) {
            SCOPED_TRACE(row.relative);
            const fs::path path = parent / row.relative;
            EXPECT_EQ(isPublishedLichtPath(path), row.published);
            const auto derived = derivedPublishedMasterPath(path);
            if (row.derived == nullptr) {
                EXPECT_FALSE(derived.has_value());
            } else {
                ASSERT_TRUE(derived.has_value());
                EXPECT_EQ(derived->filename(), fs::path(row.derived));
            }
        }
    }

} // namespace
