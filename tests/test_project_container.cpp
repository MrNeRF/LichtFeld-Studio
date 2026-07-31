/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project/crc32c.hpp"
#include "io/project_container.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <nlohmann/json.hpp>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <limits.h>
#include <sched.h>
#include <sys/mount.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace std::string_view_literals;

    const fs::path FIXTURES =
        fs::path(PROJECT_ROOT_PATH) / "tools/licht_inspect/fixtures";
    constexpr std::uint64_t FIXED_CREATION_TIME_NS =
        1'735'689'600'000'000'000;
    constexpr std::uint64_t FIXED_COMMIT_TIME_NS =
        1'735'689'601'000'000'000;

    struct TemporaryDirectory {
        TemporaryDirectory() {
            static std::atomic_uint64_t counter{0};
            path = fs::temp_directory_path() /
                   std::format("lfs-project-container-{}-{}",
                               std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count(),
                               counter.fetch_add(1));
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        fs::path path;
    };

    lfs::core::Uuid fixed_uuid(const std::uint64_t tag) {
        const auto parsed = lfs::core::Uuid::from_string(
            std::format("{:08x}-0000-4000-8000-{:012x}", tag, tag));
        if (!parsed.has_value()) {
            std::abort();
        }
        return *parsed;
    }

    ChunkKey fixed_key(const std::string_view fourcc,
                       const std::uint64_t instance_tag) {
        const auto parsed_fourcc = Fourcc::from_string(fourcc);
        if (!parsed_fourcc.has_value()) {
            std::abort();
        }
        return ChunkKey{
            .fourcc = *parsed_fourcc,
            .instance_uuid = fixed_uuid(instance_tag),
        };
    }

    std::vector<std::byte> byte_vector(const std::string_view text) {
        const auto bytes = std::as_bytes(std::span(text.data(), text.size()));
        return {bytes.begin(), bytes.end()};
    }

    std::vector<std::byte> hex_bytes(const std::string_view text) {
        auto nibble = [](const char value) -> std::uint8_t {
            if (value >= '0' && value <= '9') {
                return static_cast<std::uint8_t>(value - '0');
            }
            if (value >= 'a' && value <= 'f') {
                return static_cast<std::uint8_t>(value - 'a' + 10);
            }
            std::abort();
        };
        if (text.size() % 2 != 0) {
            std::abort();
        }
        std::vector<std::byte> result(text.size() / 2);
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = static_cast<std::byte>(
                (nibble(text[index * 2]) << 4) | nibble(text[index * 2 + 1]));
        }
        return result;
    }

    std::vector<std::byte> read_file_bytes(const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        const std::vector<char> raw{std::istreambuf_iterator<char>(input),
                                    std::istreambuf_iterator<char>()};
        std::vector<std::byte> bytes(raw.size());
        std::memcpy(bytes.data(), raw.data(), raw.size());
        return bytes;
    }

    void write_file_bytes(const fs::path& path,
                          const std::span<const std::byte> bytes) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            throw std::runtime_error(
                std::format("failed to write {}", path.string()));
        }
    }

    void write_file_range(const fs::path& path, const std::uint64_t offset,
                          const std::span<const std::byte> bytes) {
        std::fstream output(path,
                            std::ios::binary | std::ios::in | std::ios::out);
        output.seekp(static_cast<std::streamoff>(offset));
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            throw std::runtime_error(
                std::format("failed to write {} at 0x{:x}", path.string(),
                            offset));
        }
    }

    template <typename T>
    T require_result(lfs::Result<T> result) {
        if (!result) {
            throw std::runtime_error(
                lfs::format_for_developer(result.error()));
        }
        return std::move(*result);
    }

    void require_status(lfs::Result<void> result) {
        if (!result) {
            throw std::runtime_error(
                lfs::format_for_developer(result.error()));
        }
    }

    CreateOptions fixture_create_options(const std::uint64_t file_tag) {
        return CreateOptions{
            .project_uuid = fixed_uuid(1),
            .file_uuid = fixed_uuid(file_tag),
            .role = ContainerRole::Master,
            .creation_time_unix_ns = FIXED_CREATION_TIME_NS,
            .index_compression =
                IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    CommitOptions fixture_commit_options(const std::uint64_t commit_tag,
                                         const std::uint64_t snapshot_tag,
                                         const std::uint64_t generation) {
        return CommitOptions{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(commit_tag),
            .snapshot_uuid = fixed_uuid(snapshot_tag),
            .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + generation,
        };
    }

    AppendOptions fixture_append_options() {
        return AppendOptions{
            .compatibility = {},
            .index_compression =
                IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    void put_u32(std::span<std::byte> bytes, const std::size_t offset,
                 const std::uint32_t value) {
        for (std::size_t index = 0; index < 4; ++index) {
            bytes[offset + index] =
                static_cast<std::byte>(value >> (index * 8));
        }
    }

    void put_u64(std::span<std::byte> bytes, const std::size_t offset,
                 const std::uint64_t value) {
        for (std::size_t index = 0; index < 8; ++index) {
            bytes[offset + index] =
                static_cast<std::byte>(value >> (index * 8));
        }
    }

    std::uint32_t crc_range(const std::span<const std::byte> bytes,
                            const std::size_t offset,
                            const std::size_t count) {
        return crc32c(0, bytes.data() + offset, count);
    }

    void refresh_generation_envelope(
        std::vector<std::byte>& bytes, const ProjectReader& reader,
        const std::span<const std::uint32_t> head_slots) {
        const CommitInfo& commit = reader.commit();
        const std::uint32_t index_crc = crc_range(
            bytes, static_cast<std::size_t>(commit.index_offset),
            static_cast<std::size_t>(commit.index_stored_bytes));
        put_u32(bytes, static_cast<std::size_t>(commit.offset + 160),
                index_crc);
        put_u32(bytes, static_cast<std::size_t>(commit.offset + 164),
                index_crc);
        const std::uint32_t commit_crc =
            crc_range(bytes, static_cast<std::size_t>(commit.offset), 252);
        put_u32(bytes, static_cast<std::size_t>(commit.offset + 252),
                commit_crc);
        for (const std::uint32_t slot : head_slots) {
            const std::size_t head =
                static_cast<std::size_t>(HEAD_SLOT_OFFSETS[slot]);
            put_u32(bytes, head + 104, commit_crc);
            put_u32(bytes, head + 4092, crc_range(bytes, head, 4092));
        }
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

    struct FixtureOutcome {
        const char* name;
        const char* expected;
        bool verify;
    };

    constexpr std::array FIXTURE_OUTCOMES = {
        FixtureOutcome{"autosave-master.licht", "open_gen_1", true},
        FixtureOutcome{"autosave-sidecar-stale-base.licht.autosave", "open_gen_1", true},
        FixtureOutcome{"autosave-sidecar-valid.licht.autosave", "open_gen_1", true},
        FixtureOutcome{"duplicate-slot-write.licht", "open_gen_1", true},
        FixtureOutcome{"minimal-valid-single-generation.licht", "open_gen_1", true},
        FixtureOutcome{"multi-generation-append.licht", "open_gen_2", true},
        FixtureOutcome{"orphan-tail.licht", "open_gen_1", true},
        FixtureOutcome{"out-of-bounds-index-row.licht", "repair_only", false},
        FixtureOutcome{"overlapping-rows.licht", "repair_only", false},
        FixtureOutcome{"preview-locator.licht", "open_gen_2", true},
        FixtureOutcome{"split-brain.licht", "hard_fail", false},
        FixtureOutcome{"tombstone.licht", "open_gen_2", true},
        FixtureOutcome{"torn-head.licht", "open_gen_1", true},
        FixtureOutcome{"unsupported-newer-higher-head.licht", "hard_fail", false},
        FixtureOutcome{"unsupported-newer-single-head.licht", "unsupported_newer", false},
        FixtureOutcome{"write-unsafe.licht", "open_gen_1", true},
    };

    TEST(ProjectContainerFormat, Crc32cKnownVector) {
        constexpr std::string_view CHECK = "123456789";
        EXPECT_EQ(crc32c(0, CHECK.data(), CHECK.size()), 0xe3069283u);
        EXPECT_EQ(crc32c(0, CHECK.data(), 0), 0u);
        EXPECT_EQ(crc32c(crc32c(0, CHECK.data(), 3), CHECK.data() + 3, 6),
                  0xe3069283u);
    }

    TEST(ProjectContainerReader, GoldenFixtureClassificationsAndVerification) {
        for (const auto& fixture : FIXTURE_OUTCOMES) {
            SCOPED_TRACE(fixture.name);
            const fs::path path = FIXTURES / fixture.name;
            const OpenClassification classification = ProjectReader::classify(path);
            EXPECT_EQ(classification.outcome_name(), fixture.expected)
                << classification.diagnostic;
            if (!fixture.verify) {
                continue;
            }
            auto reader = ProjectReader::open(path);
            ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
            auto verified = reader->verify_all();
            EXPECT_TRUE(verified)
                << (verified ? std::string{} : lfs::format_for_developer(verified.error()));
        }
    }

    TEST(ProjectContainerReader, SelectedGenerationAndCarriedRowsMatchGrammar) {
        auto reader = ProjectReader::open(FIXTURES / "multi-generation-append.licht");
        ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
        EXPECT_EQ(reader->selected_head().slot_id, 1u);
        EXPECT_EQ(reader->selected_head().head_sequence, 2u);
        EXPECT_EQ(reader->commit().generation, 2u);
        EXPECT_EQ(reader->chunks().size(), 2u);

        const auto proj_fourcc = Fourcc::from_string("PROJ");
        const auto splt_fourcc = Fourcc::from_string("SPLT");
        ASSERT_TRUE(proj_fourcc.has_value());
        ASSERT_TRUE(splt_fourcc.has_value());
        const auto proj_uuid =
            lfs::core::Uuid::from_string("0000012c-0000-4000-8000-00000000012c");
        const auto splt_uuid =
            lfs::core::Uuid::from_string("0000012e-0000-4000-8000-00000000012e");
        ASSERT_TRUE(proj_uuid.has_value());
        ASSERT_TRUE(splt_uuid.has_value());

        const ChunkInfo* proj = reader->find(*proj_fourcc, *proj_uuid);
        const ChunkInfo* splt = reader->find(*splt_fourcc, *splt_uuid);
        ASSERT_NE(proj, nullptr);
        ASSERT_NE(splt, nullptr);
        EXPECT_EQ(proj->source_generation, 1u);
        EXPECT_EQ(splt->source_generation, 2u);
        EXPECT_TRUE(splt->block_crc_table.has_value());
        EXPECT_EQ(splt->payload_offset % TENSOR_PAYLOAD_ALIGNMENT, 0u);
    }

    TEST(ProjectContainerReader, BoundedStreamAndMappedRangeStayInsidePayload) {
        auto reader =
            ProjectReader::open(FIXTURES / "minimal-valid-single-generation.licht");
        ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
        ASSERT_EQ(reader->chunks().size(), 1u);
        const ChunkInfo& chunk = reader->chunks().front();

        auto decoded = reader->read_chunk(chunk);
        ASSERT_TRUE(decoded) << lfs::format_for_developer(decoded.error());

        auto bounded = reader->open_bounded_stream(chunk);
        ASSERT_TRUE(bounded) << lfs::format_for_developer(bounded.error());
        std::vector<std::byte> streamed(decoded->size());
        bounded->stream().read(reinterpret_cast<char*>(streamed.data()),
                               static_cast<std::streamsize>(streamed.size()));
        EXPECT_EQ(bounded->stream().gcount(),
                  static_cast<std::streamsize>(streamed.size()));
        EXPECT_EQ(streamed, *decoded);
        bounded->stream().seekg(0, std::ios::end);
        EXPECT_EQ(bounded->stream().tellg(),
                  static_cast<std::streamoff>(decoded->size()));
        bounded->stream().seekg(1, std::ios::end);
        EXPECT_TRUE(bounded->stream().fail());

        auto mapped = reader->map_stored_range(chunk, 1, chunk.stored_bytes - 1);
        ASSERT_TRUE(mapped) << lfs::format_for_developer(mapped.error());
        EXPECT_EQ(mapped->file_offset(), chunk.payload_offset + 1);
        EXPECT_TRUE(std::equal(mapped->bytes().begin(), mapped->bytes().end(),
                               decoded->begin() + 1));

        auto out_of_bounds =
            reader->map_stored_range(chunk, chunk.stored_bytes, 1);
        EXPECT_FALSE(out_of_bounds);
        EXPECT_EQ(out_of_bounds.error().code(), lfs::ErrorCode::BoundsViolation);
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

    TEST(ProjectContainerReader,
         RejectsHostileStoredSizesAndSparsePaddingBeforeAllocation) {
        TemporaryDirectory temporary;
        const fs::path huge_index_path = temporary.path / "huge-index.licht";
        fs::copy_file(FIXTURES / "minimal-valid-single-generation.licht",
                      huge_index_path);
        ProjectReader original =
            require_result(ProjectReader::open(huge_index_path));
        std::vector<std::byte> hostile = read_file_bytes(huge_index_path);
        put_u64(hostile,
                static_cast<std::size_t>(original.commit().offset + 144),
                4ull * 1024 * 1024 * 1024);
        constexpr std::array<std::uint32_t, 1> SLOT_A = {0};
        refresh_generation_envelope(hostile, original, SLOT_A);
        write_file_bytes(huge_index_path, hostile);
        const OpenClassification huge_index =
            ProjectReader::classify(huge_index_path);
        EXPECT_EQ(huge_index.outcome_name(), "repair_only")
            << huge_index.diagnostic;

        const fs::path huge_padding_path =
            temporary.path / "huge-padding.licht";
        fs::copy_file(FIXTURES / "minimal-valid-single-generation.licht",
                      huge_padding_path);
        const std::vector<std::byte> source =
            read_file_bytes(huge_padding_path);
        const std::uint64_t sparse_commit_offset =
            5ull * 1024 * 1024 * 1024;
        const std::uint64_t sparse_end =
            sparse_commit_offset + COMMIT_RECORD_BYTES;
        std::array<std::byte, COMMIT_RECORD_BYTES> commit_raw{};
        std::copy_n(
            source.begin() +
                static_cast<std::size_t>(original.commit().offset),
            commit_raw.size(), commit_raw.begin());
        put_u64(commit_raw, 176, sparse_end);
        put_u32(commit_raw, 252,
                crc32c(0, commit_raw.data(), commit_raw.size() - 4));
        std::array<std::byte, HEAD_SLOT_BYTES> head_raw{};
        std::copy_n(source.begin() +
                        static_cast<std::size_t>(HEAD_SLOT_OFFSETS[0]),
                    head_raw.size(), head_raw.begin());
        put_u64(head_raw, 80, sparse_commit_offset);
        put_u64(head_raw, 96, sparse_end);
        put_u32(head_raw, 104,
                crc32c(0, commit_raw.data(), commit_raw.size() - 4));
        put_u32(head_raw, 4092,
                crc32c(0, head_raw.data(), head_raw.size() - 4));
        fs::resize_file(huge_padding_path, sparse_end);
        write_file_range(huge_padding_path, sparse_commit_offset, commit_raw);
        write_file_range(huge_padding_path, HEAD_SLOT_OFFSETS[0], head_raw);

        const OpenClassification huge_padding =
            ProjectReader::classify(huge_padding_path);
        EXPECT_EQ(huge_padding.outcome_name(), "repair_only")
            << huge_padding.diagnostic;
    }

    TEST(ProjectContainerReader, PreviewLocatorAndConvenienceReadMatchThmbRow) {
        ProjectReader reader = require_result(
            ProjectReader::open(FIXTURES / "preview-locator.licht"));
        ASSERT_TRUE(reader.preview().has_value());
        EXPECT_EQ(reader.preview()->format, PreviewFormat::Png);
        EXPECT_LE(reader.preview()->bytes, MAX_PREVIEW_BYTES);
        const auto preview = require_result(reader.read_preview());
        ASSERT_GE(preview.size(), 8u);
        const auto png_signature =
            hex_bytes("89504e470d0a1a0a");
        EXPECT_TRUE(std::equal(png_signature.begin(), png_signature.end(),
                               preview.begin()));
        const auto matching = std::find_if(
            reader.chunks().begin(), reader.chunks().end(),
            [&](const ChunkInfo& row) {
                return row.key.fourcc == FOURCC_THMB &&
                       row.payload_offset == reader.preview()->offset &&
                       row.stored_bytes == reader.preview()->bytes;
            });
        ASSERT_NE(matching, reader.chunks().end());
        EXPECT_EQ(matching->compression, Compression::Stored);

        ProjectReader absent = require_result(ProjectReader::open(
            FIXTURES / "minimal-valid-single-generation.licht"));
        EXPECT_FALSE(absent.preview().has_value());
        auto missing = absent.read_preview();
        EXPECT_FALSE(missing);
        EXPECT_EQ(missing.error().code(), lfs::ErrorCode::NotFound);
    }

    TEST(ProjectContainerReader, UnsupportedAuthorityIsInspectOnly) {
        ReaderOptions options;
        options.allow_unsupported_inspection = true;
        auto reader =
            ProjectReader::open(FIXTURES / "unsupported-newer-single-head.licht",
                                options);
        ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
        EXPECT_EQ(reader->open_state(), OpenState::UnsupportedNewer);
        ASSERT_FALSE(reader->chunks().empty());
        auto payload = reader->read_chunk(reader->chunks().front());
        EXPECT_FALSE(payload);
        EXPECT_EQ(payload.error().code(), lfs::ErrorCode::Unsupported);
    }

    TEST(ProjectContainerReader, WriteSafetyUsesIndependentVersionAndCapabilityGates) {
        auto reader = ProjectReader::open(FIXTURES / "write-unsafe.licht");
        ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
        const WriteCompatibility compatibility = reader->write_compatibility();
        EXPECT_FALSE(compatibility.safe);
        EXPECT_EQ(compatibility.reasons.size(), 2u);
    }

    TEST(ProjectContainerReader, FullMaterializedOracleCorpusParity) {
        const char* manifest_environment =
            std::getenv("LFS_LICHT_ORACLE_MANIFEST");
        if (manifest_environment == nullptr ||
            std::string_view(manifest_environment).empty()) {
            GTEST_SKIP()
                << "set LFS_LICHT_ORACLE_MANIFEST to a materialized "
                   "oracle_corpus.py manifest";
        }
        const fs::path manifest_path = manifest_environment;
        std::ifstream input(manifest_path);
        ASSERT_TRUE(input) << manifest_path;
        const nlohmann::json manifest = nlohmann::json::parse(input);
        ASSERT_EQ(manifest.at("schema").get<int>(), 1);
        const auto& cases = manifest.at("cases");
        ASSERT_EQ(cases.size(), manifest.at("case_count").get<std::size_t>());
        ASSERT_GE(cases.size(), 10'000u);

        for (const auto& record : cases) {
            const std::string case_id =
                record.at("case_id").get<std::string>();
            SCOPED_TRACE(case_id);
            const fs::path path =
                manifest_path.parent_path() /
                record.at("mutated_file").get<std::string>();
            const std::string expected =
                record.at("expected_outcome").get<std::string>();
            const OpenClassification actual =
                ProjectReader::classify(path);
            EXPECT_EQ(actual.outcome_name(), expected)
                << actual.diagnostic;
        }
    }

    TEST(ProjectContainerWriter, ReproducesMinimalGoldenFixtureByteForByte) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "minimal.licht";
        CreateOptions create{
            .project_uuid = fixed_uuid(1),
            .file_uuid = fixed_uuid(10),
            .role = ContainerRole::Master,
            .creation_time_unix_ns = FIXED_CREATION_TIME_NS,
            .index_compression =
                IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
        auto writer = ProjectWriter::create(path, create);
        ASSERT_TRUE(writer) << lfs::format_for_developer(writer.error());
        const auto payload =
            byte_vector(R"({"fixture":"minimal","generation":1})");
        CommitOptions commit{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(100),
            .snapshot_uuid = fixed_uuid(200),
            .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 1,
        };
        auto planned = writer->plan_commit(commit);
        ASSERT_TRUE(planned)
            << lfs::format_for_developer(planned.error());
        auto preflight = writer->preflight(payload.size());
        ASSERT_TRUE(preflight)
            << lfs::format_for_developer(preflight.error());
        auto write = writer->write_chunk(fixed_key("PROJ", 300), payload);
        ASSERT_TRUE(write) << lfs::format_for_developer(write.error());
        auto published = writer->commit();
        ASSERT_TRUE(published)
            << lfs::format_for_developer(published.error());

        EXPECT_EQ(read_file_bytes(path),
                  read_file_bytes(
                      FIXTURES / "minimal-valid-single-generation.licht"));
    }

    TEST(ProjectContainerWriter, ReproducesCompleteGoldenCatalogueByteForByte) {
        TemporaryDirectory temporary;
        std::unordered_map<std::string, std::vector<std::byte>> generated;

        const fs::path minimal_path = temporary.path / "minimal.licht";
        generated["minimal-valid-single-generation.licht"] =
            create_single_chunk_fixture(
                minimal_path, 10, 100, 200, fixed_key("PROJ", 300),
                R"({"fixture":"minimal","generation":1})");

        const fs::path multi_path = temporary.path / "multi.licht";
        create_single_chunk_fixture(
            multi_path, 11, 101, 201, fixed_key("PROJ", 300),
            R"({"fixture":"multi","generation":1})");
        {
            ProjectReader prior =
                require_result(ProjectReader::open(multi_path));
            const ChunkInfo* proj =
                prior.find(fixed_key("PROJ", 300));
            ASSERT_NE(proj, nullptr);
            CleanProof proof =
                require_result(prior.make_clean_proof(*proj, 17));
            ProjectWriter writer = require_result(ProjectWriter::append(
                multi_path, fixture_append_options()));
            const auto splt_payload =
                byte_vector("fixture tensor payload\0\1\2"sv);
            require_status(writer.plan_commit(
                fixture_commit_options(102, 202, 2)));
            require_status(writer.preflight(splt_payload.size()));
            require_status(writer.reuse_if_clean(proof, 17));
            ChunkWriteOptions chunk_options{
                .chunk_version = 1,
                .compression = Compression::Stored,
                .tensor_payload = true,
                .block_crcs = true,
            };
            require_status(writer.write_chunk(fixed_key("SPLT", 302),
                                              splt_payload, chunk_options));
            require_status(writer.commit());
        }
        generated["multi-generation-append.licht"] =
            read_file_bytes(multi_path);

        const fs::path preview_path = temporary.path / "preview.licht";
        create_single_chunk_fixture(
            preview_path, 23, 118, 217, fixed_key("PROJ", 300),
            R"({"fixture":"preview","generation":1})");
        {
            ProjectReader prior =
                require_result(ProjectReader::open(preview_path));
            const ChunkInfo* proj = prior.find(fixed_key("PROJ", 300));
            ASSERT_NE(proj, nullptr);
            CleanProof proof =
                require_result(prior.make_clean_proof(*proj, 29));
            ProjectWriter writer = require_result(ProjectWriter::append(
                preview_path, fixture_append_options()));
            const auto png = hex_bytes(
                "89504e470d0a1a0a0000000d4948445200000001000000010804000000"
                "b51c0c020000000b4944415478da6364f80f00010501012718e3660000"
                "000049454e44ae426082");
            require_status(writer.plan_commit(
                fixture_commit_options(119, 218, 2)));
            require_status(writer.preflight(png.size()));
            require_status(writer.reuse_if_clean(proof, 29));
            require_status(writer.set_preview(png));
            require_status(writer.commit());
        }
        generated["preview-locator.licht"] =
            read_file_bytes(preview_path);

        const fs::path tombstone_path =
            temporary.path / "tombstone.licht";
        create_single_chunk_fixture(
            tombstone_path, 12, 103, 203, fixed_key("VIEW", 301),
            R"({"fixture":"tombstone","generation":1})");
        {
            ProjectWriter writer = require_result(ProjectWriter::append(
                tombstone_path, fixture_append_options()));
            require_status(writer.plan_commit(
                fixture_commit_options(104, 204, 2)));
            require_status(writer.preflight(0));
            require_status(writer.erase(fixed_key("VIEW", 301)));
            require_status(writer.commit());
        }
        generated["tombstone.licht"] = read_file_bytes(tombstone_path);

        std::vector<std::byte> orphan =
            generated.at("minimal-valid-single-generation.licht");
        constexpr std::string_view ORPHAN_MARKER =
            "ORPHAN-TAIL-NOT-AUTHORITY\0"sv;
        for (int repeat = 0; repeat < 3; ++repeat) {
            const auto marker = std::as_bytes(
                std::span(ORPHAN_MARKER.data(), ORPHAN_MARKER.size()));
            orphan.insert(orphan.end(), marker.begin(), marker.end());
        }
        generated["orphan-tail.licht"] = std::move(orphan);

        std::vector<std::byte> torn =
            generated.at("multi-generation-append.licht");
        torn[static_cast<std::size_t>(HEAD_SLOT_OFFSETS[1] + 4092)] ^=
            std::byte{1};
        generated["torn-head.licht"] = std::move(torn);

        const fs::path duplicate_path =
            temporary.path / "duplicate.licht";
        std::vector<std::byte> duplicate = create_single_chunk_fixture(
            duplicate_path, 13, 105, 205, fixed_key("PROJ", 300),
            R"({"fixture":"duplicate","generation":1})");
        {
            const std::size_t head_a =
                static_cast<std::size_t>(HEAD_SLOT_OFFSETS[0]);
            const std::size_t head_b =
                static_cast<std::size_t>(HEAD_SLOT_OFFSETS[1]);
            std::copy_n(duplicate.begin() + head_a,
                        static_cast<std::size_t>(HEAD_SLOT_BYTES),
                        duplicate.begin() + head_b);
            put_u32(duplicate, head_b + 8, 1);
            put_u32(duplicate, head_b + 4092,
                    crc_range(duplicate, head_b, 4092));
        }
        generated["duplicate-slot-write.licht"] =
            std::move(duplicate);

        const fs::path unsupported_single_path =
            temporary.path / "unsupported-single.licht";
        CommitOptions unsupported_single_commit =
            fixture_commit_options(114, 213, 1);
        unsupported_single_commit.min_reader_version = Version{1, 1};
        generated["unsupported-newer-single-head.licht"] =
            create_single_chunk_fixture(
                unsupported_single_path, 20, 114, 213,
                fixed_key("PROJ", 300),
                R"({"fixture":"unsupported-newer-single","generation":1})",
                &unsupported_single_commit);

        const fs::path write_unsafe_path =
            temporary.path / "write-unsafe.licht";
        CommitOptions write_unsafe_commit =
            fixture_commit_options(117, 216, 1);
        write_unsafe_commit.min_safe_writer_version = Version{1, 1};
        write_unsafe_commit.extra_writer_capabilities.set(8);
        generated["write-unsafe.licht"] = create_single_chunk_fixture(
            write_unsafe_path, 22, 117, 216,
            fixed_key("PROJ", 300),
            R"({"fixture":"write-unsafe","generation":1})",
            &write_unsafe_commit);

        const fs::path unsupported_higher_path =
            temporary.path / "unsupported-higher.licht";
        create_single_chunk_fixture(
            unsupported_higher_path, 21, 115, 214,
            fixed_key("PROJ", 300),
            R"({"fixture":"unsupported-higher","generation":1})");
        {
            ProjectReader prior =
                require_result(ProjectReader::open(unsupported_higher_path));
            const ChunkInfo* proj =
                prior.find(fixed_key("PROJ", 300));
            ASSERT_NE(proj, nullptr);
            CleanProof proof =
                require_result(prior.make_clean_proof(*proj, 23));
            ProjectWriter writer = require_result(ProjectWriter::append(
                unsupported_higher_path, fixture_append_options()));
            const auto payload = byte_vector(
                R"({"fixture":"unsupported-higher","generation":2})");
            CommitOptions commit = fixture_commit_options(116, 215, 2);
            commit.extra_reader_capabilities.set(8);
            require_status(writer.plan_commit(commit));
            require_status(writer.preflight(payload.size()));
            require_status(writer.reuse_if_clean(proof, 23));
            require_status(writer.write_chunk(fixed_key("VIEW", 301),
                                              payload));
            require_status(writer.commit());
        }
        generated["unsupported-newer-higher-head.licht"] =
            read_file_bytes(unsupported_higher_path);

        const fs::path master_path =
            temporary.path / "autosave-master.licht";
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                master_path, fixture_create_options(17)));
            const auto ckpt = byte_vector("base checkpoint bytes");
            const auto proj = byte_vector(
                R"({"fixture":"autosave-master","generation":1})");
            require_status(writer.plan_commit(
                fixture_commit_options(110, 210, 1)));
            require_status(writer.preflight(ckpt.size() + proj.size()));
            ChunkWriteOptions tensor_options{
                .chunk_version = 1,
                .compression = Compression::Stored,
                .tensor_payload = true,
            };
            require_status(writer.write_chunk(fixed_key("CKPT", 303), ckpt,
                                              tensor_options));
            require_status(
                writer.write_chunk(fixed_key("PROJ", 300), proj));
            require_status(writer.commit());
        }
        generated["autosave-master.licht"] = read_file_bytes(master_path);
        ProjectReader master =
            require_result(ProjectReader::open(master_path));

        auto create_sidecar = [&](const fs::path& path,
                                  const std::uint64_t file_tag,
                                  const lfs::core::Uuid& base_commit,
                                  const std::uint64_t sequence,
                                  const std::uint64_t commit_tag,
                                  const std::uint64_t snapshot_tag,
                                  const std::string_view checkpoint) {
            CreateOptions create{
                .project_uuid = fixed_uuid(1),
                .file_uuid = fixed_uuid(file_tag),
                .role = ContainerRole::AutosaveSidecar,
                .base_explicit_commit_uuid = base_commit,
                .autosave_sequence = sequence,
                .sidecar_snapshot_uuid = fixed_uuid(snapshot_tag),
                .creation_time_unix_ns = FIXED_CREATION_TIME_NS,
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            };
            ProjectWriter writer =
                require_result(ProjectWriter::create(path, create));
            const auto payload = byte_vector(checkpoint);
            CommitOptions commit =
                fixture_commit_options(commit_tag, snapshot_tag, 1);
            commit.kind = CommitKind::Autosave;
            require_status(writer.plan_commit(commit));
            require_status(writer.preflight(payload.size()));
            for (const ChunkInfo& row : master.chunks()) {
                require_status(writer.add_sidecar_base_reference(row));
            }
            ChunkWriteOptions tensor_options{
                .chunk_version = 1,
                .compression = Compression::Stored,
                .tensor_payload = true,
            };
            require_status(writer.write_chunk(fixed_key("CKPT", 303),
                                              payload, tensor_options));
            require_status(writer.commit());
            return read_file_bytes(path);
        };
        generated["autosave-sidecar-valid.licht.autosave"] =
            create_sidecar(
                temporary.path / "valid.licht.autosave", 18,
                fixed_uuid(110), 7, 111, 211,
                "autosaved checkpoint bytes");
        generated["autosave-sidecar-stale-base.licht.autosave"] =
            create_sidecar(
                temporary.path / "stale.licht.autosave", 19,
                fixed_uuid(113), 8, 112, 212,
                "stale autosave checkpoint");

        const fs::path out_of_bounds_path =
            temporary.path / "out-of-bounds.licht";
        create_single_chunk_fixture(
            out_of_bounds_path, 15, 108, 208,
            fixed_key("PROJ", 300),
            R"({"fixture":"oob","generation":1})");
        {
            ProjectReader reader =
                require_result(ProjectReader::open(out_of_bounds_path));
            std::vector<std::byte> bytes =
                read_file_bytes(out_of_bounds_path);
            const std::size_t row_offset = static_cast<std::size_t>(
                reader.commit().index_offset + INDEX_HEADER_BYTES);
            put_u64(bytes, row_offset + 40,
                    reader.commit().committed_file_end + 64);
            constexpr std::array<std::uint32_t, 1> SLOTS = {0};
            refresh_generation_envelope(bytes, reader, SLOTS);
            generated["out-of-bounds-index-row.licht"] =
                std::move(bytes);
        }

        const fs::path overlap_path =
            temporary.path / "overlap.licht";
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                overlap_path, fixture_create_options(16)));
            const auto first_payload = byte_vector(
                R"({"fixture":"overlap-a","generation":1})");
            const auto second_payload = byte_vector(
                R"({"fixture":"overlap-b","generation":1})");
            require_status(writer.plan_commit(
                fixture_commit_options(109, 209, 1)));
            require_status(writer.preflight(first_payload.size() +
                                            second_payload.size()));
            require_status(writer.write_chunk(fixed_key("PROJ", 300),
                                              first_payload));
            require_status(writer.write_chunk(fixed_key("VIEW", 301),
                                              second_payload));
            require_status(writer.commit());
        }
        {
            ProjectReader reader =
                require_result(ProjectReader::open(overlap_path));
            const ChunkInfo* first =
                reader.find(fixed_key("PROJ", 300));
            const ChunkInfo* second =
                reader.find(fixed_key("VIEW", 301));
            ASSERT_NE(first, nullptr);
            ASSERT_NE(second, nullptr);
            std::vector<std::byte> bytes = read_file_bytes(overlap_path);
            const std::uint64_t expanded_size =
                second->payload_offset + second->stored_bytes -
                first->payload_offset;
            const std::uint32_t expanded_crc = crc_range(
                bytes, static_cast<std::size_t>(first->payload_offset),
                static_cast<std::size_t>(expanded_size));
            put_u64(bytes,
                    static_cast<std::size_t>(first->header_offset + 32),
                    expanded_size);
            put_u64(bytes,
                    static_cast<std::size_t>(first->header_offset + 40),
                    expanded_size);
            put_u32(bytes,
                    static_cast<std::size_t>(first->header_offset + 56),
                    expanded_crc);
            const std::uint32_t header_crc = crc_range(
                bytes, static_cast<std::size_t>(first->header_offset), 60);
            put_u32(bytes,
                    static_cast<std::size_t>(first->header_offset + 60),
                    header_crc);
            const auto row_position =
                static_cast<std::size_t>(std::distance(
                    reader.chunks().begin(),
                    std::find_if(reader.chunks().begin(),
                                 reader.chunks().end(),
                                 [&](const ChunkInfo& row) {
                                     return row.key == first->key;
                                 })));
            const std::size_t row_offset = static_cast<std::size_t>(
                reader.commit().index_offset + INDEX_HEADER_BYTES +
                row_position * INDEX_ROW_BYTES);
            put_u64(bytes, row_offset + 48, expanded_size);
            put_u64(bytes, row_offset + 56, expanded_size);
            put_u32(bytes, row_offset + 72, expanded_crc);
            put_u32(bytes, row_offset + 76, header_crc);
            constexpr std::array<std::uint32_t, 1> SLOTS = {0};
            refresh_generation_envelope(bytes, reader, SLOTS);
            generated["overlapping-rows.licht"] = std::move(bytes);
        }

        const fs::path split_path = temporary.path / "split.licht";
        create_single_chunk_fixture(
            split_path, 14, 106, 206, fixed_key("PROJ", 300),
            R"({"fixture":"split-a","generation":1})");
        {
            ProjectWriter writer = require_result(ProjectWriter::append(
                split_path, fixture_append_options()));
            const auto payload = byte_vector(
                R"({"fixture":"split-b","generation":1})");
            require_status(writer.plan_commit(
                fixture_commit_options(107, 207, 1)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.erase(fixed_key("PROJ", 300)));
            require_status(
                writer.write_chunk(fixed_key("VIEW", 301), payload));
            require_status(writer.commit());
        }
        {
            ProjectReader reader =
                require_result(ProjectReader::open(split_path));
            const auto& rows = reader.chunks();
            const auto view_position = std::find_if(
                rows.begin(), rows.end(), [](const ChunkInfo& row) {
                    return row.key == fixed_key("VIEW", 301);
                });
            ASSERT_NE(view_position, rows.end());
            const std::size_t view_index =
                static_cast<std::size_t>(
                    std::distance(rows.begin(), view_position));
            const std::vector<std::byte> source =
                read_file_bytes(split_path);
            const std::size_t index_offset =
                static_cast<std::size_t>(reader.commit().index_offset);
            std::vector<std::byte> split(source.begin(),
                                         source.begin() + index_offset);
            split.insert(split.end(), source.begin() + index_offset,
                         source.begin() + index_offset + INDEX_HEADER_BYTES);
            const std::size_t source_row =
                index_offset + INDEX_HEADER_BYTES +
                view_index * INDEX_ROW_BYTES;
            split.insert(split.end(), source.begin() + source_row,
                         source.begin() + source_row + INDEX_ROW_BYTES);
            put_u64(split, index_offset + 16, 1);
            put_u64(split, index_offset + 24, 1);
            put_u32(split, index_offset + 48, 0);
            put_u64(split, index_offset + INDEX_HEADER_BYTES + 64, 1);
            while (split.size() % CHUNK_ALIGNMENT != 0) {
                split.push_back(std::byte{0});
            }
            const std::size_t commit_offset = split.size();
            const std::size_t source_commit =
                static_cast<std::size_t>(reader.commit().offset);
            split.insert(split.end(), source.begin() + source_commit,
                         source.begin() + source_commit +
                             COMMIT_RECORD_BYTES);
            put_u64(split, commit_offset + 64, 1);
            std::fill(split.begin() + commit_offset + 72,
                      split.begin() + commit_offset + 96, std::byte{0});
            put_u64(split, commit_offset + 128,
                    FIXED_COMMIT_TIME_NS + 1);
            put_u64(split, commit_offset + 136, index_offset);
            put_u64(split, commit_offset + 144,
                    INDEX_HEADER_BYTES + INDEX_ROW_BYTES);
            put_u64(split, commit_offset + 152,
                    INDEX_HEADER_BYTES + INDEX_ROW_BYTES);
            const std::uint32_t index_crc = crc_range(
                split, index_offset,
                INDEX_HEADER_BYTES + INDEX_ROW_BYTES);
            put_u32(split, commit_offset + 160, index_crc);
            put_u32(split, commit_offset + 164, index_crc);
            const std::uint64_t committed_end =
                commit_offset + COMMIT_RECORD_BYTES;
            put_u64(split, commit_offset + 176, committed_end);
            std::fill(split.begin() + commit_offset + 192,
                      split.begin() + commit_offset + 208, std::byte{0});
            const std::uint32_t commit_crc =
                crc_range(split, commit_offset, 252);
            put_u32(split, commit_offset + 252, commit_crc);

            const std::size_t head_b =
                static_cast<std::size_t>(HEAD_SLOT_OFFSETS[1]);
            put_u64(split, head_b + 16, 1);
            put_u64(split, head_b + 24, 1);
            put_u64(split, head_b + 80, commit_offset);
            put_u64(split, head_b + 96, committed_end);
            put_u32(split, head_b + 104, commit_crc);
            put_u32(split, head_b + 4092,
                    crc_range(split, head_b, 4092));
            generated["split-brain.licht"] = std::move(split);
        }

        for (const auto& [name, bytes] : generated) {
            SCOPED_TRACE(name);
            EXPECT_EQ(bytes, read_file_bytes(FIXTURES / name));
        }
        EXPECT_EQ(generated.size(), FIXTURE_OUTCOMES.size());
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
         PreviewCarriesWithCleanProofAndSurvivesCompaction) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "preview-carry.licht";
        fs::copy_file(FIXTURES / "preview-locator.licht", path);
        ProjectReader prior = require_result(ProjectReader::open(path));
        ASSERT_TRUE(prior.preview().has_value());
        const PreviewLocator prior_locator = *prior.preview();
        const auto expected_preview = require_result(prior.read_preview());
        const ChunkInfo* proj = prior.find(fixed_key("PROJ", 300));
        const auto thmb = std::find_if(
            prior.chunks().begin(), prior.chunks().end(),
            [&](const ChunkInfo& row) {
                return row.key.fourcc == FOURCC_THMB &&
                       row.payload_offset == prior_locator.offset &&
                       row.stored_bytes == prior_locator.bytes;
            });
        ASSERT_NE(proj, nullptr);
        ASSERT_NE(thmb, prior.chunks().end());
        CleanProof proj_proof =
            require_result(prior.make_clean_proof(*proj, 51));
        CleanProof thmb_proof =
            require_result(prior.make_clean_proof(*thmb, 52));

        {
            ProjectWriter writer = require_result(ProjectWriter::append(
                path, fixture_append_options()));
            const auto payload = byte_vector("metadata-only preview carry");
            require_status(writer.plan_commit(
                fixture_commit_options(826, 827, 3)));
            require_status(writer.preflight(payload.size()));
            require_status(writer.reuse_if_clean(proj_proof, 51));
            require_status(writer.reuse_if_clean(thmb_proof, 52));
            require_status(
                writer.write_chunk(fixed_key("VIEW", 828), payload));
            require_status(writer.commit());
        }
        ProjectReader carried = require_result(ProjectReader::open(path));
        ASSERT_TRUE(carried.preview().has_value());
        EXPECT_EQ(*carried.preview(), prior_locator);
        EXPECT_EQ(require_result(carried.read_preview()), expected_preview);

        CompactionOptions compaction{
            .new_file_uuid = fixed_uuid(829),
            .commit_uuid = fixed_uuid(830),
            .snapshot_uuid = fixed_uuid(831),
            .creation_time_unix_ns = FIXED_CREATION_TIME_NS + 20,
            .wallclock_unix_ns = FIXED_COMMIT_TIME_NS + 20,
            .disk_reserve_bytes = 0,
        };
        require_status(ProjectWriter::compact(path, compaction));
        ProjectReader compacted = require_result(ProjectReader::open(path));
        ASSERT_TRUE(compacted.preview().has_value());
        EXPECT_EQ(compacted.preview()->bytes, prior_locator.bytes);
        EXPECT_EQ(compacted.preview()->format, PreviewFormat::Png);
        EXPECT_NE(compacted.preview()->offset, prior_locator.offset);
        EXPECT_EQ(require_result(compacted.read_preview()), expected_preview);
    }

    TEST(ProjectContainerWriter,
         AppendPostPublishVerificationFailureReturnsPublishedDataLoss) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "post-publish.licht";
        fs::copy_file(FIXTURES / "minimal-valid-single-generation.licht",
                      path);
        ProjectReader prior = require_result(ProjectReader::open(path));
        const ChunkInfo* proj = prior.find(fixed_key("PROJ", 300));
        ASSERT_NE(proj, nullptr);
        CleanProof proof =
            require_result(prior.make_clean_proof(*proj, 61));
        const std::uint64_t new_payload_offset =
            prior.commit().committed_file_end + CHUNK_HEADER_BYTES;
        AppendOptions append = fixture_append_options();
        append.boundary_observer =
            [&](const CommitBoundary boundary) {
                if (boundary == CommitBoundary::HeadFlushed) {
                    const std::array corruption = {std::byte{0xff}};
                    write_file_range(path, new_payload_offset, corruption);
                }
            };
        ProjectWriter writer =
            require_result(ProjectWriter::append(path, append));
        const auto payload = byte_vector("published then corrupted by hook");
        require_status(
            writer.plan_commit(fixture_commit_options(832, 833, 2)));
        require_status(writer.preflight(payload.size()));
        require_status(writer.reuse_if_clean(proof, 61));
        require_status(
            writer.write_chunk(fixed_key("VIEW", 834), payload));
        auto published = writer.commit();
        ASSERT_FALSE(published);
        EXPECT_EQ(published.error().code(), lfs::ErrorCode::DataLoss);
        const std::string diagnostic =
            lfs::format_for_developer(published.error());
        EXPECT_NE(diagnostic.find("commit.post_publish_verification"),
                  std::string::npos);
        EXPECT_NE(std::string(published.error().detail()).find("generation=2"),
                  std::string::npos);
        EXPECT_NE(std::string(published.error().detail())
                      .find(fixed_uuid(832).to_string()),
                  std::string::npos);
        EXPECT_NE(std::string(published.error().detail())
                      .find("committed_file_end="),
                  std::string::npos);

        ProjectReader current = require_result(ProjectReader::open(path));
        EXPECT_EQ(current.commit().generation, 2u);
        EXPECT_FALSE(current.verify_all());
        auto retry = writer.commit();
        ASSERT_FALSE(retry);
        EXPECT_EQ(retry.error().code(),
                  lfs::ErrorCode::FailedPrecondition);
    }

    TEST(ProjectContainerWriter, RefuseWriteMatrixMutatesNoProjectBytes) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "write-unsafe.licht";
        fs::copy_file(FIXTURES / "write-unsafe.licht", path);
        const std::vector<std::byte> before = read_file_bytes(path);

        auto attempt = [&](const ReaderOptions& compatibility) {
            AppendOptions append = fixture_append_options();
            append.compatibility = compatibility;
            auto writer = ProjectWriter::append(path, append);
            EXPECT_FALSE(writer);
            if (!writer) {
                EXPECT_EQ(writer.error().code(),
                          lfs::ErrorCode::Unsupported);
            }
            EXPECT_EQ(read_file_bytes(path), before);
        };

        ReaderOptions default_options;
        attempt(default_options);

        ReaderOptions version_only;
        version_only.writer_version = Version{1, 1};
        attempt(version_only);

        ReaderOptions capability_only;
        capability_only.writer_capabilities.set(8);
        attempt(capability_only);

        ReaderOptions both;
        both.writer_version = Version{1, 1};
        both.writer_capabilities.set(8);
        {
            AppendOptions append = fixture_append_options();
            append.compatibility = both;
            auto allowed = ProjectWriter::append(path, append);
            ASSERT_TRUE(allowed)
                << lfs::format_for_developer(allowed.error());
        }
        EXPECT_EQ(read_file_bytes(path), before);

        auto compacted = ProjectWriter::compact(path);
        ASSERT_FALSE(compacted);
        EXPECT_EQ(compacted.error().code(), lfs::ErrorCode::Unsupported);
        EXPECT_EQ(read_file_bytes(path), before);
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
        EXPECT_TRUE(fs::exists(fs::path(path.string() + ".lock")));

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
                .compression = Compression::Zstd,
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
        EXPECT_EQ(held.commit().index_compression, Compression::Zstd);
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
            .keep_tombstones = false,
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
        if (const char* output =
                std::getenv("LFS_LICHT_PRODUCTION_OUTPUT");
            output != nullptr && std::string_view(output).size() != 0) {
            fs::copy_file(path, output,
                          fs::copy_options::overwrite_existing);
        }
    }

#ifndef _WIN32
    TEST(ProjectContainerWriter, ProcessKillCrashMatrixPublishesOnlyOldOrNew) {
        TemporaryDirectory temporary;
        for (int boundary_value =
                 static_cast<int>(CommitBoundary::CurrentHeadValidated);
             boundary_value <=
             static_cast<int>(CommitBoundary::HeadFlushed);
             ++boundary_value) {
            const auto target =
                static_cast<CommitBoundary>(boundary_value);
            SCOPED_TRACE(std::format("boundary {}", boundary_value));
            const fs::path path =
                temporary.path /
                std::format("crash-{}.licht", boundary_value);
            fs::copy_file(
                FIXTURES / "minimal-valid-single-generation.licht", path);

            const pid_t child = ::fork();
            ASSERT_GE(child, 0);
            if (child == 0) {
                try {
                    ProjectReader prior =
                        require_result(ProjectReader::open(path));
                    const ChunkInfo* proj =
                        prior.find(fixed_key("PROJ", 300));
                    if (proj == nullptr) {
                        ::_exit(101);
                    }
                    CleanProof proof =
                        require_result(prior.make_clean_proof(*proj, 1));
                    AppendOptions append = fixture_append_options();
                    append.boundary_observer =
                        [target](const CommitBoundary reached) {
                            if (reached == target) {
                                ::kill(::getpid(), SIGKILL);
                            }
                        };
                    ProjectWriter writer = require_result(
                        ProjectWriter::append(path, append));
                    const auto payload =
                        byte_vector("crash-boundary-payload");
                    require_status(writer.plan_commit(
                        fixture_commit_options(901, 1001, 2)));
                    require_status(writer.preflight(payload.size()));
                    require_status(writer.reuse_if_clean(proof, 1));
                    require_status(writer.write_chunk(
                        fixed_key("VIEW", 801), payload));
                    require_status(writer.commit());
                } catch (...) {
                    ::_exit(102);
                }
                ::_exit(103);
            }

            int status = 0;
            ASSERT_EQ(::waitpid(child, &status, 0), child);
            ASSERT_TRUE(WIFSIGNALED(status));
            EXPECT_EQ(WTERMSIG(status), SIGKILL);
            const OpenClassification classification =
                ProjectReader::classify(path);
            const std::string outcome = classification.outcome_name();
            if (target <= CommitBoundary::AppendFlushed) {
                EXPECT_EQ(outcome, "open_gen_1")
                    << classification.diagnostic;
            } else if (target == CommitBoundary::HeadWritten) {
                EXPECT_TRUE(outcome == "open_gen_1" ||
                            outcome == "open_gen_2")
                    << classification.diagnostic;
            } else {
                EXPECT_EQ(outcome, "open_gen_2")
                    << classification.diagnostic;
            }
        }
    }

    TEST(ProjectContainerWriter, RealEnospcTmpfsChild) {
        const char* enabled = std::getenv("LFS_RUN_ENOSPC_CHILD");
        if (enabled == nullptr || std::string_view(enabled) != "1") {
            GTEST_SKIP() << "invoked only through the isolated tmpfs parent";
        }
        TemporaryDirectory mountpoint;
        ASSERT_EQ(::mount("tmpfs", mountpoint.path.c_str(), "tmpfs", 0,
                          "size=16m,mode=0700"),
                  0)
            << std::strerror(errno);
        struct UnmountGuard {
            fs::path path;
            ~UnmountGuard() { ::umount2(path.c_str(), MNT_DETACH); }
        } unmount{mountpoint.path};

        const fs::path path = mountpoint.path / "disk-full.licht";
        fs::copy_file(
            FIXTURES / "minimal-valid-single-generation.licht", path);
        {
            ProjectReader prior =
                require_result(ProjectReader::open(path));
            const ChunkInfo* proj =
                prior.find(fixed_key("PROJ", 300));
            ASSERT_NE(proj, nullptr);
            CleanProof proof =
                require_result(prior.make_clean_proof(*proj, 3));
            ProjectWriter writer = require_result(
                ProjectWriter::append(path, fixture_append_options()));
            require_status(writer.plan_commit(
                fixture_commit_options(902, 1002, 2)));
            require_status(writer.preflight(0));
            require_status(writer.reuse_if_clean(proof, 3));
            const std::vector<std::byte> payload(32 * 1024 * 1024,
                                                 std::byte{0x5a});
            auto write =
                writer.write_chunk(fixed_key("VIEW", 802), payload);
            ASSERT_FALSE(write);
            EXPECT_EQ(write.error().code(),
                      lfs::ErrorCode::ResourceExhausted);
            const auto retry_payload = byte_vector("must not write");
            auto poisoned_write = writer.write_chunk(
                fixed_key("GUIL", 803), retry_payload);
            ASSERT_FALSE(poisoned_write);
            EXPECT_EQ(poisoned_write.error().code(),
                      lfs::ErrorCode::FailedPrecondition);
            auto poisoned_commit = writer.commit();
            ASSERT_FALSE(poisoned_commit);
            EXPECT_EQ(poisoned_commit.error().code(),
                      lfs::ErrorCode::FailedPrecondition);
            const OpenClassification classification =
                ProjectReader::classify(path);
            EXPECT_EQ(classification.outcome_name(), "open_gen_1")
                << classification.diagnostic;
            ProjectReader intact =
                require_result(ProjectReader::open(path));
            require_status(intact.verify_all());
        }
    }

    TEST(ProjectContainerWriter, RealEnospcUsesIsolatedTmpfs) {
        char executable[PATH_MAX + 1]{};
        const ssize_t length =
            ::readlink("/proc/self/exe", executable, PATH_MAX);
        ASSERT_GT(length, 0);
        executable[length] = '\0';
        const pid_t child = ::fork();
        ASSERT_GE(child, 0);
        if (child == 0) {
            ::execlp(
                "unshare", "unshare", "--user", "--map-root-user",
                "--mount", "--fork", "env", "LFS_RUN_ENOSPC_CHILD=1",
                executable,
                "--gtest_filter=ProjectContainerWriter.RealEnospcTmpfsChild",
                "--gtest_color=no", nullptr);
            ::_exit(127);
        }
        int status = 0;
        ASSERT_EQ(::waitpid(child, &status, 0), child);
        ASSERT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
#endif

} // namespace
