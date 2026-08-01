/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project/crc32c.hpp"
#include "io/project_container.hpp"
#include "io/project_recovery.hpp"
#include "licht_test_support.hpp"

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
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
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
    using namespace lfs::test::licht;
    using namespace std::string_view_literals;

    const fs::path FIXTURES =
        fs::path(PROJECT_ROOT_PATH) / "tests/fixtures/licht";
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

    void refresh_generation_envelope(
        std::vector<std::byte>& bytes, const ProjectReader& reader,
        const std::span<const std::uint32_t> head_slots) {
        const CommitInfo& commit = reader.commit();
        const std::uint32_t index_crc = crc_range(
            bytes, static_cast<std::size_t>(commit.index_offset),
            static_cast<std::size_t>(commit.index_stored_bytes));
        write_u32_le(bytes, static_cast<std::size_t>(commit.offset + 160),
                     index_crc);
        write_u32_le(bytes, static_cast<std::size_t>(commit.offset + 164),
                     index_crc);
        const std::uint32_t commit_crc =
            crc_range(bytes, static_cast<std::size_t>(commit.offset), 252);
        write_u32_le(bytes, static_cast<std::size_t>(commit.offset + 252),
                     commit_crc);
        for (const std::uint32_t slot : head_slots) {
            const std::size_t head =
                static_cast<std::size_t>(HEAD_SLOT_OFFSETS[slot]);
            write_u32_le(bytes, head + 104, commit_crc);
            write_u32_le(bytes, head + 4092, crc_range(bytes, head, 4092));
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

    TEST(ProjectContainerReader, WriteSafetyReportsBothIndependentGates) {
        auto reader = ProjectReader::open(FIXTURES / "write-unsafe.licht");
        ASSERT_TRUE(reader) << lfs::format_for_developer(reader.error());
        const auto compatibility = reader->write_compatibility();
        EXPECT_FALSE(compatibility.safe);
        EXPECT_EQ(compatibility.reasons.size(), 2u)
            << "the version and capability gates must refuse independently";
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
        write_u64_le(hostile,
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
        write_u64_le(commit_raw, 176, sparse_end);
        write_u32_le(commit_raw, 252,
                     crc32c(0, commit_raw.data(), commit_raw.size() - 4));
        std::array<std::byte, HEAD_SLOT_BYTES> head_raw{};
        std::copy_n(source.begin() +
                        static_cast<std::size_t>(HEAD_SLOT_OFFSETS[0]),
                    head_raw.size(), head_raw.begin());
        write_u64_le(head_raw, 80, sparse_commit_offset);
        write_u64_le(head_raw, 96, sparse_end);
        write_u32_le(head_raw, 104,
                     crc32c(0, commit_raw.data(), commit_raw.size() - 4));
        write_u32_le(head_raw, 4092,
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
        constexpr std::array png_signature{
            std::byte{0x89}, std::byte{0x50}, std::byte{0x4e}, std::byte{0x47},
            std::byte{0x0d}, std::byte{0x0a}, std::byte{0x1a}, std::byte{0x0a}};
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

    TEST(ProjectContainerReader,
         PreviewLocatorMismatchAndWrongCompressionInvalidateOnlyThePublishingSlot) {
        TemporaryDirectory temporary;
        const auto source = FIXTURES / "preview-locator.licht";
        const auto original = require_result(ProjectReader::open(source));
        ASSERT_EQ(original.commit().generation, 2u);
        ASSERT_TRUE(original.preview());
        const auto active_slot = original.selected_head().slot_id;
        const auto active_head =
            static_cast<std::size_t>(HEAD_SLOT_OFFSETS[active_slot]);

        const auto expect_fallback = [&](const fs::path& path) {
            auto opened = ProjectReader::open(path);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(opened.error());
            EXPECT_EQ(opened->commit().generation, 1u);
            EXPECT_FALSE(opened->warnings().empty());
            EXPECT_FALSE(opened->preview().has_value());
        };

        const auto locator_path = temporary.path / "locator-mismatch.licht";
        fs::copy_file(source, locator_path);
        auto locator_bytes = read_file_bytes(locator_path);
        write_u64_le(locator_bytes, active_head + 112,
                     original.preview()->offset + 64);
        write_u32_le(locator_bytes, active_head + 4092,
                     crc_range(locator_bytes, active_head, 4092));
        write_file_bytes(locator_path, locator_bytes);
        expect_fallback(locator_path);

        const auto compression_path =
            temporary.path / "preview-compression.licht";
        fs::copy_file(source, compression_path);
        auto compression_bytes = read_file_bytes(compression_path);
        const auto row = std::ranges::find_if(
            original.chunks(), [](const ChunkInfo& chunk) {
                return chunk.key.fourcc == FOURCC_THMB;
            });
        ASSERT_NE(row, original.chunks().end());
        const auto row_index = static_cast<std::size_t>(
            std::distance(original.chunks().begin(), row));
        const auto index_row = static_cast<std::size_t>(
            original.commit().index_offset + INDEX_HEADER_BYTES +
            row_index * INDEX_ROW_BYTES);
        compression_bytes[static_cast<std::size_t>(row->header_offset + 7)] =
            static_cast<std::byte>(Compression::Zstd);
        compression_bytes[index_row + 7] =
            static_cast<std::byte>(Compression::Zstd);
        write_u32_le(compression_bytes,
                     static_cast<std::size_t>(row->header_offset + 60),
                     crc_range(compression_bytes,
                               static_cast<std::size_t>(row->header_offset), 60));
        const std::array<std::uint32_t, 1> active_slots = {active_slot};
        refresh_generation_envelope(
            compression_bytes, original, active_slots);
        write_file_bytes(compression_path, compression_bytes);
        expect_fallback(compression_path);
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
            static constexpr char PNG[] =
                "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52"
                "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x04\x00\x00\x00\xb5\x1c\x0c"
                "\x02\x00\x00\x00\x0b\x49\x44\x41\x54\x78\xda\x63\x64\xf8\x0f\x00"
                "\x01\x05\x01\x01\x27\x18\xe3\x66\x00\x00\x00\x00\x49\x45\x4e\x44"
                "\xae\x42\x60\x82";
            const auto png_span = std::as_bytes(
                std::span(PNG, sizeof(PNG) - 1));
            const std::vector<std::byte> png(
                png_span.begin(), png_span.end());
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
            write_u32_le(duplicate, head_b + 8, 1);
            write_u32_le(duplicate, head_b + 4092,
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
        const fs::path stale_anchor_path =
            temporary.path / "stale-anchor.licht";
        create_single_chunk_fixture(
            stale_anchor_path, 118, 113, 218,
            fixed_key("PROJ", 319),
            R"({"fixture":"stale-anchor"})");

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
                .writer_lock_anchor =
                    base_commit == master.commit().commit_uuid
                        ? master_path
                        : stale_anchor_path,
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
            write_u64_le(bytes, row_offset + 40,
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
            write_u64_le(bytes,
                         static_cast<std::size_t>(first->header_offset + 32),
                         expanded_size);
            write_u64_le(bytes,
                         static_cast<std::size_t>(first->header_offset + 40),
                         expanded_size);
            write_u32_le(bytes,
                         static_cast<std::size_t>(first->header_offset + 56),
                         expanded_crc);
            const std::uint32_t header_crc = crc_range(
                bytes, static_cast<std::size_t>(first->header_offset), 60);
            write_u32_le(bytes,
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
            write_u64_le(bytes, row_offset + 48, expanded_size);
            write_u64_le(bytes, row_offset + 56, expanded_size);
            write_u32_le(bytes, row_offset + 72, expanded_crc);
            write_u32_le(bytes, row_offset + 76, header_crc);
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
            write_u64_le(split, index_offset + 16, 1);
            write_u64_le(split, index_offset + 24, 1);
            write_u32_le(split, index_offset + 48, 0);
            write_u64_le(split, index_offset + INDEX_HEADER_BYTES + 64, 1);
            while (split.size() % CHUNK_ALIGNMENT != 0) {
                split.push_back(std::byte{0});
            }
            const std::size_t commit_offset = split.size();
            const std::size_t source_commit =
                static_cast<std::size_t>(reader.commit().offset);
            split.insert(split.end(), source.begin() + source_commit,
                         source.begin() + source_commit +
                             COMMIT_RECORD_BYTES);
            write_u64_le(split, commit_offset + 64, 1);
            std::fill(split.begin() + commit_offset + 72,
                      split.begin() + commit_offset + 96, std::byte{0});
            write_u64_le(split, commit_offset + 128,
                         FIXED_COMMIT_TIME_NS + 1);
            write_u64_le(split, commit_offset + 136, index_offset);
            write_u64_le(split, commit_offset + 144,
                         INDEX_HEADER_BYTES + INDEX_ROW_BYTES);
            write_u64_le(split, commit_offset + 152,
                         INDEX_HEADER_BYTES + INDEX_ROW_BYTES);
            const std::uint32_t index_crc = crc_range(
                split, index_offset,
                INDEX_HEADER_BYTES + INDEX_ROW_BYTES);
            write_u32_le(split, commit_offset + 160, index_crc);
            write_u32_le(split, commit_offset + 164, index_crc);
            const std::uint64_t committed_end =
                commit_offset + COMMIT_RECORD_BYTES;
            write_u64_le(split, commit_offset + 176, committed_end);
            std::fill(split.begin() + commit_offset + 192,
                      split.begin() + commit_offset + 208, std::byte{0});
            const std::uint32_t commit_crc =
                crc_range(split, commit_offset, 252);
            write_u32_le(split, commit_offset + 252, commit_crc);

            const std::size_t head_b =
                static_cast<std::size_t>(HEAD_SLOT_OFFSETS[1]);
            write_u64_le(split, head_b + 16, 1);
            write_u64_le(split, head_b + 24, 1);
            write_u64_le(split, head_b + 80, commit_offset);
            write_u64_le(split, head_b + 96, committed_end);
            write_u32_le(split, head_b + 104, commit_crc);
            write_u32_le(split, head_b + 4092,
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
         MasterCreateStillRefusesCorruptAndWriteUnsafeDestinations) {
        TemporaryDirectory temporary;
        const fs::path corrupt = temporary.path / "corrupt-master.licht";
        const auto corrupt_bytes = byte_vector("not a licht project");
        write_file_bytes(corrupt, corrupt_bytes);
        auto corrupt_create =
            ProjectWriter::create(corrupt, fixture_create_options(863));
        ASSERT_FALSE(corrupt_create);
        EXPECT_EQ(read_file_bytes(corrupt), corrupt_bytes);

        const fs::path unsafe = temporary.path / "unsafe-master.licht";
        fs::copy_file(FIXTURES / "write-unsafe.licht", unsafe);
        const auto unsafe_bytes = read_file_bytes(unsafe);
        auto unsafe_create =
            ProjectWriter::create(unsafe, fixture_create_options(864));
        ASSERT_FALSE(unsafe_create);
        EXPECT_EQ(unsafe_create.error().code(), lfs::ErrorCode::Unsupported);
        EXPECT_EQ(read_file_bytes(unsafe), unsafe_bytes);
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
                .compression = Compression::Zstd,
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
                .compression = Compression::Zstd,
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

            const int status = run_child_process(
                [&] {
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
                },
                102, 103);
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

    TEST(ProjectContainerWriter,
         CompactionSigkillAtEveryBoundaryPublishesOnlyOldOrVerifiedNew) {
        TemporaryDirectory temporary;
        for (int boundary_value =
                 static_cast<int>(
                     CommitBoundary::
                         CurrentHeadValidated);
             boundary_value <=
             static_cast<int>(
                 CommitBoundary::Committed);
             ++boundary_value) {
            const auto target =
                static_cast<CommitBoundary>(
                    boundary_value);
            SCOPED_TRACE(std::format(
                "compaction boundary {}",
                boundary_value));
            const fs::path path =
                temporary.path /
                std::format(
                    "compact-crash-{}.licht",
                    boundary_value);
            fs::copy_file(
                FIXTURES /
                    "multi-generation-append.licht",
                path);
            const auto prior_bytes =
                read_file_bytes(path);
            const auto commit_uuid =
                fixed_uuid(
                    1200 + boundary_value);

            const int status = run_child_process(
                [&] {
                    auto compacted =
                        ProjectWriter::compact(
                            path,
                            CompactionOptions{
                                .compatibility =
                                    {},
                                .new_file_uuid =
                                    fixed_uuid(
                                        1300 +
                                        boundary_value),
                                .commit_uuid =
                                    commit_uuid,
                                .snapshot_uuid =
                                    fixed_uuid(
                                        1400 +
                                        boundary_value),
                                .creation_time_unix_ns =
                                    FIXED_CREATION_TIME_NS +
                                    static_cast<
                                        std::uint64_t>(
                                        boundary_value),
                                .wallclock_unix_ns =
                                    FIXED_COMMIT_TIME_NS +
                                    static_cast<
                                        std::uint64_t>(
                                        boundary_value),
                                .keep_tombstones =
                                    false,
                                .disk_reserve_bytes =
                                    0,
                                .boundary_observer =
                                    [target](
                                        const CommitBoundary
                                            reached) {
                                        if (reached ==
                                            target) {
                                            ::kill(
                                                ::getpid(),
                                                SIGKILL);
                                        }
                                    },
                            });
                    (void)compacted;
                },
                111, 112);
            ASSERT_TRUE(WIFSIGNALED(status));
            EXPECT_EQ(
                WTERMSIG(status), SIGKILL);
            ProjectReader surviving =
                require_result(
                    ProjectReader::open(path));
            require_status(
                surviving.verify_all());
            const auto after_bytes =
                read_file_bytes(path);
            if (after_bytes != prior_bytes) {
                EXPECT_EQ(
                    surviving.commit().kind,
                    CommitKind::Compaction);
                EXPECT_EQ(
                    surviving.commit()
                        .commit_uuid,
                    commit_uuid);
                EXPECT_EQ(
                    surviving.commit()
                        .generation,
                    1u);
            }

            auto hygiene =
                inspect_autosave_recovery(
                    path);
            ASSERT_TRUE(hygiene)
                << lfs::format_for_developer(
                       hygiene.error());
            for (const auto& entry :
                 fs::directory_iterator(
                     temporary.path)) {
                const auto name =
                    entry.path()
                        .filename()
                        .string();
                if (!name.starts_with(
                        path.stem()
                            .string())) {
                    continue;
                }
                EXPECT_EQ(
                    name.find(".compact."),
                    std::string::npos);
                EXPECT_EQ(
                    name.find(
                        ".replace-backup."),
                    std::string::npos);
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
                          "size=80m,mode=0700"),
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
            const std::vector<std::byte> payload(96 * 1024 * 1024,
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
        fs::remove(path);

        const fs::path sidecar_master =
            mountpoint.path /
            "sidecar-disk-full.licht";
        const fs::path sidecar =
            autosave_sidecar_path(
                sidecar_master);
        fs::copy_file(
            FIXTURES /
                "autosave-master.licht",
            sidecar_master);
        fs::copy_file(
            FIXTURES /
                "autosave-sidecar-valid.licht.autosave",
            sidecar);
        const auto master_before =
            read_file_bytes(sidecar_master);
        const auto sidecar_before =
            read_file_bytes(sidecar);
        {
            ProjectReader master =
                require_result(
                    ProjectReader::open(
                        sidecar_master));
            ProjectReader prior_sidecar =
                require_result(
                    ProjectReader::open(
                        sidecar));
            require_status(
                prior_sidecar.verify_all());
            ProjectWriter writer =
                require_result(
                    ProjectWriter::create(
                        sidecar,
                        CreateOptions{
                            .project_uuid =
                                master
                                    .superblock()
                                    .project_uuid,
                            .file_uuid =
                                fixed_uuid(1500),
                            .role =
                                ContainerRole::
                                    AutosaveSidecar,
                            .base_explicit_commit_uuid =
                                master.commit()
                                    .commit_uuid,
                            .autosave_sequence =
                                prior_sidecar
                                    .superblock()
                                    .autosave_sequence +
                                1,
                            .sidecar_snapshot_uuid =
                                fixed_uuid(1501),
                            .creation_time_unix_ns =
                                FIXED_CREATION_TIME_NS,
                            .index_compression =
                                IndexCompression::
                                    StoredForDeterministicTests,
                            .disk_reserve_bytes =
                                0,
                            .boundary_observer =
                                {},
                            .writer_lock_anchor =
                                sidecar_master,
                        }));
            CommitOptions commit =
                fixture_commit_options(
                    1502, 1501, 1);
            commit.kind =
                CommitKind::Autosave;
            require_status(
                writer.plan_commit(commit));
            // Deliberately model space disappearing after a successful
            // preflight so the failure occurs while the new sidecar temp is
            // being written.
            require_status(
                writer.preflight(0));
            const std::vector<std::byte>
                payload(
                    96 * 1024 * 1024,
                    std::byte{0x6b});
            auto write =
                writer.write_chunk(
                    fixed_key("VIEW", 1503),
                    payload);
            ASSERT_FALSE(write);
            EXPECT_EQ(
                write.error().code(),
                lfs::ErrorCode::
                    ResourceExhausted);
        }
        EXPECT_EQ(
            read_file_bytes(sidecar_master),
            master_before);
        EXPECT_EQ(
            read_file_bytes(sidecar),
            sidecar_before);
        {
            ProjectReader intact_master =
                require_result(
                    ProjectReader::open(
                        sidecar_master));
            ProjectReader intact_sidecar =
                require_result(
                    ProjectReader::open(
                        sidecar));
            require_status(
                intact_master.verify_all());
            require_status(
                intact_sidecar.verify_all());
        }
        fs::remove(sidecar);
        fs::remove(sidecar_master);

        const fs::path compact_path =
            mountpoint.path /
            "compact-disk-full.licht";
        {
            ProjectWriter writer =
                require_result(
                    ProjectWriter::create(
                        compact_path,
                        fixture_create_options(
                            1600)));
            const std::vector<std::byte>
                payload(
                    28 * 1024 * 1024,
                    std::byte{0x7c});
            require_status(
                writer.plan_commit(
                    fixture_commit_options(
                        1601, 1602, 1)));
            require_status(
                writer.preflight(
                    payload.size()));
            require_status(
                writer.write_chunk(
                    fixed_key("X7Q9", 1603),
                    payload,
                    ChunkWriteOptions{
                        .chunk_version =
                            88,
                        .compression =
                            Compression::Stored,
                        .tensor_payload =
                            true,
                        .block_crcs =
                            true,
                        .expected_stream_bytes =
                            std::nullopt,
                    }));
            require_status(writer.commit());
        }
        const auto compact_before =
            read_file_bytes(compact_path);
        const fs::path filler =
            mountpoint.path / "filler.bin";
        bool filled = false;
        auto compacted =
            ProjectWriter::compact(
                compact_path,
                CompactionOptions{
                    .compatibility = {},
                    .new_file_uuid =
                        fixed_uuid(1604),
                    .commit_uuid =
                        fixed_uuid(1605),
                    .snapshot_uuid =
                        fixed_uuid(1606),
                    .creation_time_unix_ns =
                        FIXED_CREATION_TIME_NS +
                        1,
                    .wallclock_unix_ns =
                        FIXED_COMMIT_TIME_NS +
                        1,
                    .keep_tombstones =
                        false,
                    .disk_reserve_bytes =
                        0,
                    .boundary_observer =
                        [&](const CommitBoundary
                                boundary) {
                            if (boundary !=
                                    CommitBoundary::
                                        PreflightComplete ||
                                filled) {
                                return;
                            }
                            filled = true;
                            std::ofstream output(
                                filler,
                                std::ios::binary |
                                    std::ios::trunc);
                            const std::vector<char>
                                block(
                                    1024 * 1024,
                                    '\x55');
                            while (output) {
                                output.write(
                                    block.data(),
                                    static_cast<
                                        std::streamsize>(
                                        block.size()));
                            }
                        },
                });
        ASSERT_TRUE(filled);
        ASSERT_FALSE(compacted);
        EXPECT_EQ(
            compacted.error().code(),
            lfs::ErrorCode::
                ResourceExhausted);
        fs::remove(filler);
        EXPECT_EQ(
            read_file_bytes(compact_path),
            compact_before);
        ProjectReader compact_intact =
            require_result(
                ProjectReader::open(
                    compact_path));
        require_status(
            compact_intact.verify_all());
    }

    TEST(ProjectContainerWriter, RealEnospcUsesIsolatedTmpfs) {
        // Sandboxed CI containers refuse user/mount namespaces; the disk-full
        // guarantee is then only provable on hosts that allow unshare.
        const int probe = std::system(
            "unshare --user --map-root-user --mount true 2>/dev/null");
        if (probe != 0) {
            GTEST_SKIP()
                << "unshare not permitted in this environment; "
                   "ENOSPC matrix requires a host with namespace support";
        }
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
