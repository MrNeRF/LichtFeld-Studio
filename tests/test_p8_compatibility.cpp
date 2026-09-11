/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"
#include "licht_test_support.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

    namespace fs = std::filesystem;
    using Json = nlohmann::ordered_json;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    constexpr std::uint64_t FIXED_CREATION_NS =
        1'735'689'600'000'000'000;
    constexpr std::uint64_t FIXED_COMMIT_NS =
        1'735'689'601'000'000'000;

    ProjectDocumentSaveOptions save_options(
        const std::uint64_t tag,
        const CommitKind kind = CommitKind::Explicit,
        const IndexCompression compression =
            IndexCompression::Zstd) {
        return ProjectDocumentSaveOptions{
            .commit =
                {
                    .kind = kind,
                    .commit_uuid = fixed_uuid(tag),
                    .snapshot_uuid = fixed_uuid(tag + 1),
                    .wallclock_unix_ns = FIXED_COMMIT_NS + tag,
                },
            .file_uuid = fixed_uuid(tag + 2),
            .index_compression = compression,
            .disk_reserve_bytes = 0,
        };
    }

    ProjectDocument create_document(const std::uint64_t tag,
                                    const std::string_view producer) {
        auto document = require_result(ProjectDocument::create(
            fixed_uuid(tag), FIXED_CREATION_NS + tag));
        require_status(document.edit_project().dom().set(
            "p8_producer", std::string(producer)));
        return document;
    }

    ReaderOptions declared_v1() {
        ReaderOptions options;
        options.reader_version = Version{1, 0};
        options.writer_version = Version{1, 0};
        options.reader_capabilities = CapabilitySet{};
        options.writer_capabilities = CapabilitySet{};
        // Bits 0–8: published core caps through CHUNK_BYTESHUFFLE_ZSTD_V1.
        for (std::uint8_t bit = 0; bit <= CHUNK_BYTESHUFFLE_ZSTD_V1; ++bit) {
            options.reader_capabilities.set(bit);
            options.writer_capabilities.set(bit);
        }
        return options;
    }

    void create_raw_fixture(const fs::path& path,
                            const std::uint64_t tag,
                            const CommitOptions& commit) {
        auto writer = require_result(ProjectWriter::create(
            path,
            CreateOptions{
                .project_uuid = fixed_uuid(tag),
                .file_uuid = fixed_uuid(tag + 1),
                .role = ContainerRole::Master,
                .creation_time_unix_ns = FIXED_CREATION_NS + tag,
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            }));
        const auto payload = byte_vector("synthetic compatibility envelope");
        require_status(writer.plan_commit(commit));
        require_status(writer.preflight(payload.size()));
        require_status(writer.write_chunk(
            fixed_key("X1P8", tag + 2), payload));
        require_status(writer.commit());
    }

    void append_generation_with_requirements(
        const fs::path& path, const CommitOptions& commit,
        const std::optional<std::pair<ChunkKey,
                                      ChunkWriteOptions>>& replacement =
            std::nullopt,
        const std::span<const std::byte> replacement_payload = {},
        const IndexCompression index_compression =
            IndexCompression::Zstd) {
        auto prior = require_result(ProjectReader::open(path));
        std::vector<std::pair<ChunkInfo, CleanProof>> rows;
        for (std::size_t index = 0;
             index < prior.chunks().size(); ++index) {
            rows.emplace_back(
                prior.chunks()[index],
                require_result(prior.make_clean_proof(
                    prior.chunks()[index], 90'000 + index)));
        }
        auto writer = require_result(ProjectWriter::append(
            path,
            AppendOptions{
                .compatibility = {},
                .index_compression = index_compression,
                .disk_reserve_bytes = 0,
            }));
        require_status(writer.plan_commit(commit));
        require_status(writer.preflight(replacement_payload.size()));
        for (std::size_t index = 0; index < rows.size(); ++index) {
            if (replacement &&
                rows[index].first.key == replacement->first) {
                continue;
            }
            require_status(writer.reuse_if_clean(
                rows[index].second, 90'000 + index));
        }
        if (replacement) {
            require_status(writer.write_chunk(
                replacement->first, replacement_payload,
                replacement->second));
        }
        require_status(writer.commit());
    }

    struct StoredChunkWitness {
        ChunkInfo metadata;
        std::vector<std::byte> stored;
    };

    StoredChunkWitness witness(const ProjectReader& reader,
                               const ChunkKey& key) {
        const auto* row = reader.find(key);
        if (row == nullptr) {
            throw std::runtime_error("missing witness row");
        }
        std::vector<std::byte> stored(row->stored_bytes);
        require_status(reader.read_stored_at(*row, 0, stored));
        return StoredChunkWitness{*row, std::move(stored)};
    }

    void expect_same_opaque(const StoredChunkWitness& before,
                            const ProjectReader& after) {
        const auto* row = after.find(before.metadata.key);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->key, before.metadata.key);
        EXPECT_EQ(row->chunk_version,
                  before.metadata.chunk_version);
        EXPECT_EQ(row->compression, before.metadata.compression);
        EXPECT_EQ(row->flags, before.metadata.flags);
        EXPECT_EQ(row->stored_bytes, before.metadata.stored_bytes);
        EXPECT_EQ(row->uncompressed_bytes,
                  before.metadata.uncompressed_bytes);
        EXPECT_EQ(row->payload_crc32c,
                  before.metadata.payload_crc32c);
        EXPECT_EQ(row->block_crc_table.has_value(),
                  before.metadata.block_crc_table.has_value());
        std::vector<std::byte> stored(row->stored_bytes);
        require_status(after.read_stored_at(*row, 0, stored));
        EXPECT_EQ(stored, before.stored);
    }

    TEST(P8CompatibilityTest,
         V10CorpusAndSyntheticNewerClassificationRefusePayloadAccess) {
        TemporaryDirectory temporary;
        const auto supported = temporary.path / "supported.licht";
        auto supported_document = create_document(81'000, "supported");
        auto supported_save = supported_document.save(
            supported, save_options(81'010));
        ASSERT_TRUE(supported_save)
            << lfs::format_for_developer(supported_save.error());

        const ReaderOptions v1 = declared_v1();
        const auto classification =
            ProjectReader::classify(supported, v1);
        EXPECT_EQ(classification.state, OpenState::Open);

        for (const bool capability_gate : {false, true}) {
            const auto newer = temporary.path /
                               (capability_gate
                                    ? "newer-capability.licht"
                                    : "newer-version.licht");
            CommitOptions options{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(
                    capability_gate ? 81'110 : 81'210),
                .snapshot_uuid = fixed_uuid(
                    capability_gate ? 81'111 : 81'211),
                .wallclock_unix_ns =
                    FIXED_COMMIT_NS +
                    (capability_gate ? 81'110 : 81'210),
            };
            if (capability_gate) {
                // Unassigned core bit (9+); bit 8 is CHUNK_BYTESHUFFLE_ZSTD_V1.
                options.extra_reader_capabilities.set(9);
            } else {
                options.min_reader_version = Version{1, 1};
            }
            create_raw_fixture(
                newer, capability_gate ? 81'100 : 81'200,
                options);
            EXPECT_EQ(ProjectReader::classify(newer, v1).state,
                      OpenState::UnsupportedNewer);
            auto semantic = ProjectReader::open(newer, v1);
            ASSERT_FALSE(semantic);
            EXPECT_EQ(semantic.error().code(),
                      lfs::ErrorCode::Unsupported);

            auto inspect_options = v1;
            inspect_options.allow_unsupported_inspection = true;
            auto inspection = ProjectReader::open(
                newer, inspect_options);
            ASSERT_TRUE(inspection)
                << lfs::format_for_developer(inspection.error());
            EXPECT_EQ(inspection->open_state(),
                      OpenState::UnsupportedNewer);
            auto verified = inspection->verify_all();
            ASSERT_FALSE(verified);
            EXPECT_EQ(verified.error().code(),
                      lfs::ErrorCode::Unsupported);
            ASSERT_FALSE(inspection->chunks().empty());
            auto extracted = inspection->read_chunk(
                inspection->chunks().front());
            ASSERT_FALSE(extracted);
            EXPECT_EQ(extracted.error().code(),
                      lfs::ErrorCode::Unsupported);
            auto hydrated = ProjectDocument::open(
                newer,
                ProjectDocumentOpenOptions{.reader = v1});
            ASSERT_FALSE(hydrated);
            EXPECT_EQ(hydrated.error().code(),
                      lfs::ErrorCode::Unsupported);
        }

        const auto higher = temporary.path /
                            "unsupported-higher-head.licht";
        fs::copy_file(supported, higher);
        CommitOptions higher_commit{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(81'300),
            .snapshot_uuid = fixed_uuid(81'301),
            .wallclock_unix_ns = FIXED_COMMIT_NS + 81'300,
        };
        higher_commit.extra_reader_capabilities.set(9);
        append_generation_with_requirements(
            higher, higher_commit, std::nullopt, {},
            IndexCompression::StoredForDeterministicTests);
        EXPECT_EQ(ProjectReader::classify(higher, v1).state,
                  OpenState::HardFail);
    }

    TEST(P8CompatibilityTest,
         ProhibitedOldWriterMatrixHasNoWriteEffectForEveryProducer) {
        enum class Gate { MinWriter,
                          Bit5,
                          Bit6,
                          Bit9,
                          VendorBit112 };
        constexpr std::array gates{
            Gate::MinWriter, Gate::Bit5, Gate::Bit6, Gate::Bit9,
            Gate::VendorBit112};
        TemporaryDirectory temporary;

        for (std::size_t index = 0; index < gates.size(); ++index) {
            const auto path = temporary.path /
                              std::format("gate-{}.licht", index);
            auto document = create_document(
                82'000 + index * 100,
                std::format("old-writer-gate-{}", index));
            auto options = save_options(
                82'010 + index * 100,
                CommitKind::Explicit,
                IndexCompression::StoredForDeterministicTests);
            if (gates[index] == Gate::MinWriter) {
                options.commit.min_safe_writer_version = Version{1, 1};
            } else if (gates[index] == Gate::Bit5) {
                options.commit.extra_writer_capabilities.set(5);
            } else if (gates[index] == Gate::Bit6) {
                options.commit.extra_writer_capabilities.set(6);
            } else if (gates[index] == Gate::Bit9) {
                options.commit.extra_writer_capabilities.set(9);
            } else {
                options.commit.extra_writer_capabilities.set(112);
            }
            auto initial = document.save(path, options);
            ASSERT_TRUE(initial)
                << lfs::format_for_developer(initial.error());

            ReaderOptions old = declared_v1();
            if (gates[index] == Gate::Bit5) {
                old.writer_capabilities.set(5, false);
            } else if (gates[index] == Gate::Bit6) {
                old.writer_capabilities.set(6, false);
            }
            const auto prior = require_result(
                ProjectReader::open(path, old));
            ASSERT_FALSE(prior.write_compatibility().safe);
            const auto before = read_file_bytes(path);
            const auto generation = prior.commit().generation;

            auto opened = ProjectDocument::open(
                path, ProjectDocumentOpenOptions{.reader = old});
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(opened.error());
            require_status(opened->edit_project().dom().set(
                "attempted_old_writer_edit", true));

            auto append_save = opened->save(
                path, save_options(82'500 + index * 10));
            ASSERT_FALSE(append_save);
            EXPECT_EQ(append_save.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_EQ(read_file_bytes(path), before);

            const auto save_as = temporary.path /
                                 std::format("gate-{}-save-as.licht", index);
            auto save_as_result = opened->save_as(
                save_as, save_options(82'501 + index * 10));
            ASSERT_FALSE(save_as_result);
            EXPECT_EQ(save_as_result.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_FALSE(fs::exists(save_as));
            EXPECT_EQ(read_file_bytes(path), before);

            const auto sidecar = autosave_sidecar_path(path);
            auto autosave = opened->save_autosave(
                sidecar,
                ProjectDocumentAutosaveOptions{
                    .file_uuid = fixed_uuid(82'502 + index * 10),
                    .base_explicit_commit_uuid =
                        prior.commit().commit_uuid,
                    .autosave_sequence = 1,
                    .snapshot_uuid = fixed_uuid(82'503 + index * 10),
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                });
            ASSERT_FALSE(autosave);
            EXPECT_EQ(autosave.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_FALSE(fs::exists(sidecar));
            EXPECT_EQ(read_file_bytes(path), before);

            auto raw_append = ProjectWriter::append(
                path,
                AppendOptions{
                    .compatibility = old,
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                });
            ASSERT_FALSE(raw_append);
            EXPECT_EQ(raw_append.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_EQ(read_file_bytes(path), before);

            auto compacted = ProjectWriter::compact(
                path,
                CompactionOptions{
                    .compatibility = old,
                    .new_file_uuid = fixed_uuid(82'504 + index * 10),
                    .commit_uuid = fixed_uuid(82'505 + index * 10),
                    .snapshot_uuid = fixed_uuid(82'506 + index * 10),
                    .creation_time_unix_ns = FIXED_CREATION_NS,
                    .wallclock_unix_ns = FIXED_COMMIT_NS,
                    .disk_reserve_bytes = 0,
                });
            ASSERT_FALSE(compacted);
            EXPECT_EQ(compacted.error().code(),
                      lfs::ErrorCode::Unsupported);
            EXPECT_EQ(read_file_bytes(path), before);
            EXPECT_EQ(require_result(ProjectReader::open(path, old))
                          .commit()
                          .generation,
                      generation);

            auto lease = WriterLockLease::acquire(path);
            ASSERT_TRUE(lease)
                << lfs::format_for_developer(lease.error());
        }
    }

    TEST(P8CompatibilityTest,
         OpaqueAndRetainedJsonSurviveSafeAppendAndCompaction) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "opaque-safe.licht";
        auto document = create_document(83'000, "opaque-safe");
        const Json future_uuids = Json::array(
            {fixed_uuid(83'001).to_string(),
             fixed_uuid(83'002).to_string()});
        require_status(document.edit_project().dom().set_json(
            "future_uuid_array", future_uuids));
        auto initial = document.save(path, save_options(83'010));
        ASSERT_TRUE(initial)
            << lfs::format_for_developer(initial.error());

        const auto base = require_result(ProjectReader::open(path));
        const ChunkKey view_key{
            .fourcc = FOURCC_VIEW,
            .instance_uuid = base.superblock().project_uuid,
        };
        const auto* base_view = base.find(view_key);
        ASSERT_NE(base_view, nullptr);
        const auto newer_view_payload =
            require_result(base.read_chunk(*base_view));

        CommitOptions opaque_commit{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(83'020),
            .snapshot_uuid = fixed_uuid(83'021),
            .wallclock_unix_ns = FIXED_COMMIT_NS + 83'020,
        };
        opaque_commit.extra_writer_capabilities.set(
            OPAQUE_CHUNK_PRESERVATION);
        opaque_commit.extra_writer_capabilities.set(
            RETAINED_JSON_FIELDS);
        append_generation_with_requirements(
            path, opaque_commit,
            std::pair{
                view_key,
                ChunkWriteOptions{
                    .chunk_version = 99,
                    .compression = Compression::ZstdFramed,
                }},
            newer_view_payload);

        const ChunkKey unknown_key = fixed_key("X8P8", 83'030);
        {
            auto prior = require_result(ProjectReader::open(path));
            std::vector<std::pair<ChunkInfo, CleanProof>> rows;
            for (std::size_t index = 0;
                 index < prior.chunks().size(); ++index) {
                rows.emplace_back(
                    prior.chunks()[index],
                    require_result(prior.make_clean_proof(
                        prior.chunks()[index], 91'000 + index)));
            }
            auto writer = require_result(ProjectWriter::append(
                path,
                AppendOptions{
                    .compatibility = {},
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            CommitOptions commit{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(83'031),
                .snapshot_uuid = fixed_uuid(83'032),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 83'031,
            };
            commit.extra_writer_capabilities.set(
                OPAQUE_CHUNK_PRESERVATION);
            commit.extra_writer_capabilities.set(
                RETAINED_JSON_FIELDS);
            const auto unknown_payload = byte_vector(
                "opaque bytes from a future X8P8 producer");
            require_status(writer.plan_commit(commit));
            require_status(writer.preflight(unknown_payload.size()));
            for (std::size_t index = 0; index < rows.size(); ++index) {
                require_status(writer.reuse_if_clean(
                    rows[index].second, 91'000 + index));
            }
            require_status(writer.write_chunk(
                unknown_key, unknown_payload,
                ChunkWriteOptions{
                    .chunk_version = 77,
                    .compression = Compression::ZstdFramed,
                }));
            require_status(writer.commit());
        }

        auto before = require_result(ProjectReader::open(path));
        const auto unknown_before = witness(before, unknown_key);
        const auto view_before = witness(before, view_key);
        auto opened = ProjectDocument::open(path);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(opened.error());
        EXPECT_FALSE(opened->view().dom().get<std::string>(
                                             "p8_nonexistent")
                         .has_value());
        require_status(opened->edit_project().set_modified_at_unix_ns(
            FIXED_COMMIT_NS + 83'040));
        auto appended = opened->save(
            path, save_options(83'041));
        ASSERT_TRUE(appended)
            << lfs::format_for_developer(appended.error());
        EXPECT_EQ(appended->opaque_chunks_carried, 2u);

        auto after_append = require_result(ProjectReader::open(path));
        expect_same_opaque(unknown_before, after_append);
        expect_same_opaque(view_before, after_append);
        auto appended_document = require_result(
            ProjectDocument::open(path));
        EXPECT_EQ(appended_document.project().dom().get_json(
                      "future_uuid_array"),
                  std::optional<Json>(future_uuids));

        require_status(ProjectWriter::compact(
            path,
            CompactionOptions{
                .compatibility = {},
                .new_file_uuid = fixed_uuid(83'050),
                .commit_uuid = fixed_uuid(83'051),
                .snapshot_uuid = fixed_uuid(83'052),
                .creation_time_unix_ns = FIXED_CREATION_NS + 83'050,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 83'050,
                .disk_reserve_bytes = 0,
            }));
        auto after_compact = require_result(ProjectReader::open(path));
        expect_same_opaque(unknown_before, after_compact);
        expect_same_opaque(view_before, after_compact);
        auto compacted_document = require_result(
            ProjectDocument::open(path));
        EXPECT_EQ(compacted_document.project().dom().get_json(
                      "future_uuid_array"),
                  std::optional<Json>(future_uuids));
    }

    TEST(P8CompatibilityTest,
         MissingOpaqueOrRetainedJsonCapabilityMakesDocumentReadOnly) {
        TemporaryDirectory temporary;

        const auto opaque_path = temporary.path / "missing-bit5.licht";
        auto opaque_document = create_document(84'000, "missing-bit5");
        ASSERT_TRUE(opaque_document.save(
            opaque_path, save_options(84'010)));
        {
            auto prior = require_result(ProjectReader::open(opaque_path));
            std::vector<std::pair<ChunkInfo, CleanProof>> rows;
            for (std::size_t index = 0;
                 index < prior.chunks().size(); ++index) {
                rows.emplace_back(
                    prior.chunks()[index],
                    require_result(prior.make_clean_proof(
                        prior.chunks()[index], 92'000 + index)));
            }
            auto writer = require_result(ProjectWriter::append(
                opaque_path,
                AppendOptions{
                    .compatibility = {},
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            require_status(writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(84'020),
                .snapshot_uuid = fixed_uuid(84'021),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 84'020,
            }));
            const auto payload = byte_vector("future opaque without declaration");
            require_status(writer.preflight(payload.size()));
            for (std::size_t index = 0; index < rows.size(); ++index) {
                require_status(writer.reuse_if_clean(
                    rows[index].second, 92'000 + index));
            }
            require_status(writer.write_chunk(
                fixed_key("X8N5", 84'022), payload));
            require_status(writer.commit());
        }
        auto opaque_open = require_result(
            ProjectDocument::open(opaque_path));
        require_status(opaque_open.edit_project().dom().set(
            "attempt", true));
        const auto opaque_before = read_file_bytes(opaque_path);
        auto opaque_save = opaque_open.save(
            opaque_path, save_options(84'030));
        ASSERT_FALSE(opaque_save);
        EXPECT_EQ(opaque_save.error().code(),
                  lfs::ErrorCode::Unsupported);
        EXPECT_EQ(read_file_bytes(opaque_path), opaque_before);

        const auto json_path = temporary.path / "missing-bit6.licht";
        auto json_document = require_result(ProjectDocument::create(
            fixed_uuid(84'100), FIXED_CREATION_NS + 84'100));
        ASSERT_TRUE(json_document.save(
            json_path, save_options(84'110)));
        auto prior = require_result(ProjectReader::open(json_path));
        const ChunkKey proj_key{
            .fourcc = FOURCC_PROJ,
            .instance_uuid = prior.superblock().project_uuid,
        };
        const auto* proj = prior.find(proj_key);
        ASSERT_NE(proj, nullptr);
        auto proj_payload = require_result(prior.read_chunk(*proj));
        Json proj_json = Json::parse(
            reinterpret_cast<const char*>(proj_payload.data()),
            reinterpret_cast<const char*>(proj_payload.data()) +
                proj_payload.size());
        proj_json["future_without_bit6"] = Json::array(
            {fixed_uuid(84'111).to_string()});
        const auto future_text = proj_json.dump(2);
        const auto future_payload = byte_vector(future_text);
        append_generation_with_requirements(
            json_path,
            CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(84'120),
                .snapshot_uuid = fixed_uuid(84'121),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 84'120,
            },
            std::pair{
                proj_key,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ZstdFramed,
                }},
            future_payload);
        auto json_open = require_result(
            ProjectDocument::open(json_path));
        require_status(json_open.edit_project().set_modified_at_unix_ns(
            FIXED_COMMIT_NS + 84'130));
        const auto json_before = read_file_bytes(json_path);
        auto json_save = json_open.save(
            json_path, save_options(84'131));
        ASSERT_FALSE(json_save);
        EXPECT_EQ(json_save.error().code(),
                  lfs::ErrorCode::Unsupported);
        EXPECT_EQ(read_file_bytes(json_path), json_before);
    }

    TEST(P8CompatibilityTest,
         SidecarBaseReferenceAbsentFromMasterIsRejectedByRecoveryInspection) {
        TemporaryDirectory temporary;
        const auto master_path = temporary.path / "base-reference.licht";
        const auto sidecar_path = autosave_sidecar_path(master_path);
        auto document = create_document(84'500, "base-reference-master");
        ASSERT_TRUE(document.save(master_path, save_options(84'510)));
        auto master = require_result(ProjectReader::open(master_path));
        const auto snapshot_uuid = fixed_uuid(84'520);

        {
            auto writer = require_result(ProjectWriter::create(
                sidecar_path,
                CreateOptions{
                    .project_uuid = master.superblock().project_uuid,
                    .file_uuid = fixed_uuid(84'521),
                    .role = ContainerRole::AutosaveSidecar,
                    .base_explicit_commit_uuid = master.commit().commit_uuid,
                    .autosave_sequence = 1,
                    .sidecar_snapshot_uuid = snapshot_uuid,
                    .creation_time_unix_ns = FIXED_CREATION_NS + 84'521,
                    .index_compression =
                        IndexCompression::StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                    .writer_lock_anchor = master_path,
                }));
            CommitOptions sidecar_commit{
                .kind = CommitKind::Autosave,
                .commit_uuid = fixed_uuid(84'522),
                .snapshot_uuid = snapshot_uuid,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 84'522,
            };
            sidecar_commit.extra_reader_capabilities =
                master.commit().required_reader_capabilities;
            sidecar_commit.extra_writer_capabilities =
                master.commit().required_writer_capabilities;
            require_status(writer.plan_commit(sidecar_commit));
            require_status(writer.preflight(0));
            for (const auto& row : master.chunks()) {
                if (row.is_live()) {
                    require_status(writer.add_sidecar_base_reference(row));
                }
            }
            ASSERT_FALSE(master.chunks().empty());
            ChunkInfo absent = master.chunks().front();
            absent.key = fixed_key("XBAS", 84'523);
            require_status(writer.add_sidecar_base_reference(absent));
            require_status(writer.commit());
        }

        auto inspection = inspect_autosave_recovery(master_path);
        ASSERT_TRUE(inspection)
            << lfs::format_for_developer(inspection.error());
        // Would fail if inspect still returned Invalid or left the sidecar
        // in place instead of quarantining the illegal overlay.
        EXPECT_EQ(inspection->disposition, RecoveryDisposition::StaleDeleted);
        EXPECT_FALSE(std::filesystem::exists(sidecar_path));
        bool found_aside = false;
        const auto prefix =
            sidecar_path.filename().string() + ".corrupt-";
        std::error_code error;
        for (std::filesystem::directory_iterator
                 iterator(sidecar_path.parent_path(), error),
             end;
             !error && iterator != end;
             iterator.increment(error)) {
            if (iterator->path().filename().string().starts_with(prefix)) {
                found_aside = true;
                break;
            }
        }
        EXPECT_TRUE(found_aside);
        ASSERT_FALSE(inspection->diagnostics.empty());
        EXPECT_TRUE(std::ranges::any_of(
            inspection->diagnostics,
            [](const std::string& diagnostic) {
                return diagnostic.find("invalid base reference") !=
                       std::string::npos;
            }));
    }

#ifndef _WIN32
    TEST(P8CompatibilityTest,
         AutosaveSigkillAtEveryBoundaryLeavesCompleteOldOrNewSidecar) {
        TemporaryDirectory temporary;
        for (int boundary_value =
                 static_cast<int>(CommitBoundary::CurrentHeadValidated);
             boundary_value <=
             static_cast<int>(CommitBoundary::Committed);
             ++boundary_value) {
            const auto boundary =
                static_cast<CommitBoundary>(boundary_value);
            SCOPED_TRACE(std::format("autosave boundary {}", boundary_value));
            const auto master = temporary.path /
                                std::format("autosave-crash-{}.licht",
                                            boundary_value);
            const auto sidecar = autosave_sidecar_path(master);
            auto initial = create_document(
                90'000 + boundary_value * 100, "autosave-crash");
            (void)require_result(initial.save(
                master, save_options(90'010 + boundary_value * 100)));
            const auto master_before = read_file_bytes(master);
            const auto base = require_result(ProjectReader::open(master));
            {
                auto document = require_result(ProjectDocument::open(master));
                require_status(document.edit_view().dom().set(
                    "crash_marker", std::string{"old"}));
                (void)require_result(document.save_autosave(
                    sidecar,
                    ProjectDocumentAutosaveOptions{
                        .file_uuid =
                            fixed_uuid(90'020 + boundary_value * 100),
                        .base_explicit_commit_uuid =
                            base.commit().commit_uuid,
                        .autosave_sequence = 1,
                        .snapshot_uuid =
                            fixed_uuid(90'021 + boundary_value * 100),
                        .index_compression =
                            IndexCompression::StoredForDeterministicTests,
                        .disk_reserve_bytes = 0,
                    }));
            }

            const pid_t child = ::fork();
            ASSERT_GE(child, 0);
            if (child == 0) {
                try {
                    auto document = ProjectDocument::open(master);
                    if (!document) {
                        ::_exit(101);
                    }
                    auto edited = document->edit_view().dom().set(
                        "crash_marker", std::string{"new"});
                    if (!edited) {
                        ::_exit(102);
                    }
                    (void)document->save_autosave(
                        sidecar,
                        ProjectDocumentAutosaveOptions{
                            .file_uuid =
                                fixed_uuid(90'022 + boundary_value * 100),
                            .base_explicit_commit_uuid =
                                base.commit().commit_uuid,
                            .autosave_sequence = 2,
                            .snapshot_uuid =
                                fixed_uuid(90'023 + boundary_value * 100),
                            .index_compression =
                                IndexCompression::StoredForDeterministicTests,
                            .disk_reserve_bytes = 0,
                            .boundary_observer =
                                [boundary](const CommitBoundary reached) {
                                    if (reached == boundary) {
                                        ::kill(::getpid(), SIGKILL);
                                    }
                                },
                        });
                } catch (...) {
                    ::_exit(103);
                }
                ::_exit(104);
            }
            int status = 0;
            ASSERT_EQ(::waitpid(child, &status, 0), child);
            ASSERT_TRUE(WIFSIGNALED(status));
            EXPECT_EQ(WTERMSIG(status), SIGKILL);
            EXPECT_EQ(read_file_bytes(master), master_before);

            auto inspection = inspect_autosave_recovery(master);
            ASSERT_TRUE(inspection)
                << lfs::format_for_developer(inspection.error());
            ASSERT_EQ(inspection->disposition, RecoveryDisposition::Offer);
            ASSERT_TRUE(inspection->selected_path);
            EXPECT_TRUE(inspection->autosave_sequence == 1 ||
                        inspection->autosave_sequence == 2);
            const auto recovered = temporary.path /
                                   std::format("autosave-recovered-{}.licht",
                                               boundary_value);
            require_status(materialize_recovered_project(
                master, *inspection->selected_path, recovered));
            auto reader = require_result(ProjectReader::open(recovered));
            require_status(reader.verify_all());
            auto recovered_document =
                require_result(ProjectDocument::open(recovered));
            const auto marker = recovered_document.view().dom().get<std::string>(
                "crash_marker");
            ASSERT_TRUE(marker);
            EXPECT_EQ(*marker,
                      inspection->autosave_sequence == 1 ? "old" : "new");
        }
    }

    TEST(P8LockMatrixTest,
         SecondProcessIsReadOnlyOnOriginalAndMaySaveAsElsewhere) {
        TemporaryDirectory temporary;
        const auto original = temporary.path / "locked.licht";
        const auto save_as = temporary.path / "contender-save-as.licht";
        auto document = create_document(85'000, "lock-holder");
        ASSERT_TRUE(document.save(
            original, save_options(85'010)));
        const auto before = read_file_bytes(original);
        auto lease = require_result(
            WriterLockLease::acquire(original));

        const pid_t child = ::fork();
        ASSERT_GE(child, 0);
        if (child == 0) {
            const auto fail_child = [](const int code) {
                ::_exit(code);
            };
            auto reader = ProjectReader::open(original);
            if (!reader || reader->open_state() != OpenState::Open) {
                fail_child(10);
            }
            auto contender = ProjectDocument::open(original);
            if (!contender) {
                fail_child(11);
            }
            if (!contender->edit_project().dom().set(
                    "contender", true)) {
                fail_child(12);
            }
            auto blocked_save = contender->save(
                original, save_options(85'020));
            if (blocked_save ||
                blocked_save.error().code() !=
                    lfs::ErrorCode::Unavailable ||
                read_file_bytes(original) != before) {
                fail_child(13);
            }
            auto blocked_compact = ProjectWriter::compact(
                original,
                CompactionOptions{
                    .compatibility = {},
                    .new_file_uuid = fixed_uuid(85'021),
                    .commit_uuid = fixed_uuid(85'022),
                    .snapshot_uuid = fixed_uuid(85'023),
                    .creation_time_unix_ns = FIXED_CREATION_NS,
                    .wallclock_unix_ns = FIXED_COMMIT_NS,
                    .disk_reserve_bytes = 0,
                });
            if (blocked_compact ||
                blocked_compact.error().code() !=
                    lfs::ErrorCode::Unavailable ||
                read_file_bytes(original) != before) {
                fail_child(14);
            }
            const auto sidecar = autosave_sidecar_path(original);
            auto blocked_autosave = contender->save_autosave(
                sidecar,
                ProjectDocumentAutosaveOptions{
                    .file_uuid = fixed_uuid(85'024),
                    .base_explicit_commit_uuid =
                        reader->commit().commit_uuid,
                    .autosave_sequence = 1,
                    .snapshot_uuid = fixed_uuid(85'025),
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                });
            if (blocked_autosave ||
                blocked_autosave.error().code() !=
                    lfs::ErrorCode::Unavailable ||
                fs::exists(sidecar) ||
                read_file_bytes(original) != before) {
                fail_child(15);
            }
            auto allowed_save_as = contender->save_as(
                save_as, save_options(85'030));
            if (!allowed_save_as || !fs::is_regular_file(save_as)) {
                fail_child(16);
            }
            auto saved_reader = ProjectReader::open(save_as);
            if (!saved_reader || !saved_reader->verify_all()) {
                fail_child(17);
            }
            fail_child(0);
        }

        int status = 0;
        ASSERT_EQ(::waitpid(child, &status, 0), child);
        ASSERT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0);
        EXPECT_EQ(read_file_bytes(original), before);
        EXPECT_TRUE(fs::is_regular_file(save_as));
    }
#endif

    TEST(P8CompatibilityTest, CkptContainerZstdLogicalPayloadByteVerbatim) {
        // INVARIANT: container zstd is transparent — inflate(chunk) == the
        // exact logical CKPT/LFKP bytes staged for write (standalone-serialize
        // equivalent). Exercises write_chunk Zstd + reader decompress path
        // used by ProjectDocument lazy_binary_options.
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "ckpt-zstd-verbatim.licht";

        std::vector<std::byte> lfkp_bytes(4096);
        for (std::size_t index = 0; index < lfkp_bytes.size(); ++index) {
            lfkp_bytes[index] =
                static_cast<std::byte>((index * 131u + 17u) & 0xffu);
        }
        for (std::size_t index = 0; index < 512; ++index) {
            lfkp_bytes[index] = std::byte{0};
        }

        const ChunkKey key{
            .fourcc = FOURCC_CKPT,
            .instance_uuid = fixed_uuid(93'001),
        };
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path,
                CreateOptions{
                    .project_uuid = fixed_uuid(93'000),
                    .file_uuid = fixed_uuid(93'002),
                    .role = ContainerRole::Master,
                    .creation_time_unix_ns = FIXED_CREATION_NS + 93'000,
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            require_status(writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(93'003),
                .snapshot_uuid = key.instance_uuid,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 93'100,
            }));
            require_status(writer.preflight(lfkp_bytes.size()));
            // Mirror document lazy_binary_options: Zstd + tensor_payload CKPT.
            require_status(writer.write_chunk(
                key, lfkp_bytes,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ZstdFramed,
                    .tensor_payload = true,
                    .block_crcs = false,
                    .expected_stream_bytes = lfkp_bytes.size(),
                }));
            require_status(writer.commit());
        }

        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo* row = reader.find(key.fourcc, key.instance_uuid);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->compression, Compression::ZstdFramed);
        EXPECT_EQ(row->uncompressed_bytes, lfkp_bytes.size());
        EXPECT_LT(row->stored_bytes, row->uncompressed_bytes);
        EXPECT_TRUE(reader.commit()
                        .required_reader_capabilities
                        .contains(CHUNK_ZSTD_V1));
        auto inflated = require_result(reader.read_chunk(*row));
        ASSERT_EQ(inflated.size(), lfkp_bytes.size());
        EXPECT_TRUE(std::equal(
            inflated.begin(), inflated.end(), lfkp_bytes.begin()));
    }

    TEST(P8CompatibilityTest, CkptByteShuffleZstdLogicalPayloadByteVerbatim) {
        TemporaryDirectory temporary;
        const fs::path path =
            temporary.path / "ckpt-byteshuffle-zstd-verbatim.licht";

        // Size multiple of 4 so ByteShuffle is retained (not Zstd fallback).
        std::vector<std::byte> lfkp_bytes(8192);
        for (std::size_t index = 0; index < lfkp_bytes.size(); ++index) {
            lfkp_bytes[index] =
                static_cast<std::byte>((index * 131u + 17u) & 0xffu);
        }
        for (std::size_t index = 0; index < 512; ++index) {
            lfkp_bytes[index] = std::byte{0};
        }

        const ChunkKey key{
            .fourcc = FOURCC_CKPT,
            .instance_uuid = fixed_uuid(94'001),
        };
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path,
                CreateOptions{
                    .project_uuid = fixed_uuid(94'000),
                    .file_uuid = fixed_uuid(94'002),
                    .role = ContainerRole::Master,
                    .creation_time_unix_ns = FIXED_CREATION_NS + 94'000,
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            require_status(writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(94'003),
                .snapshot_uuid = key.instance_uuid,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 94'100,
            }));
            require_status(writer.preflight(lfkp_bytes.size()));
            require_status(writer.write_chunk(
                key, lfkp_bytes,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ByteShuffleZstdFramed,
                    .tensor_payload = true,
                    .block_crcs = false,
                    .expected_stream_bytes = lfkp_bytes.size(),
                }));
            require_status(writer.commit());
        }

        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo* row = reader.find(key.fourcc, key.instance_uuid);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->compression, Compression::ByteShuffleZstdFramed);
        EXPECT_EQ(row->uncompressed_bytes, lfkp_bytes.size());
        EXPECT_LT(row->stored_bytes, row->uncompressed_bytes);
        EXPECT_TRUE(reader.commit()
                        .required_reader_capabilities
                        .contains(CHUNK_BYTESHUFFLE_ZSTD_V1));
        EXPECT_FALSE(reader.commit()
                         .required_reader_capabilities
                         .contains(CHUNK_ZSTD_V1)); // no plain Zstd chunks
        auto inflated = require_result(reader.read_chunk(*row));
        ASSERT_EQ(inflated.size(), lfkp_bytes.size());
        EXPECT_TRUE(std::equal(
            inflated.begin(), inflated.end(), lfkp_bytes.begin()));
    }

    TEST(P8CompatibilityTest, ByteShuffleFallsBackToZstdWhenSizeNotMultipleOf4) {
        TemporaryDirectory temporary;
        const fs::path path =
            temporary.path / "ckpt-byteshuffle-fallback.licht";

        std::vector<std::byte> lfkp_bytes(4097); // 4097 % 4 == 1
        for (std::size_t index = 0; index < lfkp_bytes.size(); ++index) {
            lfkp_bytes[index] =
                static_cast<std::byte>((index * 17u + 3u) & 0xffu);
        }

        const ChunkKey key{
            .fourcc = FOURCC_CKPT,
            .instance_uuid = fixed_uuid(95'001),
        };
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path,
                CreateOptions{
                    .project_uuid = fixed_uuid(95'000),
                    .file_uuid = fixed_uuid(95'002),
                    .role = ContainerRole::Master,
                    .creation_time_unix_ns = FIXED_CREATION_NS + 95'000,
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            require_status(writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(95'003),
                .snapshot_uuid = key.instance_uuid,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 95'100,
            }));
            require_status(writer.preflight(lfkp_bytes.size()));
            require_status(writer.write_chunk(
                key, lfkp_bytes,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ByteShuffleZstdFramed,
                    .tensor_payload = true,
                    .block_crcs = false,
                    .expected_stream_bytes = lfkp_bytes.size(),
                }));
            require_status(writer.commit());
        }

        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo* row = reader.find(key.fourcc, key.instance_uuid);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->compression, Compression::ZstdFramed);
        EXPECT_TRUE(reader.commit()
                        .required_reader_capabilities
                        .contains(CHUNK_ZSTD_V1));
        EXPECT_FALSE(reader.commit()
                         .required_reader_capabilities
                         .contains(CHUNK_BYTESHUFFLE_ZSTD_V1));
        auto inflated = require_result(reader.read_chunk(*row));
        ASSERT_EQ(inflated.size(), lfkp_bytes.size());
        EXPECT_TRUE(std::equal(
            inflated.begin(), inflated.end(), lfkp_bytes.begin()));
    }

    TEST(P8CompatibilityTest, MixedZstdAndByteShuffleChunksOpenCorrectly) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "mixed-encodings.licht";

        std::vector<std::byte> jsonish = {
            std::byte{'{'}, std::byte{'"'}, std::byte{'a'}, std::byte{'"'},
            std::byte{':'}, std::byte{'1'}, std::byte{'}'}};
        // Not multiple of 4 intentionally — plain Zstd only.
        std::vector<std::byte> shuffle_payload(4096);
        for (std::size_t i = 0; i < shuffle_payload.size(); ++i) {
            shuffle_payload[i] =
                static_cast<std::byte>((i * 19u + 5u) & 0xffu);
        }
        std::vector<std::byte> plain_payload(3000);
        for (std::size_t i = 0; i < plain_payload.size(); ++i) {
            plain_payload[i] =
                static_cast<std::byte>((i * 23u + 7u) & 0xffu);
        }

        const ChunkKey json_key{
            .fourcc = FOURCC_PROJ,
            .instance_uuid = fixed_uuid(96'001),
        };
        const ChunkKey shuffle_key{
            .fourcc = FOURCC_CKPT,
            .instance_uuid = fixed_uuid(96'002),
        };
        const ChunkKey plain_key{
            .fourcc = FOURCC_SPLT,
            .instance_uuid = fixed_uuid(96'003),
        };
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path,
                CreateOptions{
                    .project_uuid = fixed_uuid(96'000),
                    .file_uuid = fixed_uuid(96'010),
                    .role = ContainerRole::Master,
                    .creation_time_unix_ns = FIXED_CREATION_NS + 96'000,
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            require_status(writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(96'020),
                .snapshot_uuid = fixed_uuid(96'021),
                .wallclock_unix_ns = FIXED_COMMIT_NS + 96'100,
            }));
            const std::uint64_t preflight =
                jsonish.size() + shuffle_payload.size() + plain_payload.size();
            require_status(writer.preflight(preflight));
            require_status(writer.write_chunk(
                json_key, jsonish,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ZstdFramed,
                    .tensor_payload = false,
                    .block_crcs = false,
                    .expected_stream_bytes = std::nullopt,
                }));
            require_status(writer.write_chunk(
                shuffle_key, shuffle_payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ByteShuffleZstdFramed,
                    .tensor_payload = true,
                    .block_crcs = false,
                    .expected_stream_bytes = shuffle_payload.size(),
                }));
            require_status(writer.write_chunk(
                plain_key, plain_payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ZstdFramed,
                    .tensor_payload = true,
                    .block_crcs = false,
                    .expected_stream_bytes = plain_payload.size(),
                }));
            require_status(writer.commit());
        }

        ProjectReader reader = require_result(ProjectReader::open(path));
        EXPECT_TRUE(reader.commit()
                        .required_reader_capabilities
                        .contains(CHUNK_ZSTD_V1));
        EXPECT_TRUE(reader.commit()
                        .required_reader_capabilities
                        .contains(CHUNK_BYTESHUFFLE_ZSTD_V1));

        const ChunkInfo* j = reader.find(json_key.fourcc, json_key.instance_uuid);
        const ChunkInfo* s =
            reader.find(shuffle_key.fourcc, shuffle_key.instance_uuid);
        const ChunkInfo* p =
            reader.find(plain_key.fourcc, plain_key.instance_uuid);
        ASSERT_NE(j, nullptr);
        ASSERT_NE(s, nullptr);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(j->compression, Compression::ZstdFramed);
        EXPECT_EQ(s->compression, Compression::ByteShuffleZstdFramed);
        EXPECT_EQ(p->compression, Compression::ZstdFramed);

        auto j_bytes = require_result(reader.read_chunk(*j));
        auto s_bytes = require_result(reader.read_chunk(*s));
        auto p_bytes = require_result(reader.read_chunk(*p));
        EXPECT_EQ(j_bytes, jsonish);
        EXPECT_EQ(s_bytes, shuffle_payload);
        EXPECT_EQ(p_bytes, plain_payload);
    }

    TEST(P8CompatibilityTest, StreamingZstdLargeChunkRoundTrip) {
        // Above the 8 MiB stream threshold; exercises ZSTD_compressStream2
        // path and measures that logical bytes round-trip bit-exactly.
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "large-stream-zstd.licht";

        constexpr std::size_t kSize = 9ull * 1024 * 1024; // 9 MiB
        std::vector<std::byte> payload(kSize);
        for (std::size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<std::byte>((i * 31u + 11u) & 0xffu);
        }
        // Force %4==0 for ByteShuffle path too (9 MiB is divisible by 4).
        ASSERT_EQ(payload.size() % 4, 0u);

        const ChunkKey key{
            .fourcc = FOURCC_CKPT,
            .instance_uuid = fixed_uuid(97'001),
        };
        {
            ProjectWriter writer = require_result(ProjectWriter::create(
                path,
                CreateOptions{
                    .project_uuid = fixed_uuid(97'000),
                    .file_uuid = fixed_uuid(97'002),
                    .role = ContainerRole::Master,
                    .creation_time_unix_ns = FIXED_CREATION_NS + 97'000,
                    .index_compression = IndexCompression::Zstd,
                    .disk_reserve_bytes = 0,
                }));
            require_status(writer.plan_commit(CommitOptions{
                .kind = CommitKind::Explicit,
                .commit_uuid = fixed_uuid(97'003),
                .snapshot_uuid = key.instance_uuid,
                .wallclock_unix_ns = FIXED_COMMIT_NS + 97'100,
            }));
            require_status(writer.preflight(payload.size()));
            require_status(writer.write_chunk(
                key, payload,
                ChunkWriteOptions{
                    .chunk_version = 1,
                    .compression = Compression::ByteShuffleZstdFramed,
                    .tensor_payload = true,
                    .block_crcs = false,
                    .expected_stream_bytes = payload.size(),
                }));
            require_status(writer.commit());
        }

        ProjectReader reader = require_result(ProjectReader::open(path));
        const ChunkInfo* row = reader.find(key.fourcc, key.instance_uuid);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->compression, Compression::ByteShuffleZstdFramed);
        auto inflated = require_result(reader.read_chunk(*row));
        ASSERT_EQ(inflated.size(), payload.size());
        EXPECT_TRUE(std::equal(
            inflated.begin(), inflated.end(), payload.begin()));
    }

    // Document-backed REFS round-trips live here (not test_project_chapters.cpp)
    // so the Windows CPU-only format composition stays free of ProjectDocument /
    // CUDA includes. Chapter-only REFS mint/resolve tests remain in chapters.
    ParameterManagerSnapshot parameter_snapshot_for_refs() {
        ParameterManagerSnapshot result;
        result.active_strategy = "mrnf";
        result.mcmc_session =
            lfs::core::param::OptimizationParameters::mcmc_defaults();
        result.mrnf_session =
            lfs::core::param::OptimizationParameters::mrnf_defaults();
        result.igs_session =
            lfs::core::param::OptimizationParameters::igs_plus_defaults();
        result.mcmc_current = result.mcmc_session;
        result.mrnf_current = result.mrnf_session;
        result.igs_current = result.igs_session;
        result.mrnf_current_references.background_image_reference =
            uuid_literal("10000000-0000-4000-8000-000000000090");
        result.mrnf_current_references.ppisp_reference =
            uuid_literal("10000000-0000-4000-8000-000000000091");
        result.dataset.centralize_dataset = "cameras";
        result.dataset.timelapse_images = {"frame-a.png", "frame-b.png"};
        return result;
    }

    TEST(ProjectChapterTest,
         ParameterReferenceBindingsRoundTripThroughDocumentSaveOpen) {
        TemporaryDirectory temporary;
        const fs::path project_root = temporary.path;
        const fs::path bg_path = project_root / "bg.png";
        const fs::path ppisp_path = project_root / "model.ppisp";
        {
            std::ofstream stream(bg_path, std::ios::binary);
            stream << "bg";
        }
        {
            std::ofstream stream(ppisp_path, std::ios::binary);
            stream << "ppisp";
        }

        auto document = ProjectDocument::create(
            uuid_literal("40000000-0000-4000-8000-000000000001"), 100);
        ASSERT_TRUE(document) << lfs::format_for_developer(document.error());

        auto snapshot = parameter_snapshot_for_refs();
        snapshot.mrnf_current.bg_image_path = bg_path;
        snapshot.mrnf_current.ppisp_sidecar_path = ppisp_path;
        snapshot.mrnf_current_references = {};

        auto bg_uuid = upsert_path_reference(
            document->edit_references(), project_root, bg_path,
            "presets.mrnf.current.background_image", "background_image");
        auto ppisp_uuid = upsert_path_reference(
            document->edit_references(), project_root, ppisp_path,
            "presets.mrnf.current.ppisp_sidecar", "ppisp_sidecar");
        ASSERT_TRUE(bg_uuid);
        ASSERT_TRUE(ppisp_uuid);
        snapshot.mrnf_current_references.background_image_reference = *bg_uuid;
        snapshot.mrnf_current_references.ppisp_reference = *ppisp_uuid;
        // Paths are not serialized; only reference UUIDs.
        snapshot.mrnf_current.bg_image_path.clear();
        snapshot.mrnf_current.ppisp_sidecar_path.clear();
        require_status(document->edit_parameters().set_snapshot(snapshot));

        const auto path = temporary.path / "params-refs.licht";
        auto saved = document->save(
            path,
            ProjectDocumentSaveOptions{
                .file_uuid =
                    uuid_literal(
                        "40000000-0000-4000-8000-000000000099"),
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());

        auto reopened = ProjectDocument::open(path);
        ASSERT_TRUE(reopened) << lfs::format_for_developer(reopened.error());
        auto restored = reopened->parameters().snapshot();
        ASSERT_TRUE(restored);
        ASSERT_TRUE(restored->mrnf_current_references.background_image_reference);
        ASSERT_TRUE(restored->mrnf_current_references.ppisp_reference);
        EXPECT_EQ(
            *restored->mrnf_current_references.background_image_reference,
            *bg_uuid);
        EXPECT_EQ(
            *restored->mrnf_current_references.ppisp_reference, *ppisp_uuid);
        EXPECT_TRUE(restored->mrnf_current.bg_image_path.empty());
        EXPECT_TRUE(restored->mrnf_current.ppisp_sidecar_path.empty());

        const auto resolved_bg = resolve_path_reference(
            reopened->references(), project_root,
            *restored->mrnf_current_references.background_image_reference);
        const auto resolved_ppisp = resolve_path_reference(
            reopened->references(), project_root,
            *restored->mrnf_current_references.ppisp_reference);
        ASSERT_TRUE(resolved_bg);
        ASSERT_TRUE(resolved_ppisp);
        EXPECT_EQ(resolved_bg->lexically_normal(), bg_path.lexically_normal());
        EXPECT_EQ(resolved_ppisp->lexically_normal(),
                  ppisp_path.lexically_normal());
    }

    TEST(ProjectChapterTest,
         ViewAndSequencerReferenceUuidsRoundTripWithResolvedPaths) {
        TemporaryDirectory temporary;
        const fs::path project_root = temporary.path;
        const fs::path env_path = project_root / "env.hdr";
        const fs::path seq_dir = project_root / "seq";
        fs::create_directories(seq_dir);
        {
            std::ofstream stream(env_path, std::ios::binary);
            stream << "hdr";
        }
        {
            std::ofstream stream(seq_dir / "a.ply", std::ios::binary);
            stream << "ply";
        }

        auto document = ProjectDocument::create(
            uuid_literal("41000000-0000-4000-8000-000000000001"), 100);
        ASSERT_TRUE(document) << lfs::format_for_developer(document.error());

        auto env_uuid = upsert_path_reference(
            document->edit_references(), project_root, env_path,
            "view.environment", "environment_map");
        auto seq_uuid = upsert_path_reference(
            document->edit_references(), project_root, seq_dir,
            "sequencer.ply_sequence.clip", "ply_sequence_directory");
        ASSERT_TRUE(env_uuid);
        ASSERT_TRUE(seq_uuid);

        require_status(document->edit_view().dom().set(
            "render_settings.environment_reference_uuid",
            env_uuid->to_string()));
        require_status(document->edit_sequencer().dom().set_json(
            "ply_sequences",
            lfs::io::JsonChapterDom::Json::array({
                {
                    {"node_name", "clip"},
                    {"node_uuid",
                     uuid_literal(
                         "41000000-0000-4000-8000-000000000002")
                         .to_string()},
                    {"directory_reference_uuid",
                     seq_uuid->to_string()},
                    {"directory_hint", "seq"},
                    {"frames",
                     lfs::io::JsonChapterDom::Json::array(
                         {{{"locator", "a.ply"},
                           {"node_name", "a"},
                           {"node_uuid",
                            uuid_literal(
                                "41000000-0000-4000-8000-000000000003")
                                .to_string()}}})},
                    {"fps", 24.0f},
                },
            })));

        const auto path = temporary.path / "session-refs.licht";
        auto saved = document->save(
            path,
            ProjectDocumentSaveOptions{
                .file_uuid =
                    uuid_literal(
                        "41000000-0000-4000-8000-000000000099"),
                .index_compression =
                    IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());

        auto reopened = ProjectDocument::open(path);
        ASSERT_TRUE(reopened) << lfs::format_for_developer(reopened.error());
        const auto env_json = reopened->view().dom().get_json(
            "render_settings.environment_reference_uuid");
        ASSERT_TRUE(env_json && env_json->is_string());
        EXPECT_EQ(env_json->get<std::string>(), env_uuid->to_string());
        const auto clips =
            reopened->sequencer().dom().get_json("ply_sequences");
        ASSERT_TRUE(clips && clips->is_array() && !clips->empty());
        EXPECT_EQ((*clips)[0]["directory_reference_uuid"].get<std::string>(),
                  seq_uuid->to_string());

        const auto resolved_env = resolve_path_reference(
            reopened->references(), project_root, *env_uuid);
        const auto resolved_seq = resolve_path_reference(
            reopened->references(), project_root, *seq_uuid,
            fs::path{"seq"});
        ASSERT_TRUE(resolved_env);
        ASSERT_TRUE(resolved_seq);
        EXPECT_EQ(resolved_env->lexically_normal(),
                  env_path.lexically_normal());
        EXPECT_EQ(resolved_seq->lexically_normal(),
                  seq_dir.lexically_normal());
    }

} // namespace
