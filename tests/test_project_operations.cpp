/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/checkpoint_format.hpp"
#include "core/user_paths.hpp"
#include "io/project_document.hpp"
#include "io/project_operations.hpp"
#include "licht_test_support.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    std::vector<std::byte> checkpoint_payload(const std::int32_t iteration) {
        lfs::core::CheckpointHeader header{};
        header.iteration = iteration;
        header.num_gaussians = 4;
        header.sh_degree = 2;
        std::vector<std::byte> result(sizeof(header));
        std::memcpy(result.data(), &header, sizeof(header));
        return result;
    }

    lfs::Result<ProjectDocumentSaveReport> save_document(
        ProjectDocument& document, const fs::path& path,
        const lfs::core::Uuid& file_uuid = fixed_uuid(2000)) {
        ProjectDocumentSaveOptions options;
        options.file_uuid = file_uuid;
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        options.disk_reserve_bytes = 0;
        return document.save(path, options);
    }

    fs::path make_document(const fs::path& path) {
        auto document = require_result(ProjectDocument::create(fixed_uuid(1000), 1'700'000'000'000'000'000));
        static_cast<void>(require_result(save_document(document, path)));
        return path;
    }

    void flip_byte(const fs::path& path, const std::uint64_t offset) {
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(stream);
        stream.seekg(static_cast<std::streamoff>(offset));
        char byte = 0;
        stream.read(&byte, 1);
        ASSERT_EQ(stream.gcount(), 1);
        stream.seekp(static_cast<std::streamoff>(offset));
        byte ^= static_cast<char>(0x5a);
        stream.write(&byte, 1);
        ASSERT_TRUE(stream);
    }

    void write_u64(const fs::path& path, const std::uint64_t offset,
                   const std::uint64_t value) {
        std::array<std::byte, 8> bytes{};
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::byte>((value >> (index * 8)) & 0xff);
        }
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(stream);
        stream.seekp(static_cast<std::streamoff>(offset));
        stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(stream);
    }

    TEST(ProjectOperations, RestoreOlderSaveRekeysAndRefusesCollision) {
        TemporaryDirectory temporary;
        const auto source = make_document(temporary.path / "source.licht");
        auto document = require_result(ProjectDocument::open(source));
        static_cast<void>(document.edit_project().dom().set("old_marker", "old"));
        static_cast<void>(require_result(document.save(source, ProjectDocumentSaveOptions{
                                                                   .index_compression = IndexCompression::StoredForDeterministicTests,
                                                                   .disk_reserve_bytes = 0,
                                                               })));
        const auto destination = temporary.path / "restored.licht";
        const auto card = require_result(restore_save(source, 1, destination));
        const auto source_card = require_result(inspect_project_card(source));
        EXPECT_NE(card.project_uuid, source_card.project_uuid);
        EXPECT_NE(card.file_uuid, source_card.file_uuid);
        EXPECT_EQ(card.generation, 1u);
        EXPECT_FALSE(require_result(inspect_project_details(destination)).card.title.has_value());
        const auto collision = restore_save(source, 1, destination);
        ASSERT_FALSE(collision);
        EXPECT_EQ(collision.error().code(), lfs::ErrorCode::AlreadyExists);
    }

    TEST(ProjectOperations, RestoreSaveResurrectsDeletedCheckpoint) {
        TemporaryDirectory temporary;
        const auto source = make_document(temporary.path / "deleted-checkpoint.licht");
        const auto checkpoint_id = fixed_uuid(3500);
        const auto payload = checkpoint_payload(2200);
        {
            auto document = require_result(ProjectDocument::open(source));
            require_status(document.set_checkpoint(checkpoint_id, require_result(LazyChunkValue::from_owned(payload, checkpoint_id))));
            static_cast<void>(require_result(save_document(document, source)));
        }
        {
            auto document = require_result(ProjectDocument::open(source));
            ASSERT_TRUE(document.remove_checkpoint(checkpoint_id));
            static_cast<void>(require_result(save_document(document, source)));
        }
        const auto before = require_result(inspect_project_card(source));
        ASSERT_EQ(before.generation, 3u);
        auto deleted = require_result(ProjectReader::open(source));
        ASSERT_NE(deleted.find(FOURCC_CKPT, checkpoint_id), nullptr);
        ASSERT_EQ(deleted.find(FOURCC_CKPT, checkpoint_id)->row_kind, RowKind::Tombstone);
        const auto restored = require_result(restore_save(source, 2, source));
        EXPECT_EQ(restored.project_uuid, before.project_uuid);
        EXPECT_EQ(restored.generation, 4u);
        auto reader = require_result(ProjectReader::open(source));
        const auto* checkpoint = reader.find(FOURCC_CKPT, checkpoint_id);
        ASSERT_NE(checkpoint, nullptr);
        EXPECT_EQ(require_result(reader.read_chunk(*checkpoint)), payload);

        // Historical tombstones may be replaced, but never a resolution made
        // during the current append, whether it copied or deleted the key.
        for (const bool erase_first : {false, true}) {
            auto writer = require_result(ProjectWriter::append(source));
            require_status(writer.plan_commit());
            require_status(writer.preflight(payload.size()));
            if (erase_first) {
                require_status(writer.erase(checkpoint->key));
            } else {
                require_status(writer.copy_chunk_verbatim(reader, *checkpoint));
            }
            const auto duplicate = writer.copy_chunk_verbatim(reader, *checkpoint);
            ASSERT_FALSE(duplicate);
            EXPECT_EQ(duplicate.error().code(), lfs::ErrorCode::AlreadyExists);
        }
    }

    TEST(ProjectOperations, RestoreSaveWithRetainedThumbnailChunks) {
        TemporaryDirectory temporary;
        const auto source = make_document(temporary.path / "thumbnail-history.licht");
        const auto selected_preview = std::vector<std::byte>{
            std::byte{'p'}, std::byte{'n'}, std::byte{'g'}};
        const auto retained_thumbnail = std::vector<std::byte>{
            std::byte{'o'}, std::byte{'l'}, std::byte{'d'}};
        const auto newer_preview = std::vector<std::byte>{
            std::byte{'n'}, std::byte{'e'}, std::byte{'w'}};
        {
            auto reader = require_result(ProjectReader::open(source));
            auto writer = require_result(ProjectWriter::append(source));
            require_status(writer.plan_commit());
            std::uint64_t planned_bytes = selected_preview.size() + retained_thumbnail.size();
            for (const auto& row : reader.chunks()) {
                if (row.is_live())
                    planned_bytes += row.stored_bytes;
            }
            require_status(writer.preflight(planned_bytes));
            for (const auto& row : reader.chunks()) {
                if (row.is_live())
                    require_status(writer.copy_chunk_verbatim(reader, row));
            }
            require_status(writer.write_chunk(
                ChunkKey{FOURCC_THMB, fixed_uuid(1001)}, retained_thumbnail));
            require_status(writer.set_preview(selected_preview));
            require_status(writer.commit());
        }
        static_cast<void>(require_result(set_project_preview(source, newer_preview)));

        auto historical_reader = require_result(ProjectReader::open_generation(source, 2));
        ASSERT_TRUE(historical_reader.preview().has_value());
        EXPECT_EQ(require_result(historical_reader.read_preview()), selected_preview);

        const auto restored_path = temporary.path / "restored-thumbnail-history.licht";
        static_cast<void>(require_result(restore_save(source, 2, restored_path)));
        auto restored_reader = require_result(ProjectReader::open(restored_path));
        EXPECT_EQ(require_result(restored_reader.read_preview()), selected_preview);
        EXPECT_EQ(std::ranges::count_if(restored_reader.chunks(), [](const auto& row) {
                      return row.is_live() && row.key.fourcc == FOURCC_THMB;
                  }),
                  2);

        const auto damaged = temporary.path / "damaged-thumbnail-history.licht";
        fs::copy_file(source, damaged);
        flip_byte(damaged, HEAD_SLOT_OFFSETS[0] + 200);
        flip_byte(damaged, HEAD_SLOT_OFFSETS[1] + 200);
        const auto repaired_path = temporary.path / "repaired-thumbnail-history.licht";
        static_cast<void>(require_result(repair_project(damaged, repaired_path)));
        auto repaired = require_result(ProjectReader::open(repaired_path));
        EXPECT_EQ(require_result(repaired.read_preview()), newer_preview);
        EXPECT_EQ(std::ranges::count_if(
                      repaired.chunks(), [](const ChunkInfo& row) {
                          return row.is_live() && row.key.fourcc == FOURCC_THMB;
                      }),
                  2);

        static_cast<void>(require_result(restore_save(source, 2, source)));
        auto source_reader = require_result(ProjectReader::open(source));
        EXPECT_EQ(require_result(source_reader.read_preview()), selected_preview);
        EXPECT_EQ(std::ranges::count_if(source_reader.chunks(), [](const auto& row) {
                      return row.is_live() && row.key.fourcc == FOURCC_THMB;
                  }),
                  2);

        const auto ambiguous_destination = temporary.path / "ambiguous-preview.licht";
        const auto ambiguous = restore_save(source, 2, ambiguous_destination);
        ASSERT_FALSE(ambiguous);
        EXPECT_EQ(ambiguous.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(ambiguous.error().user_message(),
                  "The selected save's preview is ambiguous.");
        EXPECT_FALSE(fs::exists(ambiguous_destination));

        const auto ambiguous_repair_source =
            temporary.path / "ambiguous-repair-thumbnail-history.licht";
        fs::copy_file(source, ambiguous_repair_source);
        for (const auto offset : HEAD_SLOT_OFFSETS) {
            write_u64(ambiguous_repair_source, offset + 112, 1);
            flip_byte(ambiguous_repair_source, offset + 200);
        }
        const auto ambiguous_repair_destination =
            temporary.path / "ambiguous-repair-result.licht";
        const auto ambiguous_repair =
            repair_project(ambiguous_repair_source, ambiguous_repair_destination);
        ASSERT_FALSE(ambiguous_repair);
        EXPECT_EQ(ambiguous_repair.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(ambiguous_repair.error().user_message(),
                  "The recovered project preview is ambiguous.");
        EXPECT_FALSE(fs::exists(ambiguous_repair_destination));
    }

    TEST(ProjectOperations, RebindCheckpointKeepsRecoveryCopy) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "resume.licht";
        auto document = require_result(ProjectDocument::create(fixed_uuid(1100), 1'700'000'000'000'000'000));
        const auto first = fixed_uuid(1101);
        const auto second = fixed_uuid(1102);
        require_status(document.set_checkpoint(
            first, require_result(LazyChunkValue::from_owned(checkpoint_payload(10), first))));
        require_status(document.set_checkpoint(
            second, require_result(LazyChunkValue::from_owned(checkpoint_payload(20), second))));
        SceneNodeRecord node;
        node.uuid = fixed_uuid(1110);
        node.type = "splat";
        node.name = "Training model";
        node.payload = PayloadBinding{
            .fourcc = "CKPT",
            .instance_uuid = second,
            .source_kind = "checkpoint"};
        require_status(document.edit_scene_graph().upsert_node(node));
        require_status(document.edit_scene_graph().set_training_model_uuid(node.uuid));
        static_cast<void>(require_result(save_document(document, path, fixed_uuid(1111))));

        const auto before = require_result(inspect_project_card(path));
        const auto card = require_result(rebind_checkpoint(path, first));
        EXPECT_EQ(card.open_state, OpenState::Open);
        const auto backups = require_result(lfs::core::UserPaths::resolve()).backupDir() /
                             "contents" / before.project_uuid.to_string();
        EXPECT_TRUE(fs::is_regular_file(backups / (before.commit_uuid.to_string() + ".licht.bak")));
        const auto details = require_result(inspect_project_details(path));
        ASSERT_TRUE(details.scene_graph.training_node_id.has_value());
        ASSERT_TRUE(details.retained_checkpoints.size() >= 2);
        const auto bound = std::ranges::find_if(
            details.retained_checkpoints,
            [](const auto& checkpoint) { return checkpoint.binds_scene_graph; });
        ASSERT_NE(bound, details.retained_checkpoints.end());
        EXPECT_EQ(bound->instance_uuid, first);
        const auto resumed_again = require_result(rebind_checkpoint(path, second));
        EXPECT_NE(resumed_again.commit_uuid, card.commit_uuid);
        EXPECT_TRUE(fs::is_regular_file(backups / (card.commit_uuid.to_string() + ".licht.bak")));
        EXPECT_EQ(std::ranges::count_if(fs::directory_iterator(temporary.path),
                                        [](const auto& entry) { return entry.path().extension() == ".licht"; }),
                  1);
    }

    TEST(ProjectOperations, CompactPreservesRetainedCheckpointsAndWarns) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "compact.licht";
        auto document = require_result(ProjectDocument::create(fixed_uuid(1200), 1'700'000'000'000'000'000));
        const auto first = fixed_uuid(1201);
        const auto second = fixed_uuid(1202);
        require_status(document.set_checkpoint(
            first, require_result(LazyChunkValue::from_owned(checkpoint_payload(10), first))));
        require_status(document.set_checkpoint(
            second, require_result(LazyChunkValue::from_owned(checkpoint_payload(20), second))));
        static_cast<void>(require_result(save_document(document, path, fixed_uuid(1210))));
        const auto before = require_result(inspect_project_details(path));
        const auto compacted = require_result(compact_project_file(path));
        EXPECT_NE(compacted.diagnostic.find("older save points"), std::string::npos);
        const auto after = require_result(inspect_project_details(path));
        ASSERT_EQ(after.retained_checkpoints.size(), before.retained_checkpoints.size());
        EXPECT_EQ(after.card.commit_kind, CommitKind::Compaction);
        EXPECT_EQ(after.save_history.size(), 1u);
    }

    TEST(ProjectOperations, ReduceSizeDropsOnlyUnboundCheckpoints) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "reduce.licht";
        auto document = require_result(ProjectDocument::create(fixed_uuid(1250),
                                                               1'700'000'000'000'000'000));
        const auto bound_uuid = fixed_uuid(1251);
        const auto old_uuid = fixed_uuid(1252);
        require_status(document.set_checkpoint(
            bound_uuid, require_result(LazyChunkValue::from_owned(
                            checkpoint_payload(20), bound_uuid))));
        require_status(document.set_checkpoint(
            old_uuid, require_result(LazyChunkValue::from_owned(
                          checkpoint_payload(10), old_uuid))));
        const auto node_uuid = fixed_uuid(1253);
        require_status(document.edit_scene_graph().upsert_node(SceneNodeRecord{
            .uuid = node_uuid,
            .type = "splat",
            .name = "Training model",
            .payload = PayloadBinding{
                .fourcc = "CKPT",
                .instance_uuid = bound_uuid,
                .source_kind = "checkpoint",
            },
        }));
        require_status(document.edit_scene_graph().set_training_model_uuid(node_uuid));
        static_cast<void>(require_result(save_document(document, path, fixed_uuid(1254))));

        const auto plan = require_result(plan_reduce_size(path));
        ASSERT_EQ(plan.retained_checkpoints.size(), 2u);
        EXPECT_TRUE(plan.drop_checkpoints.enabled);
        const auto reduced = require_result(reduce_size(path, true, false));
        EXPECT_EQ(reduced.checkpoints_removed, 1u);
        EXPECT_TRUE(fs::is_regular_file(reduced.recovery_copy));
        const auto details = require_result(inspect_project_details(path));
        ASSERT_EQ(details.retained_checkpoints.size(), 1u);
        EXPECT_TRUE(details.retained_checkpoints.front().binds_scene_graph);
        EXPECT_EQ(details.retained_checkpoints.front().instance_uuid, bound_uuid);
    }

    TEST(ProjectOperations, ExportVisibleWriterFixtureAndRejectTruncatedRepair) {
#ifndef LFS_FORMAT_TEST_TARGET
        TemporaryDirectory temporary;
        const auto path = temporary.path / "export.licht";
        auto document = require_result(ProjectDocument::create(fixed_uuid(1260),
                                                               1'700'000'000'000'000'000));
        const auto node_uuid = fixed_uuid(1261);
        require_status(document.edit_scene_graph().upsert_node(SceneNodeRecord{
            .uuid = node_uuid,
            .type = "splat",
            .name = "Visible splat",
            .payload = PayloadBinding{
                .fourcc = "SPLT",
                .instance_uuid = node_uuid,
                .source_kind = "ply",
            },
        }));
        auto splat = require_result(SplatChapterPayload::capture(
            *make_splat(4), SplatSourceKind::ImportedPly, false));
        require_status(document.set_splat(node_uuid, std::move(splat)));
        static_cast<void>(require_result(save_document(document, path, fixed_uuid(1262))));

        const auto output = temporary.path / "visible.sog";
        const auto exported = require_result(export_project_as(
            path, ProjectExportFormat::Sog, output));
        EXPECT_EQ(exported.gaussian_count, 4u);
        EXPECT_TRUE(fs::is_regular_file(output));
        EXPECT_GT(exported.bytes_written, 0u);
#else
        GTEST_SKIP() << "Export requires the full I/O test composition";
#endif

#ifndef LFS_FORMAT_TEST_TARGET
        const char* fixture_root = std::getenv("LFS_PROJECT_INSPECT_FIXTURES");
        if (!fixture_root) {
            GTEST_SKIP() << "LFS_PROJECT_INSPECT_FIXTURES is not set";
        }
        const auto truncated = fs::path(fixture_root) / "truncated_tail.licht";
        if (!fs::is_regular_file(truncated)) {
            GTEST_SKIP() << "truncated_tail.licht fixture is unavailable";
        }
        const auto repaired = temporary.path / "repaired.licht";
        const auto result = repair_project(truncated, repaired);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_FALSE(fs::exists(repaired));
#endif
    }

    TEST(ProjectOperations, CoversCopiedInspectFixtures) {
        const char* fixture_root = std::getenv("LFS_PROJECT_INSPECT_FIXTURES");
        if (!fixture_root) {
            GTEST_SKIP() << "LFS_PROJECT_INSPECT_FIXTURES is not set";
        }
        const auto root = fs::path(fixture_root);
        constexpr std::array names{
            "blank_heads.licht", "both_heads_crc.licht", "corrupt_head_crc.licht",
            "heads_only.licht", "missing_preview.licht", "newer_reader_version.licht",
            "role_sidecar.licht", "truncated_tail.licht", "user_gen1.licht"};
        TemporaryDirectory temporary;
        for (const auto name : names) {
            const auto path = root / name;
            ASSERT_TRUE(fs::is_regular_file(path)) << path;
            const auto card = require_result(inspect_project_card(path));
            EXPECT_FALSE(card.path.empty()) << name;
        }
        const auto truncated = root / "truncated_tail.licht";
        const auto destination = temporary.path / "truncated-repaired.licht";
        const auto repaired = repair_project(truncated, destination);
        ASSERT_FALSE(repaired);
        EXPECT_EQ(repaired.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_FALSE(fs::exists(destination));
    }

    TEST(ProjectOperations, VerifyCanBeCanceledAndPreviewLicenseTitleAreExplicitSaves) {
        TemporaryDirectory temporary;
        const auto path = make_document(temporary.path / "metadata.licht");
        auto canceled = require_result(verify_project_file(
            path, {}, [] { return true; }));
        EXPECT_EQ(canceled.status, ProjectVerificationStatus::Canceled);
        auto verified = require_result(verify_project_file(path));
        EXPECT_EQ(verified.status, ProjectVerificationStatus::Verified);
        const std::vector<std::byte> preview{std::byte{'p'}, std::byte{'n'}, std::byte{'g'}};
        static_cast<void>(require_result(set_project_preview(path, preview)));
        static_cast<void>(require_result(set_project_license(path, "CC-BY-4.0", "Notice")));
        auto titled = require_result(set_project_title(path, "A useful title"));
        ASSERT_TRUE(titled.title.has_value());
        EXPECT_EQ(*titled.title, "A useful title");
        auto details = require_result(inspect_project_details(path));
        ASSERT_TRUE(details.license.has_value());
        EXPECT_EQ(details.license->identifier, "CC-BY-4.0");
        static_cast<void>(require_result(clear_project_license(path)));
        details = require_result(inspect_project_details(path));
        EXPECT_FALSE(details.license.has_value());
    }

    TEST(ProjectOperations, MutationsUseTheClosedFileWriterLockMessage) {
        TemporaryDirectory temporary;
        const auto path = make_document(temporary.path / "locked.licht");
        auto lease = require_result(WriterLockLease::acquire(path));
        const auto result = set_project_title(path, "blocked");
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::Unavailable);
        EXPECT_EQ(result.error().user_message(),
                  "The project is open for writing in another LichtFeld Studio");
    }

    TEST(ProjectOperations, ClosedFilePreviewWriteDoesNotIncludeUnsavedLiveEdits) {
        TemporaryDirectory temporary;
        const auto path = make_document(temporary.path / "stale-preview.licht");
        auto live = require_result(ProjectDocument::open(path));
        require_status(live.edit_project().dom().set("marker", "unsaved"));
        const std::vector<std::byte> preview{
            std::byte{'p'}, std::byte{'n'}, std::byte{'g'}};
        static_cast<void>(require_result(set_project_preview(path, preview)));
        auto disk = require_result(ProjectDocument::open(path));
        const auto marker = disk.project().dom().get<std::string>("marker");
        EXPECT_TRUE(!marker.has_value() || *marker != "unsaved");
        EXPECT_EQ(
            require_result(require_result(ProjectReader::open(path)).read_preview()),
            preview);
        EXPECT_EQ(live.project().dom().get<std::string>("marker"),
                  std::optional<std::string>{"unsaved"});
    }

    TEST(ProjectOperations, LiveDocumentThumbnailOnlyWritePreservesDirtyChapters) {
        TemporaryDirectory temporary;
        const auto path = make_document(temporary.path / "thumb-only.licht");
        auto live = require_result(ProjectDocument::open(path));
        require_status(live.edit_project().dom().set("marker", "unsaved"));
        ASSERT_TRUE(live.dirty());
        const std::vector<std::byte> preview{
            std::byte{'p'}, std::byte{'n'}, std::byte{'g'}};
        ProjectDocumentSaveOptions options;
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        options.disk_reserve_bytes = 0;
        options.regenerate_dataset_preview = false;
        static_cast<void>(require_result(live.save_preview(preview, options)));
        EXPECT_TRUE(live.dirty());
        EXPECT_EQ(live.project().dom().get<std::string>("marker"),
                  std::optional<std::string>{"unsaved"});
        auto disk = require_result(ProjectDocument::open(path));
        const auto disk_marker = disk.project().dom().get<std::string>("marker");
        EXPECT_TRUE(!disk_marker.has_value() || *disk_marker != "unsaved");
        EXPECT_EQ(
            require_result(require_result(ProjectReader::open(path)).read_preview()),
            preview);
        static_cast<void>(require_result(live.save(path, options)));
        auto saved = require_result(ProjectDocument::open(path));
        const auto saved_marker = saved.project().dom().get<std::string>("marker");
        ASSERT_TRUE(saved_marker.has_value());
        EXPECT_EQ(*saved_marker, "unsaved");
        EXPECT_EQ(
            require_result(require_result(ProjectReader::open(path)).read_preview()),
            preview);
    }

    TEST(ProjectOperations, ThumbnailPreservesUnsavedLazyPayloadsAndRemovals) {
        TemporaryDirectory temporary;
        const auto path = make_document(temporary.path / "thumb-payloads.licht");
        auto live = require_result(ProjectDocument::open(path));
        const auto removed = fixed_uuid(6101);
        const auto changed = fixed_uuid(6102);
        const auto added = fixed_uuid(6103);
        const auto old_payload = checkpoint_payload(1);
        const auto new_payload = checkpoint_payload(2);
        for (const auto id : {removed, changed}) {
            require_status(live.set_checkpoint(id, require_result(LazyChunkValue::from_owned(old_payload, id))));
        }
        static_cast<void>(require_result(save_document(live, path)));
        ASSERT_TRUE(live.remove_checkpoint(removed));
        for (const auto id : {changed, added}) {
            require_status(live.set_checkpoint(id, require_result(LazyChunkValue::from_owned(new_payload, id))));
        }
        const std::vector<std::byte> preview{std::byte{'p'}, std::byte{'n'}, std::byte{'g'}};
        static_cast<void>(require_result(live.save_preview(preview)));
        ASSERT_TRUE(live.dirty());
        EXPECT_EQ(live.find_checkpoint(removed), nullptr);
        ASSERT_NE(live.find_checkpoint(added), nullptr);
        auto disk = require_result(ProjectReader::open(path));
        ASSERT_NE(disk.find(FOURCC_CKPT, removed), nullptr);
        EXPECT_EQ(disk.find(FOURCC_CKPT, removed)->row_kind, RowKind::Live);
        EXPECT_EQ(require_result(disk.read_chunk(*disk.find(FOURCC_CKPT, changed))), old_payload);
        EXPECT_EQ(disk.find(FOURCC_CKPT, added), nullptr);
        static_cast<void>(require_result(save_document(live, path)));
        auto saved = require_result(ProjectReader::open(path));
        EXPECT_EQ(saved.find(FOURCC_CKPT, removed)->row_kind, RowKind::Tombstone);
        for (const auto id : {changed, added}) {
            ASSERT_NE(saved.find(FOURCC_CKPT, id), nullptr);
            EXPECT_EQ(require_result(saved.read_chunk(*saved.find(FOURCC_CKPT, id))), new_payload);
        }
        EXPECT_EQ(require_result(saved.read_preview()), preview);
    }

    TEST(ProjectOperations, LiveDocumentPreviewSaveKeepsUnsavedEdits) {
        TemporaryDirectory temporary;
        const auto path = make_document(temporary.path / "live-preview.licht");
        auto live = require_result(ProjectDocument::open(path));
        require_status(live.edit_project().dom().set("marker", "unsaved"));
        const std::vector<std::byte> preview{
            std::byte{'p'}, std::byte{'n'}, std::byte{'g'}};
        ProjectDocumentSaveOptions options;
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        options.disk_reserve_bytes = 0;
        options.preview_png = preview;
        options.regenerate_dataset_preview = false;
        static_cast<void>(require_result(live.save(path, options)));
        auto disk = require_result(ProjectDocument::open(path));
        const auto marker = disk.project().dom().get<std::string>("marker");
        ASSERT_TRUE(marker.has_value());
        EXPECT_EQ(*marker, "unsaved");
        EXPECT_EQ(
            require_result(require_result(ProjectReader::open(path)).read_preview()),
            preview);
    }

} // namespace
