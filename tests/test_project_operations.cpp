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

} // namespace
