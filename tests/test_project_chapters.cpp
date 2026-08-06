/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_chapters.hpp"
#include "licht_test_support.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    ParameterManagerSnapshot parameter_snapshot() {
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

    TEST(ProjectChapterTest, ProjectTypedMutationRetainsUnknownSubtrees) {
        const auto project_id = uuid_literal("10000000-0000-4000-8000-000000000001");
        const auto decision_id = uuid_literal("10000000-0000-4000-8000-000000000002");
        const auto node_id = uuid_literal("10000000-0000-4000-8000-000000000003");
        const std::string source = std::format(
            R"({{
  "schema_version": 1,
  "project_uuid": "{}",
  "created_at_unix_ns": 1,
  "modified_at_unix_ns": 2,
  "manifest": {{
    "application_name": "LichtFeld Studio",
    "application_version": {{"major":1,"minor":2,"patch":3}},
    "schema_version": {{"major":1,"minor":0,"patch":0}},
    "minimum_reader_version": {{"major":1,"minor":0,"patch":0}},
    "minimum_safe_writer_version": {{"major":1,"minor":0,"patch":0}},
    "required_capabilities": [],
    "optional_capabilities": [],
    "future_manifest": {{"nested":[1,2,{{"x":3}}]}}
  }},
  "georeference": {{
    "world_origin":[1.0,2.0,3.0],
    "world_unit_scale":1.0,
    "world_origin_provenance":"import",
    "future_georef":[{{"opaque":true}}]
  }},
  "embed_decisions": [{{
    "uuid":"{}",
    "node_uuid":"{}",
    "payload_fourcc":"SPLT",
    "decision":"embedded",
    "reason":"imported",
    "future_element":{{"array":[1,{{"opaque":"yes"}}]}}
  }}],
  "provenance": [],
  "embedded_payloads": [],
  "future_root":{{"objects":[{{"a":1}},{{"b":2}}]}}
}})",
            project_id.to_string(), decision_id.to_string(), node_id.to_string());
        auto chapter = ProjectChapter::parse(source);
        ASSERT_TRUE(chapter) << lfs::format_for_developer(chapter.error());

        auto manifest = chapter->manifest();
        ASSERT_TRUE(manifest);
        manifest->application_version.patch = 4;
        ASSERT_TRUE(chapter->set_manifest(*manifest));
        ASSERT_TRUE(chapter->upsert_embed_decision(EmbedDecision{
            .uuid = decision_id,
            .node_uuid = node_id,
            .payload_fourcc = "SPLT",
            .decision = "embedded",
            .reference_uuid = std::nullopt,
            .reason = "edited",
        }));
        auto georef = chapter->georeference();
        ASSERT_TRUE(georef);
        georef->world_origin[0] = 8.0;
        ASSERT_TRUE(chapter->set_georeference(*georef));

        const auto reparsed = lfs::io::JsonChapterDom::from_bytes(chapter->to_bytes());
        ASSERT_TRUE(reparsed);
        EXPECT_EQ(reparsed->get<std::int64_t>("manifest.future_manifest.nested"), std::nullopt);
        EXPECT_EQ(reparsed->get_json("manifest.future_manifest"),
                  lfs::io::JsonChapterDom::Json::parse(R"({"nested":[1,2,{"x":3}]})"));
        EXPECT_EQ(reparsed->get_json("georeference.future_georef"),
                  lfs::io::JsonChapterDom::Json::parse(R"([{"opaque":true}])"));
        const auto decision =
            reparsed->array_find("embed_decisions", decision_id.to_string());
        ASSERT_TRUE(decision);
        EXPECT_EQ(decision->get_json("future_element"),
                  lfs::io::JsonChapterDom::Json::parse(
                      R"({"array":[1,{"opaque":"yes"}]})"));
        EXPECT_EQ(reparsed->get_json("future_root"),
                  lfs::io::JsonChapterDom::Json::parse(
                      R"({"objects":[{"a":1},{"b":2}]})"));
    }

    TEST(ProjectChapterTest, ReferencesRetainUnresolvedRowsAndUnknownMembers) {
        const auto ref_id = uuid_literal("20000000-0000-4000-8000-000000000001");
        ReferencesChapter chapter;
        ReferenceRecord record{
            .uuid = ref_id,
            .key = "dataset.root",
            .kind = "dataset",
            .locator =
                {.preferred = "../missing", .base = LocatorBase::Project, .absolute_fallback = "/old/missing"},
            .fingerprint = fingerprint(3, FingerprintKind::File, 100, 200),
            .unresolved = true,
        };
        ASSERT_TRUE(chapter.upsert(record));
        auto element = chapter.dom().array_find("references", ref_id.to_string());
        ASSERT_TRUE(element);
        ASSERT_TRUE(element->set_json(
            "future", lfs::io::JsonChapterDom::Json::parse(
                          R"({"array":[{"opaque":1},2,3]})")));
        const auto before = chapter.to_bytes();

        ProjectChapter project;
        const auto project_id = uuid_literal("20000000-0000-4000-8000-000000000002");
        ASSERT_TRUE(project.set_project_uuid(project_id));
        ASSERT_TRUE(project.set_created_at_unix_ns(1));
        ASSERT_TRUE(project.set_modified_at_unix_ns(2));
        ASSERT_TRUE(project.set_dataset_reference(ref_id));
        SceneGraphChapter scene;
        auto index = build_reverse_reference_index(chapter, project, scene);
        ASSERT_TRUE(index);
        ASSERT_EQ(index->at(ref_id).size(), 1);
        EXPECT_EQ(index->at(ref_id)[0].chapter, "PROJ");

        const auto reopened = ReferencesChapter::from_bytes(before);
        ASSERT_TRUE(reopened);
        const auto after = reopened->to_bytes();
        EXPECT_EQ(before, after);
        auto retained =
            reopened->dom().array_find("references", ref_id.to_string());
        ASSERT_TRUE(retained);
        EXPECT_EQ(retained->get_json("future"),
                  lfs::io::JsonChapterDom::Json::parse(
                      R"({"array":[{"opaque":1},2,3]})"));
        const auto rows = reopened->records();
        ASSERT_TRUE(rows);
        ASSERT_EQ(rows->size(), 1);
        EXPECT_TRUE((*rows)[0].unresolved);
        EXPECT_EQ((*rows)[0], record);
    }

    TEST(ProjectChapterTest, FingerprintMtimeIsFastPathButSizeAndHashAreDecisive) {
        TemporaryDirectory temporary;
        const fs::path path = temporary.path / "reference.bin";
        {
            std::ofstream stream(path, std::ios::binary);
            stream << "same-content";
        }
        auto baseline = fingerprint_path(path);
        ASSERT_TRUE(baseline) << lfs::format_for_developer(baseline.error());

        ReferencesChapter chapter;
        const auto ref_id = uuid_literal("30000000-0000-4000-8000-000000000001");
        ASSERT_TRUE(chapter.upsert(ReferenceRecord{
            .uuid = ref_id,
            .key = "background",
            .kind = "background_image",
            .locator = {.preferred = "reference.bin", .base = LocatorBase::Project},
            .fingerprint = *baseline,
            .unresolved = false,
        }));
        fs::last_write_time(
            path, fs::last_write_time(path) + std::chrono::seconds(5));
        auto mtime_only = chapter.verify_and_refresh(ref_id, path);
        ASSERT_TRUE(mtime_only) << lfs::format_for_developer(mtime_only.error());
        EXPECT_EQ(mtime_only->disposition,
                  FingerprintDisposition::MatchMtimeRefreshed);
        auto refreshed = chapter.find(ref_id);
        ASSERT_TRUE(refreshed);
        ASSERT_TRUE(*refreshed);
        EXPECT_EQ((*refreshed)->fingerprint.mtime_unix_ns,
                  mtime_only->observed->mtime_unix_ns);

        auto stale_metadata = **refreshed;
        stale_metadata.fingerprint.size += 17;
        stale_metadata.fingerprint.mtime_unix_ns -= 31;
        ASSERT_TRUE(chapter.upsert(stale_metadata));
        auto content_match = chapter.verify_and_refresh(ref_id, path);
        ASSERT_TRUE(content_match)
            << lfs::format_for_developer(content_match.error());
        EXPECT_EQ(content_match->disposition,
                  FingerprintDisposition::MatchMtimeRefreshed);
        auto metadata_refreshed = chapter.find(ref_id);
        ASSERT_TRUE(metadata_refreshed && *metadata_refreshed);
        EXPECT_EQ((*metadata_refreshed)->fingerprint.size,
                  fs::file_size(path));
        EXPECT_EQ((*metadata_refreshed)->fingerprint.mtime_unix_ns,
                  content_match->observed->mtime_unix_ns);

        auto relink_metadata = **metadata_refreshed;
        relink_metadata.fingerprint.size += 41;
        relink_metadata.fingerprint.mtime_unix_ns -= 73;
        relink_metadata.unresolved = true;
        ASSERT_TRUE(chapter.upsert(relink_metadata));
        ASSERT_TRUE(chapter.relink(
            ref_id,
            ReferenceLocator{
                .preferred = "moved/reference.bin",
                .base = LocatorBase::Project,
            },
            path, false));
        auto relinked = chapter.find(ref_id);
        ASSERT_TRUE(relinked && *relinked);
        EXPECT_EQ((*relinked)->locator.preferred,
                  "moved/reference.bin");
        EXPECT_EQ((*relinked)->fingerprint.size,
                  fs::file_size(path));
        EXPECT_FALSE((*relinked)->unresolved);

        {
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            stream << "different-size";
        }
        auto size_mismatch = chapter.verify_and_refresh(ref_id, path);
        ASSERT_FALSE(size_mismatch);
        EXPECT_EQ(size_mismatch.error().code(), lfs::ErrorCode::FailedPrecondition);

        {
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            stream << "same-content";
        }
        auto reset = fingerprint_path(path);
        ASSERT_TRUE(reset);
        auto current = chapter.find(ref_id);
        ASSERT_TRUE(current && *current);
        auto accepted = **current;
        accepted.fingerprint = *reset;
        accepted.unresolved = false;
        ASSERT_TRUE(chapter.upsert(accepted));
        {
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            stream << "same-contenu";
        }
        fs::last_write_time(
            path, fs::last_write_time(path) + std::chrono::seconds(5));
        auto hash_mismatch = chapter.verify_and_refresh(ref_id, path);
        ASSERT_FALSE(hash_mismatch);
        EXPECT_EQ(hash_mismatch.error().code(), lfs::ErrorCode::FailedPrecondition);
    }

    TEST(ProjectChapterTest, SceneGraphRetentionHierarchyAndReverseOwners) {
        const auto ref_id = uuid_literal("40000000-0000-4000-8000-000000000001");
        const auto root_id = uuid_literal("40000000-0000-4000-8000-000000000002");
        const auto splat_id = uuid_literal("40000000-0000-4000-8000-000000000003");
        SceneGraphChapter scene;
        ASSERT_TRUE(scene.upsert_node(SceneNodeRecord{
            .uuid = root_id,
            .type = "group",
            .name = "Root",
            .parent_uuid = std::nullopt,
            .child_order = 0,
        }));
        SceneNodeRecord splat{
            .uuid = splat_id,
            .type = "splat",
            .name = "Live RAD",
            .parent_uuid = root_id,
            .child_order = 0,
            .payload =
                PayloadBinding{
                    .fourcc = "REFS",
                    .instance_uuid = ref_id,
                    .reference_uuid = ref_id,
                    .source_kind = "rad",
                },
        };
        ASSERT_TRUE(scene.upsert_node(splat));
        auto element = scene.dom().array_find("nodes", splat_id.to_string());
        ASSERT_TRUE(element);
        ASSERT_TRUE(element->set_json(
            "future_node_state",
            lfs::io::JsonChapterDom::Json::parse(
                R"({"array":[{"new_type":"future"}]})")));
        splat.visible = false;
        ASSERT_TRUE(scene.upsert_node(splat));
        ASSERT_TRUE(scene.validate_hierarchy());
        const auto retained =
            scene.dom().array_find("nodes", splat_id.to_string());
        ASSERT_TRUE(retained);
        EXPECT_EQ(retained->get_json("future_node_state"),
                  lfs::io::JsonChapterDom::Json::parse(
                      R"({"array":[{"new_type":"future"}]})"));

        ReferencesChapter refs;
        ASSERT_TRUE(refs.upsert(ReferenceRecord{
            .uuid = ref_id,
            .key = "rad.live",
            .kind = "rad",
            .locator = {.preferred = "missing.rad", .base = LocatorBase::Project},
            .fingerprint = fingerprint(5, FingerprintKind::File, 100, 200),
            .unresolved = true,
        }));
        ProjectChapter project;
        ASSERT_TRUE(project.set_project_uuid(
            uuid_literal("40000000-0000-4000-8000-000000000004")));
        ASSERT_TRUE(project.set_created_at_unix_ns(1));
        ASSERT_TRUE(project.set_modified_at_unix_ns(2));
        auto index = build_reverse_reference_index(
            refs, project, scene,
            std::array{ReferenceOwnerBinding{
                .reference_uuid = ref_id,
                .chapter = "VIEW",
                .owner_uuid = std::nullopt,
                .field = "environment.reference_uuid",
            }});
        ASSERT_TRUE(index);
        ASSERT_EQ(index->at(ref_id).size(), 2);
        EXPECT_EQ(index->at(ref_id)[0].chapter, "SCNG");
        EXPECT_EQ(index->at(ref_id)[1].chapter, "VIEW");
        EXPECT_EQ(index->at(ref_id)[0].owner_uuid, splat_id);
    }

    TEST(ProjectChapterTest, SceneGraphRejectsParentCyclesAndDuplicateNodeUuids) {
        const auto root_id =
            uuid_literal("41000000-0000-4000-8000-000000000001");
        const auto child_id =
            uuid_literal("41000000-0000-4000-8000-000000000002");
        SceneGraphChapter chapter;
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = root_id,
            .type = "group",
            .name = "Root",
            .child_order = 0,
        }));
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = child_id,
            .type = "group",
            .name = "Child",
            .parent_uuid = root_id,
            .child_order = 0,
        }));

        using Json = lfs::io::JsonChapterDom::Json;
        const auto baseline = Json::parse(chapter.dom().dump());

        auto cycle = baseline;
        cycle["nodes"][0]["parent_uuid"] = child_id.to_string();
        auto cycle_result = SceneGraphChapter::parse(cycle.dump());
        ASSERT_FALSE(cycle_result);
        EXPECT_EQ(cycle_result.error().code(),
                  lfs::ErrorCode::DataLoss);

        auto duplicate = baseline;
        duplicate["nodes"].push_back(duplicate["nodes"][1]);
        auto duplicate_result = SceneGraphChapter::parse(duplicate.dump());
        ASSERT_FALSE(duplicate_result);
        EXPECT_EQ(duplicate_result.error().code(),
                  lfs::ErrorCode::DataLoss);
    }

    TEST(ProjectChapterTest, ParametersMutationRetainsUnknownNestedObjects) {
        ParametersChapter chapter;
        auto snapshot = parameter_snapshot();
        ASSERT_TRUE(chapter.set_snapshot(snapshot));
        ASSERT_TRUE(chapter.dom().set_json(
            "future_root", lfs::io::JsonChapterDom::Json::parse(
                               R"([{"opaque":{"v":1}},2])")));
        ASSERT_TRUE(chapter.dom().set_json(
            "presets.mrnf.current.future_parameter",
            lfs::io::JsonChapterDom::Json::parse(
                R"({"nested":[1,{"v":2}]})")));
        snapshot.mrnf_current.means_lr = 0.123f;
        snapshot.dataset.timelapse_every = 77;
        ASSERT_TRUE(chapter.set_snapshot(snapshot));

        auto reparsed = ParametersChapter::from_bytes(chapter.to_bytes());
        ASSERT_TRUE(reparsed) << lfs::format_for_developer(reparsed.error());
        auto restored = reparsed->snapshot();
        ASSERT_TRUE(restored);
        EXPECT_FLOAT_EQ(restored->mrnf_current.means_lr, 0.123f);
        EXPECT_EQ(
            restored->mrnf_current_references,
            snapshot.mrnf_current_references);
        EXPECT_EQ(restored->dataset.timelapse_every, 77);
        EXPECT_EQ(reparsed->dom().get_json("future_root"),
                  lfs::io::JsonChapterDom::Json::parse(
                      R"([{"opaque":{"v":1}},2])"));
        EXPECT_EQ(reparsed->dom().get_json(
                      "presets.mrnf.current.future_parameter"),
                  lfs::io::JsonChapterDom::Json::parse(
                      R"({"nested":[1,{"v":2}]})"));
        EXPECT_FALSE(reparsed->dom().get_json("presets.mrnf.current.headless"));
        EXPECT_FALSE(
            reparsed->dom().get_json(
                "presets.mrnf.current.bg_image_path"));
        EXPECT_EQ(
            reparsed->dom().get<std::string>(
                "presets.mrnf.current.background_image_reference_uuid"),
            snapshot.mrnf_current_references
                .background_image_reference->to_string());
    }

    TEST(ProjectChapterTest, PathReferenceMintAndResolveRoundTrip) {
        TemporaryDirectory temporary;
        const fs::path project_root = temporary.path / "project";
        const fs::path assets = project_root / "assets";
        fs::create_directories(assets);
        const fs::path env_path = assets / "studio.hdr";
        const fs::path bg_path = assets / "background.png";
        const fs::path seq_dir = assets / "frames";
        fs::create_directories(seq_dir);
        {
            std::ofstream stream(env_path, std::ios::binary);
            stream << "environment-hdr-bytes";
        }
        {
            std::ofstream stream(bg_path, std::ios::binary);
            stream << "background-image-bytes";
        }
        {
            std::ofstream stream(seq_dir / "frame_000.ply", std::ios::binary);
            stream << "ply";
        }

        ReferencesChapter references;
        auto env_uuid = upsert_path_reference(
            references, project_root, env_path, "view.environment",
            "environment_map");
        ASSERT_TRUE(env_uuid) << lfs::format_for_developer(env_uuid.error());
        auto bg_uuid = upsert_path_reference(
            references, project_root, bg_path,
            "presets.mrnf.current.background_image", "background_image");
        ASSERT_TRUE(bg_uuid) << lfs::format_for_developer(bg_uuid.error());
        auto seq_uuid = upsert_path_reference(
            references, project_root, seq_dir, "sequencer.ply_sequence.clip",
            "ply_sequence_directory");
        ASSERT_TRUE(seq_uuid) << lfs::format_for_developer(seq_uuid.error());

        // Re-mint with same key reuses the UUID.
        auto env_again = upsert_path_reference(
            references, project_root, env_path, "view.environment",
            "environment_map");
        ASSERT_TRUE(env_again);
        EXPECT_EQ(*env_again, *env_uuid);

        const auto resolved_env = resolve_path_reference(
            references, project_root, *env_uuid);
        const auto resolved_bg = resolve_path_reference(
            references, project_root, *bg_uuid);
        const auto resolved_seq = resolve_path_reference(
            references, project_root, *seq_uuid, fs::path{"frames"});
        ASSERT_TRUE(resolved_env);
        ASSERT_TRUE(resolved_bg);
        ASSERT_TRUE(resolved_seq);
        EXPECT_EQ(resolved_env->lexically_normal(),
                  env_path.lexically_normal());
        EXPECT_EQ(resolved_bg->lexically_normal(),
                  bg_path.lexically_normal());
        EXPECT_EQ(resolved_seq->lexically_normal(),
                  seq_dir.lexically_normal());

        auto rows = references.records();
        ASSERT_TRUE(rows);
        ASSERT_EQ(rows->size(), 3u);
        for (const auto& row : *rows) {
            EXPECT_EQ(row.locator.base, LocatorBase::Project);
            EXPECT_FALSE(row.unresolved);
        }
    }

} // namespace
