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

    TEST(ProjectChapterTest, SceneGraphBatchedUpsertRetainsUnknownNodeMembers) {
        const auto node_id = uuid_literal("43000000-0000-4000-8000-000000000001");
        const std::string source = std::format(
            R"({{
  "schema_version": 1,
  "training_model_uuid": null,
  "nodes": [{{
    "uuid": "{}",
    "type": "group",
    "name": "Root",
    "parent_uuid": null,
    "child_order": 0,
    "local_transform": [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "visible": true,
    "locked": false,
    "training_enabled": true,
    "payload_diverged": false,
    "vendor_extra": {{"keep": 42, "nested": [1, {{"x": true}}]}}
  }}]
}})",
            node_id.to_string());
        auto chapter = SceneGraphChapter::parse(source);
        ASSERT_TRUE(chapter) << lfs::format_for_developer(chapter.error());
        auto found = chapter->find(node_id);
        ASSERT_TRUE(found) << lfs::format_for_developer(found.error());
        ASSERT_TRUE(*found);
        auto node = **found;
        node.name = "Renamed";
        node.visible = false;
        ASSERT_TRUE(chapter->upsert_node(node));

        const auto dumped = lfs::io::JsonChapterDom::Json::parse(chapter->dom().dump());
        ASSERT_EQ(dumped["nodes"].size(), 1);
        EXPECT_EQ(dumped["nodes"][0]["name"], "Renamed");
        EXPECT_EQ(dumped["nodes"][0]["visible"], false);
        const auto extra = dumped["nodes"][0]["vendor_extra"];
        ASSERT_TRUE(extra.is_object());
        EXPECT_EQ(extra["keep"], 42);
        ASSERT_TRUE(extra["nested"].is_array());
        ASSERT_EQ(extra["nested"].size(), 2);
        EXPECT_EQ(extra["nested"][0], 1);
        EXPECT_EQ(extra["nested"][1]["x"], true);
    }

    TEST(SceneGraphChapterScaleTest, UpsertParseAndEnumerateThousandsOfCameraNodes) {
        using clock = std::chrono::steady_clock;
        const auto started = clock::now();

        const auto root_id = uuid_literal("52000000-0000-4000-8000-000000000001");
        const auto group_id = uuid_literal("52000000-0000-4000-8000-000000000002");

        SceneGraphChapter chapter;
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = root_id,
            .type = "dataset",
            .name = "Dataset",
            .parent_uuid = std::nullopt,
            .child_order = 0,
        }));
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = group_id,
            .type = "camera_group",
            .name = "Cameras",
            .parent_uuid = root_id,
            .child_order = 0,
        }));

        constexpr int camera_count = 1998;
        for (int i = 0; i < camera_count; ++i) {
            const auto camera_id = uuid_literal(std::format(
                "52000000-0000-4000-8000-{:012x}", static_cast<unsigned>(i + 3)));
            CameraRecord camera;
            camera.uid = i;
            camera.camera_id = 1;
            camera.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            camera.translation = {0.0f, 0.0f, static_cast<float>(i)};
            camera.focal_x = 800.0f;
            camera.focal_y = 800.0f;
            camera.center_x = 640.0f;
            camera.center_y = 360.0f;
            camera.camera_width = 1280;
            camera.camera_height = 720;
            camera.image_width = 1280;
            camera.image_height = 720;
            camera.image_name = std::format("cam_{:04}.png", i);
            camera.image_path = camera.image_name;
            camera.split = "train";
            ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
                .uuid = camera_id,
                .type = "camera",
                .name = camera.image_name,
                .parent_uuid = group_id,
                .child_order = static_cast<std::uint32_t>(i),
                .camera = std::move(camera),
            })) << "failed to upsert camera "
                << i;
        }

        auto hierarchy = chapter.validate_hierarchy();
        ASSERT_TRUE(hierarchy) << lfs::format_for_developer(hierarchy.error());
        const auto bytes = chapter.to_bytes();
        auto reparsed = SceneGraphChapter::from_bytes(bytes);
        ASSERT_TRUE(reparsed) << lfs::format_for_developer(reparsed.error());
        auto nodes = reparsed->nodes();
        ASSERT_TRUE(nodes) << lfs::format_for_developer(nodes.error());
        ASSERT_EQ(nodes->size(), 2000u);
        EXPECT_EQ((*nodes)[0].uuid, root_id);
        EXPECT_EQ((*nodes)[0].type, "dataset");
        EXPECT_EQ((*nodes)[0].name, "Dataset");
        EXPECT_EQ((*nodes)[1].uuid, group_id);
        EXPECT_EQ((*nodes)[1].type, "camera_group");
        EXPECT_EQ((*nodes)[1].parent_uuid, root_id);
        EXPECT_EQ((*nodes)[2].type, "camera");
        ASSERT_TRUE((*nodes)[2].camera);
        EXPECT_EQ((*nodes)[2].camera->uid, 0);
        EXPECT_EQ((*nodes)[2].camera->image_name, "cam_0000.png");
        const auto& last = nodes->back();
        EXPECT_EQ(last.type, "camera");
        ASSERT_TRUE(last.camera);
        EXPECT_EQ(last.camera->uid, 1997);
        EXPECT_EQ(last.camera->image_name, "cam_1997.png");
        EXPECT_EQ(last.parent_uuid, group_id);
        EXPECT_EQ(last.child_order, 1997u);

        const auto elapsed = clock::now() - started;
        EXPECT_LT(elapsed, std::chrono::seconds(30))
            << "scale pass took " << std::chrono::duration<double>(elapsed).count() << "s";
    }

    CameraRecord make_chapter_camera(const bool has_image = true) {
        CameraRecord camera;
        camera.uid = 3;
        camera.camera_id = 1;
        camera.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        camera.translation = {0.0f, 0.0f, 1.0f};
        camera.focal_x = 800.0f;
        camera.focal_y = 800.0f;
        camera.center_x = 640.0f;
        camera.center_y = 360.0f;
        camera.camera_width = 1280;
        camera.camera_height = 720;
        camera.image_width = 1280;
        camera.image_height = 720;
        camera.image_name = "frame_0003.png";
        camera.image_path = "images/frame_0003.png";
        camera.split = "train";
        camera.has_image = has_image;
        return camera;
    }

    TEST(ProjectChapterTest,
         SceneGraphCameraVisibleAndTrainingEnabledRoundTripSeparatesHasImage) {
        const auto hidden_id =
            uuid_literal("53000000-0000-4000-8000-000000000001");
        const auto disabled_id =
            uuid_literal("53000000-0000-4000-8000-000000000002");
        const auto missing_id =
            uuid_literal("53000000-0000-4000-8000-000000000003");

        SceneGraphChapter chapter;
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = hidden_id,
            .type = "camera",
            .name = "hidden",
            .child_order = 0,
            .visible = false,
            .training_enabled = true,
            .camera = make_chapter_camera(true),
        }));
        auto disabled_camera = make_chapter_camera(true);
        disabled_camera.uid = 1;
        disabled_camera.image_name = "frame_0001.png";
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = disabled_id,
            .type = "camera",
            .name = "disabled",
            .child_order = 1,
            .visible = true,
            .training_enabled = false,
            .camera = std::move(disabled_camera),
        }));
        auto missing_camera = make_chapter_camera(false);
        missing_camera.uid = 2;
        missing_camera.image_name = "frame_0002.png";
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = missing_id,
            .type = "camera",
            .name = "missing-image",
            .child_order = 2,
            .visible = true,
            .training_enabled = true,
            .camera = std::move(missing_camera),
        }));

        auto reparsed =
            SceneGraphChapter::from_bytes(chapter.to_bytes());
        ASSERT_TRUE(reparsed)
            << lfs::format_for_developer(reparsed.error());
        auto nodes = reparsed->nodes();
        ASSERT_TRUE(nodes)
            << lfs::format_for_developer(nodes.error());
        ASSERT_EQ(nodes->size(), 3u);

        EXPECT_EQ((*nodes)[0].uuid, hidden_id);
        EXPECT_FALSE((*nodes)[0].visible);
        EXPECT_TRUE((*nodes)[0].training_enabled);
        ASSERT_TRUE((*nodes)[0].camera);
        EXPECT_TRUE((*nodes)[0].camera->has_image);

        EXPECT_EQ((*nodes)[1].uuid, disabled_id);
        EXPECT_TRUE((*nodes)[1].visible);
        EXPECT_FALSE((*nodes)[1].training_enabled);
        ASSERT_TRUE((*nodes)[1].camera);
        EXPECT_TRUE((*nodes)[1].camera->has_image);

        EXPECT_EQ((*nodes)[2].uuid, missing_id);
        EXPECT_TRUE((*nodes)[2].visible);
        EXPECT_TRUE((*nodes)[2].training_enabled);
        ASSERT_TRUE((*nodes)[2].camera);
        EXPECT_FALSE((*nodes)[2].camera->has_image);
    }

    TEST(ProjectChapterTest, SceneGraphCameraHasImageDefaultsTrueWhenAbsent) {
        const auto node_id =
            uuid_literal("53000000-0000-4000-8000-000000000010");
        SceneGraphChapter chapter;
        ASSERT_TRUE(chapter.upsert_node(SceneNodeRecord{
            .uuid = node_id,
            .type = "camera",
            .name = "legacy-camera",
            .child_order = 0,
            .visible = true,
            .training_enabled = false,
            .camera = make_chapter_camera(false),
        }));

        auto dumped =
            lfs::io::JsonChapterDom::Json::parse(chapter.dom().dump());
        ASSERT_EQ(dumped["nodes"].size(), 1);
        ASSERT_TRUE(dumped["nodes"][0].contains("camera"));
        dumped["nodes"][0]["camera"].erase("has_image");
        EXPECT_FALSE(dumped["nodes"][0]["camera"].contains("has_image"));

        auto reparsed = SceneGraphChapter::parse(dumped.dump());
        ASSERT_TRUE(reparsed)
            << lfs::format_for_developer(reparsed.error());
        auto found = reparsed->find(node_id);
        ASSERT_TRUE(found)
            << lfs::format_for_developer(found.error());
        ASSERT_TRUE(*found);
        EXPECT_FALSE((*found)->training_enabled);
        ASSERT_TRUE((*found)->camera);
        EXPECT_TRUE((*found)->camera->has_image);
    }

} // namespace
