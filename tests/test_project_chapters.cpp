/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_chapters.hpp"

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

    lfs::core::Uuid uuid(const std::string_view text) {
        const auto result = lfs::core::Uuid::from_string(text);
        EXPECT_TRUE(result);
        return result.value_or(lfs::core::Uuid{});
    }

    class TemporaryDirectory {
    public:
        TemporaryDirectory()
            : path(fs::temp_directory_path() /
                   ("lfs-p3-chapters-" + lfs::core::generate_uuid_v4().to_string())) {
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        fs::path path;
    };

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
            uuid("10000000-0000-4000-8000-000000000090");
        result.mrnf_current_references.ppisp_reference =
            uuid("10000000-0000-4000-8000-000000000091");
        result.dataset.centralize_dataset = "cameras";
        result.dataset.timelapse_images = {"frame-a.png", "frame-b.png"};
        return result;
    }

    ReferenceFingerprint fake_fingerprint(const std::uint8_t tag) {
        ReferenceFingerprint result;
        result.size = 100 + tag;
        result.mtime_unix_ns = 200 + tag;
        result.head_xxh3.bytes.fill(tag);
        result.tail_xxh3.bytes.fill(static_cast<std::uint8_t>(tag + 1));
        return result;
    }

    TEST(ProjectChapterTest, ProjectTypedMutationRetainsUnknownSubtrees) {
        const auto project_id = uuid("10000000-0000-4000-8000-000000000001");
        const auto decision_id = uuid("10000000-0000-4000-8000-000000000002");
        const auto node_id = uuid("10000000-0000-4000-8000-000000000003");
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
        const auto ref_id = uuid("20000000-0000-4000-8000-000000000001");
        ReferencesChapter chapter;
        ReferenceRecord record{
            .uuid = ref_id,
            .key = "dataset.root",
            .kind = "dataset",
            .locator =
                {.preferred = "../missing", .base = LocatorBase::Project, .absolute_fallback = "/old/missing"},
            .fingerprint = fake_fingerprint(3),
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
        const auto project_id = uuid("20000000-0000-4000-8000-000000000002");
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
        const auto ref_id = uuid("30000000-0000-4000-8000-000000000001");
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
        const auto ref_id = uuid("40000000-0000-4000-8000-000000000001");
        const auto root_id = uuid("40000000-0000-4000-8000-000000000002");
        const auto splat_id = uuid("40000000-0000-4000-8000-000000000003");
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
            .fingerprint = fake_fingerprint(5),
            .unresolved = true,
        }));
        ProjectChapter project;
        ASSERT_TRUE(project.set_project_uuid(
            uuid("40000000-0000-4000-8000-000000000004")));
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

} // namespace
