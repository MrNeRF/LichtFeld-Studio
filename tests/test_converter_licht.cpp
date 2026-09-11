/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "app/include/app/converter.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "io/formats/ply.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "training/checkpoint.hpp"
#include "training/strategies/mcmc.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::core::param;
using namespace lfs::io;
using namespace lfs::io::project;
using lfs::test::licht::fingerprint;
using lfs::test::licht::fixed_uuid_in_namespace;
using lfs::test::licht::make_empty_document;
using lfs::test::licht::make_splat;
using lfs::test::licht::require_status;

namespace {

    constexpr std::uint32_t kUuidNamespace = 0x7c000000;

    Uuid test_uuid(const std::uint64_t tag) {
        return fixed_uuid_in_namespace(kUuidNamespace, tag);
    }

    ProjectDocumentSaveOptions save_options(
        const std::uint64_t identity_tag, const std::uint64_t wallclock) {
        return lfs::test::licht::deterministic_document_save_options(
            kUuidNamespace, identity_tag, wallclock);
    }

    ConvertParameters make_convert_params(const fs::path& input, const fs::path& output) {
        ConvertParameters params;
        params.input_path = input;
        params.output_path = output;
        params.format = OutputFormat::PLY;
        params.sh_degree = -1;
        params.overwrite = true;
        return params;
    }

    class ConverterLichtTest : public ::testing::Test {
    protected:
        const fs::path temp_dir = fs::temp_directory_path() / "lfs_converter_licht_test";

        void SetUp() override {
            fs::remove_all(temp_dir);
            fs::create_directories(temp_dir);
        }

        void TearDown() override {
            fs::remove_all(temp_dir);
        }

        void add_root_group(ProjectDocument& document, const Uuid& root_uuid) {
            require_status(document.edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = root_uuid,
                .type = "group",
                .name = "Root",
                .child_order = 0,
            }));
        }

        void add_embedded_splat(
            ProjectDocument& document,
            const Uuid& node_uuid,
            const Uuid& parent_uuid,
            const std::uint32_t child_order,
            const SplatData& model,
            const std::string& name,
            const std::uint8_t fingerprint_tag) {
            require_status(document.edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = node_uuid,
                .type = "splat",
                .name = name,
                .parent_uuid = parent_uuid,
                .child_order = child_order,
                .payload =
                    PayloadBinding{
                        .fourcc = "SPLT",
                        .instance_uuid = node_uuid,
                        .source_kind = "ply",
                    },
            }));
            auto captured = SplatChapterPayload::capture(
                model, SplatSourceKind::ImportedPly, false);
            ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
            require_status(document.set_splat(node_uuid, std::move(*captured)));

            auto& project = document.edit_project();
            require_status(project.upsert_embed_decision(EmbedDecision{
                .uuid = node_uuid,
                .node_uuid = node_uuid,
                .payload_fourcc = "SPLT",
                .decision = "embedded",
                .reason = "convert licht test",
            }));
            require_status(project.upsert_embedded_payload_provenance(
                EmbeddedPayloadProvenance{
                    .uuid = node_uuid,
                    .node_uuid = node_uuid,
                    .fourcc = "SPLT",
                    .import_locator =
                        {
                            .preferred = "assets/splat.ply",
                            .base = LocatorBase::Project,
                        },
                    .import_fingerprint = fingerprint(fingerprint_tag),
                    .content_xxh3_128 = {},
                }));
        }

        void write_empty_project(const fs::path& path) {
            auto document = make_empty_document(test_uuid(1), 100);
            const auto saved = document->save(path, save_options(10, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        void write_splt_project(
            const fs::path& path, const SplatData& model, const std::uint64_t identity_tag) {
            auto document = make_empty_document(test_uuid(identity_tag), 100);
            const auto root_uuid = test_uuid(identity_tag + 1);
            const auto splat_uuid = test_uuid(identity_tag + 2);
            add_root_group(*document, root_uuid);
            add_embedded_splat(
                *document, splat_uuid, root_uuid, 0, model, "Splat",
                static_cast<std::uint8_t>(identity_tag & 0xff));
            const auto saved = document->save(path, save_options(identity_tag + 10, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        void write_two_splt_project(const fs::path& path) {
            auto document = make_empty_document(test_uuid(80), 100);
            const auto root_uuid = test_uuid(81);
            add_root_group(*document, root_uuid);
            add_embedded_splat(*document, test_uuid(82), root_uuid, 0, *make_splat(2), "A", 2);
            add_embedded_splat(*document, test_uuid(83), root_uuid, 1, *make_splat(3), "B", 3);
            const auto saved = document->save(path, save_options(90, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        void write_ckpt_project(const fs::path& path, const int gaussian_count) {
            auto model = make_splat(static_cast<std::size_t>(gaussian_count));
            lfs::training::MCMC strategy(*model);
            TrainingParameters parameters;
            parameters.optimization = OptimizationParameters::mcmc_defaults();
            parameters.optimization.sh_degree = 0;
            parameters.optimization.max_cap = gaussian_count;

            std::ostringstream stream(std::ios::binary | std::ios::out);
            const auto serialized = lfs::training::serialize_checkpoint(
                stream, 17, strategy, parameters, nullptr, nullptr, nullptr, nullptr);
            ASSERT_TRUE(serialized) << lfs::format_for_developer(serialized.error());
            const auto encoded = stream.str();
            ASSERT_FALSE(encoded.empty());
            std::vector<std::byte> bytes(encoded.size());
            std::memcpy(bytes.data(), encoded.data(), encoded.size());

            const auto project_uuid = test_uuid(200);
            const auto root_uuid = test_uuid(201);
            const auto training_uuid = test_uuid(202);
            const auto checkpoint_uuid = test_uuid(203);
            auto document = make_empty_document(project_uuid, 100);
            add_root_group(*document, root_uuid);
            require_status(document->edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = training_uuid,
                .type = "splat",
                .name = "Training",
                .parent_uuid = root_uuid,
                .child_order = 0,
                .payload =
                    PayloadBinding{
                        .fourcc = "CKPT",
                        .instance_uuid = checkpoint_uuid,
                        .source_kind = "training",
                    },
            }));
            require_status(document->edit_scene_graph().set_training_model_uuid(training_uuid));
            auto payload = LazyChunkValue::from_owned(std::move(bytes), checkpoint_uuid);
            ASSERT_TRUE(payload) << lfs::format_for_developer(payload.error());
            require_status(document->set_checkpoint(checkpoint_uuid, std::move(*payload)));

            auto options = save_options(210, 200);
            options.commit.snapshot_uuid = checkpoint_uuid;
            const auto saved = document->save(path, options);
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }
    };

    TEST_F(ConverterLichtTest, ConvertsSingleSpltProjectToPly) {
        const auto project = temp_dir / "single-splt.licht";
        const auto output = temp_dir / "single-splt.ply";
        constexpr std::size_t kCount = 4;
        write_splt_project(project, *make_splat(kCount), 20);

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        (void)testing::internal::GetCapturedStdout();
        ASSERT_EQ(rc, 0);
        ASSERT_TRUE(fs::exists(output));

        auto loaded = load_ply(output);
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(loaded->value.size(), kCount);
        EXPECT_EQ(loaded->value.get_max_sh_degree(), 0);
    }

    TEST_F(ConverterLichtTest, ConvertsSingleCkptProjectToPly) {
        const auto project = temp_dir / "single-ckpt.licht";
        const auto output = temp_dir / "single-ckpt.ply";
        constexpr int kCount = 3;
        write_ckpt_project(project, kCount);

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        (void)testing::internal::GetCapturedStdout();
        ASSERT_EQ(rc, 0);
        ASSERT_TRUE(fs::exists(output));

        auto loaded = load_ply(output);
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(loaded->value.size(), static_cast<std::size_t>(kCount));
        EXPECT_EQ(loaded->value.get_max_sh_degree(), 0);
    }

    TEST_F(ConverterLichtTest, RejectsProjectWithNoSplatModel) {
        const auto project = temp_dir / "empty.licht";
        const auto output = temp_dir / "empty.ply";
        write_empty_project(project);

        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        const auto stdout_text = testing::internal::GetCapturedStdout();
        const auto stderr_text = testing::internal::GetCapturedStderr();

        EXPECT_NE(rc, 0);
        EXPECT_NE(stderr_text.find("project contains no splat model"), std::string::npos);
        EXPECT_EQ(stdout_text.find("Loaded"), std::string::npos);
        EXPECT_FALSE(fs::exists(output));
    }

    TEST_F(ConverterLichtTest, RejectsProjectWithMultipleSplatModels) {
        const auto project = temp_dir / "two-models.licht";
        const auto output = temp_dir / "two-models.ply";
        write_two_splt_project(project);

        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        const auto stdout_text = testing::internal::GetCapturedStdout();
        const auto stderr_text = testing::internal::GetCapturedStderr();

        EXPECT_NE(rc, 0);
        EXPECT_NE(stderr_text.find("0 checkpoint model(s)"), std::string::npos);
        EXPECT_NE(stderr_text.find("2 splat model(s)"), std::string::npos);
        EXPECT_NE(stderr_text.find("LichtFeld Studio GUI"), std::string::npos);
        EXPECT_EQ(stdout_text.find("Loaded"), std::string::npos);
        EXPECT_FALSE(fs::exists(output));
    }

    TEST_F(ConverterLichtTest, DirectoryScanFindsLichtProjects) {
        const auto input_dir = temp_dir / "input";
        const auto output_dir = temp_dir / "output";
        fs::create_directories(input_dir);
        fs::create_directories(output_dir);
        constexpr std::size_t kCount = 2;
        write_splt_project(input_dir / "scene.licht", *make_splat(kCount), 40);

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(make_convert_params(input_dir, output_dir));
        (void)testing::internal::GetCapturedStdout();
        ASSERT_EQ(rc, 0);

        const auto output = output_dir / "scene_converted.ply";
        ASSERT_TRUE(fs::exists(output));
        auto loaded = load_ply(output);
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(loaded->value.size(), kCount);
    }

} // namespace
