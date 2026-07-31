/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "checkpoint_fixture.hpp"
#include "ppisp_fixture.hpp"

#include "core/checkpoint_format.hpp"
#include "core/tensor.hpp"
#include "gui/layout_state.hpp"
#include "io/project_document.hpp"
#include "training/components/ppisp_file.hpp"
#include "training/strategies/mcmc.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

    namespace fs = std::filesystem;
    using Json = nlohmann::ordered_json;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    using namespace lfs::io::project;

    constexpr std::string_view AUTHORITY_SHA =
        "8ca8028e6214b1f424c373b24d479cd90ff2e918";

    Uuid fixed_uuid(const std::uint64_t tag) {
        const auto parsed = Uuid::from_string(std::format(
            "{:08x}-0000-4000-8000-{:012x}", tag, tag));
        if (!parsed) {
            std::abort();
        }
        return *parsed;
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
        ASSERT_TRUE(result)
            << (result ? std::string{}
                       : lfs::format_for_developer(result.error()));
    }

    std::vector<std::byte> read_bytes(const fs::path& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot read " + path.string());
        }
        const std::vector<char> raw{
            std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
        std::vector<std::byte> result(raw.size());
        if (!raw.empty()) {
            std::memcpy(result.data(), raw.data(), raw.size());
        }
        return result;
    }

    void write_text(const fs::path& path, const std::string_view text) {
        fs::create_directories(path.parent_path());
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot write " + path.string());
        }
        stream.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!stream) {
            throw std::runtime_error("cannot finish " + path.string());
        }
    }

    fs::path fixture_root() {
        return fs::path(PROJECT_ROOT_PATH) /
               "tools/licht_inspect/fixtures/migration";
    }

    Json oracle() {
        std::ifstream stream(fixture_root() / "oracle.json");
        if (!stream) {
            throw std::runtime_error("migration oracle is missing");
        }
        return Json::parse(stream);
    }

    std::unique_ptr<lfs::core::SplatData> make_splat(
        const std::size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index);
            rotations[index * 4] = 1.0f;
        }
        return std::make_unique<lfs::core::SplatData>(
            0,
            Tensor::from_vector(means, {count, std::size_t{3}}, Device::CPU),
            Tensor::zeros({count, std::size_t{1}, std::size_t{3}},
                          Device::CPU, DataType::Float32),
            Tensor{},
            Tensor::zeros({count, std::size_t{3}}, Device::CPU,
                          DataType::Float32),
            Tensor::from_vector(rotations, {count, std::size_t{4}},
                                Device::CPU),
            Tensor::zeros({count, std::size_t{1}}, Device::CPU,
                          DataType::Float32),
            1.0f);
    }

    lfs::core::param::TrainingParameters migration_parameters() {
        lfs::core::param::TrainingParameters parameters;
        parameters.optimization =
            lfs::core::param::OptimizationParameters::mcmc_defaults();
        parameters.optimization.strategy = "mcmc";
        parameters.optimization.iterations = 90;
        parameters.optimization.sh_degree = 0;
        parameters.optimization.max_cap = 2;
        parameters.optimization.use_ppisp = true;
        parameters.dataset.data_path = "/p8/frozen/dataset";
        parameters.dataset.images = "images";
        return parameters;
    }

    ParameterManagerSnapshot parameter_snapshot(
        const lfs::core::param::TrainingParameters& parameters) {
        ParameterManagerSnapshot snapshot;
        snapshot.active_strategy = "mcmc";
        snapshot.mcmc_session = parameters.optimization;
        snapshot.mcmc_current = parameters.optimization;
        snapshot.mrnf_session =
            lfs::core::param::OptimizationParameters::mrnf_defaults();
        snapshot.mrnf_current = snapshot.mrnf_session;
        snapshot.igs_session =
            lfs::core::param::OptimizationParameters::igs_plus_defaults();
        snapshot.igs_current = snapshot.igs_session;
        snapshot.dataset = parameters.dataset;
        return snapshot;
    }

    ProjectDocumentSaveOptions save_options(const std::uint64_t tag) {
        return ProjectDocumentSaveOptions{
            .commit =
                {
                    .kind = CommitKind::Explicit,
                    .commit_uuid = fixed_uuid(tag),
                    .snapshot_uuid = fixed_uuid(tag + 1),
                    .wallclock_unix_ns = 1'735'689'601'000'000'000 + tag,
                },
            .file_uuid = fixed_uuid(tag + 2),
            .index_compression = IndexCompression::Zstd,
            .disk_reserve_bytes = 0,
        };
    }

    class TemporaryDirectory {
    public:
        TemporaryDirectory()
            : path(fs::temp_directory_path() /
                   ("lfs-p8-migration-" +
                    lfs::core::generate_uuid_v4().to_string())) {
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        fs::path path;
    };

#ifndef _WIN32
    class ScopedXdgConfigHome {
    public:
        explicit ScopedXdgConfigHome(const fs::path& path) {
            if (const char* previous = std::getenv("XDG_CONFIG_HOME")) {
                previous_ = previous;
            }
            setenv("XDG_CONFIG_HOME", path.string().c_str(), 1);
        }

        ~ScopedXdgConfigHome() {
            if (previous_) {
                setenv("XDG_CONFIG_HOME", previous_->c_str(), 1);
            } else {
                unsetenv("XDG_CONFIG_HOME");
            }
        }

    private:
        std::optional<std::string> previous_;
    };
#endif

    Json fixed_arrangement(const GuiLayoutChapter& chapter) {
        const Json root = Json::parse(chapter.dom().dump());
        const auto& areas = root.at("layouts").at(0).at("areas");
        if (areas.size() != 1) {
            throw std::runtime_error("GUIL oracle does not have one area");
        }
        for (const auto& space : areas.at(0).at("spaces")) {
            if (space.value("type", std::string{}) == "fixed_arrangement") {
                return space.at("opaque_payload");
            }
        }
        throw std::runtime_error("GUIL oracle has no fixed arrangement");
    }

    void emit_frozen_inputs(const fs::path& output) {
        fs::create_directories(output);
        auto model = make_splat(2);
        lfs::training::MCMC strategy(*model);
        auto parameters = migration_parameters();
        strategy.initialize(parameters.optimization);
        lfs::training::PPISP checkpoint_ppisp(
            parameters.optimization.iterations);
        checkpoint_ppisp.register_frame(101, 10);
        checkpoint_ppisp.register_frame(102, 20);
        checkpoint_ppisp.register_frame(103, 20);
        checkpoint_ppisp.finalize();
        const auto checkpoint = lfs::test::write_checkpoint_fixture(
            output, 37, strategy, parameters, nullptr, &checkpoint_ppisp,
            nullptr, nullptr);
        if (!checkpoint) {
            throw std::runtime_error(checkpoint.error());
        }

        const lfs::training::PPISPFileMetadata metadata{
            .dataset_path_utf8 = "/p8/frozen/dataset",
            .images_folder = "images",
            .frame_image_names = {"a.png", "b.png", "c.png"},
            .frame_camera_ids = {10, 20, 20},
            .camera_ids = {10, 20},
        };
        const auto ppisp = lfs::test::write_ppisp_fixture(
            output / "appearance.ppisp", checkpoint_ppisp, nullptr,
            &metadata);
        if (!ppisp) {
            throw std::runtime_error(ppisp.error());
        }

        write_text(
            output / "layout_config/LichtFeldStudio/layout.json",
            R"({"right_panel_width":451.0,"scene_panel_ratio":0.61,"python_console_width":377.0,"bottom_dock_height":222.0,"left_dock_width":417.0,"show_sequencer":true})");
        write_text(output / "live.rad",
                   "LichtFeld P8 external RAD reference witness\n");
        write_text(
            output / "half_migrated/LichtFeldStudio/layout.json",
            R"({"left_dock_width":999.0,"bottom_dock_height":888.0,"scene_panel_ratio":0.77,"show_sequencer":true})");
        fs::create_directories(output / "half_migrated");
        fs::copy_file(
            fs::path(PROJECT_ROOT_PATH) /
                "tools/licht_inspect/fixtures/release_corpus/save.licht",
            output / "half_migrated/project.licht",
            fs::copy_options::overwrite_existing);
    }

    TEST(P8MigrationFixtureTest, FrozenInputManifestIsAuthorityLocked) {
        if (const char* requested =
                std::getenv("LFS_P8_MIGRATION_CORPUS_OUTPUT");
            requested != nullptr && std::string_view(requested).size() != 0) {
            emit_frozen_inputs(requested);
            return;
        }
        const Json expected = oracle();
        EXPECT_EQ(expected.at("authority_sha").get<std::string>(),
                  AUTHORITY_SHA);
        ASSERT_EQ(expected.at("inputs").size(), 6u);
        for (const auto& input : expected.at("inputs")) {
            const auto path = fixture_root() /
                              input.at("path").get<std::string>();
            ASSERT_TRUE(fs::is_regular_file(path)) << path;
            EXPECT_EQ(fs::file_size(path), input.at("bytes").get<std::uint64_t>())
                << path;
        }
    }

    TEST(P8MigrationFixtureTest,
         LegacyCheckpointMapsToCkptAndPrmsChapterOracle) {
        const auto expected = oracle().at("checkpoint");
        const auto path = fixture_root() / "checkpoint.resume";
        const auto header = lfs::core::load_checkpoint_header(path);
        ASSERT_TRUE(header) << header.error();
        EXPECT_EQ(header->iteration, expected.at("iteration").get<int>());
        EXPECT_EQ(header->num_gaussians,
                  expected.at("num_gaussians").get<std::uint32_t>());
        EXPECT_TRUE(lfs::core::has_flag(
            header->flags, lfs::core::CheckpointFlags::HAS_PPISP));
        const auto parameters = lfs::core::load_checkpoint_params(path);
        ASSERT_TRUE(parameters) << parameters.error();
        EXPECT_EQ(parameters->optimization.strategy,
                  expected.at("strategy").get<std::string>());
        EXPECT_EQ(parameters->optimization.iterations,
                  expected.at("max_iter").get<int>());

        const auto project_uuid = Uuid::from_string(
            expected.at("project_uuid").get<std::string>());
        const auto checkpoint_uuid = Uuid::from_string(
            expected.at("checkpoint_uuid").get<std::string>());
        ASSERT_TRUE(project_uuid && checkpoint_uuid);
        auto document = require_result(ProjectDocument::create(
            *project_uuid, 1'735'689'600'000'000'000));
        require_status(document.edit_parameters().set_snapshot(
            parameter_snapshot(*parameters)));
        require_status(document.edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = *checkpoint_uuid,
                .type = "splat",
                .name = "Migrated checkpoint",
                .child_order = 0,
                .payload = PayloadBinding{
                    .fourcc = "CKPT",
                    .instance_uuid = *checkpoint_uuid,
                    .source_kind = "training",
                },
            }));
        require_status(document.edit_scene_graph().set_training_model_uuid(
            *checkpoint_uuid));
        const auto source_bytes = read_bytes(path);
        auto lazy = require_result(LazyChunkValue::from_owned(
            source_bytes, *checkpoint_uuid));
        require_status(document.set_checkpoint(
            *checkpoint_uuid, std::move(lazy)));

        TemporaryDirectory temporary;
        const auto converted = temporary.path / "checkpoint.licht";
        auto options = save_options(0xC301);
        options.commit.snapshot_uuid = *checkpoint_uuid;
        require_result(document.save(converted, options));
        const auto reader = require_result(ProjectReader::open(converted));
        EXPECT_NE(reader.find(FOURCC_CKPT, *checkpoint_uuid), nullptr);
        EXPECT_NE(reader.find(FOURCC_PRMS, *project_uuid), nullptr);
        EXPECT_EQ(reader.find(FOURCC_PPIS, *checkpoint_uuid), nullptr);
        const auto reopened = require_result(ProjectDocument::open(converted));
        ASSERT_EQ(reopened.checkpoint_uuids(),
                  std::vector<Uuid>{*checkpoint_uuid});
        ASSERT_TRUE(reopened.ppisp_uuids().empty());
        const auto* embedded = reopened.find_checkpoint(*checkpoint_uuid);
        ASSERT_NE(embedded, nullptr);
        std::vector<std::byte> roundtrip(source_bytes.size());
        require_status(embedded->read_at(0, roundtrip));
        EXPECT_EQ(roundtrip, source_bytes);
        const auto pending = require_result(reopened.parameters().snapshot());
        EXPECT_EQ(pending.active_strategy,
                  expected.at("strategy").get<std::string>());
        EXPECT_EQ(pending.mcmc_current.iterations,
                  expected.at("max_iter").get<int>());
    }

    TEST(P8MigrationFixtureTest,
         LegacyPpispMapsToPrmsAndMatrixQualifiedPayloadOracle) {
        const auto expected = oracle().at("ppisp");
        const auto path = fixture_root() / "appearance.ppisp";
        lfs::training::PPISP loaded(90);
        lfs::training::PPISPFileMetadata metadata;
        const auto parsed = lfs::training::load_ppisp_file(
            path, loaded, nullptr, &metadata);
        ASSERT_TRUE(parsed) << parsed.error();
        EXPECT_EQ(loaded.num_cameras(),
                  expected.at("num_cameras").get<std::size_t>());
        EXPECT_EQ(loaded.num_frames(),
                  expected.at("num_frames").get<std::size_t>());
        EXPECT_EQ(metadata.images_folder,
                  expected.at("images_folder").get<std::string>());

        const auto project_uuid = Uuid::from_string(
            expected.at("project_uuid").get<std::string>());
        const auto ppisp_uuid = Uuid::from_string(
            expected.at("ppisp_uuid").get<std::string>());
        ASSERT_TRUE(project_uuid && ppisp_uuid);
        auto document = require_result(ProjectDocument::create(
            *project_uuid, 1'735'689'600'000'000'100));
        const auto parameters = migration_parameters();
        require_status(document.edit_parameters().set_snapshot(
            parameter_snapshot(parameters)));
        const auto source_bytes = read_bytes(path);
        auto lazy = require_result(LazyChunkValue::from_owned(
            source_bytes, *ppisp_uuid));
        require_status(document.set_ppisp(*ppisp_uuid, std::move(lazy)));

        TemporaryDirectory temporary;
        const auto converted = temporary.path / "ppisp.licht";
        require_result(document.save(converted, save_options(0xC311)));
        const auto reader = require_result(ProjectReader::open(converted));
        EXPECT_NE(reader.find(FOURCC_PPIS, *ppisp_uuid), nullptr);
        EXPECT_NE(reader.find(FOURCC_PRMS, *project_uuid), nullptr);
        EXPECT_EQ(reader.find(FOURCC_CKPT, *ppisp_uuid), nullptr);
        const auto reopened = require_result(ProjectDocument::open(converted));
        ASSERT_TRUE(reopened.checkpoint_uuids().empty());
        ASSERT_EQ(reopened.ppisp_uuids(), std::vector<Uuid>{*ppisp_uuid});
        const auto* embedded = reopened.find_ppisp(*ppisp_uuid);
        ASSERT_NE(embedded, nullptr);
        std::vector<std::byte> roundtrip(source_bytes.size());
        require_status(embedded->read_at(0, roundtrip));
        EXPECT_EQ(roundtrip, source_bytes);
    }

#ifndef _WIN32
    TEST(P8MigrationFixtureTest, LegacyLayoutMapsToGuilAreaTreeOracle) {
        const auto expected = oracle().at("layout");
        const ScopedXdgConfigHome config(fixture_root() / "layout_config");
        lfs::vis::gui::LayoutState legacy;
        legacy.load();
        EXPECT_FLOAT_EQ(legacy.left_dock_width,
                        expected.at("left_dock_width").get<float>());
        EXPECT_FLOAT_EQ(legacy.bottom_dock_height,
                        expected.at("bottom_dock_height").get<float>());
        EXPECT_FLOAT_EQ(legacy.scene_panel_ratio,
                        expected.at("scene_panel_ratio").get<float>());
        EXPECT_EQ(legacy.show_sequencer,
                  expected.at("sequencer_visible").get<bool>());

        GuiLayoutChapter chapter;
        const Json known{
            {"layouts",
             Json::array({Json{
                 {"areas",
                  Json::array({Json{
                      {"spaces",
                       Json::array({Json{
                           {"type", "fixed_arrangement"},
                           {"opaque_payload",
                            Json{
                                {"right_panel_width",
                                 legacy.right_panel_width},
                                {"scene_panel_ratio",
                                 legacy.scene_panel_ratio},
                                {"python_console_width",
                                 legacy.python_console_width},
                                {"bottom_dock_height",
                                 legacy.bottom_dock_height},
                                {"left_dock_width",
                                 legacy.left_dock_width},
                                {"sequencer_visible",
                                 legacy.show_sequencer},
                            }},
                       }})},
                  }})},
             }})},
        };
        require_status(chapter.merge_known_state(known));
        require_status(chapter.validate());
        const auto fixed = fixed_arrangement(chapter);
        EXPECT_FLOAT_EQ(fixed.at("left_dock_width").get<float>(),
                        expected.at("left_dock_width").get<float>());
        EXPECT_FLOAT_EQ(fixed.at("bottom_dock_height").get<float>(),
                        expected.at("bottom_dock_height").get<float>());
        EXPECT_FLOAT_EQ(fixed.at("scene_panel_ratio").get<float>(),
                        expected.at("scene_panel_ratio").get<float>());
        EXPECT_EQ(fixed.at("sequencer_visible").get<bool>(),
                  expected.at("sequencer_visible").get<bool>());
    }
#endif

    TEST(P8MigrationFixtureTest, LiveRadReferenceAndRelinkMatchOracle) {
        const auto expected = oracle().at("rad");
        const auto source = fixture_root() / "live.rad";
        const auto fingerprint = require_result(fingerprint_path(source));
        const auto project_uuid = Uuid::from_string(
            expected.at("project_uuid").get<std::string>());
        const auto node_uuid = Uuid::from_string(
            expected.at("node_uuid").get<std::string>());
        const auto reference_uuid = Uuid::from_string(
            expected.at("reference_uuid").get<std::string>());
        ASSERT_TRUE(project_uuid && node_uuid && reference_uuid);
        auto document = require_result(ProjectDocument::create(
            *project_uuid, 1'735'689'600'000'000'200));
        require_status(document.edit_references().upsert(ReferenceRecord{
            .uuid = *reference_uuid,
            .key = expected.at("key").get<std::string>(),
            .kind = "rad",
            .locator =
                {
                    .preferred = "live.rad",
                    .base = LocatorBase::Project,
                    .absolute_fallback = source.string(),
                },
            .fingerprint = fingerprint,
            .unresolved = false,
        }));
        require_status(document.edit_project().upsert_embed_decision(
            EmbedDecision{
                .uuid = *node_uuid,
                .node_uuid = *node_uuid,
                .payload_fourcc = "REFS",
                .decision = "external",
                .reference_uuid = *reference_uuid,
                .reason = "legacy live RAD remains external",
            }));
        require_status(document.edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = *node_uuid,
                .type = "splat",
                .name = "Legacy live RAD",
                .child_order = 0,
                .payload = PayloadBinding{
                    .fourcc = "REFS",
                    .instance_uuid = *reference_uuid,
                    .reference_uuid = *reference_uuid,
                    .source_kind = "rad",
                },
            }));

        TemporaryDirectory temporary;
        const auto converted = temporary.path / "rad.licht";
        require_result(document.save(converted, save_options(0xC321)));
        auto reopened = require_result(ProjectDocument::open(converted));
        auto records = require_result(reopened.references().records());
        ASSERT_EQ(records.size(), 1u);
        EXPECT_EQ(records.front().uuid, *reference_uuid);
        EXPECT_EQ(records.front().key,
                  expected.at("key").get<std::string>());
        EXPECT_EQ(records.front().kind, "rad");
        EXPECT_EQ(records.front().fingerprint, fingerprint);

        const auto relinked = temporary.path / "moved.rad";
        fs::copy_file(source, relinked);
        fs::last_write_time(
            relinked,
            fs::last_write_time(relinked) + std::chrono::seconds(5));
        require_status(reopened.edit_references().relink(
            *reference_uuid,
            ReferenceLocator{
                .preferred = "moved.rad",
                .base = LocatorBase::Project,
            },
            relinked, false));
        require_result(reopened.save(converted, save_options(0xC331)));
        const auto after = require_result(ProjectDocument::open(converted));
        const auto updated = require_result(after.references().find(
            *reference_uuid));
        ASSERT_TRUE(updated);
        EXPECT_EQ(updated->locator.preferred, "moved.rad");
        EXPECT_EQ(updated->fingerprint.head_xxh3,
                  fingerprint.head_xxh3);
        EXPECT_EQ(updated->fingerprint.tail_xxh3,
                  fingerprint.tail_xxh3);
        EXPECT_FALSE(updated->unresolved);
    }

#ifndef _WIN32
    TEST(P8MigrationFixtureTest,
         HalfMigratedDirectoryChoosesProjectGuilOverLegacyLayout) {
        const auto expected = oracle().at("half_migrated");
        const ScopedXdgConfigHome config(
            fixture_root() / "half_migrated");
        lfs::vis::gui::LayoutState legacy;
        legacy.load();
        EXPECT_FLOAT_EQ(legacy.left_dock_width,
                        expected.at("legacy_left_dock_width").get<float>());
        const auto project = require_result(ProjectDocument::open(
            fixture_root() / "half_migrated/project.licht"));
        const auto fixed = fixed_arrangement(project.gui_layout());
        EXPECT_EQ(expected.at("winner").get<std::string>(),
                  "project.licht:GUIL");
        EXPECT_FLOAT_EQ(fixed.at("left_dock_width").get<float>(),
                        expected.at("project_left_dock_width").get<float>());
        EXPECT_NE(fixed.at("left_dock_width").get<float>(),
                  legacy.left_dock_width);
    }
#endif

} // namespace
