/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/licht_command.hpp"
#include "core/argument_parser.hpp"
#include "io/embedded_dataset.hpp"
#include "io/formats/transforms.hpp"
#include "io/loader.hpp"
#include "io/project_chapters.hpp"
#include "io/project_document.hpp"
#include "io/project_operations.hpp"
#include "licht_test_support.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <zlib.h>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    std::vector<std::byte> valid_png(const std::byte red = std::byte{0xff}) {
        const auto append_u32 = [](std::vector<std::byte>& output, const std::uint32_t value) {
            for (int shift = 24; shift >= 0; shift -= 8)
                output.push_back(static_cast<std::byte>((value >> shift) & 0xff));
        };
        std::vector<std::byte> result{
            std::byte{0x89}, std::byte{'P'}, std::byte{'N'}, std::byte{'G'},
            std::byte{0x0d}, std::byte{0x0a}, std::byte{0x1a}, std::byte{0x0a}};
        const auto append_chunk = [&](const std::array<char, 4> type,
                                      const std::span<const std::byte> payload) {
            append_u32(result, static_cast<std::uint32_t>(payload.size()));
            const auto type_begin = result.size();
            for (const char value : type)
                result.push_back(static_cast<std::byte>(value));
            result.insert(result.end(), payload.begin(), payload.end());
            const auto crc = crc32(0, reinterpret_cast<const Bytef*>(result.data() + type_begin),
                                   static_cast<uInt>(4 + payload.size()));
            append_u32(result, static_cast<std::uint32_t>(crc));
        };
        const std::array<std::byte, 13> header{
            std::byte{0}, std::byte{0}, std::byte{0}, std::byte{1},
            std::byte{0}, std::byte{0}, std::byte{0}, std::byte{1},
            std::byte{8}, std::byte{2}, std::byte{0}, std::byte{0}, std::byte{0}};
        append_chunk({'I', 'H', 'D', 'R'}, header);
        const std::array<std::byte, 4> raw{
            std::byte{0}, red, std::byte{0}, std::byte{0}};
        uLongf compressed_size = compressBound(raw.size());
        std::vector<std::byte> compressed(compressed_size);
        EXPECT_EQ(compress2(reinterpret_cast<Bytef*>(compressed.data()), &compressed_size,
                            reinterpret_cast<const Bytef*>(raw.data()), raw.size(), Z_BEST_SPEED),
                  Z_OK);
        compressed.resize(compressed_size);
        append_chunk({'I', 'D', 'A', 'T'}, compressed);
        append_chunk({'I', 'E', 'N', 'D'}, std::span<const std::byte>{});
        return result;
    }

    void write_text(const fs::path& path, const std::string_view value) {
        fs::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary);
        output << value;
        ASSERT_TRUE(output.good());
    }

    fs::path make_colmap_dataset(const fs::path& root,
                                 const std::string_view layout) {
        const auto sparse = layout == "flat" ? root : root / "sparse" / (layout == "sparse" ? "" : "0");
        fs::create_directories(sparse);
        fs::create_directories(root / "images");
        write_text(sparse / "cameras.txt", "1 PINHOLE 1 1 1 1 0.5 0.5\n");
        write_text(sparse / "images.txt", "1 1 0 0 0 0 0 0 1 frame.png\n\n");
        write_text(sparse / "points3D.txt", "# 3D point list\n");
        write_file_bytes(root / "images/frame.png", valid_png());
        return root;
    }

    fs::path make_transforms_dataset(const fs::path& root) {
        for (const auto* folder : {"train", "test", "masks", "depths", "normals", "points"})
            fs::create_directories(root / folder);
        const auto image = valid_png();
        write_file_bytes(root / "train/frame_a.png", image);
        write_file_bytes(root / "test/frame_b.png", image);
        write_file_bytes(root / "masks/frame_a.png", image);
        write_file_bytes(root / "depths/frame_a.png", image);
        write_file_bytes(root / "normals/frame_a.png", image);
        write_text(root / "points/cloud.ply",
                   "ply\nformat ascii 1.0\nelement vertex 1\n"
                   "property float x\nproperty float y\nproperty float z\n"
                   "end_header\n0 0 0\n");
        const nlohmann::json identity = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0},
        };
        const nlohmann::json transforms = {
            {"w", 1},
            {"h", 1},
            {"fl_x", 1.0},
            {"fl_y", 1.0},
            {"cx", 0.5},
            {"cy", 0.5},
            {"ply_file_path", "points/cloud.ply"},
            {"frames", nlohmann::json::array({
                           {{"file_path", "train/frame_a"}, {"mask_path", "masks/frame_a.png"}, {"transform_matrix", identity}},
                           {{"file_path", "test/frame_b.png"},
                            {"transform_matrix", identity}},
                       })},
        };
        write_text(root / "transforms_train.json", transforms.dump());
        return root;
    }

    fs::path make_project(const fs::path& path, const fs::path& dataset) {
        auto document = require_result_ptr(ProjectDocument::create(
            fixed_uuid(static_cast<std::uint64_t>(path.native().size()) +
                       static_cast<std::uint64_t>(dataset.native().size()) + 5000),
            100));
        const auto reference = require_result(upsert_path_reference(
            document->edit_references(), path.parent_path(), dataset,
            "dataset", "dataset"));
        require_status(document->edit_project().set_dataset_reference(reference));
        ProjectDocumentSaveOptions options;
        options.disk_reserve_bytes = 0;
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        static_cast<void>(require_result(document->save(path, options)));
        return path;
    }

    std::vector<std::byte> bytes(const fs::path& path) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        const auto size = input.tellg();
        std::vector<std::byte> result(static_cast<std::size_t>(size));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(result.data()), size);
        return result;
    }

    std::optional<lfs::core::args::LichtMode> parse_licht(
        std::vector<std::string> arguments) {
        std::vector<const char*> argv;
        argv.reserve(arguments.size());
        for (const auto& argument : arguments)
            argv.push_back(argument.c_str());
        auto parsed = lfs::core::args::parse_args(
            static_cast<int>(argv.size()), argv.data());
        if (!parsed)
            return std::nullopt;
        if (auto* mode = std::get_if<lfs::core::args::LichtMode>(&*parsed))
            return *mode;
        return std::nullopt;
    }

    void expect_extracted_file(const fs::path& project, const fs::path& dataset,
                               const std::string& rel_path,
                               const fs::path& extract_root) {
        auto document = require_result_ptr(ProjectDocument::open(project));
        ASSERT_TRUE(require_result(extract_embedded_dataset(*document, extract_root)));
        EXPECT_EQ(bytes(dataset / rel_path), bytes(extract_root / rel_path));
    }

    TEST(LichtCommand, EmbedsReferencedColmapDatasetInSupportedLayouts) {
        for (const auto layout : {"sparse0", "sparse", "flat"}) {
            TemporaryDirectory temporary;
            const auto dataset = make_colmap_dataset(temporary.path / layout, layout);
            const auto project = make_project(temporary.path / "project.licht", dataset);
            const auto mode = parse_licht({"LichtFeld-Studio", "licht",
                                           project.string(), "--embed"});
            ASSERT_TRUE(mode);
            ASSERT_EQ(lfs::app::run_licht_command(*mode), 0);
            auto document = require_result_ptr(ProjectDocument::open(project));
            const auto manifest = require_result(document->parameters().embedded_dataset());
            ASSERT_TRUE(manifest);
            EXPECT_EQ(manifest->entries.size(), 4u);
            EXPECT_NE(std::ranges::find_if(manifest->entries, [](const auto& entry) {
                          return entry.rel_path == "images/frame.png";
                      }),
                      manifest->entries.end());
            expect_extracted_file(project, dataset, "images/frame.png",
                                  temporary.path / "extracted");
        }
    }

    TEST(LichtCommand, RelativeOverrideUsesCurrentDirectoryAndReplacesPayload) {
        TemporaryDirectory temporary;
        const auto first = make_colmap_dataset(temporary.path / "first", "sparse0");
        const auto second = make_colmap_dataset(temporary.path / "selected", "flat");
        const auto changed_image = valid_png(std::byte{0x11});
        write_file_bytes(second / "images/frame.png", changed_image);
        const auto project = make_project(temporary.path / "project.licht", first);
        const auto previous = fs::current_path();
        fs::current_path(temporary.path);
        const auto mode = parse_licht({"LichtFeld-Studio", "licht",
                                       project.string(), "--embed", "selected"});
        fs::current_path(previous);
        ASSERT_TRUE(mode);
        ASSERT_TRUE(mode->dataset_path);
        EXPECT_EQ(*mode->dataset_path, second);
        ASSERT_EQ(lfs::app::run_licht_command(*mode), 0);
        auto document = require_result_ptr(ProjectDocument::open(project));
        const auto manifest = require_result(document->parameters().embedded_dataset());
        ASSERT_TRUE(manifest);
        EXPECT_EQ(document->dataset_source_uuids().size(), manifest->entries.size());
        expect_extracted_file(project, second, "images/frame.png",
                              temporary.path / "extracted");
        EXPECT_EQ(lfs::io::Loader::getDatasetType(first), lfs::io::DatasetType::COLMAP);
        const auto unchanged = bytes(project);
        const auto absolute_mode = parse_licht({"LichtFeld-Studio", "licht",
                                                project.string(), "--embed", second.string()});
        ASSERT_TRUE(absolute_mode);
        ASSERT_EQ(lfs::app::run_licht_command(*absolute_mode), 0);
        EXPECT_EQ(bytes(project), unchanged);

        const auto same_content_path = make_colmap_dataset(
            temporary.path / "same-content", "flat");
        write_file_bytes(same_content_path / "images/frame.png", changed_image);
        const auto before_same_content_override = bytes(project);
        const auto same_content_mode = parse_licht({"LichtFeld-Studio", "licht", project.string(), "--embed",
                                                    same_content_path.string()});
        ASSERT_TRUE(same_content_mode);
        ASSERT_EQ(lfs::app::run_licht_command(*same_content_mode), 0);
        EXPECT_NE(bytes(project), before_same_content_override);
        auto replaced_document = require_result_ptr(ProjectDocument::open(project));
        const auto replaced_manifest = require_result(
            replaced_document->parameters().embedded_dataset());
        ASSERT_TRUE(replaced_manifest);
        EXPECT_EQ(replaced_document->dataset_source_uuids().size(),
                  replaced_manifest->entries.size());
    }

    TEST(LichtCommand, EmbedsTransformsFramesAndSidecars) {
        TemporaryDirectory temporary;
        const auto dataset = make_transforms_dataset(temporary.path / "transforms");
        const auto project = make_project(temporary.path / "project.licht", dataset);
        const auto mode = parse_licht({"LichtFeld-Studio", "licht",
                                       project.string(), "--embed"});
        ASSERT_TRUE(mode);
        ASSERT_EQ(lfs::app::run_licht_command(*mode), 0);
        auto document = require_result_ptr(ProjectDocument::open(project));
        const auto manifest = require_result(document->parameters().embedded_dataset());
        ASSERT_TRUE(manifest);
        EXPECT_EQ(std::ranges::count(manifest->entries, "image", &EmbeddedDatasetEntry::kind), 2);
        EXPECT_EQ(std::ranges::count(manifest->entries, "mask", &EmbeddedDatasetEntry::kind), 1);
        EXPECT_EQ(std::ranges::count(manifest->entries, "depth", &EmbeddedDatasetEntry::kind), 1);
        EXPECT_EQ(std::ranges::count(manifest->entries, "normal", &EmbeddedDatasetEntry::kind), 1);
        EXPECT_EQ(std::ranges::count(manifest->entries, "sparse", &EmbeddedDatasetEntry::kind), 1);
        const auto extract_root = temporary.path / "extracted";
        ASSERT_TRUE(require_result(extract_embedded_dataset(*document, extract_root)));
        const auto [cameras, center, splits] =
            lfs::io::read_transforms_cameras_and_images(extract_root, {});
        (void)center;
        (void)splits;
        EXPECT_EQ(cameras.size(), 2u);
        auto loader = lfs::io::Loader::create();
        ASSERT_NE(loader, nullptr);
        auto loaded = loader->load(extract_root);
        ASSERT_TRUE(loaded) << loaded.error().format();
        const auto& scene = std::get<lfs::io::LoadedScene>(loaded->data);
        ASSERT_NE(scene.point_cloud, nullptr);
        EXPECT_EQ(scene.point_cloud->size(), 1);
    }

    TEST(LichtCommand, UnknownAndUnreachableDatasetsFailWithoutProjectRewrite) {
        TemporaryDirectory temporary;
        const auto unknown = temporary.path / "unknown";
        fs::create_directories(unknown);
        const auto unknown_project = make_project(temporary.path / "unknown.licht", unknown);
        const auto unknown_before = bytes(unknown_project);
        auto mode = parse_licht({"LichtFeld-Studio", "licht",
                                 unknown_project.string(), "--embed"});
        ASSERT_TRUE(mode);
        EXPECT_EQ(lfs::app::run_licht_command(*mode), 1);
        EXPECT_EQ(bytes(unknown_project), unknown_before);

        const auto dataset = make_colmap_dataset(temporary.path / "unreachable", "flat");
        const auto missing_project = make_project(temporary.path / "missing.licht", dataset);
        fs::rename(dataset, temporary.path / "moved");
        const auto missing_before = bytes(missing_project);
        mode = parse_licht({"LichtFeld-Studio", "licht",
                            missing_project.string(), "--embed"});
        ASSERT_TRUE(mode);
        EXPECT_EQ(lfs::app::run_licht_command(*mode), 1);
        EXPECT_EQ(bytes(missing_project), missing_before);
    }

    TEST(LichtCommand, CompleteDatasetIsAlreadyEmbeddedWithoutProjectRewrite) {
        TemporaryDirectory temporary;
        const auto dataset = make_colmap_dataset(temporary.path / "dataset", "sparse0");
        const auto project = make_project(temporary.path / "project.licht", dataset);
        const auto mode = parse_licht({"LichtFeld-Studio", "licht",
                                       project.string(), "--embed"});
        ASSERT_TRUE(mode);
        ASSERT_EQ(lfs::app::run_licht_command(*mode), 0);
        const auto first_generation = bytes(project);
        ASSERT_EQ(lfs::app::run_licht_command(*mode), 0);
        EXPECT_EQ(bytes(project), first_generation);

        const auto changed_image = valid_png(std::byte{0x11});
        write_file_bytes(dataset / "images/frame.png", changed_image);
        const auto changed_mode = parse_licht({"LichtFeld-Studio", "licht", project.string(), "--embed",
                                               dataset.string()});
        ASSERT_TRUE(changed_mode);
        ASSERT_EQ(lfs::app::run_licht_command(*changed_mode), 0);
        EXPECT_NE(bytes(project), first_generation);
        expect_extracted_file(project, dataset, "images/frame.png",
                              temporary.path / "changed-extracted");
    }

} // namespace
