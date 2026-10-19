/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "io/project_document.hpp"
#include "io/project_operations.hpp"
#include "licht_test_support.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    void write_test_png(const fs::path& path) {
        auto image = lfs::core::Tensor::empty(
            {3, 4, 3}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
        for (std::size_t i = 0; i < image.numel(); ++i) {
            image.ptr<std::uint8_t>()[i] = static_cast<std::uint8_t>(i * 17);
        }
        lfs::core::save_image_u8(path, image);
    }

    fs::path make_project_with_external_image(
        const fs::path& root, const lfs::core::Uuid& project_uuid,
        const lfs::core::Uuid& file_uuid) {
        const auto dataset_root = root / "dataset";
        const auto image_path = dataset_root / "images" / "frame.png";
        fs::create_directories(image_path.parent_path());
        write_test_png(image_path);

        auto document = require_result(ProjectDocument::create(
            project_uuid, 1'700'000'000'000'000'000));
        auto snapshot = require_result(document.parameters().snapshot());
        snapshot.dataset.images = "images";
        require_status(document.edit_parameters().set_snapshot(snapshot));
        const auto dataset_uuid = require_result(upsert_path_reference(
            document.edit_references(), {}, dataset_root, "dataset.root",
            "dataset"));
        require_status(document.edit_project().set_dataset_reference(dataset_uuid));

        ProjectDocumentSaveOptions options;
        options.file_uuid = file_uuid;
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        options.disk_reserve_bytes = 0;
        const auto project_path = root / "thumbnail-sources.licht";
        static_cast<void>(require_result(document.save(project_path, options)));
        return project_path;
    }

    void embed_image(const fs::path& project_path, const fs::path& image_path,
                     const std::vector<std::byte>& image_bytes,
                     const lfs::core::Uuid& image_uuid,
                     const lfs::core::Uuid& file_uuid) {
        auto document = require_result(ProjectDocument::open(project_path));
        const EmbeddedDatasetEntry image_entry{
            .rel_path = "images/frame.png",
            .kind = "image",
            .chunk_uuid = image_uuid,
            .bytes = image_bytes.size(),
            .xxh3_128 = xxh3_128(image_bytes),
        };
        const EmbeddedDatasetManifest manifest{
            .schema_version = 1,
            .images_folder = "images",
            .complete = true,
            .entries = {image_entry},
        };
        const std::array sources{
            DatasetEmbedSource{.entry = image_entry, .source_path = image_path},
        };
        ProjectDocumentSaveOptions options;
        options.file_uuid = file_uuid;
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        options.disk_reserve_bytes = 0;
        static_cast<void>(require_result(document.embed_dataset_batch(
            manifest, std::span(sources), options)));
    }

    TEST(ProjectThumbnailSources, DecodesBothSourcesWithoutWritingProject) {
        TemporaryDirectory temporary;
        const auto project_path = make_project_with_external_image(
            temporary.path, fixed_uuid(2190), fixed_uuid(2191));
        const auto image_path = temporary.path / "dataset/images/frame.png";
        const auto image_bytes = read_file_bytes(image_path);
        embed_image(project_path, image_path, image_bytes, fixed_uuid(2192),
                    fixed_uuid(2193));

        const auto bytes_before = read_file_bytes(project_path);
        const auto card_before = require_result(inspect_project_card(project_path));
        const auto availability = require_result(
            inspect_project_thumbnail_sources(project_path));
        const auto bytes_after = read_file_bytes(project_path);
        const auto card_after = require_result(inspect_project_card(project_path));

        EXPECT_TRUE(availability.first_dataset_image);
        EXPECT_TRUE(availability.first_embedded_image);
        EXPECT_EQ(bytes_after, bytes_before);
        EXPECT_EQ(card_after.generation, card_before.generation);
        EXPECT_EQ(card_after.commit_uuid, card_before.commit_uuid);
    }

    TEST(ProjectThumbnailSources, InvalidEmbeddedImageDoesNotHideExternalSource) {
        TemporaryDirectory temporary;
        const auto project_path = make_project_with_external_image(
            temporary.path, fixed_uuid(2200), fixed_uuid(2201));
        const auto invalid_path = temporary.path / "invalid-image.bin";
        const std::vector<std::byte> invalid_bytes{
            std::byte{'n'}, std::byte{'o'}, std::byte{'t'}, std::byte{'a'},
            std::byte{'n'}, std::byte{'i'}, std::byte{'m'}, std::byte{'a'},
            std::byte{'g'}, std::byte{'e'}};
        {
            std::ofstream output(invalid_path, std::ios::binary);
            ASSERT_TRUE(output.good());
            output.write(reinterpret_cast<const char*>(invalid_bytes.data()),
                         static_cast<std::streamsize>(invalid_bytes.size()));
            ASSERT_TRUE(output.good());
        }
        embed_image(project_path, invalid_path, invalid_bytes, fixed_uuid(2202),
                    fixed_uuid(2203));

        const auto bytes_before = read_file_bytes(project_path);
        const auto card_before = require_result(inspect_project_card(project_path));
        const auto availability = require_result(
            inspect_project_thumbnail_sources(project_path));
        const auto bytes_after = read_file_bytes(project_path);
        const auto card_after = require_result(inspect_project_card(project_path));

        EXPECT_TRUE(availability.first_dataset_image);
        EXPECT_FALSE(availability.first_embedded_image);
        EXPECT_EQ(bytes_after, bytes_before);
        EXPECT_EQ(card_after.generation, card_before.generation);
        EXPECT_EQ(card_after.commit_uuid, card_before.commit_uuid);
    }

    TEST(ProjectThumbnailSources, InvalidEmbeddedManifestDoesNotHideExternalSource) {
        TemporaryDirectory temporary;
        const auto project_path = make_project_with_external_image(
            temporary.path, fixed_uuid(2210), fixed_uuid(2211));
        auto document = require_result(ProjectDocument::open(project_path));
        using Json = lfs::io::JsonChapterDom::Json;
        require_status(document.edit_parameters().dom().set_json(
            "dataset.embedded_dataset", Json{{"schema_version", 1}}));
        ProjectDocumentSaveOptions options;
        options.file_uuid = fixed_uuid(2212);
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        options.disk_reserve_bytes = 0;
        static_cast<void>(require_result(document.save(project_path, options)));

        const auto bytes_before = read_file_bytes(project_path);
        const auto card_before = require_result(inspect_project_card(project_path));
        const auto availability = require_result(
            inspect_project_thumbnail_sources(project_path));
        const auto bytes_after = read_file_bytes(project_path);
        const auto card_after = require_result(inspect_project_card(project_path));

        EXPECT_TRUE(availability.first_dataset_image);
        EXPECT_FALSE(availability.first_embedded_image);
        EXPECT_EQ(bytes_after, bytes_before);
        EXPECT_EQ(card_after.generation, card_before.generation);
        EXPECT_EQ(card_after.commit_uuid, card_before.commit_uuid);
    }

} // namespace
