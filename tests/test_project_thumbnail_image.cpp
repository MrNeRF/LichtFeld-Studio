/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/checkpoint_format.hpp"
#include "core/image_io.hpp"
#include "io/project_document.hpp"
#include "io/project_operations.hpp"
#include "licht_test_support.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <span>
#include <vector>

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

    void save_document(ProjectDocument& document, const fs::path& path) {
        ProjectDocumentSaveOptions options;
        options.file_uuid = fixed_uuid(2173);
        options.index_compression = IndexCompression::StoredForDeterministicTests;
        options.disk_reserve_bytes = 0;
        static_cast<void>(require_result(document.save(path, options)));
    }

    bool starts_with(const std::vector<std::byte>& bytes,
                     const std::span<const unsigned char> signature) {
        if (bytes.size() < signature.size()) {
            return false;
        }
        for (std::size_t i = 0; i < signature.size(); ++i) {
            if (std::to_integer<unsigned char>(bytes[i]) != signature[i]) {
                return false;
            }
        }
        return true;
    }

    TEST(ProjectThumbnailImage, ReencodesJpegAndPreservesProjectData) {
        TemporaryDirectory temporary;
        const auto project_path = temporary.path / "image-source.licht";
        auto document = require_result(ProjectDocument::create(
            fixed_uuid(2170), 1'700'000'000'000'000'000));
        const auto checkpoint_uuid = fixed_uuid(2171);
        require_status(document.set_checkpoint(
            checkpoint_uuid,
            require_result(LazyChunkValue::from_owned(
                checkpoint_payload(42), checkpoint_uuid))));
        SceneNodeRecord node;
        node.uuid = fixed_uuid(2172);
        node.type = "splat";
        node.name = "Thumbnail test model";
        node.payload = PayloadBinding{
            .fourcc = "CKPT",
            .instance_uuid = checkpoint_uuid,
            .source_kind = "checkpoint"};
        require_status(document.edit_scene_graph().upsert_node(node));
        require_status(document.edit_scene_graph().set_training_model_uuid(node.uuid));
        save_document(document, project_path);

        const auto image_path = temporary.path / "selected.jpg";
        auto image = lfs::core::Tensor::empty(
            {768, 1024, 3}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
        for (std::size_t i = 0; i < image.numel(); ++i) {
            image.ptr<std::uint8_t>()[i] = static_cast<std::uint8_t>(i * 13);
        }
        lfs::core::save_image_u8(image_path, image);
        const auto jpeg_bytes = read_file_bytes(image_path);
        constexpr std::array<unsigned char, 2> jpeg_signature{0xff, 0xd8};
        ASSERT_TRUE(starts_with(jpeg_bytes, jpeg_signature));

        // This is the previous byte-oriented operation: it reports success
        // while storing the JPEG stream directly as the project thumbnail.
        static_cast<void>(require_result(
            set_project_preview(project_path, jpeg_bytes)));
        auto raw_reader = require_result(ProjectReader::open(project_path));
        const auto raw_preview = require_result(raw_reader.read_preview());
        EXPECT_TRUE(starts_with(raw_preview, jpeg_signature));
        constexpr std::array<unsigned char, 8> png_signature{
            0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
        EXPECT_FALSE(starts_with(raw_preview, png_signature));
        const auto* scene_row = raw_reader.find(
            FOURCC_SCNG, raw_reader.superblock().project_uuid);
        const auto* checkpoint_row = raw_reader.find(FOURCC_CKPT, checkpoint_uuid);
        ASSERT_NE(scene_row, nullptr);
        ASSERT_NE(checkpoint_row, nullptr);
        const auto scene_before = require_result(raw_reader.read_chunk(*scene_row));
        const auto checkpoint_before = require_result(
            raw_reader.read_chunk(*checkpoint_row));

        static_cast<void>(require_result(
            preview_from_image_file(project_path, image_path)));
        auto reader = require_result(ProjectReader::open(project_path));
        const auto preview = require_result(reader.read_preview());
        ASSERT_TRUE(starts_with(preview, png_signature));
        const auto [pixels, width, height, channels] =
            lfs::core::load_image_from_memory(
                reinterpret_cast<const std::uint8_t*>(preview.data()),
                preview.size());
        ASSERT_NE(pixels, nullptr);
        EXPECT_LE(std::max(width, height), 512);
        EXPECT_GT(width, 0);
        EXPECT_GT(height, 0);
        EXPECT_GE(channels, 1);
        lfs::core::free_image(pixels);

        const auto* updated_scene_row = reader.find(
            FOURCC_SCNG, reader.superblock().project_uuid);
        const auto* updated_checkpoint_row = reader.find(
            FOURCC_CKPT, checkpoint_uuid);
        ASSERT_NE(updated_scene_row, nullptr);
        ASSERT_NE(updated_checkpoint_row, nullptr);
        EXPECT_EQ(require_result(reader.read_chunk(*updated_scene_row)), scene_before);
        EXPECT_EQ(require_result(reader.read_chunk(*updated_checkpoint_row)),
                  checkpoint_before);

        const auto invalid_path = temporary.path / "invalid.jpg";
        {
            std::ofstream output(invalid_path, std::ios::binary);
            ASSERT_TRUE(output.good());
            output << "not a decodable image";
        }
        const auto bytes_before_invalid = read_file_bytes(project_path);
        const auto card_before_invalid = require_result(
            inspect_project_card(project_path));
        const auto invalid_result = preview_from_image_file(
            project_path, invalid_path);
        ASSERT_FALSE(invalid_result);
        EXPECT_EQ(read_file_bytes(project_path), bytes_before_invalid);
        const auto card_after_invalid = require_result(
            inspect_project_card(project_path));
        EXPECT_EQ(card_after_invalid.generation, card_before_invalid.generation);
        EXPECT_EQ(card_after_invalid.commit_uuid, card_before_invalid.commit_uuid);
    }

} // namespace
