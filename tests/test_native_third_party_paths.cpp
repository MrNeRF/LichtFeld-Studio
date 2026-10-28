/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "gui/rmlui/rml_file_interface.hpp"
#include <archive.h>
#include <archive_entry.h>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
extern "C" {
#include <libavformat/avformat.h>
}
#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <vector>

namespace {
    namespace fs = lfs::core::native_fs;
    class NativeThirdPartyPaths : public ::testing::Test {
    protected:
        void SetUp() override {
#ifdef _WIN32
            const auto enabled = fs::process_long_paths_enabled();
            ASSERT_TRUE(enabled.has_value());
            ASSERT_FALSE(*enabled);
#endif
            root_ = fs::temp_directory_path() / ("lfs-boundaries-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            deep_ = root_;
            for (int i = 0; i < 12; ++i)
                deep_ /= lfs::core::utf8_to_path("directory_\u6e2c\u8a66_01234567890123456789");
            ASSERT_GT(deep_.native().size(), 260u);
            fs::create_directories(deep_);
        }
        void TearDown() override {
            std::error_code ec;
            fs::remove_all(root_, ec);
            EXPECT_FALSE(ec);
        }
        fs::path root_, deep_;
    };
    TEST_F(NativeThirdPartyPaths, RmlFileInterfaceReadsUtf8ResourcesWithoutAGui) {
        const auto path = deep_ / lfs::core::utf8_to_path("resource_\u6e2c\u8a66.rml");
        {
            std::ofstream output(lfs::core::native_fs::native_path(path), std::ios::binary);
            output << "<rml>resource</rml>";
        }
        lfs::vis::gui::RmlFileInterface io;
        const auto handle = io.Open(lfs::core::path_to_utf8(path));
        ASSERT_NE(handle, Rml::FileHandle{});
        struct Close {
            lfs::vis::gui::RmlFileInterface& io;
            Rml::FileHandle handle;
            ~Close() { io.Close(handle); }
        } close{io, handle};
        std::array<char, 32> bytes{};
        const auto count = io.Read(bytes.data(), bytes.size(), handle);
        EXPECT_EQ(std::string(bytes.data(), count), "<rml>resource</rml>");
        EXPECT_EQ(io.Tell(handle), count);
        EXPECT_TRUE(io.Seek(handle, 5, SEEK_SET));
        EXPECT_EQ(io.Tell(handle), 5u);
        EXPECT_EQ(io.Open(lfs::core::path_to_utf8(deep_ / "missing.rml")), Rml::FileHandle{});
    }
    TEST_F(NativeThirdPartyPaths, AssimpFindsRelativeMaterialBesideLongUtf8MeshPath) {
        const auto path = deep_ / lfs::core::utf8_to_path("mesh_\u6e2c\u8a66.obj");
        {
            std::ofstream output(lfs::core::native_fs::native_path(path), std::ios::binary);
            output << "mtllib material.mtl\no triangle\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl expected-material\nf 1 2 3\n";
        }
        {
            std::ofstream output(lfs::core::native_fs::native_path(deep_ / "material.mtl"), std::ios::binary);
            output << "newmtl expected-material\nKd 0.25 0.5 0.75\n";
        }
        Assimp::Importer importer;
        const auto* scene = importer.ReadFile(lfs::core::path_to_native_utf8(path), aiProcess_Triangulate);
        ASSERT_NE(scene, nullptr) << importer.GetErrorString();
        ASSERT_EQ(scene->mNumMeshes, 1u);
        aiString material_name;
        ASSERT_EQ(scene->mMaterials[scene->mMeshes[0]->mMaterialIndex]->Get(AI_MATKEY_NAME, material_name), AI_SUCCESS);
        EXPECT_STREQ(material_name.C_Str(), "expected-material");
    }
    TEST_F(NativeThirdPartyPaths, LibarchivePreservesUnicodeMemberNamesAndLongArchivePaths) {
        const auto path = deep_ / lfs::core::utf8_to_path("archive_\u6e2c\u8a66.zip");
        const auto name = std::string("texture_\u6e2c\u8a66.bin");
        auto* writer = archive_write_new();
        ASSERT_NE(writer, nullptr);
        struct Writer {
            struct archive* value;
            ~Writer() { archive_write_free(value); }
        } release_writer{writer};
        ASSERT_EQ(archive_write_set_format_zip(writer), ARCHIVE_OK);
        ASSERT_EQ(archive_write_set_format_option(writer, "zip", "hdrcharset", "UTF-8"), ARCHIVE_OK);
        ASSERT_EQ(archive_write_open_filename_w(writer, fs::native_path(path).c_str()), ARCHIVE_OK);
        auto* entry = archive_entry_new();
        ASSERT_NE(entry, nullptr);
        struct Entry {
            archive_entry* value;
            ~Entry() { archive_entry_free(value); }
        } release_entry{entry};
        archive_entry_set_pathname_utf8(entry, name.c_str());
        archive_entry_set_filetype(entry, AE_IFREG);
        archive_entry_set_perm(entry, 0644);
        archive_entry_set_size(entry, 7);
        ASSERT_EQ(archive_write_header(writer, entry), ARCHIVE_OK) << archive_error_string(writer);
        ASSERT_EQ(archive_write_data(writer, "payload", 7), 7);
        ASSERT_EQ(archive_write_close(writer), ARCHIVE_OK);
        auto* reader = archive_read_new();
        ASSERT_NE(reader, nullptr);
        struct Reader {
            struct archive* value;
            ~Reader() { archive_read_free(value); }
        } release_reader{reader};
        archive_read_support_filter_all(reader);
        archive_read_support_format_zip(reader);
        ASSERT_EQ(archive_read_open_filename_w(reader, fs::native_path(path).c_str(), 10240), ARCHIVE_OK);
        archive_entry* loaded = nullptr;
        ASSERT_EQ(archive_read_next_header(reader, &loaded), ARCHIVE_OK);
        ASSERT_NE(archive_entry_pathname_utf8(loaded), nullptr);
        EXPECT_EQ(archive_entry_pathname_utf8(loaded), name);
        std::array<char, 7> data{};
        ASSERT_EQ(archive_read_data(reader, data.data(), data.size()), 7);
        EXPECT_EQ(std::string(data.data(), data.size()), "payload");
    }
    TEST_F(NativeThirdPartyPaths, FfmpegDemuxesAndWritesLongUtf8Paths) {
        const auto path = deep_ / lfs::core::utf8_to_path("audio_\u6e2c\u8a66.wav");
        std::vector<unsigned char> wav;
        const auto text = [&](const char* s) { wav.insert(wav.end(), s, s + 4); };
        const auto u16 = [&](unsigned value) { wav.push_back(value & 255); wav.push_back((value >> 8) & 255); };
        const auto u32 = [&](unsigned value) { for(int i=0;i<4;++i) wav.push_back((value >> (8*i)) & 255); };
        text("RIFF");
        u32(36 + 1600);
        text("WAVE");
        text("fmt ");
        u32(16);
        u16(1);
        u16(1);
        u32(8000);
        u32(16000);
        u16(2);
        u16(16);
        text("data");
        u32(1600);
        wav.resize(wav.size() + 1600);
        {
            std::ofstream output(lfs::core::native_fs::native_path(path), std::ios::binary);
            output.write(reinterpret_cast<const char*>(wav.data()), wav.size());
        }
        AVFormatContext* context = nullptr;
        ASSERT_GE(avformat_open_input(&context, lfs::core::path_to_native_utf8(path).c_str(), nullptr, nullptr), 0);
        struct Close {
            AVFormatContext*& value;
            ~Close() { avformat_close_input(&value); }
        } close{context};
        ASSERT_GE(avformat_find_stream_info(context, nullptr), 0);
        ASSERT_EQ(context->nb_streams, 1u);
        EXPECT_EQ(context->streams[0]->codecpar->sample_rate, 8000);
        AVIOContext* output = nullptr;
        const auto destination = deep_ / lfs::core::utf8_to_path("export_\u6e2c\u8a66.bin");
        ASSERT_GE(avio_open(&output, lfs::core::path_to_native_utf8(destination).c_str(), AVIO_FLAG_WRITE), 0);
        avio_write(output, wav.data(), static_cast<int>(wav.size()));
        EXPECT_GE(avio_closep(&output), 0);
        EXPECT_EQ(fs::file_size(destination), wav.size());
    }
} // namespace
