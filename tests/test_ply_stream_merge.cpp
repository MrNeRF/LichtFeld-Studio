/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/ply_stream_merge.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
    class PlyStreamMergeTest : public ::testing::Test {
    protected:
        void SetUp() override {
            static std::atomic<unsigned> sequence{0};
            root = std::filesystem::temp_directory_path() /
                   ("lfs_ply_stream_merge_" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                    std::to_string(sequence++));
            std::filesystem::create_directories(root);
        }

        void TearDown() override {
            std::error_code error;
            std::filesystem::remove_all(root, error);
        }

        void write_part(const std::string& name, const std::string& second_property,
                        const std::vector<float>& values, bool truncate = false) const {
            std::ofstream out(root / name, std::ios::binary);
            out << "ply\nformat binary_little_endian 1.0\n"
                << "comment lichtfeld_provenance {}\n"
                << "element vertex " << values.size() / 2 << "\n"
                << "property float x\nproperty float " << second_property << "\nend_header\n";
            const auto bytes = static_cast<std::streamsize>((values.size() - (truncate ? 1 : 0)) * sizeof(float));
            out.write(reinterpret_cast<const char*>(values.data()), bytes);
        }

        std::filesystem::path root;
    };

    TEST_F(PlyStreamMergeTest, ConcatenatesCompleteBinaryRowsInOrder) {
        write_part("a.ply", "f_rest_0", {1.0f, 10.0f});
        write_part("b.ply", "f_rest_0", {2.0f, 20.0f, 3.0f, 30.0f});
        const auto output = root / "merged.ply";
        const auto result = lfs::io::merge_binary_ply_streaming(
            {root / "a.ply", root / "b.ply"}, output);
        ASSERT_TRUE(result.has_value()) << result.error().format();
        std::ifstream in(output, std::ios::binary);
        std::string line;
        bool saw_count = false;
        while (std::getline(in, line)) {
            if (line == "element vertex 3") saw_count = true;
            if (line == "end_header") break;
        }
        ASSERT_TRUE(saw_count);
        float values[6]{};
        in.read(reinterpret_cast<char*>(values), sizeof(values));
        ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(sizeof(values)));
        EXPECT_EQ(std::vector<float>(values, values + 6),
                  (std::vector<float>{1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f}));
    }

    TEST_F(PlyStreamMergeTest, RejectsDifferentShSchemasWithoutReplacingOutput) {
        write_part("a.ply", "f_rest_0", {1.0f, 10.0f});
        write_part("b.ply", "f_rest_1", {2.0f, 20.0f});
        const auto output = root / "merged.ply";
        std::ofstream(output) << "previous result";
        const auto result = lfs::io::merge_binary_ply_streaming(
            {root / "a.ply", root / "b.ply"}, output);
        ASSERT_FALSE(result.has_value());
        std::ifstream in(output);
        std::string contents;
        std::getline(in, contents);
        EXPECT_EQ(contents, "previous result");
    }

    TEST_F(PlyStreamMergeTest, RejectsTruncatedInput) {
        write_part("a.ply", "f_rest_0", {1.0f, 10.0f}, true);
        EXPECT_FALSE(lfs::io::merge_binary_ply_streaming(
            {root / "a.ply"}, root / "merged.ply").has_value());
        EXPECT_FALSE(std::filesystem::exists(root / "merged.ply"));
    }

    TEST_F(PlyStreamMergeTest, CancellationKeepsPreviousOutput) {
        write_part("a.ply", "f_rest_0", {1.0f, 10.0f});
        const auto output = root / "merged.ply";
        std::ofstream(output) << "previous result";
        int checks = 0;
        const auto result = lfs::io::merge_binary_ply_streaming(
            {root / "a.ply"}, output, [&checks] { return ++checks >= 3; });
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CANCELLED);
        std::ifstream in(output);
        std::string contents;
        std::getline(in, contents);
        EXPECT_EQ(contents, "previous result");
    }
} // namespace
