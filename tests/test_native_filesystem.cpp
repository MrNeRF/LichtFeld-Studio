/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/filesystem.hpp"
#include "core/path_utils.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif
namespace {
    namespace fs = lfs::core::fs;
    class NativeFilesystemTest : public ::testing::Test {
    protected:
        void SetUp() override {
#ifdef _WIN32
            using Query = BOOLEAN(NTAPI*)();
            auto query = reinterpret_cast<Query>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlAreLongPathsEnabled"));
            ASSERT_NE(query, nullptr);
            RecordProperty("process_long_paths_enabled", query() ? "true" : "false");
#ifdef LFS_REQUIRE_LONG_PATHS_DISABLED
            ASSERT_FALSE(query()) << "This target must exercise Windows without long-path opt-in";
#endif
#endif
            root_ = std::filesystem::temp_directory_path() / ("lfs-native-fs-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            deep_ = root_;
            for (int i = 0; i < 12; ++i)
                deep_ /= lfs::core::utf8_to_path("unicode_測試_à_component_01234567890123456789");
            ASSERT_GT(deep_.native().size(), 500u);
            ASSERT_TRUE(fs::create_directories(deep_));
        }
        void TearDown() override {
            std::error_code error;
            fs::remove_all(root_, error);
            EXPECT_FALSE(error) << error.message();
            EXPECT_FALSE(fs::exists(root_));
        }
        fs::path root_, deep_;
    };
    TEST_F(NativeFilesystemTest, StreamsAndFileOperationsWorkBeyondTraditionalLimit) {
        const auto file = deep_ / "source.bin";
        const std::string payload("native\0filesystem\r\n", 19);
        {
            std::ofstream out;
            ASSERT_TRUE(lfs::core::open_file_for_write(file, std::ios::binary, out));
            out.write(payload.data(), payload.size());
        }
        ASSERT_TRUE(fs::exists(file));
        ASSERT_TRUE(fs::is_regular_file(file));
        EXPECT_EQ(fs::file_size(file), payload.size());
        {
            fs::ifstream stream(file, std::ios::binary);
            EXPECT_TRUE(stream.is_open());
        }
        std::ifstream in;
        ASSERT_TRUE(lfs::core::open_file_for_read(file, std::ios::binary, in));
        const std::string actual{std::istreambuf_iterator<char>(in), {}};
        EXPECT_EQ(actual, payload);
        in.close();
        auto* raw = lfs::core::open_file(file, "rb");
        ASSERT_NE(raw, nullptr);
        EXPECT_EQ(std::fclose(raw), 0);
        const auto copy = deep_ / "copy.bin", renamed = deep_ / "renamed.bin";
        ASSERT_TRUE(fs::copy_file(file, copy));
        fs::rename(copy, renamed);
        EXPECT_FALSE(fs::exists(copy));
        EXPECT_EQ(fs::file_size(renamed), payload.size());
        const auto link = deep_ / "hardlink.bin";
        fs::create_hard_link(file, link);
        EXPECT_TRUE(fs::equivalent(file, link));
        EXPECT_GT(fs::space(deep_).capacity, 0u);
        const auto time = fs::last_write_time(file);
        fs::last_write_time(file, time);
        EXPECT_TRUE(fs::remove(link));
        EXPECT_TRUE(fs::remove(renamed));
    }
    TEST_F(NativeFilesystemTest, IteratorsKeepApplicationPathsOutOfExtendedNamespace) {
        const auto file = deep_ / "item.bin";
        {
            std::ofstream out;
            ASSERT_TRUE(lfs::core::open_file_for_write(file, out));
            out << "data";
        }
        int files = 0;
        for (const auto& entry : fs::directory_iterator(deep_)) {
            EXPECT_EQ(entry.path(), file);
            EXPECT_TRUE(entry.is_regular_file());
            ++files;
#ifdef _WIN32
            EXPECT_FALSE(entry.path().native().starts_with(L"\\\\?\\"));
#endif
        }
        EXPECT_EQ(files, 1);
        files = 0;
        for (const auto& entry : fs::recursive_directory_iterator(root_))
            if (entry.is_regular_file()) {
                EXPECT_EQ(entry.path(), file);
                ++files;
            }
        EXPECT_EQ(files, 1);
        std::error_code error;
        fs::directory_iterator missing(deep_ / "missing", error);
        EXPECT_TRUE(error);
        EXPECT_EQ(missing, fs::directory_iterator{});
    }
    TEST_F(NativeFilesystemTest, CanonicalizationHandlesMissingRemainderAndAlternateSpellings) {
        std::error_code error;
        const auto resolved = fs::weakly_canonical(deep_ / "missing" / ".." / "new" / "output.bin", error);
        ASSERT_FALSE(error);
        EXPECT_EQ(resolved, fs::canonical(deep_) / "new" / "output.bin");
        EXPECT_EQ(fs::relative(deep_ / "new" / "output.bin", deep_), fs::path("new") / "output.bin");
#ifdef _WIN32
        EXPECT_EQ(fs::native_path(fs::native_path(deep_)), fs::native_path(deep_));
        EXPECT_EQ(fs::display_path(fs::native_path(deep_)), deep_);
#endif
    }
} // namespace
