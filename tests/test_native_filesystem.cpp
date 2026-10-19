/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif
namespace {
    namespace fs = lfs::core::native_fs;
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
                deep_ /= lfs::core::utf8_to_path("unicode_æ¸¬è©¦_Ã _component_01234567890123456789");
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
            std::ifstream stream(lfs::core::native_fs::native_path(file), std::ios::binary);
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
    TEST_F(NativeFilesystemTest, EntryPathsUseLogicalSpelling) {
        const auto path = root_ / "short.bin";
        {
            std::ofstream output(lfs::core::native_fs::native_path(path), std::ios::binary);
            output << "short";
        }
        bool found = false;
        for (const auto& entry : fs::directory_iterator(root_)) {
            if (entry.path().filename() == "short.bin") {
                EXPECT_EQ(fs::entry_path(root_, entry.path()), path);
                found = true;
            }
        }
        EXPECT_TRUE(found);
    }
    TEST_F(NativeFilesystemTest, Utf8StringStreamArgumentsDoNotRoundTripThroughTheWindowsCodePage) {
        const auto path = deep_ / lfs::core::utf8_to_path("stream_\u6e2c\u8a66.bin");
        const auto name = lfs::core::path_to_utf8(path);
        {
            std::ofstream output(lfs::core::native_fs::native_path(lfs::core::native_fs::from_utf8(name)), std::ios::binary);
            ASSERT_TRUE(output);
            output << "unicode";
        }
        std::ifstream input(lfs::core::native_fs::native_path(lfs::core::native_fs::from_utf8(name)), std::ios::binary);
        ASSERT_TRUE(input);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(input), {}), "unicode");
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
            EXPECT_EQ(fs::display_path(entry.path()), file);
            EXPECT_TRUE(entry.is_regular_file());
            ++files;
#ifdef _WIN32
            EXPECT_FALSE(fs::entry_path(deep_, entry.path()).native().starts_with(L"\\\\?\\"));
#endif
        }
        EXPECT_EQ(files, 1);
        files = 0;
        for (const auto& entry : fs::recursive_directory_iterator(root_))
            if (entry.is_regular_file()) {
                EXPECT_EQ(fs::display_path(entry.path()), file);
                ++files;
            }
        EXPECT_EQ(files, 1);
        std::error_code error;
        std::filesystem::directory_iterator missing(lfs::core::native_fs::traversal_path(deep_ / "missing"), error);
        EXPECT_TRUE(error);
        EXPECT_EQ(missing, std::filesystem::directory_iterator{});
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

    TEST_F(NativeFilesystemTest, OrdinaryCanonicalizationKeepsStandardSemantics) {
        for (const auto& value : {fs::path{}, fs::path("missing-relative-path"), root_, root_ / "missing" / ".." / "new"}) {
            std::error_code expected_error, actual_error;
            const auto expected = std::filesystem::weakly_canonical(value, expected_error);
            const auto actual = fs::weakly_canonical(value, actual_error);
            EXPECT_EQ(actual_error, expected_error);
            EXPECT_EQ(actual, expected);
        }
    }

    TEST_F(NativeFilesystemTest, RelativePathsAndRecursiveCopyWork) {
        const auto previous = fs::current_path();
        struct Restore {
            fs::path value;
            ~Restore() {
                std::error_code ec;
                fs::current_path(value, ec);
            }
        } restore{previous};
        fs::current_path(root_.parent_path());
        const auto relative = fs::relative(deep_, fs::current_path());
        ASSERT_FALSE(relative.empty());
        {
            std::ofstream file(lfs::core::native_fs::native_path(relative / "relative.bin"), std::ios::binary);
            ASSERT_TRUE(file);
            file << "payload";
        }
        for (const auto& entry : fs::directory_iterator(relative)) {
            EXPECT_EQ(fs::entry_path(relative, entry.path()), relative / "relative.bin");
            EXPECT_EQ(fs::entry_path(relative, entry.path()), relative / "relative.bin");
        }
        const auto copy = root_.parent_path() / (root_.filename().string() + "-copy");
        struct Cleanup {
            fs::path value;
            ~Cleanup() {
                std::error_code ec;
                fs::remove_all(value, ec);
            }
        } cleanup{copy};
        fs::copy(root_, copy, fs::copy_options::recursive);
        EXPECT_EQ(fs::file_size(copy / deep_.lexically_relative(root_) / "relative.bin"), 7u);
        std::error_code ec;
        EXPECT_FALSE(fs::copy_file(relative / "relative.bin", deep_ / "missing-parent" / "file.bin", ec));
        EXPECT_TRUE(ec);
    }
#ifdef _WIN32
    TEST_F(NativeFilesystemTest, UncOperationsWorkWhenTheLocalShareIsAvailable) {
        const auto drive = root_.root_name().native();
        if (drive.size() != 2 || drive[1] != L':')
            GTEST_SKIP() << "Temporary root is not on a drive-backed local volume";
        fs::path share(L"\\\\localhost\\" + drive.substr(0, 1) + L"$" + root_.native().substr(2));
        fs::path cleanup_root;
        const auto size = GetEnvironmentVariableW(L"LFS_TEST_UNC_ROOT", nullptr, 0);
        if (size) {
            std::wstring configured(size, L'\0');
            configured.resize(GetEnvironmentVariableW(L"LFS_TEST_UNC_ROOT", configured.data(), size));
            share = fs::path(configured) / root_.filename();
            cleanup_root = share;
            fs::create_directories(share / deep_.lexically_relative(root_));
        }
        struct Cleanup {
            fs::path value;
            ~Cleanup() {
                if (!value.empty()) {
                    std::error_code ec;
                    fs::remove_all(value, ec);
                }
            }
        } cleanup{cleanup_root};
        std::error_code ec;
        if (!fs::is_directory(share, ec)) {
            if (GetEnvironmentVariableW(L"LFS_TEST_REQUIRE_UNC", nullptr, 0))
                FAIL() << "Required UNC share unavailable: " << ec.message();
            GTEST_SKIP() << "Local administrative share unavailable: " << ec.message();
        }
        const auto network_deep = share / deep_.lexically_relative(root_);
        const auto file = network_deep / "network.bin";
        {
            std::ofstream output(lfs::core::native_fs::native_path(file), std::ios::binary);
            ASSERT_TRUE(output);
            output << "network";
        }
        EXPECT_EQ(fs::file_size(file), 7u);
        const auto renamed = network_deep / "renamed.bin";
        fs::rename(file, renamed);
        EXPECT_TRUE(fs::exists(renamed));
        EXPECT_EQ(fs::display_path(fs::native_path(renamed)), renamed);
        const auto missing = fs::weakly_canonical(network_deep / "new" / "file.bin", ec);
        ASSERT_FALSE(ec) << ec.message();
        EXPECT_EQ(missing.filename().string(), "file.bin");
        auto lower_unc = fs::native_path(network_deep / "new" / "file.bin").native();
        lower_unc.replace(4, 3, L"unc");
        EXPECT_EQ(fs::weakly_canonical(fs::path(lower_unc), ec), missing);
        EXPECT_FALSE(ec);
        EXPECT_EQ(fs::display_path(fs::path(lower_unc)), network_deep / "new" / "file.bin");
        size_t files = 0;
        for (const auto& entry : fs::directory_iterator(network_deep))
            if (entry.is_regular_file())
                ++files;
        EXPECT_EQ(files, 1u);
    }
    TEST_F(NativeFilesystemTest, RelativeSymlinkTargetsRemainRelative) {
        {
            std::ofstream output(lfs::core::native_fs::native_path(deep_ / "target.bin"), std::ios::binary);
            output << "target";
        }
        std::error_code ec;
        fs::create_symlink(fs::path("target.bin"), deep_ / "link.bin", ec);
        if (ec == std::error_code(ERROR_PRIVILEGE_NOT_HELD, std::system_category()))
            GTEST_SKIP() << "Windows symlink privilege is unavailable";
        ASSERT_FALSE(ec) << ec.message();
        EXPECT_EQ(fs::read_symlink(deep_ / "link.bin"), fs::path("target.bin"));
        EXPECT_TRUE(fs::equivalent(deep_ / "link.bin", deep_ / "target.bin"));
    }
    TEST_F(NativeFilesystemTest, VerbatimNamesDoNotAliasOrdinaryFiles) {
        const auto ordinary = root_ / "name";
        {
            std::ofstream output(lfs::core::native_fs::native_path(ordinary), std::ios::binary);
            output << "ordinary";
        }
        for (const auto* name : {"name.", "name ", "CON.licht"}) {
            const auto path = root_ / name;
            {
                std::ofstream output(lfs::core::native_fs::native_path(path), std::ios::binary);
                ASSERT_TRUE(output);
                output << "verbatim";
            }
            std::ifstream input(lfs::core::native_fs::native_path(path), std::ios::binary);
            EXPECT_EQ(std::string(std::istreambuf_iterator<char>(input), {}), "verbatim");
            input.close();
            EXPECT_TRUE(fs::remove(path));
            EXPECT_TRUE(fs::exists(ordinary));
        }
        std::ifstream input(lfs::core::native_fs::native_path(ordinary), std::ios::binary);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(input), {}), "ordinary");
        EXPECT_EQ(fs::native_path(fs::path("NUL")), fs::path("NUL"));
    }
    TEST_F(NativeFilesystemTest, LongCurrentDirectoryStillSupportsRelativeIo) {
        const auto previous = fs::current_path();
        struct Restore {
            fs::path value;
            ~Restore() {
                std::error_code ec;
                fs::current_path(value, ec);
            }
        } restore{previous};
        std::error_code ec;
        fs::current_path(deep_, ec);
        if (ec.value() == ERROR_FILENAME_EXCED_RANGE)
            GTEST_SKIP() << "SetCurrentDirectoryW rejects long working directories without process opt-in; absolute file I/O is tested separately";
        ASSERT_FALSE(ec) << ec.message();
        EXPECT_TRUE(fs::equivalent(fs::current_path(), deep_));
        {
            std::ofstream output(lfs::core::native_fs::native_path("relative.bin"), std::ios::binary);
            ASSERT_TRUE(output);
            output << "long-cwd";
        }
        EXPECT_EQ(fs::file_size(deep_ / "relative.bin"), 8u);
        for (const auto& entry : fs::directory_iterator(fs::path(".")))
            EXPECT_EQ(fs::entry_path(fs::path("."), entry.path()), fs::path(".") / "relative.bin");
    }
    TEST_F(NativeFilesystemTest, LongTemporaryDirectoryDoesNotFallBackToProfile) {
        class EnvironmentGuard {
        public:
            explicit EnvironmentGuard(const wchar_t* key) : key_(key) {
                const auto size = GetEnvironmentVariableW(key_, nullptr, 0);
                if (size) {
                    value_.resize(size);
                    const auto length = GetEnvironmentVariableW(key_, value_.data(), size);
                    value_.resize(length);
                    present_ = true;
                }
            }
            ~EnvironmentGuard() { SetEnvironmentVariableW(key_, present_ ? value_.c_str() : nullptr); }

        private:
            const wchar_t* key_;
            std::wstring value_;
            bool present_ = false;
        } tmp(L"TMP"), temp(L"TEMP");
        ASSERT_TRUE(SetEnvironmentVariableW(L"TMP", deep_.c_str()));
        ASSERT_TRUE(SetEnvironmentVariableW(L"TEMP", deep_.c_str()));
        std::error_code ec;
        const auto actual = fs::temp_directory_path(ec);
        ASSERT_FALSE(ec) << ec.message();
        EXPECT_TRUE(fs::equivalent(actual, deep_));
        EXPECT_FALSE(actual.native().starts_with(L"\\\\?\\"));
    }
    TEST(NativeFilesystemPath, ExtendedUncAndDevicePathsAreIdempotent) {
        const fs::path unc = L"\\\\server\\share\\" + std::wstring(250, L'a');
        EXPECT_TRUE(fs::native_path(unc).native().starts_with(L"\\\\?\\UNC\\server\\share\\"));
        EXPECT_EQ(fs::display_path(fs::native_path(unc)), unc);
        const fs::path device = L"\\\\.\\NUL";
        EXPECT_EQ(fs::native_path(device), device);
        EXPECT_EQ(fs::native_path(fs::native_path(unc)), fs::native_path(unc));
    }
#endif
} // namespace
