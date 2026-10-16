/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "io/atomic_output.hpp"
#include "long_path_test_support.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

    namespace fs = std::filesystem;
    namespace native_fs = lfs::core::native_fs;
    using namespace lfs::test::long_paths;

    std::string read_stream(const fs::path& path) {
        std::ifstream in(native_fs::native_path(path), std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    std::string read_stdio(const fs::path& path) {
        std::FILE* const file = lfs::core::open_file(path, "rb");
        if (file == nullptr) {
            return "<open failed>";
        }
        std::string text(64, '\0');
        text.resize(std::fread(text.data(), 1, text.size(), file));
        std::fclose(file);
        return text;
    }

    // The format test executable has no long-path manifest, so on Windows any operation that reaches the OS
    // with the plain path fails beyond 259 units (247 for folders), whatever the machine policy says.
    void exercise_file_operations(const fs::path& root) {
        const auto leaf = lfs::core::utf8_to_path("файл 🙂.bin");
        const std::string payload = "native path payload";
        for (const auto length : kLengthsAroundExtendedPathSwitch) {
            SCOPED_TRACE(length);
            const auto file = with_native_length(root / unicode_folder() / std::to_string(length), length, leaf);
            ASSERT_EQ(file.native().size(), length);
            const auto folder = file.parent_path();
            std::error_code error;
            native_fs::create_directories(folder, error);
            ASSERT_FALSE(error) << error.message();
            ASSERT_TRUE(native_fs::is_directory(folder, error)) << error.message();
            {
                std::ofstream out(native_fs::native_path(file), std::ios::binary);
                out << payload;
                ASSERT_TRUE(out.good());
            }
            EXPECT_TRUE(native_fs::exists(file, error));
            EXPECT_TRUE(native_fs::is_regular_file(file, error));
            EXPECT_EQ(native_fs::status(file, error).type(), fs::file_type::regular);
            EXPECT_EQ(native_fs::file_size(file, error), payload.size());
            EXPECT_EQ(read_stream(file), payload);
            EXPECT_EQ(read_stdio(file), payload);

            const auto copy = folder / lfs::core::utf8_to_path("копия 🙂.bin");
            const auto moved = folder / lfs::core::utf8_to_path("перемещено 🙂.bin");
            ASSERT_TRUE(native_fs::copy_file(file, copy, error)) << error.message();
            native_fs::rename(copy, moved, error);
            ASSERT_FALSE(error) << error.message();
            EXPECT_FALSE(native_fs::exists(copy, error));
            native_fs::resize_file(moved, 6, error);
            ASSERT_FALSE(error) << error.message();
            EXPECT_EQ(read_stream(moved), payload.substr(0, 6));
            native_fs::last_write_time(moved, error);
            EXPECT_FALSE(error) << error.message();

            std::vector<std::string> listed;
            for (const auto& entry : fs::directory_iterator(native_fs::native_path(folder), error)) {
                listed.push_back(lfs::core::path_to_utf8(native_fs::display_path(entry.path())));
            }
            ASSERT_FALSE(error) << error.message();
            std::sort(listed.begin(), listed.end());
            std::vector<std::string> expected{lfs::core::path_to_utf8(file), lfs::core::path_to_utf8(moved)};
            std::sort(expected.begin(), expected.end());
            EXPECT_EQ(listed, expected);

            const auto resolved = lfs::core::path_to_utf8(native_fs::weakly_canonical(folder / "missing" / leaf, error));
            ASSERT_FALSE(error) << error.message();
            EXPECT_FALSE(resolved.starts_with(R"(\\?\)")) << resolved;
            EXPECT_TRUE(resolved.ends_with(lfs::core::path_to_utf8(fs::path("missing") / leaf))) << resolved;

            EXPECT_TRUE(native_fs::remove(moved, error)) << error.message();
            EXPECT_EQ(native_fs::remove_all(folder, error), 2u) << error.message();
            EXPECT_FALSE(native_fs::exists(folder, error));
        }
    }

    // Exports publish through a temporary sibling about 30 units longer than the final name, so creating,
    // flushing or moving it fails before the final path itself would reach the limit.
    void exercise_atomic_outputs(const fs::path& root) {
        const auto leaf = lfs::core::utf8_to_path("экспорт 🙂.ply");
        for (const auto length : kLengthsAroundExtendedPathSwitch) {
            SCOPED_TRACE(length);
            const auto output = with_native_length(root / unicode_folder() / std::to_string(length), length, leaf);
            ASSERT_EQ(output.native().size(), length);
            ASSERT_TRUE(lfs::io::ensure_output_parent_directory(output));
            for (const auto durability : {lfs::io::AtomicOutputDurability::Atomic, lfs::io::AtomicOutputDurability::Durable}) {
                const std::string content = durability == lfs::io::AtomicOutputDurability::Atomic ? "first" : "replaced";
                lfs::io::ScopedAtomicOutputFile file(output, lfs::io::AtomicOutputTempName::AppendSuffix, durability);
                {
                    std::ofstream out;
                    ASSERT_TRUE(lfs::core::open_file_for_write(file.temp_path(), std::ios::binary, out));
                    out << content;
                }
                const auto committed = file.commit();
                ASSERT_TRUE(committed) << committed.error().message;
                EXPECT_EQ(read_stream(output), content);
                std::error_code error;
                EXPECT_FALSE(native_fs::exists(file.temp_path(), error));
            }
        }
    }

    // Directory walks append child names to the root they start from, so a short root has to be extended for
    // descendants past the limit to stay reachable while iterating, copying and removing recursively.
    TEST(NativeFs, ShortRootsReachDescendantsPastTheLimit) {
        LongPathDirectory temporary;
        const auto tree = temporary.path / unicode_folder();
        const auto deep = with_native_length(tree, 320, lfs::core::utf8_to_path("tief 🙂.bin"));
        std::error_code error;
        native_fs::create_directories(deep.parent_path(), error);
        ASSERT_FALSE(error) << error.message();
        {
            std::ofstream out(native_fs::native_path(deep), std::ios::binary);
            out << "deep";
            ASSERT_TRUE(out.good());
        }

        std::vector<std::string> found;
        for (std::filesystem::recursive_directory_iterator it(native_fs::traversal_path(temporary.path), error), end;
             !error && it != end; it.increment(error)) {
            if (it->is_regular_file())
                found.push_back(lfs::core::path_to_utf8(native_fs::entry_path(temporary.path, it->path())));
        }
        ASSERT_FALSE(error) << error.message();
        EXPECT_EQ(found, std::vector<std::string>{lfs::core::path_to_utf8(deep)});

        const auto copy = temporary.path / "copy";
        native_fs::copy(tree, copy, fs::copy_options::recursive, error);
        ASSERT_FALSE(error) << error.message();
        EXPECT_EQ(read_stream(copy / deep.lexically_relative(tree)), "deep");

        const auto inside = deep.lexically_relative(tree);
        const auto components = static_cast<std::uintmax_t>(std::distance(inside.begin(), inside.end()));
        EXPECT_EQ(native_fs::remove_all(tree, error), components + 1) << error.message();
        EXPECT_FALSE(native_fs::exists(deep, error));
    }

    // Failures report paths as the caller wrote them; error text never shows the extended spelling.
    TEST(NativeFs, FailuresReportTheCallersPath) {
        LongPathDirectory temporary;
        const auto missing = with_native_length(temporary.path / "missing", 300, fs::path("missing.bin"));
        const auto expect_logical = [&](const fs::filesystem_error& error, const fs::path& first) {
            EXPECT_EQ(lfs::core::path_to_utf8(error.path1()), lfs::core::path_to_utf8(first));
            EXPECT_EQ(std::string(error.what()).find(R"(\\?\)"), std::string::npos) << error.what();
        };
        try {
            (void)native_fs::file_size(missing);
            ADD_FAILURE() << "file_size of a missing file must throw";
        } catch (const fs::filesystem_error& error) {
            expect_logical(error, missing);
        }
        try {
            (void)native_fs::directory_iterator(missing.parent_path());
            ADD_FAILURE() << "iterating a missing folder must throw";
        } catch (const fs::filesystem_error& error) {
            expect_logical(error, missing.parent_path());
        }
        try {
            native_fs::rename(missing, temporary.path / "renamed.bin");
            ADD_FAILURE() << "renaming a missing file must throw";
        } catch (const fs::filesystem_error& error) {
            expect_logical(error, missing);
            EXPECT_EQ(lfs::core::path_to_utf8(error.path2()), lfs::core::path_to_utf8(temporary.path / "renamed.bin"));
        }
    }

    // Queries answer a missing path exactly like the standard: false or not_found, never an exception.
    TEST(NativeFs, MissingPathsAnswerLikeTheStandard) {
        LongPathDirectory temporary;
        const auto missing = with_native_length(temporary.path / "missing", 300, fs::path("missing.bin"));
        EXPECT_NO_THROW({
            EXPECT_FALSE(native_fs::exists(missing));
            EXPECT_FALSE(native_fs::is_directory(missing));
            EXPECT_FALSE(native_fs::is_regular_file(missing));
            EXPECT_EQ(native_fs::status(missing).type(), fs::file_type::not_found);
        });
    }

    // relative() must not hand extended operands to MSVC's own resolution, which fails for missing tails.
    TEST(NativeFs, RelativeResolvesExtendedOperands) {
        LongPathDirectory temporary;
        const auto existing = with_native_length(temporary.path / unicode_folder(), 300, lfs::core::utf8_to_path("da 🙂.bin"));
        std::error_code error;
        native_fs::create_directories(existing.parent_path(), error);
        ASSERT_FALSE(error) << error.message();
        {
            std::ofstream out(native_fs::native_path(existing), std::ios::binary);
            out << "x";
        }
        const auto missing = existing.parent_path() / "fehlt" / "noch.bin";
        for (const auto& target : {existing, missing}) {
            const auto relative = native_fs::relative(native_fs::native_path(target), native_fs::traversal_path(temporary.path), error);
            ASSERT_FALSE(error) << error.message();
            EXPECT_EQ(lfs::core::path_to_utf8(relative), lfs::core::path_to_utf8(target.lexically_relative(temporary.path)));
        }
    }

    TEST(NativeFs, EntryPathsKeepTheCallersSpelling) {
        for (const fs::path root : {fs::path("relative") / "folder", fs::temp_directory_path() / "folder"}) {
            const auto entry = native_fs::traversal_path(root) / "sub" / lfs::core::utf8_to_path("bild 🙂.png");
            EXPECT_EQ(lfs::core::path_to_utf8(native_fs::entry_path(root, entry)),
                      lfs::core::path_to_utf8(root / "sub" / lfs::core::utf8_to_path("bild 🙂.png")));
        }
    }

    TEST(NativeFs, FileOperationsWorkOnBothSidesOfTheExtendedPathSwitch) {
        LongPathDirectory temporary;
        exercise_file_operations(temporary.path);
    }

    TEST(NativeFs, AtomicOutputsPublishOnBothSidesOfTheExtendedPathSwitch) {
        LongPathDirectory temporary;
        exercise_atomic_outputs(temporary.path);
    }

#ifdef _WIN32
    TEST(NativeFs, FileOperationsAndAtomicOutputsWorkThroughANetworkShare) {
        const auto share = network_share();
        if (!share) {
            if (network_share_required()) {
                FAIL() << "LFS_TEST_UNC_ROOT and LFS_TEST_UNC_LOCAL must name a reachable network share";
            }
            GTEST_SKIP() << "No network share provisioned (LFS_TEST_UNC_ROOT, LFS_TEST_UNC_LOCAL)";
        }
        LongPathDirectory run(share->local);
        const auto network = share->network / run.path.filename();
        exercise_file_operations(network);
        exercise_atomic_outputs(network);
    }

    // Proves that the helpers, not the machine, make long paths work: this executable has no long-path manifest,
    // so Windows must reject the plain spelling of a path past the limit while the native spelling succeeds.
    TEST(NativeFs, PlainSpellingOfAnOverLimitPathFailsInThisProcess) {
        LongPathDirectory temporary;
        const auto file = with_native_length(temporary.path / unicode_folder(), 300, lfs::core::utf8_to_path("probe.bin"));
        std::error_code error;
        native_fs::create_directories(file.parent_path(), error);
        ASSERT_FALSE(error) << error.message();
        const auto open = [](const fs::path& path) {
            const HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                                              nullptr);
            const bool opened = handle != INVALID_HANDLE_VALUE;
            if (opened)
                CloseHandle(handle);
            return opened;
        };
        EXPECT_FALSE(open(file)) << "This process accepts plain long paths, so passing tests here prove nothing";
        EXPECT_TRUE(open(native_fs::native_path(file)));
    }

    TEST(NativeFs, NativeAndDisplayFormsRoundTrip) {
        const auto long_tail = fs::path(std::wstring(150, L'a')) / std::wstring(120, L'b') / L"scene.licht";
        for (const fs::path root : {fs::path(L"C:\\projects"), fs::path(L"\\\\server\\share")}) {
            const auto plain = root / long_tail;
            const auto native = native_fs::native_path(plain);
            EXPECT_TRUE(native.native().starts_with(root.native().starts_with(L"\\\\") ? L"\\\\?\\UNC\\server\\share\\"
                                                                                       : L"\\\\?\\C:\\"));
            EXPECT_EQ(native_fs::native_path(native), native);
            EXPECT_EQ(native_fs::display_path(native), plain);
            EXPECT_EQ(native_fs::native_path(root / L"short.licht"), root / L"short.licht");
        }
        const fs::path device = L"\\\\.\\pipe\\lfs";
        EXPECT_EQ(native_fs::native_path(device), device);
    }
#endif

} // namespace
