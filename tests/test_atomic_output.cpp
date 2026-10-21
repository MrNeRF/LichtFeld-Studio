/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/native_fs.hpp"
#include "io/atomic_output.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

#ifndef _WIN32
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

    namespace fs = lfs::core::native_fs;

    std::string read_file(const fs::path& path) {
        std::ifstream stream(lfs::core::native_fs::native_path(path), std::ios::binary);
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    void write_file(const fs::path& path, const std::string& contents) {
        std::ofstream stream(lfs::core::native_fs::native_path(path), std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(stream.is_open());
        stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        stream.close();
        ASSERT_TRUE(stream);
    }

    TEST(AtomicOutputPathTest, LongLegalNameCanBeCommitted) {
        const auto directory = fs::temp_directory_path() / "lfs_atomic_long_name";
        fs::create_directories(directory);
        for (const auto style : {lfs::io::AtomicOutputTempName::AppendSuffix,
                                 lfs::io::AtomicOutputTempName::PreserveExtension}) {
            const auto output = directory / (std::string(250, 'x') + ".usd");
            SCOPED_TRACE(static_cast<int>(style));
            write_file(output, "old");
            lfs::io::ScopedAtomicOutputFile replacement(output, style);
            EXPECT_EQ(replacement.temp_path().parent_path(), directory);
            if (style == lfs::io::AtomicOutputTempName::PreserveExtension)
                EXPECT_EQ(replacement.temp_path().extension(), ".usd");
            write_file(replacement.temp_path(), "new");
            ASSERT_TRUE(replacement.commit().has_value());
            EXPECT_EQ(read_file(output), "new");
        }
        std::error_code ec;
        fs::remove_all(directory, ec);
        EXPECT_FALSE(ec);
    }

    TEST(AtomicOutputPathTest, ShortNameKeepsExistingNaming) {
        const auto output = fs::temp_directory_path() / "normal.usd";
        const auto appended = lfs::io::make_atomic_temp_output_path(output);
        const auto preserved = lfs::io::make_atomic_temp_output_path(
            output, lfs::io::AtomicOutputTempName::PreserveExtension);
        EXPECT_TRUE(appended.filename().string().starts_with("normal.usd."));
        EXPECT_TRUE(appended.filename().string().ends_with(".tmp"));
        EXPECT_TRUE(preserved.filename().string().starts_with("normal."));
        EXPECT_TRUE(preserved.filename().string().ends_with(".tmp.usd"));
    }

#ifndef _WIN32
    TEST(CheckpointAtomicReplaceFaultTest, KillAtEveryCommitBoundaryLeavesValidCheckpoint) {
        const auto directory = fs::temp_directory_path() /
                               ("lfs_checkpoint_atomic_replace_" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(directory, ec);
        ASSERT_TRUE(fs::create_directories(directory));

        const auto checkpoint = directory / "checkpoint.resume";
        const std::string old_contents(64 * 1024, 'o');
        const std::string new_contents(64 * 1024, 'n');

        constexpr std::array stages{
            lfs::io::AtomicOutputCommitStage::FileSynced,
            lfs::io::AtomicOutputCommitStage::DestinationReplaced,
            lfs::io::AtomicOutputCommitStage::DirectorySynced};

        for (const auto kill_stage : stages) {
            SCOPED_TRACE(static_cast<int>(kill_stage));
            write_file(checkpoint, old_contents);

            const pid_t child = ::fork();
            ASSERT_GE(child, 0);
            if (child == 0) {
                lfs::io::ScopedAtomicOutputFile output(
                    checkpoint,
                    lfs::io::AtomicOutputTempName::AppendSuffix,
                    lfs::io::AtomicOutputDurability::Durable);
                std::ofstream stream(lfs::core::native_fs::native_path(output.temp_path()), std::ios::binary | std::ios::trunc);
                stream.write(new_contents.data(), static_cast<std::streamsize>(new_contents.size()));
                stream.close();
                if (!stream) {
                    ::_exit(2);
                }
                const auto result = output.commit([kill_stage](const auto stage) {
                    if (stage == kill_stage) {
                        ::raise(SIGKILL);
                    }
                });
                ::_exit(result ? 0 : 3);
            }

            int status = 0;
            ASSERT_EQ(::waitpid(child, &status, 0), child);
            ASSERT_TRUE(WIFSIGNALED(status));
            EXPECT_EQ(WTERMSIG(status), SIGKILL);

            ASSERT_TRUE(fs::exists(checkpoint));
            const auto contents = read_file(checkpoint);
            if (kill_stage == lfs::io::AtomicOutputCommitStage::FileSynced) {
                EXPECT_EQ(contents, old_contents);
            } else {
                EXPECT_EQ(contents, new_contents);
            }

            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.path() != checkpoint) {
                    fs::remove_all(entry.path(), ec);
                    ASSERT_FALSE(ec);
                }
            }
        }

        fs::remove_all(directory, ec);
        EXPECT_FALSE(ec);
    }
#endif

} // namespace
