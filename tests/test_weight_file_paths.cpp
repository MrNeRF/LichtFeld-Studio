/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/native_fs.hpp"
#include "core/nn/weight_file.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>
#include <string_view>

namespace {
    class WeightSourceTest : public ::testing::Test {
    protected:
        void SetUp() override {
            root = std::filesystem::temp_directory_path() /
                   ("lfs-weight-source-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            ASSERT_TRUE(lfs::core::native_fs::create_directories(root));
            path = root / lfs::core::utf8_to_path("weights_重み.lfw");
            std::ofstream output;
            ASSERT_TRUE(lfs::core::open_file_for_write(path, std::ios::binary, output));
            constexpr std::string_view json = R"({"tensors":{}})";
            std::array<char, 64> bytes{};
            bytes[0] = 'L';
            bytes[1] = 'F';
            bytes[2] = 'W';
            bytes[3] = '1';
            bytes[4] = static_cast<char>(json.size());
            std::copy(json.begin(), json.end(), bytes.begin() + 8);
            output.write(bytes.data(), bytes.size());
            ASSERT_TRUE(output.good());
        }
        void TearDown() override {
            std::error_code error;
            lfs::core::native_fs::remove_all(root, error);
            EXPECT_FALSE(error) << error.message();
        }
        std::filesystem::path root, path;
    };

    TEST_F(WeightSourceTest, RetainedModelProtectsSourceAfterMappingCloses) {
        using lfs::core::nn::WeightFile;
        std::shared_ptr<const void> model_source;
        {
            auto file = WeightFile::open(path);
            ASSERT_TRUE(file) << file.error().detail();
            model_source = file->retain_source();
        }
        EXPECT_TRUE(WeightFile::source_in_use(path));
        const auto blocked = WeightFile::remove_unused_source(path);
        ASSERT_FALSE(blocked);
        EXPECT_EQ(blocked.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_TRUE(lfs::core::native_fs::exists(path));
        model_source.reset();
        EXPECT_FALSE(WeightFile::source_in_use(path));
        const auto removed = WeightFile::remove_unused_source(path);
        ASSERT_TRUE(removed);
        EXPECT_TRUE(*removed);
        const auto again = WeightFile::remove_unused_source(path);
        ASSERT_TRUE(again);
        EXPECT_FALSE(*again);
    }

    TEST_F(WeightSourceTest, AliasAndIndependentLoadsShareProtection) {
        using lfs::core::nn::WeightFile;
        auto first = WeightFile::open(path);
        ASSERT_TRUE(first);
        const auto alias = root / "alias.lfw";
        lfs::core::native_fs::create_hard_link(path, alias);
        auto second = WeightFile::open(alias);
        ASSERT_TRUE(second);
        auto model_source = second->retain_source();
        *first = WeightFile{};
        *second = WeightFile{};
        EXPECT_TRUE(WeightFile::source_in_use(path));
        EXPECT_FALSE(WeightFile::remove_unused_source(path));
        model_source.reset();
        const auto removed = WeightFile::remove_unused_source(path);
        ASSERT_TRUE(removed);
        EXPECT_TRUE(*removed);
    }

    TEST_F(WeightSourceTest, AcquisitionProtectsHandoffToTheLoader) {
        using lfs::core::nn::WeightFile;
        auto prepared = WeightFile::retain_source(path);
        EXPECT_FALSE(WeightFile::remove_unused_source(path));
        auto file = WeightFile::open(path);
        ASSERT_TRUE(file);
        auto loaded_model = file->retain_source();
        prepared.reset();
        *file = WeightFile{};
        EXPECT_FALSE(WeightFile::remove_unused_source(path));
        loaded_model.reset();
        const auto removed = WeightFile::remove_unused_source(path);
        ASSERT_TRUE(removed);
        EXPECT_TRUE(*removed);
    }

    TEST_F(WeightSourceTest, CacheOperationRemainsExclusiveWithoutAnOpenModel) {
        using lfs::core::nn::WeightFile;
        const auto missing = root / "not-downloaded.lfw";
        const auto operation = WeightFile::source_operation(missing);
        std::lock_guard lock(*operation);
        EXPECT_FALSE(WeightFile::source_in_use(missing));
        auto contender = std::async(std::launch::async, [missing] {
            const auto other = WeightFile::source_operation(missing);
            std::unique_lock attempt(*other, std::try_to_lock);
            return attempt.owns_lock();
        });
        EXPECT_FALSE(contender.get());
    }

    TEST_F(WeightSourceTest, RemovalDoesNotDeleteADirectoryAtTheArchivePath) {
        const auto directory = root / "directory.lfw";
        ASSERT_TRUE(lfs::core::native_fs::create_directories(directory));
        const auto removed = lfs::core::nn::WeightFile::remove_unused_source(directory);
        ASSERT_FALSE(removed);
        EXPECT_EQ(removed.error().code(), lfs::ErrorCode::Unsupported);
        EXPECT_TRUE(lfs::core::native_fs::is_directory(directory));
    }

    TEST_F(WeightSourceTest, LongUnicodePathSharesReservationWithItsNativeSpelling) {
        using lfs::core::nn::WeightFile;
        auto deep = root;
        for (int i = 0; i < 12; ++i)
            deep /= lfs::core::utf8_to_path("weights_重み_component_01234567890123456789");
        ASSERT_GT(deep.native().size(), 400u);
        ASSERT_TRUE(lfs::core::native_fs::create_directories(deep));
        const auto archive = deep / "model.lfw";
        ASSERT_TRUE(lfs::core::native_fs::copy_file(path, archive));
        auto file = WeightFile::open(lfs::core::native_fs::native_path(archive));
        ASSERT_TRUE(file);
        auto model = file->retain_source();
        *file = WeightFile{};
        EXPECT_TRUE(WeightFile::source_in_use(archive));
        EXPECT_FALSE(WeightFile::remove_unused_source(archive));
        model.reset();
        const auto removed = WeightFile::remove_unused_source(archive);
        ASSERT_TRUE(removed);
        EXPECT_TRUE(*removed);
    }
} // namespace

TEST(WeightFileTest, MissingUnicodePathIsReportedAsUtf8) {
    const auto path = std::filesystem::temp_directory_path() /
                      lfs::core::utf8_to_path("missing_weights_重み.lfw");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ASSERT_FALSE(ec) << ec.message();

    const auto result = lfs::core::nn::WeightFile::open(path);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), lfs::ErrorCode::NotFound);
    EXPECT_NE(result.error().detail().find(lfs::core::path_to_utf8(path)),
              std::string_view::npos);
}
