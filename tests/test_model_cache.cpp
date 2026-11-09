/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/native_fs.hpp"
#include "core/nn/weight_file.hpp"
#include "core/nn/weight_source.hpp"
#include "core/path_utils.hpp"
#include "licht_test_support.hpp"
#include "preprocessing/preprocess.hpp"

#include <chrono>
#include <future>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::preprocessing;
    namespace fs = lfs::core::native_fs;

    class ModelCacheTest : public ::testing::Test {
    protected:
        lfs::test::licht::TemporaryDirectory home{"lfs-model-cache"};
        lfs::test::licht::ScopedEnvironmentVariable environment{
#ifdef _WIN32
            "USERPROFILE",
#else
            "HOME",
#endif
            lfs::core::path_to_utf8(home.path)};
        fs::path path;

        void SetUp() override {
            const auto state = inspect_cached_model(CachedModel::Sam2);
            ASSERT_TRUE(state);
            path = state->path;
            // Verify isolation before any write or removal can touch the cache.
            ASSERT_EQ(path.parent_path(), home.path / ".lichtfeld" / "onnx");
            ASSERT_TRUE(fs::create_directories(path.parent_path()));
            std::ofstream output;
            ASSERT_TRUE(lfs::core::open_file_for_write(path, std::ios::binary, output));
            output << "unverified weights";
            ASSERT_TRUE(output.good());
        }
    };

    TEST_F(ModelCacheTest, PresenceDoesNotImplyIntegrityAndVerificationKeepsTheFile) {
        const auto state = inspect_cached_model(CachedModel::Sam2);
        ASSERT_TRUE(state);
        EXPECT_TRUE(state->present);
        EXPECT_GT(state->bytes, 0);
        std::vector<std::pair<std::uintmax_t, std::uintmax_t>> progress;
        const auto verified = prepare_cached_model(CachedModel::Sam2, false, {}, [&](ModelPreparationPhase phase, auto done, auto total) {
            if (phase == ModelPreparationPhase::Verifying)
                progress.emplace_back(done, total);
        });
        EXPECT_FALSE(verified);
        ASSERT_GE(progress.size(), 2u);
        EXPECT_EQ(progress.front().first, 0u);
        EXPECT_EQ(progress.back().first, state->bytes);
        EXPECT_EQ(progress.back().second, state->bytes);
        EXPECT_EQ(fs::file_size(path), state->bytes);
        EXPECT_FALSE(lfs::core::nn::WeightFile::source_in_use(path));
    }

    TEST_F(ModelCacheTest, PreparedConsumerPreventsRemovalUntilReleased) {
        {
            const lfs::core::nn::WeightSource consumer{path};
            const auto state = inspect_cached_model(CachedModel::Sam2);
            ASSERT_TRUE(state);
            EXPECT_TRUE(state->in_use);
            const auto removed = remove_cached_model(CachedModel::Sam2);
            ASSERT_FALSE(removed);
            EXPECT_EQ(removed.error().code(), lfs::ErrorCode::FailedPrecondition);
            EXPECT_TRUE(fs::exists(path));
        }
        const auto removed = remove_cached_model(CachedModel::Sam2);
        ASSERT_TRUE(removed);
        EXPECT_TRUE(*removed);
        EXPECT_FALSE(fs::exists(path));
    }

    TEST_F(ModelCacheTest, DirectoryAtTheWeightPathIsReportedAndPreserved) {
        ASSERT_TRUE(fs::remove(path));
        ASSERT_TRUE(fs::create_directory(path));
        const auto state = inspect_cached_model(CachedModel::Sam2);
        ASSERT_FALSE(state);
        EXPECT_EQ(state.error().code(), lfs::ErrorCode::Unsupported);
        const auto prepared = prepare_cached_model(CachedModel::Sam2, false);
        ASSERT_FALSE(prepared);
        EXPECT_EQ(prepared.error().code(), lfs::ErrorCode::Unsupported);
        EXPECT_TRUE(fs::is_directory(path));
    }

    TEST_F(ModelCacheTest, WaitingCallerCanCancelWithoutReleasingAnotherOperation) {
        const auto operation = lfs::core::nn::WeightFile::source_operation(path);
        std::unique_lock lock(*operation);
        auto waiting = std::async(std::launch::async, [] {
            int checks = 0;
            return prepare_cached_model(CachedModel::Sam2, false, [&] { return ++checks > 2; });
        });
        if (waiting.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            lock.unlock();
            FAIL() << "Waiting model preparation did not honor cancellation";
        }
        const auto canceled = waiting.get();
        ASSERT_FALSE(canceled);
        EXPECT_EQ(canceled.error().code(), lfs::ErrorCode::Cancelled);
        // Observe from another thread, just as UI inspection observes a worker's gate.
        const auto blocked = std::async(std::launch::async, [] {
                                 return std::pair{inspect_cached_model(CachedModel::Sam2), remove_cached_model(CachedModel::Sam2)};
                             }).get();
        const auto& state = blocked.first;
        const auto& removed = blocked.second;
        ASSERT_TRUE(state);
        EXPECT_TRUE(state->busy);
        ASSERT_FALSE(removed);
        EXPECT_EQ(removed.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_TRUE(fs::exists(path));
    }

    TEST_F(ModelCacheTest, CancellationBeforePreparationDoesNotReadOrDownloadAnyModel) {
        for (const auto& model : model_catalog()) {
            const auto result = prepare_cached_model(model.id, true, [] { return true; });
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
        }
        EXPECT_TRUE(fs::exists(path));
    }
} // namespace
