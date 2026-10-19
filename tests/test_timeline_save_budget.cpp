/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/path_utils.hpp"
#include "sequencer/timeline.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
    using lfs::sequencer::Timeline;
    constexpr size_t JSON_FILE_BUDGET = 16ULL * 1024ULL * 1024ULL;

    // Keep the timeline valid for the in-memory reader; only its serialized byte
    // count reaches the file budget. Equal-time camera keys are supported.
    nlohmann::json timelineJsonWithSize(const size_t bytes) {
        Timeline source;
        source.ensureAnimationClip().setName("");
        lfs::sequencer::Keyframe key;
        source.addKeyframe(key);
        auto json = source.saveToJson();
        const auto first_size = json.dump(2).size();
        const auto saved_key = json["keyframes"][0];
        json["keyframes"].push_back(saved_key);
        const auto key_bytes = json.dump(2).size() - first_size;
        const auto key_count = 1 + (bytes - first_size) / key_bytes;
        json["keyframes"] = std::vector<nlohmann::json>(key_count, saved_key);
        json["animation_clip"]["name"] = std::string(bytes - (first_size + (key_count - 1) * key_bytes), 'x');
        return json;
    }

    std::string readBytes(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    class TimelineSaveBudgetTest : public ::testing::Test {
    protected:
        std::filesystem::path directory = std::filesystem::temp_directory_path() /
                                          ("timeline-save-budget-" + std::to_string(
                                                                         std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::path path = directory / std::filesystem::path(u8"camera-\u00e8-\u65e5.json");

        void SetUp() override { ASSERT_TRUE(std::filesystem::create_directory(directory)); }
        void TearDown() override {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
        }
    };

    TEST_F(TimelineSaveBudgetTest, ExactReadBudgetSavesAndLoads) {
        const auto json = timelineJsonWithSize(JSON_FILE_BUDGET);
        ASSERT_EQ(json.dump(2).size(), JSON_FILE_BUDGET);
        Timeline source;
        ASSERT_TRUE(source.loadFromJson(json));
        ASSERT_TRUE(source.saveToJson(lfs::core::path_to_utf8(path)));
        EXPECT_EQ(std::filesystem::file_size(path), JSON_FILE_BUDGET);
        Timeline restored;
        ASSERT_TRUE(restored.loadFromJson(lfs::core::path_to_utf8(path)));
        EXPECT_EQ(restored.saveToJson(), source.saveToJson());
    }

    TEST_F(TimelineSaveBudgetTest, OneByteOverBudgetPreservesExistingFile) {
        Timeline original;
        original.addKeyframe(lfs::sequencer::Keyframe{});
        ASSERT_TRUE(original.saveToJson(lfs::core::path_to_utf8(path)));
        const auto original_bytes = readBytes(path);
        ASSERT_FALSE(original_bytes.empty());
        const auto json = timelineJsonWithSize(JSON_FILE_BUDGET + 1);
        ASSERT_EQ(json.dump(2).size(), JSON_FILE_BUDGET + 1);
        Timeline source;
        ASSERT_TRUE(source.loadFromJson(json));
        EXPECT_FALSE(source.saveToJson(lfs::core::path_to_utf8(path)));
        EXPECT_TRUE(readBytes(path) == original_bytes);
        Timeline restored;
        ASSERT_TRUE(restored.loadFromJson(lfs::core::path_to_utf8(path)));
        EXPECT_EQ(restored.saveToJson(), original.saveToJson());
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory),
                                std::filesystem::directory_iterator()),
                  1);
    }

    TEST_F(TimelineSaveBudgetTest, OversizedSaveDoesNotCreateOutputDirectories) {
        Timeline source;
        ASSERT_TRUE(source.loadFromJson(timelineJsonWithSize(JSON_FILE_BUDGET + 1)));
        const auto parent = directory / "new-directory";
        EXPECT_FALSE(source.saveToJson(lfs::core::path_to_utf8(parent / "timeline.json")));
        EXPECT_FALSE(std::filesystem::exists(parent));
    }

    TEST_F(TimelineSaveBudgetTest, SupportedSaveStillReplacesExistingFile) {
        Timeline source;
        source.addKeyframe(lfs::sequencer::Keyframe{});
        ASSERT_TRUE(source.saveToJson(lfs::core::path_to_utf8(path)));
        const auto original_bytes = readBytes(path);
        ASSERT_FALSE(original_bytes.empty());
        source.setClipDuration(60.0f);
        ASSERT_TRUE(source.saveToJson(lfs::core::path_to_utf8(path)));
        EXPECT_NE(readBytes(path), original_bytes);
        Timeline restored;
        ASSERT_TRUE(restored.loadFromJson(lfs::core::path_to_utf8(path)));
        EXPECT_EQ(restored.saveToJson(), source.saveToJson());
    }
} // namespace
