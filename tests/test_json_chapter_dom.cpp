/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/json_chapter_dom.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>

namespace {

    using lfs::io::JsonChapterDom;
    using Json = nlohmann::ordered_json;

    constexpr std::string_view UUID_A = "00112233-4455-6677-8899-aabbccddeeff";
    constexpr std::string_view UUID_B = "10213243-5465-7687-98a9-bacbdcedfe0f";
    constexpr std::string_view UUID_C = "abcdef01-2345-6789-abcd-ef0123456789";

    TEST(JsonChapterDomTest, UnknownStructureSurvivesTargetedEdits) {
        constexpr std::string_view input = R"json({
  "chapter_version": 1,
  "unknown_sibling": "keep me",
  "view": {
    "split": {"mode": "single", "future_nested": {"enabled": true}},
    "unknown_view_member": [3, 2, 1]
  },
  "bookmarks": [
    {
      "uuid": "00112233-4455-6677-8899-aabbccddeeff",
      "name": "old",
      "future_element_member": {"vendor": "preserved", "revision": 7}
    }
  ]
})json";

        auto dom = JsonChapterDom::parse(input);
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());
        ASSERT_TRUE(dom->set("view.split.mode", "side_by_side"));

        auto bookmark = dom->array_find("bookmarks", UUID_A);
        ASSERT_TRUE(bookmark);
        ASSERT_TRUE(bookmark->set("name", "edited"));

        const Json dumped = Json::parse(dom->dump());
        EXPECT_EQ(dumped["unknown_sibling"], "keep me");
        EXPECT_EQ(dumped["view"]["split"]["future_nested"]["enabled"], true);
        EXPECT_EQ(dumped["view"]["unknown_view_member"], Json({3, 2, 1}));
        EXPECT_EQ(dumped["bookmarks"][0]["future_element_member"]["vendor"], "preserved");
        EXPECT_EQ(dumped["bookmarks"][0]["future_element_member"]["revision"], 7);
        EXPECT_EQ(dumped["view"]["split"]["mode"], "side_by_side");
        EXPECT_EQ(dumped["bookmarks"][0]["name"], "edited");
    }

    TEST(JsonChapterDomTest, ArrayUpsertAppendsMissingAndEditsExistingInPlace) {
        const std::string input = std::string(R"json({"items":[{"uuid":")json") +
                                  std::string(UUID_A) +
                                  R"json(","name":"first","unknown":11},{"uuid":")json" +
                                  std::string(UUID_B) +
                                  R"json(","name":"second","unknown":22}]})json";
        auto dom = JsonChapterDom::parse(input);
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());

        auto existing = dom->array_upsert("items", UUID_A);
        ASSERT_TRUE(existing) << lfs::format_for_developer(existing.error());
        ASSERT_TRUE(existing->set("name", "first-edited"));

        auto appended = dom->array_upsert("items", UUID_C);
        ASSERT_TRUE(appended) << lfs::format_for_developer(appended.error());
        ASSERT_TRUE(appended->set("name", "third"));

        auto existing_again = dom->array_upsert("items", UUID_A);
        ASSERT_TRUE(existing_again) << lfs::format_for_developer(existing_again.error());

        const Json dumped = Json::parse(dom->dump());
        ASSERT_EQ(dumped["items"].size(), 3);
        EXPECT_EQ(dumped["items"][0]["uuid"], std::string(UUID_A));
        EXPECT_EQ(dumped["items"][0]["name"], "first-edited");
        EXPECT_EQ(dumped["items"][0]["unknown"], 11);
        EXPECT_EQ(dumped["items"][1]["uuid"], std::string(UUID_B));
        EXPECT_EQ(dumped["items"][1]["unknown"], 22);
        EXPECT_EQ(dumped["items"][2]["uuid"], std::string(UUID_C));
        EXPECT_EQ(dumped["items"][2]["name"], "third");
    }

    TEST(JsonChapterDomTest, MalformedAndPathologicalInputReturnTypedErrors) {
        auto malformed = JsonChapterDom::parse(R"json({"view": [1, 2,})json");
        ASSERT_FALSE(malformed);
        EXPECT_EQ(malformed.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(malformed.error().domain(), lfs::ErrorDomain::IO);
        EXPECT_NE(malformed.error().detail().find("byte"), std::string_view::npos);
        EXPECT_NE(malformed.error().detail().find("Malformed JSON chapter"), std::string_view::npos);

        const std::string pathological(2048, '[');
        auto too_deep_and_truncated = JsonChapterDom::parse(pathological);
        ASSERT_FALSE(too_deep_and_truncated);
        EXPECT_EQ(too_deep_and_truncated.error().code(),
                  lfs::ErrorCode::ResourceExhausted);
        EXPECT_EQ(too_deep_and_truncated.error().domain(), lfs::ErrorDomain::IO);

        auto duplicate = JsonChapterDom::parse(
            R"json({"outer":{"same":1,"same":2}})json");
        ASSERT_FALSE(duplicate);
        EXPECT_EQ(duplicate.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_NE(duplicate.error().detail().find("Duplicate JSON object key 'same'"),
                  std::string_view::npos);

        const std::string oversized(64ull * 1024 * 1024 + 1, ' ');
        auto too_large = JsonChapterDom::parse(oversized);
        ASSERT_FALSE(too_large);
        EXPECT_EQ(too_large.error().code(),
                  lfs::ErrorCode::ResourceExhausted);
    }

    TEST(JsonChapterDomTest, FailedSetAcrossNonObjectIsTypedAndTransactional) {
        auto dom = JsonChapterDom::parse(R"json({"view":{"split":3},"unknown":{"keep":true}})json");
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());
        const std::string before = dom->dump();

        auto result = dom->set("view.split.mode", "side_by_side");
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::IO);
        EXPECT_NE(result.error().detail().find("view.split"), std::string_view::npos);
        EXPECT_EQ(dom->dump(), before);
    }

    TEST(JsonChapterDomTest, CanonicalDumpAndChunkBytesAreIdempotent) {
        auto dom = JsonChapterDom::parse(
            R"json({"z":1,"a":{"later":true,"first":false},"items":[3,2,1]})json");
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());

        const std::string first_dump = dom->dump();
        auto reparsed = JsonChapterDom::parse(first_dump);
        ASSERT_TRUE(reparsed) << lfs::format_for_developer(reparsed.error());
        EXPECT_EQ(reparsed->dump(), first_dump);

        const auto bytes = dom->to_bytes();
        auto from_bytes = JsonChapterDom::from_bytes(bytes);
        ASSERT_TRUE(from_bytes) << lfs::format_for_developer(from_bytes.error());
        EXPECT_EQ(from_bytes->dump(), first_dump);
        EXPECT_LT(first_dump.find("\"z\""), first_dump.find("\"a\""));
        EXPECT_NE(first_dump.find("\n  \"z\""), std::string::npos);
    }

} // namespace
