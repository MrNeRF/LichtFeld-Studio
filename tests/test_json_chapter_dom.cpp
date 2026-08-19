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

    TEST(JsonChapterDomTest, ArrayItemsReturnsOrderedPairsAndTypedErrors) {
        const std::string input = std::string(R"json({"items":[{"uuid":")json") +
                                  std::string(UUID_A) +
                                  R"json(","name":"first"},{"uuid":")json" +
                                  std::string(UUID_B) +
                                  R"json(","name":"second"}]})json";
        auto dom = JsonChapterDom::parse(input);
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());

        auto items = dom->array_items("items");
        ASSERT_TRUE(items) << lfs::format_for_developer(items.error());
        ASSERT_EQ(items->size(), 2);
        EXPECT_EQ((*items)[0].first, std::string(UUID_A));
        EXPECT_EQ((*items)[0].second["name"], "first");
        EXPECT_EQ((*items)[1].first, std::string(UUID_B));
        EXPECT_EQ((*items)[1].second["name"], "second");

        auto missing = dom->array_items("absent");
        ASSERT_TRUE(missing) << lfs::format_for_developer(missing.error());
        EXPECT_TRUE(missing->empty());

        auto duplicate = JsonChapterDom::parse(
            std::string(R"json({"items":[{"uuid":")json") + std::string(UUID_A) +
            R"json("},{"uuid":")json" + std::string(UUID_A) + R"json("}]})json");
        ASSERT_TRUE(duplicate) << lfs::format_for_developer(duplicate.error());
        auto duplicate_items = duplicate->array_items("items");
        ASSERT_FALSE(duplicate_items);
        EXPECT_EQ(duplicate_items.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_NE(duplicate_items.error().detail().find("occurs more than once"),
                  std::string_view::npos);

        auto non_object = JsonChapterDom::parse(R"json({"items":[1]})json");
        ASSERT_TRUE(non_object) << lfs::format_for_developer(non_object.error());
        auto non_object_items = non_object->array_items("items");
        ASSERT_FALSE(non_object_items);
        EXPECT_EQ(non_object_items.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_NE(non_object_items.error().detail().find("expected an object"),
                  std::string_view::npos);
    }

    TEST(JsonChapterDomTest, ArrayGetCopiesPresentElementAndNulloptsAbsence) {
        const std::string input = std::string(R"json({"items":[{"uuid":")json") +
                                  std::string(UUID_A) +
                                  R"json(","name":"first","extra":[1,2]}]})json";
        auto dom = JsonChapterDom::parse(input);
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());

        auto present = dom->array_get("items", UUID_A);
        ASSERT_TRUE(present);
        EXPECT_EQ((*present)["name"], "first");
        EXPECT_EQ((*present)["extra"], Json({1, 2}));

        EXPECT_FALSE(dom->array_get("items", UUID_B));
        EXPECT_FALSE(dom->array_get("missing", UUID_A));
        EXPECT_FALSE(dom->array_get("items", "not-a-canonical-uuid"));
    }

    TEST(JsonChapterDomTest, ArrayPutInsertsAtEndReplacesInPlaceAndRejectsTypedErrors) {
        const std::string input = std::string(R"json({"items":[{"uuid":")json") +
                                  std::string(UUID_A) +
                                  R"json(","name":"first"},{"uuid":")json" +
                                  std::string(UUID_B) +
                                  R"json(","name":"second"}]})json";
        auto dom = JsonChapterDom::parse(input);
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());

        auto inserted = dom->array_put(
            "items", UUID_C, Json{{"uuid", std::string(UUID_C)}, {"name", "third"}});
        ASSERT_TRUE(inserted) << lfs::format_for_developer(inserted.error());

        auto replaced = dom->array_put(
            "items", UUID_B,
            Json{{"uuid", std::string(UUID_B)}, {"name", "second-edited"}, {"keep", 7}});
        ASSERT_TRUE(replaced) << lfs::format_for_developer(replaced.error());

        const Json dumped = Json::parse(dom->dump());
        ASSERT_EQ(dumped["items"].size(), 3);
        EXPECT_EQ(dumped["items"][0]["uuid"], std::string(UUID_A));
        EXPECT_EQ(dumped["items"][1]["uuid"], std::string(UUID_B));
        EXPECT_EQ(dumped["items"][1]["name"], "second-edited");
        EXPECT_EQ(dumped["items"][1]["keep"], 7);
        EXPECT_EQ(dumped["items"][2]["uuid"], std::string(UUID_C));
        EXPECT_EQ(dumped["items"][2]["name"], "third");
        EXPECT_LT(dom->dump().find(std::string(UUID_A)),
                  dom->dump().find(std::string(UUID_B)));
        EXPECT_LT(dom->dump().find(std::string(UUID_B)),
                  dom->dump().find(std::string(UUID_C)));

        auto mismatch = dom->array_put(
            "items", UUID_A, Json{{"uuid", std::string(UUID_B)}, {"name", "wrong"}});
        ASSERT_FALSE(mismatch);
        EXPECT_EQ(mismatch.error().code(), lfs::ErrorCode::InvalidArgument);

        auto non_object = dom->array_put("items", UUID_A, Json("not-an-object"));
        ASSERT_FALSE(non_object);
        EXPECT_EQ(non_object.error().code(), lfs::ErrorCode::InvalidArgument);

        auto non_canonical =
            dom->array_put("items", "Not-Canonical", Json{{"uuid", "Not-Canonical"}});
        ASSERT_FALSE(non_canonical);
        EXPECT_EQ(non_canonical.error().code(), lfs::ErrorCode::InvalidArgument);
        EXPECT_EQ(Json::parse(dom->dump())["items"].size(), 3);
    }

    TEST(JsonChapterDomTest, StaticReadHelpersResolveDetachedElementScalarsAndSubtrees) {
        const Json element{
            {"uuid", std::string(UUID_A)},
            {"name", "detached"},
            {"count", 4},
            {"flag", true},
            {"nested", Json{{"value", 9}, {"label", "inner"}}},
        };

        EXPECT_EQ(JsonChapterDom::read<std::string>(element, "name"), "detached");
        EXPECT_EQ(JsonChapterDom::read<int>(element, "count"), 4);
        EXPECT_EQ(JsonChapterDom::read<bool>(element, "flag"), true);
        EXPECT_EQ(JsonChapterDom::read<int>(element, "nested.value"), 9);
        EXPECT_EQ(JsonChapterDom::read<std::string>(element, "missing"), std::nullopt);
        EXPECT_EQ(JsonChapterDom::read_json(element, "nested"),
                  Json({{"value", 9}, {"label", "inner"}}));
        EXPECT_EQ(JsonChapterDom::read_json(element, "missing"), std::nullopt);
    }

    TEST(JsonChapterDomTest, FailedElementSetAcrossNonObjectIsTypedAndTransactional) {
        const std::string input = std::string(R"json({"items":[{"uuid":")json") +
                                  std::string(UUID_A) +
                                  R"json(","view":{"split":3},"keep":true}]})json";
        auto dom = JsonChapterDom::parse(input);
        ASSERT_TRUE(dom) << lfs::format_for_developer(dom.error());
        const std::string before = dom->dump();

        auto element = dom->array_find("items", UUID_A);
        ASSERT_TRUE(element);
        auto result = element->set("view.split.mode", "side_by_side");
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::IO);
        EXPECT_NE(result.error().detail().find("view.split"), std::string_view::npos);
        EXPECT_EQ(dom->dump(), before);
        EXPECT_EQ(Json::parse(dom->dump())["items"][0]["keep"], true);
    }

} // namespace
