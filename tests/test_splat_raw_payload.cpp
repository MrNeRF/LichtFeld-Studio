/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "io/formats/ply.hpp"
#include "io/project_chapters.hpp"
#include "io/splat_chapter.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "licht_test_support.hpp"

namespace {

    using lfs::io::project::SplatChapterPayload;

    struct ShValueQuantOverrideGuard {
        ShValueQuantOverrideGuard() {
            lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);
        }

        ~ShValueQuantOverrideGuard() {
            lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(
                std::nullopt);
        }
    };

    void put_u32(std::vector<std::byte>& bytes, const std::size_t at,
                 const std::uint32_t value) {
        for (std::size_t i = 0; i < 4; ++i)
            bytes[at + i] = static_cast<std::byte>(value >> (8 * i));
    }

    void put_u64(std::vector<std::byte>& bytes, const std::size_t at,
                 const std::uint64_t value) {
        for (std::size_t i = 0; i < 8; ++i)
            bytes[at + i] = static_cast<std::byte>(value >> (8 * i));
    }

    std::uint64_t get_u64(const std::vector<std::byte>& bytes, const std::size_t at) {
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(
                         std::to_integer<std::uint8_t>(bytes[at + i]))
                     << (8 * i);
        return value;
    }

    class SplatRawPayloadTest : public ::testing::Test {
    protected:
        static std::vector<std::byte> payload;

        static void SetUpTestSuite() {
            const auto path = std::filesystem::path(PROJECT_ROOT_PATH) /
                              "tests/data/bike.ply";
            if (!std::filesystem::exists(path)) {
                return;
            }
            auto loaded = lfs::io::load_ply(path);
            ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
            auto captured = SplatChapterPayload::capture(
                loaded->value, lfs::io::project::SplatSourceKind::ImportedPly,
                false);
            ASSERT_TRUE(captured.has_value()) << lfs::format_for_developer(captured.error());
            payload.assign(captured->bytes().begin(), captured->bytes().end());
        }

        void SetUp() override {
            if (payload.empty()) {
                GTEST_SKIP() << "tests/data/bike.ply is not available";
            }
        }

        static void expect_invalid(std::vector<std::byte> bytes) {
            auto parsed = SplatChapterPayload::from_lfsp(bytes);
            ASSERT_TRUE(parsed.has_value()) << lfs::format_for_developer(parsed.error());
            auto hydrated = parsed->hydrate();
            EXPECT_FALSE(hydrated.has_value());

            std::string storage(reinterpret_cast<const char*>(bytes.data()),
                                bytes.size());
            std::istringstream stream(storage, std::ios::binary);
            auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
                stream, bytes.size());
            EXPECT_FALSE(streamed.has_value());
            if (hydrated.has_value() == false && streamed.has_value() == false) {
                EXPECT_EQ(hydrated.error().code(), streamed.error().code());
            }
        }

        static std::istringstream stream_of(const std::span<const std::byte> bytes) {
            std::string storage(reinterpret_cast<const char*>(bytes.data()),
                                bytes.size());
            return std::istringstream(std::move(storage), std::ios::binary);
        }
    };

    std::vector<std::byte> SplatRawPayloadTest::payload;

    TEST_F(SplatRawPayloadTest, BadTensorCount) {
        auto bytes = payload;
        put_u32(bytes, 24, 9);
        expect_invalid(std::move(bytes));
    }

    TEST_F(SplatRawPayloadTest, OverlappingAndOutOfRangeOffsets) {
        auto overlap = payload;
        const auto first_offset = get_u64(overlap, 40 + 40);
        put_u64(overlap, 40 + 64 + 40, first_offset);
        expect_invalid(std::move(overlap));

        auto out_of_range = payload;
        put_u64(out_of_range, 40 + 40,
                static_cast<std::uint64_t>(out_of_range.size()) + 1);
        expect_invalid(std::move(out_of_range));
    }

    TEST_F(SplatRawPayloadTest, DtypeAndShapeMismatch) {
        auto dtype = payload;
        dtype[40 + 4] = std::byte{4};
        expect_invalid(std::move(dtype));

        auto shape = payload;
        put_u64(shape, 40 + 8, get_u64(shape, 40 + 8) + 1);
        expect_invalid(std::move(shape));
    }

    TEST_F(SplatRawPayloadTest, TotalSizeAndTruncatedTensor) {
        auto extra = payload;
        extra.push_back(std::byte{0});
        expect_invalid(std::move(extra));

        auto truncated = payload;
        const auto descriptor = 40 + 5 * 64;
        const auto length = get_u64(truncated, descriptor + 48);
        ASSERT_GT(length, 0u);
        truncated.resize(truncated.size() - 1);
        put_u64(truncated, descriptor + 48, length - 1);
        expect_invalid(std::move(truncated));
    }

    TEST_F(SplatRawPayloadTest, PlyRoundTripPreservesTensorBytes) {
        const auto path = std::filesystem::path(PROJECT_ROOT_PATH) /
                          "tests/data/bike.ply";
        auto loaded = lfs::io::load_ply(path);
        ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
        auto captured = SplatChapterPayload::capture(
            loaded->value, lfs::io::project::SplatSourceKind::ImportedPly, false);
        ASSERT_TRUE(captured.has_value()) << lfs::format_for_developer(captured.error());
        auto parsed = SplatChapterPayload::from_lfsp(
            std::vector<std::byte>(captured->bytes().begin(), captured->bytes().end()));
        ASSERT_TRUE(parsed.has_value()) << lfs::format_for_developer(parsed.error());
        auto hydrated = parsed->hydrate();
        ASSERT_TRUE(hydrated.has_value()) << lfs::format_for_developer(hydrated.error());

        const auto equal = [](const lfs::core::Tensor& lhs,
                              const lfs::core::Tensor& rhs) {
            const auto left = lhs.contiguous().cpu();
            const auto right = rhs.contiguous().cpu();
            return left.shape() == right.shape() && left.dtype() == right.dtype() &&
                   left.bytes() == right.bytes() &&
                   std::memcmp(left.data_ptr(), right.data_ptr(), left.bytes()) == 0;
        };
        EXPECT_TRUE(equal(loaded->value.means(), (*hydrated)->means()));
        EXPECT_TRUE(equal(loaded->value.sh0(), (*hydrated)->sh0()));
        EXPECT_TRUE(equal(loaded->value.shN_canonical(), (*hydrated)->shN_canonical()));
        EXPECT_TRUE(equal(loaded->value.scaling_raw(), (*hydrated)->scaling_raw()));
        EXPECT_TRUE(equal(loaded->value.rotation_raw(), (*hydrated)->rotation_raw()));
        EXPECT_TRUE(equal(loaded->value.opacity_raw(), (*hydrated)->opacity_raw()));
    }

    TEST_F(SplatRawPayloadTest, CaptureAfterSelectionOnViewerQ16Model) {
        ShValueQuantOverrideGuard quant_override;
        const auto path = std::filesystem::path(PROJECT_ROOT_PATH) /
                          "tests/data/bike.ply";
        auto loaded = lfs::io::load_ply(path);
        ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
        auto& model = loaded->value;

        ASSERT_TRUE(lfs::training::sh_value::apply_shN_value_quant(model));
        ASSERT_TRUE(model.shN_value_quantized());

        const auto count = static_cast<std::size_t>(model.size());
        ASSERT_GT(count, 0u);
        auto selection = lfs::core::Tensor::zeros(
            {count}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        selection.slice(0, 0, std::min<std::size_t>(count, 64)).fill_(1.0f);
        model.soft_delete(selection);
        ASSERT_TRUE(model.deleted().is_valid());

        auto captured = SplatChapterPayload::capture(
            model, lfs::io::project::SplatSourceKind::Generated, false);
        ASSERT_TRUE(captured.has_value()) << lfs::format_for_developer(captured.error());
        auto hydrated = captured->hydrate();
        ASSERT_TRUE(hydrated.has_value()) << lfs::format_for_developer(hydrated.error());

        EXPECT_EQ((*hydrated)->size(), model.size());
        EXPECT_EQ((*hydrated)->means().shape(), model.means().shape());
        ASSERT_TRUE((*hydrated)->deleted().is_valid());
        const auto deleted = (*hydrated)->deleted().cpu().to_vector_bool();
        EXPECT_TRUE(std::ranges::any_of(deleted, [](const bool value) {
            return value;
        }));
    }

    TEST_F(SplatRawPayloadTest, StreamHydrateMatchesRawBytesAndContentHash) {
        auto stream = stream_of(payload);
        auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
            stream, payload.size());
        ASSERT_TRUE(streamed.has_value())
            << lfs::format_for_developer(streamed.error());
        EXPECT_EQ(streamed->content_xxh3_128,
                  lfs::io::project::xxh3_128(payload));

        auto parsed = SplatChapterPayload::from_lfsp(payload);
        ASSERT_TRUE(parsed.has_value()) << lfs::format_for_developer(parsed.error());
        auto hydrated = parsed->hydrate();
        ASSERT_TRUE(hydrated.has_value())
            << lfs::format_for_developer(hydrated.error());

        const auto equal = [](const lfs::core::Tensor& lhs,
                              const lfs::core::Tensor& rhs) {
            const auto left = lhs.contiguous().cpu();
            const auto right = rhs.contiguous().cpu();
            return left.shape() == right.shape() && left.dtype() == right.dtype() &&
                   left.bytes() == right.bytes() &&
                   std::memcmp(left.data_ptr(), right.data_ptr(), left.bytes()) == 0;
        };
        EXPECT_EQ((*hydrated)->get_active_sh_degree(),
                  streamed->splat->get_active_sh_degree());
        EXPECT_EQ((*hydrated)->get_max_sh_degree(),
                  streamed->splat->get_max_sh_degree());
        EXPECT_EQ((*hydrated)->get_scene_scale(),
                  streamed->splat->get_scene_scale());
        EXPECT_EQ((*hydrated)->frozen_ranges().size(),
                  streamed->splat->frozen_ranges().size());
        EXPECT_TRUE(equal((*hydrated)->means(), streamed->splat->means()));
        EXPECT_TRUE(equal((*hydrated)->scaling_raw(),
                          streamed->splat->scaling_raw()));
        EXPECT_TRUE(equal((*hydrated)->rotation_raw(),
                          streamed->splat->rotation_raw()));
        EXPECT_TRUE(equal((*hydrated)->opacity_raw(),
                          streamed->splat->opacity_raw()));
        EXPECT_TRUE(equal((*hydrated)->sh0(), streamed->splat->sh0()));
        EXPECT_TRUE(equal((*hydrated)->shN(), streamed->splat->shN()));
    }

    TEST_F(SplatRawPayloadTest, StreamHydrateProgressIsMonotonicAcrossWindows) {
        struct WindowGuard {
            explicit WindowGuard(const std::size_t bytes) {
                lfs::io::project::detail::set_splat_stream_window_bytes_for_testing(
                    bytes);
            }
            WindowGuard(const WindowGuard&) = delete;
            WindowGuard& operator=(const WindowGuard&) = delete;
            ~WindowGuard() {
                lfs::io::project::detail::set_splat_stream_window_bytes_for_testing(
                    std::nullopt);
            }
        };
        const WindowGuard window(128);
        std::vector<std::pair<std::size_t, std::size_t>> reports;
        auto stream = stream_of(payload);
        auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
            stream, payload.size(), {},
            [&](const std::size_t consumed, const std::size_t total) {
                reports.emplace_back(consumed, total);
            });
        ASSERT_TRUE(streamed.has_value())
            << lfs::format_for_developer(streamed.error());
        ASSERT_GE(reports.size(), 2u);
        EXPECT_EQ(reports.back().first, reports.back().second);
        EXPECT_EQ(reports.back().first, payload.size());
        for (std::size_t i = 0; i < reports.size(); ++i) {
            EXPECT_EQ(reports[i].second, payload.size());
            if (i > 0) {
                EXPECT_LE(reports[i - 1].first, reports[i].first);
            }
        }
    }

    TEST_F(SplatRawPayloadTest, StreamHydrateRejectsSameCorruptionsAsHydrateRaw) {
        auto bad_magic = payload;
        bad_magic[0] = std::byte{
            static_cast<unsigned char>(~std::to_integer<unsigned char>(bad_magic[0]))};
        {
            std::string storage(reinterpret_cast<const char*>(bad_magic.data()),
                                bad_magic.size());
            std::istringstream stream(storage, std::ios::binary);
            auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
                stream, bad_magic.size());
            EXPECT_FALSE(streamed.has_value());
            if (!streamed) {
                EXPECT_EQ(streamed.error().code(), lfs::ErrorCode::DataLoss);
            }
        }

        auto overlap = payload;
        const auto first_offset = get_u64(overlap, 40 + 40);
        put_u64(overlap, 40 + 64 + 40, first_offset);
        expect_invalid(std::move(overlap));

        auto truncated = payload;
        ASSERT_GT(truncated.size(), 8u);
        truncated.resize(truncated.size() - 8);
        expect_invalid(std::move(truncated));
    }

    std::vector<std::byte> captured_raw_payload() {
        auto model = lfs::test::licht::make_splat(24);
        model->set_frozen_ranges({{2, 4}, {10, 3}});
        auto captured = SplatChapterPayload::capture(
            *model, lfs::io::project::SplatSourceKind::ImportedPly, false);
        if (!captured) {
            throw std::runtime_error(lfs::format_for_developer(captured.error()));
        }
        return {captured->bytes().begin(), captured->bytes().end()};
    }

    std::istringstream bytes_to_stream(const std::vector<std::byte>& bytes) {
        std::string storage(reinterpret_cast<const char*>(bytes.data()),
                            bytes.size());
        return std::istringstream(std::move(storage), std::ios::binary);
    }

    TEST(SplatStreamHydrateTest, MatchesMaterializedTensorsAndXxh3) {
        const auto bytes = captured_raw_payload();
        auto parsed = SplatChapterPayload::from_lfsp(bytes);
        ASSERT_TRUE(parsed.has_value()) << lfs::format_for_developer(parsed.error());
        auto hydrated = parsed->hydrate();
        ASSERT_TRUE(hydrated.has_value())
            << lfs::format_for_developer(hydrated.error());

        auto stream = bytes_to_stream(bytes);
        auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
            stream, bytes.size());
        ASSERT_TRUE(streamed.has_value())
            << lfs::format_for_developer(streamed.error());
        EXPECT_EQ(streamed->content_xxh3_128,
                  lfs::io::project::xxh3_128(bytes));

        const auto equal = [](const lfs::core::Tensor& lhs,
                              const lfs::core::Tensor& rhs) {
            const auto left = lhs.contiguous().cpu();
            const auto right = rhs.contiguous().cpu();
            return left.shape() == right.shape() && left.dtype() == right.dtype() &&
                   left.bytes() == right.bytes() &&
                   std::memcmp(left.data_ptr(), right.data_ptr(), left.bytes()) == 0;
        };
        EXPECT_EQ((*hydrated)->get_active_sh_degree(),
                  streamed->splat->get_active_sh_degree());
        EXPECT_EQ((*hydrated)->get_max_sh_degree(),
                  streamed->splat->get_max_sh_degree());
        EXPECT_EQ((*hydrated)->get_scene_scale(),
                  streamed->splat->get_scene_scale());
        ASSERT_EQ((*hydrated)->frozen_ranges().size(),
                  streamed->splat->frozen_ranges().size());
        for (std::size_t i = 0; i < (*hydrated)->frozen_ranges().size(); ++i) {
            EXPECT_EQ((*hydrated)->frozen_ranges()[i].start,
                      streamed->splat->frozen_ranges()[i].start);
            EXPECT_EQ((*hydrated)->frozen_ranges()[i].count,
                      streamed->splat->frozen_ranges()[i].count);
        }
        EXPECT_TRUE(equal((*hydrated)->means(), streamed->splat->means()));
        EXPECT_TRUE(equal((*hydrated)->scaling_raw(),
                          streamed->splat->scaling_raw()));
        EXPECT_TRUE(equal((*hydrated)->rotation_raw(),
                          streamed->splat->rotation_raw()));
        EXPECT_TRUE(equal((*hydrated)->opacity_raw(),
                          streamed->splat->opacity_raw()));
        EXPECT_TRUE(equal((*hydrated)->sh0(), streamed->splat->sh0()));
        EXPECT_TRUE(equal((*hydrated)->shN(), streamed->splat->shN()));
    }

    TEST(SplatStreamHydrateTest, ProgressIsMonotonicAcrossWindows) {
        struct WindowGuard {
            explicit WindowGuard(const std::size_t bytes) {
                lfs::io::project::detail::set_splat_stream_window_bytes_for_testing(
                    bytes);
            }
            WindowGuard(const WindowGuard&) = delete;
            WindowGuard& operator=(const WindowGuard&) = delete;
            ~WindowGuard() {
                lfs::io::project::detail::set_splat_stream_window_bytes_for_testing(
                    std::nullopt);
            }
        };
        const auto bytes = captured_raw_payload();
        const WindowGuard window(64);
        std::vector<std::pair<std::size_t, std::size_t>> reports;
        auto stream = bytes_to_stream(bytes);
        auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
            stream, bytes.size(), {},
            [&](const std::size_t consumed, const std::size_t total) {
                reports.emplace_back(consumed, total);
            });
        ASSERT_TRUE(streamed.has_value())
            << lfs::format_for_developer(streamed.error());
        ASSERT_GE(reports.size(), 2u);
        EXPECT_EQ(reports.back().first, reports.back().second);
        EXPECT_EQ(reports.back().first, bytes.size());
        for (std::size_t i = 0; i < reports.size(); ++i) {
            EXPECT_EQ(reports[i].second, bytes.size());
            if (i > 0) {
                EXPECT_LE(reports[i - 1].first, reports[i].first);
            }
        }
    }

    TEST(SplatStreamHydrateTest, RejectsSameCorruptionsAsHydrateRaw) {
        const auto bytes = captured_raw_payload();
        const auto expect_both_reject = [](std::vector<std::byte> corrupted) {
            auto parsed = SplatChapterPayload::from_lfsp(corrupted);
            ASSERT_TRUE(parsed.has_value())
                << lfs::format_for_developer(parsed.error());
            auto hydrated = parsed->hydrate();
            EXPECT_FALSE(hydrated.has_value());
            auto stream = bytes_to_stream(corrupted);
            auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
                stream, corrupted.size());
            EXPECT_FALSE(streamed.has_value());
            if (!hydrated && !streamed) {
                EXPECT_EQ(hydrated.error().code(), streamed.error().code());
            }
        };

        auto bad_magic = bytes;
        bad_magic[0] = std::byte{static_cast<unsigned char>(
            ~std::to_integer<unsigned char>(bad_magic[0]))};
        {
            auto stream = bytes_to_stream(bad_magic);
            auto streamed = SplatChapterPayload::hydrate_lfsp_stream(
                stream, bad_magic.size());
            EXPECT_FALSE(streamed.has_value());
            if (!streamed) {
                EXPECT_EQ(streamed.error().code(), lfs::ErrorCode::DataLoss);
            }
        }

        auto overlap = bytes;
        const auto first_offset = get_u64(overlap, 40 + 40);
        put_u64(overlap, 40 + 64 + 40, first_offset);
        expect_both_reject(std::move(overlap));

        auto truncated = bytes;
        ASSERT_GT(truncated.size(), 8u);
        truncated.resize(truncated.size() - 8);
        expect_both_reject(std::move(truncated));
    }

} // namespace
