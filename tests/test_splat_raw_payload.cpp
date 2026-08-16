/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "io/formats/ply.hpp"
#include "io/splat_chapter.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"

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
            auto loaded = lfs::io::load_ply(path);
            ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
            auto captured = SplatChapterPayload::capture(
                loaded->value, lfs::io::project::SplatSourceKind::ImportedPly,
                false);
            ASSERT_TRUE(captured.has_value()) << lfs::format_for_developer(captured.error());
            payload.assign(captured->bytes().begin(), captured->bytes().end());
        }

        static void expect_invalid(std::vector<std::byte> bytes) {
            auto parsed = SplatChapterPayload::from_lfsp(std::move(bytes));
            ASSERT_TRUE(parsed.has_value()) << lfs::format_for_developer(parsed.error());
            auto hydrated = parsed->hydrate();
            EXPECT_FALSE(hydrated.has_value());
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

} // namespace
