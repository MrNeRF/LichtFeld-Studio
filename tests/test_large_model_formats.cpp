/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The export formats' limits, at real sizes: past its limit each export must fail with an error and leave no file,
// and up to it the file must hold every row.
// - SOG (and the HTML export, which embeds one SOG): its reader refuses more than 100M splats, so the export refuses
//   them before writing; a model past the limit whose visible splats are below it still exports. (SSOG writes its
//   LOD tree's chunks as SOGs of their own, each far below the limit.)
// - SPZ: at most 30M visible splats.
// - NuRec USDZ: msgpack stores each tensor in one binary of at most 2^32 - 1 bytes, so the export refuses a model
//   whose specular SH would pass it (47.7M visible splats at SH degree 3); below that, a payload past 4 GiB (36.4M
//   splats at SH degree 3) must reach zlib and libarchive in pieces and load back row for row.
// RAD's 32-bit count is not tested: it needs 4.29G splats.
//
// The files go to a fresh folder under std::filesystem::temp_directory_path() (TMPDIR), removed afterwards.
// Tiers by device memory: see large_model_test_support.hpp. Each test's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "io/exporter.hpp"
#include "io/loaders/usd_loader.hpp"
#include "lfs/training/sh_value_codec.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace io = lfs::io;

    constexpr size_t kSogLimit = 100'000'000;
    constexpr size_t kSpzLimit = 30'000'000;
    constexpr size_t kMsgpackLimitSh3 = 47'721'859;   // 90 bytes of specular SH a splat pass 2^32 - 1 bytes
    static_assert(kMsgpackLimitSh3 * 90 > (size_t{1} << 32) - 1 && (kMsgpackLimitSh3 - 1) * 90 <= (size_t{1} << 32) - 1);

    // Means, sh0, scaling, rotation and opacity: 14 floats a splat.
    size_t attribute_bytes(const size_t rows) { return rows * 14 * sizeof(float); }

    class LargeModelFormats : public ::testing::Test {
    protected:
        void SetUp() override {
            static std::atomic<int> counter{0};
            folder_ = fs::temp_directory_path() /
                      std::format("lfs_large_model_formats_{}_{}", std::chrono::steady_clock::now().time_since_epoch().count(),
                                  counter++);
            fs::create_directories(folder_);
        }

        void TearDown() override {
            std::error_code ignored;
            fs::remove_all(folder_, ignored);
        }

        // A model of `rows` zeroed splats at SH degree 0 on `device`.
        static SplatData sh0_model(const size_t rows, const Device device) {
            return SplatData(0, Tensor::zeros({rows, 3}, device), Tensor::zeros({rows, 1, 3}, device), Tensor{},
                             Tensor::zeros({rows, 3}, device), Tensor::full({rows, 4}, 0.5f, device),
                             Tensor::zeros({rows, 1}, device), 1.0f);
        }

        // Nothing at the path, and nothing else in the folder either (no temporary left behind).
        void expect_no_file(const fs::path& path) const {
            EXPECT_FALSE(fs::exists(path)) << path.filename() << " was written";
            size_t entries = 0;
            for ([[maybe_unused]] const auto& entry : fs::directory_iterator(folder_))
                ++entries;
            EXPECT_EQ(entries, 0u) << "the export left files in its folder";
        }

        fs::path folder_;
    };

#define LFS_LARGE_MODEL_NEEDS(device_bytes, host_bytes)                    \
    do {                                                                   \
        if (const auto reason = lm::skip_reason(device_bytes, host_bytes)) \
            GTEST_SKIP() << *reason;                                       \
    } while (false)

    // SOG past the reader's 100M: INVALID_DATASET before anything is written.
    TEST_F(LargeModelFormats, SogRefusesPastTheReaderLimit) {
        LFS_LARGE_MODEL_NEEDS(attribute_bytes(kSogLimit + 1), 0);
        const SplatData model = sh0_model(kSogLimit + 1, Device::GPU);
        const fs::path path = folder_ / "past_limit.sog";
        const auto result = io::save_sog(model, {.output_path = path});
        ASSERT_FALSE(result.has_value()) << "a SOG of " << kSogLimit + 1 << " splats was written";
        EXPECT_EQ(result.error().code, io::ErrorCode::INVALID_DATASET) << result.error().format();
        expect_no_file(path);
    }

    // The SOG limit counts visible splats: the same model with all but 1,000 deleted exports.
    TEST_F(LargeModelFormats, SogCountsVisibleSplatsAgainstTheLimit) {
        LFS_LARGE_MODEL_NEEDS(attribute_bytes(kSogLimit + 1) + kSogLimit, 0);
        SplatData model = sh0_model(kSogLimit + 1, Device::GPU);
        {
            Tensor mask = Tensor::ones_bool({kSogLimit + 1}, Device::GPU);
            mask.slice(0, 0, 1000).copy_(Tensor::zeros_bool({1000}, Device::GPU));
            (void)model.soft_delete(mask);
        }
        // visible_count() itself is checked in test_large_model_counts.cpp: it is a float sum of the deleted mask.
        const fs::path path = folder_ / "visible_below_limit.sog";
        const auto result = io::save_sog(model, {.output_path = path});
        ASSERT_TRUE(result.has_value()) << result.error().format();
        EXPECT_TRUE(fs::exists(path));
    }

    // The HTML viewer embeds one SOG and refuses the same model with the SOG's error. (SSOG is not refused: it
    // writes the model as chunks of its LOD tree, each its own SOG far below the limit.)
    TEST_F(LargeModelFormats, HtmlRefusesPastTheSogLimit) {
        LFS_LARGE_MODEL_NEEDS(attribute_bytes(kSogLimit + 1), 0);
        const SplatData model = sh0_model(kSogLimit + 1, Device::GPU);
        const fs::path html = folder_ / "past_limit.html";
        const auto result = io::export_html(model, {.output_path = html});
        ASSERT_FALSE(result.has_value()) << "an HTML viewer of " << kSogLimit + 1 << " splats was written";
        EXPECT_EQ(result.error().code, io::ErrorCode::INVALID_DATASET) << result.error().format();
        expect_no_file(html);
    }

    // SPZ past 30M visible splats: INVALID_DATASET, no file.
    TEST_F(LargeModelFormats, SpzRefusesPastThirtyMillion) {
        LFS_LARGE_MODEL_NEEDS(attribute_bytes(kSpzLimit + 1), 0);
        const SplatData model = sh0_model(kSpzLimit + 1, Device::GPU);
        const fs::path path = folder_ / "past_limit.spz";
        const auto result = io::save_spz(model, {.output_path = path});
        ASSERT_FALSE(result.has_value()) << "an SPZ of " << kSpzLimit + 1 << " splats was written";
        EXPECT_EQ(result.error().code, io::ErrorCode::INVALID_DATASET) << result.error().format();
        expect_no_file(path);
    }

    // NuRec past msgpack's 4 GiB a tensor: 47,721,859 visible splats at SH degree 3 (Q16 SH, zeroed), refused with
    // INVALID_DATASET before anything is written.
    TEST_F(LargeModelFormats, NurecRefusesTensorsPastMsgpack) {
        const size_t rows = kMsgpackLimitSh3;
        LFS_LARGE_MODEL_NEEDS(attribute_bytes(rows) + lm::store_bytes(lm::ShFormat::Q16, rows), 0);
        SplatData model(3, Tensor::zeros({rows, 3}, Device::GPU), Tensor::zeros({rows, 1, 3}, Device::GPU),
                        Tensor::zeros({lm::value_elements(lm::ShFormat::Q16, rows)}, Device::GPU, DataType::Float16),
                        Tensor::zeros({rows, 3}, Device::GPU), Tensor::full({rows, 4}, 0.5f, Device::GPU),
                        Tensor::zeros({rows, 1}, Device::GPU), 1.0f, SplatData::ShNLayout::Swizzled);
        model.shN_value_bounds() = Tensor::zeros({lm::bound_elements(lm::ShFormat::Q16, rows)}, Device::GPU);
        ASSERT_TRUE(model.shN_value_quantized());
        const fs::path path = folder_ / "past_msgpack.usdz";
        const auto result = io::save_nurec_usdz(model, {.output_path = path});
        ASSERT_FALSE(result.has_value()) << "a NuRec USDZ of " << rows << " splats at SH 3 was written";
        EXPECT_EQ(result.error().code, io::ErrorCode::INVALID_DATASET) << result.error().format();
        expect_no_file(path);
    }

    // Values exact in half precision, a function of the row and the attribute's column. Rotations (columns 9-12) are
    // the identity: the export writes normalized quaternions.
    float half_exact(const size_t row, const size_t column) {
        if (column >= 9 && column <= 12)
            return column == 9 ? 1.0f : 0.0f;
        return static_cast<float>(static_cast<int>((row * 61 + column * 17) % 2047) - 1023) / 1024.0f;
    }

    // A NuRec export and import of `rows` splats at SH degree 3, every value checked. The payload is about 118 bytes
    // a splat: past 2 GiB (libarchive's write size) from 18.2M splats and past 4 GiB (zlib's input size) from 36.4M.
    void round_trip(const fs::path& folder, const size_t rows) {
        lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(false);
        struct Restore {
            ~Restore() { lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt); }
        } restore;

        // The model on the host: means 0-2, sh0 3-5, scaling 6-8, rotation 9-12, opacity 13, SH rest 14-58.
        const auto column_tensor = [&](const size_t width, const size_t first_column,
                                       const std::initializer_list<size_t> shape) {
            Tensor host = Tensor::empty(lfs::core::TensorShape(shape), Device::CPU, DataType::Float32);
            auto* const data = host.ptr<float>();
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < width; ++c)
                    data[r * width + c] = half_exact(r, first_column + c);
            return host;
        };
        SplatData model(3, column_tensor(3, 0, {rows, 3}), column_tensor(3, 3, {rows, 1, 3}),
                        column_tensor(45, 14, {rows, 15, 3}), column_tensor(3, 6, {rows, 3}),
                        column_tensor(4, 9, {rows, 4}), column_tensor(1, 13, {rows, 1}), 1.0f);

        const fs::path path = folder / std::format("round_trip_{}.usdz", rows);
        const auto saved = io::save_nurec_usdz(model, {.output_path = path});
        ASSERT_TRUE(saved.has_value()) << saved.error().format();

        io::USDLoader loader;
        auto loaded = loader.load(path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().format();
        const auto* const data = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
        ASSERT_NE(data, nullptr);
        ASSERT_TRUE(*data);
        const SplatData& back = **data;
        ASSERT_EQ(static_cast<size_t>(back.size()), rows);
        ASSERT_EQ(back.get_max_sh_degree(), 3);

        const auto check = [&](const Tensor& tensor, const size_t width, const size_t first_column, const char* name) {
            const auto values = lm::host_copy<float>(tensor.contiguous());
            ASSERT_EQ(values.size(), rows * width) << name;
            size_t wrong = 0, first = 0;
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < width; ++c)
                    if (values[r * width + c] != half_exact(r, first_column + c) && wrong++ == 0)
                        first = r;
            EXPECT_EQ(wrong, 0u) << name << ": " << wrong << " of " << rows * width << " values differ; first in row "
                                 << first;
        };
        check(back.means(), 3, 0, "positions");
        check(back.sh0(), 3, 3, "albedo (sh0)");
        check(back.scaling_raw(), 3, 6, "scales");
        check(back.rotation_raw(), 4, 9, "rotations");
        check(back.opacity_raw(), 1, 13, "densities (opacity)");
        check(back.shN_canonical_cpu(), 45, 14, "specular SH");
    }

    // 18.3M splats: a payload of about 2.16 GB, past libarchive's INT_MAX bytes a write.
    TEST_F(LargeModelFormats, NurecRoundTripPast2GiBPayload) {
        constexpr size_t kRowsPast2GiB = 18'300'000;
        LFS_LARGE_MODEL_NEEDS(kRowsPast2GiB * 59 * sizeof(float) * 2, kRowsPast2GiB * 59 * sizeof(float) * 4);
        round_trip(folder_, kRowsPast2GiB);
    }

    // 36.5M splats: a payload of about 4.3 GB, past zlib's 32-bit input size.
    TEST_F(LargeModelFormats, NurecRoundTripPast4GiBPayload) {
        constexpr size_t kRowsPast4GiB = 36'500'000;
        LFS_LARGE_MODEL_NEEDS(kRowsPast4GiB * 59 * sizeof(float) * 2, kRowsPast4GiB * 59 * sizeof(float) * 4);
        round_trip(folder_, kRowsPast4GiB);
    }

#undef LFS_LARGE_MODEL_NEEDS

} // namespace
