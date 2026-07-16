/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/formats/ply.hpp"
#include "io/loaders/ply_loader.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

    namespace fs = std::filesystem;

    constexpr std::string_view kGaussianProperties =
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float opacity\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n";

    class LogHandlerGuard {
    public:
        explicit LogHandlerGuard(lfs::core::LogHandler handler)
            : token_(lfs::core::Logger::get().add_log_handler(std::move(handler))) {}

        ~LogHandlerGuard() {
            lfs::core::Logger::get().remove_log_handler(token_);
        }

        LogHandlerGuard(const LogHandlerGuard&) = delete;
        LogHandlerGuard& operator=(const LogHandlerGuard&) = delete;

    private:
        lfs::core::LogHandlerToken token_;
    };

    std::string make_binary_header(const size_t vertex_count,
                                   const std::string_view properties) {
        return std::format(
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex {}\n"
            "{}"
            "end_header\n",
            vertex_count, properties);
    }

    void append_float(std::string& bytes, const float value) {
        const size_t offset = bytes.size();
        bytes.resize(offset + sizeof(value));
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    template <size_t N>
    void append_row(std::string& bytes, const std::array<float, N>& values) {
        for (const float value : values) {
            append_float(bytes, value);
        }
    }

    void write_binary_file(const fs::path& path,
                           const std::string_view header,
                           const std::string_view body = {}) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(stream.is_open());
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        stream.write(body.data(), static_cast<std::streamsize>(body.size()));
        ASSERT_TRUE(stream.good());
    }

    template <class T>
    std::optional<T> find_field(const lfs::SmallFields& fields,
                                const std::string_view key) {
        for (const auto& entry : fields.entries()) {
            if (entry.key == key) {
                if (const auto* value = std::get_if<T>(&entry.value)) {
                    return *value;
                }
            }
        }
        return std::nullopt;
    }

    template <class T>
    std::optional<T> find_error_field(const lfs::Error& error,
                                      const std::string_view key) {
        for (const auto& frame : error.frames()) {
            if (auto value = find_field<T>(frame.fields, key)) {
                return value;
            }
        }
        return std::nullopt;
    }

    bool is_failure_log_level(const lfs::core::LogLevel level) {
        return level == lfs::core::LogLevel::Error ||
               level == lfs::core::LogLevel::Critical;
    }

    class PlyErrorTaxonomyTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::core::reset_failure_report_dedup_for_testing();
            previous_log_level_ = lfs::core::Logger::get().level();
            lfs::core::Logger::get().set_level(lfs::core::LogLevel::Info);

            static std::atomic<std::uint64_t> sequence{0};
            const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
            temp_dir_ = fs::temp_directory_path() /
                        std::format("lfs_ply_error_taxonomy_{}_{}", tick,
                                    sequence.fetch_add(1, std::memory_order_relaxed));
            ASSERT_TRUE(fs::create_directories(temp_dir_));
        }

        void TearDown() override {
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
            lfs::core::Logger::get().set_level(previous_log_level_);
        }

        fs::path path(const std::string_view filename) const {
            return temp_dir_ / filename;
        }

        fs::path temp_dir_;
        lfs::core::LogLevel previous_log_level_ = lfs::core::LogLevel::Info;
    };

    TEST_F(PlyErrorTaxonomyTest, MissingFileReturnsNotFoundWithPathAndNoFailureLog) {
        const fs::path missing = path("missing.ply");
        std::atomic<size_t> failure_logs{0};
        const LogHandlerGuard handler(
            [&](const lfs::core::LogLevel level, const lfs::core::SourceSite&,
                const std::string_view) {
                if (is_failure_log_level(level)) {
                    failure_logs.fetch_add(1, std::memory_order_relaxed);
                }
            });

        const auto result = lfs::io::load_ply(missing);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::NotFound);
        EXPECT_EQ(find_error_field<std::string>(result.error(), "path"),
                  lfs::core::path_to_utf8(missing));
        EXPECT_EQ(failure_logs.load(std::memory_order_relaxed), 0u);

        lfs::io::PLYLoader loader;
        const auto legacy_result = loader.load(missing);
        ASSERT_FALSE(legacy_result.has_value());
        EXPECT_EQ(legacy_result.error().code, lfs::io::ErrorCode::PATH_NOT_FOUND);
        EXPECT_EQ(failure_logs.load(std::memory_order_relaxed), 0u);
    }

    TEST_F(PlyErrorTaxonomyTest, InvalidHeaderReturnsInvalidArgument) {
        const fs::path input = path("invalid_header.ply");
        write_binary_file(input, "not a PLY header\n");

        const auto result = lfs::io::load_ply(input);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
    }

    TEST_F(PlyErrorTaxonomyTest, CheckedBodySizeOverflowReturnsResourceExhausted) {
        const fs::path input = path("overflow.ply");
        const std::string header = std::format(
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex {}\n"
            "property float x\n"
            "end_header\n",
            std::numeric_limits<size_t>::max());
        write_binary_file(input, header);

        const auto result = lfs::io::load_ply(input);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
    }

    TEST_F(PlyErrorTaxonomyTest, TruncatedBodyReturnsDataLossWithByteCounts) {
        const fs::path input = path("truncated.ply");
        const std::string header = make_binary_header(
            2,
            "property float x\n"
            "property float y\n"
            "property float z\n");
        std::string body;
        append_row(body, std::array{1.0f, 2.0f, 3.0f});
        write_binary_file(input, header, body);

        const auto result = lfs::io::load_ply(input);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "declared_vertices"), 2);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "vertex_stride"), 12);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "required_bytes"), 24);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "available_bytes"), 12);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "complete_vertices"), 1);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "missing_bytes"), 12);
    }

    TEST_F(PlyErrorTaxonomyTest, AllInvalidRowsReturnDataLoss) {
        const fs::path input = path("all_invalid.ply");
        std::string body;
        append_row(body, std::array<float, 11>{
                             std::numeric_limits<float>::quiet_NaN(), 2.0f, 3.0f,
                             -2.0f, -2.0f, -2.0f, 0.25f,
                             1.0f, 0.0f, 0.0f, 0.0f});
        write_binary_file(input, make_binary_header(1, kGaussianProperties), body);

        const auto result = lfs::io::load_ply(input);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::DataLoss);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "invalid_rows"), 1);
        EXPECT_EQ(find_error_field<std::int64_t>(result.error(), "vertex_count"), 1);
    }

    TEST_F(PlyErrorTaxonomyTest, PartialInvalidRowsReturnStructuredWarning) {
        const fs::path input = path("partial_invalid.ply");
        const std::string properties = std::format(
            "{}"
            "property float f_rest_0\n"
            "property float f_rest_1\n"
            "property float f_rest_2\n"
            "property float f_rest_3\n"
            "property float f_rest_4\n"
            "property float f_rest_5\n"
            "property float f_rest_6\n"
            "property float f_rest_7\n"
            "property float f_rest_8\n",
            kGaussianProperties);
        std::string body;
        append_row(body, std::array<float, 20>{
                             1.0f, 2.0f, 3.0f,
                             -2.0f, -2.0f, -2.0f, 0.25f,
                             1.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        append_row(body, std::array<float, 20>{
                             std::numeric_limits<float>::quiet_NaN(), 5.0f, 6.0f,
                             -2.0f, -2.0f, -2.0f, 0.25f,
                             1.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        write_binary_file(input, make_binary_header(2, properties), body);
        lfs::io::LoadOptions options;
        options.splat_tensor_allocator =
            [](lfs::core::TensorShape shape, const size_t, const lfs::core::DataType dtype,
               const std::string_view) {
                return lfs::core::Tensor::empty(
                    std::move(shape), lfs::core::Device::CPU, dtype);
            };

        auto result = lfs::io::load_ply(input, options);

        ASSERT_TRUE(result.has_value()) << lfs::format_for_developer(result.error());
        EXPECT_EQ(result->value.size(), 1u);
        ASSERT_EQ(result->warnings.size(), 1u);
        const lfs::io::Diagnostic& warning = result->warnings.front();
        EXPECT_EQ(warning.code, lfs::ErrorCode::DataLoss);
        EXPECT_EQ(find_field<std::int64_t>(warning.fields, "invalid_rows"), 1);
        EXPECT_NE(warning.message.find("1 invalid splat(s)"), std::string::npos);

        lfs::io::PLYLoader loader;
        const auto loader_result = loader.load(input, options);
        ASSERT_TRUE(loader_result.has_value()) << loader_result.error().format();
        ASSERT_EQ(loader_result->warnings.size(), 1u);
        EXPECT_NE(loader_result->warnings.front().find("1 invalid splat(s)"),
                  std::string::npos);
    }

    TEST_F(PlyErrorTaxonomyTest, CancellationReturnsCancelled) {
        const fs::path input = path("cancelled.ply");
        write_binary_file(input, "not a PLY header\n");
        lfs::io::LoadOptions options;
        options.cancel_requested = [] { return true; };

        const auto result = lfs::io::load_ply(input, options);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
    }

    TEST_F(PlyErrorTaxonomyTest, TruncatedBodyReportsExactlyOnceAtLoaderOwnerBoundary) {
        const fs::path input = path("owner_log_truncated.ply");
        std::string body;
        append_row(body, std::array<float, 11>{
                             1.0f, 2.0f, 3.0f,
                             -2.0f, -2.0f, -2.0f, 0.25f,
                             1.0f, 0.0f, 0.0f, 0.0f});
        write_binary_file(input, make_binary_header(2, kGaussianProperties), body);

        std::atomic<size_t> failure_logs{0};
        const LogHandlerGuard handler(
            [&](const lfs::core::LogLevel level, const lfs::core::SourceSite&,
                const std::string_view) {
                if (is_failure_log_level(level)) {
                    failure_logs.fetch_add(1, std::memory_order_relaxed);
                }
            });

        lfs::io::PLYLoader loader;
        const auto result = loader.load(input);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
        EXPECT_EQ(failure_logs.load(std::memory_order_relaxed), 1u);
    }

} // namespace
