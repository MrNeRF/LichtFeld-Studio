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
#include <random>
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

    void expect_point_cloud_failure(const fs::path& input,
                                    const std::string_view expected_message) {
        const auto direct_result = lfs::io::load_ply_point_cloud(input, {});
        ASSERT_FALSE(direct_result.has_value());
        EXPECT_NE(direct_result.error().find(expected_message), std::string::npos);

        lfs::io::PLYLoader loader;
        const auto legacy_result = loader.load(input);
        ASSERT_FALSE(legacy_result.has_value());
        EXPECT_EQ(legacy_result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
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

    TEST_F(PlyErrorTaxonomyTest, HeaderStructuralMalformationsReturnInvalidArgument) {
        struct Case {
            std::string description;
            std::string header;
            std::vector<std::pair<std::string, std::int64_t>> integer_fields;
            std::vector<std::pair<std::string, std::string>> string_fields;
        };

        const auto header_with_properties = [](const std::string_view properties) {
            return make_binary_header(1, properties);
        };
        const std::vector<Case> cases{
            {"duplicate format",
             "ply\nformat binary_little_endian 1.0\nformat binary_little_endian 1.0\n"
             "element vertex 1\nproperty float x\nend_header\n"},
            {"element before format",
             "ply\nelement vertex 1\nformat binary_little_endian 1.0\n"
             "property float x\nend_header\n"},
            {"duplicate vertex element",
             "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
             "element vertex 1\nproperty float x\nend_header\n"},
            {"property trailing text", header_with_properties("property float x trailing\n")},
            {"invalid property name",
             header_with_properties(std::string("property float invalid\x01name\n")),
             {},
             {{"property_name", std::string("invalid\x01name")}}},
            {"duplicate property name",
             header_with_properties("property float x\nproperty float x\n"),
             {},
             {{"property_name", "x"}}},
            {"unparseable f_dc suffix",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float f_dc_bad\n"),
             {{"max_exclusive", 48}},
             {{"property_name", "f_dc_bad"}}},
            {"out-of-range f_rest suffix",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float f_rest_999\n"),
             {{"max_exclusive", 135}},
             {{"property_name", "f_rest_999"}}},
            {"unparseable scale suffix",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float scale_bad\n"),
             {},
             {{"property_name", "scale_bad"}}},
            {"out-of-range rotation suffix",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float rot_4\n"),
             {},
             {{"property_name", "rot_4"}}},
            {"recognized property with wrong type",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property uchar opacity\n"),
             {},
             {{"property_name", "opacity"}, {"property_type", "uchar"}}},
            {"incomplete scale triplet",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float scale_0\nproperty float scale_1\n"),
             {{"scale_properties_present", 2}}},
            {"incomplete rotation quad",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float rot_0\nproperty float rot_1\n"
                                    "property float rot_2\n"),
             {{"rotation_properties_present", 3}}},
            {"sparse f_dc indices",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float f_dc_0\nproperty float f_dc_5\n"),
             {{"missing_index", 1}, {"dc_count", 6}}},
            {"f_dc count not divisible by three",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float f_dc_0\n"),
             {{"dc_count", 1}}},
            {"f_rest count not divisible by three",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float f_rest_0\n"),
             {{"rest_count", 1}}},
            {"incomplete SH band",
             header_with_properties("property float x\nproperty float y\nproperty float z\n"
                                    "property float f_rest_0\nproperty float f_rest_1\n"
                                    "property float f_rest_2\nproperty float f_rest_3\n"
                                    "property float f_rest_4\nproperty float f_rest_5\n"),
             {{"rest_count", 6}, {"coefficients_per_channel", 2}, {"candidate_root", 1}}},
        };

        size_t case_index = 0;
        for (const auto& test_case : cases) {
            SCOPED_TRACE(test_case.description);
            const fs::path input = path(std::format("structural_{}.ply", case_index++));
            write_binary_file(input, test_case.header, std::string(600, '\0'));

            const auto result = lfs::io::load_ply(input);

            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
            for (const auto& [key, expected] : test_case.integer_fields) {
                EXPECT_EQ(find_error_field<std::int64_t>(result.error(), key), expected);
            }
            for (const auto& [key, expected] : test_case.string_fields) {
                EXPECT_EQ(find_error_field<std::string>(result.error(), key), expected);
            }
        }
    }

    TEST_F(PlyErrorTaxonomyTest, UnsupportedSchemaVariantsReturnUnsupported) {
        struct Case {
            std::string_view description;
            std::string header;
            std::optional<std::string> observed_format;
        };
        const std::array<Case, 3> cases{
            Case{"ASCII format",
                 "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n",
                 "format ascii 1.0"},
            Case{"big-endian format",
                 "ply\nformat binary_big_endian 1.0\nelement vertex 1\n"
                 "property float x\nend_header\n",
                 "format binary_big_endian 1.0"},
            Case{"list property",
                 "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                 "property list uchar float x\nend_header\n",
                 std::nullopt},
        };

        size_t case_index = 0;
        for (const auto& test_case : cases) {
            SCOPED_TRACE(test_case.description);
            const fs::path input = path(std::format("unsupported_{}.ply", case_index++));
            write_binary_file(input, test_case.header);

            const auto result = lfs::io::load_ply(input);

            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Unsupported);
            if (test_case.observed_format) {
                EXPECT_EQ(find_error_field<std::string>(result.error(), "observed_format"),
                          test_case.observed_format);
            }
        }
    }

    TEST_F(PlyErrorTaxonomyTest, ZeroVertexPointCloudReturnsLegacyDataLoss) {
        const fs::path input = path("point_cloud_zero_vertices.ply");
        write_binary_file(input, make_binary_header(
                                     0, "property float x\nproperty float y\nproperty float z\n"));

        expect_point_cloud_failure(input, "at least one vertex");
    }

    TEST_F(PlyErrorTaxonomyTest, NonFinitePointCloudPositionReturnsLegacyDataLoss) {
        const fs::path input = path("point_cloud_non_finite_position.ply");
        std::string body;
        append_row(body, std::array<float, 3>{
                             std::numeric_limits<float>::quiet_NaN(), 2.0f, 3.0f});
        write_binary_file(input,
                          make_binary_header(
                              1, "property float x\nproperty float y\nproperty float z\n"),
                          body);

        expect_point_cloud_failure(input, "flat_index=0");
    }

    TEST_F(PlyErrorTaxonomyTest, NonFinitePointCloudFloatColorReturnsLegacyDataLoss) {
        const fs::path input = path("point_cloud_non_finite_color.ply");
        std::string body;
        append_row(body, std::array<float, 6>{
                             1.0f, 2.0f, 3.0f,
                             std::numeric_limits<float>::quiet_NaN(), 0.5f, 1.0f});
        write_binary_file(input,
                          make_binary_header(
                              1, "property float x\nproperty float y\nproperty float z\n"
                                 "property float red\nproperty float green\nproperty float blue\n"),
                          body);

        expect_point_cloud_failure(input, "flat_index=0");
    }

    TEST_F(PlyErrorTaxonomyTest, NonFinitePointCloudNormalReturnsLegacyDataLoss) {
        const fs::path input = path("point_cloud_non_finite_normal.ply");
        std::string body;
        append_row(body, std::array<float, 6>{
                             1.0f, 2.0f, 3.0f,
                             std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f});
        write_binary_file(input,
                          make_binary_header(
                              1, "property float x\nproperty float y\nproperty float z\n"
                                 "property float nx\nproperty float ny\nproperty float nz\n"),
                          body);

        expect_point_cloud_failure(input, "flat_index=0");
    }

    TEST_F(PlyErrorTaxonomyTest, HostileHeaderDoesNotEmitTensorContractViolation) {
        const fs::path input = path("duplicate_format_no_contract_report.ply");
        write_binary_file(
            input,
            "ply\nformat binary_little_endian 1.0\nformat binary_little_endian 1.0\n"
            "element vertex 1\nproperty float x\nend_header\n");
        std::atomic<bool> saw_tensor_contract{false};
        const LogHandlerGuard handler(
            [&](const lfs::core::LogLevel, const lfs::core::SourceSite&,
                const std::string_view message) {
                if (message.contains("tensor contract violation")) {
                    saw_tensor_contract.store(true, std::memory_order_relaxed);
                }
            });

        const auto result = lfs::io::load_ply(input);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
        EXPECT_FALSE(saw_tensor_contract.load(std::memory_order_relaxed));
    }

    TEST_F(PlyErrorTaxonomyTest, DeterministicHeaderTruncationNeverSucceedsOrReturnsInternal) {
        std::string full_bytes = make_binary_header(3, kGaussianProperties);
        const std::array<float, 11> row{
            1.0f, 2.0f, 3.0f,
            -2.0f, -2.0f, -2.0f, 0.25f,
            1.0f, 0.0f, 0.0f, 0.0f};
        append_row(full_bytes, row);
        append_row(full_bytes, row);
        append_row(full_bytes, row);

        std::mt19937 random(12345u);
        std::uniform_int_distribution<size_t> truncation_length(0, full_bytes.size() - 1);
        for (size_t iteration = 0; iteration < 200; ++iteration) {
            SCOPED_TRACE(iteration);
            const fs::path input = path(std::format("truncated_fuzz_{}.ply", iteration));
            const size_t length = truncation_length(random);
            write_binary_file(input, std::string_view(full_bytes).substr(0, length));

            const auto result = lfs::io::load_ply(input);

            ASSERT_FALSE(result.has_value());
            const lfs::ErrorCode code = result.error().code();
            EXPECT_TRUE(code == lfs::ErrorCode::DataLoss ||
                        code == lfs::ErrorCode::InvalidArgument ||
                        code == lfs::ErrorCode::Unsupported ||
                        code == lfs::ErrorCode::ResourceExhausted)
                << "unexpected code " << static_cast<int>(code)
                << " for truncation length " << length;
        }
    }

} // namespace
