/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "checkpoint_fixture.hpp"
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "core/checkpoint_format.hpp"
#include "core/error.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/checkpoint.hpp"
#include "training/strategies/mcmc.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

    constexpr uint32_t BILATERAL_MAGIC = 0x4C464247;
    constexpr uint32_t BILATERAL_LEGACY_VERSION = 1;
    constexpr uint32_t BILATERAL_VERSION = 2;
    constexpr size_t BILATERAL_CONFIG_OFFSET = 24;
    constexpr size_t BILATERAL_CONFIG_BLOCK_BYTES = 60;

    constexpr uint32_t PPISP_MAGIC = 0x4C465050;
    constexpr uint32_t PPISP_LEGACY_VERSION = 2;
    constexpr uint32_t PPISP_VERSION = 3;
    constexpr size_t PPISP_CONFIG_OFFSET = 16;
    constexpr size_t PPISP_CONFIG_BLOCK_BYTES = 84;

    constexpr uint32_t CONTROLLER_POOL_MAGIC = 0x4C465043;
    constexpr uint32_t CONTROLLER_POOL_LEGACY_VERSION = 1;
    constexpr uint32_t CONTROLLER_POOL_VERSION = 2;
    constexpr size_t CONTROLLER_POOL_CONFIG_OFFSET = 16;
    constexpr size_t CONTROLLER_POOL_CONFIG_BLOCK_BYTES = 60;

    constexpr uint32_t CONFIG_SCHEMA_VERSION = 1;
    constexpr uint32_t OPTIMIZER_CONFIG_V1_BYTES = 52;
    constexpr uint32_t PPISP_CONFIG_V1_BYTES = 76;

    struct LegacyBilateralConfigV1 {
        double lr;
        double beta1;
        double beta2;
        double eps;
        int warmup_steps;
        double warmup_start_factor;
        double final_lr_factor;
    };

    struct LegacyPPISPConfigV2 {
        double lr;
        double beta1;
        double beta2;
        double eps;
        int warmup_steps;
        double warmup_start_factor;
        double final_lr_factor;
        float exposure_mean;
        float vig_center;
        float vig_channel;
        float vig_non_pos;
        float color_mean;
        float crf_channel;
    };

    struct LegacyControllerPoolConfigV1 {
        double lr;
        double beta1;
        double beta2;
        double eps;
        int warmup_steps;
        double warmup_start_factor;
        double final_lr_factor;
    };

    static_assert(std::is_standard_layout_v<LegacyBilateralConfigV1>);
    static_assert(sizeof(LegacyBilateralConfigV1) == 56);
    static_assert(offsetof(LegacyBilateralConfigV1, warmup_steps) == 32);
    static_assert(offsetof(LegacyBilateralConfigV1, warmup_start_factor) == 40);
    static_assert(offsetof(LegacyBilateralConfigV1, final_lr_factor) == 48);

    static_assert(std::is_standard_layout_v<LegacyPPISPConfigV2>);
    static_assert(sizeof(LegacyPPISPConfigV2) == 80);
    static_assert(offsetof(LegacyPPISPConfigV2, warmup_steps) == 32);
    static_assert(offsetof(LegacyPPISPConfigV2, warmup_start_factor) == 40);
    static_assert(offsetof(LegacyPPISPConfigV2, final_lr_factor) == 48);
    static_assert(offsetof(LegacyPPISPConfigV2, exposure_mean) == 56);
    static_assert(offsetof(LegacyPPISPConfigV2, crf_channel) == 76);

    static_assert(std::is_standard_layout_v<LegacyControllerPoolConfigV1>);
    static_assert(sizeof(LegacyControllerPoolConfigV1) == 56);
    static_assert(offsetof(LegacyControllerPoolConfigV1, warmup_steps) == 32);
    static_assert(offsetof(LegacyControllerPoolConfigV1, warmup_start_factor) == 40);
    static_assert(offsetof(LegacyControllerPoolConfigV1, final_lr_factor) == 48);

    template <typename T>
    using ScalarBits = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;

    template <typename T>
    T read_little_endian(const std::string& bytes, const size_t offset) {
        static_assert(std::is_trivially_copyable_v<T>);
        EXPECT_LE(offset + sizeof(T), bytes.size());
        ScalarBits<T> bits = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            bits |= static_cast<ScalarBits<T>>(static_cast<unsigned char>(bytes[offset + i])) << (i * 8);
        }
        return std::bit_cast<T>(bits);
    }

    void write_little_endian_u32(std::string& bytes, const size_t offset, const uint32_t value) {
        ASSERT_LE(offset + sizeof(value), bytes.size());
        for (size_t i = 0; i < sizeof(value); ++i) {
            bytes[offset + i] = static_cast<char>((value >> (i * 8)) & 0xffU);
        }
    }

    std::string component_header(const uint32_t magic, const uint32_t version) {
        std::string bytes(sizeof(uint32_t) * 2, '\0');
        write_little_endian_u32(bytes, 0, magic);
        write_little_endian_u32(bytes, sizeof(uint32_t), version);
        return bytes;
    }

    template <typename LegacyConfig>
    std::string make_legacy_blob(
        std::string bytes,
        const size_t config_offset,
        const size_t new_config_bytes,
        const uint32_t legacy_version,
        const LegacyConfig& legacy_config) {
        EXPECT_LE(config_offset + new_config_bytes, bytes.size());
        write_little_endian_u32(bytes, sizeof(uint32_t), legacy_version);
        bytes.replace(
            config_offset,
            new_config_bytes,
            reinterpret_cast<const char*>(&legacy_config),
            sizeof(legacy_config));
        return bytes;
    }

    lfs::training::BilateralGrid::Config bilateral_config() {
        return {
            .lr = 0.0125,
            .beta1 = 0.71,
            .beta2 = 0.923,
            .eps = 3.25e-9,
            .warmup_steps = 7,
            .warmup_start_factor = 0.23,
            .final_lr_factor = 0.37,
        };
    }

    lfs::training::PPISP::Config ppisp_config() {
        return {
            .lr = 0.0175,
            .beta1 = 0.73,
            .beta2 = 0.947,
            .eps = 4.5e-10,
            .warmup_steps = 9,
            .warmup_start_factor = 0.19,
            .final_lr_factor = 0.41,
            .exposure_mean = 0.31f,
            .vig_center = 0.07f,
            .vig_channel = 0.13f,
            .vig_non_pos = 0.17f,
            .color_mean = 0.29f,
            .crf_channel = 0.23f,
        };
    }

    lfs::training::PPISPControllerPool::Config controller_pool_config() {
        return {
            .lr = 0.0075,
            .beta1 = 0.67,
            .beta2 = 0.937,
            .eps = 6.25e-7,
            .warmup_steps = 11,
            .warmup_start_factor = 0.27,
            .final_lr_factor = 0.43,
        };
    }

    template <typename Expected, typename Actual>
    void expect_optimizer_config_eq(const Expected& expected, const Actual& actual) {
        EXPECT_DOUBLE_EQ(actual.lr, expected.lr);
        EXPECT_DOUBLE_EQ(actual.beta1, expected.beta1);
        EXPECT_DOUBLE_EQ(actual.beta2, expected.beta2);
        EXPECT_DOUBLE_EQ(actual.eps, expected.eps);
        EXPECT_EQ(actual.warmup_steps, expected.warmup_steps);
        EXPECT_DOUBLE_EQ(actual.warmup_start_factor, expected.warmup_start_factor);
        EXPECT_DOUBLE_EQ(actual.final_lr_factor, expected.final_lr_factor);
    }

    void expect_ppisp_config_eq(
        const lfs::training::PPISP::Config& expected,
        const lfs::training::PPISP::Config& actual) {
        expect_optimizer_config_eq(expected, actual);
        EXPECT_FLOAT_EQ(actual.exposure_mean, expected.exposure_mean);
        EXPECT_FLOAT_EQ(actual.vig_center, expected.vig_center);
        EXPECT_FLOAT_EQ(actual.vig_channel, expected.vig_channel);
        EXPECT_FLOAT_EQ(actual.vig_non_pos, expected.vig_non_pos);
        EXPECT_FLOAT_EQ(actual.color_mean, expected.color_mean);
        EXPECT_FLOAT_EQ(actual.crf_channel, expected.crf_channel);
    }

    template <typename Config>
    void expect_optimizer_config_wire(
        const std::string& bytes,
        size_t offset,
        const Config& config) {
        EXPECT_EQ(read_little_endian<uint32_t>(bytes, offset), CONFIG_SCHEMA_VERSION);
        offset += sizeof(uint32_t);
        EXPECT_EQ(read_little_endian<uint32_t>(bytes, offset), OPTIMIZER_CONFIG_V1_BYTES);
        offset += sizeof(uint32_t);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.lr);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.beta1);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.beta2);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.eps);
        offset += sizeof(double);
        EXPECT_EQ(read_little_endian<int32_t>(bytes, offset), config.warmup_steps);
        offset += sizeof(int32_t);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.warmup_start_factor);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.final_lr_factor);
    }

    void expect_ppisp_config_wire(
        const std::string& bytes,
        size_t offset,
        const lfs::training::PPISP::Config& config) {
        EXPECT_EQ(read_little_endian<uint32_t>(bytes, offset), CONFIG_SCHEMA_VERSION);
        offset += sizeof(uint32_t);
        EXPECT_EQ(read_little_endian<uint32_t>(bytes, offset), PPISP_CONFIG_V1_BYTES);
        offset += sizeof(uint32_t);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.lr);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.beta1);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.beta2);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.eps);
        offset += sizeof(double);
        EXPECT_EQ(read_little_endian<int32_t>(bytes, offset), config.warmup_steps);
        offset += sizeof(int32_t);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.warmup_start_factor);
        offset += sizeof(double);
        EXPECT_DOUBLE_EQ(read_little_endian<double>(bytes, offset), config.final_lr_factor);
        offset += sizeof(double);
        EXPECT_FLOAT_EQ(read_little_endian<float>(bytes, offset), config.exposure_mean);
        offset += sizeof(float);
        EXPECT_FLOAT_EQ(read_little_endian<float>(bytes, offset), config.vig_center);
        offset += sizeof(float);
        EXPECT_FLOAT_EQ(read_little_endian<float>(bytes, offset), config.vig_channel);
        offset += sizeof(float);
        EXPECT_FLOAT_EQ(read_little_endian<float>(bytes, offset), config.vig_non_pos);
        offset += sizeof(float);
        EXPECT_FLOAT_EQ(read_little_endian<float>(bytes, offset), config.color_mean);
        offset += sizeof(float);
        EXPECT_FLOAT_EQ(read_little_endian<float>(bytes, offset), config.crf_channel);
    }

    template <typename Load>
    void expect_typed_future_version_failure(
        const std::string_view component,
        const uint32_t magic,
        const uint32_t current_version,
        Load&& load) {
        std::stringstream stream(component_header(magic, current_version + 1));
        try {
            load(stream);
            FAIL() << "future " << component << " version was accepted";
        } catch (const lfs::Exception& exception) {
            EXPECT_EQ(exception.error().code(), lfs::ErrorCode::Unsupported);
            EXPECT_EQ(exception.error().domain(), lfs::ErrorDomain::Training);
            EXPECT_NE(exception.error().detail().find(component), std::string_view::npos);
            EXPECT_NE(
                exception.error().detail().find(std::to_string(current_version + 1)),
                std::string_view::npos);
        } catch (const std::exception& exception) {
            FAIL() << "future " << component << " version returned untyped error: " << exception.what();
        }
    }

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3] = static_cast<float>(i);
            rotations[i * 4] = 1.0f;
        }
        return std::make_unique<lfs::core::SplatData>(
            0,
            lfs::core::Tensor::from_vector(means, {count, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros(
                {count, size_t{1}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            lfs::core::Tensor::zeros({size_t{0}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            lfs::core::Tensor::zeros(
                {count, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            lfs::core::Tensor::from_vector(rotations, {count, size_t{4}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros(
                {count, size_t{1}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            1.0f);
    }

    class ScopedTestDirectory {
    public:
        explicit ScopedTestDirectory(const std::string_view name)
            : path_(std::filesystem::temp_directory_path() / std::string(name)) {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
            std::filesystem::create_directories(path_ / "checkpoints");
        }

        ~ScopedTestDirectory() {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }

        const std::filesystem::path& path() const { return path_; }

    private:
        std::filesystem::path path_;
    };

    TEST(BilateralGridConfigSerializationTest, NewFormatRoundTripPreservesEveryField) {
        const auto config = bilateral_config();
        lfs::training::BilateralGrid source(1, 2, 3, 4, 100, config);
        std::stringstream stream;
        source.serialize(stream);

        const auto bytes = stream.str();
        EXPECT_EQ(read_little_endian<uint32_t>(bytes, sizeof(uint32_t)), BILATERAL_VERSION);
        expect_optimizer_config_wire(bytes, BILATERAL_CONFIG_OFFSET, config);

        lfs::training::BilateralGrid loaded(1, 1, 1, 1, 1);
        loaded.deserialize(stream);
        expect_optimizer_config_eq(config, loaded.get_config());
    }

    TEST(BilateralGridConfigSerializationTest, LegacyVersionOneRawConfigLoads) {
        const auto config = bilateral_config();
        lfs::training::BilateralGrid source(1, 2, 3, 4, 100, config);
        std::stringstream current;
        source.serialize(current);

        const LegacyBilateralConfigV1 legacy{
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.warmup_steps,
            config.warmup_start_factor,
            config.final_lr_factor,
        };
        std::stringstream stream(make_legacy_blob(
            current.str(),
            BILATERAL_CONFIG_OFFSET,
            BILATERAL_CONFIG_BLOCK_BYTES,
            BILATERAL_LEGACY_VERSION,
            legacy));
        lfs::training::BilateralGrid loaded(1, 1, 1, 1, 1);
        loaded.deserialize(stream);
        expect_optimizer_config_eq(config, loaded.get_config());
    }

    TEST(PPISPConfigSerializationTest, NewFormatRoundTripPreservesEveryField) {
        const auto config = ppisp_config();
        lfs::training::PPISP source(100, config);
        source.register_frame(7, 11);
        source.finalize();
        std::stringstream stream;
        source.serialize(stream);

        const auto bytes = stream.str();
        EXPECT_EQ(read_little_endian<uint32_t>(bytes, sizeof(uint32_t)), PPISP_VERSION);
        expect_ppisp_config_wire(bytes, PPISP_CONFIG_OFFSET, config);

        lfs::training::PPISP loaded(1);
        loaded.deserialize(stream);
        expect_ppisp_config_eq(config, loaded.get_config());
    }

    TEST(PPISPConfigSerializationTest, LegacyVersionTwoRawConfigLoads) {
        const auto config = ppisp_config();
        lfs::training::PPISP source(100, config);
        source.register_frame(7, 11);
        source.finalize();
        std::stringstream current;
        source.serialize(current);

        const LegacyPPISPConfigV2 legacy{
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.warmup_steps,
            config.warmup_start_factor,
            config.final_lr_factor,
            config.exposure_mean,
            config.vig_center,
            config.vig_channel,
            config.vig_non_pos,
            config.color_mean,
            config.crf_channel,
        };
        std::stringstream stream(make_legacy_blob(
            current.str(),
            PPISP_CONFIG_OFFSET,
            PPISP_CONFIG_BLOCK_BYTES,
            PPISP_LEGACY_VERSION,
            legacy));
        lfs::training::PPISP loaded(1);
        loaded.deserialize(stream);
        expect_ppisp_config_eq(config, loaded.get_config());
    }

    TEST(PPISPControllerPoolConfigSerializationTest, NewFormatRoundTripPreservesEveryField) {
        const auto config = controller_pool_config();
        lfs::training::PPISPControllerPool source(1, 100, config);
        std::stringstream stream;
        source.serialize(stream);

        const auto bytes = stream.str();
        EXPECT_EQ(read_little_endian<uint32_t>(bytes, sizeof(uint32_t)), CONTROLLER_POOL_VERSION);
        expect_optimizer_config_wire(bytes, CONTROLLER_POOL_CONFIG_OFFSET, config);

        lfs::training::PPISPControllerPool loaded(1, 1);
        loaded.deserialize(stream);
        expect_optimizer_config_eq(config, loaded.get_config());
    }

    TEST(PPISPControllerPoolConfigSerializationTest, LegacyVersionOneRawConfigLoads) {
        const auto config = controller_pool_config();
        lfs::training::PPISPControllerPool source(1, 100, config);
        std::stringstream current;
        source.serialize(current);

        const LegacyControllerPoolConfigV1 legacy{
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.warmup_steps,
            config.warmup_start_factor,
            config.final_lr_factor,
        };
        std::stringstream stream(make_legacy_blob(
            current.str(),
            CONTROLLER_POOL_CONFIG_OFFSET,
            CONTROLLER_POOL_CONFIG_BLOCK_BYTES,
            CONTROLLER_POOL_LEGACY_VERSION,
            legacy));
        lfs::training::PPISPControllerPool loaded(1, 1);
        loaded.deserialize(stream);
        expect_optimizer_config_eq(config, loaded.get_config());
    }

    TEST(ComponentConfigSerializationTest, FutureComponentVersionsReturnTypedUnsupported) {
        {
            SCOPED_TRACE("BilateralGrid");
            lfs::training::BilateralGrid loaded(1, 1, 1, 1, 1);
            expect_typed_future_version_failure(
                "BilateralGrid", BILATERAL_MAGIC, BILATERAL_VERSION,
                [&](std::istream& stream) { loaded.deserialize(stream); });
        }
        {
            SCOPED_TRACE("PPISP");
            lfs::training::PPISP loaded(1);
            expect_typed_future_version_failure(
                "PPISP", PPISP_MAGIC, PPISP_VERSION,
                [&](std::istream& stream) { loaded.deserialize(stream); });
        }
        {
            SCOPED_TRACE("PPISPControllerPool");
            lfs::training::PPISPControllerPool loaded(1, 1);
            expect_typed_future_version_failure(
                "PPISPControllerPool", CONTROLLER_POOL_MAGIC, CONTROLLER_POOL_VERSION,
                [&](std::istream& stream) { loaded.deserialize(stream); });
        }
    }

    TEST(CheckpointComponentConfigRoundTripTest, AllConfigsRoundTripInsideLfkp) {
        const ScopedTestDirectory temp_dir("lfs_checkpoint_component_config_roundtrip");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir.path();
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 100;
        params.optimization.sh_degree = 0;
        params.optimization.max_cap = 16;

        auto source_model = make_test_splat(2);
        lfs::training::MCMC source_strategy(*source_model);
        const auto bilateral = bilateral_config();
        const auto ppisp = ppisp_config();
        const auto controller = controller_pool_config();
        lfs::training::BilateralGrid source_bilateral(1, 2, 3, 4, 100, bilateral);
        lfs::training::PPISP source_ppisp(100, ppisp);
        source_ppisp.register_frame(7, 11);
        source_ppisp.finalize();
        lfs::training::PPISPControllerPool source_controller(1, 100, controller);

        const auto saved = lfs::test::write_checkpoint_fixture(
            temp_dir.path(),
            17,
            source_strategy,
            params,
            &source_bilateral,
            &source_ppisp,
            &source_controller,
            nullptr);
        ASSERT_TRUE(saved.has_value()) << saved.error();

        const auto checkpoint = lfs::test::checkpoint_fixture_path(temp_dir.path());
        const auto header = lfs::core::load_checkpoint_header(checkpoint);
        ASSERT_TRUE(header.has_value()) << header.error();
        EXPECT_EQ(header->version, lfs::core::CHECKPOINT_VERSION_FIELDWISE_CONFIGS);

        auto target_model = make_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        lfs::training::BilateralGrid target_bilateral(1, 1, 1, 1, 1);
        lfs::training::PPISP target_ppisp(1);
        lfs::training::PPISPControllerPool target_controller(1, 1);
        auto loaded_params = params;
        const auto loaded = lfs::training::load_checkpoint(
            checkpoint,
            target_strategy,
            loaded_params,
            &target_bilateral,
            &target_ppisp,
            &target_controller,
            nullptr);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        EXPECT_EQ(*loaded, 17);
        expect_optimizer_config_eq(bilateral, target_bilateral.get_config());
        expect_ppisp_config_eq(ppisp, target_ppisp.get_config());
        expect_optimizer_config_eq(controller, target_controller.get_config());
    }

} // namespace
