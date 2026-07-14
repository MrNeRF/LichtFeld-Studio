/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/abi.hpp"
#include "core/tensor.hpp"
#include "io/formats/colmap.hpp"
#include "io/formats/ply.hpp"
#include "lfs_core_abi_stamp.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    class AssertHardeningExpectedFailTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            temp_dir_ = fs::temp_directory_path() /
                        ("lfs_assert_hardening_" + std::to_string(nonce) + "_" +
                         std::to_string(reinterpret_cast<std::uintptr_t>(this)));
            fs::create_directories(temp_dir_);
        }

        void TearDown() override {
            std::error_code error;
            fs::remove_all(temp_dir_, error);
        }

        void write_text(const fs::path& path, const std::string& contents) const {
            std::ofstream stream(path, std::ios::binary);
            ASSERT_TRUE(stream.is_open());
            stream << contents;
            ASSERT_TRUE(stream.good());
        }

        template <size_t N>
        void write_binary_ply(const fs::path& path,
                              const std::string& header,
                              const std::array<float, N>& payload) const {
            std::ofstream stream(path, std::ios::binary);
            ASSERT_TRUE(stream.is_open());
            stream.write(header.data(), static_cast<std::streamsize>(header.size()));
            stream.write(reinterpret_cast<const char*>(payload.data()),
                         static_cast<std::streamsize>(sizeof(payload)));
            ASSERT_TRUE(stream.good());
        }

        fs::path temp_dir_;
    };

    TEST_F(AssertHardeningExpectedFailTest, RejectsBrickCapacitySmallerThanLogicalRows) {
        EXPECT_THROW((void)Tensor::zeros_direct({4, 3}, 3, Device::CUDA),
                     std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsNonBooleanMaskedSelectMask) {
        const auto values = Tensor::from_vector({1.0f, 2.0f}, {2}, Device::CPU);
        const auto invalid_mask = Tensor::from_vector({1.0f, 0.0f}, {2}, Device::CPU);

        EXPECT_THROW((void)values.masked_select(invalid_mask), std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsOutOfBoundsIndexBeforeGathering) {
        const auto values = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, Device::CPU);
        const auto indices = Tensor::from_vector({3}, {1}, Device::CPU);

        EXPECT_THROW((void)values.index_select(0, indices), std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsUnsupportedPartialGatherDispatch) {
        const auto values = Tensor::from_vector({1, 2, 3}, {3}, Device::CPU);
        const auto indices = Tensor::from_vector({0}, {1}, Device::CPU);

        EXPECT_THROW((void)values.gather(0, indices), std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsUnsupportedMultiTensorIndexing) {
        auto values = Tensor::zeros({2}, Device::CPU);
        const auto first = Tensor::from_vector({0}, {1}, Device::CPU);
        const auto second = Tensor::from_vector({1}, {1}, Device::CPU);
        const std::vector<Tensor> indices{first, second};

        EXPECT_THROW((void)values[indices], std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsRowAssignmentAcrossDtypes) {
        auto destination = Tensor::zeros({2, 2}, Device::CPU, DataType::Float32);
        const auto source = Tensor::from_vector({1, 2}, {2}, Device::CPU);

        EXPECT_THROW(destination[0] = source, std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsUnrepresentableIntegralMaskedFill) {
        auto values = Tensor::zeros({2}, Device::CPU, DataType::UInt8);
        const auto mask = Tensor::from_vector({true, false}, {2}, Device::CPU);

        EXPECT_THROW(values.masked_fill_(mask, 256.0f), std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsAllCloseOnUnsupportedDtype) {
        const auto lhs = Tensor::from_vector({1, 2}, {2}, Device::CPU);
        const auto rhs = Tensor::from_vector({1, 2}, {2}, Device::CPU);

        EXPECT_THROW((void)lhs.all_close(rhs), std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsUnimplementedArgmaxDispatch) {
        const auto values = Tensor::from_vector({1.0f, 3.0f, 2.0f}, {3}, Device::CPU);

        EXPECT_THROW((void)values.argmax(), std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsDuplicatePlyVertexProperties) {
        const fs::path path = temp_dir_ / "duplicate_property.ply";
        write_binary_ply(
            path,
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n",
            std::array{0.0f, 0.0f, 0.0f, 0.0f});

        const auto result = lfs::io::load_ply(path);
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsNonFinitePlySphericalHarmonics) {
        const fs::path path = temp_dir_ / "non_finite_sh.ply";
        write_binary_ply(
            path,
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n"
            "end_header\n",
            std::array{0.0f, 0.0f, 0.0f,
                       std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f});

        const auto result = lfs::io::load_ply(path);
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsMalformedColmapTrackPairs) {
        write_text(temp_dir_ / "points3D.txt",
                   "1 0 0 0 255 255 255 0.1 malformed 0\n");

        EXPECT_THROW(
            (void)lfs::io::read_colmap_point_cloud_text_with_stats(temp_dir_),
            std::runtime_error);
    }

    TEST_F(AssertHardeningExpectedFailTest, RejectsNonNormalizedColmapQuaternion) {
        write_text(temp_dir_ / "cameras.txt",
                   "1 PINHOLE 640 480 500 500 320 240\n");
        write_text(temp_dir_ / "images.txt",
                   "1 2 0 0 0 0 0 0 1 frame.png\n\n");

        EXPECT_THROW((void)lfs::io::read_colmap_cameras_only(temp_dir_),
                     std::runtime_error);
    }

    TEST(AssertHardeningRegression, MaskedSelectPreservesInt64Exactly) {
        auto values = Tensor::empty({3}, Device::CPU, DataType::Int64);
        auto* data = values.ptr<int64_t>();
        data[0] = (int64_t{1} << 54) + 1;
        data[1] = -((int64_t{1} << 55) + 3);
        data[2] = 7;
        const auto mask = Tensor::from_vector({true, true, false}, {3}, Device::CPU);

        const auto selected = values.masked_select(mask);

        EXPECT_EQ(selected.dtype(), DataType::Int64);
        EXPECT_EQ(selected.to_vector_int64(),
                  (std::vector<int64_t>{data[0], data[1]}));
    }

    TEST(AssertHardeningRegression, AbiTripwireMatchesOnlyCurrentCore) {
        EXPECT_STREQ(lfs_core_abi_stamp(), LFS_CORE_ABI_STAMP);
        EXPECT_TRUE(lfs_core_abi_matches(LFS_CORE_ABI_STAMP));
        EXPECT_FALSE(lfs_core_abi_matches("stale-core-build"));
        EXPECT_FALSE(lfs_core_abi_matches(nullptr));
    }

    TEST(AssertHardeningRegression, CudaRowProxyPreservesInt64Exactly) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }

        auto cpu = Tensor::empty({2}, Device::CPU, DataType::Int64);
        auto* values = cpu.ptr<int64_t>();
        values[0] = (int64_t{1} << 54) + 1;
        values[1] = -((int64_t{1} << 55) + 3);

        const auto cuda = cpu.to(Device::CUDA);

        EXPECT_EQ(cuda[0].item_int64(), values[0]);
        EXPECT_EQ(cuda[1].item_int64(), values[1]);
    }

    TEST(AssertHardeningRegression, ZeroLengthMatrixFactoriesDoNotPoisonCuda) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        const auto empty = Tensor::empty({0}, Device::CUDA);
        const auto diagonal = Tensor::diag(empty);
        const auto identity = Tensor::eye(0, Device::CUDA);

        EXPECT_EQ(diagonal.shape(), lfs::core::TensorShape({0, 0}));
        EXPECT_EQ(identity.shape(), lfs::core::TensorShape({0, 0}));
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

    TEST(AssertHardeningRegression, CpuBooleanSumUsesInt64Storage) {
        const auto values =
            Tensor::from_vector(std::vector<bool>{true, false, true}, {3}, Device::CPU);

        const auto sum = values.sum();

        EXPECT_EQ(sum.dtype(), DataType::Int64);
        EXPECT_EQ(sum.item<int64_t>(), 2);
    }

    TEST(AssertHardeningRegression, TensorSerializationUsesUntypedByteAccess) {
        const auto source = Tensor::from_vector({1.0f, -2.5f, 3.25f}, {3}, Device::CPU);
        std::stringstream stream;

        stream << source;
        Tensor loaded;
        stream >> loaded;

        EXPECT_EQ(loaded.dtype(), DataType::Float32);
        EXPECT_EQ(loaded.shape(), source.shape());
        EXPECT_EQ(loaded.to_vector(), source.to_vector());
    }

} // namespace
