/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>
#include <torch/torch.h>

using namespace lfs::core;
namespace fs = std::filesystem;

class TensorSerializationTest : public ::testing::Test {
protected:
    fs::path temp_dir_;

    void SetUp() override {
        temp_dir_ = fs::temp_directory_path() / "tensor_serialization_test";
        fs::create_directories(temp_dir_);
        Tensor::manual_seed(42);
    }

    void TearDown() override { fs::remove_all(temp_dir_); }

    std::string temp_file(const std::string& name) const { return (temp_dir_ / name).string(); }

    template <typename T>
    void check_exact(const Tensor& a, const Tensor& b) const {
        ASSERT_EQ(a.numel(), b.numel());
        const auto a_cpu = a.device() == Device::CPU ? a : a.cpu();
        const auto b_cpu = b.device() == Device::CPU ? b : b.cpu();
        for (size_t i = 0; i < a.numel(); ++i) {
            EXPECT_EQ(a_cpu.ptr<T>()[i], b_cpu.ptr<T>()[i]);
        }
    }

    void check_float(const Tensor& a, const Tensor& b, float tol = 0) const {
        ASSERT_EQ(a.numel(), b.numel());
        const auto av = a.cpu().to_vector();
        const auto bv = b.cpu().to_vector();
        for (size_t i = 0; i < av.size(); ++i) {
            EXPECT_LE(std::abs(av[i] - bv[i]), tol + 1e-7f);
        }
    }
};

TEST_F(TensorSerializationTest, Float32) {
    const auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, Device::CPU);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    check_float(t, loaded);
}

TEST_F(TensorSerializationTest, Int32) {
    const auto t = Tensor::from_vector({1, 2, 3, 4}, {2, 2}, Device::CPU);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    check_exact<int32_t>(t, loaded);
}

TEST_F(TensorSerializationTest, Bool) {
    const auto t = Tensor::from_vector({true, false, true, false}, {2, 2}, Device::CPU);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    check_exact<uint8_t>(t, loaded);
}

TEST_F(TensorSerializationTest, Scalar) {
    const auto t = Tensor::full({}, 42.0f, Device::CPU);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    EXPECT_EQ(loaded.ndim(), 0);
    EXPECT_EQ(loaded.numel(), 1);
    check_float(t, loaded);
}

TEST_F(TensorSerializationTest, HighDimensional) {
    const auto t = Tensor::randn({2, 3, 4, 5}, Device::CPU);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    EXPECT_EQ(loaded.ndim(), 4);
    check_float(t, loaded, 1e-5f);
}

TEST_F(TensorSerializationTest, Empty) {
    const auto t = Tensor::empty({0}, Device::CPU);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    EXPECT_EQ(loaded.numel(), 0);
}

TEST_F(TensorSerializationTest, Int64) {
    auto t = Tensor::empty({4}, Device::CPU, DataType::Int64);
    for (size_t i = 0; i < 4; ++i)
        t.ptr<int64_t>()[i] = static_cast<int64_t>(i) * 1000000000LL;
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    check_exact<int64_t>(t, loaded);
}

TEST_F(TensorSerializationTest, UInt8) {
    auto t = Tensor::empty({16}, Device::CPU, DataType::UInt8);
    for (size_t i = 0; i < 16; ++i)
        t.ptr<uint8_t>()[i] = static_cast<uint8_t>(i);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    check_exact<uint8_t>(t, loaded);
}

TEST_F(TensorSerializationTest, CudaTensor) {
    const auto t = Tensor::randn({32, 32}, Device::CUDA);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    EXPECT_EQ(loaded.device(), Device::CPU);
    check_float(t, loaded, 1e-5f);
}

TEST_F(TensorSerializationTest, FileIO) {
    const auto t = Tensor::randn({64, 64}, Device::CUDA);
    save_tensor(t, temp_file("t.lft"));
    const auto loaded = load_tensor(temp_file("t.lft"));
    check_float(t, loaded, 1e-5f);
}

TEST_F(TensorSerializationTest, MultipleTensors) {
    const auto t1 = Tensor::randn({8, 8}, Device::CPU);
    const auto t2 = Tensor::randint({4, 4}, 0, 100, Device::CPU);
    std::ofstream ofs(temp_file("multi.lft"), std::ios::binary);
    ofs << t1 << t2;
    ofs.close();
    std::ifstream ifs(temp_file("multi.lft"), std::ios::binary);
    Tensor l1, l2;
    ifs >> l1 >> l2;
    check_float(t1, l1, 1e-6f);
    check_exact<int32_t>(t2, l2);
}

TEST_F(TensorSerializationTest, SpecialFloatValues) {
    auto t = Tensor::empty({4}, Device::CPU);
    t.ptr<float>()[0] = 0.0f;
    t.ptr<float>()[1] = std::numeric_limits<float>::infinity();
    t.ptr<float>()[2] = -std::numeric_limits<float>::infinity();
    t.ptr<float>()[3] = std::numeric_limits<float>::max();
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    EXPECT_EQ(loaded.ptr<float>()[0], 0.0f);
    EXPECT_TRUE(std::isinf(loaded.ptr<float>()[1]) && loaded.ptr<float>()[1] > 0);
    EXPECT_TRUE(std::isinf(loaded.ptr<float>()[2]) && loaded.ptr<float>()[2] < 0);
}

TEST_F(TensorSerializationTest, NaN) {
    auto t = Tensor::empty({1}, Device::CPU);
    t.ptr<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    EXPECT_TRUE(std::isnan(loaded.ptr<float>()[0]));
}

TEST_F(TensorSerializationTest, LargeTensor) {
    const auto t = Tensor::randn({1000, 1000}, Device::CUDA);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    check_float(t, loaded, 1e-5f);
}

TEST_F(TensorSerializationTest, InvalidMagicNumber) {
    std::stringstream ss;
    const uint32_t bad = 0xDEADBEEF;
    ss.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
    Tensor loaded;
    EXPECT_THROW(ss >> loaded, std::runtime_error);
}

TEST_F(TensorSerializationTest, RejectsUnsupportedDtypeWithoutMutatingDestination) {
    std::stringstream ss;
    const TensorFileHeader header{
        .magic = TENSOR_FILE_MAGIC,
        .version = TENSOR_FILE_VERSION,
        .dtype = 0xff,
        .device = static_cast<uint8_t>(Device::CPU),
        .rank = 1,
        .numel = 1,
    };
    ss.write(reinterpret_cast<const char*>(&header), sizeof(header));

    Tensor destination = Tensor::from_vector({42.0f}, {1}, Device::CPU);
    EXPECT_THROW(ss >> destination, std::runtime_error);
    ASSERT_TRUE(destination.is_valid());
    EXPECT_FLOAT_EQ(destination.ptr<float>()[0], 42.0f);
}

TEST_F(TensorSerializationTest, RejectsExcessiveRankBeforeReadingDimensions) {
    std::stringstream ss;
    const TensorFileHeader header{
        .magic = TENSOR_FILE_MAGIC,
        .version = TENSOR_FILE_VERSION,
        .dtype = static_cast<uint8_t>(DataType::Float32),
        .device = static_cast<uint8_t>(Device::CPU),
        .rank = static_cast<uint16_t>(MAX_SERIALIZED_TENSOR_RANK + 1),
        .numel = 1,
    };
    ss.write(reinterpret_cast<const char*>(&header), sizeof(header));

    Tensor loaded;
    EXPECT_THROW(ss >> loaded, std::runtime_error);
    EXPECT_FALSE(loaded.is_valid());
}

TEST_F(TensorSerializationTest, RejectsPayloadOverBudgetBeforeAllocation) {
    std::stringstream ss;
    const uint64_t numel = MAX_SERIALIZED_TENSOR_BYTES / sizeof(float) + 1;
    const TensorFileHeader header{
        .magic = TENSOR_FILE_MAGIC,
        .version = TENSOR_FILE_VERSION,
        .dtype = static_cast<uint8_t>(DataType::Float32),
        .device = static_cast<uint8_t>(Device::CPU),
        .rank = 1,
        .numel = numel,
    };
    ss.write(reinterpret_cast<const char*>(&header), sizeof(header));
    ss.write(reinterpret_cast<const char*>(&numel), sizeof(numel));

    Tensor loaded;
    EXPECT_THROW(ss >> loaded, std::runtime_error);
    EXPECT_FALSE(loaded.is_valid());
}

TEST_F(TensorSerializationTest, RejectsTruncatedPayloadBeforeAllocation) {
    std::stringstream ss;
    constexpr uint64_t numel = 4;
    const TensorFileHeader header{
        .magic = TENSOR_FILE_MAGIC,
        .version = TENSOR_FILE_VERSION,
        .dtype = static_cast<uint8_t>(DataType::Float32),
        .device = static_cast<uint8_t>(Device::CPU),
        .rank = 1,
        .numel = numel,
    };
    constexpr float partial_payload = 1.0f;
    ss.write(reinterpret_cast<const char*>(&header), sizeof(header));
    ss.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
    ss.write(reinterpret_cast<const char*>(&partial_payload), sizeof(partial_payload));

    Tensor loaded;
    EXPECT_THROW(ss >> loaded, std::runtime_error);
    EXPECT_FALSE(loaded.is_valid());
}

TEST_F(TensorSerializationTest, NonexistentFile) {
    EXPECT_THROW(load_tensor("/nonexistent/path/tensor.lft"), std::runtime_error);
}

TEST_F(TensorSerializationTest, InvalidTensor) {
    std::stringstream ss;
    EXPECT_THROW(ss << Tensor(), std::runtime_error);
}

TEST_F(TensorSerializationTest, Transpose) {
    const auto base = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, Device::CPU);
    const auto t = base.t();
    EXPECT_FALSE(t.is_contiguous());
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    EXPECT_TRUE(loaded.is_contiguous());
    EXPECT_EQ(loaded.size(0), 3);
    EXPECT_EQ(loaded.size(1), 2);
    const std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    const auto v = loaded.to_vector();
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(v[i], expected[i]);
}

TEST_F(TensorSerializationTest, Slice) {
    const auto base = Tensor::arange(100.0f).reshape({10, 10}).cpu();
    const auto sliced = base.slice(0, 2, 7);
    std::stringstream ss;
    ss << sliced;
    Tensor loaded;
    ss >> loaded;
    EXPECT_EQ(loaded.size(0), 5);
    check_float(sliced.contiguous(), loaded);
}

TEST_F(TensorSerializationTest, RoundTripCuda) {
    const auto t = Tensor::randn({64, 64}, Device::CUDA);
    std::stringstream ss;
    ss << t;
    Tensor loaded;
    ss >> loaded;
    check_float(t, loaded.cuda(), 1e-5f);
}

TEST_F(TensorSerializationTest, ScalarMatchesLibTorch) {
    const auto torch_scalar = torch::full({}, 42.0f);
    const auto lfs_scalar = Tensor::full({}, 42.0f, Device::CPU);
    EXPECT_EQ(lfs_scalar.ndim(), static_cast<size_t>(torch_scalar.dim()));
    EXPECT_EQ(lfs_scalar.numel(), static_cast<size_t>(torch_scalar.numel()));
}
