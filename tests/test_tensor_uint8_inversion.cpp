/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>

using namespace lfs::core;

class TensorUInt8InversionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(Tensor::zeros({1}, Device::GPU).is_valid());
    }

    // Helper to create test mask [1,0,1,0,1]
    static Tensor createTestMask(Device device) {
        auto mask = Tensor::zeros({5}, Device::CPU, DataType::UInt8);
        auto* p = mask.ptr<uint8_t>();
        p[0] = 1;
        p[1] = 0;
        p[2] = 1;
        p[3] = 0;
        p[4] = 1;
        return device == Device::GPU ? mask.cuda() : mask;
    }

    // Verify inverted pattern [0,1,0,1,0]
    template <typename T>
    static void verifyInverted(const T* r) {
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 1);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(r[4], 0);
    }
};

TEST_F(TensorUInt8InversionTest, SubtractionCPU) {
    const auto mask = createTestMask(Device::CPU);
    const auto ones = Tensor::ones({5}, Device::CPU, DataType::UInt8);
    const auto inverted = ones - mask;

    ASSERT_EQ(inverted.dtype(), DataType::UInt8);
    verifyInverted(inverted.ptr<uint8_t>());
}

TEST_F(TensorUInt8InversionTest, SubtractionCUDA) {
    const auto mask = createTestMask(Device::GPU);
    const auto ones = Tensor::ones({5}, Device::GPU, DataType::UInt8);
    const auto inverted = (ones - mask).cpu();

    ASSERT_EQ(inverted.dtype(), DataType::UInt8);
    verifyInverted(inverted.ptr<uint8_t>());
}

TEST_F(TensorUInt8InversionTest, LogicalNot) {
    const auto mask = createTestMask(Device::GPU);
    const auto inverted = mask.logical_not().cpu();

    ASSERT_EQ(inverted.dtype(), DataType::Bool);
    const auto* r = inverted.ptr<bool>();
    EXPECT_FALSE(r[0]);
    EXPECT_TRUE(r[1]);
    EXPECT_FALSE(r[2]);
    EXPECT_TRUE(r[3]);
    EXPECT_FALSE(r[4]);
}

TEST_F(TensorUInt8InversionTest, EqZero) {
    const auto mask = createTestMask(Device::GPU);
    const auto eq_result = mask.eq(0);

    ASSERT_EQ(eq_result.dtype(), DataType::Bool);
    const auto cpu = eq_result.cpu();
    const auto* r = cpu.ptr<bool>();
    EXPECT_FALSE(r[0]);
    EXPECT_TRUE(r[1]);
    EXPECT_FALSE(r[2]);
    EXPECT_TRUE(r[3]);
    EXPECT_FALSE(r[4]);
}

TEST_F(TensorUInt8InversionTest, BoolToUInt8Conversion) {
    const auto mask = createTestMask(Device::GPU);
    const auto inverted = mask.eq(0).to(DataType::UInt8).cpu();

    ASSERT_EQ(inverted.dtype(), DataType::UInt8);
    verifyInverted(inverted.ptr<uint8_t>());
}

TEST_F(TensorUInt8InversionTest, LargeScale) {
    constexpr size_t N = 1000000;

    auto mask = Tensor::zeros({N}, Device::CPU, DataType::UInt8);
    auto* p = mask.ptr<uint8_t>();
    for (size_t i = 0; i < N; i += 2)
        p[i] = 1;
    mask = mask.cuda();

    const auto ones = Tensor::ones({N}, Device::GPU, DataType::UInt8);
    const auto inverted = ones - mask;
    const float sum = inverted.to(DataType::Float32).sum_scalar();

    EXPECT_NEAR(sum, N / 2.0f, 1.0f);
}
