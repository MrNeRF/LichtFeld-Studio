/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_config.hpp"
#include <gtest/gtest.h>
#include <optional>

using namespace lfs::core;

namespace {

    class LazyModeOverrideGuard {
    public:
        explicit LazyModeOverrideGuard(std::optional<LazyMode> mode) {
            internal::set_lazy_mode_override_for_testing(mode);
        }

        ~LazyModeOverrideGuard() {
            internal::set_lazy_mode_override_for_testing(std::nullopt);
        }
    };

} // namespace

TEST(TensorLazyRuntimeTest, ParseModeString) {
    EXPECT_EQ(internal::parse_lazy_mode_string("off"), LazyMode::Off);
    EXPECT_EQ(internal::parse_lazy_mode_string("shadow"), LazyMode::Shadow);
    EXPECT_EQ(internal::parse_lazy_mode_string("on"), LazyMode::On);

    EXPECT_EQ(internal::parse_lazy_mode_string("  OFF  "), LazyMode::Off);
    EXPECT_EQ(internal::parse_lazy_mode_string(" Compare "), LazyMode::Shadow);
    EXPECT_EQ(internal::parse_lazy_mode_string("TRUE"), LazyMode::On);

    EXPECT_EQ(internal::parse_lazy_mode_string("bogus", LazyMode::Shadow), LazyMode::Shadow);
}

TEST(TensorLazyRuntimeTest, OverrideControlsMode) {
    {
        LazyModeOverrideGuard guard(LazyMode::Off);
        EXPECT_EQ(Tensor::lazy_mode(), LazyMode::Off);
        EXPECT_STREQ(Tensor::lazy_mode_name(), "off");
        EXPECT_FALSE(Tensor::lazy_enabled());
        EXPECT_FALSE(Tensor::lazy_shadow_enabled());
    }

    {
        LazyModeOverrideGuard guard(LazyMode::Shadow);
        EXPECT_EQ(Tensor::lazy_mode(), LazyMode::Shadow);
        EXPECT_STREQ(Tensor::lazy_mode_name(), "shadow");
        EXPECT_FALSE(Tensor::lazy_enabled());
        EXPECT_TRUE(Tensor::lazy_shadow_enabled());
    }

    {
        LazyModeOverrideGuard guard(LazyMode::On);
        EXPECT_EQ(Tensor::lazy_mode(), LazyMode::On);
        EXPECT_STREQ(Tensor::lazy_mode_name(), "on");
        EXPECT_TRUE(Tensor::lazy_enabled());
        EXPECT_FALSE(Tensor::lazy_shadow_enabled());
    }
}

TEST(TensorLazyRuntimeTest, TelemetryTracksMaterializationBytes) {
    Tensor::reset_lazy_telemetry();

    const auto initial = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(initial.materializations, 0u);
    EXPECT_EQ(initial.allocated_bytes, 0u);

    auto a = Tensor::empty_unpinned({2, 3}, DataType::Float32);
    auto b = Tensor::empty_unpinned({8}, DataType::Int32);

    const uint64_t min_expected_bytes = static_cast<uint64_t>(a.bytes() + b.bytes());
    const auto snapshot = Tensor::lazy_telemetry_snapshot();

    EXPECT_GE(snapshot.materializations, 2u);
    EXPECT_GE(snapshot.allocated_bytes, min_expected_bytes);

    Tensor::reset_lazy_telemetry();
    const auto after_reset = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(after_reset.materializations, 0u);
    EXPECT_EQ(after_reset.allocated_bytes, 0u);
}

