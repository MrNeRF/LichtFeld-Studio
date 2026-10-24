/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_backend.hpp"
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

namespace lfs::test {
    struct TensorBackendTarget {
        std::string name;
        std::optional<core::GpuBackend> backend;
    };

    // Enumerate compiled backends without initializing any device at test discovery.
    inline std::vector<TensorBackendTarget> tensor_backend_targets() {
        std::vector<TensorBackendTarget> targets{{"CPU", std::nullopt}};
        for (const auto backend : core::kCompiledGpuBackends)
            targets.push_back({core::gpu_backend_name(backend), backend});
        return targets;
    }

    inline std::string tensor_backend_parameter_name(const testing::TestParamInfo<TensorBackendTarget>& info) {
        return info.param.name;
    }

    class TensorBackendContract : public testing::TestWithParam<TensorBackendTarget> {
    protected:
        void SetUp() override {
            if (const auto backend = GetParam().backend) {
                if (!core::gpu_backend_available(*backend))
                    GTEST_SKIP() << GetParam().name << " device unavailable";
                scope_.emplace(*backend);
            }
        }

        core::Device device() const { return GetParam().backend ? core::Device::GPU : core::Device::CPU; }

        void expect_placement(const core::Tensor& tensor) const {
            EXPECT_EQ(tensor.device(), device());
            EXPECT_EQ(core::gpu_backend_of(tensor), GetParam().backend);
        }

    private:
        // Derived fixture members are destroyed first, while their scope is active.
        std::optional<core::GpuBackendScope> scope_;
    };
} // namespace lfs::test
