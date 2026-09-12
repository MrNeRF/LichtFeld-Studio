/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/bilateral_grid.hpp"
#include "core/tensor.hpp"
#include "tensor_hardening_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::BilateralGrid;
    using lfs::training::BilateralGridParameterization;

    class BilateralGridExposureChromaTest : public tensor_hardening::CudaTest {};

    std::vector<float> cpu_copy(const Tensor& tensor) {
        return tensor.cpu().contiguous().to_vector();
    }

    Tensor random_image(const lfs::core::TensorShape& shape, const float seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.15f + 0.7f * std::fmod(seed * 0.173f * static_cast<float>(i + 1), 1.0f);
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    float max_abs_diff(const Tensor& a, const Tensor& b) {
        const auto da = cpu_copy(a);
        const auto db = cpu_copy(b);
        float max_diff = 0.0f;
        for (size_t i = 0; i < da.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(da[i] - db[i]));
        }
        return max_diff;
    }

    float relative_error(const float analytical, const float numerical) {
        return std::abs(analytical - numerical) /
               std::max({std::abs(analytical), std::abs(numerical), 1e-6f});
    }

    TEST_F(BilateralGridExposureChromaTest, ZeroGridIsIdentityHWC) {
        BilateralGrid grid(1, 2, 2, 2, 10, {}, BilateralGridParameterization::ExposureChroma);
        const auto image = random_image({6, 6, 3}, 1.0f);
        const auto out = grid.apply(image, 0);
        EXPECT_LE(max_abs_diff(out, image), 1e-5f);
    }

    TEST_F(BilateralGridExposureChromaTest, ZeroGridIsIdentityCHW) {
        BilateralGrid grid(1, 2, 2, 2, 10, {}, BilateralGridParameterization::ExposureChroma);
        const auto image = random_image({3, 6, 6}, 2.0f);
        const auto out = grid.apply(image, 0);
        EXPECT_LE(max_abs_diff(out, image), 1e-5f);
    }

    void fill_nonzero_grid(BilateralGrid& grid) {
        auto host = cpu_copy(grid.grids());
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = 0.15f * std::sin(0.31f * static_cast<float>(i + 1));
        }
        grid.grids().copy_from(Tensor::from_vector(host, grid.grids().shape(), Device::GPU));
    }

    void check_finite_difference(BilateralGrid& grid, const Tensor& image,
                                 const float rel_tol = 1e-2f, const float abs_tol = 2.5e-3f) {
        const auto output = grid.apply(image, 0);
        const auto grad_output = output * 2.0f;
        grid.zero_grad();
        const auto grad_rgb = grid.backward(image, grad_output, 0);
        const auto grad_grid = cpu_copy(grid.grad_slice());
        const auto rgb_grad = cpu_copy(grad_rgb);
        auto grid_host = cpu_copy(grid.grids());
        auto image_host = cpu_copy(image);
        constexpr float kEps = 1e-3f;

        auto set_grid = [&](const std::vector<float>& values) {
            grid.grids().copy_from(Tensor::from_vector(values, grid.grids().shape(), Device::GPU));
        };
        auto loss_of = [&](const Tensor& img) {
            const auto out = cpu_copy(grid.apply(img, 0));
            double loss = 0.0;
            for (float v : out)
                loss += static_cast<double>(v) * static_cast<double>(v);
            return static_cast<float>(loss);
        };

        for (size_t i = 0; i < grid_host.size(); ++i) {
            auto plus = grid_host;
            auto minus = grid_host;
            plus[i] += kEps;
            minus[i] -= kEps;
            set_grid(plus);
            const float lp = loss_of(image);
            set_grid(minus);
            const float lm = loss_of(image);
            set_grid(grid_host);
            const float numerical = (lp - lm) / (2.0f * kEps);
            const float abs_diff = std::abs(grad_grid[i] - numerical);
            if (std::max(std::abs(grad_grid[i]), std::abs(numerical)) < 1e-3f && abs_diff < 1e-3f)
                continue;
            EXPECT_TRUE(relative_error(grad_grid[i], numerical) < rel_tol || abs_diff < abs_tol)
                << "grid index " << i << " analytical=" << grad_grid[i]
                << " numerical=" << numerical;
        }

        for (size_t i = 0; i < image_host.size(); ++i) {
            auto plus = image_host;
            auto minus = image_host;
            plus[i] += kEps;
            minus[i] -= kEps;
            const auto img_plus = Tensor::from_vector(plus, image.shape(), Device::GPU);
            const auto img_minus = Tensor::from_vector(minus, image.shape(), Device::GPU);
            const float numerical = (loss_of(img_plus) - loss_of(img_minus)) / (2.0f * kEps);
            const float abs_diff = std::abs(rgb_grad[i] - numerical);
            if (std::max(std::abs(rgb_grad[i]), std::abs(numerical)) < 1e-3f && abs_diff < 1e-3f)
                continue;
            EXPECT_TRUE(relative_error(rgb_grad[i], numerical) < rel_tol || abs_diff < abs_tol)
                << "rgb index " << i << " analytical=" << rgb_grad[i]
                << " numerical=" << numerical;
        }
    }

    TEST_F(BilateralGridExposureChromaTest, FiniteDifferenceHWC) {
        BilateralGrid grid(1, 2, 2, 2, 10, {}, BilateralGridParameterization::ExposureChroma);
        fill_nonzero_grid(grid);
        check_finite_difference(grid, random_image({6, 6, 3}, 3.0f));
    }

    TEST_F(BilateralGridExposureChromaTest, FiniteDifferenceCHW) {
        BilateralGrid grid(1, 2, 2, 2, 10, {}, BilateralGridParameterization::ExposureChroma);
        fill_nonzero_grid(grid);
        check_finite_difference(grid, random_image({3, 6, 6}, 4.0f));
    }

    TEST_F(BilateralGridExposureChromaTest, NegativeNeutralLatentStaysFiniteAndMatchesFD) {
        BilateralGrid grid(1, 2, 2, 2, 10, {}, BilateralGridParameterization::ExposureChroma);
        auto host = cpu_copy(grid.grids());
        const int cells = 2 * 2 * 2;
        for (int cell = 0; cell < cells; ++cell) {
            host[static_cast<size_t>(7 * cells + cell)] = -8.0f;
            host[static_cast<size_t>(8 * cells + cell)] = -8.0f;
        }
        grid.grids().copy_from(Tensor::from_vector(host, grid.grids().shape(), Device::GPU));

        std::vector<float> rgb(3 * 4 * 4, 1.0f);
        const auto image = Tensor::from_vector(rgb, {3, 4, 4}, Device::GPU);
        const auto out = grid.apply(image, 0);
        const auto out_host = cpu_copy(out);
        for (float v : out_host) {
            EXPECT_TRUE(std::isfinite(v));
        }
        // No sign-flip inversion of a positive input into large negatives.
        float min_v = out_host[0];
        for (float v : out_host)
            min_v = std::min(min_v, v);
        EXPECT_GT(min_v, -1.0f);

        // Clamp at z<=0 makes the VJP piecewise; float32 FD at the kink is a few
        // parts in 1e-2, so allow a slightly looser floor than the smooth case.
        check_finite_difference(grid, image, 3e-2f, 5e-3f);
    }

} // namespace
