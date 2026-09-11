/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/bilateral_grid.hpp"
#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/bilateral_grid.cuh"
#include "tensor_hardening_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::BilateralGrid;
    using lfs::training::BilateralGridParameterization;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_chw;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_chw_reference;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_exposure_chroma_chw;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_exposure_chroma_reference;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_reference;

    class BilateralGridBackwardTiledTest : public tensor_hardening::CudaTest {};

    std::vector<float> cpu_copy(const Tensor& tensor) {
        return tensor.cpu().contiguous().to_vector();
    }

    void fill_grid(BilateralGrid& grid, const float seed) {
        auto host = cpu_copy(grid.grids());
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] += 0.08f * std::sin(seed * 0.17f * static_cast<float>(i + 1));
        }
        grid.grids().copy_from(Tensor::from_vector(host, grid.grids().shape(), Device::CUDA));
        grid.project_mean(grid.parameterization() == BilateralGridParameterization::ExposureChroma);
    }

    Tensor random_image(const lfs::core::TensorShape& shape, const float seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.12f + 0.76f * std::fmod(seed * 0.173f * static_cast<float>(i + 1), 1.0f);
        }
        return Tensor::from_vector(values, shape, Device::CUDA);
    }

    Tensor random_grad(const lfs::core::TensorShape& shape, const float seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.2f * std::sin(seed * 0.31f * static_cast<float>(i + 3));
        }
        return Tensor::from_vector(values, shape, Device::CUDA);
    }

    void expect_rel_near(const std::vector<float>& got,
                         const std::vector<float>& ref,
                         const float rel,
                         const std::string& ctx) {
        ASSERT_EQ(got.size(), ref.size()) << ctx;
        // Shared-memory tiling changes the order of float32 atomicAdds into each
        // grid cell. Keep a 1e-4 relative bound, plus an absolute floor so
        // near-zero cancellations (and ~1e-6 atomic-order residuals on a real
        // image) are not judged on relative error alone. Bicycle images_4
        // peaks around 4e-6 abs; 1e-6 is too tight.
        constexpr float kAbsFloor = 1e-5f;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float abs_diff = std::abs(got[i] - ref[i]);
            const float scale = std::max({std::abs(ref[i]), std::abs(got[i]), 1.0e-6f});
            EXPECT_TRUE(abs_diff / scale <= rel || abs_diff <= kAbsFloor)
                << ctx << " index " << i << " got=" << got[i] << " ref=" << ref[i]
                << " abs=" << abs_diff << " rel=" << (abs_diff / scale);
        }
    }

    using ReferenceLauncher = void (*)(const float*, const float*, const float*, float*, float*,
                                       int, int, int, int, int, const float*, cudaStream_t);

    void compare_new_vs_reference(BilateralGrid& grid,
                                  const Tensor& image,
                                  const Tensor& grad_output,
                                  const bool chw,
                                  ReferenceLauncher reference,
                                  const std::string& ctx) {
        grid.zero_grad();
        const auto grad_rgb_new = grid.backward(image, grad_output, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<float> new_grid = cpu_copy(grid.grad_slice());
        const std::vector<float> new_rgb = cpu_copy(grad_rgb_new);

        auto grad_grid_ref = Tensor::zeros(grid.grad_slice().shape(), Device::CUDA);
        auto grad_rgb_ref = Tensor::empty(image.shape(), Device::CUDA);
        const auto rgb_cont = image.contiguous();
        const auto gout_cont = grad_output.contiguous();
        const int h = chw ? static_cast<int>(image.shape()[1]) : static_cast<int>(image.shape()[0]);
        const int w = chw ? static_cast<int>(image.shape()[2]) : static_cast<int>(image.shape()[1]);
        reference(grid.grids().ptr<float>(),
                  rgb_cont.ptr<float>(),
                  gout_cont.ptr<float>(),
                  grad_grid_ref.ptr<float>(),
                  grad_rgb_ref.ptr<float>(),
                  grid.grid_guidance(), grid.grid_height(), grid.grid_width(),
                  h, w, grid.shared_offset().ptr<float>(), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        expect_rel_near(new_grid, cpu_copy(grad_grid_ref), 1e-4f, ctx + " grad_grid");
        expect_rel_near(new_rgb, cpu_copy(grad_rgb_ref), 1e-4f, ctx + " grad_rgb");
    }

    Tensor load_bicycle_hwc() {
        const std::filesystem::path path =
            std::filesystem::path(TEST_DATA_DIR) / "bicycle/images_4/_DSC8679.JPG";
        auto [pixels, width, height, channels] = lfs::core::load_image(path);
        if (pixels == nullptr || width <= 0 || height <= 0 || channels < 3) {
            if (pixels)
                lfs::core::free_image(pixels);
            return {};
        }
        std::vector<float> hwc(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int src = (y * width + x) * channels;
                const int dst = (y * width + x) * 3;
                hwc[static_cast<size_t>(dst + 0)] = static_cast<float>(pixels[src + 0]) / 255.0f;
                hwc[static_cast<size_t>(dst + 1)] = static_cast<float>(pixels[src + 1]) / 255.0f;
                hwc[static_cast<size_t>(dst + 2)] = static_cast<float>(pixels[src + 2]) / 255.0f;
            }
        }
        lfs::core::free_image(pixels);
        return Tensor::from_vector(
            hwc,
            {static_cast<size_t>(height), static_cast<size_t>(width), size_t{3}},
            Device::CUDA);
    }

    Tensor hwc_to_chw(const Tensor& hwc) {
        const int h = static_cast<int>(hwc.shape()[0]);
        const int w = static_cast<int>(hwc.shape()[1]);
        const auto host = cpu_copy(hwc);
        std::vector<float> chw(static_cast<size_t>(3 * h * w));
        const int hw = h * w;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int pix = y * w + x;
                chw[static_cast<size_t>(0 * hw + pix)] = host[static_cast<size_t>(pix * 3 + 0)];
                chw[static_cast<size_t>(1 * hw + pix)] = host[static_cast<size_t>(pix * 3 + 1)];
                chw[static_cast<size_t>(2 * hw + pix)] = host[static_cast<size_t>(pix * 3 + 2)];
            }
        }
        return Tensor::from_vector(
            chw, {size_t{3}, static_cast<size_t>(h), static_cast<size_t>(w)}, Device::CUDA);
    }

    TEST_F(BilateralGridBackwardTiledTest, RandomImageMatchesReferenceBothLayouts) {
        {
            BilateralGrid grid(1, 16, 16, 8, 20);
            fill_grid(grid, 1.3f);
            const auto image = random_image({256, 192, 3}, 2.0f);
            const auto gout = random_grad(image.shape(), 3.0f);
            compare_new_vs_reference(grid, image, gout, false,
                                     launch_bilateral_grid_slice_backward_reference,
                                     "affine HWC random");
        }
        {
            BilateralGrid grid(1, 16, 16, 8, 20);
            fill_grid(grid, 1.7f);
            const auto image = random_image({3, 256, 192}, 4.0f);
            const auto gout = random_grad(image.shape(), 5.0f);
            compare_new_vs_reference(grid, image, gout, true,
                                     launch_bilateral_grid_slice_backward_chw_reference,
                                     "affine CHW random");
        }
        {
            BilateralGrid grid(1, 16, 16, 8, 20, {}, BilateralGridParameterization::ExposureChroma);
            fill_grid(grid, 2.1f);
            const auto image = random_image({256, 192, 3}, 6.0f);
            const auto gout = random_grad(image.shape(), 7.0f);
            compare_new_vs_reference(grid, image, gout, false,
                                     launch_bilateral_grid_slice_backward_exposure_chroma_reference,
                                     "chroma HWC random");
        }
        {
            BilateralGrid grid(1, 16, 16, 8, 20, {}, BilateralGridParameterization::ExposureChroma);
            fill_grid(grid, 2.5f);
            const auto image = random_image({3, 256, 192}, 8.0f);
            const auto gout = random_grad(image.shape(), 9.0f);
            compare_new_vs_reference(grid, image, gout, true,
                                     launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference,
                                     "chroma CHW random");
        }
    }

    TEST_F(BilateralGridBackwardTiledTest, BicycleImageMatchesReferenceBothParameterizations) {
        const auto hwc = load_bicycle_hwc();
        if (!hwc.is_valid() || hwc.numel() == 0) {
            GTEST_SKIP() << "missing data/bicycle/images_4/_DSC8679.JPG";
        }
        const auto chw = hwc_to_chw(hwc);
        const auto gout_hwc = random_grad(hwc.shape(), 11.0f);
        const auto gout_chw = hwc_to_chw(gout_hwc);

        {
            BilateralGrid grid(1, 16, 16, 8, 20);
            fill_grid(grid, 3.1f);
            compare_new_vs_reference(grid, hwc, gout_hwc, false,
                                     launch_bilateral_grid_slice_backward_reference,
                                     "affine HWC bicycle");
            compare_new_vs_reference(grid, chw, gout_chw, true,
                                     launch_bilateral_grid_slice_backward_chw_reference,
                                     "affine CHW bicycle");
        }
        {
            BilateralGrid grid(1, 16, 16, 8, 20, {}, BilateralGridParameterization::ExposureChroma);
            fill_grid(grid, 3.7f);
            compare_new_vs_reference(grid, hwc, gout_hwc, false,
                                     launch_bilateral_grid_slice_backward_exposure_chroma_reference,
                                     "chroma HWC bicycle");
            compare_new_vs_reference(grid, chw, gout_chw, true,
                                     launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference,
                                     "chroma CHW bicycle");
        }
    }

    template <typename Launch>
    float cuda_event_ms(Launch&& launch, const int warmup, const int iters) {
        for (int i = 0; i < warmup; ++i)
            launch();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        EXPECT_EQ(cudaEventCreate(&start), cudaSuccess);
        EXPECT_EQ(cudaEventCreate(&stop), cudaSuccess);
        EXPECT_EQ(cudaEventRecord(start), cudaSuccess);
        for (int i = 0; i < iters; ++i)
            launch();
        EXPECT_EQ(cudaEventRecord(stop), cudaSuccess);
        EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);
        float ms = 0.0f;
        EXPECT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms / static_cast<float>(iters);
    }

    TEST_F(BilateralGridBackwardTiledTest, MicroBenchChw1237x822) {
        constexpr int kH = 822;
        constexpr int kW = 1237;
        auto hwc = load_bicycle_hwc();
        Tensor chw;
        Tensor gout;
        if (hwc.is_valid() && hwc.numel() > 0 &&
            static_cast<int>(hwc.shape()[0]) == kH &&
            static_cast<int>(hwc.shape()[1]) == kW) {
            chw = hwc_to_chw(hwc);
            gout = hwc_to_chw(random_grad(hwc.shape(), 13.0f));
        } else {
            chw = random_image({3, kH, kW}, 13.0f);
            gout = random_grad(chw.shape(), 17.0f);
        }

        auto run_param = [&](BilateralGridParameterization param, const char* name) {
            BilateralGrid grid(1, 16, 16, 8, 20, {}, param);
            fill_grid(grid, 4.1f);
            auto grad_grid = Tensor::zeros(grid.grad_slice().shape(), Device::CUDA);
            auto grad_rgb = Tensor::empty(chw.shape(), Device::CUDA);
            const float* grid_ptr = grid.grids().ptr<float>();
            const float* rgb_ptr = chw.ptr<float>();
            const float* gout_ptr = gout.ptr<float>();
            float* gg_ptr = grad_grid.ptr<float>();
            float* gr_ptr = grad_rgb.ptr<float>();
            const float* off = grid.shared_offset().ptr<float>();
            const int L = grid.grid_guidance();
            const int H = grid.grid_height();
            const int W = grid.grid_width();

            const auto ref = [&]() {
                if (param == BilateralGridParameterization::ExposureChroma) {
                    launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference(
                        grid_ptr, rgb_ptr, gout_ptr, gg_ptr, gr_ptr, L, H, W, kH, kW, off, nullptr);
                } else {
                    launch_bilateral_grid_slice_backward_chw_reference(
                        grid_ptr, rgb_ptr, gout_ptr, gg_ptr, gr_ptr, L, H, W, kH, kW, off, nullptr);
                }
            };
            const auto tiled = [&]() {
                if (param == BilateralGridParameterization::ExposureChroma) {
                    launch_bilateral_grid_slice_backward_exposure_chroma_chw(
                        grid_ptr, rgb_ptr, gout_ptr, gg_ptr, gr_ptr, L, H, W, kH, kW, off, nullptr, false);
                } else {
                    launch_bilateral_grid_slice_backward_chw(
                        grid_ptr, rgb_ptr, gout_ptr, gg_ptr, gr_ptr, L, H, W, kH, kW, off, nullptr, false);
                }
            };
            const auto agg = [&]() {
                if (param == BilateralGridParameterization::ExposureChroma) {
                    launch_bilateral_grid_slice_backward_exposure_chroma_chw(
                        grid_ptr, rgb_ptr, gout_ptr, gg_ptr, gr_ptr, L, H, W, kH, kW, off, nullptr, true);
                } else {
                    launch_bilateral_grid_slice_backward_chw(
                        grid_ptr, rgb_ptr, gout_ptr, gg_ptr, gr_ptr, L, H, W, kH, kW, off, nullptr, true);
                }
            };

            const float ref_ms = cuda_event_ms(ref, 3, 20);
            const float tiled_ms = cuda_event_ms(tiled, 3, 20);
            const float agg_ms = cuda_event_ms(agg, 3, 20);
            std::cout << "[BwdMicroBench] " << name << " CHW " << kW << "x" << kH
                      << " scatter=" << ref_ms << " ms  tiled_r3=" << tiled_ms
                      << " ms  warp_agg=" << agg_ms << " ms/call\n";
            EXPECT_GT(ref_ms, 0.0f);
            EXPECT_GT(tiled_ms, 0.0f);
            EXPECT_GT(agg_ms, 0.0f);
        };

        run_param(BilateralGridParameterization::Affine, "affine");
        run_param(BilateralGridParameterization::ExposureChroma, "chroma");
    }

} // namespace
