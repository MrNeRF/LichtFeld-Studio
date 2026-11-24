/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/bilateral_grid.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/gpu_elapsed.hpp"
#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/kernels/bilateral_grid.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::BilateralGrid;
    using lfs::training::BilateralGridParameterization;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_chw_reference;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_exposure_chroma_reference;
    using lfs::training::kernels::launch_bilateral_grid_slice_backward_reference;

    class BilateralGridBackwardTiledTest : public lfs::test::CudaDeviceTest {};

    std::vector<float> cpu_copy(const Tensor& tensor) {
        return tensor.cpu().contiguous().to_vector();
    }

    void fill_grid(BilateralGrid& grid, const float seed) {
        auto host = cpu_copy(grid.grids());
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] += 0.08f * std::sin(seed * 0.17f * static_cast<float>(i + 1));
        }
        grid.grids().copy_from(Tensor::from_vector(host, grid.grids().shape(), Device::GPU));
        grid.project_mean(grid.parameterization() == BilateralGridParameterization::ExposureChroma);
    }

    Tensor random_image(const lfs::core::TensorShape& shape, const float seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.12f + 0.76f * std::fmod(seed * 0.173f * static_cast<float>(i + 1), 1.0f);
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor random_grad(const lfs::core::TensorShape& shape, const float seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.2f * std::sin(seed * 0.31f * static_cast<float>(i + 3));
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    void expect_rel_near(const std::vector<float>& got,
                         const std::vector<float>& ref,
                         const float rel,
                         const std::string& ctx,
                         const float abs_floor = 1e-5f) {
        ASSERT_EQ(got.size(), ref.size()) << ctx;
        float max_diff = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float abs_diff = std::abs(got[i] - ref[i]);
            max_diff = std::max(max_diff, abs_diff);
            const float scale = std::max({std::abs(ref[i]), std::abs(got[i]), 1.0e-6f});
            EXPECT_TRUE(abs_diff / scale <= rel || abs_diff <= abs_floor)
                << ctx << " index " << i << " got=" << got[i] << " ref=" << ref[i]
                << " abs=" << abs_diff << " rel=" << (abs_diff / scale);
        }
        std::cout << std::format("[bilateral] {} max_abs={:.9g}\n", ctx, max_diff);
    }

    using ReferenceLauncher = void (*)(const float*, const float*, const float*, float*, float*,
                                       int, int, int, int, int, const float*, cudaStream_t);

    // Evaluate the independent scatter reference on its native device, even when the
    // implementation under test uses the Vulkan backend selected on the command line.
    std::pair<std::vector<float>, std::vector<float>> reference_gradients(
        ReferenceLauncher launch, BilateralGrid& grid, const Tensor& image, const Tensor& upstream, int h, int w) {
        const auto convert = [](const Tensor& tensor) {
            return tensor.to(lfs::core::GpuBackend::CUDA).contiguous();
        };
        auto coefficients = convert(grid.grids()), rgb = convert(image), grad = convert(upstream);
        auto offset = convert(grid.shared_offset());
        lfs::core::GpuBackendScope scope(lfs::core::GpuBackend::CUDA);
        auto grid_result = Tensor::zeros(grid.grad_slice().shape(), Device::GPU);
        auto rgb_result = Tensor::empty(image.shape(), Device::GPU);
        launch(coefficients.ptr<float>(), rgb.ptr<float>(), grad.ptr<float>(), grid_result.ptr<float>(), rgb_result.ptr<float>(),
               grid.grid_guidance(), grid.grid_height(), grid.grid_width(), h, w, offset.ptr<float>(), nullptr);
        return {cpu_copy(grid_result), cpu_copy(rgb_result)};
    }

    void compare_new_vs_reference(BilateralGrid& grid,
                                  const Tensor& image,
                                  const Tensor& grad_output,
                                  const bool chw,
                                  ReferenceLauncher reference,
                                  const std::string& ctx,
                                  const float grid_abs_floor = 1e-5f) {
        grid.zero_grad();
        const auto grad_rgb_new = grid.backward(image, grad_output, 0);
        ASSERT_NO_THROW(lfs::core::gpu_device_barrier(lfs::core::default_gpu_backend()));
        const std::vector<float> new_grid = cpu_copy(grid.grad_slice());
        const std::vector<float> new_rgb = cpu_copy(grad_rgb_new);

        const int h = static_cast<int>(image.shape()[chw ? 1 : 0]);
        const int w = static_cast<int>(image.shape()[chw ? 2 : 1]);
        const auto [grad_grid_ref, grad_rgb_ref] = reference_gradients(reference, grid, image, grad_output, h, w);
        expect_rel_near(new_grid, grad_grid_ref, 1e-4f, ctx + " grad_grid", grid_abs_floor);
        expect_rel_near(new_rgb, grad_rgb_ref, 1e-4f, ctx + " grad_rgb");
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
            Device::GPU);
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
            chw, {size_t{3}, static_cast<size_t>(h), static_cast<size_t>(w)}, Device::GPU);
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

        // Large-image grid gradients sum thousands of signed float32 terms per
        // cell. Even repeated scatter-reference runs differ by up to 3e-5 due
        // to atomicAdd ordering. Allow 5e-5 only for these accumulated gradients;
        // keep the relative, per-pixel, and small-image bounds unchanged.
        constexpr float kGridAbsFloor = 5e-5f;

        {
            BilateralGrid grid(1, 16, 16, 8, 20);
            fill_grid(grid, 3.1f);
            compare_new_vs_reference(grid, hwc, gout_hwc, false,
                                     launch_bilateral_grid_slice_backward_reference,
                                     "affine HWC bicycle", kGridAbsFloor);
            compare_new_vs_reference(grid, chw, gout_chw, true,
                                     launch_bilateral_grid_slice_backward_chw_reference,
                                     "affine CHW bicycle", kGridAbsFloor);
        }
        {
            BilateralGrid grid(1, 16, 16, 8, 20, {}, BilateralGridParameterization::ExposureChroma);
            fill_grid(grid, 3.7f);
            compare_new_vs_reference(grid, hwc, gout_hwc, false,
                                     launch_bilateral_grid_slice_backward_exposure_chroma_reference,
                                     "chroma HWC bicycle", kGridAbsFloor);
            compare_new_vs_reference(grid, chw, gout_chw, true,
                                     launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference,
                                     "chroma CHW bicycle", kGridAbsFloor);
        }
    }

    float hash_uniform(const uint32_t i, const uint32_t seed) {
        uint32_t x = i * 0x9E3779B1u ^ (seed * 0x85EBCA77u + 0x165667B1u);
        x ^= x >> 16;
        x *= 0x7FEB352Du;
        x ^= x >> 15;
        x *= 0x846CA68Bu;
        x ^= x >> 16;
        return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
    }

    // Noise puts every pixel into an unrelated luminance cell; Smooth is a photo-like pattern with fine texture,
    // so neighbouring pixels mostly share a grid cell.
    enum class Content { Noise,
                         Smooth };

    Tensor make_image(const int h, const int w, const bool chw, const Content content, const uint32_t seed) {
        const size_t hw = static_cast<size_t>(h) * static_cast<size_t>(w);
        std::vector<float> values(3 * hw);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t pix = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
                const float u = static_cast<float>(x) / static_cast<float>(w);
                const float v = static_cast<float>(y) / static_cast<float>(h);
                const float base = 0.5f + 0.28f * std::sin(6.1f * u + 1.3f) * std::cos(4.7f * v) +
                                   0.1f * std::sin(0.11f * static_cast<float>(x) + 0.07f * static_cast<float>(y)) +
                                   0.04f * std::sin(0.7f * static_cast<float>(x)) * std::sin(0.9f * static_cast<float>(y));
                for (int c = 0; c < 3; ++c) {
                    const uint32_t index = static_cast<uint32_t>(pix * 3 + static_cast<size_t>(c));
                    const float value =
                        content == Content::Noise
                            ? 0.02f + 0.96f * hash_uniform(index, seed)
                            : base + 0.06f * static_cast<float>(c - 1) * std::sin(3.0f * u + 2.0f * v + static_cast<float>(c)) +
                                  0.03f * (hash_uniform(index, seed) - 0.5f);
                    values[chw ? static_cast<size_t>(c) * hw + pix : pix * 3 + static_cast<size_t>(c)] = value;
                }
            }
        }
        return Tensor::from_vector(values,
                                   chw ? lfs::core::TensorShape{size_t{3}, static_cast<size_t>(h), static_cast<size_t>(w)}
                                       : lfs::core::TensorShape{static_cast<size_t>(h), static_cast<size_t>(w), size_t{3}},
                                   Device::GPU);
    }

    Tensor make_grad(const lfs::core::TensorShape& shape, const uint32_t seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = 0.2f * (2.0f * hash_uniform(static_cast<uint32_t>(i), seed) - 1.0f);
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    struct BackwardVariant {
        const char* name;
        BilateralGridParameterization param;
        bool chw;
        ReferenceLauncher reference;
    };

    const BackwardVariant kVariants[] = {
        {"affine HWC", BilateralGridParameterization::Affine, false,
         launch_bilateral_grid_slice_backward_reference},
        {"affine CHW", BilateralGridParameterization::Affine, true,
         launch_bilateral_grid_slice_backward_chw_reference},
        {"chroma HWC", BilateralGridParameterization::ExposureChroma, false,
         launch_bilateral_grid_slice_backward_exposure_chroma_reference},
        {"chroma CHW", BilateralGridParameterization::ExposureChroma, true,
         launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference},
    };

    // Normwise: max |got - ref| over max |ref|. Elementwise: |got - ref| over |ref|, with |ref| floored at 1 % of
    // max |ref| so cells whose terms cancel to almost nothing do not divide by zero.
    struct RelError {
        double normwise = 0.0;
        double elementwise = 0.0;
    };

    RelError relative_error(const std::vector<float>& got, const std::vector<float>& ref) {
        double max_ref = 0.0;
        for (const float r : ref)
            max_ref = std::max(max_ref, static_cast<double>(std::abs(r)));
        RelError err;
        if (max_ref == 0.0)
            return err;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double diff = std::abs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
            err.normwise = std::max(err.normwise, diff / max_ref);
            err.elementwise =
                std::max(err.elementwise, diff / std::max(static_cast<double>(std::abs(ref[i])), 0.01 * max_ref));
        }
        return err;
    }

    TEST_F(BilateralGridBackwardTiledTest, GridGradientMatchesReferenceAcrossSizes) {
        struct Size {
            int w;
            int h;
        };
        // 640x480 and 4000x3000 as the small and the large per-cell footprint; the odd and tiny sizes cover
        // partial tiles and tiles wider than the grid.
        const Size sizes[] = {{640, 480}, {4000, 3000}, {333, 97}, {37, 23}};
        uint32_t seed = 101;
        for (const Size size : sizes) {
            for (const Content content : {Content::Noise, Content::Smooth}) {
                for (const BackwardVariant& variant : kVariants) {
                    ++seed;
                    const std::string ctx = std::string(variant.name) + " " + std::to_string(size.w) + "x" +
                                            std::to_string(size.h) +
                                            (content == Content::Noise ? " noise" : " smooth");
                    BilateralGrid grid(1, 16, 16, 8, 20, {}, variant.param);
                    fill_grid(grid, 0.37f * static_cast<float>(seed));
                    const auto image = make_image(size.h, size.w, variant.chw, content, seed);
                    const auto gout = make_grad(image.shape(), seed + 7);

                    grid.zero_grad();
                    const auto grad_rgb = grid.backward(image, gout, 0);
                    ASSERT_NO_THROW(lfs::core::gpu_device_barrier(lfs::core::default_gpu_backend())) << ctx;
                    const auto got_grid = cpu_copy(grid.grad_slice());
                    const auto got_rgb = cpu_copy(grad_rgb);

                    const auto [ref_grid, ref_rgb] = reference_gradients(variant.reference, grid, image, gout, size.h, size.w);
                    const RelError grid_err = relative_error(got_grid, ref_grid);
                    const RelError rgb_err = relative_error(got_rgb, ref_rgb);
                    std::cout << std::format("[GridGrad] {} grad_grid rel normwise={:.9g} elementwise={:.9g} grad_rgb rel normwise={:.9g}\n",
                                             ctx, grid_err.normwise, grid_err.elementwise, rgb_err.normwise);
                    // The reference adds each pixel to global memory with its own float atomic, so its rounding
                    // error grows with the pixels per cell: 1.3e-5 / 2.6e-4 at 640x480, 1.3e-4 / 2.6e-3 at
                    // 4000x3000, several times what two runs of the reference differ by.
                    const double pixels = static_cast<double>(size.w) * static_cast<double>(size.h);
                    EXPECT_LT(grid_err.normwise, 1e-5 + 1e-11 * pixels) << ctx;
                    EXPECT_LT(grid_err.elementwise, 2e-4 + 2e-10 * pixels) << ctx;
                    EXPECT_LT(rgb_err.normwise, 1e-5) << ctx;

                    // grad_rgb may overwrite grad_output: the grid gradient must read it first.
                    auto in_place = gout.clone();
                    grid.backward_in_place(image, in_place, 0);
                    ASSERT_NO_THROW(lfs::core::gpu_device_barrier(lfs::core::default_gpu_backend())) << ctx;
                    EXPECT_EQ(cpu_copy(in_place), got_rgb) << ctx;
                    EXPECT_LT(relative_error(cpu_copy(grid.grad_slice()), got_grid).normwise, 1e-5) << ctx;
                }
            }
        }
    }

    uint64_t fnv1a(const std::vector<float>& values) {
        uint64_t hash = 1469598103934665603ull;
        for (const float value : values) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int b = 0; b < 4; ++b) {
                hash ^= (bits >> (8 * b)) & 0xFFu;
                hash *= 1099511628211ull;
            }
        }
        return hash;
    }

    // Median of 50 timed backward launches (grid gradient + grad_rgb) per variant, at 20 MP and 1 MP with a 16x16x8
    // grid. Prints a hash of grad_rgb so runs of two builds can be checked bit for bit. Run with
    // --gtest_also_run_disabled_tests.
    TEST_F(BilateralGridBackwardTiledTest, DISABLED_MicroBenchBackward) {
        struct Size {
            int w;
            int h;
        };
        const Size sizes[] = {{5262, 3927}, {1297, 840}};
        for (const Size size : sizes) {
            for (const Content content : {Content::Smooth, Content::Noise}) {
                for (const BackwardVariant& variant : kVariants) {
                    BilateralGrid grid(1, 16, 16, 8, 20, {}, variant.param);
                    fill_grid(grid, 4.1f);
                    const auto image = make_image(size.h, size.w, variant.chw, content, 21);
                    const auto gout = make_grad(image.shape(), 23);
                    auto grad_grid = Tensor::zeros(grid.grad_slice().shape(), Device::GPU);
                    auto grad_rgb = Tensor::empty(image.shape(), Device::GPU);
                    const auto backend = lfs::core::default_gpu_backend();
                    const auto coefficients = grid.grids().to(backend);
                    const auto* ops = lfs::training::training_ops(backend).bilateral;
                    ASSERT_NE(ops, nullptr);
                    const lfs::gpu_ops::GridSliceParams params{
                        .layout = variant.chw ? lfs::gpu_ops::Layout::CHW : lfs::gpu_ops::Layout::HWC,
                        .transform = variant.param == BilateralGridParameterization::ExposureChroma
                                         ? lfs::gpu_ops::GridTransform::ExposureChroma
                                         : lfs::gpu_ops::GridTransform::Affine};
                    const auto launch = [&] {
                        ops->slice_backward(coefficients, image, gout, grid.shared_offset(), grad_grid, grad_rgb, params);
                    };
                    for (int i = 0; i < 5; ++i)
                        launch();
                    lfs::core::gpu_device_barrier(backend);
                    lfs::core::GpuElapsed timer(backend, 2);
                    ASSERT_TRUE(timer.ready());
                    const auto queue = lfs::core::TensorExecutionTarget::current();
                    std::vector<float> times;
                    for (int i = 0; i < 50; ++i) {
                        ASSERT_TRUE(timer.mark(0, queue));
                        launch();
                        ASSERT_TRUE(timer.mark(1, queue));
                        ASSERT_TRUE(timer.wait_event(1));
                        const auto ms = timer.milliseconds(0, 1);
                        ASSERT_TRUE(ms.has_value());
                        times.push_back(*ms);
                    }
                    std::sort(times.begin(), times.end());
                    const float median = 0.5f * (times[24] + times[25]);
                    std::cout << "[BwdMicroBench] " << variant.name << " " << size.w << "x" << size.h
                              << (content == Content::Noise ? " noise " : " smooth ") << "median_ms=" << median
                              << " grad_rgb_fnv=" << std::hex << fnv1a(cpu_copy(grad_rgb)) << std::dec << "\n";
                }
            }
        }
    }

} // namespace
