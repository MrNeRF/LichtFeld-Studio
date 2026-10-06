/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_device_runtime.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/facade_trace.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"
#include "core/tensor_ppisp.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

namespace {
    using namespace lfs::core;
    class ImageRegionBackends : public testing::TestWithParam<std::optional<GpuBackend>> {
    protected:
        std::optional<GpuBackendScope> scope;
        Device device() const { return GetParam() ? Device::GPU : Device::CPU; }
        void SetUp() override {
            if (GetParam()) {
                if (!gpu_backend_available(*GetParam()))
                    GTEST_SKIP() << "Backend unavailable";
                scope.emplace(*GetParam());
            }
        }
        void TearDown() override {
            if (GetParam() && *GetParam() != GpuBackend::CUDA && gpu_backend_live(*GetParam())) {
                gpu_device_barrier(*GetParam());
                EXPECT_TRUE(shutdown_gpu_backend(*GetParam()));
            }
            scope.reset();
        }
        UndistortParams params(CameraModelType model, int width = 33, int height = 17) {
            UndistortParams p{};
            p.src_width = p.dst_width = width;
            p.src_height = p.dst_height = height;
            p.src_fx = p.dst_fx = 0.8f * width;
            p.src_fy = p.dst_fy = 1.1f * height;
            p.src_cx = 0.47f * width;
            p.src_cy = 0.53f * height;
            p.dst_cx = p.src_cx + 0.75f;
            p.dst_cy = p.src_cy - 0.35f;
            p.model_type = model;
            p.num_distortion = model == CameraModelType::PINHOLE ? 8 : model == CameraModelType::FISHEYE ? 4
                                                                                                         : 12;
            p.distortion[0] = -0.08f;
            p.distortion[1] = 0.012f;
            p.distortion[2] = 0.001f;
            p.distortion[3] = -0.0015f;
            if (model == CameraModelType::THIN_PRISM_FISHEYE) {
                p.distortion[8] = 0.0008f;
                p.distortion[10] = -0.0006f;
            }
            return p;
        }
        Tensor rgb(int width = 33, int height = 17) {
            std::vector<uint8_t> bytes(3 * width * height);
            for (size_t i = 0; i < bytes.size(); ++i)
                bytes[i] = uint8_t((i * 37 + i / 11) % 256);
            return Tensor::from_blob(bytes.data(), {3, size_t(height), size_t(width)}, Device::CPU, DataType::UInt8).clone();
        }
        void near(const Tensor& result, const Tensor& reference, float tolerance = 4e-4f) {
            EXPECT_EQ(result.dtype(), DataType::Float32);
            EXPECT_EQ(result.device(), device());
            EXPECT_EQ(gpu_backend_of(result), GetParam());
            const auto a = result.to_vector(), b = reference.to_vector();
            ASSERT_EQ(a.size(), b.size());
            float maximum = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                ASSERT_TRUE(std::isfinite(a[i])) << i;
                maximum = std::max(maximum, std::abs(a[i] - b[i]));
            }
            EXPECT_LE(maximum, tolerance);
        }
    };

    TEST_P(ImageRegionBackends, UInt8FullAndRegionsMatchFloatAcrossModelsBordersAndOddSizes) {
        for (const auto model : {CameraModelType::PINHOLE, CameraModelType::FISHEYE, CameraModelType::THIN_PRISM_FISHEYE}) {
            SCOPED_TRACE(static_cast<int>(model));
            for (const auto size : {std::array{19, 13}, std::array{33, 17}}) {
                const auto p = params(model, size[0], size[1]);
                const auto cpu = rgb(size[0], size[1]);
                const auto input = cpu.to(device());
                const auto normalized = cpu.to(DataType::Float32).div(255.0f);
                const auto oracle = undistort_image(normalized, p, nullptr);
                near(undistort_image(input, p, nullptr), oracle);
                near(undistort_image(normalized.to(device()), p, nullptr), oracle);
                for (const auto origin : {std::array{0, 0}, std::array{size[0] - 9, size[1] - 7}, std::array{3, 2}}) {
                    const auto expected = oracle.slice(1, origin[1], origin[1] + 7).slice(2, origin[0], origin[0] + 9).contiguous();
                    near(undistort_image_region(input, p, origin[0], origin[1], 9, 7, nullptr), expected);
                }
                // A strided CHW view preserves UInt8 sampling and only needs one
                // materialization when prepared as a retained native source.
                const auto strided = cpu.permute({0, 2, 1}).to(device());
                const auto transposed = params(model, size[1], size[0]);
                const auto strided_ref = undistort_image(strided.cpu().to(DataType::Float32).div(255.0f), transposed, nullptr);
                near(undistort_image_region(strided, transposed, 1, 2, 7, 9, nullptr),
                     strided_ref.slice(1, 2, 11).slice(2, 1, 8).contiguous());
            }
        }
    }

    TEST_P(ImageRegionBackends, FloatMasksRetainSamplingAndBorderSemantics) {
        for (const auto model : {CameraModelType::PINHOLE, CameraModelType::FISHEYE, CameraModelType::THIN_PRISM_FISHEYE}) {
            const auto p = params(model);
            const auto mask = rgb().slice(0, 0, 1).squeeze(0).to(DataType::Float32).div(255.0f);
            near(undistort_mask_area(mask.to(device()), p, nullptr), undistort_mask_area(mask, p, nullptr));
        }
    }

    TEST_P(ImageRegionBackends, PpispFullRowsAndRectangularCropsAgree) {
        const auto input = rgb().to(DataType::Float32).div(255.0f).to(device());
        PpispParams settings;
        settings.exposure_factor = 1.12f;
        settings.full_width = 33;
        settings.full_height = 17;
        for (int c = 0; c < 3; ++c) {
            settings.vignetting[c * 5] = 0.43f + 0.01f * c;
            settings.vignetting[c * 5 + 1] = 0.54f;
            settings.vignetting[c * 5 + 2] = 0.08f;
            settings.vignetting[c * 5 + 3] = -0.02f;
            settings.vignetting[c * 5 + 4] = 0.01f;
        }
        const auto full = ppisp_apply(input, settings);
        const auto cpu = ppisp_apply(input.cpu(), settings);
        near(full, cpu, 2e-5f);
        for (const auto region : {std::array{0, 3, 33, 7}, std::array{5, 4, 19, 9}, std::array{26, 12, 7, 5}}) {
            auto cropped = settings;
            cropped.x_offset = region[0];
            cropped.y_offset = region[1];
            const auto slice = input.slice(1, region[1], region[1] + region[3]).slice(2, region[0], region[0] + region[2]);
            const auto expected = full.slice(1, region[1], region[1] + region[3]).slice(2, region[0], region[0] + region[2]).contiguous();
            near(ppisp_apply(slice, cropped), expected, 2e-5f);
        }
    }

    TEST_P(ImageRegionBackends, RepeatedEightKPanKeepsWorkingStorageBoundedAndNeverConvertsWholeSource) {
        if (!GetParam())
            GTEST_SKIP() << "GPU allocation counters required; CPU numeric parity is covered separately";
        constexpr int width = 8193, height = 4321;
        const auto p = params(CameraModelType::PINHOLE, width, height);
        // Exactly one owned contiguous RGB8 source is prepared and uploaded.
        const auto input = Tensor::zeros({3, size_t(height), size_t(width)}, Device::CPU, DataType::UInt8).to(device()).contiguous();
        {
            const auto warm = undistort_image_region(input, p, 1300, 1400, 63, 47, nullptr);
            gpu_device_barrier(*GetParam());
        }
        const auto baseline = gpu_backend_memory_info(*GetParam(), true);
        struct TraceScope {
            bool previous = internal::g_facade_trace_enabled.exchange(true);
            ~TraceScope() { internal::g_facade_trace_enabled.store(previous); }
        } trace;
        const auto before = internal::facade_trace_snapshot_for_testing();
        size_t maximum = baseline.allocated_bytes;
        for (int pan = 0; pan < 12; ++pan) {
            const auto crop = undistort_image_region(input, p, 1300 + pan * 17, 1400 + pan * 11, 63, 47, nullptr);
            gpu_device_barrier(*GetParam());
            EXPECT_EQ(crop.bytes(), size_t(3 * 63 * 47 * sizeof(float)));
            maximum = std::max(maximum, gpu_backend_memory_info(*GetParam(), true).allocated_bytes);
        }
        const auto after = internal::facade_trace_snapshot_for_testing();
        for (const auto entry : {internal::FacadeEntry::convert_type, internal::FacadeEntry::service_copy_device_to_host,
                                 internal::FacadeEntry::service_enqueue_readback}) {
            EXPECT_EQ(after[size_t(entry)], before[size_t(entry)]) << internal::facade_entry_name(entry);
        }
        const size_t increase = maximum - baseline.allocated_bytes;
        RecordProperty("source_bytes", std::to_string(input.bytes()));
        RecordProperty("crop_bytes", std::to_string(3 * 63 * 47 * sizeof(float)));
        RecordProperty("warm_allocated_bytes", std::to_string(baseline.allocated_bytes));
        RecordProperty("maximum_allocated_increase_bytes", std::to_string(increase));
        EXPECT_LT(increase, size_t(64 * 1024 * 1024));
    }

    const auto region_backends = [] {
        std::vector<std::optional<GpuBackend>> values{std::nullopt};
        for (const auto backend : kCompiledGpuBackends)
            values.emplace_back(backend);
        return values;
    }();
    INSTANTIATE_TEST_SUITE_P(Backends, ImageRegionBackends, testing::ValuesIn(region_backends),
                             [](const testing::TestParamInfo<std::optional<GpuBackend>>& info) {
                                 return !info.param ? "Cpu" : *info.param == GpuBackend::CUDA ? "Cuda"
                                                          : *info.param == GpuBackend::Vulkan ? "Vulkan"
                                                                                              : "Metal";
                             });
} // namespace
