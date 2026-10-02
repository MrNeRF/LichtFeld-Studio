/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "lfs/training/ops/registry.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs::core;
    namespace image_kernels = lfs::io::cuda;
    constexpr size_t H = 13, W = 17;
    Tensor image_pattern(const TensorShape& shape, DataType dtype = DataType::Float32) {
        auto cpu = Tensor::empty(shape, Device::CPU, dtype);
        if (dtype == DataType::Float32) {
            for (size_t i = 0; i < cpu.numel(); ++i)
                cpu.ptr<float>()[i] = float((i * 7919 + 104729) % 2003) / 2002.f;
        } else {
            auto* bytes = static_cast<uint8_t*>(cpu.data_ptr());
            for (size_t i = 0; i < cpu.bytes(); ++i)
                bytes[i] = static_cast<uint8_t>((i * 71 + 149) % 256);
        }
        return cpu.gpu();
    }

    std::vector<uint8_t> bytes(const Tensor& value) {
        auto cpu = value.cpu().contiguous();
        const auto* p = static_cast<const uint8_t*>(cpu.data_ptr());
        return {p, p + cpu.bytes()};
    }
    template <typename Run>
    void expect_same(Run&& run) {
        const auto expected = run(false);
        const auto actual = run(true);
        ASSERT_EQ(cudaStreamSynchronize(getCurrentCUDAStream()), cudaSuccess);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(actual[i].shape(), expected[i].shape());
            EXPECT_EQ(bytes(actual[i]), bytes(expected[i]));
        }
    }
    using SharedImageOpsTest = lfs::test::CudaBackendTest;

    TEST_F(SharedImageOpsTest, SentinelsMatchDirectLaunchers) {
        const auto& ops = *lfs::training::training_ops(GpuBackend::CUDA).shared_image;
        TensorCudaStream owned_stream;
        for (const auto stream : {cudaStream_t{nullptr}, owned_stream.get()}) {
            const CUDAStreamGuard scope(stream);
            {
                auto run = [&](bool dispatch) {
                    auto output = Tensor::empty({1031}, Device::GPU, DataType::UInt8);
                    auto unchanged = Tensor::full({1}, 1.f, Device::GPU, DataType::UInt32);
                    if (dispatch) {
                        ops.sentinel_fill(output, 0x932abe71u);
                        ops.sentinel_check(output, unchanged, 0x932abe71u);
                    } else {
                        image_kernels::launch_fill_u8_sentinel(output.ptr<uint8_t>(), output.bytes(), 0x932abe71u, stream);
                        image_kernels::launch_flag_u8_sentinel_unchanged(output.ptr<uint8_t>(), output.bytes(), 0x932abe71u, unchanged.ptr<uint32_t>(), stream);
                    }
                    return std::vector<Tensor>{output, unchanged};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto output = image_pattern({1031}, DataType::UInt8);
                    auto unchanged = Tensor::full({1}, 1.f, Device::GPU, DataType::UInt32);
                    if (dispatch) {
                        ops.sentinel_check(output, unchanged, 0x932abe71u);
                    } else {
                        image_kernels::launch_flag_u8_sentinel_unchanged(output.ptr<uint8_t>(), output.bytes(), 0x932abe71u, unchanged.ptr<uint32_t>(), stream);
                    }
                    return std::vector<Tensor>{unchanged};
                };
                expect_same(run);
            }
        }
    }

    TEST_F(SharedImageOpsTest, ConversionsMatchDirectLaunchers) {
        const auto& ops = *lfs::training::training_ops(GpuBackend::CUDA).shared_image;
        TensorCudaStream owned_stream;
        for (const auto stream : {cudaStream_t{nullptr}, owned_stream.get()}) {
            const CUDAStreamGuard scope(stream);
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::UInt8);
                    auto output = image_pattern({H, W, 3}, DataType::Float32);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::U8HWCToF32CHW, H, W, 3, {});
                    } else {
                        image_kernels::launch_uint8_hwc_to_float32_chw(static_cast<uint8_t*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, 3, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 6}, DataType::UInt8);
                    auto output = image_pattern({H, W, 3}, DataType::Float32);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::U16HWCToF32CHW, H, W, 3, {});
                    } else {
                        image_kernels::launch_uint16_hwc_to_float32_chw(static_cast<uint16_t*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, 3, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::Float32);
                    auto output = image_pattern({H, W, 6}, DataType::UInt8);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::F32HWCToU16HWC, H, W, 3, {});
                    } else {
                        image_kernels::launch_float32_hwc_to_uint16_hwc(static_cast<float*>(source.data_ptr()), static_cast<uint16_t*>(output.data_ptr()), H, W, 3, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 6}, DataType::UInt8);
                    auto output = image_pattern({H, W, 3}, DataType::Float32);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::U16HWCToF32HWC, H, W, 3, {});
                    } else {
                        image_kernels::launch_uint16_hwc_to_float32_hwc(static_cast<uint16_t*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, 3, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::Float32);
                    auto output = image_pattern({H, W, 3}, DataType::Float32);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::NormalCHWToJ2KHWC, H, W, 3, {});
                    } else {
                        image_kernels::launch_normal_chw_to_jpeg2k_hwc(static_cast<float*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::Float32);
                    auto output = image_pattern({H, W, 3}, DataType::Float32);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::J2KHWCToNormalCHW, H, W, 3, {});
                    } else {
                        image_kernels::launch_jpeg2k_hwc_to_normal_chw(static_cast<float*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::UInt8);
                    auto output = image_pattern({H, W, 3}, DataType::Float32);
                    lfs::gpu_ops::NormalPriorTransform transform{.srgb = true, .flip_yz = true, .world_to_camera = true, .w2c = {0, -1, 0, 1, 0, 0, 0, 0, 1}};
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::NormalPriorU8, H, W, 3, transform);
                    } else {
                        image_kernels::launch_normal_prior_u8_hwc_to_float32_chw(static_cast<uint8_t*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, transform, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 6}, DataType::UInt8);
                    auto output = image_pattern({H, W, 3}, DataType::Float32);
                    lfs::gpu_ops::NormalPriorTransform transform{.srgb = true, .flip_yz = true, .world_to_camera = true, .w2c = {0, -1, 0, 1, 0, 0, 0, 0, 1}};
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::NormalPriorU16, H, W, 3, transform);
                    } else {
                        image_kernels::launch_normal_prior_u16_hwc_to_float32_chw(static_cast<uint16_t*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, transform, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::UInt8);
                    auto output = image_pattern({H, W, 3}, DataType::UInt8);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::U8HWCToU8CHW, H, W, 3, {});
                    } else {
                        image_kernels::launch_uint8_hwc_to_uint8_chw(static_cast<uint8_t*>(source.data_ptr()), static_cast<uint8_t*>(output.data_ptr()), H, W, 3, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 6}, DataType::UInt8);
                    auto output = image_pattern({H, W, 3}, DataType::UInt8);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::U16HWCToU8CHW, H, W, 3, {});
                    } else {
                        image_kernels::launch_uint16_hwc_to_uint8_chw(static_cast<uint16_t*>(source.data_ptr()), static_cast<uint8_t*>(output.data_ptr()), H, W, 3, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::Float32);
                    auto output = image_pattern({H, W, 3}, DataType::UInt8);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW, H, W, 3, {});
                    } else {
                        image_kernels::launch_float32_chw_to_uint8_chw(static_cast<float*>(source.data_ptr()), static_cast<uint8_t*>(output.data_ptr()), H, W, 3, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W}, DataType::UInt8);
                    auto output = image_pattern({H, W}, DataType::Float32);
                    if (dispatch) {
                        ops.convert(source, output, lfs::gpu_ops::ImageConversion::U8HWToF32HW, H, W, 1, {});
                    } else {
                        image_kernels::launch_uint8_hw_to_float32_hw(static_cast<uint8_t*>(source.data_ptr()), static_cast<float*>(output.data_ptr()), H, W, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
        }
    }

    TEST_F(SharedImageOpsTest, RgbaSplitMatchDirectLaunchers) {
        const auto& ops = *lfs::training::training_ops(GpuBackend::CUDA).shared_image;
        TensorCudaStream owned_stream;
        for (const auto stream : {cudaStream_t{nullptr}, owned_stream.get()}) {
            const CUDAStreamGuard scope(stream);
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 4}, DataType::UInt8);
                    auto rgb = image_pattern({3, H, W}, DataType::Float32);
                    auto alpha = image_pattern({H, W});
                    if (dispatch) {
                        ops.rgba_split(source, rgb, alpha);
                    } else {
                        image_kernels::launch_uint8_rgba_split_to_float32_rgb_and_alpha(source.ptr<uint8_t>(), rgb.ptr<float>(), alpha.ptr<float>(), H, W, stream);
                    }
                    return std::vector<Tensor>{rgb, alpha};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 4}, DataType::UInt8);
                    auto rgb = image_pattern({3, H, W}, DataType::UInt8);
                    auto alpha = image_pattern({H, W});
                    if (dispatch) {
                        ops.rgba_split(source, rgb, alpha);
                    } else {
                        image_kernels::launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(source.ptr<uint8_t>(), rgb.ptr<uint8_t>(), alpha.ptr<float>(), H, W, stream);
                    }
                    return std::vector<Tensor>{rgb, alpha};
                };
                expect_same(run);
            }
        }
    }

    TEST_F(SharedImageOpsTest, MasksMatchDirectLaunchers) {
        const auto& ops = *lfs::training::training_ops(GpuBackend::CUDA).shared_image;
        TensorCudaStream owned_stream;
        for (const auto stream : {cudaStream_t{nullptr}, owned_stream.get()}) {
            const CUDAStreamGuard scope(stream);
            {
                auto run = [&](bool dispatch) {
                    auto output = image_pattern({H, W});
                    if (dispatch) {
                        ops.mask(output, lfs::gpu_ops::MaskTransform::Invert, 0.61f);
                    } else {
                        image_kernels::launch_mask_invert(output.ptr<float>(), H, W, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto output = image_pattern({H, W});
                    if (dispatch) {
                        ops.mask(output, lfs::gpu_ops::MaskTransform::Threshold, 0.61f);
                    } else {
                        image_kernels::launch_mask_threshold(output.ptr<float>(), H, W, 0.61f, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
        }
    }

    TEST_F(SharedImageOpsTest, ResizeMatchDirectLaunchers) {
        const auto& ops = *lfs::training::training_ops(GpuBackend::CUDA).shared_image;
        TensorCudaStream owned_stream;
        for (const auto stream : {cudaStream_t{nullptr}, owned_stream.get()}) {
            const CUDAStreamGuard scope(stream);
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::UInt8);
                    Tensor output;
                    if (dispatch) {
                        output = ops.resize(source, 7, 9, lfs::gpu_ops::Resample::LanczosRGB, 2);
                    } else {
                        output = lanczos_resize(source, 7, 9, 2, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W, 3}, DataType::Float32);
                    Tensor output;
                    if (dispatch) {
                        output = ops.resize(source, 7, 9, lfs::gpu_ops::Resample::LanczosRGB, 2);
                    } else {
                        output = lanczos_resize(source, 7, 9, 2, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W}, DataType::UInt8);
                    Tensor output;
                    if (dispatch) {
                        output = ops.resize(source, 7, 9, lfs::gpu_ops::Resample::LanczosGray, 2);
                    } else {
                        output = lanczos_resize_grayscale(source, 7, 9, 2, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W}, DataType::Float32);
                    Tensor output;
                    if (dispatch) {
                        output = ops.resize(source, 7, 9, lfs::gpu_ops::Resample::LanczosGray, 2);
                    } else {
                        output = lanczos_resize_grayscale(source, 7, 9, 2, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({3, H, W}, DataType::Float32);
                    Tensor output;
                    if (dispatch) {
                        output = ops.resize(source, 7, 9, lfs::gpu_ops::Resample::LanczosFloatCHW, 2);
                    } else {
                        output = lanczos_resize_float_chw(source, 7, 9, 2, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W}, DataType::Float32);
                    Tensor output;
                    if (dispatch) {
                        output = ops.resize(source, 7, 9, lfs::gpu_ops::Resample::DepthPrior, 2);
                    } else {
                        output = resize_depth_prior(source, 7, 9, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({3, H, W}, DataType::Float32);
                    Tensor output;
                    if (dispatch) {
                        output = ops.resize(source, 7, 9, lfs::gpu_ops::Resample::NormalPrior, 2);
                    } else {
                        output = resize_normal_prior(source, 7, 9, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
        }
    }

    TEST_F(SharedImageOpsTest, UndistortMatchDirectLaunchers) {
        const auto& ops = *lfs::training::training_ops(GpuBackend::CUDA).shared_image;
        TensorCudaStream owned_stream;
        for (const auto stream : {cudaStream_t{nullptr}, owned_stream.get()}) {
            const CUDAStreamGuard scope(stream);
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({3, H, W});
                    UndistortParams params{};
                    params.src_fx = params.dst_fx = 15.f;
                    params.src_fy = params.dst_fy = 14.f;
                    params.src_cx = params.dst_cx = 8.f;
                    params.src_cy = params.dst_cy = 6.f;
                    params.src_width = W;
                    params.src_height = H;
                    params.dst_width = 15;
                    params.dst_height = 11;
                    params.model_type = CameraModelType::PINHOLE;
                    params.num_distortion = 4;
                    params.distortion[0] = 0.08f;
                    params.distortion[1] = -0.03f;
                    params.distortion[2] = 0.001f;
                    params.distortion[3] = -0.002f;
                    Tensor output;
                    if (dispatch) {
                        output = ops.undistort(source, params, false);
                    } else {
                        output = undistort_image(source, params, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
            {
                auto run = [&](bool dispatch) {
                    auto source = image_pattern({H, W});
                    UndistortParams params{};
                    params.src_fx = params.dst_fx = 15.f;
                    params.src_fy = params.dst_fy = 14.f;
                    params.src_cx = params.dst_cx = 8.f;
                    params.src_cy = params.dst_cy = 6.f;
                    params.src_width = W;
                    params.src_height = H;
                    params.dst_width = 15;
                    params.dst_height = 11;
                    params.model_type = CameraModelType::PINHOLE;
                    params.num_distortion = 4;
                    params.distortion[0] = 0.08f;
                    params.distortion[1] = -0.03f;
                    params.distortion[2] = 0.001f;
                    params.distortion[3] = -0.002f;
                    Tensor output;
                    if (dispatch) {
                        output = ops.undistort(source, params, true);
                    } else {
                        output = undistort_mask(source, params, stream);
                    }
                    return std::vector<Tensor>{output};
                };
                expect_same(run);
            }
        }
    }

    TEST_F(SharedImageOpsTest, RegistrySharesCoreTableAndRejectsMissingBackends) {
        EXPECT_EQ(lfs::training::training_ops(GpuBackend::CUDA).shared_image, shared_image_ops(GpuBackend::CUDA));
        lfs::core::param::TrainingParameters params;
        EXPECT_EQ(lfs::training::training_ops(GpuBackend::Vulkan).shared_image, shared_image_ops(GpuBackend::Vulkan));
        EXPECT_NE(shared_image_ops(GpuBackend::Vulkan), nullptr);
        EXPECT_FALSE(lfs::training::unavailable_training_family(GpuBackend::Vulkan, lfs::training::Family::SharedImage));
        for (auto backend : {GpuBackend::Metal}) {
            EXPECT_EQ(shared_image_ops(backend), nullptr);
            EXPECT_EQ(lfs::training::training_ops(backend).shared_image, nullptr);
            const auto reason = lfs::training::unavailable_training_reason(params, backend, lfs::training::training_loader_dependencies(params));
            ASSERT_TRUE(reason);
            EXPECT_NE(reason->find("SharedImage"), std::string::npos);
        }
    }
} // namespace
