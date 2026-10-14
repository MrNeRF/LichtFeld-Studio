/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

TEST(Jpeg2kExplicitLayout, RoundTripsNarrowChw) {
    int devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    ASSERT_GT(devices, 0);
    lfs::io::NvCodecImageLoader::Options options;
    options.decoder_pool_size = 1;
    lfs::io::NvCodecImageLoader loader(options);

    // 5x3 is CHW [3,H,3]. 3x5 has a side of 3. Shape guessing mishandles both.
    for (const auto [height, width] : {std::pair{5, 3}, std::pair{3, 5}}) {
        const size_t count = static_cast<size_t>(3 * height * width);
        std::vector<uint16_t> codes(count);
        for (size_t i = 0; i < count; ++i)
            codes[i] = static_cast<uint16_t>(i + 1);

        const lfs::core::TensorShape shape(
            {size_t{3}, static_cast<size_t>(height), static_cast<size_t>(width)});
        auto packed = lfs::core::Tensor::empty(shape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
        ASSERT_EQ(cudaMemcpy(packed.data_ptr(), codes.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice),
                  cudaSuccess);
        auto chw = lfs::core::Tensor::empty(shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        lfs::io::cuda::launch_uint16_hwc_to_float32_hwc(
            reinterpret_cast<const uint16_t*>(packed.data_ptr()), chw.ptr<float>(),
            static_cast<size_t>(height), static_cast<size_t>(width), 3);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto decoded = loader.decode_jpeg2k_16bit_from_memory_gpu(
            loader.encode_to_jpeg2k(chw, lfs::io::RgbLayout::CHW));
        ASSERT_EQ(decoded.shape()[0], static_cast<size_t>(height));
        ASSERT_EQ(decoded.shape()[1], static_cast<size_t>(width));
        ASSERT_EQ(decoded.shape()[2], size_t{3});
        const auto actual = decoded.permute({2, 0, 1}).contiguous().cpu().to_vector();
        const auto expected = chw.cpu().to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            ASSERT_EQ(actual[i], expected[i]) << height << "x" << width << " index " << i;
    }
}
