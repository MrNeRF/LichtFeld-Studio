/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Standalone contract test linked to the actual core DLL and CUDA kernels.
#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include "core/splat_preview.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <stdexcept>

using namespace lfs::core;

namespace {
    void require(bool value) {
        if (!value)
            throw std::runtime_error("Selected SH preview contract failed");
    }
    template <typename F>
    void rejects(F action) {
        bool rejected = false;
        try {
            action();
        } catch (const std::logic_error&) { rejected = true; }
        require(rejected);
    }

    void check(Device device, int degree, int format) {
        constexpr size_t n = 300;
        const uint32_t rest = sh_rest_coefficients_for_degree(degree);
        const size_t cells = format == 2 ? sh_swizzled_padded_n(n) * rest * 3u : sh_swizzled_float_count(n, rest);
        auto storage = Tensor::empty_pageable_host({cells}, format ? DataType::Float16 : DataType::Float32);
        auto bounds = Tensor::empty_pageable_host({size_t{4}}, DataType::Float32);
        for (int i = 0; i < 4; ++i)
            bounds.ptr<float>()[i] = float(i * 2 - 3);
        for (size_t i = 0; i < cells; ++i) {
            if (format == 2)
                static_cast<uint16_t*>(storage.data_ptr())[i] = static_cast<uint16_t>((i * 17u) % 65536u);
            else if (format == 1)
                storage.ptr<__half>()[i] = __float2half(float(i % 101) / 101.f - .5f);
            else
                storage.ptr<float>()[i] = float(i % 101) / 101.f - .5f;
        }
        SplatData model(degree, Tensor::zeros({n, size_t{3}}, device), Tensor::zeros({n, size_t{1}, size_t{3}}, device),
                        storage.to(device), Tensor::zeros({n, size_t{3}}, device), Tensor::zeros({n, size_t{4}}, device),
                        Tensor::zeros({n, size_t{1}}, device), 1.f, SplatData::ShNLayout::Swizzled);
        if (format == 2)
            model.set_active_sh_degree(degree, bounds.to(device));
        auto rows = Tensor::from_vector(std::vector<int>{299, 0, 31, 32, 255, 256, 257, 31}, {size_t{8}}, Device::CPU);
        const auto before = model.shN_raw().to_pageable_host();
        for (int selected_degree = 0; selected_degree <= degree; ++selected_degree) {
            const auto output = splat_sh_preview(model, rows, selected_degree);
            const uint32_t selected_rest = sh_rest_coefficients_for_degree(selected_degree);
            require(output.device() == Device::CPU && output.dtype() == DataType::Float32);
            require(output.shape() == TensorShape({size_t{8}, selected_rest, size_t{3}}));
            for (size_t i = 0; i < 8; ++i) {
                const auto row = static_cast<size_t>(rows.ptr<int>()[i]);
                for (size_t c = 0; c < selected_rest * 3u; ++c) {
                    float expected;
                    if (format == 2) {
                        const size_t offset = (row / 32u) * (rest * 3u) * 32u + c * 32u + row % 32u;
                        const auto q = static_cast<const uint16_t*>(storage.data_ptr())[offset];
                        const auto* b = bounds.ptr<float>();
                        const volatile float range = b[row / 256u * 2u + 1u] - b[row / 256u * 2u];
                        const volatile float normalized = static_cast<float>(q) * (1.f / 65535.f);
                        expected = std::fma(static_cast<float>(range), static_cast<float>(normalized), b[row / 256u * 2u]);
                    } else {
                        const size_t slots = (rest * 3u + 3u) / 4u;
                        const size_t offset = ((row / 32u) * slots * 32u + (c / 4u) * 32u + row % 32u) * 4u + c % 4u;
                        expected = format == 1 ? __half2float(storage.ptr<__half>()[offset]) : storage.ptr<float>()[offset];
                    }
                    require(output.ptr<float>()[i * selected_rest * 3u + c] == expected);
                }
            }
            const auto int64_output = splat_sh_preview(model, rows.to(DataType::Int64), selected_degree);
            require(int64_output.bytes() == output.bytes());
            require(output.bytes() == 0 || std::memcmp(output.data_ptr(), int64_output.data_ptr(), output.bytes()) == 0);
        }
        const auto after = model.shN_raw().to_pageable_host();
        require(std::memcmp(before.data_ptr(), after.data_ptr(), before.bytes()) == 0);
        require(splat_sh_preview(model, Tensor::empty_pageable_host({size_t{0}}, DataType::Int64), degree).numel() == 0);
        rejects([&] { (void)splat_sh_preview(model, rows, -1); });
        rejects([&] { (void)splat_sh_preview(model, rows, degree + 1); });
        rejects([&] { (void)splat_sh_preview(model, rows.to(DataType::Float32), degree); });
        rejects([&] { (void)splat_sh_preview(model, rows.reshape({size_t{2}, size_t{4}}), degree); });
        for (int invalid : {-1, 300}) {
            rejects([&] { (void)splat_sh_preview(model, Tensor::from_vector(std::vector<int>{invalid}, {size_t{1}}, Device::CPU), degree); });
        }
    }
} // namespace

int main() try {
    for (auto device : {Device::CPU, Device::CUDA})
        for (int degree = 1; degree <= 3; ++degree)
            for (int format = 0; format < 3; ++format)
                check(device, degree, format);
    (void)std::puts("SELECTED_SH_PREVIEW_CORE_OK");
    return 0;
} catch (const std::exception& error) {
    (void)std::fprintf(stderr, "%s\n", error.what());
    return 1;
} catch (...) {
    return 2;
}
