/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../metrics/flip_config.hpp"
#include "core/tensor.hpp"
#include <limits>

namespace lfs::training::flip_detail {
    struct Parameters {
        uint64_t image = 0, taps = 0, planes = 0, reference = 0, features = 0, error = 0, output = 0;
        int32_t width = 0, height = 0, csf_radius = 0, feature_radius = 0;
        float max_color_error = 0;
        uint32_t operation = 0, group_offset = 0;
    };
    static_assert(sizeof(Parameters) == 88 && offsetof(Parameters, width) == 56);

    template <class Backend>
    struct Implementation {
        using Tensor = core::Tensor;
        static Tensor map(const Tensor& reference, const Tensor& test, float ppd) {
            const auto h = reference.shape()[1], w = reference.shape()[2];
            LFS_ASSERT(w > 0 && h > 0 && w <= size_t(std::numeric_limits<int32_t>::max()) && h <= size_t(std::numeric_limits<int32_t>::max()));
            Parameters p;
            p.width = static_cast<int32_t>(w);
            p.height = static_cast<int32_t>(h);
            const auto host_taps = flip_taps(ppd, p.csf_radius, p.feature_radius);
            const auto taps = Tensor::from_vector(host_taps, {host_taps.size()}, core::Device::GPU);
            auto planes = Tensor::empty({HORIZONTAL_PLANES, h, w}, core::Device::GPU);
            auto features = Tensor::empty({FEATURE_PLANES, h, w}, core::Device::GPU);
            auto error = Tensor::empty({h, w}, core::Device::GPU);
            const auto a = reference.contiguous(), b = test.contiguous();
            const auto green = hunt_adjusted_lab(0, 1, 0), blue = hunt_adjusted_lab(0, 0, 1);
            p.max_color_error = std::pow(hyab(green, blue), COLOR_EXPONENT);
            p.taps = Backend::address(taps);
            p.planes = Backend::address(planes);
            p.image = Backend::address(a);
            const size_t groups = (w + 255) / 256 * h;
            Backend::launch(p, {&a, &taps}, {&planes}, groups);
            p.operation = 1;
            p.features = Backend::address(features);
            Backend::launch(p, {&planes, &taps}, {&features}, groups);
            p.operation = 0;
            p.image = Backend::address(b);
            Backend::launch(p, {&b, &taps}, {&planes}, groups);
            p.operation = 1;
            p.reference = Backend::address(features);
            p.features = 0;
            p.error = Backend::address(error);
            Backend::launch(p, {&planes, &taps, &features}, {&error}, groups);
            return error;
        }
        static Tensor image(const Tensor& error_map) {
            const auto source = error_map.contiguous();
            auto result = Tensor::empty({3, source.shape()[0], source.shape()[1]}, core::Device::GPU, core::DataType::UInt8);
            if (!source.numel())
                return result;
            auto colors = Tensor::empty({std::size(MAGMA)}, core::Device::CPU, core::DataType::UInt8);
            std::copy(std::begin(MAGMA), std::end(MAGMA), colors.ptr<uint8_t>());
            const auto palette = colors.to(core::Device::GPU);
            Parameters p;
            p.operation = 2;
            p.width = static_cast<int32_t>(source.shape()[1]);
            p.height = static_cast<int32_t>(source.shape()[0]);
            p.image = Backend::address(source);
            p.taps = Backend::address(palette);
            p.output = Backend::address(result);
            Backend::launch(p, {&source, &palette}, {&result}, (source.numel() + 255) / 256);
            return result;
        }
    };
} // namespace lfs::training::flip_detail
