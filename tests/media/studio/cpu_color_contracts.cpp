// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/tensor_color.hpp"
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace {
    namespace core = lfs::core;
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    // Frozen dev 4a4ab916 producer: independent tensor-expression regression oracle.
    core::Yuv420Planes legacyRgbToYuv420p(const core::Tensor& rgb) {
        const int height = static_cast<int>(rgb.size(0));
        const int width = static_cast<int>(rgb.size(1));
        const auto bytes = (rgb.clamp(0.0f, 1.0f) * 255.0f + 0.5f).floor();
        const auto channel = [](const core::Tensor& image, const size_t c) {
            return image.slice(2, c, c + 1).reshape({static_cast<int>(image.size(0)), static_cast<int>(image.size(1))});
        };
        const auto y = ((channel(bytes, 0) * 66.0f +
                         channel(bytes, 1) * 129.0f +
                         channel(bytes, 2) * 25.0f + 128.0f) /
                        256.0f)
                           .floor()
                           .add(16.0f)
                           .to(core::DataType::UInt8);
        const auto chroma = (bytes.reshape({height / 2, 2, width / 2, 2, 3})
                                 .sum({1, 3}) /
                             4.0f)
                                .floor();
        const auto u = ((channel(chroma, 0) * -38.0f +
                         channel(chroma, 1) * -74.0f +
                         channel(chroma, 2) * 112.0f + 128.0f) /
                        256.0f)
                           .floor()
                           .add(128.0f)
                           .clamp(0.0f, 255.0f)
                           .to(core::DataType::UInt8);
        const auto v = ((channel(chroma, 0) * 112.0f +
                         channel(chroma, 1) * -94.0f +
                         channel(chroma, 2) * -18.0f + 128.0f) /
                        256.0f)
                           .floor()
                           .add(128.0f)
                           .clamp(0.0f, 255.0f)
                           .to(core::DataType::UInt8);
        return {y, u, v};
    }

} // namespace
nlohmann::json runCpuColorContracts() {
    using namespace lfs::core;
    std::vector<float> pixels(2 * 256 * 3);
    for (size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = float((i * 719) % 1024) / 511.f - .5f;
    for (int i = 0; i < 255; ++i) {
        const auto edge = (i + .5f) / 255.f;
        pixels[3 * i] = std::nextafter(edge, 0.f);
        pixels[3 * i + 1] = edge;
        pixels[3 * i + 2] = std::nextafter(edge, 1.f);
    }
    pixels[1000] = std::numeric_limits<float>::quiet_NaN();
    pixels[1001] = std::numeric_limits<float>::infinity();
    pixels[1002] = -std::numeric_limits<float>::infinity();
    auto input = Tensor::from_blob(pixels.data(), {2, 256, 3}, Device::CPU, DataType::Float32);
    const auto expected = legacyRgbToYuv420p(input);
    auto result = rgb_to_yuv420p(input);
    require(result.has_value(), "CPU RGB conversion must succeed");
    const auto equal = [&](const Yuv420Planes& actual) {
        require(actual.y.to_vector_uint8() == expected.y.to_vector_uint8(), "CPU luma differs from frozen dev producer");
        require(actual.u.to_vector_uint8() == expected.u.to_vector_uint8(), "CPU U differs from frozen dev producer");
        require(actual.v.to_vector_uint8() == expected.v.to_vector_uint8(), "CPU V differs from frozen dev producer");
    };
    equal(*result);
    auto strided = Tensor::cat({input, Tensor::zeros_like(input)}, 1).slice(1, 0, 256);
    require(!strided.is_contiguous(), "CPU fixture must be strided");
    auto reused = rgb_to_yuv420p_into(strided, *result);
    require(reused.has_value(), "CPU reusable output accepts strided input");
    equal(*result);
    auto alias = *result;
    alias.v = alias.u;
    require(!rgb_to_yuv420p_into(input, alias), "CPU output alias rejected");
    auto wrong_shape = *result;
    wrong_shape.y = wrong_shape.y.slice(1, 0, 255);
    require(!rgb_to_yuv420p_into(input, wrong_shape), "CPU wrong output shape rejected");
    require(!rgb_to_yuv420p(Tensor{}), "Missing CPU input rejected");
    require(!rgb_to_yuv420p(input.slice(1, 0, 255)), "Odd CPU input rejected");
    return {{"success", true}, {"checked_bytes", 768}, {"backend", "cpu"}};
}
