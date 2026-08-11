/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs_onnx_vulkan/onnx_vulkan.hpp"

#include <array>
#include <iostream>
#include <vector>

int main(const int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: lfs_onnx_smoke <model.onnx>\n";
        return 2;
    }
    auto session = lfs::onnx_vulkan::VulkanSession::create(argv[1]);
    if (!session) {
        std::cerr << session.error().message << '\n';
        return 1;
    }
    std::cout << "device " << session->device_name() << '\n';
    constexpr std::int64_t height = 350;
    constexpr std::int64_t width = 518;
    std::vector<float> image(static_cast<std::size_t>(3 * height * width));
    const std::array<std::int64_t, 4> image_shape{1, 3, height, width};
    const std::int64_t tokens = 1800;
    const std::array<lfs::onnx_vulkan::NamedTensorView, 2> inputs{{
        {"image", {lfs::onnx_vulkan::ElementType::Float32, image_shape,
                   std::as_bytes(std::span(image))}},
        {"num_tokens", {lfs::onnx_vulkan::ElementType::Int64, {},
                        std::as_bytes(std::span(&tokens, 1))}},
    }};
    auto outputs = session->run(inputs);
    if (!outputs) {
        std::cerr << outputs.error().message << '\n';
        return 1;
    }
    for (const auto& output : *outputs) {
        std::cout << output.name << ' ';
        for (const auto extent : output.tensor.shape())
            std::cout << extent << 'x';
        std::cout << " bytes=" << output.tensor.bytes().size() << '\n';
    }
    return 0;
}
